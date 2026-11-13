"""Records tracker fixtures in TouchDesigner: per frame, a wrapper's Info DAT (what the tracker
reads), its `results` / `result_delay` channels, whether each tracker script cooked, and the
tables the Python tracker wrote. tests/tracking_replay.py replays a fixture through the Python
tracker (and later TrackerDAT's replay tool) and diffs the tables.

Driven by tests/td_record_tracking.py (POST it to the bridge's /run). Runs one session at a time
in /project1/TrackingLab, one frame per step via run() (never a background thread). The movie is
stepped by index (Play Mode: Specify), so a session sees every frame of the clip whatever TD's
frame rate.

Fixture file (gzipped JSON lines), build/tracking/<session>.jsonl.gz:
	line 0: {"meta": {session, wrapper, mode, source, frames, tracker_pars, frame_size, ...}}
	then per frame: {"f", "info" (text, or null = unchanged), "results", "result_delay",
		"cooked": [tracks, joints, bones], "tracks" / "joints" / "bones" (text, null = unchanged)}
"""
import gzip
import json
import math
import sys
from pathlib import Path

from td import app, op, project, run  # noqa: F401 (TouchDesigner's globals, in an imported module)

LAB = '/project1/TrackingLab'
OUT_DIR = Path(project.folder) / 'build' / 'tracking'
TABLES = ('tracks', 'joints', 'bones')
WARMUP = 5	# frames cooked on the first movie frame before recording (engines load, MediaPipe starts)

# The recording in progress (module state, not COMP storage: a save mid-recording stays small).
_state = {}


def _builder():
	python_dir = (Path(project.folder) / 'python').as_posix()
	if python_dir not in sys.path:
		sys.path.insert(0, python_dir)
	import wrapper_builder
	return wrapper_builder


def _lab():
	lab = op(LAB)
	if lab is None:
		lab = op('/project1').create('baseCOMP', LAB.rsplit('/', 1)[1])
		lab.nodeX, lab.nodeY = 0, -1200
	source = lab.op('source')
	if source is None:
		source = lab.create('moviefileinTOP', 'source')
		source.nodeX, source.nodeY = 0, 0
	motion = lab.op('motion')
	if motion is None:
		motion = lab.create('transformTOP', 'motion')
		motion.nodeX, motion.nodeY = 150, 0
		motion.inputConnectors[0].connect(source)
		motion.par.extend = 'zero'
	return lab


def _move(lab, session, frame):
	"""A session with 'motion': the still photo drifts, scales and turns, the same every run."""
	motion = lab.op('motion')
	amount = session.get('motion', 0.0) if frame >= 0 else 0.0
	motion.par.tx = 0.25 * amount * math.sin(frame * 0.045)
	motion.par.ty = 0.1 * amount * math.sin(frame * 0.07 + 1.0)
	s = 1.0 + 0.35 * amount * math.sin(frame * 0.03)
	motion.par.sx = motion.par.sy = s
	motion.par.rotate = 15.0 * amount * math.sin(frame * 0.05)


def _wrapper(lab, session):
	"""The session's wrapper, built from its spec on first use (live python/ modules)."""
	name = session.get('instance') or session['wrapper']
	comp = lab.op(name)
	if comp is None:
		builder = _builder()
		# The Python tracker: the fixtures' reference while the C++ port is checked against it.
		comp = builder.build(builder.load_spec(session['wrapper']), lab, name=name, tracker_engine='python')
		index = len([c for c in lab.children if c.fetch('wrapper_spec', None)]) - 1
		comp.nodeX, comp.nodeY = 300, -index * 200
	if not comp.inputConnectors[0].connections or comp.inputConnectors[0].connections[0].owner != lab.op('motion'):
		comp.inputConnectors[0].connect(lab.op('motion'))
	return comp


def start(sessions):
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	lab = _lab()
	lab.allowCooking = True
	_state.clear()
	_state.update(queue=list(sessions), done=[], step=None)
	run('args[0]()', _begin, delayFrames=1)


def _modules(comp):
	return comp.op('local/modules/onnxtop_tables').module, comp.op('local/modules/object_tracker').module


def _begin():
	lab = op(LAB)
	state = _state
	if not state['queue']:
		lab.allowCooking = False
		(OUT_DIR / 'recorded.json').write_text(json.dumps(state['done'], indent=1))
		print('[TrackingRecorder] done:', state['done'])
		return
	session = state['queue'][0]
	comp = _wrapper(lab, session)
	for child in lab.children:
		if child.fetch('wrapper_spec', None):
			child.allowCooking = child is comp
	for name, value in session.get('pars', {}).items():
		setattr(comp.par, name, value)
	source = lab.op('source')
	source.par.file = session['source']
	source.par.playmode = 'specify'
	source.par.indexunit = 'indices'
	source.par.index = session.get('start', 0)
	state['step'] = {'frame': -WARMUP, 'path': comp.path, 'last': {}}
	run('args[0]()', _step, delayFrames=1)


def _reset(comp):
	"""A known starting state: no tracks, ids from 1, no skip memory. Done in the first step,
	after the session's image and motion are set: a cook in between would otherwise run on the
	previous session's image, unrecorded."""
	tracker = comp.op('tracker')
	tables, objects = _modules(comp)
	tables.reset(tracker)
	objects.Track._next_id = 1
	objects.KeypointTrack._next_id = 1
	for key in [k for k in tables._written if k[1] == tracker.path]:
		del tables._written[key]
	return [tracker.op('script_' + t).totalCooks for t in TABLES]


def _text(dat):
	return dat.text if dat is not None else None


def _step():
	lab = op(LAB)
	state = _state
	session, step = state['queue'][0], state['step']
	comp = op(step['path'])
	tracker = comp.op('tracker')
	source = lab.op('source')
	frame = step['frame']
	source.par.index = session.get('start', 0) + max(frame, 0)
	_move(lab, session, frame)
	if frame == -WARMUP:
		step['counts'] = _reset(comp)
	scripts = [tracker.op('script_' + t) for t in TABLES]
	comp.op('debug').cook(force=True)
	for t in TABLES:
		comp.op('out_' + t).cook()
	stats = comp.op('inference_stats')
	stats.cook()
	# Cooks since the last step (from the reset in _begin): normally 0 or 1; more means TD cooked
	# the tracker between steps too (a viewer open on it), which the replay can't reproduce exactly.
	counts = [s.totalCooks for s in scripts]
	cooked = [c - b for c, b in zip(counts, step['counts'])]
	step['counts'] = counts

	if frame == -WARMUP:
		tables, _ = _modules(comp)
		meta = {
			'session': session['name'], 'wrapper': session['wrapper'], 'source': session['source'],
			'start': session.get('start', 0), 'frames': session['frames'], 'pars': session.get('pars', {}),
			'motion': session.get('motion', 0.0),
			'mode': tracker.par.Trackermode.eval(),
			'tracker_pars': {p.name: p.eval() for p in tracker.customPars if not p.isPulse},
			'frame_size': list(tables._frame_size(tracker)),
			'td_build': app.build,
		}
		step['file'] = str(OUT_DIR / f"{session['name']}.jsonl.gz")
		step['lines'] = [json.dumps({'meta': meta})]
	# Warm-up frames (f < 0) are recorded too: they advance the tracker.
	record = {'f': frame, 'cooked': cooked}
	info = tracker.inputs[0] if tracker.inputs else None
	texts = {'info': _text(info), **{t: _text(s) for t, s in zip(TABLES, scripts)}}
	for key, text in texts.items():
		record[key] = None if step['last'].get(key) == text else text
		step['last'][key] = text
	for chan in ('results', 'result_delay'):
		c = stats.chan(chan) if stats is not None else None
		record[chan] = c.eval() if c is not None else None
	step['lines'].append(json.dumps(record))
	step['frame'] = frame + 1
	if step['frame'] >= session['frames']:
		with gzip.open(step['file'], 'wt', encoding='utf-8') as f:
			f.write('\n'.join(step['lines']) + '\n')
		state['done'].append({'session': session['name'], 'file': step['file'], 'frames': session['frames']})
		state['queue'].pop(0)
		state['step'] = None
		run('args[0]()', _begin, delayFrames=1)
		return
	run('args[0]()', _step, delayFrames=1)
