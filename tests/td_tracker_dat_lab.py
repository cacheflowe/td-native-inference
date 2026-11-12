"""Tracker DAT vs the Python tracker, live in TouchDesigner. POST to the bridge's /run:

	curl -X POST --data-binary @tests/td_tracker_dat_lab.py http://127.0.0.1:3031/run

Builds /project1/TrackerDATLab: one wrapper per SESSIONS entry on a clip stepped one frame per TD
frame, its usual (Python) `tracker` plus `tracker_cpp`, a second tracker COMP of Tracker DATs
(wrapper_builder engine 'cpp') on the same Info DAT, result counters and parameters. Every frame
both trackers' tracks / joints / bones are pulled and compared as text, and each node's cook time
is read. Returns at once; the report lands in build/tracker-dat-lab.json. The Python tracker's id
counter starts where the DLL's is (read from an unwired Tracker DAT's Info CHOP), and other demos
stop cooking during the run, so ids and colors must match too.
"""
import json
import sys
from pathlib import Path

ROOT = Path(project.folder)
LAB = '/project1/TrackerDATLab'
REPORT = ROOT / 'build' / 'tracker-dat-lab.json'
MEDIA = (ROOT / 'media').as_posix()
SESSIONS = [
	{'name': 'holistic_gpu', 'wrapper': 'MediaPipeHolisticGPU', 'source': f'{MEDIA}/dancer.mp4', 'frames': 300},
	{'name': 'holistic_gpu_noface', 'wrapper': 'MediaPipeHolisticGPU', 'source': f'{MEDIA}/dancer.mp4', 'frames': 300,
		'pars': {'Holisticface': False}},
	{'name': 'detect', 'wrapper': 'Yolo26Detect', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'segment', 'wrapper': 'Yolo26Seg', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'yunet', 'wrapper': 'YuNet', 'source': f'{MEDIA}/people-walking-forward.mp4', 'frames': 300},
]
ONLY = []
TABLES = ('tracks', 'joints', 'bones')

python_dir = (ROOT / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder  # noqa: E402

state = {'queue': [s for s in SESSIONS if not ONLY or s['name'] in ONLY], 'reports': []}


def lab():
	comp = op(LAB)
	if comp is None:
		comp = op('/project1').create('baseCOMP', LAB.rsplit('/', 1)[1])
		comp.nodeX, comp.nodeY = 0, -1400
	if comp.op('source') is None:
		comp.create('moviefileinTOP', 'source')
	return comp


def build(session):
	comp = lab()
	source = comp.op('source')
	source.par.file = session['source']
	source.par.playmode = 'specify'
	source.par.indexunit = 'frames'
	source.par.index = 0
	spec = wrapper_builder.load_spec(session['wrapper'])
	wrapper = wrapper_builder.build(spec, comp, name=session['name'], tracker_engine='python')
	wrapper.nodeX, wrapper.nodeY = 200, 0
	wrapper.inputConnectors[0].connect(source)
	if hasattr(wrapper.par, 'Debugoverlay'):
		wrapper.par.Debugoverlay = False
	for name, value in session.get('pars', {}).items():
		setattr(wrapper.par, name, value)
	# The second tracker: built as 'tracker' by the same code, then renamed.
	python_tracker = wrapper.op('tracker')
	python_tracker.name = 'tracker_python'
	source_name = python_tracker.inputs[0].name
	# The spec's own tracker may already be the Tracker DAT; the comparison needs one of each.
	cpp_tracker = wrapper_builder._build_tracker(wrapper, spec['tracker'], source_name, 'cpp')
	cpp_tracker.name = 'tracker_cpp'
	python_tracker.name = 'tracker'
	cpp_tracker.nodeX, cpp_tracker.nodeY = python_tracker.nodeX, python_tracker.nodeY - 200
	for par in cpp_tracker.customPars:
		if hasattr(python_tracker.par, par.name):
			par.expr = f"op('tracker').par.{par.name}"
	for t in TABLES:	# the plugin's own timing: cook_ms (all of it), track_ms (without writing the table)
		info = cpp_tracker.create('infoCHOP', 'info_' + t)
		info.par.op = 'script_' + t
	return wrapper


def sync_ids(wrapper):
	"""Clear the Python tracker's caches and start its ids at the Tracker DATs' next id (one
	counter per process, shared by every Tracker DAT)."""
	modules = wrapper.op('local/modules')
	tables = modules.op('onnxtop_tables').module
	for cache in (tables._pipelines, tables._written, tables._header_cache, tables._layout_cache, tables._class_scores):
		cache.clear()
	comp = lab()
	probe = comp.op('id_probe') or wrapper_builder._tracker_dat(comp, 'id_probe', 'tracks')
	info = comp.op('id_probe_info') or comp.create('infoCHOP', 'id_probe_info')
	info.par.op = probe
	probe.cook(force=True)
	info.cook(force=True)
	tracker_module = modules.op('object_tracker').module
	tracker_module.KeypointTrack._next_id = int(info['next_track_id'].eval())
	tracker_module.Track._next_id = int(info['next_box_track_id'].eval())


def begin():
	if not state['queue']:
		finish()
		return
	session = state['queue'].pop(0)
	comp = lab()
	comp.allowCooking = True
	paused = [c for c in op('/project1').children if c.isCOMP and c.path != LAB and c.allowCooking
		and c.name in ('VisionDemo', 'WrapperLab', 'PuppetLab', 'TrackingLab')]
	for c in paused:
		c.allowCooking = False
	state.setdefault('paused', []).extend(paused)
	wrapper = build(session)
	sync_ids(wrapper)
	state['run'] = {'session': session, 'wrapper': wrapper, 'frame': 0, 'compared': 0, 'differing': 0, 'diffs': [],
		'ms': {'python': {t: [] for t in TABLES}, 'cpp': {t: [] for t in TABLES}},
		'plugin': {t: {'cook_ms': [], 'track_ms': [], 'input_ms': []} for t in TABLES}}
	run('args[0]()', step, delayFrames=1)


def first_difference(a, b):
	if a == b:
		return None
	ra, rb = a.splitlines(), b.splitlines()
	if len(ra) != len(rb):
		return {'rows': [len(ra), len(rb)]}
	for r, (x, y) in enumerate(zip(ra, rb)):
		if x != y:
			cx, cy = x.split('\t'), y.split('\t')
			if len(cx) != len(cy):
				return {'row': r, 'cols': [len(cx), len(cy)]}
			c = next(i for i, (p, q) in enumerate(zip(cx, cy)) if p != q)
			header = ra[0].split('\t')
			return {'row': r, 'col': header[c] if c < len(header) else c, 'python': cx[c], 'cpp': cy[c]}
	return None


def step():
	st = state['run']
	session, wrapper = st['session'], st['wrapper']
	frame = absTime.frame
	texts = {}
	for engine, name in (('python', 'tracker'), ('cpp', 'tracker_cpp')):
		tracker = wrapper.op(name)
		for t in TABLES:
			node = tracker.op('script_' + t)
			texts[engine, t] = node.text
			if node.cookAbsFrame == frame:
				st['ms'][engine][t].append(node.cookTime)
				if engine == 'cpp':
					info = tracker.op('info_' + t)
					for chan in ('cook_ms', 'track_ms', 'input_ms'):
						st['plugin'][t][chan].append(info[chan].eval())
	differs = False
	for t in TABLES:
		st['compared'] += 1
		diff = first_difference(texts['python', t], texts['cpp', t])
		if diff:
			differs = True
			if len(st['diffs']) < 8:
				st['diffs'].append({'f': st['frame'], 'table': t, **diff})
	st['differing'] += differs
	st['frame'] += 1
	if st['frame'] >= session['frames']:
		report(st)
		run('args[0]()', begin, delayFrames=1)
		return
	wrapper.parent().op('source').par.index = st['frame']
	run('args[0]()', step, delayFrames=1)


def report(st):
	def mean(values):
		return round(sum(values) / len(values), 4) if values else None

	timing = {e: {t: {'mean_ms': mean(v), 'cooks': len(v)} for t, v in tables.items()} for e, tables in st['ms'].items()}
	for e in timing:
		timing[e]['total_mean_ms'] = round(sum(x['mean_ms'] or 0 for x in timing[e].values()), 4)
	timing['cpp_inside'] = {t: {c: mean(v) for c, v in chans.items()} for t, chans in st['plugin'].items()}
	state['reports'].append({'session': st['session']['name'], 'frames': st['frame'], 'tables_compared': st['compared'],
		'differing_frames': st['differing'], 'diffs': st['diffs'], 'timing': timing})


def finish():
	REPORT.parent.mkdir(parents=True, exist_ok=True)
	REPORT.write_text(json.dumps(state['reports'], indent=1))
	lab().allowCooking = False
	for c in state.get('paused', []):
		c.allowCooking = True
	print(f'[TrackerDATLab] wrote {REPORT}')


run('args[0]()', begin, delayFrames=1)
print(f"[TrackerDATLab] running {[s['name'] for s in state['queue']]}")
