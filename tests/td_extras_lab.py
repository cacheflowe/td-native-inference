"""The rest of the wrappers' per-frame Python vs its C++ (Tracker DAT) version, live in TouchDesigner.
POST to the bridge's /run:

	curl -X POST --data-binary @tests/td_extras_lab.py http://127.0.0.1:3031/run

Builds /project1/ExtrasLab: one Python-engine wrapper per session on a clip stepped one frame per TD
frame, plus the C++ node beside the Python one on the same inputs, and compares them every frame:
	rtmw      RTMW3D's rtmw_info (rtmw_convert) vs a Tracker DAT in Convert RTMW3D mode (text)
	scores    HSEmotion's scores (write_scores) vs a Tracker DAT in Class Scores mode (text)
	channels  a C++-engine wrapper's `channels`, the only thing pulled each frame, vs write_channels'
	          conversion of its tracks table at that moment: the track values must be current every
	          frame (an Info CHOP alone wouldn't cook the Tracker DAT). The conversion itself is checked
	          on every recorded table offline (tests/test_tracking_extras.py); the trackers by
	          td_tracker_dat_lab.py. Comparing against the Python wrapper's own channels doesn't work:
	          write_channels force-cooks inference_stats, which recooks the plugin mid-frame.
Report: build/extras-lab.json.
"""
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(project.folder)
LAB = '/project1/ExtrasLab'
REPORT = ROOT / 'build' / 'extras-lab.json'
MEDIA = (ROOT / 'media').as_posix()
SESSIONS = [
	{'name': 'rtmw', 'wrapper': 'RTMW3D', 'check': 'rtmw', 'source': f'{MEDIA}/dancer.mp4', 'frames': 200},
	{'name': 'scores', 'wrapper': 'HSEmotion', 'check': 'scores', 'source': f'{MEDIA}/people-walking-forward.mp4', 'frames': 200},
	{'name': 'channels_holistic', 'wrapper': 'MediaPipeHolisticGPU', 'check': 'channels', 'source': f'{MEDIA}/dancer.mp4',
		'frames': 150},
	{'name': 'channels_detect', 'wrapper': 'Yolo26Detect', 'check': 'channels', 'source': f'{MEDIA}/people-walking.mp4',
		'frames': 150},
]
ONLY = []

python_dir = (ROOT / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder  # noqa: E402

state = {'queue': [s for s in SESSIONS if not ONLY or s['name'] in ONLY], 'reports': [], 'paused': []}


def lab():
	comp = op(LAB)
	if comp is None:
		comp = op('/project1').create('baseCOMP', LAB.rsplit('/', 1)[1])
		comp.nodeX, comp.nodeY = 0, -1600
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
	check = session['check']
	engine = 'cpp' if check == 'channels' else 'python'
	wrapper = wrapper_builder.build(spec, comp, name=session['name'], tracker_engine=engine)
	wrapper.nodeX, wrapper.nodeY = 200, 0
	wrapper.inputConnectors[0].connect(source)
	if hasattr(wrapper.par, 'Debugoverlay'):
		wrapper.par.Debugoverlay = False
	modules = wrapper.op('local/modules')
	tables = modules.op('onnxtop_tables').module
	for cache in (tables._pipelines, tables._written, tables._header_cache, tables._layout_cache, tables._class_scores):
		cache.clear()
	if check == 'rtmw':
		node = wrapper_builder._tracker_dat(wrapper, 'rtmw_info_cpp', 'convert_rtmw')
		node.par.Frametop = 'input'
		node.inputConnectors[0].connect(wrapper.op('detections'))
		return wrapper, ('rtmw_info', 'rtmw_info_cpp')
	if check == 'scores':
		node = wrapper_builder._tracker_dat(wrapper, 'scores_cpp', 'scores')
		for name in ('Classprefix', 'Classsmoothing'):
			if hasattr(wrapper.par, name):
				node.par[name].expr = f'parent().par.{name}'
		node.inputConnectors[0].connect(wrapper.op('crop_boxes'))
		node.inputConnectors[1].connect(wrapper.op('detections'))
		return wrapper, ('scores', 'scores_cpp')
	return wrapper, ('channels', 'tracker/script_tracks')


def begin():
	if not state['queue']:
		finish()
		return
	session = state['queue'].pop(0)
	for c in op('/project1').children:
		if c.isCOMP and c.path != LAB and c.allowCooking and c.name in ('VisionDemo', 'WrapperLab', 'PuppetLab', 'TrackingLab'):
			c.allowCooking = False
			state['paused'].append(c)
	lab().allowCooking = True
	wrapper, pair = build(session)
	state['run'] = {'session': session, 'wrapper': wrapper, 'pair': pair, 'frame': 0, 'compared': 0, 'differing': 0,
		'diffs': [], 'nonempty': 0, 'ms': {'python': [], 'cpp': []}}
	run('args[0]()', step, delayFrames=1)


def f32(value):
	return struct.unpack('f', struct.pack('f', value))[0]


def expected_channels(text):
	"""wrapper_runtime.write_channels' track channels for a tracks table."""
	rows = [line.split('	') for line in text.splitlines()]
	if not rows or 'track_id' not in rows[0]:
		return []
	id_col = rows[0].index('track_id')
	out = []
	for row in rows[1:]:
		for name, cell in zip(rows[0], row):
			try:
				value = float(cell)
			except ValueError:
				continue
			out.append((re.sub(r'[^a-zA-Z0-9_]', '_', f't{row[id_col]}_{name}'), value))
	return out


def text_diff(a, b):
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
	py_node, cpp_node = (wrapper.op(n) for n in st['pair'])
	if session['check'] == 'channels':
		channels, track = py_node, cpp_node
		channels.cook()
		got = [(c.name, c[0]) for c in channels.chans() if re.match(r't\d', c.name)]
		expected = expected_channels(track.text)
		diff = None
		if [n for n, _ in expected] != [n for n, _ in got]:
			diff = {'names': [len(expected), len(got)]}
		else:
			bad = [(n, x, y) for (n, x), (_, y) in zip(expected, got) if f32(x) != y and not (x != x and y != y)]
			if bad:
				diff = {'values': len(bad), 'first': bad[0]}
		st['nonempty'] += bool(expected)
	else:
		diff = text_diff(py_node.text, cpp_node.text)
		st['nonempty'] += py_node.numRows > 1
	frame = absTime.frame
	for engine, node in (('python', py_node), ('cpp', cpp_node)):
		if node.cookAbsFrame == frame:
			st['ms'][engine].append(node.cookTime)
	st['compared'] += 1
	if diff:
		st['differing'] += 1
		if len(st['diffs']) < 5:
			st['diffs'].append({'f': st['frame'], **diff})
	st['frame'] += 1
	if st['frame'] >= session['frames']:
		mean = lambda v: round(sum(v) / len(v), 4) if v else None
		state['reports'].append({'session': session['name'], 'frames': st['frame'], 'differing_frames': st['differing'],
			'frames_with_data': st['nonempty'], 'diffs': st['diffs'],
			'mean_ms': {e: mean(v) for e, v in st['ms'].items()}})
		run('args[0]()', begin, delayFrames=1)
		return
	wrapper.parent().op('source').par.index = st['frame']
	run('args[0]()', step, delayFrames=1)


def finish():
	REPORT.parent.mkdir(parents=True, exist_ok=True)
	REPORT.write_text(json.dumps(state['reports'], indent=1))
	lab().allowCooking = False
	for c in state['paused']:
		c.allowCooking = True
	print(f'[ExtrasLab] wrote {REPORT}')


run('args[0]()', begin, delayFrames=1)
print(f"[ExtrasLab] running {[s['name'] for s in state['queue']]}")
