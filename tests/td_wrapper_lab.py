"""Live check of every generated wrapper in /project1/WrapperLab, one at a time, sliced across
frames (never blocks TD). POST to the bridge's /run; the report lands in
build/wrapper-lab.json when done (poll for it).

Each wrapper gets a suitable official MediaPipe example image (build/*.jpg, see docs/RELEASING.md)
and is cooked until it reports a detection (or, for image models, a non-empty result), or until
MAX_FRAMES. Afterwards the lab source is restored.
"""

import json
from pathlib import Path

MAX_FRAMES = 1200
IMAGES = {'MediaPipeHands': 'thumbs_up.jpg', 'MediaPipeFace': 'portrait.jpg', 'MediaPipeFaceDetector': 'portrait.jpg',
	'YuNet': 'portrait.jpg', 'HSEmotion': 'portrait.jpg', 'MediaPipeSegment': 'portrait.jpg'}
DEFAULT_IMAGE = 'pose.jpg'
LAB = '/project1/WrapperLab'
REPORT = Path(project.folder) / 'build' / 'wrapper-lab.json'


ONLY = []	# limit to these wrapper names (empty = all)


def _wrappers():
	names = sorted(child.name for child in op(LAB).children if child.fetch('wrapper_spec', None))
	return [name for name in names if not ONLY or name in ONLY]


def start():
	REPORT.unlink(missing_ok=True)
	state = {'queue': _wrappers(), 'results': {}, 'frames': 0,
		'original': op(LAB).op('source').par.file.eval()}
	op(LAB).store('lab_test', state)
	_begin_next()


def _begin_next():
	state = op(LAB).fetch('lab_test')
	if not state['queue']:
		return _finish()
	name = state['queue'][0]
	op(LAB).op('source').par.file = 'build/' + IMAGES.get(name, DEFAULT_IMAGE)
	state['frames'] = 0
	op(LAB).store('lab_test', state)
	run('args[0]()', _poll, delayFrames=2)


def _poll():
	state = op(LAB).fetch('lab_test')
	name = state['queue'][0]
	comp = op(LAB).op(name)
	for output in ('debug', 'out_channels', 'out_boxes'):
		node = comp.op(output)
		if node is not None:
			node.cook(force=True)
	state['frames'] += 1
	done = _detected(comp)
	if not done and state['frames'] < MAX_FRAMES:
		op(LAB).store('lab_test', state)
		run('args[0]()', _poll, delayFrames=1)
		return
	state['results'][name] = _measure(comp, state['frames'], done)
	state['queue'].pop(0)
	op(LAB).store('lab_test', state)
	run('args[0]()', _begin_next, delayFrames=1)


def _detected(comp):
	spec = comp.fetch('wrapper_spec')
	if spec['kind'] == 'image':
		return comp.op('debug').numpyArray()[..., :3].max() > 0
	tracks = comp.op('tracks')
	header = [cell.val for cell in tracks.row(0)] if tracks.numRows else []
	# While a model loads the table can hold a status row; only a real track counts.
	if tracks.numRows < 2 or 'track_id' not in header:
		return False
	if spec.get('result') == 'inference' and comp.op('debug').numpyArray()[..., :3].max() <= 0:
		return False
	if spec['kind'] == 'classifier':
		return 'emotion_label' in header and bool(tracks[1, 'emotion_label'].val)
	return True


def _measure(comp, frames, detected):
	spec = comp.fetch('wrapper_spec')
	nodes = [comp, *comp.findChildren()]
	tracks = comp.op('tracks')
	header = [cell.val for cell in tracks.row(0)] if tracks.numRows else []
	label = next((column for column in ('emotion_label', 'gesture', 'label') if column in header), None)
	debug = comp.op('debug').numpyArray()
	result = {
		'detected': bool(detected), 'frames': frames, 'kind': spec['kind'],
		'tracks': max(0, tracks.numRows - 1),
		'labels': [tracks[row, label].val for row in range(1, tracks.numRows)][:4] if label else [],
		'joints': comp.op('joints').numRows - 1 if comp.op('joints') is not None else None,
		'debug_size': [comp.op('debug').width, comp.op('debug').height],
		'debug_alpha_pixels': int((debug[..., 3] > 0).sum()),
		'debug_rgb_mean': round(float(debug[..., :3].mean()), 4),
		'channels': comp.op('out_channels').numChans,
		'errors': {node.path: node.errors() for node in nodes if node.errors()},
		'warnings': {node.path: node.warnings() for node in nodes if node.warnings()},
	}
	if comp.op('out_boxes') is not None:
		result['out_boxes_alpha_pixels'] = int((comp.op('out_boxes').numpyArray()[..., 3] > 0).sum())
	return result


def _finish():
	state = op(LAB).fetch('lab_test')
	op(LAB).op('source').par.file = state['original']
	op(LAB).unstore('lab_test')
	REPORT.parent.mkdir(parents=True, exist_ok=True)
	REPORT.write_text(json.dumps(state['results'], indent=1, default=str), encoding='utf-8')


start()
print(json.dumps({'scheduled': _wrappers(), 'report': REPORT.as_posix()}))
