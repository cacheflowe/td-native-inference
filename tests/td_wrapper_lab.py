"""Live check of every generated wrapper in /project1/WrapperLab, one at a time, sliced across
frames (never blocks TD). POST to the bridge's /run; the report lands in
build/wrapper-lab.json when done (poll for it).

Each wrapper gets a suitable example image (docs/images/sources/<name>, else the official
MediaPipe test image build/<name>) and is cooked until it reports a detection (or, for image
models, a non-empty result), or until MAX_FRAMES, then SETTLE_FRAMES more. With THUMBNAILS on,
docs/images/models/<Name>.jpg shows its overlay over the input (image results beside it), for the
README. Afterwards the lab source is restored.
"""

import json
from pathlib import Path

MAX_FRAMES = 1200
SETTLE_FRAMES = 90
IMAGES = {'MediaPipeHands': 'thumbs_up.jpg', 'MediaPipeFace': 'portrait.jpg', 'MediaPipeFaceDetector': 'portrait.jpg',
	'YuNet': 'portrait.jpg', 'HSEmotion': 'portrait.jpg', 'MediaPipeSegment': 'portrait.jpg'}
DEFAULT_IMAGE = 'pose.jpg'
LAB = '/project1/WrapperLab'
REPORT = Path(project.folder) / 'build' / 'wrapper-lab.json'


ONLY = []	# limit to these wrapper names (empty = all)
THUMBNAILS = True	# save docs/images/models/<Name>.jpg once a wrapper detects (README / MODELS.md)
THUMB_HEIGHT = 320
THUMB_DIR = Path(project.folder) / 'docs' / 'images' / 'models'


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
	image = IMAGES.get(name, DEFAULT_IMAGE)
	# Images in docs/images/sources/ (publishable, for the README thumbnails) replace the
	# MediaPipe test images in build/ of the same name.
	override = Path(project.folder) / 'docs' / 'images' / 'sources' / image
	op(LAB).op('source').par.file = override.as_posix() if override.exists() else 'build/' + image
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
	if 'detected_at' not in state and (_detected(comp) or state['frames'] >= MAX_FRAMES):
		state['detected_at'] = state['frames'] if _detected(comp) else None
	# Keep running SETTLE_FRAMES past the first detection: right after the input changes,
	# late MediaPipe results from the previous image and smoothed keypoints still converging
	# would show up as ghost tracks and offset skeletons.
	settled = 'detected_at' in state and (state['detected_at'] is None
		or state['frames'] >= state['detected_at'] + SETTLE_FRAMES)
	if not settled:
		op(LAB).store('lab_test', state)
		run('args[0]()', _poll, delayFrames=1)
		return
	done = state.pop('detected_at') is not None and _detected(comp)
	state['results'][name] = _measure(comp, state['frames'], done)
	if THUMBNAILS and done:
		state['results'][name]['thumbnail'] = _thumbnail(comp)
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


def _thumbnail(comp):
	"""The overlay composited over the input; image results (depth, masks) beside the input."""
	spec = comp.fetch('wrapper_spec')
	work = op(LAB).create('baseCOMP', '_thumbnail')
	try:
		def select(path):
			node = work.create('selectTOP')
			node.par.top = comp.path + '/' + path
			return node

		def over(top, bottom):
			node = work.create('compositeTOP')
			node.par.operand = 'over'
			node.inputConnectors[0].connect(top)
			node.inputConnectors[1].connect(bottom)
			return node

		source = select('passthrough')
		if spec['kind'] == 'image':
			panels = [source, select('debug')]
		elif spec.get('result') == 'inference':
			panels = [over(select('out_boxes'), source), select('debug')]
		else:
			panels = [over(select('debug'), source)]
		source.cook(force=True)
		width = int(round(THUMB_HEIGHT * source.width / max(1, source.height)))
		layout = work.create('layoutTOP')
		for index, panel in enumerate(panels):
			# Every panel at the input's aspect (the seg matte is at inference size).
			sized = work.create('resolutionTOP')
			sized.inputConnectors[0].connect(panel)
			sized.par.outputresolution = 'custom'
			sized.par.resolutionw, sized.par.resolutionh = width, THUMB_HEIGHT
			sized.par.outputaspect = 'resolution'
			layout.inputConnectors[index].connect(sized)
		layout.par.align = 'horizlr'
		layout.par.fit = 'nativeres'
		layout.par.outputresolution = 'custom'
		layout.par.resolutionw = width * len(panels)
		layout.par.resolutionh = THUMB_HEIGHT
		layout.par.outputaspect = 'resolution'	# else it keeps the input's aspect and squashes
		layout.cook(force=True)
		THUMB_DIR.mkdir(parents=True, exist_ok=True)
		destination = THUMB_DIR / f'{comp.name}.jpg'
		layout.save(destination.as_posix(), quality=0.85)
		return destination.relative_to(Path(project.folder)).as_posix()
	finally:
		work.destroy()


def _finish():
	state = op(LAB).fetch('lab_test')
	op(LAB).op('source').par.file = state['original']
	op(LAB).unstore('lab_test')
	REPORT.parent.mkdir(parents=True, exist_ok=True)
	REPORT.write_text(json.dumps(state['results'], indent=1, default=str), encoding='utf-8')


start()
print(json.dumps({'scheduled': _wrappers(), 'report': REPORT.as_posix()}))
