"""Live check of every generated wrapper in /project1/WrapperLab, one at a time, sliced across
frames (never blocks TD). POST to the bridge's /run; the report lands in
build/wrapper-lab.json when done (poll for it).

Each wrapper gets a suitable example image, docs/images/sources/hand.jpg, face.jpg or body.jpg
(replace those files to change the README thumbnails), and is cooked until it reports a detection
(or, for image models, a non-empty result), or until MAX_FRAMES, then SETTLE_FRAMES more. Audio
wrappers listen to the lab's audio_source (a speech clip) until a sentence shows as a caption. With
THUMBNAILS on, docs/images/models/<Name>.jpg shows its overlay over the input (image results beside
it). Afterwards the lab source is restored.
"""

import json
from pathlib import Path

MAX_FRAMES = 1200
SETTLE_FRAMES = 90
SOURCES = Path(project.folder) / 'docs' / 'images' / 'sources'
IMAGES = {'MediaPipeHands': 'hand.jpg', 'MediaPipeFace': 'face.jpg', 'MediaPipeFaceDetector': 'face.jpg',
	'YuNet': 'face.jpg', 'HSEmotion': 'face.jpg', 'MediaPipeSegment': 'face.jpg', 'DAViDNormals': 'face.jpg'}
DEFAULT_IMAGE = 'body.jpg'
LAB = '/project1/WrapperLab'
REPORT = Path(project.folder) / 'build' / 'wrapper-lab.json'


ONLY = []	# limit to these wrapper names (empty = all)
# Save docs/images/models/<Name>.jpg once a wrapper detects (README / MODELS.md). Off for routine
# runs (it rewrites the committed images); turn on when preparing a release.
THUMBNAILS = False
THUMB_HEIGHT = 320
THUMB_DIR = Path(project.folder) / 'docs' / 'images' / 'models'


def _wrappers():
	names = sorted(child.name for child in op(LAB).children if child.fetch('wrapper_spec', None))
	return [name for name in names if not ONLY or name in ONLY]


def start():
	REPORT.unlink(missing_ok=True)
	state = {'queue': _wrappers(), 'results': {}, 'frames': 0,
		'original': op(LAB).op('source').par.file.eval(), 'cooking': op(LAB).allowCooking}
	# The wrappers must cook to be tested (their forced output cooks don't reach a plugin inside
	# a COMP with cooking off); _finish restores the flag.
	op(LAB).allowCooking = True
	op(LAB).store('lab_test', state)
	_begin_next()


def _begin_next():
	state = op(LAB).fetch('lab_test')
	if not state['queue']:
		return _finish()
	name = state['queue'][0]
	op(LAB).op('source').par.file = (SOURCES / IMAGES.get(name, DEFAULT_IMAGE)).as_posix()
	comp = op(LAB).op(name)
	if comp.fetch('wrapper_spec')['kind'] == 'speech':
		# Muted while tested; a new line in the input DAT is what gets spoken.
		state['play_audio'] = comp.par.Playaudio.eval()
		comp.par.Playaudio = False
		comp.unstore('lab_heard')
		op(LAB).op('text_source').text = f'This is the wrapper lab, test number {absTime.frame}.'
	state['frames'] = 0
	op(LAB).store('lab_test', state)
	run('args[0]()', _poll, delayFrames=2)


def _poll():
	state = op(LAB).fetch('lab_test')
	name = state['queue'][0]
	comp = op(LAB).op(name)
	for output in ('debug', 'out_channels', 'out_boxes', 'out_audio'):
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
	if 'play_audio' in state:
		comp.par.Playaudio = state.pop('play_audio')
	state['queue'].pop(0)
	op(LAB).store('lab_test', state)
	run('args[0]()', _begin_next, delayFrames=1)


def _detected(comp):
	spec = comp.fetch('wrapper_spec')
	if spec['kind'] == 'transcript':
		# A final sentence, and the caption showing (it's cleared again after Hold seconds).
		latest, lines = comp.op('latest'), comp.op('caption_lines')
		return latest.numRows > 1 and bool(latest[1, 'text'].val) and bool(lines.numRows and lines[0, 0].val)
	if spec['kind'] == 'scores':
		# A confident sound other than Silence (the clip has pauses), for a useful thumbnail.
		top = comp.op('top')
		return top.numRows > 1 and top[1, 'class'].val != 'Silence' and float(top[1, 'score'].val or 0) > 0.5
	if spec['kind'] == 'speech':
		# Audible output (on any cook so far: speech has pauses) and the caption showing the line.
		audio = comp.op('out_audio')['audio'].vals
		if audio and max(abs(v) for v in audio) > 0.01:
			comp.store('lab_heard', True)
		text = comp.op('speaking_text')
		return bool(comp.fetch('lab_heard', False)) and bool(text.numRows and text[0, 0].val)
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
	if spec['kind'] == 'transcript':
		transcript = comp.op('transcript')
		return {
			'detected': bool(detected), 'frames': frames, 'kind': spec['kind'],
			'sentences': [transcript[row, 'text'].val for row in range(1, transcript.numRows)][-4:],
			'caption': comp.op('caption_lines')[0, 0].val if comp.op('caption_lines').numRows else '',
			'channels': {channel.name: round(channel[0], 4) for channel in comp.op('out_channels').chans()},
			'debug_size': [comp.op('debug').width, comp.op('debug').height],
			'errors': {node.path: node.errors() for node in nodes if node.errors()},
			'warnings': {node.path: node.warnings() for node in nodes if node.warnings()},
		}
	if spec['kind'] == 'speech':
		lines = comp.op('out_lines')
		return {
			'detected': bool(detected), 'frames': frames, 'kind': spec['kind'],
			'lines': [[cell.val for cell in row] for row in lines.rows()],
			'caption': comp.op('speaking_text')[0, 0].val if comp.op('speaking_text').numRows else '',
			'audio_rate': comp.op('out_audio').rate,
			'channels': {channel.name: round(channel[0], 3) for channel in comp.op('out_channels').chans()},
			'errors': {node.path: node.errors() for node in nodes if node.errors()},
			'warnings': {node.path: node.warnings() for node in nodes if node.warnings()},
		}
	if spec['kind'] == 'scores':
		top = comp.op('top')
		return {
			'detected': bool(detected), 'frames': frames, 'kind': spec['kind'],
			'top': [f"{top[row, 'label'].val} {top[row, 'score'].val}" for row in range(1, top.numRows)],
			'events': {channel.name: round(channel[0], 3) for channel in comp.op('out_channels').chans()},
			'score_channels': comp.op('out_scores').numChans,
			'errors': {node.path: node.errors() for node in nodes if node.errors()},
			'warnings': {node.path: node.warnings() for node in nodes if node.warnings()},
		}
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

		if spec.get('input') in ('audio', 'text'):
			# The caption / top results over a dark backdrop at the debug image's size.
			source = work.create('constantTOP')
			source.par.outputresolution = 'custom'
			source.par.resolutionw, source.par.resolutionh = comp.op('debug').width, comp.op('debug').height
			source.par.colorr, source.par.colorg, source.par.colorb = 0.12, 0.13, 0.15
		else:
			source = select('passthrough')
		if spec.get('input') in ('audio', 'text'):
			panels = [over(select('debug'), source)]
		elif spec['kind'] == 'image':
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
	# Back to the lab's own Cooking flag. A real on -> off switch also stops nodes that would
	# otherwise go on cooking every frame inside a COMP with cooking off.
	op(LAB).allowCooking = state.get('cooking', True)


start()
print(json.dumps({'scheduled': _wrappers(), 'report': REPORT.as_posix()}))
