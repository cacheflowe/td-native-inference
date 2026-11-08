"""Clean-process wrapper test: loads exported wrapper toxes into a fresh TD process.

Build the driver tox from the dev project (POST this file to the bridge's /run), then launch
TouchDesigner.exe on build/clean-wrapper/RunCleanWrapper.tox (tests/run_clean_wrapper.ps1 does
this) with:
	TDNI_CASES   JSON list of {"tox": path, "image": path}, tested one after another
	TDNI_RESULT  JSON report path (TD quits after writing it)
Each case reports the inner node type, errors/warnings, tracks, joints and overlay pixels. A case
passes when it has a real track (or, for image models, a non-empty result image). Audio wrappers
get "image" as an audio file and pass on a sentence / a non-silent sound; text to speech gets a
line of text (muted) and passes once it's heard.
"""

import json
import os
from pathlib import Path

MAX_FRAMES = 900


def _issues(nodes, method):
	return {node.path: getattr(node, method)() for node in nodes if getattr(node, method)()}


def start():
	container = me.parent()
	container.store('cases', json.loads(os.environ['TDNI_CASES']))
	container.store('report', {})
	_next()


def _next():
	container = me.parent()
	cases = container.fetch('cases')
	if not cases:
		Path(os.environ['TDNI_RESULT']).write_text(json.dumps(container.fetch('report'), indent=1, default=str), encoding='utf-8')
		project.quit(force=True)
		return
	case = cases[0]
	for child in list(container.children):
		if child.name not in ('test_driver', 'startup'):
			child.destroy()
	wrapper = container.loadTox(case['tox'])
	kind = wrapper.fetch('wrapper_spec', {}).get('input', 'video')
	if kind == 'audio':
		source = container.create('audiofileinCHOP', 'source')
		source.par.file = case['image']
	elif kind == 'text':
		wrapper.par.Playaudio = False
		source = container.create('textDAT', 'source')
	else:
		source = container.create('moviefileinTOP', 'source')
		source.par.file = case['image']
	wrapper.inputConnectors[0].connect(source)
	# Text to speech ignores what's in its input at startup: _poll sends the line a bit later.
	container.store('say_at', 60 if kind == 'text' else None)
	container.store('frames', 0)
	run('args[0]()', _poll, delayFrames=1)


def _wrapper():
	container = me.parent()
	return next(child for child in container.children if child.family == 'COMP' and child.op('inference') is not None)


def _detected(wrapper):
	kind = wrapper.fetch('wrapper_spec', {}).get('kind')
	if kind == 'transcript':
		latest = wrapper.op('latest')
		return latest.numRows > 1 and bool(latest[1, 'text'].val)
	if kind == 'scores':
		top = wrapper.op('top')
		return top.numRows > 1 and top[1, 'class'].val != 'Silence' and float(top[1, 'score'].val or 0) > 0.5
	if kind == 'speech':
		audio = wrapper.op('out_audio')['audio'].vals
		if audio and max(abs(v) for v in audio) > 0.01:
			wrapper.store('heard', True)
		return bool(wrapper.fetch('heard', False))
	tracks = wrapper.op('tracks')
	header = [cell.val for cell in tracks.row(0)] if tracks is not None and tracks.numRows else []
	if 'track_id' in header and tracks.numRows > 1:
		return True
	spec = wrapper.fetch('wrapper_spec', {})
	stats = wrapper.op('inference_stats')
	loaded = stats is not None and stats['loaded'] is not None and stats['loaded'][0] >= 1
	return bool(spec.get('kind') == 'image' and loaded and wrapper.op('debug').numpyArray()[..., :3].max() > 0)


def _poll():
	container = me.parent()
	wrapper = _wrapper()
	frames = container.fetch('frames') + 1
	container.store('frames', frames)
	if frames == container.fetch('say_at', None):
		container.op('source').text = 'This line was spoken in a clean project.'
	for name in ('debug', 'out_channels', 'out_audio'):
		node = wrapper.op(name)
		if node is not None:
			node.cook(force=True)
	if frames < MAX_FRAMES and not _detected(wrapper):
		run('args[0]()', _poll, delayFrames=1)
		return
	_record(wrapper, frames)
	cases = container.fetch('cases')
	cases.pop(0)
	container.store('cases', cases)
	run('args[0]()', _next, delayFrames=1)


def _record(wrapper, frames):
	nodes = [wrapper, *wrapper.findChildren()]
	inference = wrapper.op('inference')
	tracks = wrapper.op('tracks')
	header = [cell.val for cell in tracks.row(0)] if tracks is not None and tracks.numRows else []
	rows = [[cell.val for cell in row] for row in tracks.rows()[1:]] if 'track_id' in header else []
	label = next((column for column in ('emotion_label', 'gesture', 'class_name') if column in header), None)
	debug = wrapper.op('debug').numpyArray()
	report = me.parent().fetch('report')
	report[wrapper.name] = {
		'passed': _detected(wrapper), 'frames': frames, 'project_folder': project.folder,
		'inference_type': inference.OPType,
		'errors': _issues(nodes, 'errors'), 'warnings': _issues(nodes, 'warnings'),
		'tracks': len(rows), 'labels': [row[header.index(label)] for row in rows] if label else [],
		'joints': wrapper.op('joints').numRows - 1 if wrapper.op('joints') is not None else None,
		'overlay_pixels': int((debug[..., 3] > 0).sum()),
		'pages': {page.name: [par.name for par in page.pars] for page in wrapper.customPages},
	}
	for name in ('latest', 'top', 'out_lines'):	# audio wrappers: what they heard / said
		table = wrapper.op(name)
		if table is not None:
			report[wrapper.name][name] = [[cell.val for cell in row] for row in table.rows()][:4]
	# A partial report after each case shows how far a run got if it never finishes.
	Path(os.environ['TDNI_RESULT'] + '.partial').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
	me.parent().store('report', report)


def build_driver():
	"""Run in the dev project: save build/clean-wrapper/RunCleanWrapper.tox."""
	destination = Path(project.folder) / 'build' / 'clean-wrapper' / 'RunCleanWrapper.tox'
	container = op('/project1').create('baseCOMP', 'RunCleanWrapper')
	try:
		driver = container.create('textDAT', 'test_driver')
		driver.text = (Path(project.folder) / 'tests' / 'td_clean_wrapper.py').read_text(encoding='utf-8')
		startup = container.create('executeDAT', 'startup')
		startup.par.create = True
		startup.par.start = True
		startup.text = ("import os\ndef onStart():\n\tif os.environ.get('TDNI_RESULT') and not parent().fetch('started', False):\n"
			"\t\tparent().store('started', True)\n\t\trun(\"op(args[0]).module.start()\", parent().op('test_driver').path, delayFrames=30)\n"
			"def onCreate():\n\tonStart()\n")
		container.save(destination.as_posix(), createFolders=True)
	finally:
		container.destroy()
	print(json.dumps({'driver': destination.as_posix()}))


if me.name != 'test_driver':
	build_driver()
