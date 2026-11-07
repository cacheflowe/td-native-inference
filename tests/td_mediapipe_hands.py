"""Run via TD /run; also embeds a clean-process test driver in build/."""

import json
import os
from pathlib import Path
import traceback


def schedule():
	run("op(args[0]).module.step()", me.path, delayFrames=10)


def start():
	root = Path(os.environ.get('TD_HANDS_PACKAGE', project.folder))
	container = me.parent()
	container.store('report', {'passed': False, 'package': root.as_posix(), 'pid': os.getpid()})
	container.store('attempts', 0)
	try:
		wrapper = container.loadTox((root / 'tox/MediaPipeHands.tox').as_posix())
		wrapper.name = 'subject'
		discovered = wrapper.par.Packageroot.eval()
		if not discovered:
			wrapper.par.Packageroot = root.as_posix()
			container.fetch('report')['root_discovery'] = 'explicit_override_required_for_loadTox'
		else:
			container.fetch('report')['root_discovery'] = 'automatic'
		actual = Path(wrapper.par.Packageroot.eval()).resolve()
		assert actual == root.resolve(), f'Package Root mismatch: {actual}; filePath={wrapper.filePath}'
		image = container.create('moviefileinTOP', 'test_image')
		image.par.file = os.environ.get('TD_HANDS_IMAGE', (root / 'build/thumbs_up.jpg').as_posix())
		wrapper.inputConnectors[0].connect(image.outputConnectors[0])
		wrapper.par.Active = True
		schedule()
	except Exception:
		report = container.fetch('report')
		if container.op('subject') is not None:
			subject = container.op('subject')
			report['root_expression'] = subject.par.Packageroot.expr
			report['component_errors'] = subject.errors(recurse=True)
			try:
				report['direct_root'] = subject.op('component').module.default_package_root(subject)
			except Exception:
				report['direct_root_error'] = traceback.format_exc()
		finish(traceback.format_exc())


def step():
	container = me.parent()
	try:
		wrapper = container.op('subject')
		wrapper.op('inference_stats').cook(force=True)
		wrapper.op('tracker/script_tracks').cook(force=True)
		for name in ('debug', 'tracks', 'joints', 'bones', 'channels'):
			wrapper.op(name).cook(force=True)
		attempts = container.fetch('attempts') + 1
		container.store('attempts', attempts)
		stats = wrapper.op('inference_stats')
		native = {channel.name: float(channel[0]) for channel in stats.chans()}
		container.fetch('report')['native_stats'] = native
		ready = native.get('loaded', 0) == 1 and native.get('num_detections', 0) > 0
		tracks = wrapper.op('tracks')
		gesture_ready = tracks.numRows > 1 and tracks[1, 'gesture'].val == 'Thumb_Up'
		if (not ready or not gesture_ready) and attempts < 60:
			schedule()
			return
		check(wrapper)
		finish()
	except Exception:
		finish(traceback.format_exc())


def check(wrapper):
	import numpy as np
	import sys

	report = me.parent().fetch('report')
	for node in [wrapper, *wrapper.findChildren()]:
		assert not node.errors(), f'{node.path}: {node.errors()}'
		for name in ('syncfile', 'loadonstart', 'externaltox', 'enableexternaltox'):
			parameter = getattr(node.par, name, None)
			assert parameter is None or not parameter.eval(), f'Remaining {name}: {node.path}'
		if node.family == 'DAT' and getattr(node.par, 'file', None) is not None:
			assert not node.par.file.eval(), f'Remaining file: {node.path}'
	tracks = wrapper.op('tracks')
	assert tracks.numRows > 1, 'Known hand image produced no tracks'
	assert wrapper.op('inference').par.Task.eval() == 'gesture', 'Wrong native Task'
	assert tracks[1, 'gesture'].val == 'Thumb_Up', 'Known gesture was not recognized'
	header = [cell.val for cell in tracks.row(0)]
	for name in ('track_id', 'presence', 'gesture', 'gesture_score', 'gesture_frames', 'gesture_onset', 'handedness'):
		assert name in header, f'Missing track column: {name}'
	assert wrapper.op('joints').numRows >= 22, 'Missing hand joints'
	assert wrapper.op('bones').numRows >= 22, 'Missing hand bones'
	channels = wrapper.op('channels')
	names = [channel.name for channel in channels.chans()]
	assert 'loaded' in names and channels['loaded'][0] == 1, 'Native model not loaded'
	assert any(name.endswith('_gesture_onset') for name in names), 'Missing gesture trigger channel'
	assert channels['num_detections'][0] > 0, 'No native detections'
	modules = wrapper.op('local/modules')
	tables = modules.op('onnxtop_tables').module
	assert wrapper.op('tracker/script_tracks_callbacks').module.onnxtop_tables is tables
	assert tables.object_tracker is modules.op('object_tracker').module
	assert tables.pose_tracker_pipeline is modules.op('pose_tracker_pipeline').module
	passthrough = wrapper.op('passthrough').numpyArray()
	assert passthrough is not None and np.ptp(passthrough[:, :, :3]) > 0.1, 'Blank passthrough'
	wrapper.par.Debugoverlay = False
	wrapper.op('debug').cook(force=True)
	off = wrapper.op('debug').numpyArray()
	assert off.shape == passthrough.shape, 'Debug-off resolution differs from passthrough'
	assert np.allclose(off, 0), 'Debug-off output is not transparent black'
	wrapper.par.Debugoverlay = True
	wrapper.op('debug').cook(force=True)
	on = wrapper.op('debug').numpyArray()
	assert on.shape == passthrough.shape and on.shape[2] == 4, 'Overlay must match the input in RGBA'
	transparent_pixels = int(np.count_nonzero(on[:, :, 3] < 0.01))
	assert transparent_pixels > on.shape[0] * on.shape[1] * 0.7, 'Overlay still contains an opaque image background'
	assert np.count_nonzero(on[:, :, 3] > 0.01) > 10, 'Overlay has no visible alpha'
	changed = int(np.count_nonzero(np.max(np.abs(on - off), axis=2) > 0.01))
	assert changed > 10, 'Overlay has no visible pixels'
	layers = {}
	layer_pixels = {}
	for mode, source in (('both', 'overlay'), ('boxes', 'debug_boxes/out1'), ('joints', 'debug_keypoints/out1')):
		wrapper.par.Debuglayers = mode
		wrapper.op('debug').cook(force=True)
		image = wrapper.op('debug').numpyArray().copy()
		assert np.allclose(image, wrapper.op(source).numpyArray()), f'Wrong debug source for {mode}'
		assert image.shape == passthrough.shape, f'Wrong debug resolution for {mode}'
		assert np.count_nonzero(image[:, :, 3] < 0.01) > image.shape[0] * image.shape[1] * 0.7, f'Opaque background in {mode}'
		layer_pixels[mode] = int(np.count_nonzero(image[:, :, 3] > 0.01))
		assert layer_pixels[mode] > 10, f'No visible overlay for {mode}'
		layers[mode] = image
		wrapper.par.Debugoverlay = False
		wrapper.op('debug').cook(force=True)
		assert np.allclose(wrapper.op('debug').numpyArray(), 0), f'Debug-off is not transparent for {mode}'
		wrapper.par.Debugoverlay = True
	assert not np.allclose(layers['boxes'], layers['joints']), 'Boxes and joints modes are identical'
	assert not np.allclose(layers['both'], layers['boxes']), 'Both mode omits joints'
	assert not np.allclose(layers['both'], layers['joints']), 'Both mode omits boxes'
	wrapper.par.Debuglayers = 'both'
	assert np.allclose(wrapper.op('passthrough').numpyArray(), passthrough), 'Debug toggle changed the image output'
	package = Path(report['package'])
	assert Path(wrapper.op('inference').par.plugin.eval()).resolve() == (package / 'plugins/MediaPipeTOP.dll').resolve()
	assert Path(wrapper.par.Gesturemodel.eval()).resolve() == (package / 'models/mediapipe/gesture_recognizer.task').resolve()
	report.update({
		'passed': True, 'tracks': tracks.numRows - 1,
		'gesture': tracks[1, 'gesture'].val, 'joints': wrapper.op('joints').numRows - 1,
		'bones': wrapper.op('bones').numRows - 1, 'overlay_pixels': changed,
		'debug_overlay_only': True, 'transparent_background_pixels': transparent_pixels,
		'debug_layers_verified': True, 'debug_layer_pixels': layer_pixels,
		'channels': names, 'inference_ms': float(channels['infer_ms'][0]),
		'package_root': wrapper.par.Packageroot.eval(), 'file_path': wrapper.filePath,
		'embedded_imports': True, 'no_source_links': True,
		'source_modules_on_sys_path': any((Path(entry or '.') / 'onnxtop_tables.py').is_file() for entry in sys.path),
	})


def finish(error=None):
	container = me.parent()
	report = container.fetch('report')
	if error:
		report['passed'] = False
		report['error'] = error
	destination = Path(os.environ.get('TD_HANDS_RESULT', Path(project.folder) / 'build/mediapipe-hands-live.json'))
	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
	if os.environ.get('TD_HANDS_RESULT'):
		project.quit(force=True)
	else:
		run("op(args[0]).destroy()", container.path, delayFrames=1)


def launch_live():
	root = Path(project.folder)
	parent_comp = op('/project1/ONNX_Native')
	assert parent_comp.op('_hands_validation') is None, 'Validation component already exists'
	dev = parent_comp.op('MediaPipeHands')
	assert dev is not None, 'Build the development wrapper first'
	def snapshot():
		return [(node.path, par.name, str(par.mode), str(par.val), par.expr, par.bindExpr)
			for node in [dev, *dev.findChildren()] for par in node.pars()
			if par.name in ('syncfile', 'loadonstart', 'file', 'externaltox', 'enableexternaltox', 'Packageroot')]
	before = snapshot()
	sources = {path: path.read_bytes() for path in [*root.glob('python/*.py'), *root.glob('tox/shared/*.tox')]}
	dev.op('component').module.export(dev, root / 'build/MediaPipeHands-restoration.tox')
	assert snapshot() == before, 'Export did not restore development parameter values/modes/expressions'
	assert all(path.read_bytes() == content for path, content in sources.items()), 'Export changed a development source file'
	container = parent_comp.create('baseCOMP', '_hands_validation')
	driver = container.create('textDAT', 'test_driver')
	driver.text = (root / 'tests/td_mediapipe_hands.py').read_text(encoding='utf-8')
	startup = container.create('executeDAT', 'startup')
	startup.par.create = True
	startup.par.start = True
	startup.text = '''import os
def launch():
	if os.environ.get('TD_HANDS_RESULT') and not parent().fetch('started', False):
		parent().store('started', True)
		run("op(args[0]).module.start()", parent().op('test_driver').path, delayFrames=5)
def onCreate():
	launch()
def onStart():
	launch()
'''
	container.save((root / 'build/RunMediaPipeHandsTests.tox').as_posix(), createFolders=True)
	driver.module.start()
	print(json.dumps({'scheduled': container.path, 'result': 'build/mediapipe-hands-live.json',
		'development_restored': True, 'source_files_unchanged': True}))


if me.name != 'test_driver':
	launch_live()