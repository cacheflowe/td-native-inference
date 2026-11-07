"""Frame-sliced integration check for the default-off model workbench; run via /run."""

import json
from pathlib import Path
import traceback


def schedule():
	run('args[0].module.step()', me, delayFrames=10)


def finish(error=None):
	container = me.parent()
	demo = container.fetch('demo')
	for name in container.fetch('names'):
		demo.par[name.capitalize()] = False
	source = container.fetch('previous_source')
	for connector in reversed(demo.inputConnectors):
		connector.disconnect()
	demo.inputConnectors[0].connect(source.outputConnectors[0])
	container.fetch('image').destroy()
	report = container.fetch('report')
	report['passed'] = error is None
	if error:
		report['error'] = error
	destination = Path(project.folder) / 'build/model-wrappers-live.json'
	destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
	run('args[0].destroy()', container, delayFrames=1)


def step():
	container = me.parent()
	demo = container.fetch('demo')
	report = container.fetch('report')
	try:
		index = container.fetch('index', 0)
		names = container.fetch('names')
		stage = container.fetch('stage', 'idle')
		if stage == 'idle':
			for name in names:
				assert not demo.op(name).allowCooking, f'{name} is cooking while inactive'
			if index == len(names):
				finish()
				return
			name = names[index]
			wrapper = demo.op(name)
			container.fetch('initial_counts')[name] = wrapper.op('inference').totalCooks
			image = container.fetch('image')
			filename = 'thumbs_up.jpg' if name == 'MediaPipeHands' else 'portrait.jpg' if name in ('MediaPipeFace', 'MediaPipeFaceDetector', 'YuNet', 'HSEmotion') else 'pose.jpg'
			image.par.file = (Path(project.folder) / 'build' / filename).as_posix()
			image.cook(force=True)
			assert not image.errors(), image.errors()
			assert image.width > 128 and image.height > 128, f'Fixture did not load: {image.par.file.eval()}'
			demo.par[name.capitalize()] = True
			container.store('stage', 'running')
			container.store('attempts', 0)
		elif stage == 'running':
			name = names[index]
			wrapper = demo.op(name)
			assert wrapper.allowCooking and wrapper.par.Active.eval(), f'{name} did not wake'
			wrapper.op('input').cook(force=True)
			assert wrapper.op('input').width == container.fetch('image').width, f'{name}: input {wrapper.op("input").width}, fixture {container.fetch("image").width}, demo {demo.op("input").width}, null {demo.op("input_frame").width}'
			for other in names:
				if other != name:
					assert not demo.op(other).allowCooking, f'{other} unexpectedly cooking'
			for suffix in ('out_tracks', 'out_channels', 'debug', 'passthrough'):
				wrapper.op(suffix).cook(force=True)
			stats = wrapper.op('inference_stats')
			stats.cook(force=True)
			native = {channel.name: float(channel[0]) for channel in stats.chans()}
			errors = wrapper.errors(recurse=True)
			assert not errors, f'{name}: {errors}'
			attempts = container.fetch('attempts') + 1
			container.store('attempts', attempts)
			positive = native.get('num_detections', 0) > 0
			ready = native.get('loaded', 0) == 1
			if not ready or (not positive and attempts < 15):
				assert attempts < 120, f'{name} did not load: {native}; {wrapper.warnings(recurse=True)}'
				schedule()
				return
			assert wrapper.op('inference').totalCooks > container.fetch('initial_counts')[name]
			assert wrapper.op('channels')['loaded'][0] == 1
			assert wrapper.op('tracks').numRows >= 1
			image = wrapper.op('debug').numpyArray()
			assert image is not None and image.shape[0] > 1 and image.shape[1] > 1
			fixture = container.fetch('image')
			if name == 'Yolo26Seg':
				import numpy as np
				assert np.allclose(image, wrapper.op('inference').numpyArray()), 'YOLO segmentation result differs from its native matte'
				assert wrapper.op('out_boxes') is not None, 'YOLO segmentation boxes output missing'
			else:
				assert (image.shape[1], image.shape[0]) == (fixture.width, fixture.height), f'{name}: debug resolution mismatch'
			assert (wrapper.op('passthrough').width, wrapper.op('passthrough').height) == (fixture.width, fixture.height), f'{name}: passthrough resolution mismatch'
			wrapper.par.Debugoverlay = False
			wrapper.op('debug').cook(force=True)
			import numpy as np
			assert np.allclose(wrapper.op('debug').numpyArray(), 0), f'{name} debug-off is not transparent'
			wrapper.par.Debugoverlay = True
			report['models'][name] = dict(native=native, rows=wrapper.op('tracks').numRows,
				positive_detections=positive, output_shape=list(image.shape), loaded=True,
				warnings=wrapper.warnings(recurse=True))
			demo.par[name.capitalize()] = False
			container.store('stage', 'stopped')
		elif stage == 'stopped':
			wrapper = demo.op(names[index])
			assert not wrapper.allowCooking
			container.store('stopped_count', wrapper.op('inference').totalCooks)
			container.store('stage', 'check_stopped')
		else:
			wrapper = demo.op(names[index])
			assert wrapper.op('inference').totalCooks == container.fetch('stopped_count'), 'Disabled native kept cooking'
			container.store('index', index + 1)
			container.store('stage', 'idle')
		schedule()
	except Exception:
		finish(traceback.format_exc())


def check_demo_composite(demo):
	import numpy as np

	container = me.parent()
	frame = container.create('constantTOP', 'input_frame')
	frame.par.resolutionw = 32
	frame.par.resolutionh = 24
	names = ['Front', 'Back']
	container.store('completed_wrappers', names)
	page = container.appendCustomPage('Models')
	for name, color in zip(names, ((1, 0, 0), (0, 1, 0))):
		page.appendToggle(name)[0].val = True
		wrapper = container.create('baseCOMP', name)
		wrapper.store('model_wrapper_spec', {'kind': 'landmarks'})
		constant = wrapper.create('constantTOP', 'image')
		constant.par.resolutionw = 32
		constant.par.resolutionh = 24
		for channel, value in zip(('colorr', 'colorg', 'colorb'), color):
			constant.par[channel] = value
		constant.par.alpha = 1
		output = wrapper.create('outTOP', 'debug')
		output.inputConnectors[0].connect(constant.outputConnectors[0])
		output.cook(force=True)
	helper = demo.op('component').module
	helper.setup_demo_composite(container)
	output = container.op('out_debug')
	output.cook(force=True)
	assert np.allclose(output.numpyArray(), [1, 0, 0, 1]), 'Over layer order is incorrect'
	container.par.Front = False
	container.op('Front').allowCooking = False
	count = container.op('Front/image').totalCooks
	output.cook(force=True)
	assert np.allclose(output.numpyArray(), [0, 1, 0, 1]), 'Disabled front layer did not clear'
	assert container.op('Front/image').totalCooks == count, 'Disabled layer cooked'
	container.par.Back = False
	container.op('Back').allowCooking = False
	counts = {name: container.op(name + '/image').totalCooks for name in names}
	for _ in range(3):
		output.cook(force=True)
		assert np.allclose(output.numpyArray(), 0), 'All-off output is not transparent'
	assert counts == {name: container.op(name + '/image').totalCooks for name in names}
	assert (output.width, output.height) == (32, 24)
	assert output.par.connectorder.eval() == 1
	assert container.op('out_passthrough').par.connectorder.eval() == 2
	return dict(over_order=True, disabled_layers_cleared=True, all_off_transparent=True,
		inactive_cook_counts_stable=True, output_size=[32, 24])


def check_segmentation_output(demo):
	import numpy as np

	wrapper = me.parent().create('baseCOMP', 'segmentation_probe')
	wrapper.appendCustomPage('Debug').appendToggle('Debugoverlay')[0].val = True
	for name, color in (('input', (0, 0, 1, 1)), ('inference', (0, 1, 0, 1)),
		('overlay', (1, 0, 0, 1)), ('transparent', (0, 0, 0, 0))):
		node = wrapper.create('constantTOP', name)
		for channel, value in zip(('colorr', 'colorg', 'colorb', 'alpha'), color):
			node.par[channel] = value
	passthrough = wrapper.create('nullTOP', 'null_passthrough')
	passthrough.inputConnectors[0].connect(wrapper.op('inference').outputConnectors[0])
	preview = wrapper.create('switchTOP', 'preview')
	for index, (name, node) in enumerate((('debug', preview), ('passthrough', passthrough)), 1):
		output = wrapper.create('outTOP', name)
		output.par.connectorder = index
		output.inputConnectors[0].connect(node.outputConnectors[0])
	demo.op('component').module.setup_segmentation_output(wrapper)
	for name, expected in (('debug', 'inference'), ('passthrough', 'input'), ('out_boxes', 'overlay')):
		wrapper.op(name).cook(force=True)
		assert np.allclose(wrapper.op(name).numpyArray(), wrapper.op(expected).numpyArray()), f'Wrong segmentation {name} source'
	wrapper.par.Debugoverlay = False
	wrapper.op('debug').cook(force=True)
	assert np.allclose(wrapper.op('debug').numpyArray(), 0), 'Disabled segmentation result is not transparent'
	assert wrapper.fetch('model_wrapper_spec')['image_output']
	return dict(primary_matches_inference=True, passthrough_matches_input=True,
		boxes_preserved=True, disabled_transparent=True)


def launch():
	root = Path(project.folder)
	demo = op('/project1/ModelDemo') or op('/project1/ONNX_Native/ModelDemo')
	assert demo is not None, 'ModelDemo not found'
	parent_comp = demo.parent()
	assert demo.fetch('model_builder_stage') == 'complete'
	assert parent_comp.op('_model_validation') is None
	names = demo.fetch('completed_wrappers')
	for name in names:
		wrapper = demo.op(name)
		assert not wrapper.par.Active.eval() and not wrapper.allowCooking
		assert wrapper.op('inference').type == 'cplusplus'
		assert wrapper.inputs[0] == demo.op('input_frame')
	container = parent_comp.create('baseCOMP', '_model_validation')
	container.store('demo', demo)
	container.store('names', names)
	container.store('initial_counts', {})
	container.store('previous_source', demo.inputs[0])
	container.store('report', {'passed': False, 'models': {}, 'default_off': True})
	for name in names:
		subject = container.loadTox((root / 'tox' / f'{name}.tox').as_posix())
		subject.par.Packageroot = root.as_posix()
		assert not subject.par.Active.eval()
		for node in [subject, *subject.findChildren()]:
			for parameter_name in ('syncfile', 'loadonstart', 'externaltox', 'enableexternaltox'):
				parameter = getattr(node.par, parameter_name, None)
				assert parameter is None or not parameter.eval(), f'{name}: external link {node.path}.{parameter_name}'
			if node.family == 'DAT' and getattr(node.par, 'file', None) is not None:
				assert not node.par.file.eval(), f'{name}: linked DAT {node.path}'
		subject.destroy()
	subject = container.loadTox((root / 'tox/ModelDemo.tox').as_posix())
	subject.par.Packageroot = root.as_posix()
	for name in names:
		assert not subject.par[name.capitalize()].eval()
		assert not subject.op(name).allowCooking, f'Exported demo {name} cooks by default'
		assert subject.op(name).par.Packageroot.eval() == root.as_posix()
	subject.destroy()
	container.fetch('report')['demo_export_default_off'] = True
	image = parent_comp.create('moviefileinTOP', '_model_test_image')
	container.store('image', image)
	for connector in reversed(demo.inputConnectors):
		connector.disconnect()
	demo.inputConnectors[0].connect(image.outputConnectors[0])
	driver = container.create('textDAT', 'test_driver')
	driver.text = (root / 'tests/td_model_wrappers.py').read_text(encoding='utf-8')
	driver.module.schedule()
	print(json.dumps({'scheduled': len(names), 'result': 'build/model-wrappers-live.json', 'exports_embedded': True}))


if me.name not in ('test_driver', 'composite_driver'):
	launch()