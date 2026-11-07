"""Run twice via TD /run: register native DLLs, then configure/export the model demo."""

import json
from pathlib import Path


ROOT = Path(project.folder)
PARENT = op('/project1/ONNX_Native')
MODULES = ('object_tracker', 'mediapipe_skeletons', 'box_tracker_pipeline',
	'face_tracker_pipeline', 'pose_tracker_pipeline', 'onnxtop_tables')
SPECS = (
	('MediaPipeHands', 'MP_Gesture', 'mp_hands', 'landmarks'),
	('MediaPipePose', 'MP_Pose', 'mp_pose', 'landmarks'),
	('MediaPipeFace', 'MP_Face', 'mp_face', 'landmarks'),
	('MediaPipeFaceDetector', 'MP_FaceDetector', 'mp_facedet', 'landmarks'),
	('MediaPipeHolistic', 'MP_Holistic', 'mp_holistic', 'landmarks'),
	('MediaPipeSegment', 'MP_Segment', None, 'image'),
	('Yolo26Pose', 'Onnx_YOLO_Pose', 'pose', 'landmarks'),
	('Yolo26Detect', 'Onnx_YOLO_Object', 'detect', 'boxes'),
	('Yolo26Seg', 'Onnx_YOLO_SEG', 'segment', 'boxes'),
	('MoveNet', 'Onnx_MoveNet', 'movenet', 'landmarks'),
	('YuNet', 'Onnx_YuNet', 'face', 'landmarks'),
	('DepthAnything', 'Onnx_DepthAnything', None, 'image'),
	('HSEmotion', 'Onnx_Emotion', None, 'classifier'),
)


def connect(destination, *sources):
	for index, source in enumerate(sources):
		destination.inputConnectors[index].connect(source.outputConnectors[0])


def linked_dat(comp, name, source):
	dat = comp.create('textDAT', name)
	dat.par.file = (ROOT / source).as_posix()
	dat.par.syncfile = True
	assert dat.text.strip(), f'Missing DAT source: {source}'
	return dat


def expose(page, source, prefix=''):
	name = prefix + source.tupletName.lower() if prefix else source.tupletName
	group = page.appendPar(name, par=source)
	for parameter, original in zip(group, source.tuplet):
		parameter.val = original.eval()
		if not original.isPulse:
			original.expr = f'parent().par.{parameter.name}.eval()'
	return group


def create_tracker(comp, mode):
	tracker = comp.create('baseCOMP', 'tracker')
	tables = comp.op('local/modules/onnxtop_tables').module
	tables.setup_parameters(tracker)
	tracker.par.Trackermode = mode
	tables.setup_parameters(tracker)
	info = tracker.create('inDAT', 'in_info')
	tracks = None
	for index, suffix in enumerate(('tracks', 'joints', 'bones'), 1):
		script = tracker.create('scriptDAT', 'script_' + suffix)
		callbacks = tracker.op('script_' + suffix + '_callbacks')
		assert callbacks is not None
		arguments = 'scriptOp, parent(), scriptOp.inputs[0]' if suffix == 'tracks' else 'scriptOp, parent()'
		callbacks.text = f'import onnxtop_tables\ndef onCook(scriptOp):\n\tonnxtop_tables.write_{suffix}({arguments})\n'
		connect(script, info if suffix == 'tracks' else tracks)
		if suffix == 'tracks':
			tracks = script
		output = tracker.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		connect(output, script)
	connect(tracker, comp.op('detections'))
	tracking = comp.appendCustomPage('Tracking')
	seen = set()
	for parameter in list(tracker.customPars):
		if parameter.page.name == 'Tracking' and parameter.tupletName not in seen:
			seen.add(parameter.tupletName)
			expose(tracking, parameter, 'Track')
	return tracker


def set_image_reference(comp, raw_mask=False):
	passthrough = comp.op('null_passthrough')
	passthrough.inputConnectors[0].disconnect()
	connect(passthrough, comp.op('input'))
	comp.par.Inputaspectratio.expr = "me.op('input').width / max(1, me.op('input').height)"
	boxes = comp.op('debug_boxes')
	if boxes is not None:
		boxes.inputConnectors[0].disconnect()
		boxes.inputConnectors[0].connect(comp.op('input').outputConnectors[0])
	else:
		comp.op('overlay').destroy()
		overlay = comp.create('resolutionTOP', 'overlay')
		connect(overlay, comp.op('inference'))
		comp.op('preview').inputConnectors[1].connect(overlay.outputConnectors[0])
		overlay.par.outputresolution = 'custom'
		overlay.par.resolutionw.expr = "op('input').width"
		overlay.par.resolutionh.expr = "op('input').height"
	comp.op('transparent').par.resolutionw.expr = "op('input').width"
	comp.op('transparent').par.resolutionh.expr = "op('input').height"
	if raw_mask:
		helper = comp.parent().op('component') or comp.op('component')
		helper.module.setup_segmentation_output(comp)


def configure(comp, spec, demo):
	name, source_name, mode, kind = spec
	source = PARENT.op(source_name)
	plugin = comp.op('inference')
	assert getattr(plugin.par, 'Active', None) is not None, f'DLL not registered: {name}; retry later'
	setup = comp.appendCustomPage('Setup')
	setup.appendFolder('Packageroot', label='Package Root')[0].val = ROOT.as_posix()
	comp.store('package_root_hint', ROOT.as_posix())
	model_page = comp.appendCustomPage('Model')
	seen = set()
	pulses = []
	for parameter in source.pars():
		if not parameter.name[0].isupper() or parameter.tupletName in seen:
			continue
		seen.add(parameter.tupletName)
		native = plugin.par[parameter.name]
		assert native is not None, f'Missing native parameter: {name}.{parameter.name}'
		group = expose(model_page, native)
		for target, original in zip(group, parameter.tuplet):
			if target.isPulse:
				pulses.append(target.name)
				continue
			target.val = original.eval()
			if target.name.lower().endswith('model'):
				path = Path(str(original.eval()))
				if str(path) != '.':
					relative = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
					target.expr = f"me.op('component').module.package_path(me, {relative!r})"
	comp.par.Active = False
	plugin.par.unloadplugin.expr = 'not parent().par.Active.eval()'
	plugin.par.plugin.expr = f"parent().op('component').module.package_path(parent(), 'plugins/{'MediaPipeTOP' if source_name.startswith('MP_') else 'OnnxTOP'}.dll')"
	if getattr(comp.par, 'Cachefolder', None) is not None:
		comp.par.Cachefolder.expr = f"me.op('component').module.package_path(me, 'cache/trt/{name}')"
	if getattr(comp.par, 'Task', None) is not None:
		if name == 'MediaPipeHands':
			comp.par.Task.menuNames = ['hands', 'gesture']
			comp.par.Task.menuLabels = ['Hands', 'Gesture']
			comp.par.Task = 'gesture'
			comp.par.Maxinstances = 2
		else:
			comp.par.Task.readOnly = True
	if getattr(comp.par, 'Mode', None) is not None:
		comp.par.Mode.readOnly = True
	debug = comp.appendCustomPage('Debug')
	debug.appendToggle('Debugoverlay', label='Debug Overlay')[0].val = True
	debug.appendFloat('Inputaspectratio', label='Input Aspect Ratio')[0].expr = "me.op('null_passthrough').width / max(1, me.op('null_passthrough').height)"
	comp.par.Inputaspectratio.readOnly = True
	controller = comp.create('parameterexecuteDAT', 'model_pulses')
	controller.par.op = '..'
	controller.par.pars = ' '.join(pulses)
	controller.par.onpulse = True
	controller.text = "def onPulse(par):\n\tparent().op('inference').par[par.name].pulse()\n"
	startup = comp.create('executeDAT', 'initialize')
	startup.par.create = True
	startup.par.start = True
	startup.text = "def launch():\n\trun(\"args[0].op('component').module.initialize(args[0])\", parent(), delayFrames=1)\ndef onCreate():\n\tlaunch()\ndef onStart():\n\tlaunch()\n"
	connect(plugin, comp.op('input'))
	info = comp.create('infoDAT', 'detections')
	info.par.op = 'inference'
	passthrough = comp.create('nullTOP', 'null_passthrough')
	connect(passthrough, comp.op('input') if kind in ('image', 'classifier') else plugin)
	if mode:
		create_tracker(comp, mode)
	if kind == 'classifier':
		face = comp.copy(demo.op('YuNet'), name='face_detector')
		face.par.Active.bindExpr = ''
		face.par.Active.expr = 'parent().par.Active.eval()'
		face.par.Packageroot.expr = 'parent().par.Packageroot.eval()'
		face.allowCooking = True
		connect(face, comp.op('input'))
		crops = comp.create('selectDAT', 'crop_boxes')
		crops.par.dat = 'face_detector/tracker/script_tracks'
		comp.par.Cropdat.expr = "me.op('crop_boxes')"
		setup.appendStr('Classprefix', label='Class Prefix')[0].val = 'emotion'
		setup.appendFloat('Classsmoothing', label='Class Smoothing')[0].val = 0.5
		tracks = comp.create('scriptDAT', 'tracks')
		comp.op('tracks_callbacks').text = "import onnxtop_tables\ndef onCook(scriptOp):\n\tonnxtop_tables.write_scores(scriptOp, parent(), scriptOp.inputs[0], scriptOp.inputs[1])\n"
		connect(tracks, crops, info)
	else:
		tracks = comp.create('selectDAT', 'tracks')
		tracks.par.dat = 'tracker/out_tracks' if mode else 'detections'
	if mode or kind == 'classifier':
		for suffix in ('joints', 'bones'):
			data = comp.create('selectDAT', suffix)
			data.par.dat = ('face_detector/' if kind == 'classifier' else '') + 'tracker/out_' + suffix
	if kind == 'image':
		overlay = comp.create('nullTOP', 'overlay')
		connect(overlay, plugin)
	else:
		boxes = comp.loadTox((ROOT / 'tox/shared/DebugBB.tox').as_posix())
		boxes.name = 'debug_boxes'
		boxes.par.externaltox = (ROOT / 'tox/shared/DebugBB.tox').as_posix()
		boxes.par.Inputaspectratio.expr = 'parent().par.Inputaspectratio'
		connect(boxes, passthrough, tracks)
		overlay = comp.create('nullTOP' if kind == 'boxes' else 'compositeTOP', 'overlay')
		if kind != 'boxes':
			overlay.par.operand = 'over'
		if kind in ('landmarks', 'classifier'):
			keypoints = comp.loadTox((ROOT / 'tox/shared/DebugKeypoints.tox').as_posix())
			keypoints.name = 'debug_keypoints'
			keypoints.par.externaltox = (ROOT / 'tox/shared/DebugKeypoints.tox').as_posix()
			connect(keypoints, comp.op('joints'), comp.op('bones'))
			connect(overlay, keypoints, boxes)
		else:
			connect(overlay, boxes)
	transparent = comp.create('constantTOP', 'transparent')
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	transparent.par.resolutionw.expr = "op('null_passthrough').width"
	transparent.par.resolutionh.expr = "op('null_passthrough').height"
	preview = comp.create('switchTOP', 'preview')
	preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
	connect(preview, transparent, overlay)
	if kind in ('landmarks', 'classifier'):
		comp.op('component').module.setup_debug_layers(comp)
	for index, (suffix, source_node) in enumerate((('debug', preview), ('passthrough', passthrough)), 1):
		output = comp.create('outTOP', suffix)
		output.par.connectorder = index
		connect(output, source_node)
	for index, suffix in enumerate(('tracks', 'joints', 'bones') if mode or kind == 'classifier' else ('tracks',), 1):
		output = comp.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		connect(output, comp.op(suffix))
	stats = comp.create('infoCHOP', 'inference_stats')
	stats.par.op = 'inference'
	channels = comp.create('scriptCHOP', 'channels')
	comp.op('channels_callbacks').text = "def onCook(scriptOp):\n\tparent().op('component').module.write_channels(scriptOp, parent())\ndef onGetCookLevel(scriptOp):\n\treturn CookLevel.WHEN_USED\n"
	output = comp.create('outCHOP', 'out_channels')
	connect(output, channels)
	set_image_reference(comp, name == 'Yolo26Seg')
	for node in comp.findChildren():
		node.viewer = False
	comp.op('component').module.layout(comp)
	if kind == 'classifier':
		for suffix, coordinates in {'face_detector': (240, -360), 'crop_boxes': (480, -360),
			'tracks_callbacks': (960, -1440)}.items():
			node = comp.op(suffix)
			node.nodeX, node.nodeY = coordinates
	destination = ROOT / 'tox' / f'{name}.tox'
	if destination.exists():
		backup = ROOT / 'build' / 'before-model-wrappers' / destination.name
		if not backup.exists():
			backup.parent.mkdir(parents=True, exist_ok=True)
			backup.write_bytes(destination.read_bytes())
	comp.op('component').module.export(comp, destination)
	control = demo.par[name.capitalize()]
	comp.par.Active.bindExpr = f'parent().par.{control.name}'
	comp.allowCooking = False
	connect(comp, demo.op('input_frame'))
	comp.store('model_wrapper_spec', dict(name=name, kind=kind, source=source_name, mode=mode,
		image_output=name in ('Yolo26Seg', 'MediaPipeSegment', 'DepthAnything')))
	return name


def finalize_demo(demo):
	if getattr(demo.par, 'Packageroot', None) is None:
		demo.appendCustomPage('Setup').appendFolder('Packageroot', label='Package Root')
		linked_dat(demo, 'component', 'python/mediapipe_hands_component.py')
	demo.par.Packageroot = ROOT.as_posix()
	demo.store('package_root_hint', ROOT.as_posix())
	for name, *_ in SPECS:
		demo.op(name).par.Packageroot.expr = 'parent().par.Packageroot.eval()'
	controller = demo.op('model_controls')
	controller.text = 'NAMES = ' + repr({name.capitalize(): name for name, *_ in SPECS}) + "\ndef onValueChange(par, prev):\n\tpar.owner.op(NAMES[par.name]).allowCooking = bool(par.eval())\ndef initialize():\n\tfor control, name in NAMES.items():\n\t\tparent().op(name).allowCooking = bool(parent().par[control].eval())\n"
	startup = demo.op('initialize') or demo.create('executeDAT', 'initialize')
	startup.par.create = True
	startup.par.start = True
	startup.text = "def launch():\n\trun(\"args[0].op('component').module.initialize(args[0]); args[0].op('model_controls').module.initialize()\", parent(), delayFrames=1)\ndef onCreate():\n\tlaunch()\ndef onStart():\n\tlaunch()\n"
	output = demo.op('out_input') or demo.op('out_passthrough') or demo.create('outTOP', 'out_input')
	connect(output, demo.op('input_frame'))
	output.nodeX, output.nodeY = 960, 200
	demo.op('component').nodeX, demo.op('component').nodeY = 0, -600
	startup.nodeX, startup.nodeY = 240, -600
	if not demo.inputs or demo.inputs[0] == PARENT.op('null1'):
		source = demo.parent().op('ModelDemoInput')
		if source is None:
			source = demo.parent().create('moviefileinTOP', 'ModelDemoInput')
			source.par.file = PARENT.op('people_walking').par.file.eval()
			source.nodeX, source.nodeY = demo.nodeX - 240, demo.nodeY
		for connector in reversed(demo.inputConnectors):
			connector.disconnect()
		connect(demo, source)
	demo.op('component').module.setup_demo_composite(demo)
	demo.op('component').module.export_demo(demo, ROOT / 'tox/ModelDemo.tox')


def build():
	assert PARENT is not None
	demo = op('/project1/ModelDemo') or PARENT.op('ModelDemo')
	if demo is not None:
		if demo.fetch('model_builder_stage', '') == 'complete':
			for name, source_name, mode, kind in SPECS:
				comp = demo.op(name)
				assert not demo.par[name.capitalize()].eval(), f'Disable {name} before refreshing'
				if kind == 'boxes':
					comp.op('overlay').destroy()
					overlay = comp.create('nullTOP', 'overlay')
					connect(overlay, comp.op('debug_boxes'))
					comp.op('preview').inputConnectors[1].connect(overlay.outputConnectors[0])
				set_image_reference(comp, name == 'Yolo26Seg')
				comp.op('component').module.layout(comp)
				comp.par.Active.bindExpr = ''
				comp.par.Active = False
				comp.allowCooking = True
				comp.op('component').module.export(comp, ROOT / 'tox' / f'{name}.tox')
				comp.par.Active.bindExpr = f'parent().par.{name.capitalize()}'
				comp.allowCooking = False
			finalize_demo(demo)
			print(json.dumps({'refreshed': len(SPECS)}))
			return
		assert demo.fetch('model_builder_stage', '') == 'registered', 'Existing ModelDemo is not an unfinished owned build'
		completed = demo.fetch('completed_wrappers', [])
		for spec in SPECS:
			if spec[0] not in completed:
				completed.append(configure(demo.op(spec[0]), spec, demo))
				demo.store('completed_wrappers', completed)
		finalize_demo(demo)
		demo.store('model_builder_stage', 'complete')
		print(json.dumps({'built': completed, 'demo': demo.path, 'all_inactive': True}, indent=2))
		return
	demo = op('/project1').create('baseCOMP', 'ModelDemo')
	demo.nodeX = 3000
	demo.nodeY = 500
	controls = demo.appendCustomPage('Models')
	input_top = demo.create('inTOP', 'input')
	input_top.nodeX, input_top.nodeY = 0, 0
	frame = demo.create('nullTOP', 'input_frame')
	frame.nodeX, frame.nodeY = 240, 0
	connect(frame, input_top)
	for index, spec in enumerate(SPECS):
		name, source_name, mode, kind = spec
		assert PARENT.op(source_name) is not None, f'Missing reference model: {source_name}'
		controls.appendToggle(name.capitalize(), label=name)[0].val = False
		comp = demo.create('baseCOMP', name)
		comp.nodeX, comp.nodeY = 600, -index * 200
		linked_dat(comp, 'component', 'python/mediapipe_hands_component.py')
		modules = comp.create('baseCOMP', 'local').create('baseCOMP', 'modules')
		for module_name in MODULES:
			linked_dat(modules, module_name, f'python/{module_name}.py')
		comp.create('inTOP', 'input')
		plugin = comp.create('cplusplusTOP', 'inference')
		plugin.par.plugin = (ROOT / 'plugins' / ('MediaPipeTOP.dll' if source_name.startswith('MP_') else 'OnnxTOP.dll')).as_posix()
		plugin.viewer = False
		plugin.cook(force=True)
	gate = demo.create('parameterexecuteDAT', 'model_controls')
	gate.par.op = '..'
	gate.par.pars = ' '.join(name.capitalize() for name, *_ in SPECS)
	gate.par.onpulse = False
	gate.text = 'NAMES = ' + repr({name.capitalize(): name for name, *_ in SPECS}) + "\ndef onValueChange(par, prev):\n\tpar.owner.op(NAMES[par.name]).allowCooking = bool(par.eval())\n"
	gate.nodeX, gate.nodeY = 0, -400
	demo.store('model_builder_stage', 'registered')
	print(json.dumps({'stage': 'native_plugins_registered', 'next': 'Run again to configure/export'}))


build()