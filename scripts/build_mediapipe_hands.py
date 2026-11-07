"""Run twice via TD's /run route: construct, then configure the loaded DLL."""

import json
from pathlib import Path


ROOT = Path(project.folder)
PARENT = op('/project1/ONNX_Native')
NAME = '_MediaPipeHands_build'
MODULES = (
	'object_tracker', 'mediapipe_skeletons', 'box_tracker_pipeline',
	'face_tracker_pipeline', 'pose_tracker_pipeline', 'onnxtop_tables',
)


def linked_dat(comp, name, source):
	dat = comp.create('textDAT', name)
	dat.par.file = (ROOT / source).as_posix()
	dat.par.syncfile = True
	assert dat.text.strip(), f'Module did not load: {name}'
	return dat


def connect(destination, *sources):
	for index, source in enumerate(sources):
		destination.inputConnectors[index].connect(source.outputConnectors[0])


def configure(comp):
	plugin = comp.op('inference')
	assert getattr(plugin.par, 'Task', None) is not None, 'DLL is still loading; retry the second invocation'
	setup = comp.appendCustomPage('Setup')
	root = setup.appendFolder('Packageroot', label='Package Root')[0]
	root.val = ROOT.as_posix()
	comp.store('package_root_hint', ROOT.as_posix())
	model_page = comp.appendCustomPage('Model')
	for name in ('Task', 'Gesturemodel', 'Handmodel', 'Active', 'Reload', 'Maxinstances',
		'Detectwidth', 'Mindetection', 'Minpresence', 'Mintracking', 'Outputz', 'Outputworld'):
		source = plugin.par[name]
		parameter = model_page.appendPar(name, par=source)[0]
		parameter.val = source.eval()
		if not source.isPulse:
			source.expr = f'parent().par.{name}.eval()'
	comp.par.Task.menuNames = ['hands', 'gesture']
	comp.par.Task.menuLabels = ['Hands', 'Gesture']
	comp.par.Task = 'gesture'
	comp.par.Maxinstances = 2
	comp.par.Gesturemodel.expr = "me.op('component').module.package_path(me, 'models/mediapipe/gesture_recognizer.task')"
	comp.par.Handmodel.expr = "me.op('component').module.package_path(me, 'models/mediapipe/hand_landmarker.task')"
	plugin.par.plugin.expr = "parent().op('component').module.package_path(parent(), 'plugins/MediaPipeTOP.dll')"
	tracker = comp.op('tracker')
	tracker.par.Trackermode = 'mp_hands'
	tracking = comp.appendCustomPage('Tracking')
	for source in tracker.customPars:
		if source.page.name != 'Tracking':
			continue
		parameter = tracking.appendPar(source.name, par=source)[0]
		parameter.val = source.eval()
		source.expr = f'parent().par.{source.name}'
	debug = comp.appendCustomPage('Debug')
	debug.appendToggle('Debugoverlay', label='Debug Overlay')[0].val = True
	debug.appendFloat('Inputaspectratio', label='Input Aspect Ratio')[0].expr = "me.op('null_passthrough').width / max(1, me.op('null_passthrough').height)"
	comp.par.Inputaspectratio.readOnly = True
	pulses = comp.create('parameterexecuteDAT', 'model_pulses')
	pulses.par.op = '..'
	pulses.par.pars = 'Reload'
	pulses.par.onpulse = True
	pulses.text = "def onPulse(par):\n\tparent().op('inference').par.Reload.pulse()\n"
	startup = comp.create('executeDAT', 'initialize')
	startup.par.create = True
	startup.par.start = True
	startup.text = "def launch():\n\trun(\"args[0].op('component').module.initialize(args[0])\", parent(), delayFrames=1)\ndef onCreate():\n\tlaunch()\ndef onStart():\n\tlaunch()\n"
	passthrough = comp.create('nullTOP', 'null_passthrough')
	connect(passthrough, plugin)
	tracks = comp.create('selectDAT', 'tracks')
	tracks.par.dat = 'tracker/out_tracks'
	joints = comp.create('selectDAT', 'joints')
	joints.par.dat = 'tracker/out_joints'
	bones = comp.create('selectDAT', 'bones')
	bones.par.dat = 'tracker/out_bones'
	boxes = comp.loadTox((ROOT / 'tox/shared/DebugBB.tox').as_posix())
	boxes.name = 'debug_boxes'
	boxes.par.externaltox = (ROOT / 'tox/shared/DebugBB.tox').as_posix()
	boxes.par.Inputaspectratio.expr = 'parent().par.Inputaspectratio'
	connect(boxes, passthrough, tracks)
	keypoints = comp.loadTox((ROOT / 'tox/shared/DebugKeypoints.tox').as_posix())
	keypoints.name = 'debug_keypoints'
	keypoints.par.externaltox = (ROOT / 'tox/shared/DebugKeypoints.tox').as_posix()
	connect(keypoints, joints, bones)
	overlay = comp.create('compositeTOP', 'overlay')
	overlay.par.operand = 'over'
	connect(overlay, keypoints, boxes)
	transparent = comp.create('constantTOP', 'transparent')
	for name in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[name] = 0
	transparent.par.resolutionw.expr = "op('null_passthrough').width"
	transparent.par.resolutionh.expr = "op('null_passthrough').height"
	preview = comp.create('switchTOP', 'preview')
	preview.par.index.expr = 'int(parent().par.Debugoverlay)'
	connect(preview, transparent, overlay)
	for index, (name, source) in enumerate((('debug', preview), ('passthrough', passthrough)), 1):
		output = comp.create('outTOP', name)
		output.par.connectorder = index
		connect(output, source)
	for index, source in enumerate((tracks, joints, bones), 1):
		output = comp.create('outDAT', f'out_{source.name}')
		output.par.connectorder = index
		connect(output, source)
	stats = comp.create('infoCHOP', 'inference_stats')
	stats.par.op = 'inference'
	channels = comp.create('scriptCHOP', 'channels')
	callbacks = comp.op('channels_callbacks') or comp.create('textDAT', 'channels_callbacks')
	callbacks.par.language = 'python'
	callbacks.text = "def onCook(scriptOp):\n\tparent().op('component').module.write_channels(scriptOp, parent())\ndef onGetCookLevel(scriptOp):\n\treturn CookLevel.WHEN_USED\n"
	channels.par.callbacks = 'channels_callbacks'
	output = comp.create('outCHOP', 'out_channels')
	connect(output, channels)
	comp.op('component').module.setup_debug_layers(comp)
	for node in comp.children:
		node.viewer = False
	comp.op('component').module.layout(comp)
	comp.name = 'MediaPipeHands'
	comp.unstore('hands_builder_owned')
	comp.op('component').module.export(comp, ROOT / 'tox/MediaPipeHands.tox')
	print(json.dumps({'built': comp.path, 'exported': 'tox/MediaPipeHands.tox',
		'plugin_errors': plugin.errors()}, indent=2))


def build():
	assert PARENT is not None, 'Open the native-inference development project first'
	existing = PARENT.op(NAME)
	if existing is not None and existing.fetch('hands_builder_owned', False):
		if getattr(existing.par, 'Packageroot', None) is None:
			try:
				configure(existing)
			except Exception:
				existing.destroy()
				raise
			return
		existing.destroy()
	assert PARENT.op(NAME) is None and PARENT.op('MediaPipeHands') is None, 'Refusing to overwrite an existing component'
	assert not (ROOT / 'tox/MediaPipeHands.tox').exists(), 'Refusing to overwrite an existing tox'
	comp = PARENT.create('baseCOMP', NAME)
	comp.store('hands_builder_owned', True)
	try:
		linked_dat(comp, 'component', 'python/mediapipe_hands_component.py')
		modules = comp.create('baseCOMP', 'local').create('baseCOMP', 'modules')
		for name in MODULES:
			linked_dat(modules, name, f'python/{name}.py')
		input_top = comp.create('inTOP', 'input')
		plugin = comp.create('cplusplusTOP', 'inference')
		plugin.par.plugin = (ROOT / 'plugins/MediaPipeTOP.dll').as_posix()
		connect(plugin, input_top)
		tracker = comp.copy(PARENT.op('MPTables_Gesture'), name='tracker')
		info = comp.create('infoDAT', 'detections')
		info.par.op = 'inference'
		connect(tracker, info)
		plugin.cook(force=True)
		print(json.dumps({'stage': 'constructed', 'next': 'Run this script again to configure and export'}))
	except Exception:
		comp.destroy()
		raise


build()