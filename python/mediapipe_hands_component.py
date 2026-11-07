"""Component-local support for portable model wrappers and their demo."""

from pathlib import Path
import re


def setup_debug_layers(comp):
	parameter = getattr(comp.par, 'Debuglayers', None)
	if parameter is None:
		page = next((page for page in comp.customPages if page.name == 'Debug'), None)
		if page is None:
			page = comp.appendCustomPage('Debug')
		parameter = page.appendMenu('Debuglayers', label='Overlay Layers')[0]
		parameter.menuNames = ['both', 'boxes', 'joints']
		parameter.menuLabels = ['Both', 'Bounding Boxes', 'Joints']
		parameter.val = 'both'
	parameter.enableExpr = 'me.par.Debugoverlay.eval()'
	preview = comp.op('preview')
	for index, name in enumerate(('transparent', 'overlay', 'debug_boxes', 'debug_keypoints')):
		preview.inputConnectors[index].connect(comp.op(name).outputConnectors[0])
	preview.par.index.expr = '0 if not parent().par.Debugoverlay else parent().par.Debuglayers.menuIndex + 1'


def setup_segmentation_output(comp):
	preview = comp.op('preview')
	for connector in reversed(preview.inputConnectors):
		connector.disconnect()
	preview.inputConnectors[0].connect(comp.op('transparent').outputConnectors[0])
	preview.inputConnectors[1].connect(comp.op('inference').outputConnectors[0])
	preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
	passthrough = comp.op('null_passthrough')
	passthrough.inputConnectors[0].disconnect()
	passthrough.inputConnectors[0].connect(comp.op('input').outputConnectors[0])
	boxes = comp.op('out_boxes') or comp.create('outTOP', 'out_boxes')
	boxes.par.connectorder = 3
	boxes.inputConnectors[0].connect(comp.op('overlay').outputConnectors[0])
	boxes.nodeX, boxes.nodeY = 1920, -1080
	spec = dict(comp.fetch('model_wrapper_spec', {}))
	spec['image_output'] = True
	comp.store('model_wrapper_spec', spec)


def setup_demo_composite(comp):
	names = comp.fetch('completed_wrappers')
	assert names, 'No model wrappers registered in this demo'
	transparent = comp.op('debug_transparent') or comp.create('constantTOP', 'debug_transparent')
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	transparent.par.resolutionw.expr = "op('input_frame').width"
	transparent.par.resolutionh.expr = "op('input_frame').height"
	transparent.nodeX, transparent.nodeY = 600, 300
	layers = []
	for name in names:
		wrapper = comp.op(name)
		layer = comp.op('layer_' + name) or comp.create('switchTOP', 'layer_' + name)
		for connector in reversed(layer.inputConnectors):
			connector.disconnect()
		layer.inputConnectors[0].connect(transparent.outputConnectors[0])
		debug_connector = next(connector for connector in wrapper.outputConnectors if connector.outOP == wrapper.op('debug'))
		layer.inputConnectors[1].connect(debug_connector)
		layer.par.index.expr = f'int(parent().par.{name.capitalize()}.eval())'
		layer.nodeX, layer.nodeY = 960, wrapper.nodeY
		layer.viewer = False
		layers.append((name, layer))
	layers.sort(key=lambda item: comp.op(item[0]).fetch('model_wrapper_spec', {}).get('image_output', False)
		or comp.op(item[0]).fetch('model_wrapper_spec', {}).get('kind') == 'image')
	composite = comp.op('debug_composite') or comp.create('compositeTOP', 'debug_composite')
	composite.par.operand = 'over'
	for connector in reversed(composite.inputConnectors):
		connector.disconnect()
	for index, (_, layer) in enumerate(layers):
		composite.inputConnectors[index].connect(layer.outputConnectors[0])
	composite.nodeX, composite.nodeY = 1280, 0
	output = comp.op('out_debug') or comp.create('outTOP', 'out_debug')
	output.par.connectorder = 1
	output.inputConnectors[0].connect(composite.outputConnectors[0])
	output.nodeX, output.nodeY = 1520, 0
	passthrough = comp.op('out_input') or comp.op('out_passthrough')
	if passthrough is None:
		passthrough = comp.create('outTOP', 'out_passthrough')
		passthrough.inputConnectors[0].connect(comp.op('input_frame').outputConnectors[0])
	passthrough.par.connectorder = 2
	passthrough.nodeX, passthrough.nodeY = 1520, 200
	comp.par.opviewer = 'out_debug'
	return [name for name, _ in layers]


def layout(comp):
	positions = {
		'input': (0, 200), 'inference': (240, 200), 'null_passthrough': (480, 200),
		'detections': (480, -120), 'tracker': (720, -120),
		'tracks': (960, -120), 'joints': (960, -360), 'bones': (960, -600),
		'debug_boxes': (1200, 0), 'debug_keypoints': (1200, -360),
		'overlay': (1440, 0), 'transparent': (1440, 200), 'preview': (1680, 0),
		'debug': (1920, 0), 'passthrough': (1920, 200),
		'out_tracks': (1920, -240), 'out_joints': (1920, -480), 'out_bones': (1920, -720),
		'inference_stats': (720, -960), 'channels': (1440, -960), 'out_channels': (1920, -960),
		'channels_callbacks': (1440, -1200), 'component': (0, -1200), 'local': (240, -1200),
		'initialize': (0, -1440), 'model_pulses': (240, -1440),
		'tracker/in_info': (0, 0),
		'tracker/script_tracks': (240, 0), 'tracker/out_tracks': (720, 0),
		'tracker/script_tracks_callbacks': (240, -160),
		'tracker/script_joints': (480, -400), 'tracker/out_joints': (720, -400),
		'tracker/script_joints_callbacks': (480, -560),
		'tracker/script_bones': (480, -800), 'tracker/out_bones': (720, -800),
		'tracker/script_bones_callbacks': (480, -960), 'tracker/parexec_mode': (0, -1200),
		'local/modules': (0, 0),
		'local/modules/object_tracker': (0, 0), 'local/modules/mediapipe_skeletons': (0, -240),
		'local/modules/box_tracker_pipeline': (280, 0),
		'local/modules/face_tracker_pipeline': (280, -240),
		'local/modules/pose_tracker_pipeline': (280, -480),
		'local/modules/onnxtop_tables': (560, -240),
	}
	for path, (horizontal, vertical) in positions.items():
		node = comp.op(path)
		if node is not None:
			node.nodeX = horizontal
			node.nodeY = vertical


def default_package_root(comp):
	reference = comp.par.externaltox.eval()
	source = Path(reference) if reference else Path(comp.filePath)
	if source.suffix.lower() == '.tox':
		root = source.parent.parent.as_posix()
		comp.store('package_root_hint', root)
		return root
	return comp.fetch('package_root_hint', '')


def package_path(comp, relative):
	root = comp.par.Packageroot.eval()
	if not root:
		raise ValueError('Package Root is empty; select the folder containing plugins and models')
	return (Path(root) / relative).as_posix()


def initialize(comp):
	parameter = comp.par.Packageroot
	if parameter.expr == "me.op('component').module.default_package_root(me)":
		root = default_package_root(comp)
		if root:
			comp.par.Packageroot = root


def write_channels(script_op, comp):
	script_op.clear()
	script_op.numSamples = 1
	tracks = comp.op('tracks')
	tracks.cook()
	if tracks.numRows:
		header = [cell.val for cell in tracks.row(0)]
		if 'track_id' in header:
			for row in tracks.rows()[1:]:
				tid = row[header.index('track_id')].val
				for name, cell in zip(header, row):
					try:
						value = float(cell.val)
					except ValueError:
						continue
					channel = re.sub(r'[^a-zA-Z0-9_]', '_', f't{tid}_{name}')
					script_op.appendChan(channel)[0] = value
	stats = comp.op('inference_stats')
	stats.cook(force=True)
	for channel in stats.chans():
		script_op.appendChan(channel.name)[0] = channel[0]


def export_demo(comp, destination):
	states = []
	try:
		for name in comp.fetch('completed_wrappers'):
			parameter = comp.par[name.capitalize()]
			wrapper = comp.op(name)
			states.append((parameter, parameter.val, parameter.expr, parameter.bindExpr, parameter.mode, wrapper, wrapper.allowCooking))
			parameter.val = False
			wrapper.allowCooking = False
		return export(comp, destination)
	finally:
		for parameter, value, expression, binding, mode, wrapper, cooking in states:
			parameter.val = value
			parameter.expr = expression
			parameter.bindExpr = binding
			parameter.mode = mode
			wrapper.allowCooking = cooking


def export(comp, destination):
	changed = []
	hint = comp.storage.pop('package_root_hint', None)
	try:
		for node in [comp, *comp.findChildren()]:
			settings = [('externaltox', ''), ('enableexternaltox', False)]
			if node.family == 'DAT':
				settings = [('syncfile', False), ('loadonstart', False), ('file', '')] + settings
			if node == comp:
				settings += [('Packageroot', None)]
			for name, value in settings:
				par = getattr(node.par, name, None)
				if par is None:
					continue
				changed.append((par, par.val, par.expr, par.bindExpr, par.mode))
				if name == 'Packageroot':
					par.expr = "me.op('component').module.default_package_root(me)"
				else:
					par.val = value
		return comp.save(str(destination), createFolders=True)
	finally:
		for par, value, expression, binding, mode in reversed(changed):
			par.val = value
			par.expr = expression
			par.bindExpr = binding
			par.mode = mode
		comp.storage.pop('package_root_hint', None)
		if hint is not None:
			comp.store('package_root_hint', hint)