"""Builds model wrapper COMPs from wrappers/<Name>.json and exports them as release toxes.

Runs inside TD (scripts/build_wrappers.py through the bridge's /run). A generated wrapper is build
output: to change one, edit its spec or this file and rebuild. Never hand-edit the COMP.

Every wrapper has the same skeleton:
	input -> inference (the native custom OP, preset from the spec)
	      -> detections (Info DAT) -> tracker -> tracks / joints / bones
	      -> debug visualizers -> preview (transparent when Debug Overlay is off)
Outputs: TOP 1 debug overlay (or the image result), TOP 2 the input image, DATs tracks / joints /
bones, CHOP per-track values plus native stats.

The spec decides what differs: which custom OP and presets, the tracker mode, the output kind
(landmarks / boxes / image) and which native and tracker parameters the user sees. Everything
not exposed stays preset on the inner nodes.
"""

import json
from pathlib import Path

import td	# imported modules don't get TD's globals (project, op)

MODULES = ('object_tracker', 'mediapipe_skeletons', 'box_tracker_pipeline',
	'face_tracker_pipeline', 'pose_tracker_pipeline', 'onnxtop_tables')
DEBUG_BB = 'tox/shared/DebugBB.tox'
DEBUG_KEYPOINTS = 'tox/shared/DebugKeypoints.tox'
KINDS = ('landmarks', 'boxes', 'image')
ALWAYS_EXPOSED = ('Active',)
ALWAYS_PULSES = ('Reload',)

LAYOUT = {
	'input': (0, 200), 'inference': (240, 200), 'null_passthrough': (480, 200),
	'detections': (480, -120), 'tracker': (720, -120),
	'tracks': (960, -120), 'joints': (960, -360), 'bones': (960, -600),
	'debug_boxes': (1200, 0), 'debug_keypoints': (1200, -360),
	'overlay': (1440, 0), 'transparent': (1440, 200), 'preview': (1680, 0),
	'debug': (1920, 0), 'passthrough': (1920, 200),
	'out_tracks': (1920, -240), 'out_joints': (1920, -480), 'out_bones': (1920, -720),
	'inference_stats': (720, -960), 'channels': (1440, -960), 'out_channels': (1920, -960),
	'channels_callbacks': (1440, -1200), 'component': (0, -1200), 'local': (240, -1200),
	'model_pulses': (480, -1200),
	'tracker/in_info': (0, 0),
	'tracker/script_tracks': (240, 0), 'tracker/out_tracks': (720, 0),
	'tracker/script_tracks_callbacks': (240, -160),
	'tracker/script_joints': (480, -400), 'tracker/out_joints': (720, -400),
	'tracker/script_joints_callbacks': (480, -560),
	'tracker/script_bones': (480, -800), 'tracker/out_bones': (720, -800),
	'tracker/script_bones_callbacks': (480, -960),
	'local/modules': (0, 0),
	'local/modules/object_tracker': (0, 0), 'local/modules/mediapipe_skeletons': (0, -240),
	'local/modules/box_tracker_pipeline': (280, 0),
	'local/modules/face_tracker_pipeline': (280, -240),
	'local/modules/pose_tracker_pipeline': (280, -480),
	'local/modules/onnxtop_tables': (560, -240),
}


def root():
	return Path(td.project.folder)


def spec_names():
	return sorted(path.stem for path in (root() / 'wrappers').glob('*.json'))


def load_spec(name):
	spec = json.loads((root() / 'wrappers' / f'{name}.json').read_text(encoding='utf-8'))
	assert spec.get('name') == name, f'[WrapperBuilder] {name}.json names itself {spec.get("name")!r}'
	assert spec.get('kind') in KINDS, f'[WrapperBuilder] {name}: kind must be one of {KINDS}'
	return spec


def version():
	path = root() / 'VERSION'
	return path.read_text(encoding='utf-8').strip() if path.exists() else '0.0.0'


def _connect(destination, *sources):
	for index, source in enumerate(sources):
		destination.inputConnectors[index].connect(source.outputConnectors[0])


def _synced_dat(parent_comp, name, relative):
	dat = parent_comp.create('textDAT', name)
	dat.par.file = relative
	dat.par.syncfile = True
	assert dat.text.strip(), f'[WrapperBuilder] {relative} did not load'
	return dat


def _tuplet(node, name):
	"""A parameter tuplet by tuplet name (Dynamicsize) or member name (Dynamicsize1)."""
	for parameter in node.pars():
		if parameter.tupletName == name or parameter.name == name:
			return parameter.tuplet
	names = sorted({parameter.tupletName for parameter in node.pars() if parameter.name[0].isupper()})
	raise KeyError(f'[WrapperBuilder] {node.path} has no parameter {name!r}. Custom parameters: {names}')


def _expose(page, node, name, exposed_name=None):
	"""Copy a parameter onto the wrapper's page and drive the inner one from it. Returns the
	new pars and the names of any pulses (forwarded by the model_pulses DAT)."""
	tuplet = _tuplet(node, name)
	group = page.appendPar(exposed_name or tuplet[0].tupletName, par=tuplet[0])
	pulses = []
	for target, source in zip(group, tuplet):
		if source.isPulse:
			pulses.append(target.name)
			continue
		target.val = source.eval()
		target.default = source.default
		# .eval() so menu values forward by name even when the wrapper's menu is shorter.
		source.expr = f'parent().par.{target.name}.eval()'
	return group, pulses


def _restrict_menu(parameter, choices):
	names, labels = list(choices), list(choices.values())
	current = parameter.eval()
	parameter.menuNames = names
	parameter.menuLabels = labels
	if current not in names:
		parameter.val = names[0]
	if parameter.default not in names:
		parameter.default = names[0]


def _add_code(comp):
	_synced_dat(comp, 'component', 'python/wrapper_runtime.py')
	modules = comp.create('baseCOMP', 'local').create('baseCOMP', 'modules')
	for module_name in MODULES:
		_synced_dat(modules, module_name, f'python/{module_name}.py')


def _build_tracker(comp, mode):
	tracker = comp.create('baseCOMP', 'tracker')
	tables = comp.op('local/modules/onnxtop_tables').module
	tables.setup_parameters(tracker)
	tracker.par.Trackermode = mode
	tables.setup_parameters(tracker)
	info = tracker.create('inDAT', 'in_info')
	tracks = None
	for index, suffix in enumerate(('tracks', 'joints', 'bones'), 1):
		script = tracker.create('scriptDAT', 'script_' + suffix)
		arguments = 'scriptOp, parent(), scriptOp.inputs[0]' if suffix == 'tracks' else 'scriptOp, parent()'
		tracker.op('script_' + suffix + '_callbacks').text = (
			f'import onnxtop_tables\ndef onCook(scriptOp):\n\tonnxtop_tables.write_{suffix}({arguments})\n')
		_connect(script, info if suffix == 'tracks' else tracks)
		if suffix == 'tracks':
			tracks = script
		output = tracker.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		_connect(output, script)
	_connect(tracker, comp.op('detections'))
	return tracker


def _load_shared(comp, name, relative):
	node = comp.loadTox((root() / relative).as_posix())
	node.name = name
	node.par.externaltox = relative	# provenance; instances are refreshed by rebuilding
	node.par.enableexternaltox = False
	return node


def _build_visuals(comp, kind):
	"""Wire the overlay (or image result) into `preview`, a Switch whose input 0 is transparent."""
	source = comp.op('input')
	transparent = comp.create('constantTOP', 'transparent')
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	transparent.par.outputresolution = 'custom'
	transparent.par.resolutionw.expr = "op('input').width"
	transparent.par.resolutionh.expr = "op('input').height"
	preview = comp.create('switchTOP', 'preview')
	if kind == 'image':
		overlay = comp.create('resolutionTOP', 'overlay')
		_connect(overlay, comp.op('inference'))
		overlay.par.outputresolution = 'custom'
		overlay.par.resolutionw.expr = "op('input').width"
		overlay.par.resolutionh.expr = "op('input').height"
		_connect(preview, transparent, overlay)
		preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
		return
	boxes = _load_shared(comp, 'debug_boxes', DEBUG_BB)
	boxes.par.Inputaspectratio.expr = 'parent().par.Inputaspectratio'
	_connect(boxes, source, comp.op('tracks'))
	if kind == 'boxes':
		overlay = comp.create('nullTOP', 'overlay')
		_connect(overlay, boxes)
		_connect(preview, transparent, overlay)
		preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
		return
	keypoints = _load_shared(comp, 'debug_keypoints', DEBUG_KEYPOINTS)
	_connect(keypoints, comp.op('joints'), comp.op('bones'))
	overlay = comp.create('compositeTOP', 'overlay')
	overlay.par.operand = 'over'
	_connect(overlay, keypoints, boxes)
	_connect(preview, transparent, overlay, boxes, keypoints)
	preview.par.index.expr = '0 if not parent().par.Debugoverlay else parent().par.Debuglayers.menuIndex + 1'


def _build_pages(comp, spec, plugin, tracker):
	model_page = comp.appendCustomPage('Model')
	pulses = []
	for name in (*ALWAYS_EXPOSED, *spec.get('model', ()), *ALWAYS_PULSES):
		_, found = _expose(model_page, plugin, name)
		pulses += found
	for name, choices in spec.get('menus', {}).items():
		_restrict_menu(comp.par[name], choices)
	if tracker is not None and spec.get('tracking'):
		tracking_page = comp.appendCustomPage('Tracking')
		for name in spec['tracking']:
			assert _tuplet(tracker, name)[0].page.name == 'Tracking', f'[WrapperBuilder] {name} is not a tracking parameter'
			_expose(tracking_page, tracker, name, 'Track' + name.lower())
	debug_page = comp.appendCustomPage('Debug')
	overlay = debug_page.appendToggle('Debugoverlay', label='Debug Overlay')[0]
	overlay.val = overlay.default = True
	if spec['kind'] == 'landmarks':
		layers = debug_page.appendMenu('Debuglayers', label='Overlay Layers')[0]
		layers.menuNames = ['both', 'boxes', 'joints']
		layers.menuLabels = ['Both', 'Bounding Boxes', 'Joints']
		layers.val = layers.default = 'both'
		layers.enableExpr = 'me.par.Debugoverlay.eval()'
	aspect = debug_page.appendFloat('Inputaspectratio', label='Input Aspect Ratio')[0]
	aspect.expr = "me.op('input').width / max(1, me.op('input').height)"
	aspect.readOnly = True
	about = comp.appendCustomPage('About')
	release = about.appendStr('Version', label='Version')[0]
	release.val = release.default = version()
	release.readOnly = True
	if pulses:
		forward = comp.create('parameterexecuteDAT', 'model_pulses')
		forward.par.op = '..'
		forward.par.pars = ' '.join(pulses)
		forward.par.valuechange = False
		forward.par.onpulse = True
		forward.text = "def onPulse(par):\n\tparent().op('inference').par[par.name].pulse()\n"


def build(spec, parent_comp):
	"""(Re)build one wrapper inside parent_comp, keeping an existing one's position and wires."""
	name = spec['name']
	position, inputs, outputs = (parent_comp.nodeX, parent_comp.nodeY), [], []
	old = parent_comp.op(name)
	if old is not None:
		position = (old.nodeX, old.nodeY)
		inputs = [connector.connections[0].owner if connector.connections else None for connector in old.inputConnectors]
		outputs = [(index, connection.owner, connection.index) for index, connector in enumerate(old.outputConnectors)
			for connection in connector.connections]
		old.destroy()
	comp = parent_comp.create('baseCOMP', name)
	comp.nodeX, comp.nodeY = position
	comp.store('wrapper_spec', spec)
	_add_code(comp)
	comp.create('inTOP', 'input')
	plugin = comp.create(spec['plugin'], 'inference')
	for parameter, value in spec.get('preset', {}).items():
		_tuplet(plugin, parameter)[0].val = value
	_connect(plugin, comp.op('input'))
	info = comp.create('infoDAT', 'detections')
	info.par.op = 'inference'
	stats = comp.create('infoCHOP', 'inference_stats')
	stats.par.op = 'inference'
	passthrough = comp.create('nullTOP', 'null_passthrough')
	_connect(passthrough, comp.op('input'))
	tracker = _build_tracker(comp, spec['tracker']) if spec.get('tracker') else None
	_build_pages(comp, spec, plugin, tracker)
	if tracker is not None:
		for suffix in ('tracks', 'joints', 'bones'):
			comp.create('selectDAT', suffix).par.dat = 'tracker/out_' + suffix
	else:
		comp.create('selectDAT', 'tracks').par.dat = 'detections'
	_build_visuals(comp, spec['kind'])
	for index, (suffix, source) in enumerate((('debug', 'preview'), ('passthrough', 'null_passthrough')), 1):
		output = comp.create('outTOP', suffix)
		output.par.connectorder = index
		_connect(output, comp.op(source))
	for index, suffix in enumerate(('tracks', 'joints', 'bones') if tracker is not None else ('tracks',), 1):
		output = comp.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		_connect(output, comp.op(suffix))
	channels = comp.create('scriptCHOP', 'channels')
	comp.op('channels_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_channels(scriptOp, parent())\n")
	_connect(comp.create('outCHOP', 'out_channels'), channels)
	for node in comp.findChildren():
		node.viewer = False
	for path, (x, y) in LAYOUT.items():
		node = comp.op(path)
		if node is not None:
			node.nodeX, node.nodeY = x, y
	comp.par.opviewer = 'debug'
	for index, source in enumerate(inputs):
		if source is not None:
			comp.inputConnectors[index].connect(source)
	for index, owner, input_index in outputs:
		comp.outputConnectors[index].connect(owner.inputConnectors[input_index])
	return comp


def export(comp, destination):
	"""Save comp as a self-contained tox: DAT file syncs and external-tox links are cleared in
	the saved copy, then restored, so the development COMP keeps its links."""
	changed = []
	try:
		for node in [comp, *comp.findChildren()]:
			settings = [('externaltox', ''), ('enableexternaltox', False)]
			if node.family == 'DAT':
				settings = [('syncfile', False), ('loadonstart', False), ('file', '')] + settings
			for name, value in settings:
				parameter = getattr(node.par, name, None)
				if parameter is None:
					continue
				changed.append((parameter, parameter.val, parameter.expr, parameter.bindExpr, parameter.mode))
				parameter.mode = type(parameter.mode).CONSTANT	# ParMode isn't reachable from a module
				parameter.val = value
		destination = Path(destination)
		comp.save(destination.as_posix(), createFolders=True)
		return destination.as_posix()
	finally:
		for parameter, value, expression, binding, mode in reversed(changed):
			parameter.val = value
			parameter.expr = expression
			parameter.bindExpr = binding
			parameter.mode = mode


def check_export(path):
	"""Load an exported tox into a throwaway COMP and list anything still linked to dev files."""
	probe = td.op('/project1').create('baseCOMP', '_wrapper_export_check')
	try:
		subject = probe.loadTox(str(path))
		problems = []
		folder = td.project.folder.lower()
		for node in [subject, *subject.findChildren()]:
			for name in ('syncfile', 'externaltox', 'enableexternaltox'):
				parameter = getattr(node.par, name, None)
				if parameter is not None and parameter.eval() not in ('', False, None, 0):
					problems.append(f'{node.path}.{name}={parameter.eval()}')
			for parameter in node.pars():
				for text in (str(parameter.val), parameter.expr or ''):	# raw: eval() can raise
					if folder in text.replace('\\', '/').lower():
						problems.append(f'{node.path}.{parameter.name}={text}')
		return problems
	finally:
		probe.destroy()
