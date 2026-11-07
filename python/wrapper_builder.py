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

# Columns 200 apart, in data-flow order. Bands, top to bottom: the input-image lane
# (null_passthrough -> passthrough, nothing else at that height), the inference/visualizer band,
# the DAT outputs (Select DATs right beside their Out DATs), the CHOP outputs, then code/support
# nodes under the input. Wires only run between adjacent columns or along an empty lane, and the
# sources of every multi-input node sit in the same top-to-bottom order as its inputs.
COLUMN = 200
LAYOUT = {
	'null_passthrough': (2, 420), 'passthrough': (7, 420),
	'input': (0, 180), 'inference': (1, 180), 'detections': (2, 180), 'tracker': (3, 160),
	'debug_boxes': (4, 250), 'debug_keypoints': (4, 60),
	'transparent': (5, 350), 'overlay': (5, 160), 'preview': (6, 230), 'debug': (7, 230),
	'tracks': (6, -100), 'out_tracks': (7, -100),
	'joints': (6, -240), 'out_joints': (7, -240),
	'bones': (6, -380), 'out_bones': (7, -380),
	'inference_stats': (5, -540), 'channels': (6, -540), 'out_channels': (7, -540),
	'component': (0, -100), 'model_pulses': (1, -100), 'local': (0, -260),
	'tracker/in_info': (0, 0), 'tracker/script_tracks': (1, 0), 'tracker/out_tracks': (3, 0),
	'tracker/script_joints': (2, -150), 'tracker/out_joints': (3, -150),
	'tracker/script_bones': (2, -300), 'tracker/out_bones': (3, -300),
	'local/modules': (0, 0),
	'local/modules/object_tracker': (0, 0), 'local/modules/mediapipe_skeletons': (0, -150),
	'local/modules/box_tracker_pipeline': (1, 0), 'local/modules/face_tracker_pipeline': (1, -150),
	'local/modules/pose_tracker_pipeline': (1, -300), 'local/modules/onnxtop_tables': (2, -150),
}
# Image models have no tracker or visualizers: the resized result sits right after inference.
LAYOUT_IMAGE = {'overlay': (2, 320), 'detections': (2, 120)}
DOCKED = ('channels_callbacks', 'tracker/script_tracks_callbacks',
	'tracker/script_joints_callbacks', 'tracker/script_bones_callbacks')


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
	"""Build `overlay` (the visualizers, or the resized image result) and `preview`, a two-input
	Switch: transparent when Debug Overlay is off, so the visualizers don't cook then."""
	transparent = comp.create('constantTOP', 'transparent')
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	transparent.par.outputresolution = 'custom'
	transparent.par.resolutionw.expr = "op('input').width"
	transparent.par.resolutionh.expr = "op('input').height"
	preview = comp.create('switchTOP', 'preview')
	preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
	if kind == 'image':
		overlay = comp.create('resolutionTOP', 'overlay')
		_connect(overlay, comp.op('inference'))
		overlay.par.outputresolution = 'custom'
		overlay.par.resolutionw.expr = "op('input').width"
		overlay.par.resolutionh.expr = "op('input').height"
		_connect(preview, transparent, overlay)
		return
	tracker = comp.op('tracker')
	def table(suffix):
		# Straight from the tracker's outputs (adjacent column); the Select DATs feed the Out DATs.
		if tracker is None:
			return comp.op(suffix).outputConnectors[0]
		return next(c for c in tracker.outputConnectors if c.outOP == tracker.op('out_' + suffix))
	boxes = _load_shared(comp, 'debug_boxes', DEBUG_BB)
	boxes.par.Inputaspectratio.expr = 'parent().par.Inputaspectratio'
	# DebugBB only reads its image input's resolution.
	boxes.inputConnectors[0].connect(comp.op('null_passthrough'))
	boxes.inputConnectors[1].connect(table('tracks'))
	if kind == 'boxes':
		overlay = comp.create('nullTOP', 'overlay')
		_connect(overlay, boxes)
		_connect(preview, transparent, overlay)
		return
	keypoints = _load_shared(comp, 'debug_keypoints', DEBUG_KEYPOINTS)
	keypoints.inputConnectors[0].connect(table('joints'))
	keypoints.inputConnectors[1].connect(table('bones'))
	overlay = comp.create('compositeTOP', 'overlay')
	overlay.par.operand = 'over'
	overlay.par.swaporder = True	# boxes wired first (it sits above), keypoints drawn over it
	_connect(overlay, boxes, keypoints)
	# Overlay Layers: Both composites; Bounding Boxes / Joints pass one input through.
	overlay.par.selectinput.expr = "parent().par.Debuglayers.eval() != 'both'"
	overlay.par.inputindex.expr = "0 if parent().par.Debuglayers.eval() == 'boxes' else 1"
	_connect(preview, transparent, overlay)


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


def _layout(comp, kind):
	positions = dict(LAYOUT, **(LAYOUT_IMAGE if kind == 'image' else {}))
	for path, (column, y) in positions.items():
		node = comp.op(path)
		if node is not None:
			node.nodeX, node.nodeY = column * COLUMN, y
	for path in DOCKED:
		callbacks = comp.op(path)
		host = comp.op(path.removesuffix('_callbacks'))
		if callbacks is not None and host is not None:
			callbacks.nodeX, callbacks.nodeY = host.nodeX, host.nodeY - 110
			callbacks.dock = host


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
	_layout(comp, spec['kind'])
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
