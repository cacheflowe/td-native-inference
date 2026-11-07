"""Builds model wrapper COMPs from wrappers/<Name>.json and exports them as release toxes.

Runs inside TD (scripts/build_wrappers.py through the bridge's /run). A generated wrapper is build
output: to change one, edit its spec or this file and rebuild. Never hand-edit the COMP.

Every wrapper has the same skeleton:
	input -> inference (the native custom OP, preset from the spec)
	      -> detections (Info DAT) -> tracker -> tracks / joints / bones
	      -> debug visualizers -> preview (transparent when Debug Overlay is off)
Outputs: TOP 1 debug overlay (or the image result), TOP 2 the input image, DATs tracks / joints /
bones, CHOP per-track values plus native stats.

The spec decides what differs:
	plugin, preset   the custom OP type and its fixed settings (member names like Dynamicsize1 ok)
	kind             landmarks | boxes | image | classifier (second stage on a nested detector)
	tracker          onnxtop_tables tracker mode (omit for image models)
	result           'inference' makes TOP 1 the native image (e.g. a seg matte); boxes go to TOP 3
	model, tracking  the native / tracker parameters the user sees, in order; menus narrows menus
	detector, classes  classifier only: the nested detector spec and the score column settings
Everything not exposed stays preset on the inner nodes.
"""

import json
from pathlib import Path

import td	# imported modules don't get TD's globals (project, op)

MODULES = ('object_tracker', 'mediapipe_skeletons', 'box_tracker_pipeline',
	'face_tracker_pipeline', 'pose_tracker_pipeline', 'onnxtop_tables')
DEBUG_BB = 'tox/shared/DebugBB.tox'
DEBUG_KEYPOINTS = 'tox/shared/DebugKeypoints.tox'
KINDS = ('landmarks', 'boxes', 'image', 'classifier')
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
LAYOUT_KIND = {
	# No tracker or visualizers: the resized result sits right after inference.
	'image': {'overlay': (2, 320), 'detections': (2, 120)},
	# The nested detector feeds the crops (above) and the keypoint visualizer (below).
	'classifier': {'detector': (1, -20), 'crop_boxes': (2, 320), 'scores': (3, 160)},
}
LAYOUT_RESULT_INFERENCE = {'matte': (5, 160), 'overlay': (5, 20), 'out_boxes': (7, 20)}
DOCKED = ('channels_callbacks', 'scores_callbacks', 'tracker/script_tracks_callbacks',
	'tracker/script_joints_callbacks', 'tracker/script_bones_callbacks')

DEMO_ROW = 180


def root():
	return Path(td.project.folder)


def spec_names():
	return sorted(path.stem for path in (root() / 'wrappers').glob('*.json'))


def load_spec(name):
	spec = json.loads((root() / 'wrappers' / f'{name}.json').read_text(encoding='utf-8'))
	assert spec.get('name') == name, f'[WrapperBuilder] {name}.json names itself {spec.get("name")!r}'
	assert spec.get('kind') in KINDS, f'[WrapperBuilder] {name}: kind must be one of {KINDS}'
	assert spec['kind'] != 'classifier' or spec.get('detector'), f'[WrapperBuilder] {name}: classifier needs a detector'
	return spec


def version():
	path = root() / 'VERSION'
	return path.read_text(encoding='utf-8').strip() if path.exists() else '0.0.0'


def _connect(destination, *sources):
	for index, source in enumerate(sources):
		destination.inputConnectors[index].connect(source.outputConnectors[0])


def _output(comp, out_name):
	"""A COMP's output connector by its Out OP (mixed-family COMPs don't number them by family)."""
	return next(connector for connector in comp.outputConnectors if connector.outOP == comp.op(out_name))


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


def _set(node, name, value):
	parameter = getattr(node.par, name, None) or _tuplet(node, name)[0]
	parameter.val = value


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
		target.default = source.eval()
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


def _build_classifier(comp, spec):
	"""Second stage: a nested detector wrapper finds and tracks faces; this node classifies their
	crops and `scores` joins the class scores onto the detector's tracks."""
	detector = build(load_spec(spec['detector']), comp, name='detector')
	detector.par.Active.expr = 'parent().par.Active.eval()'
	detector.par.Debugoverlay = False	# its own overlay is unused; ours draws its tables
	_connect(detector, comp.op('input'))
	# script_tracks, not out_tracks: the crops must not depend on the scored table.
	comp.create('selectDAT', 'crop_boxes').par.dat = 'detector/tracker/script_tracks'
	scores = comp.create('scriptDAT', 'scores')
	comp.op('scores_callbacks').text = ('import onnxtop_tables\ndef onCook(scriptOp):\n'
		'\tonnxtop_tables.write_scores(scriptOp, parent(), scriptOp.inputs[0], scriptOp.inputs[1])\n')
	_connect(scores, comp.op('crop_boxes'), comp.op('detections'))
	return detector


def _load_shared(comp, name, relative):
	node = comp.loadTox((root() / relative).as_posix())
	node.name = name
	node.par.externaltox = relative	# provenance; instances are refreshed by rebuilding
	node.par.enableexternaltox = False
	return node


def _sources(comp, kind):
	"""Output connectors feeding the visualizers: tracker outputs, or the classifier's tables."""
	tracker, detector = comp.op('tracker'), comp.op('detector')
	if kind == 'classifier':
		return {'tracks': comp.op('scores').outputConnectors[0],
			'joints': _output(detector, 'out_joints'), 'bones': _output(detector, 'out_bones')}
	if tracker is not None:
		return {suffix: _output(tracker, 'out_' + suffix) for suffix in ('tracks', 'joints', 'bones')}
	return {'tracks': comp.op('tracks').outputConnectors[0]}


def _build_visuals(comp, spec):
	"""Build `overlay` (the visualizers, or the resized image result) and `preview`, a two-input
	Switch: transparent when Debug Overlay is off, so the visualizers don't cook then."""
	kind = spec['kind']
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
	sources = _sources(comp, kind)
	boxes = _load_shared(comp, 'debug_boxes', DEBUG_BB)
	boxes.par.Inputaspectratio.expr = 'parent().par.Inputaspectratio'
	# DebugBB only reads its image input's resolution.
	boxes.inputConnectors[0].connect(comp.op('null_passthrough'))
	boxes.inputConnectors[1].connect(sources['tracks'])
	if kind == 'boxes':
		overlay = comp.create('nullTOP', 'overlay')
		_connect(overlay, boxes)
	else:
		keypoints = _load_shared(comp, 'debug_keypoints', DEBUG_KEYPOINTS)
		keypoints.inputConnectors[0].connect(sources['joints'])
		keypoints.inputConnectors[1].connect(sources['bones'])
		overlay = comp.create('compositeTOP', 'overlay')
		overlay.par.operand = 'over'
		overlay.par.swaporder = True	# boxes wired first (it sits above), keypoints drawn over it
		_connect(overlay, boxes, keypoints)
		# Overlay Layers: Both composites; Bounding Boxes / Joints pass one input through.
		overlay.par.selectinput.expr = "parent().par.Debuglayers.eval() != 'both'"
		overlay.par.inputindex.expr = "0 if parent().par.Debuglayers.eval() == 'boxes' else 1"
	if spec.get('result') == 'inference':
		# The native image (e.g. a segmentation matte, at inference size) is the primary result;
		# referenced, not wired, so no wire crosses the visualizers.
		matte = comp.create('selectTOP', 'matte')
		matte.par.top = 'inference'
		_connect(preview, transparent, matte)
		boxes_out = comp.create('outTOP', 'out_boxes')
		boxes_out.par.connectorder = 3
		_connect(boxes_out, overlay)
	else:
		_connect(preview, transparent, overlay)


def _build_pages(comp, spec, plugin, tracker):
	model_page = comp.appendCustomPage('Model')
	pulses = []
	for name in (*ALWAYS_EXPOSED, *spec.get('model', ()), *ALWAYS_PULSES):
		_, found = _expose(model_page, plugin, name)
		pulses += found
	for name, choices in spec.get('menus', {}).items():
		_restrict_menu(comp.par[name], choices)
	if spec['kind'] == 'classifier':
		classes = spec.get('classes', {})
		prefix = model_page.appendStr('Classprefix', label='Class Column Prefix')[0]
		prefix.val = prefix.default = classes.get('prefix', 'class')
		prefix.help = 'Score columns become <prefix>_label and <prefix>_<class name>.'
		smoothing = model_page.appendFloat('Classsmoothing', label='Class Score Smoothing')[0]
		smoothing.val = smoothing.default = classes.get('smoothing', 0.5)
		smoothing.normMin, smoothing.normMax = 0, 0.99
		smoothing.help = 'Per-track smoothing of class scores (0 = raw, higher = steadier).'
	if tracker is not None and spec.get('tracking'):
		tracking_page = comp.appendCustomPage('Tracking')
		for name in spec['tracking']:
			assert _tuplet(tracker, name)[0].page.name == 'Tracking', f'[WrapperBuilder] {name} is not a tracking parameter'
			_expose(tracking_page, tracker, name, 'Track' + name.lower())
	debug_page = comp.appendCustomPage('Debug')
	label = 'Show Result' if spec['kind'] == 'image' or spec.get('result') == 'inference' else 'Debug Overlay'
	overlay = debug_page.appendToggle('Debugoverlay', label=label)[0]
	overlay.val = overlay.default = True
	if spec['kind'] in ('landmarks', 'classifier'):
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


def _layout(comp, spec):
	positions = dict(LAYOUT, **LAYOUT_KIND.get(spec['kind'], {}))
	if spec.get('result') == 'inference':
		positions.update(LAYOUT_RESULT_INFERENCE)
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


def _take_place(parent_comp, name):
	"""Destroy an existing node, returning its position and wires so a rebuild can restore them."""
	old = parent_comp.op(name)
	if old is None:
		return None, [], []
	position = (old.nodeX, old.nodeY)
	inputs = [connector.connections[0].owner if connector.connections else None for connector in old.inputConnectors]
	outputs = [(index, connection.owner, connection.index) for index, connector in enumerate(old.outputConnectors)
		for connection in connector.connections]
	old.destroy()
	return position, inputs, outputs


def _restore_wires(comp, inputs, outputs):
	for index, source in enumerate(inputs):
		if source is not None and index < len(comp.inputConnectors):
			comp.inputConnectors[index].connect(source)
	for index, owner, input_index in outputs:
		if index < len(comp.outputConnectors):
			comp.outputConnectors[index].connect(owner.inputConnectors[input_index])


def build(spec, parent_comp, name=None):
	"""(Re)build one wrapper inside parent_comp, keeping an existing one's position and wires."""
	name = name or spec['name']
	position, inputs, outputs = _take_place(parent_comp, name)
	comp = parent_comp.create('baseCOMP', name)
	if position is not None:
		comp.nodeX, comp.nodeY = position
	comp.store('wrapper_spec', spec)
	_add_code(comp)
	comp.create('inTOP', 'input')
	plugin = comp.create(spec['plugin'], 'inference')
	for parameter, value in spec.get('preset', {}).items():
		_set(plugin, parameter, value)
	_connect(plugin, comp.op('input'))
	info = comp.create('infoDAT', 'detections')
	info.par.op = 'inference'
	stats = comp.create('infoCHOP', 'inference_stats')
	stats.par.op = 'inference'
	passthrough = comp.create('nullTOP', 'null_passthrough')
	_connect(passthrough, comp.op('input'))
	kind = spec['kind']
	tracker = _build_tracker(comp, spec['tracker']) if spec.get('tracker') else None
	detector = _build_classifier(comp, spec) if kind == 'classifier' else None
	_build_pages(comp, spec, plugin, tracker)
	if tracker is not None:
		for suffix in ('tracks', 'joints', 'bones'):
			comp.create('selectDAT', suffix).par.dat = 'tracker/out_' + suffix
	elif detector is not None:
		comp.create('selectDAT', 'tracks').par.dat = 'scores'
		for suffix in ('joints', 'bones'):
			comp.create('selectDAT', suffix).par.dat = 'detector/out_' + suffix
	else:
		comp.create('selectDAT', 'tracks').par.dat = 'detections'
	_build_visuals(comp, spec)
	for index, (suffix, source) in enumerate((('debug', 'preview'), ('passthrough', 'null_passthrough')), 1):
		output = comp.create('outTOP', suffix)
		output.par.connectorder = index
		_connect(output, comp.op(source))
	tables = ('tracks', 'joints', 'bones') if comp.op('joints') is not None else ('tracks',)
	for index, suffix in enumerate(tables, 1):
		output = comp.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		_connect(output, comp.op(suffix))
	channels = comp.create('scriptCHOP', 'channels')
	comp.op('channels_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_channels(scriptOp, parent())\n")
	_connect(comp.create('outCHOP', 'out_channels'), channels)
	for node in comp.findChildren():
		node.viewer = False
	_layout(comp, spec)
	comp.par.opviewer = 'debug'
	_restore_wires(comp, inputs, outputs)
	return comp


def build_demo(parent_comp, names, name='ModelDemo'):
	"""(Re)build the workbench: one wrapper per spec, each enabled from the Models page. A disabled
	wrapper doesn't cook at all (allowCooking, set from outside it: a COMP can't wake itself)."""
	specs = [load_spec(spec_name) for spec_name in names]
	# Over puts earlier inputs on top: box/joint overlays first, opaque image results last.
	specs.sort(key=lambda spec: spec['kind'] == 'image' or spec.get('result') == 'inference')
	position, inputs, outputs = _take_place(parent_comp, name)
	demo = parent_comp.create('baseCOMP', name)
	if position is not None:
		demo.nodeX, demo.nodeY = position
	models = demo.appendCustomPage('Models')
	controls = {}
	for spec in specs:
		toggle = models.appendToggle(spec['name'].capitalize(), label=spec.get('label', spec['name']))[0]
		toggle.val = toggle.default = False
		controls[toggle.name] = spec['name']
	about = demo.appendCustomPage('About')
	release = about.appendStr('Version', label='Version')[0]
	release.val = release.default = version()
	release.readOnly = True
	source = demo.create('inTOP', 'input')
	frame = demo.create('nullTOP', 'input_frame')
	_connect(frame, source)
	transparent = demo.create('constantTOP', 'transparent')
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	transparent.par.outputresolution = 'custom'
	transparent.par.resolutionw.expr = "op('input_frame').width"
	transparent.par.resolutionh.expr = "op('input_frame').height"
	composite = demo.create('compositeTOP', 'debug_composite')
	composite.par.operand = 'over'
	for index, spec in enumerate(specs):
		toggle = next(key for key, value in controls.items() if value == spec['name'])
		wrapper = build(spec, demo)
		wrapper.allowCooking = False
		wrapper.par.Active.expr = f'parent().par.{toggle}.eval()'
		_connect(wrapper, frame)
		wrapper.nodeX, wrapper.nodeY = 2 * COLUMN, -index * DEMO_ROW
		# Referenced, not wired: a disabled wrapper keeps its last frame, so pick transparent.
		layer = demo.create('selectTOP', 'layer_' + spec['name'])
		layer.par.top.expr = f"'{spec['name']}/debug' if parent().par.{toggle} else 'transparent'"
		layer.nodeX, layer.nodeY = 3 * COLUMN, -index * DEMO_ROW + 20
		layer.viewer = False
		composite.inputConnectors[index].connect(layer)
	output = demo.create('outTOP', 'out_debug')
	output.par.connectorder = 1
	_connect(output, composite)
	passthrough = demo.create('outTOP', 'out_input')
	passthrough.par.connectorder = 2
	_connect(passthrough, frame)
	gate = demo.create('parameterexecuteDAT', 'model_controls')
	gate.par.op = '..'
	gate.par.pars = ' '.join(controls)
	gate.par.onpulse = False
	gate.text = (f'NAMES = {controls!r}\n'
		'def onValueChange(par, prev):\n\tpar.owner.op(NAMES[par.name]).allowCooking = bool(par.eval())\n'
		'def apply():\n\tfor control, name in NAMES.items():\n'
		'\t\tparent().op(name).allowCooking = bool(parent().par[control].eval())\n')
	startup = demo.create('executeDAT', 'initialize')
	startup.par.create = True
	startup.par.start = True
	startup.text = ("def onCreate():\n\trun(\"args[0].module.apply()\", parent().op('model_controls'), delayFrames=1)\n"
		"def onStart():\n\tonCreate()\n")
	middle = -(len(specs) - 1) * DEMO_ROW // 2
	for node, (column, y) in {source: (0, 200), frame: (1, 200), passthrough: (5, 200), transparent: (3, 200),
		composite: (4, middle), output: (5, middle), gate: (0, -200), startup: (0, -350)}.items():
		node.nodeX, node.nodeY = column * COLUMN, y
	for node in (source, frame, transparent, composite, gate, startup):
		node.viewer = False
	demo.par.opviewer = 'out_debug'
	_restore_wires(demo, inputs, outputs)
	return demo


def export_demo(demo, destination):
	"""Export with every model off (and not cooking), then restore the live selection."""
	toggles = [demo.par[name] for name in demo.op('model_controls').module.NAMES]
	states = [(toggle, toggle.val) for toggle in toggles]
	try:
		for toggle in toggles:
			toggle.val = False
		demo.op('model_controls').module.apply()
		return export(demo, destination)
	finally:
		for toggle, value in states:
			toggle.val = value
		demo.op('model_controls').module.apply()


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
		subject.allowCooking = False
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
