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
	tracker          the Tracker DAT's mode (tracking_pars.MODES; omit for image models)
	tracker_defaults tracker parameter values that differ from the mode's defaults (e.g. smoothing
	                 for Holistic GPU, which unlike MediaPipe's own Holistic has no landmark filter)
	result           'inference' makes TOP 1 the native image (e.g. a seg matte); boxes go to TOP 3
	instances        seg only: TOP 4 colors each tracked object's pixels (Matte Encoding = Instances)
	extra_outputs    image only: the inference node's other color buffers as TOPs 3, 4, ...; the
	                 VisionDemo then gets a Show menu, its first entry labelled "result_label"
	flow             image only: an optical flow model (frame pair): TOP 1 the flow in the Optical
	                 Flow TOP's format, TOP 3 the flow as color (_build_flow)
	model, tracking  the native / tracker parameters the user sees, in order; menus narrows menus
	detector, classes  classifier only: the nested detector spec and the score column settings
	input            'audio' builds an audio wrapper instead (_build_audio): audio CHOP in, the
	                 MediaPipe audio CHOP, kind transcript (DATs + closed-caption TOP)
Everything not exposed stays preset on the inner nodes.
"""

import json
from pathlib import Path

import td	# imported modules don't get TD's globals (project, op)

import tracking_pars

# A landmarks spec with a "detector" nests a person detector before OnnxTOP (Pose Crops, Holistic);
# its optional "converter" (RTMW3D: "rtmw_convert", the Tracker DAT's Convert RTMW3D table) reshapes
# the Info DAT for the tracker.
DEBUG_BB = 'tox/shared/DebugBB.tox'
DEBUG_KEYPOINTS = 'tox/shared/DebugKeypoints.tox'
KINDS = ('landmarks', 'boxes', 'image', 'classifier', 'transcript', 'scores', 'speech')
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
	'track_values': (4, -680), 'track_channels': (5, -680), 'tracker_stats': (4, -820),
	'component': (0, -100), 'model_pulses': (1, -100), 'local': (0, -260),
	'tracker/in_info': (0, 0), 'tracker/script_tracks': (1, 0), 'tracker/out_tracks': (3, 0),
	'tracker/script_joints': (2, -150), 'tracker/out_joints': (3, -150),
	'tracker/script_bones': (2, -300), 'tracker/out_bones': (3, -300),
	'detector': (1, -20), 'crop_boxes': (2, 320), 'rtmw_info': (2, 60), 'detector_gate': (0, 320),
}
LAYOUT_KIND = {
	# No tracker or visualizers: the resized result sits right after inference.
	'image': {'overlay': (2, 320), 'detections': (2, 120)},
	# The nested detector feeds the crops (above) and the keypoint visualizer (below).
	'classifier': {'detector': (1, -20), 'crop_boxes': (2, 320), 'scores': (3, 160)},
}
LAYOUT_RESULT_INFERENCE = {'matte': (5, 160), 'overlay': (5, 20), 'out_boxes': (7, 20),
	# Instances: a band under the CHOP outputs. instance_matte above instance_lookup (input order).
	'instance_matte': (5, -680), 'instance_colors': (4, -820), 'instance_lookup': (5, -820),
	'instances': (6, -750), 'out_instances': (7, -750)}
DOCKED = ('rtmw_info_callbacks', 'channels_callbacks', 'track_channels_callbacks', 'scores_callbacks', 'instance_colors_callbacks', 'instances_pixel', 'tracker/script_tracks_callbacks',
	'tracker/script_joints_callbacks', 'tracker/script_bones_callbacks')

# Audio wrappers, same banding: the caption chain on top, the DAT outputs, then the CHOP outputs.
LAYOUT_AUDIO = {
	'input': (0, 180), 'inference': (1, 180), 'keep_cooking': (1, 330),
	'caption_lines': (3, 180), 'caption_text': (4, 300), 'caption_canvas': (4, 160),
	'captions': (5, 230), 'transparent': (5, 380), 'preview': (6, 230), 'debug': (7, 230),
	'transcript': (2, -100), 'out_transcript': (7, -100), 'latest': (3, -240), 'out_latest': (7, -240),
	'inference_stats': (0, -540), 'status': (1, -540), 'channels': (2, -470), 'out_channels': (7, -470),
	'component': (0, -100), 'model_pulses': (0, -260),
}
LAYOUT_AUDIO_SCORES = {
	'input': (0, 180), 'inference': (1, 180), 'keep_cooking': (1, 330), 'info': (1, 20),
	'scores': (2, 180), 'out_scores': (7, 180),
	'top_lines': (3, 380), 'top_text': (4, 380), 'transparent': (5, 520), 'preview': (6, 450), 'debug': (7, 450),
	'top': (2, -100), 'out_top': (7, -100),
	'events': (2, -300), 'inference_stats': (0, -540), 'status': (1, -540), 'channels': (3, -420),
	'out_channels': (7, -420), 'component': (0, -100), 'model_pulses': (0, -260),
}
LAYOUT_AUDIO_SPEECH = {
	'input': (0, 20), 'inference': (1, 180), 'keep_cooking': (1, 330),
	'voice_left': (2, 400), 'voice_right': (2, 290), 'stereo': (3, 345), 'audio_out': (4, 345),
	'out_audio': (7, 180),
	'speaking_text': (3, 600), 'caption_text': (4, 680), 'caption_canvas': (4, 540), 'captions': (5, 610),
	'transparent': (5, 760), 'preview': (6, 610), 'debug': (7, 610),
	'info': (2, -100), 'out_lines': (7, -100),
	'inference_stats': (0, -540), 'status': (1, -540), 'out_channels': (7, -540),
	'component': (0, -260), 'model_pulses': (0, -400),
}
DOCKED_AUDIO = ('caption_lines_callbacks', 'latest_callbacks', 'top_callbacks', 'events_callbacks',
	'top_lines_callbacks', 'speaking_text_callbacks')
# Non-video wrappers by input: audio (an audio CHOP) or text (an optional DAT).
INPUT_KINDS = {'audio': ('transcript', 'scores'), 'text': ('speech',)}

DEMO_ROW = 180


def root():
	return Path(td.project.folder)


def spec_names(input_kind=None):
	"""Every spec name, or only those for one input ('video', 'audio' or 'text')."""
	names = sorted(path.stem for path in (root() / 'wrappers').glob('*.json'))
	return names if input_kind is None else [name for name in names if load_spec(name).get('input', 'video') == input_kind]


def load_spec(name):
	spec = json.loads((root() / 'wrappers' / f'{name}.json').read_text(encoding='utf-8'))
	assert spec.get('name') == name, f'[WrapperBuilder] {name}.json names itself {spec.get("name")!r}'
	assert spec.get('kind') in KINDS, f'[WrapperBuilder] {name}: kind must be one of {KINDS}'
	assert spec['kind'] != 'classifier' or spec.get('detector'), f'[WrapperBuilder] {name}: classifier needs a detector'
	source = spec.get('input', 'video')
	expected = next((kind for kind, kinds in INPUT_KINDS.items() if spec['kind'] in kinds), 'video')
	assert source == expected, f'[WrapperBuilder] {name}: kind {spec["kind"]} needs input {expected!r}, not {source!r}'
	return spec


def version():
	path = root() / 'VERSION'
	return path.read_text(encoding='utf-8').strip() if path.exists() else '0.0.0'


def refresh_cooking(comp, frames=5):
	"""Re-apply a COMP's Cooking-off flag: on now, off again a few frames later. Nodes rebuilt or
	force-cooked inside a COMP with cooking off can go on cooking every frame regardless (seen in
	the wrapper lab: whole wrappers cooking, ~350 ms/s); a fresh off-switch stops them. Does
	nothing while cooking is on."""
	if comp is None or comp.allowCooking:
		return
	comp.allowCooking = True
	td.run('args[0].allowCooking = False', comp, delayFrames=frames)


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


TRACKER_DAT_PLUGIN = 'plugins/TrackerDAT.dll'


def _tracker_dat(parent_comp, name, table):
	"""A Tracker DAT: the registered custom OP, or (before TD has registered it, i.e. until the next
	restart after the first build) a CPlusPlus DAT loading plugins/TrackerDAT.dll."""
	try:
		node = parent_comp.create('TrackerDAT', name)
	except Exception:
		node = parent_comp.create('cplusplusDAT', name)
		node.par.plugin = (root() / TRACKER_DAT_PLUGIN).as_posix()
		node.cook(force=True)	# loads the plugin, which adds its parameters
	node.par.Table = table
	return node


def _build_tracker(comp, mode, source='detections', defaults=None):
	"""The tracker COMP: its Setup / Tracking pages (tracking_pars) and three Tracker DATs writing
	tracks, joints and bones, the Tracks node's parameters bound to the COMP's."""
	assert mode in tracking_pars.MODES, f'[WrapperBuilder] no tracker mode {mode}'
	tracker = comp.create('baseCOMP', 'tracker')
	tracking_pars.build_pages(tracker, mode)
	for name, value in (defaults or {}).items():
		parameter = getattr(tracker.par, name)
		parameter.default = value
		parameter.val = value
	info = tracker.create('inDAT', 'in_info')
	tracks = None
	for index, suffix in enumerate(('tracks', 'joints', 'bones'), 1):
		script = _tracker_dat(tracker, 'script_' + suffix, suffix)
		if suffix == 'tracks':
			for par in script.pars():
				if par.page is None or par.page.name not in ('Tracker', 'Tracking'):
					continue
				if par.name == 'Trackermode' or (par.page.name == 'Tracking' and hasattr(tracker.par, par.name)):
					par.expr = f'parent().par.{par.name}'
			if comp.op('inference_stats') is not None:
				script.par.Resultschop = '../inference_stats'
			if comp.op('null_passthrough') is not None:
				script.par.Frametop = '../null_passthrough'	# solvePnP's frame size
		_connect(script, info if suffix == 'tracks' else tracks)
		if suffix == 'tracks':
			tracks = script
		output = tracker.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		_connect(output, script)
	_connect(tracker, comp.op(source))
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
	scores = _tracker_dat(comp, 'scores', 'scores')
	for name in ('Classprefix', 'Classsmoothing'):	# created with the pages, after this
		scores.par[name].expr = f'parent().par.{name}'
	_connect(scores, comp.op('crop_boxes'), comp.op('detections'))
	return detector


def _build_posecrops(comp, spec):
	"""A landmarks wrapper with a nested person detector: the detector finds and tracks the people,
	`inference` (OnnxTOP Pose Crops / Holistic) takes their boxes from `crop_boxes`. With a
	"converter" (RTMW3D: rtmw_convert), `rtmw_info` reshapes the keypoints into the tracker's
	Holistic-like table; without one the tracker reads the Info DAT directly."""
	detector = build(load_spec(spec['detector']), comp, name='detector')
	detector.par.Active.expr = 'parent().par.Active.eval()'
	detector.par.Debugoverlay = False
	detector.par.Classfilter = 0	# people only (COCO class 0)
	_connect(detector, comp.op('input'))
	# script_tracks, not out_tracks: the crops must not depend on the pose table.
	comp.create('selectDAT', 'crop_boxes').par.dat = 'detector/tracker/script_tracks'
	if spec.get('preset', {}).get('Mode') == 'holistic':
		# Holistic tracks each person from their own landmarks; the detector only finds new people. It
		# runs while fewer are tracked than Max Detections, without waiting for its result (a person
		# is picked up a frame later). Set at frame start, not by expression: an expression on the
		# tracked count would be a cook loop (the inference reads the detector's boxes).
		detector.par.Active.expr = ''
		detector.par.Waitforresult = False
		gate = comp.create('executeDAT', 'detector_gate')
		gate.par.framestart = True
		gate.text = ("def onFrameStart(frame):\n"
			"\tw = parent()\n"
			"\tif not w.allowCooking:\n"
			"\t\treturn\n"
			"\tstats = w.op('inference_stats')\n"
			"\ttracked = stats['num_detections'].eval() if stats['num_detections'] is not None else 0\n"
			"\tneed = bool(w.par.Active.eval()) and tracked < w.par.Maxdetections.eval()\n"
			"\tdetector = w.op('detector')\n"
			"\tif bool(detector.par.Active.eval()) != need:\n"
			"\t\tdetector.par.Active = need\n")
	converter = spec.get('converter')
	if converter:
		assert converter == 'rtmw_convert', f'[WrapperBuilder] no converter {converter}'
		convert = _tracker_dat(comp, 'rtmw_info', 'convert_rtmw')
		convert.par.Frametop = 'input'
		_connect(convert, comp.op('detections'))
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
		result = _build_flow(comp) if spec.get('flow') else _remapped(comp, comp.op('inference'), spec.get('remap'), 'remap')
		_connect(overlay, result)
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
		debug_page = next(page for page in comp.customPages if page.name == 'Debug')
		for name in ('Jointsize', 'Jointoutline'):
			_expose(debug_page, keypoints, name)
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
		if spec.get('instances'):
			_build_instances(comp)
	else:
		_connect(preview, transparent, overlay)


# Optical flow (a spec's "flow": true; OnnxTOP Raw mode on a frame-pair model, Letterbox off): the raw
# output is the flow in pixels of the model's input, x right, y down. TOP 1 gets it in the Optical Flow
# TOP's format, so the two are interchangeable: R and G in screen widths per second (G too is in
# widths: a 16:9 frame's height is 0.5625 widths), x right, y up, times Gain X / Y.
FLOW_SHADER = '''// Flow in pixels of the model's input (x right, y down) -> the Optical Flow TOP's format: screen
// widths per second, x right, y up.
uniform vec2 uModelSize;	// the model's input, pixels
uniform vec2 uFrame;		// x: the frame's height / width, y: seconds between the pair's two frames
uniform vec2 uGain;
out vec4 fragColor;
void main()
{
	vec2 f = texture(sTD2DInputs[0], vUV.st).rg;
	float perSecond = 1.0 / max(uFrame.y, 1e-4);
	float x = f.x / uModelSize.x * perSecond * uGain.x;
	float y = -f.y / uModelSize.y * uFrame.x * perSecond * uGain.y;
	fragColor = TDOutputSwizzle(vec4(x, y, 0.0, 1.0));
}
'''
FLOW_SMOOTH_SHADER = '''// Flow Smoothing: this flow blended with the last smoothed one (input 2, a Feedback TOP).
uniform float uSmoothing;	// 0 = raw
out vec4 fragColor;
void main()
{
	vec4 now = texture(sTD2DInputs[0], vUV.st);
	vec4 before = texture(sTD2DInputs[1], vUV.st);
	fragColor = TDOutputSwizzle(mix(now, before, uSmoothing));
}
'''
FLOW_GATE_SHADER = '''// Noise Floor: speeds below uFloor.x become zero; above it the flow fades back in over uFloor.y.
uniform vec2 uFloor;	// x: floor, y: ease (screen widths per second)
out vec4 fragColor;
void main()
{
	vec4 f = texture(sTD2DInputs[0], vUV.st);
	float speed = length(f.rg);
	float keep = uFloor.y > 0.0 ? smoothstep(uFloor.x, uFloor.x + uFloor.y, speed) : step(uFloor.x, speed);
	fragColor = TDOutputSwizzle(vec4(f.rg * keep, f.ba));
}
'''
FLOW_COLOR_SHADER = '''// Flow as color, like viewing the Optical Flow TOP: R = rightward, G = upward, scaled so uFullSpeed is
// full brightness. Left and down (negative) and still show black.
uniform float uFullSpeed;	// screen widths per second shown at full brightness
out vec4 fragColor;
void main()
{
	vec2 f = texture(sTD2DInputs[0], vUV.st).rg;
	vec2 c = clamp(f / max(uFullSpeed, 1e-4), 0.0, 1.0);
	fragColor = TDOutputSwizzle(vec4(c, 0.0, 1.0));
}
'''


def _glsl(comp, name, shader, source):
	"""A GLSL TOP `name` running `shader` (a pixel shader) on `source`, 32-bit float out."""
	glsl = comp.create('glslTOP', name)
	pixel = comp.op(name + '_pixel') or comp.create('textDAT', name + '_pixel')
	pixel.text = shader
	glsl.par.pixeldat = pixel.name
	glsl.par.format = 'rgba32float'
	_connect(glsl, source)
	return glsl


def _build_flow(comp):
	"""`flow`: the flow in the Optical Flow TOP's format (Gain X / Y on the Model page), using the
	plugin's pair_interval (the measured time between the pair's frames; a project frame until known).
	`flow_smooth`: Flow Smoothing over it (a Feedback TOP). `flow_gate`, the result (TOP 1): the Noise
	Floor (after the smoothing, which steadies the speeds it compares). `flow_color` (TOP 3):
	R = rightward, G = upward, like viewing the Optical Flow TOP (Color Full Speed on the Debug page)."""
	flow = _glsl(comp, 'flow', FLOW_SHADER, comp.op('inference'))
	stat = "op('inference_stats')['{}'].eval() if op('inference_stats')['{}'] is not None else 0"
	flow.par.vec0name = 'uModelSize'
	flow.par.vec0valuex.expr = f"max(1, {stat.format('model_w', 'model_w')})"
	flow.par.vec0valuey.expr = f"max(1, {stat.format('model_h', 'model_h')})"
	flow.par.vec1name = 'uFrame'
	flow.par.vec1valuex.expr = "op('input').height / max(1, op('input').width)"
	flow.par.vec1valuey.expr = f"({stat.format('pair_interval', 'pair_interval')}) or 1.0 / me.time.rate"
	flow.par.vec2name = 'uGain'
	flow.par.vec2valuex.expr = 'parent().par.Gainx'
	flow.par.vec2valuey.expr = 'parent().par.Gainy'
	model = next(page for page in comp.customPages if page.name == 'Model')
	_page_par(model, 'Float', 'Gainx', 'Gain X', 1.0, 'Multiplies the horizontal flow (R), like the Optical Flow TOP.', 0.0, 4.0)
	_page_par(model, 'Float', 'Gainy', 'Gain Y', 1.0, 'Multiplies the vertical flow (G), like the Optical Flow TOP.', 0.0, 4.0)
	smoothing = _page_par(model, 'Float', 'Flowsmoothing', 'Flow Smoothing', 0.3,
		'Blends each new flow with the last result (0 = raw, higher = steadier but slower to follow).', 0.0, 0.95)
	smoothing.clampMin, smoothing.clampMax, smoothing.min, smoothing.max = True, True, 0.0, 0.99
	feedback = comp.create('feedbackTOP', 'flow_feedback')
	_connect(feedback, flow)
	feedback.par.top = 'flow_smooth'
	smooth = _glsl(comp, 'flow_smooth', FLOW_SMOOTH_SHADER, flow)
	smooth.inputConnectors[1].connect(feedback)
	smooth.par.vec0name = 'uSmoothing'
	smooth.par.vec0valuex.expr = 'parent().par.Flowsmoothing'
	floor = _page_par(model, 'Float', 'Noisefloor', 'Noise Floor', 0.02,
		'Speeds below this (screen widths per second) become zero: the model\'s baseline noise on still areas.', 0.0, 0.2)
	floor.clampMin, floor.min = True, 0.0
	ease = _page_par(model, 'Float', 'Floorease', 'Floor Ease', 0.03,
		'Above the Noise Floor the flow fades back in over this much more speed (0 = a hard cut).', 0.0, 0.2)
	ease.clampMin, ease.min = True, 0.0
	gate = _glsl(comp, 'flow_gate', FLOW_GATE_SHADER, smooth)
	gate.par.vec0name = 'uFloor'
	gate.par.vec0valuex.expr = 'parent().par.Noisefloor'
	gate.par.vec0valuey.expr = 'parent().par.Floorease'
	flow_color = _glsl(comp, 'flow_color', FLOW_COLOR_SHADER, gate)
	flow_color.par.format = 'rgba8fixed'
	flow_color.par.vec0name = 'uFullSpeed'
	flow_color.par.vec0valuex.expr = 'parent().par.Colorfullspeed'
	debug = next(page for page in comp.customPages if page.name == 'Debug')
	_page_par(debug, 'Float', 'Colorfullspeed', 'Color Full Speed', 1.0,
		'TOP 3 (the flow as color): the speed, in screen widths per second, shown at full brightness.', 0.05, 5.0)
	output = comp.create('outTOP', 'out_color')
	output.par.connectorder = 3
	_connect(output, flow_color)
	return gate


def _remapped(comp, source, remap, name):
	"""`source`, or a Math TOP `name` on it applying a spec's "remap": [gain, offset] (value * gain +
	offset: DAViD's -1..1 normals -> 0..1 with [0.5, 0.5], depth flipped to near = bright with [-1, 1])."""
	if not remap:
		return source
	math = comp.create('mathTOP', name)
	math.par.gain, math.par.postoff = remap
	_connect(math, source)
	return math


def _build_extra_outputs(comp, spec):
	"""A spec's "extra_outputs": [{"name", "buffer", "remap"?}, ...]: the inference node's other color
	buffers (OnnxTOP Raw mode, All Outputs) as TOP outputs 3, 4, ... in a band under the CHOPs."""
	for index, extra in enumerate(spec.get('extra_outputs', ())):
		name, y = extra['name'], -700 - index * 140
		select = comp.create('renderselectTOP', name + '_select')
		select.par.top = 'inference'
		select.par.bufferindex = extra['buffer']
		result = _remapped(comp, select, extra.get('remap'), name + '_remap')
		output = comp.create('outTOP', 'out_' + name)
		output.par.connectorder = 3 + index
		_connect(output, result)
		for node, column in ((select, 5), (result, 6), (output, 7)):
			node.nodeX, node.nodeY = column * COLUMN, y


INSTANCES_GLSL = '''// Instances: each pixel's owner (the matte's G = detection index + 1) in its track's color.
// Input 0: the instance-encoded matte; input 1: one pixel per detection index + 1 (instance_colors).
out vec4 fragColor;
void main()
{
	vec4 matte = texture(sTD2DInputs[0], vUV.st);
	int owner = int(matte.g + 0.5);
	vec4 color = vec4(0.0);
	if (owner > 0 && owner < textureSize(sTD2DInputs[1], 0).x)
		color = texelFetch(sTD2DInputs[1], ivec2(owner, 0), 0);
	float alpha = matte.a * color.a;
	fragColor = TDOutputSwizzle(vec4(color.rgb * alpha, alpha));
}
'''


def _build_instances(comp):
	"""TOP 4 `instances`: each tracked object's pixels in its track color (Matte Encoding =
	Instances). The tracks table's det_index maps this frame's detections to tracks; that lookup
	is a one-row texture, so Python only handles one value per object, never pixels."""
	colors = comp.create('scriptCHOP', 'instance_colors')
	comp.op('instance_colors_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_instance_colors(scriptOp, parent())\n")
	_per_frame(colors)
	lookup = comp.create('choptoTOP', 'instance_lookup')
	lookup.par.chop = 'instance_colors'
	lookup.par.dataformat = 'rgba'
	lookup.par.format = 'rgba16float'
	matte = comp.create('selectTOP', 'instance_matte')
	matte.par.top = 'inference'
	instances = comp.create('glslTOP', 'instances')
	for name in ('instances_compute', 'instances_info'):
		if comp.op(name) is not None:
			comp.op(name).destroy()
	comp.op('instances_pixel').text = INSTANCES_GLSL
	instances.par.inputfiltertype = 'nearest'	# ids must not be blended between neighbours
	instances.par.format = 'rgba8fixed'
	_connect(instances, matte, lookup)
	output = comp.create('outTOP', 'out_instances')
	output.par.connectorder = 4
	_connect(output, instances)


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
	if tracker is not None and hasattr(comp.par, 'Holisticface') and hasattr(tracker.par, 'Facemesh'):
		# Holistic GPU: one switch. Without face inference there's no mesh to track.
		tracker.par.Facemesh.expr = 'parent().par.Holisticface.eval()'
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
	if spec.get('input') in INPUT_KINDS:
		positions = {'scores': LAYOUT_AUDIO_SCORES, 'speech': LAYOUT_AUDIO_SPEECH}.get(spec['kind'], LAYOUT_AUDIO)
		docked = DOCKED_AUDIO
	else:
		positions, docked = dict(LAYOUT, **LAYOUT_KIND.get(spec['kind'], {})), DOCKED
		if spec.get('result') == 'inference':
			positions.update(LAYOUT_RESULT_INFERENCE)
		if spec.get('remap'):
			positions.update({'remap': (2, 320), 'overlay': (3, 320)})
		if spec.get('flow'):
			positions.update({'flow': (2, 320), 'flow_feedback': (2, 460), 'flow_smooth': (3, 320), 'flow_gate': (4, 320),
				'flow_color': (4, 460), 'overlay': (5, 200), 'out_color': (7, -700)})
			docked = list(docked) + ['flow_pixel', 'flow_smooth_pixel', 'flow_gate_pixel', 'flow_color_pixel']
	for path, (column, y) in positions.items():
		node = comp.op(path)
		if node is not None:
			node.nodeX, node.nodeY = column * COLUMN, y
	for path in docked:
		callbacks = comp.op(path)
		host = comp.op(path.removesuffix('_callbacks').removesuffix('_pixel'))	# GLSL TOPs: <name>_pixel
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
	if spec.get('input') in INPUT_KINDS:
		_build_audio(comp, spec)
		_layout(comp, spec)
		_restore_wires(comp, inputs, outputs)
		return comp
	posecrops = spec['kind'] == 'landmarks' and bool(spec.get('detector'))
	converter = spec.get('converter') if posecrops else None
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
	detector = (_build_classifier(comp, spec) if kind == 'classifier'
		else _build_posecrops(comp, spec) if posecrops else None)
	tracker = _build_tracker(comp, spec['tracker'], 'rtmw_info' if converter else 'detections',
		spec.get('tracker_defaults')) if spec.get('tracker') else None
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
	_build_extra_outputs(comp, spec)
	tables = ('tracks', 'joints', 'bones') if comp.op('joints') is not None else ('tracks',)
	for index, suffix in enumerate(tables, 1):
		output = comp.create('outDAT', 'out_' + suffix)
		output.par.connectorder = index
		_connect(output, comp.op(suffix))
	_connect(comp.create('outCHOP', 'out_channels'), _build_channels(comp))
	_build_performance(comp, spec)
	for node in comp.findChildren():
		node.viewer = False
	_layout(comp, spec)
	_add_icon(comp, comp.name, './debug')
	_restore_wires(comp, inputs, outputs)
	return comp


# The t{id}_{column} track values on a Tracker DAT's Info CHOP (its stats and TD's standard info
# channels don't match: brackets are a numeric range in TD patterns).
TRACK_CHANNELS = 't[0-9]*'


def track_channels(comp, track_path, values_name='track_values', name='track_channels'):
	"""The t{id}_{column} channels of a Tracker DAT, current every cook: an Info CHOP scoped to them,
	copied by a Script CHOP that first reads the Tracker DAT. An Info CHOP alone only reports a node's
	latest numbers, it doesn't cook it (nothing else pulling the tables, the channels stood still);
	reading the DAT cooks it and makes the Script CHOP depend on it. One C++ copy per cook."""
	values = comp.create('infoCHOP', values_name)
	values.par.op = track_path
	values.par.iscope = TRACK_CHANNELS
	pull = comp.create('scriptCHOP', name)
	comp.op(name + '_callbacks').text = (
		"def onCook(scriptOp):\n"
		f"\tif parent().op('{track_path}').numRows >= 0:\n"
		f"\t\tvalues = parent().op('{values_name}')\n"
		"\t\tvalues.cook(force=True)\n"
		"\t\tscriptOp.copy(values)\n")
	return pull


def _track_node(comp):
	"""The node writing the wrapper's final tracks table: the tracker's Tracks node, or a classifier's
	scores. None for image wrappers."""
	if comp.op('tracker') is not None:
		return comp.op('tracker/script_tracks')
	return comp.op('scores')


def _build_channels(comp):
	"""`channels`: t{id}_{column} for every numeric tracks cell, then the plugin's Info CHOP stats. The
	Tracker DAT publishes the track values on its Info CHOP (built only when asked; see
	track_channels); a Merge adds the plugin's stats."""
	track = _track_node(comp)
	if track is None:
		# Image models: no tracks, so the channels are the plugin's stats.
		channels = comp.create('nullCHOP', 'channels')
		_connect(channels, comp.op('inference_stats'))
		return channels
	channels = comp.create('mergeCHOP', 'channels')
	_connect(channels, track_channels(comp, track.path[len(comp.path) + 1:]), comp.op('inference_stats'))
	return channels


# TD caches an expression's value until something it depends on changes; cook times aren't tracked
# dependencies, so these also read absTime.frame to be re-evaluated every frame they're shown.
FRESH = ' + absTime.frame * 0'


def _cooked_ms(name):
	"""A node's main-thread time last frame: a COMP's children, else the node's own cook (0 when it
	didn't cook, rather than its last, stale cook time)."""
	node = f"me.op('{name}')"
	return f"({node}.childrenCookTime if {node}.isCOMP else ({node}.cookTime if {node}.cookAbsFrame >= absTime.frame - 1 else 0.0))"


def _perf(page, name, label, expression, help_text):
	parameter = page.appendFloat(name, label=label)[0]
	parameter.expr = expression
	parameter.readOnly = True
	parameter.help = help_text
	return parameter


def _build_performance(comp, spec):
	"""The Performance page: read-only timings, evaluated only while something shows them (the
	parameter dialog on this page), so they cost nothing otherwise. Plugin and tracker numbers are
	the plugins' own running averages; Info DAT and overlay are TouchDesigner's cook times."""
	page = comp.appendCustomPage('Performance')
	onnx = spec['plugin'] == 'OnnxTOP'
	stat = "me.op('inference_stats')['{}'].eval() if me.op('inference_stats')['{}'] is not None else 0.0"
	_perf(page, 'Perfinfer', 'Inference ms', stat.format('infer_ms_avg', 'infer_ms_avg'),
		'The model itself: GPU time (OnnxTOP, CUDA events) or the MediaPipe worker thread\'s time. Off TD\'s main '
		'thread unless Wait for Result is on.' if onnx else 'The MediaPipe worker thread\'s time per result (off TD\'s main thread).')
	_perf(page, 'Perfplugin', 'Plugin Cook ms', stat.format('execute_ms_avg', 'execute_ms_avg'),
		"The plugin's cook on TD's main thread: uploading the frame, launching the model, reading back results.")
	mode = spec.get('preset', {}).get('Mode')
	if onnx and mode != 'holistic':	# Holistic runs its models on a worker thread
		_perf(page, 'Perflaunch', 'Model Launch ms', stat.format('run_ms_avg', 'run_ms_avg'),
			'Part of Plugin Cook: the CPU time ONNX Runtime takes to launch a run.')
	if onnx and mode not in ('holistic', 'crops', 'posecrops'):
		_perf(page, 'Perfdecode', 'Decode ms', stat.format('decode_ms_avg', 'decode_ms_avg'),
			'Part of Plugin Cook: reading back and decoding the outputs (boxes, keypoints, images), including '
			'waiting for the GPU when the result is needed this frame.')
	_perf(page, 'Perfinfodat', 'Info DAT ms', _cooked_ms('detections') + FRESH,
		"Writing the detections into the Info DAT (TD's cook time, last frame).")
	track = _track_node(comp)
	if track is not None:
		# A second Info CHOP, unscoped: track_values only carries the track channels.
		stats = comp.create('infoCHOP', 'tracker_stats')
		stats.par.op = track.path[len(comp.path) + 1:]
		stats.par.iscope = 'tracker_ms_avg'
		expression = "me.op('tracker_stats')['tracker_ms_avg'].eval() if me.op('tracker_stats')['tracker_ms_avg'] is not None else 0.0"
		help_text = 'The Tracker DAT: tracking and writing the tracks, joints and bones tables (its own timing).'
		_perf(page, 'Perftracker', 'Tracker ms', expression, help_text)
	if comp.op('detector') is not None:
		_perf(page, 'Perfdetector', 'Detector ms', "me.op('detector').par.Perftotal.eval()",
			"The nested person / face detector wrapper's main-thread total (its own Performance page).")
	overlays = [n for n in ('debug_boxes', 'debug_keypoints', 'overlay', 'instances') if comp.op(n) is not None]
	if overlays:
		_perf(page, 'Perfoverlay', 'Debug Overlay ms', ' + '.join(_cooked_ms(n) for n in overlays) + FRESH,
			"Drawing the debug overlay (TD's cook times, last frame); 0 with Debug Overlay off.")
	_perf(page, 'Perftotal', 'Main Thread Total ms', 'me.childrenCookTime' + FRESH,
		"Everything inside this wrapper on TD's main thread last frame (TD's measure; inference on the GPU or a "
		"worker thread isn't in it).")
	delay = page.appendInt('Perfdelay', label='Result Delay (frames)')[0]
	delay.expr = stat.format('result_delay', 'result_delay')
	delay.readOnly = True
	delay.help = 'Input frames between the shown result and the newest frame (0 = this frame\'s result).'
	if spec['plugin'] != 'OnnxTOP':
		dropped = page.appendInt('Perfdropped', label='Dropped Frames')[0]
		dropped.expr = stat.format('dropped_frames', 'dropped_frames')
		dropped.readOnly = True
		dropped.help = 'Frames the MediaPipe worker skipped because it was still busy.'
	comp.sortCustomPages(*[p.name for p in comp.customPages if p.name != 'About'], 'About')


ICON_SIZE = (320, 260)	# the 16:13 card at node size: sharp in the network, 4x smaller in the tox than 2x


def _add_icon(comp, name, fallback_viewer):
	"""Show docs/images/icons/<name>.png (scripts/build_icons.py) as the COMP's node viewer: loaded,
	resized and locked, so the pixels are saved in the tox and nothing renders at runtime. Without
	the PNG (or for nested wrappers like HSEmotion's detector) the viewer stays `fallback_viewer`."""
	png = root() / 'docs' / 'images' / 'icons' / f'{name}.png'
	if not png.exists():
		comp.par.opviewer = fallback_viewer
		if name != 'detector':
			print(f'[WrapperBuilder] no icon for {name}: run scripts/build_icons.py')
		return None
	icon = comp.create('moviefileinTOP', 'icon')
	icon.par.file = png.as_posix()
	icon.par.outputresolution = 'custom'
	icon.par.resolutionw, icon.par.resolutionh = ICON_SIZE
	icon.cook(force=True)
	icon.lock = True
	icon.par.file = ''	# the locked pixels stay; no path back to this checkout
	icon.viewer = False
	icon.nodeX, icon.nodeY = 0, 560
	comp.par.opviewer = './icon'	# ./: a bare name resolves next to the COMP, not inside it
	comp.viewer = True	# a dropped-in tox shows its card right away
	return icon


def _page_par(page, kind, name, label, value, help_text, low=None, high=None):
	parameter = getattr(page, 'append' + kind)(name, label=label)[0]
	parameter.val = parameter.default = value
	if low is not None:
		parameter.normMin, parameter.normMax = low, high
	parameter.help = help_text
	return parameter


def _build_audio(comp, spec):
	"""Audio wrapper: input (an audio CHOP, or for text input an optional DAT) -> the MediaPipe audio
	CHOP (`inference`, cooked every frame), `info` (its Info DAT), the Model page, then the kind's
	own outputs (_build_transcript, _build_scores, _build_speech) and the About page."""
	_add_code(comp)
	text_input = spec['input'] == 'text'
	comp.create('inDAT' if text_input else 'inCHOP', 'input')
	plugin = comp.create(spec['plugin'], 'inference')
	for parameter, value in spec.get('preset', {}).items():
		_set(plugin, parameter, value)
	if not text_input:
		_connect(plugin, comp.op('input'))
	# Nothing may need this wrapper's outputs on a given frame, but audio keeps arriving: cook the
	# native node every frame so nothing is missed.
	# Text input is polled here too: a DAT Execute on the In DAT only fires if something cooks it.
	keep = comp.create('executeDAT', 'keep_cooking')
	keep.par.framestart = True
	keep.text = ("def onFrameStart(frame):\n\tif parent().par.Active.eval():\n"
		+ ("\t\tparent().op('component').module.speak_input(parent())\n" if text_input else '')
		+ "\t\tparent().op('inference').cook()\n")
	comp.create('infoDAT', 'info').par.op = 'inference'
	comp.create('infoCHOP', 'inference_stats').par.op = 'inference'
	# The plugin's own info channels only (the Info CHOP adds cook stats and a second sample_rate).
	status = comp.create('selectCHOP', 'status')
	status.par.chop = 'inference_stats'
	status.par.channames = 'loaded loading infer_ms results' + (' queued' if text_input else '')

	model_page = comp.appendCustomPage('Model')
	pulses = []
	for name in (*ALWAYS_EXPOSED, *spec.get('model', ()), *ALWAYS_PULSES):
		pulses += _expose(model_page, plugin, name)[1]
	if pulses:
		forward = comp.create('parameterexecuteDAT', 'model_pulses')
		forward.par.op = '..'
		forward.par.pars = ' '.join(pulses)
		forward.par.valuechange = False
		forward.par.onpulse = True
		forward.text = "def onPulse(par):\n\tparent().op('inference').par[par.name].pulse()\n"

	{'transcript': _build_transcript, 'scores': _build_scores, 'speech': _build_speech}[spec['kind']](comp, spec)
	release = comp.appendCustomPage('About').appendStr('Version', label='Version')[0]
	release.val = release.default = version()
	release.readOnly = True
	for node in comp.findChildren():
		node.viewer = False
	_add_icon(comp, comp.name, './debug')


def _transparent_switch(comp, shown, width_expr, height_expr):
	"""`preview` -> Out TOP `debug`: `shown` when Debug Overlay is on, else transparent."""
	transparent = comp.create('constantTOP', 'transparent')
	transparent.par.outputresolution = 'custom'
	transparent.par.resolutionw.expr = width_expr
	transparent.par.resolutionh.expr = height_expr
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		transparent.par[channel] = 0
	preview = comp.create('switchTOP', 'preview')
	preview.par.index.expr = 'int(parent().par.Debugoverlay.eval())'
	_connect(preview, transparent, shown)
	_connect(comp.create('outTOP', 'debug'), preview)


def _out_dats(comp, names):
	for index, name in enumerate(names, 1):
		output = comp.create('outDAT', 'out_' + name)
		output.par.connectorder = index
		_connect(output, comp.op(name))


def _per_frame(script):
	"""A custom parameter that changes every frame, so the script re-cooks whenever it's pulled."""
	frame = script.appendCustomPage('Cook').appendInt('Frame', label='Frame')[0]
	frame.expr = 'absTime.frame'
	frame.readOnly = True


def _build_transcript(comp, spec):
	"""Speech to text: transcript (the Info DAT) and latest (newest sentence + interim) DATs, the
	native channels plus status as a CHOP, and a closed-caption TOP: caption_lines (text) ->
	caption_text over a transparent canvas."""
	comp.op('info').name = 'transcript'
	transcript = comp.op('transcript')
	captions_page = comp.appendCustomPage('Captions')
	_page_par(captions_page, 'Int', 'Maxchars', 'Max Characters', 90,
		'Show at most this many characters (the newest, cut at a word boundary).', 20, 200)
	_page_par(captions_page, 'Float', 'Newcaption', 'New Caption After (Seconds)', 1.0,
		'Speech after this many seconds without new text starts a fresh caption (0 = keep adding).', 0, 5)
	_page_par(captions_page, 'Float', 'Hold', 'Hold (Seconds)', 4.0,
		'Clear the caption after this many seconds without new text.', 0.5, 15)
	_caption_size_pars(captions_page)
	_page_par(comp.appendCustomPage('Debug'), 'Toggle', 'Debugoverlay', 'Show Captions', True,
		'Off = TOP 1 stays transparent and the caption chain does not cook.')

	latest = comp.create('scriptDAT', 'latest')
	comp.op('latest_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_latest(scriptOp, parent())\n")
	_connect(latest, transcript)
	lines = comp.create('scriptDAT', 'caption_lines')
	comp.op('caption_lines_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_captions(scriptOp, parent())\n")
	_connect(lines, transcript)
	_per_frame(lines)	# the Hold timeout needs a cook every frame while the captions are shown
	_caption_image(comp, 'caption_lines')

	_out_dats(comp, ('transcript', 'latest'))
	channels = comp.create('mergeCHOP', 'channels')
	_connect(channels, comp.op('inference'), comp.op('status'))
	_connect(comp.create('outCHOP', 'out_channels'), channels)


def _caption_size_pars(page):
	_page_par(page, 'Float', 'Fontsize', 'Font Size', 56, 'Caption text size in pixels.', 16, 120)
	resolution = page.appendInt('Captionresolution', label='Resolution', size=2)
	for parameter, value in zip(resolution, (1920, 1080)):
		parameter.val = parameter.default = value
	resolution[0].help = 'Size of the caption image (transparent; the caption sits along the bottom).'


def _caption_image(comp, dat_name):
	"""TOP 1: the text in cell [0, 0] of `dat_name` in a dark strip along the bottom of a
	transparent Captionresolution image (caption_text over caption_canvas), hidden when empty."""
	shown = f"op('{dat_name}')[0, 0] is not None and op('{dat_name}')[0, 0].val"
	text = comp.create('textTOP', 'caption_text')
	text.par.outputresolution = 'custom'
	text.par.resolutionw.expr = 'int(parent().par.Captionresolution1 * 0.78)'
	text.par.resolutionh.expr = 'int(parent().par.Fontsize * 3.4)'
	text.par.dat = dat_name
	text.par.text = ''	# else its default text is drawn as well
	text.par.wordwrap = True
	text.par.alignx, text.par.aligny = 'center', 'center'
	text.par.fontsizex.expr = 'parent().par.Fontsize'
	text.par.fontcolorr = text.par.fontcolorg = text.par.fontcolorb = 1.0
	text.par.bgcolorr = text.par.bgcolorg = text.par.bgcolorb = 0.0
	text.par.bgalpha.expr = f'0.65 if {shown} else 0'
	text.par.fontalpha.expr = f'1 if {shown} else 0'
	canvas = comp.create('constantTOP', 'caption_canvas')
	canvas.par.outputresolution = 'custom'
	canvas.par.resolutionw.expr = 'parent().par.Captionresolution1'
	canvas.par.resolutionh.expr = 'parent().par.Captionresolution2'
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		canvas.par[channel] = 0
	captions = comp.create('overTOP', 'captions')
	# Over TOP: input 1 goes over input 2, and the output takes input 2's resolution (the canvas).
	_connect(captions, text, canvas)
	captions.par.prefit = 'nativeres'
	captions.par.justifyv, captions.par.justifyh = 'bottom', 'center'
	captions.par.tunit = 'pixels'
	captions.par.ty.expr = 'int(parent().par.Captionresolution2 * 0.055)'
	_transparent_switch(comp, captions, 'parent().par.Captionresolution1', 'parent().par.Captionresolution2')


def _build_speech(comp, spec):
	"""Text to speech: Text + Speak on the Model page, or a DAT wired in (its newest row is spoken
	whenever it changes, while Speak Input is on). CHOP 1 is the voice as 24 kHz audio plus
	`speaking`, also played on the default audio device while Play Audio is on; CHOP 2 the status;
	DAT `lines` (speaking + queued); TOP 1 a caption of the line being spoken."""
	page = comp.appendCustomPage('Output')
	_page_par(page, 'Toggle', 'Speakinput', 'Speak Input DAT', True,
		"Speak the newest row of the DAT wired into the input whenever it changes (its 'text' column, "
		'else its last cell).')
	_page_par(page, 'Toggle', 'Playaudio', 'Play Audio', True,
		'Play the voice on the default audio device. Off = only CHOP 1 (route it yourself).')
	_page_par(page, 'Float', 'Volume', 'Volume', 1.0, 'Playback volume.', 0, 1)
	_caption_size_pars(comp.appendCustomPage('Captions'))
	_page_par(comp.appendCustomPage('Debug'), 'Toggle', 'Debugoverlay', 'Show Captions', True,
		'Off = TOP 1 stays transparent and the caption chain does not cook.')

	plugin = comp.op('inference')	# the input DAT is read every frame by keep_cooking (speak_input)
	# Audio Device Out plays a mono CHOP on one side only: send the voice to both.
	sides = []
	for side in ('left', 'right'):
		voice = comp.create('selectCHOP', 'voice_' + side)
		voice.par.channames = 'audio'
		voice.par.renamefrom, voice.par.renameto = 'audio', side
		_connect(voice, plugin)
		sides.append(voice)
	stereo = comp.create('mergeCHOP', 'stereo')
	_connect(stereo, *sides)
	device = comp.create('audiodeviceoutCHOP', 'audio_out')
	_connect(device, stereo)
	device.par.active.expr = 'parent().par.Active.eval() and parent().par.Playaudio.eval()'
	device.par.volume.expr = 'parent().par.Volume'

	speaking = comp.create('scriptDAT', 'speaking_text')
	comp.op('speaking_text_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_speaking(scriptOp, parent())\n")
	_per_frame(speaking)
	_caption_image(comp, 'speaking_text')

	_out_dats(comp, ('info',))
	comp.op('out_info').name = 'out_lines'
	audio_out = comp.create('outCHOP', 'out_audio')
	audio_out.par.connectorder = 1
	_connect(audio_out, plugin)
	status_out = comp.create('outCHOP', 'out_channels')
	status_out.par.connectorder = 2
	_connect(status_out, comp.op('status'))


def _build_scores(comp, spec):
	"""Sound classification: CHOP 1 every class score, smoothed (Lag CHOP `scores`); CHOP 2
	`events`: per watched class its score, _active and _onset, plus status; DAT `top` (the top
	results of the smoothed scores) and a top-results TOP."""
	classes = spec.get('classes', {})
	page = comp.appendCustomPage('Classes')
	_page_par(page, 'Float', 'Smoothing', 'Score Smoothing (Seconds)', classes.get('smoothing', 0.3),
		'Lag on every class score: higher = steadier, slower to react.', 0, 2)
	_page_par(page, 'Int', 'Topcount', 'Top Results', 5, 'Rows in the top DAT and the debug display.', 1, 20)
	_page_par(page, 'Str', 'Watch', 'Watch Classes', classes.get('watch', 'Speech Music Clapping'),
		'Space-separated class channel names (as in CHOP 1, e.g. Speech Dog Finger_snapping). Each gets '
		'<name>, <name>_active and <name>_onset channels in CHOP 2.')
	_page_par(page, 'Float', 'Threshold', 'Active Threshold', classes.get('threshold', 0.3),
		'A watched class is active while its smoothed score is above this.', 0, 1)
	_page_par(comp.appendCustomPage('Debug'), 'Toggle', 'Debugoverlay', 'Show Top Results', True,
		'Off = TOP 1 stays transparent and the display does not cook.')

	scores = comp.create('lagCHOP', 'scores')
	_connect(scores, comp.op('inference'))
	scores.par.lag1.expr = scores.par.lag2.expr = 'parent().par.Smoothing'
	top = comp.create('scriptDAT', 'top')
	comp.op('top_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_top(scriptOp, parent())\n")
	_per_frame(top)
	events = comp.create('scriptCHOP', 'events')
	comp.op('events_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_events(scriptOp, parent())\n")
	_per_frame(events)

	# Reads `top` by name (no wire across the CHOP lane); per-frame like `top`.
	lines = comp.create('scriptDAT', 'top_lines')
	comp.op('top_lines_callbacks').text = (
		"def onCook(scriptOp):\n\tparent().op('component').module.write_top_lines(scriptOp, parent())\n")
	_per_frame(lines)
	text = comp.create('textTOP', 'top_text')
	text.par.outputresolution = 'custom'
	text.par.resolutionw = 640
	text.par.resolutionh.expr = '44 + 29 * parent().par.Topcount'	# one ~29 px line per result
	text.par.dat = 'top_lines'
	text.par.text = ''
	text.par.font = 'Consolas'
	text.par.alignx, text.par.aligny = 'left', 'top'
	text.par.position1, text.par.position2 = 24, -24
	text.par.fontsizex = 19	# 38 monospace characters per line fit the 640 px width
	text.par.fontcolorr = text.par.fontcolorg = text.par.fontcolorb = 1.0
	text.par.bgcolorr = text.par.bgcolorg = text.par.bgcolorb = 0.0
	text.par.bgalpha = 0.65
	_transparent_switch(comp, text, "op('top_text').width", "op('top_text').height")

	_out_dats(comp, ('top',))
	_connect(comp.create('outCHOP', 'out_scores'), scores)
	channels = comp.create('mergeCHOP', 'channels')
	_connect(channels, events, comp.op('status'))
	output = comp.create('outCHOP', 'out_channels')
	_connect(output, channels)
	comp.op('out_scores').par.connectorder = 1
	output.par.connectorder = 2


def build_demo(parent_comp, names, name='VisionDemo'):
	"""(Re)build the workbench: one wrapper per spec, each enabled from the Models page. A disabled
	wrapper doesn't cook at all (allowCooking, set from outside it: a COMP can't wake itself).
	A spec with "demo": false (an experimental wrapper whose model doesn't ship yet) is left out."""
	specs = [spec for spec in (load_spec(spec_name) for spec_name in names) if spec.get('demo', True)]
	# Over puts earlier inputs on top: box/joint overlays first, opaque image results last.
	specs.sort(key=lambda spec: spec['kind'] == 'image' or spec.get('result') == 'inference')
	position, inputs, outputs = _take_place(parent_comp, name)
	demo = parent_comp.create('baseCOMP', name)
	if position is not None:
		demo.nodeX, demo.nodeY = position
	models = demo.appendCustomPage('Models')
	controls = {}
	views = {}	# wrapper name -> its Show menu (wrappers with extra outputs: which one is its layer)
	for spec in specs:
		toggle = models.appendToggle(spec['name'].capitalize(), label=spec.get('label', spec['name']))[0]
		toggle.val = toggle.default = False
		controls[toggle.name] = spec['name']
		extras = [extra['name'] for extra in spec.get('extra_outputs', ())]
		if extras:
			menu = models.appendMenu(spec['name'].capitalize() + 'show', label='    Show')[0]
			menu.menuNames = ['result'] + extras
			menu.menuLabels = [spec.get('result_label', 'Result')] + [name.capitalize() for name in extras]
			menu.val = menu.default = 'result'
			menu.help = f"Which of {spec.get('label', spec['name'])}'s outputs is its layer."
			views[spec['name']] = menu.name
		elif spec.get('flow'):
			# The color view (TOP 3) by default; Flow Data is TOP 1 (signed, so negative motion reads black).
			menu = models.appendMenu(spec['name'].capitalize() + 'show', label='    Show')[0]
			menu.menuNames = ['color', 'result']
			menu.menuLabels = ['Color', 'Flow Data']
			menu.val = menu.default = 'color'
			menu.help = f"Which of {spec.get('label', spec['name'])}'s outputs is its layer."
			views[spec['name']] = menu.name
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
	composite.par.format = 'rgba32float'	# float layers (flow, depth) keep their values: 8-bit clamps to 0-1
	for index, spec in enumerate(specs):
		toggle = next(key for key, value in controls.items() if value == spec['name'])
		wrapper = build(spec, demo)
		wrapper.allowCooking = False
		wrapper.par.Active.expr = f'parent().par.{toggle}.eval()'
		_connect(wrapper, frame)
		wrapper.nodeX, wrapper.nodeY = 2 * COLUMN, -index * DEMO_ROW
		# Referenced, not wired: a disabled wrapper keeps its last frame, so pick transparent.
		layer = demo.create('selectTOP', 'layer_' + spec['name'])
		shown = f"'{spec['name']}/debug'"
		if spec['name'] in views:
			view = f"parent().par.{views[spec['name']]}.eval()"
			shown = f"({shown} if {view} == 'result' else '{spec['name']}/out_' + {view})"
		layer.par.top.expr = f"{shown} if parent().par.{toggle} else 'transparent'"
		layer.nodeX, layer.nodeY = 3 * COLUMN, -index * DEMO_ROW + 20
		layer.viewer = False
		composite.inputConnectors[index].connect(layer)
	output = demo.create('outTOP', 'out_debug')
	output.par.connectorder = 1
	_connect(output, composite)
	passthrough = demo.create('outTOP', 'out_input')
	passthrough.par.connectorder = 2
	_connect(passthrough, frame)
	gate, startup = _model_gate(demo, controls)
	middle = -(len(specs) - 1) * DEMO_ROW // 2
	for node, (column, y) in {source: (0, 200), frame: (1, 200), passthrough: (5, 200), transparent: (3, 200),
		composite: (4, middle), output: (5, middle), gate: (0, -200), startup: (0, -350)}.items():
		node.nodeX, node.nodeY = column * COLUMN, y
	for node in (source, frame, transparent, composite, gate, startup):
		node.viewer = False
	_add_icon(demo, name, './out_debug')
	_restore_wires(demo, inputs, outputs)
	return demo


def _model_gate(demo, controls):
	"""`model_controls` switches each wrapper's cooking with its Models toggle ({toggle: wrapper});
	`initialize` applies the toggles when the COMP is created or the project starts."""
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
	return gate, startup


def build_audio_demo(parent_comp, name='AudioDemo'):
	"""(Re)build the audio workbench: an audio CHOP into the sound classifier and speech to text, and
	text to speech fed by a DAT input or (Speak What It Hears) by speech to text's newest sentence.
	Each model is enabled from the Models page and doesn't cook while off. TOP 1: the classifier's top
	results (top left) over the speech captions; CHOP 1 the spoken audio; DATs the transcript and the
	top results."""
	classifier, listener, speaker = 'MediaPipeSoundClassifier', 'MediaPipeSpeechToText', 'MediaPipeTextToSpeech'
	position, inputs, outputs = _take_place(parent_comp, name)
	demo = parent_comp.create('baseCOMP', name)
	if position is not None:
		demo.nodeX, demo.nodeY = position
	models = demo.appendCustomPage('Models')
	controls = {}
	for wrapper_name in (classifier, listener, speaker):
		spec = load_spec(wrapper_name)
		toggle = models.appendToggle(wrapper_name.capitalize(), label=spec.get('label', wrapper_name))[0]
		toggle.val = toggle.default = False
		controls[toggle.name] = wrapper_name
	echo = models.appendToggle('Echo', label='Speak What It Hears')[0]
	echo.val = echo.default = False
	echo.help = ('Text to speech says each sentence speech to text hears. With speakers (not headphones) '
		'the mic hears the voice too and it repeats itself.')
	about = demo.appendCustomPage('About')
	release = about.appendStr('Version', label='Version')[0]
	release.val = release.default = version()
	release.readOnly = True

	audio_in = demo.create('inCHOP', 'audio_in')
	text_in = demo.create('inDAT', 'text_in')
	audio = demo.create('nullCHOP', 'audio')
	_connect(audio, audio_in)
	wrappers = {}
	for wrapper_name in (classifier, listener, speaker):
		wrapper = build(load_spec(wrapper_name), demo)
		toggle = next(key for key, value in controls.items() if value == wrapper_name)
		wrapper.allowCooking = False
		wrapper.par.Active.expr = f'parent().par.{toggle}.eval()'
		wrappers[wrapper_name] = wrapper
	_connect(wrappers[classifier], audio)
	_connect(wrappers[listener], audio)
	# Text to speech's input: the DAT wired in, or speech to text's newest sentence.
	heard = demo.create('selectDAT', 'heard')
	heard.par.dat = f'{listener}/out_latest'
	to_speak = demo.create('switchDAT', 'to_speak')
	_connect(to_speak, text_in, heard)
	to_speak.par.index.expr = 'int(parent().par.Echo.eval())'
	_connect(wrappers[speaker], to_speak)

	listener_toggle, classifier_toggle = (next(k for k, v in controls.items() if v == n) for n in (listener, classifier))
	canvas = demo.create('constantTOP', 'transparent')
	canvas.par.outputresolution = 'custom'
	canvas.par.resolutionw, canvas.par.resolutionh = 1920, 1080
	for channel in ('colorr', 'colorg', 'colorb', 'alpha'):
		canvas.par[channel] = 0
	# Referenced, not wired: a disabled wrapper keeps its last frame, so pick transparent.
	captions = demo.create('selectTOP', 'layer_captions')
	captions.par.top.expr = f"'{listener}/debug' if parent().par.{listener_toggle} else 'transparent'"
	readout = demo.create('selectTOP', 'layer_top_results')
	readout.par.top.expr = f"'{classifier}/debug' if parent().par.{classifier_toggle} else 'transparent'"
	view = demo.create('overTOP', 'debug_composite')
	_connect(view, readout, captions)	# the readout over the captions, at the captions' size
	view.par.prefit = 'nativeres'
	view.par.justifyh, view.par.justifyv = 'left', 'top'
	view.par.tunit = 'pixels'
	view.par.tx, view.par.ty = 40, -40

	outputs_spec = (('outTOP', 'out_debug', view), ('outCHOP', 'out_voice', wrappers[speaker]),
		('outDAT', 'out_transcript', wrappers[listener]), ('outDAT', 'out_top', wrappers[classifier]))
	out_nodes = {}
	for order, (op_type, out_name, source) in enumerate(outputs_spec, 1):
		output = demo.create(op_type, out_name)
		output.par.connectorder = order
		out_nodes[out_name] = output
	_connect(out_nodes['out_debug'], view)
	out_nodes['out_voice'].inputConnectors[0].connect(_output(wrappers[speaker], 'out_audio'))
	out_nodes['out_transcript'].inputConnectors[0].connect(_output(wrappers[listener], 'out_transcript'))
	out_nodes['out_top'].inputConnectors[0].connect(_output(wrappers[classifier], 'out_top'))
	gate, startup = _model_gate(demo, controls)

	for node, (column, y) in {audio_in: (0, 0), audio: (1, 0), wrappers[classifier]: (2, 180),
		wrappers[listener]: (2, 0), text_in: (0, -320), heard: (1, -180), to_speak: (2, -250),
		wrappers[speaker]: (3, -250), canvas: (3, 330), readout: (3, 180), captions: (3, 30),
		view: (4, 110), out_nodes['out_debug']: (5, 110), out_nodes['out_top']: (5, 250),
		out_nodes['out_transcript']: (5, -40), out_nodes['out_voice']: (5, -250),
		gate: (0, -480), startup: (0, -620)}.items():
		node.nodeX, node.nodeY = column * COLUMN, y
	for node in (audio_in, text_in, audio, heard, to_speak, canvas, readout, captions, view, gate, startup):
		node.viewer = False
	_add_icon(demo, name, './out_debug')
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
	fallbacks = [node.path for node in comp.findChildren() if node.OPType == 'cplusplusDAT']
	assert not fallbacks, (f'[WrapperBuilder] {fallbacks}: CPlusPlus DATs loading a DLL by path (the Tracker DAT '
		'before TD registered it). Restart TD, rebuild, then export.')
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
