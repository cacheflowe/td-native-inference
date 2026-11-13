"""Builds the MixamoPuppet COMP and exports tox/MixamoPuppet.tox (run inside TD through
scripts/build_puppet_tox.py). Build output: change this file, never the COMP.

    In TOP (the tracked image) + In DAT (a Holistic / Pose tox's out_tracks, World Landmark
    Columns on) -> solver (the Puppet CHOP, cpp/Puppet: bone rotations + placement) -> rotations
    -> the FBX COMP's bones -> render (Center camera or an Over Image camera) -> outputs.
The solver reads the rig's rest pose from `rest_pose`, which puppet_runtime.capture_rest
fills from the FBX COMP's bones when the character loads.
Outputs: TOP 1 the puppet render (transparent background), TOP 2 the same over the input image
(Over Image placement; Center: the render alone), CHOP the bone rotations.
The exported tox contains no character: the FBX is imported at runtime from the FBX File par.
"""

import wrapper_builder as wb

NAME = 'MixamoPuppet'
PUPPET_CHOP_PLUGIN = 'plugins/PuppetCHOP.dll'
PAR_EXEC = '''def onValueChange(par, prev):
	if par.name == 'Fbxfile':
		parent().op('component').module.load_character(parent())


def onPulse(par):
	if par.name == 'Loadcharacter':
		parent().op('component').module.load_character(parent())
	elif par.name == 'Capturerest':
		parent().op('component').module.capture_rest(parent())
'''
STARTUP = '''def onCreate():
	run("args[0].module.load_character(args[0].parent())", parent().op('component'), delayFrames=1)


def onStart():
	onCreate()
'''
POSITIONS = {
	'in_image': (0, 100), 'in_tracks': (0, -100), 'solver': (200, -100), 'rotations': (400, -100),
	'out_rotations': (600, -100), 'fbx': (400, 100), 'light': (400, 300), 'camera': (400, 450),
	'camera_image': (400, 600), 'render': (600, 100), 'over': (800, 0), 'stage': (1000, 0),
	'puppet': (800, 200), 'over_image': (1200, 0), 'component': (0, -300), 'par_exec': (0, -450),
	'startup': (0, -600), 'rest_pose': (200, 100),
}


def _expr(par, text):
	par.expr = text
	par.mode = type(par.mode).EXPRESSION


# The Puppet CHOP's Smoothing page, exposed on the tox (name, label, default, slider range, help).
SMOOTHING = [
	('Smoothcutoff', 'Landmark Cutoff (Hz)', 1.5, 0.0, 5.0, 'Smooths the tracked landmarks before solving (a One Euro '
		"filter, like the Filter CHOP's): the cutoff while still. Lower is smoother and slower. 0 = off."),
	('Smoothspeed', 'Speed Coefficient', 4.0, 0.0, 20.0, 'How fast the smoothing lets go as the person moves faster: '
		'higher follows quick moves more closely.'),
	('Smoothslope', 'Slope Cutoff (Hz)', 1.0, 0.1, 5.0, 'Smooths the speed the filter reacts to.'),
	('Twistcutoff', 'Hand Roll Cutoff (Hz)', 2.0, 0.0, 5.0, "Smooths the forearms', hands' and fingers' roll (from "
		'the short knuckle line) and holds it while a hand is seen edge-on or not at all. 0 = off.'),
]
# Its toggles (name, label, help), on by default.
SMOOTHING_TOGGLES = [
	('Kneelimits', 'Knee Limits', "Knees bend like hinges, within a natural range: depth noise can't bend them backward "
		'or sideways.'),
	('Relaxunseen', 'Relax Unseen Limbs', 'Joints the tracker reports as unseen (a person framed from the chest up): '
		"the hips go under the shoulders and the arms hang, instead of the character's rest pose. Legs stay straight."),
]
SMOOTHING_PARS = [s[0] for s in SMOOTHING] + [t[0] for t in SMOOTHING_TOGGLES]


def _puppet_chop(comp, name):
	"""The Puppet CHOP: the registered custom OP, or (until TD has registered it, i.e. the next
	restart after the first build) a CPlusPlus CHOP loading plugins/PuppetCHOP.dll, which export
	refuses."""
	try:
		return comp.create('PuppetCHOP', name)
	except Exception:
		node = comp.create('cplusplusCHOP', name)
		node.par.plugin = (wb.root() / PUPPET_CHOP_PLUGIN).as_posix()
		node.cook(force=True)	# loads the plugin, which adds its parameters
		return node


def _pages(comp):
	page = comp.appendCustomPage('Puppet')
	wb._page_par(page, 'File', 'Fbxfile', 'FBX File', '', 'A Mixamo character (FBX, with skin). Loads when set.')
	page.appendPulse('Loadcharacter', label='Load Character')
	status = wb._page_par(page, 'Str', 'Status', 'Status', '', 'What the last load did.')
	status.readOnly = True
	menu = page.appendMenu('Placement', label='Placement')[0]
	menu.menuNames, menu.menuLabels = ['center', 'image'], ['Center', 'Over Image']
	menu.default = menu.val = 'image'
	menu.help = ('Center: the character stays at the origin, seen by a fixed camera. Over Image: it is drawn '
		'over the tracked person in the input image, by a camera matching Image FOV.')
	wb._page_par(page, 'Float', 'Fov', 'Image FOV', 50.0, 'Horizontal field of view assumed for the input camera (Over Image).', 20.0, 90.0)
	wb._page_par(page, 'Toggle', 'Mirror', 'Mirror', False, 'Center placement: swap left and right, for a mirrored (selfie) camera image.')
	wb._page_par(page, 'Float', 'Placesmoothing', 'Placement Smoothing', 0.6,
		"Smooths the character's position over the image (depth is the noisiest). 0 = raw.", 0.0, 0.99)
	size = page.appendWH('Resolution', label='Center Resolution')
	size[0].default = size[0].val = 1280
	size[1].default = size[1].val = 720
	page.appendPulse('Capturerest', label='Capture Rest Pose')
	smoothing = comp.appendCustomPage('Smoothing')
	for name, label, default, lo, hi, text in SMOOTHING:
		wb._page_par(smoothing, 'Float', name, label, default, text, lo, hi)
	for name, label, text in SMOOTHING_TOGGLES:
		wb._page_par(smoothing, 'Toggle', name, label, True, text)
	about = comp.appendCustomPage('About')
	wb._page_par(about, 'Str', 'Version', 'Version', wb.version(),
		'The td-native-inference release this tox was built with.').readOnly = True


def build(parent_comp):
	old = parent_comp.op(NAME)
	# The input wires as connectors: a TOP and a DAT out of other COMPs can't be rewired by owner.
	wires = [[c for c in connector.connections] for connector in old.inputConnectors] if old is not None else []
	position, _, outputs = wb._take_place(parent_comp, NAME)
	comp = parent_comp.create('baseCOMP', NAME)
	if position is not None:
		comp.nodeX, comp.nodeY = position
	wb._synced_dat(comp, 'component', 'python/puppet_runtime.py')
	_pages(comp)

	image = comp.create('inTOP', 'in_image')
	image.par.connectorder = 1
	tracks = comp.create('inDAT', 'in_tracks')
	tracks.par.connectorder = 2

	fbx = comp.create('fbxCOMP', 'fbx')
	solver = _puppet_chop(comp, 'solver')
	comp.create('tableDAT', 'rest_pose').clear()
	solver.par.Restpose = 'rest_pose'
	for name in SMOOTHING_PARS:
		_expr(solver.par[name], f'parent().par.{name}')
	solver.par.Tracks, solver.par.Image = 'in_tracks', 'in_image'
	for name in ('Placement', 'Fov', 'Mirror', 'Placesmoothing'):
		_expr(solver.par[name], f'parent().par.{name}')
	rotations = comp.create('nullCHOP', 'rotations')
	rotations.inputConnectors[0].connect(solver)
	comp.create('outCHOP', 'out_rotations').inputConnectors[0].connect(rotations)

	light = comp.create('lightCOMP', 'light')
	light.par.tx, light.par.ty, light.par.tz = 200, 300, 400
	camera = comp.create('cameraCOMP', 'camera')
	camera.par.tx, camera.par.ty, camera.par.tz = 0, 95, 650
	camera.par.fov, camera.par.far = 40, 5000
	camera_image = comp.create('cameraCOMP', 'camera_image')
	camera_image.par.tx = camera_image.par.ty = camera_image.par.tz = 0
	_expr(camera_image.par.fov, 'parent().par.Fov')
	camera_image.par.near, camera_image.par.far = 1, 100000
	render = comp.create('renderTOP', 'render')
	_expr(render.par.camera, "'camera_image' if parent().par.Placement == 'image' else 'camera'")
	render.par.lights = 'light'
	render.par.outputresolution = 'custom'
	_expr(render.par.resolutionw, "op('in_image').width if parent().par.Placement == 'image' else parent().par.Resolutionw")
	_expr(render.par.resolutionh, "op('in_image').height if parent().par.Placement == 'image' else parent().par.Resolutionh")
	render.par.bgcolora = 0
	over = comp.create('compositeTOP', 'over')
	over.par.operand = 'over'
	over.inputConnectors[0].connect(render)
	over.inputConnectors[1].connect(image)
	stage = comp.create('switchTOP', 'stage')
	stage.inputConnectors[0].connect(render)
	stage.inputConnectors[1].connect(over)
	_expr(stage.par.index, 'parent().par.Placement.menuIndex')
	puppet = comp.create('outTOP', 'puppet')
	puppet.par.connectorder = 1
	puppet.inputConnectors[0].connect(render)
	over_image = comp.create('outTOP', 'over_image')
	over_image.par.connectorder = 2
	over_image.inputConnectors[0].connect(stage)

	par_exec = comp.create('parameterexecuteDAT', 'par_exec')
	par_exec.text = PAR_EXEC
	_expr(par_exec.par.op, 'parent()')
	par_exec.par.pars = 'Fbxfile Loadcharacter Capturerest'
	par_exec.par.valuechange = True
	par_exec.par.onpulse = True
	startup = comp.create('executeDAT', 'startup')
	startup.text = STARTUP
	for name, (x, y) in POSITIONS.items():
		node = comp.op(name)
		if node is not None:
			node.nodeX, node.nodeY = x, y
	for node in comp.findChildren():
		node.viewer = False
	wb._add_icon(comp, NAME, './puppet')
	for index, connections in enumerate(wires):
		for connection in connections:
			comp.inputConnectors[index].connect(connection)
	wb._restore_wires(comp, [], outputs)
	return comp


def export(comp, destination):
	"""Save a copy with the character removed: the FBX COMP's content (the imported mesh and its
	bound bones) is Mixamo's, so the tox carries an empty FBX COMP and the Load Character step."""
	holder = comp.parent().create('baseCOMP', '_puppet_export')	# the copy keeps the COMP's own name
	copy = holder.copyOPs([comp])[0]
	try:
		old = copy.op('fbx')
		x, y = old.nodeX, old.nodeY
		old.destroy()
		fbx = copy.create('fbxCOMP', 'fbx')
		fbx.nodeX, fbx.nodeY = x, y
		fbx.viewer = False
		copy.par.Fbxfile = ''
		copy.par.Status = ''
		copy.op('render').par.geometry = ''
		solver = copy.op('solver')
		assert solver.OPType != 'cplusplusCHOP', ('[PuppetBuilder] the solver is a CPlusPlus CHOP loading a DLL by '
			'path: restart TouchDesigner so it registers the Puppet CHOP, then rebuild')
		copy.op('rest_pose').clear()	# the character's skeleton: Mixamo's too
		return wb.export(copy, destination)
	finally:
		holder.destroy()
