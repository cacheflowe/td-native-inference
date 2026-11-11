"""Builds the MixamoPuppet COMP and exports tox/MixamoPuppet.tox (run inside TD through
scripts/build_puppet_tox.py). Build output: change this file, never the COMP.

    In TOP (the tracked image) + In DAT (a Holistic / Pose tox's out_tracks, World Landmark
    Columns on) -> solver (puppet_solver.py: bone rotations + placement) -> rotations
    -> the FBX COMP's bones -> render (Center camera or an Over Image camera) -> outputs.
Outputs: TOP 1 the puppet render (transparent background), TOP 2 the same over the input image
(Over Image placement; Center: the render alone), CHOP the bone rotations.
The exported tox contains no character: the FBX is imported at runtime from the FBX File par.
"""

import wrapper_builder as wb

MODULES = ('puppet_solver', 'mixamo_retarget', 'hand_retarget')
NAME = 'MixamoPuppet'
CALLBACKS = '''# Puppet solver: see puppet_solver.py (local/modules).
import puppet_solver


def onSetupParameters(scriptOp):
	puppet_solver.setup(scriptOp)


def onPulse(par):
	puppet_solver.pulse(par)


def onCook(scriptOp):
	puppet_solver.cook(scriptOp)
'''
PAR_EXEC = '''def onValueChange(par, prev):
	if par.name == 'Fbxfile':
		parent().op('component').module.load_character(parent())


def onPulse(par):
	if par.name == 'Loadcharacter':
		parent().op('component').module.load_character(parent())
	elif par.name == 'Capturerest':
		parent().op('solver').par.Capturerest.pulse()
	elif par.name == 'Reloadmodules':
		parent().op('solver').par.Reloadmodule.pulse()
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
	'startup': (0, -600), 'local': (200, -300),
}


def _expr(par, text):
	par.expr = text
	par.mode = type(par.mode).EXPRESSION


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
	page.appendPulse('Reloadmodules', label='Reload Solver Modules')
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
	modules = comp.create('baseCOMP', 'local').create('baseCOMP', 'modules')
	for module_name in MODULES:
		wb._synced_dat(modules, module_name, f'python/{module_name}.py')
	wb._synced_dat(comp, 'component', 'python/puppet_runtime.py')
	_pages(comp)

	image = comp.create('inTOP', 'in_image')
	image.par.connectorder = 1
	tracks = comp.create('inDAT', 'in_tracks')
	tracks.par.connectorder = 2

	fbx = comp.create('fbxCOMP', 'fbx')
	solver = comp.create('scriptCHOP', 'solver')
	callbacks = comp.op('solver_callbacks') or comp.create('textDAT', 'solver_callbacks')
	callbacks.text = CALLBACKS
	solver.par.callbacks = callbacks.name
	callbacks.dock = solver
	solver.destroyCustomPars()
	modules.op('puppet_solver').module.setup(solver)
	solver.par.Tracks, solver.par.Rig, solver.par.Image = 'in_tracks', 'fbx', 'in_image'
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
	par_exec.par.pars = 'Fbxfile Loadcharacter Capturerest Reloadmodules'
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
		copy.op('solver').unstore('rest_bones')
		return wb.export(copy, destination)
	finally:
		holder.destroy()
