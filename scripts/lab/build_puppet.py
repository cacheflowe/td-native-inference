"""Puppeteer lab for /project1/PuppetLab (POST to the bridge's /run). Two separate paths:
	body: clip / dancer / webcam -> source -> MediaPipe Holistic GPU or CPU (Body Tracker; world
	      landmarks, its hands also drive the fingers) -> solver -> rotations -> the Mixamo
	      character's bones (assets/models/y-bot.fbx)
	hand: a close-up hand clip / webcam -> hand_source -> MediaPipe Hands -> hand_solver
	      -> hand_rotations -> a rigged hand's bones (assets/models/hand/handRig_02.fbx)
Each path's solver is the Puppet CHOP (cpp/Puppet), with the lab's Smoothing settings; its rig
comes from `rest_pose`.
Models and media aren't committed (their licenses allow use, not redistribution).

The lab COMP's Puppet page drives both: Body Source and Hand Input pick the videos; Placement
(Center: each model on its own stage; Over Image: rendered over its video by a camera matching
the image, root_tx/ty/tz from the solvers) with an Image FOV per path; Tracked Hand. The output
has its own output, body_out and hand_out: the tracked video over the puppet.

Re-runnable: nodes are reused by name (trackers, FBX COMPs and their settings stay) and
positioned left to right in two bands (body on top, hand under it). Nodes of earlier versions
and '_' probes are removed; other nodes someone added are left alone. Network boxes are drawn
afterwards by scripts/lab/annotate_puppet.py (see GROUPS).

The rest pose is captured from the bones' constant values (which binding leaves untouched) into
`rest_pose`; rebuild after re-importing an FBX."""

import sys

if project.folder + '/python' not in sys.path:
	sys.path.insert(0, project.folder + '/python')
import importlib  # noqa: E402

import puppet_builder  # noqa: E402
import puppet_runtime  # noqa: E402

importlib.reload(puppet_runtime)
importlib.reload(puppet_builder)

lab = op('/project1/PuppetLab') or op('/project1').create(baseCOMP, 'PuppetLab')
COL, ROW = 220, 170
HAND_ROW = 4	# the hand band's main row
BODY_FBX = 'assets/models/y-bot.fbx'
HAND_FBX = 'assets/models/hand/handRig_02.fbx'
HAND_MAPS = {'hand_color': 'hand_Co.jpeg', 'hand_normal': 'hand_No.png', 'hand_spec': 'hand_Sp.jpeg'}
CLIPS = [('clip', 'media/shacarri.mp4'), ('dancer', 'media/dancer.mp4')]
HAND_CLIP = 'media/hands.mp4'
WEBCAM = '/project1/videodevin1'
# Network boxes, drawn by the caller through /annotate: title -> node names.
GROUPS = {
	'Body source': ['clip', 'dancer', 'webcam', 'source'],
	'Body tracking': ['holistic', 'holistic_gpu', 'overlay_cpu', 'overlay_gpu', 'overlay'],
	'Body puppet: Mixamo character': ['tracks', 'rest_pose', 'solver', 'rotations', 'ybot', 'camera', 'light',
		'camera_image', 'render', 'mirror', 'image', 'over_body', 'body_stage', 'body_fit_tracking', 'body_fit_puppet', 'body_view', 'body_out', 'out1'],
	'Hand source': ['hand_clip', 'hand_webcam', 'hand_source'],
	'Hand tracking': ['hands', 'hand_overlay'],
	'Hand puppet': ['hand_tracks', 'hand_rest_pose', 'hand_solver', 'hand_rotations', 'hand', 'hand_camera',
		'hand_light', 'hand_camera_image', 'render_hand', 'hand_mirror', 'hand_image', 'over_hand', 'hand_stage',
		'hand_fit_tracking', 'hand_fit_puppet', 'hand_view', 'hand_out', 'out2',
		'hand_mat', *HAND_MAPS],
}
P = "parent().par."


def place(o, col, row):
	o.nodeX, o.nodeY = int(col * COL), int(-row * ROW)
	return o


def node(kind, name, col, row):
	"""The lab's op called name, (re)created if it isn't a `kind`."""
	o = lab.op(name)
	if o is not None and type(o) is not kind:
		o.destroy()
		o = None
	o = o or lab.create(kind, name)
	return place(o, col, row)


def expr(par, text):
	par.expr = text
	par.mode = ParMode.EXPRESSION


def wire(o, *inputs):
	for i, src in enumerate(inputs):
		o.inputConnectors[i].connect(src)
	return o


# ---------------------------------------------------------------- lab controls
kept = {p.name: p.eval() for p in lab.customPars}
lab.destroyCustomPars()
page = lab.appendCustomPage('Puppet')
src = page.appendMenu('Source', label='Body Source')[0]
src.menuNames, src.menuLabels = [n for n, _ in CLIPS] + ['webcam'], ['Clip', 'Dancer', 'Webcam']
pl = page.appendMenu('Placement', label='Placement')[0]
pl.menuNames, pl.menuLabels = ['center', 'image'], ['Center', 'Over Image']
pl.default = 'image'
pl.help = 'Center: each puppet on its own stage. Over Image: drawn over its video where the person / hand is.'
fov = page.appendFloat('Fov', label='Body Image FOV')[0]
fov.default, fov.normMin, fov.normMax = 50.0, 20.0, 90.0
fov.help = 'Horizontal field of view assumed for the body video\'s camera (Over Image).'
hi = page.appendMenu('Handinput', label='Hand Input')[0]
hi.menuNames, hi.menuLabels = ['clip', 'webcam'], ['Hand Clip', 'Webcam']
hfov = page.appendFloat('Handfov', label='Hand Image FOV')[0]
hfov.default, hfov.normMin, hfov.normMax = 50.0, 20.0, 90.0
hfov.help = 'Horizontal field of view assumed for the hand video\'s camera (Over Image).'
hand_par = page.appendMenu('Hand', label='Tracked Hand')[0]
hand_par.menuNames, hand_par.menuLabels = ['any', 'right', 'left'], ['Any', 'Right', 'Left']
hand_par.help = 'Which of the person\'s hands drives the hand model (Any: the one in view); the other side mirrors onto it.'
tr = page.appendMenu('Bodytracker', label='Body Tracker')[0]
tr.menuNames, tr.menuLabels = ['holistic_gpu', 'holistic'], ['Holistic GPU', 'Holistic (CPU)']
tr.help = "Which tracker drives the body puppet. CPU Holistic smooths its landmarks inside MediaPipe; the GPU one doesn't."
smoothing = lab.appendCustomPage('Smoothing')
for name, label, default, lo, hi, text in puppet_builder.SMOOTHING:
	p = smoothing.appendFloat(name, label=label)[0]
	p.default, p.normMin, p.normMax, p.help = default, lo, hi, text
	p.clampMin, p.min = True, 0.0
for name, label, text in puppet_builder.SMOOTHING_TOGGLES:
	t = smoothing.appendToggle(name, label=label)[0]
	t.default, t.help = True, text
for name, default in (('Source', 'dancer'), ('Placement', 'image'), ('Fov', 50.0), ('Handinput', 'clip'),
		('Handfov', 50.0), ('Hand', 'any'), ('Bodytracker', 'holistic_gpu'),
		*[(s[0], s[2]) for s in puppet_builder.SMOOTHING], *[(t[0], True) for t in puppet_builder.SMOOTHING_TOGGLES]):
	try:
		getattr(lab.par, name).val = kept.get(name, default)
	except Exception:	# a menu entry that no longer exists
		getattr(lab.par, name).val = default


def tracker(tox, name, col, row, source):
	"""The tracker tox, reloaded every build (its Python is embedded, so this is how tracker fixes
	arrive), keeping the old instance's parameter values and expressions."""
	old = lab.op(name)
	kept = {}
	if old is not None:
		kept = {p.name: (p.mode, p.expr, p.eval()) for p in old.customPars if not p.readOnly and not p.isPulse}
		old.destroy()
	comp = lab.loadTox(project.folder + '/tox/' + tox)
	comp.name = name
	for par_name, (mode, text, value) in kept.items():
		par = getattr(comp.par, par_name, None)
		if par is None or par.readOnly:
			continue
		if mode == ParMode.EXPRESSION:
			expr(par, text)
		else:
			try:
				par.val = value
			except Exception:	# a menu entry that no longer exists
				pass
	place(comp, col, row)
	wire(comp, source)
	if hasattr(comp.par, 'Outputworld'):	# Holistic GPU always outputs world landmarks
		comp.par.Outputworld = True
	return comp, {c.outOP.name: c for c in comp.outputConnectors}	# order isn't by family


def fbx_comp(name, file, col, row):
	fbx = lab.op(name)
	if fbx is None:
		fbx = lab.create(fbxCOMP, name)
		fbx.par.file = file
		fbx.par.imp.pulse()
	return place(fbx, col, row)


def puppet(prefix, tracks_dat, fbx, image, fov_par, row):
	"""tracks -> solver (the Puppet CHOP) -> rotations, bound to the FBX COMP's bones and root.
	tracks_dat: a DAT path, or '=' + an expression for one."""
	tracks = node(selectDAT, prefix + 'tracks', 3, row)
	if tracks_dat.startswith('='):
		expr(tracks.par.dat, tracks_dat[1:])
	else:
		tracks.par.dat = tracks_dat
	# Rest pose from the bones' constant values (binding them leaves those).
	rest = node(tableDAT, prefix + 'rest_pose', 3, row + 0.7)
	rest.clear()
	for r in puppet_runtime.rest_rows(fbx):
		rest.appendRow(r)
	solver = lab.op(prefix + 'solver')
	if solver is not None and solver.OPType != 'PuppetCHOP':
		solver.destroy()
		solver = None
	solver = place(solver or puppet_builder._puppet_chop(lab, prefix + 'solver'), 4, row)
	solver.par.Restpose = rest.name
	solver.par.Tracks = tracks.name
	solver.par.Image = image.name
	expr(solver.par.Placement, P + 'Placement')
	expr(solver.par.Fov, P + fov_par)
	for name in puppet_builder.SMOOTHING_PARS:
		expr(solver.par[name], P + name)
	rotations = wire(node(nullCHOP, prefix + 'rotations', 5, row), solver)
	solver.cook(force=True)

	# The import exports the FBX's animation (exportAnim) onto some bones, which would override
	# the binding: switch that off. Bind every bone's rotation (constant values stay as the rest
	# pose) and the COMP's translate.
	for chop in fbx.findChildren(type=CHOP):
		chop.export = False
	depth = len(lab.path.split('/'))
	for bone in fbx.findChildren(type=nullCOMP):
		up = '../' * (len(bone.path.split('/')) - depth - 1)	# from the bone's network to the lab
		for axis in ('rx', 'ry', 'rz'):
			expr(bone.par[axis], f"op('{up}{rotations.name}')['{bone.name}_{axis}']")
	for axis in ('tx', 'ty', 'tz'):
		expr(fbx.par[axis], f"op('{rotations.name}')['root_{axis}']")
	fbx.par.sx.mode = ParMode.CONSTANT
	fbx.par.sx = 1
	return solver


def stage(prefix, render_name, geometry, image, video, fov_par, row, camera, light_pos, size):
	"""Light, over-image camera and render for one puppet, then its stage: the render alone
	(Center) or over its video (Over Image)."""
	light = node(lightCOMP, prefix + 'light', 7, row + 0.6)
	light.par.tx, light.par.ty, light.par.tz = light_pos
	cam_image = node(cameraCOMP, prefix + 'camera_image', 7, row + 1.4)
	cam_image.par.tx = cam_image.par.ty = cam_image.par.tz = 0
	expr(cam_image.par.fov, P + fov_par)
	cam_image.par.near, cam_image.par.far = 1, 100000
	r = node(renderTOP, render_name, 8, row)
	r.par.geometry = geometry
	expr(r.par.camera, f"'{cam_image.name}' if {P}Placement == 'image' else '{camera.name}'")
	r.par.lights = light.name
	r.par.outputresolution = 'custom'
	expr(r.par.resolutionw, f"op('{video}').width if {P}Placement == 'image' else {size[0]}")
	expr(r.par.resolutionh, f"op('{video}').height if {P}Placement == 'image' else {size[1]}")
	r.par.bgcolora = 0
	# The other hand is solved mirrored onto a one-hand model: flip its render back.
	mirror = wire(node(flipTOP, prefix + 'mirror', 9, row - 0.7), r)
	expr(mirror.par.flipx, f"op('{prefix}rotations')['mirror'] > 0.5")
	over = node(compositeTOP, f'over_{prefix[:-1] or "body"}', 9, row + 0.7)
	over.par.operand = 'over'
	wire(over, mirror, image)
	st = wire(node(switchTOP, f'{prefix or "body_"}stage', 10, row), mirror, over)
	expr(st.par.index, P + 'Placement.menuIndex')
	return st


def output(prefix, tracked, stage_top, row):
	"""This path's own output: the tracked video (with its debug overlay) on top, the puppet
	under it, each fitted into 1280 x 720."""
	halves = []
	for i, (part, top) in enumerate((('tracking', tracked), ('puppet', stage_top))):
		fit = wire(node(fitTOP, f'{prefix}fit_{part}', 11, row - 0.4 + 0.8 * i), top)
		fit.par.fit = 'fitbest'
		fit.par.outputresolution = 'custom'
		fit.par.resolutionw, fit.par.resolutionh = 1280, 720
		halves.append(fit)
	v = wire(node(layoutTOP, prefix + 'view', 12, row), *halves)
	for par in v.pars():	# a known state, whatever an older build or a hand edit left
		if not par.readOnly and par.mode == ParMode.CONSTANT:
			par.val = par.default
	v.par.align = 'verttb'
	v.par.fit = 'fitbest'
	v.par.scaleres = True	# the first input's 1280 x 720, one cell per input: 1280 x 1440
	out = wire(node(nullTOP, prefix + 'out', 13, row), v)
	out.viewer = True
	return out


def lab_output(name, top, row):
	"""An Out TOP, so the lab COMP exposes this path's output."""
	return wire(node(outTOP, name, 13.75, row), top)


# ---------------------------------------------------------------- body path (top band)
for i, (name, file) in enumerate(CLIPS):
	node(moviefileinTOP, name, 0, i - 0.5).par.file = file
webcam = node(selectTOP, 'webcam', 0, len(CLIPS) - 0.5)
webcam.par.top = WEBCAM
source = wire(node(switchTOP, 'source', 1, 0.5), *[lab.op(n) for n, _ in CLIPS], webcam)
expr(source.par.index, P + 'Source.menuIndex')
holistic, holistic_out = tracker('MediaPipeHolistic.tox', 'holistic', 2, 0, source)
holistic_gpu, holistic_gpu_out = tracker('MediaPipeHolisticGPU.tox', 'holistic_gpu', 2, 1.2, source)
overlay_cpu = wire(node(overTOP, 'overlay_cpu', 1.4, -1.2), holistic_out['debug'], holistic_out['passthrough'])
overlay_gpu = wire(node(overTOP, 'overlay_gpu', 2, -1.2), holistic_gpu_out['debug'], holistic_gpu_out['passthrough'])
overlay = wire(node(switchTOP, 'overlay', 2.6, -1.2), overlay_gpu, overlay_cpu)
expr(overlay.par.index, P + 'Bodytracker.menuIndex')

ybot = fbx_comp('ybot', BODY_FBX, 6, 0)
image = node(selectTOP, 'image', 8, 1.4)
image.par.top = 'source'
body_solver = puppet('', "=parent().par.Bodytracker + '/out_tracks'", ybot, image, 'Fov', 0)
camera = node(cameraCOMP, 'camera', 7, -0.6)
camera.par.tx, camera.par.ty, camera.par.tz = 0, 95, 650
camera.par.fov, camera.par.far = 40, 5000
body_stage = stage('', 'render', 'ybot ybot/Alpha_Joints ybot/Alpha_Surface', image, 'source', 'Fov', 0,
	camera, (200, 300, 400), (1280, 720))
lab_output('out1', output('body_', overlay, body_stage, 0), 0)

# ---------------------------------------------------------------- hand path (bottom band)
H = HAND_ROW
hand_clip = node(moviefileinTOP, 'hand_clip', 0, H - 0.5)
hand_clip.par.file = HAND_CLIP
hand_webcam = node(selectTOP, 'hand_webcam', 0, H + 0.5)
hand_webcam.par.top = WEBCAM
hand_source = wire(node(switchTOP, 'hand_source', 1, H), hand_clip, hand_webcam)
expr(hand_source.par.index, P + 'Handinput.menuIndex')
hands, hands_out = tracker('MediaPipeHands.tox', 'hands', 2, H, hand_source)
hands.par.Active.mode = ParMode.CONSTANT
hands.par.Active = True
hand_overlay = wire(node(overTOP, 'hand_overlay', 2, H - 1.2), hands_out['debug'], hands_out['passthrough'])

hand = fbx_comp('hand', HAND_FBX, 6, H)
hand_image = node(selectTOP, 'hand_image', 8, H + 1.4)
hand_image.par.top = 'hand_source'
puppet('hand_', 'hands/out_tracks', hand, hand_image, 'Handfov', H)
expr(lab.op('hand_solver').par.Hand, P + 'Hand')
hand_camera = node(cameraCOMP, 'hand_camera', 7, H - 0.6)
hand_camera.par.tx, hand_camera.par.ty, hand_camera.par.tz = 0, 0, 260
hand_camera.par.fov, hand_camera.par.far = 40, 5000
hand_stage = stage('hand_', 'render_hand', 'hand/handSmooth_001', hand_image, 'hand_source', 'Handfov', H,
	hand_camera, (150, 250, 500), (720, 720))
lab_output('out2', output('hand_', hand_overlay, hand_stage, H), H)
for i, (name, file) in enumerate(HAND_MAPS.items()):
	node(moviefileinTOP, name, 3 + i, H + 2.4).par.file = 'assets/models/hand/' + file
mat = node(phongMAT, 'hand_mat', 6, H + 2.4)
mat.par.colormap, mat.par.normalmap, mat.par.specmap = HAND_MAPS
hand.op('handSmooth_001').par.material = mat.path
# The import's own texture loaders look for a hand_Co.jpg that doesn't resolve; hand_mat replaces them.
for top in hand.findChildren(type=TOP):
	top.bypass = True

# Nodes earlier versions of this lab made, probes ('_...') and the network boxes go (the caller
# redraws the boxes); anything else someone added stays.
RETIRED = ['view', 'null_view', 'side_by_side', 'over_image', 'stage', 'hands1', 'solver_callbacks', 'hand_solver_callbacks',
	'solver_cpp', 'hand_solver_cpp', 'rest_cpp', 'hand_rest_cpp', 'solver_py', 'solver_py_callbacks', 'solver_choice',
	'hand_solver_py', 'hand_solver_py_callbacks', 'hand_solver_choice']
for child in list(lab.findChildren(depth=1, includeUtility=True)):
	if child.valid and (child.name in RETIRED or child.name.startswith('_') or child.OPType == 'annotateCOMP'):
		child.destroy()

print({'body_channels': body_solver.numChans, 'hand_channels': lab.op('hand_solver').numChans,
	'errors': {o.name: o.errors() for o in lab.children if o.errors()}})
