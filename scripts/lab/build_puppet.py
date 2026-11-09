"""Puppeteer lab for /project1/PuppetLab (POST to the bridge's /run): a clip or the webcam ->
MediaPipeHolistic (World Landmark Columns on) -> solver (Script CHOP, python/mixamo_retarget.py)
-> rotations -> the Mixamo character's bone rotations (assets/models/y-bot.fbx, not committed:
Mixamo's license allows use, not redistribution) -> a render. Rebuilds everything except the FBX
COMP, so an imported character and its materials stay. Laid out on the 150 x 100 lab grid: the
tracking band on top, the character and render band under it.

The solver keeps the character's rest pose in its storage (captured from the bones' constant
values, which binding leaves untouched); pulse Capture Rest Pose after re-importing the FBX."""

lab = op('/project1/PuppetLab') or op('/project1').create(baseCOMP, 'PuppetLab')
COL, ROW, BAND = 150, 100, -250
FBX_FILE = 'assets/models/y-bot.fbx'
CLIP = 'media/shacarri.mp4'
WEBCAM = '/project1/videodevin1'

CALLBACKS = '''# Mixamo puppet: the first track's world landmarks (Tracks DAT, a MediaPipeHolistic or
# MediaPipePose tracks table with World Landmark Columns on) -> one rx/ry/rz channel per bone of
# the FBX COMP's skeleton (local Euler degrees, rotate order xyz). See python/mixamo_retarget.py.
import importlib

import mixamo_retarget

_rigs = {}		# Script CHOP path -> Rig (fetch() returns a copy, so build once, not per frame)
_columns = {}	# Tracks DAT header -> [(landmark name, wx column, conf column or None)]


def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Puppet')
	p = page.appendInt('Frame', label='Frame')
	p[0].expr = 'absTime.frame'	# cook every frame, not only when a render pulls the bones
	p[0].readOnly = True
	page.appendDAT('Tracks', label='Tracks DAT')
	page.appendCOMP('Rig', label='Character FBX COMP')
	p = page.appendToggle('Mirror', label='Mirror')
	p[0].help = 'Swap left and right, for a mirrored (selfie) camera image.'
	page.appendPulse('Capturerest', label='Capture Rest Pose')
	page.appendPulse('Reloadmodule', label='Reload Solver Module')


def onPulse(par):
	if par.name == 'Capturerest':
		_capture(par.owner)
	elif par.name == 'Reloadmodule':
		importlib.reload(mixamo_retarget)
		_rigs.pop(par.owner.path, None)
	par.owner.cook(force=True)


def _capture(scriptOp):
	"""Rest pose from the bones' constant values (.val: binding them to this CHOP leaves those)."""
	fbx = scriptOp.par.Rig.eval()
	bones = {}
	for bone in fbx.findChildren(type=nullCOMP, maxDepth=1) if fbx else ():
		bones[bone.name] = {
			'parent': bone.inputCOMPs[0].name if bone.inputCOMPs else None,
			't': [bone.par.tx.val, bone.par.ty.val, bone.par.tz.val],
			'r': [bone.par.rx.val, bone.par.ry.val, bone.par.rz.val],
		}
	scriptOp.store('rest_bones', bones)
	_rigs.pop(scriptOp.path, None)
	return bones


def _rig(scriptOp):
	if scriptOp.path not in _rigs:
		bones = scriptOp.fetch('rest_bones', None) or _capture(scriptOp)
		_rigs[scriptOp.path] = mixamo_retarget.Rig(bones) if bones else None
	return _rigs[scriptOp.path]


def _world_columns(tracks):
	"""Per world landmark: (name without holistic's 'pose_' prefix, _wx column, _conf column)."""
	key = (tracks.numCols, tracks[0, 0].val, tracks[0, tracks.numCols - 1].val)
	cols = _columns.get(key)
	if cols is None:
		header = [c.val for c in tracks.row(0)]
		index = {name: i for i, name in enumerate(header)}
		cols = []
		for i, name in enumerate(header):
			if name.endswith('_wx'):
				joint = name[:-3]
				cols.append((joint[len('pose_'):] if joint.startswith('pose_') else joint, i, index.get(joint + '_conf')))
		cols = _columns[key] = (cols, index.get('lost_frames'), index.get('total_frames'))
	return cols


def _landmarks(tracks):
	"""The current person's world landmarks: {name: [x, y, z]}, holistic's 'pose_' prefix
	dropped, and a joint whose keypoint is absent (conf 0: an unseen hand) left out. With more
	than one track (one held after a dropout), the one matched this frame, longest-lived first."""
	if tracks is None or tracks.numRows < 2:
		return None
	cols, lost, total = _world_columns(tracks)
	if not cols:
		return None	# World Landmark Columns off
	r = 1
	if lost is not None and total is not None and tracks.numRows > 2:
		r = min(range(1, tracks.numRows), key=lambda i: (float(tracks[i, lost]), -float(tracks[i, total])))
	row = tracks.row(r)
	out = {}
	for name, i, conf in cols:
		if conf is None or float(row[conf].val) > 0.0:
			out[name] = [float(row[i].val), float(row[i + 1].val), float(row[i + 2].val)]
	return out


def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.numSamples = 1
	rig = _rig(scriptOp)
	if rig is None:
		return
	landmarks = _landmarks(scriptOp.par.Tracks.eval())
	if landmarks:
		rotations = mixamo_retarget.solve(rig, landmarks, mirror=scriptOp.par.Mirror.eval())
	else:
		rotations = {name: rig.bones[name]['r'] for name in rig.order}
	for bone, values in rotations.items():
		for axis, value in zip(('rx', 'ry', 'rz'), values):
			scriptOp.appendChan(f'{bone}_{axis}')[0] = value
'''


def node(kind, name, col, row, band=0):
	n = lab.create(kind, name)
	n.nodeX, n.nodeY = col * COL, band * BAND - row * ROW
	return n


# Rebuild everything but the character, keeping the source choice and Mirror.
kept = {'source': lab.op('source').par.index.eval() if lab.op('source') else 0,
	'mirror': bool(lab.op('solver').par.Mirror.eval()) if lab.op('solver') and hasattr(lab.op('solver').par, 'Mirror') else False}
for child in list(lab.children):
	if child.valid and child.name != 'ybot':	# docked DATs go with their op
		child.destroy()

# ---- tracking band
clip = node(moviefileinTOP, 'clip', 0, 0)
clip.par.file = CLIP
webcam = node(selectTOP, 'webcam', 0, 1)
webcam.par.top = WEBCAM
source = node(switchTOP, 'source', 1, 0)
source.inputConnectors[0].connect(clip)
source.inputConnectors[1].connect(webcam)
source.par.index = kept['source']

holistic = lab.loadTox(project.folder + '/tox/MediaPipeHolistic.tox')
holistic.name = 'holistic'
holistic.nodeX, holistic.nodeY = 2 * COL, 0
holistic.inputConnectors[0].connect(source)
holistic.par.Outputworld = True

tracks = node(selectDAT, 'tracks', 3, 0)
tracks.par.dat = 'holistic/out_tracks'
solver = node(scriptCHOP, 'solver', 4, 0)
lab.op('solver_callbacks').text = CALLBACKS
lab.op('solver_callbacks').nodeX, lab.op('solver_callbacks').nodeY = 4 * COL, -ROW
solver.par.setuppars.pulse()	# runs onSetupParameters
rotations = node(nullCHOP, 'rotations', 5, 0)
rotations.inputConnectors[0].connect(solver)

overlay = node(overTOP, 'overlay', 3, 1)
outputs = {c.outOP.name: c for c in holistic.outputConnectors}	# order isn't by family
overlay.inputConnectors[0].connect(outputs['debug'])
overlay.inputConnectors[1].connect(outputs['passthrough'])

# ---- character band
fbx = lab.op('ybot')
if fbx is None:
	fbx = lab.create(fbxCOMP, 'ybot')
	fbx.par.file = FBX_FILE
	fbx.par.imp.pulse()
fbx.nodeX, fbx.nodeY = 0, BAND

solver.par.Tracks = 'tracks'
solver.par.Rig = 'ybot'
solver.par.Mirror = kept['mirror']
solver.par.Capturerest.pulse()
solver.cook(force=True)

# The import exports the FBX's animation (exportAnim) onto some bones (forearms, hands, head,
# fingers), which would override the binding: switch that off.
for chop in fbx.findChildren(type=CHOP):
	chop.export = False
# Bind every bone's rotation to the solver (constant values stay as the rest pose).
for bone in fbx.findChildren(type=nullCOMP, maxDepth=1):
	for axis in ('rx', 'ry', 'rz'):
		bone.par[axis].expr = f"op('../rotations')['{bone.name}_{axis}']"
		bone.par[axis].mode = ParMode.EXPRESSION

camera = node(cameraCOMP, 'camera', 1, 0, 1)
camera.par.tx, camera.par.ty, camera.par.tz = 0, 95, 650
camera.par.fov = 40
camera.par.far = 5000
light = node(lightCOMP, 'light', 2, 0, 1)
light.par.tx, light.par.ty, light.par.tz = 200, 300, 400
render = node(renderTOP, 'render', 3, 0, 1)
render.par.camera = 'camera'
render.par.geometry = 'ybot'
render.par.lights = 'light'
render.par.resolutionw, render.par.resolutionh = 1280, 720
render.par.outputresolution = 'custom'
render.par.bgcolora = 0

view = node(layoutTOP, 'view', 4, 0, 1)
view.inputConnectors[0].connect(overlay)
view.inputConnectors[1].connect(render)
view.par.align = 'horizlr'
view.par.fit = 'nativeres'
null = node(nullTOP, 'null_view', 5, 0, 1)
null.inputConnectors[0].connect(view)
null.viewer = True

print({'bones': len(fbx.findChildren(type=nullCOMP, maxDepth=1)), 'channels': solver.numChans,
	'solver_errors': solver.errors(), 'holistic': holistic.path})
