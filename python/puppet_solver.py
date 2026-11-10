"""Script CHOP side of the puppets: a tracks table in, one rx/ry/rz channel per bone of an FBX
COMP's skeleton out, plus root_tx/ty/tz for the FBX COMP itself and mirror. Bind the bones'
rotations and the COMP's translate to these channels, and flip the render horizontally while
mirror is 1.

Rigs (mixamo_retarget.Rig, captured from the FBX COMP's bones):
	- a Mixamo character, from MediaPipe Holistic or Pose world landmarks (the first track)
	- a rigged hand (hand_retarget), from Holistic's left or right hand, or a MediaPipe Hands
	  track of that handedness. The model is one hand; the other hand is solved mirrored and
	  its render flipped back (mirror = 1). Mirroring the model itself (sx = -1) turns its
	  shading inside out.
The tracks table needs World Landmark Columns on (<joint>_wx/_wy/_wz).

Placement: Center leaves the FBX COMP at the origin; Over Image moves it so the posed model,
seen by a camera at the origin looking down -z (horizontal FOV = the Image FOV par, the Image
TOP's aspect), lies over the tracked person or hand (mixamo_retarget.place). Over Image ignores
Mirror (the model copies the image as it is).

Callbacks DAT:
	import puppet_solver
	def onSetupParameters(scriptOp): puppet_solver.setup(scriptOp)
	def onPulse(par): puppet_solver.pulse(par)
	def onCook(scriptOp): puppet_solver.cook(scriptOp)
"""
import importlib

import hand_retarget
import mixamo_retarget

_rigs = {}		# Script CHOP path -> (Rig, HandSpec or None); fetch() returns a copy, so build once
_columns = {}	# Tracks DAT header key -> column layout
_roots = {}		# Script CHOP path -> smoothed root translation
_last = {}		# Script CHOP path -> last solved rotations
_center = {}	# Script CHOP path -> last Center root of a hand
_sides = {}		# Script CHOP path -> the hand side last driven
_keys = {}		# Script CHOP path -> what its channels were last solved from
_chans = {}		# Script CHOP path -> (its channel names, its Channel objects)


def setup(scriptOp):
	_chans.pop(scriptOp.path, None)
	_keys.pop(scriptOp.path, None)
	page = scriptOp.appendCustomPage('Puppet')
	p = page.appendInt('Frame', label='Frame')
	p[0].expr = 'absTime.frame'	# cook every frame, not only when a render pulls the bones
	p[0].readOnly = True
	page.appendDAT('Tracks', label='Tracks DAT')
	page.appendCOMP('Rig', label='Character FBX COMP')
	p = page.appendMenu('Hand', label='Tracked Hand')
	p[0].menuNames, p[0].menuLabels = ['any', 'right', 'left'], ['Any', 'Right', 'Left']
	p[0].help = ('Hand rigs: which of the person\'s hands drives the model (Any: the one in view). '
		'The other side than the model\'s mirrors onto it.')
	p = page.appendToggle('Mirror', label='Mirror')
	p[0].help = 'Body rigs, Center placement: swap left and right, for a mirrored (selfie) camera image.'
	p = page.appendMenu('Placement', label='Placement')
	p[0].menuNames, p[0].menuLabels = ['center', 'image'], ['Center', 'Over Image']
	p[0].help = 'Center: the model stays at the origin. Over Image: root_tx/ty/tz put it over the tracked person in the image.'
	p = page.appendFloat('Fov', label='Image FOV')
	p[0].default = p[0].val = 50.0
	p[0].normMin, p[0].normMax = 20.0, 90.0
	p[0].help = 'Horizontal field of view assumed for the source camera; give the overlay camera the same.'
	p = page.appendTOP('Image', label='Image TOP')
	p[0].help = 'The tracked image, for its aspect ratio.'
	p = page.appendFloat('Placesmoothing', label='Placement Smoothing')
	p[0].default = p[0].val = 0.6
	p[0].normMin, p[0].normMax = 0.0, 0.99
	p[0].help = 'Smooths root_tx/ty/tz (depth from the image is the noisiest). 0 = raw.'
	page.appendPulse('Capturerest', label='Capture Rest Pose')
	page.appendPulse('Reloadmodule', label='Reload Solver Modules')


def pulse(par):
	owner = par.owner
	if par.name == 'Capturerest':
		capture(owner)
	elif par.name == 'Reloadmodule':
		importlib.reload(mixamo_retarget)
		importlib.reload(hand_retarget)
		_rigs.pop(owner.path, None)
		_keys.pop(owner.path, None)
	owner.cook(force=True)


def capture(scriptOp):
	"""Rest pose from the bones' constant values (.val: binding them to this CHOP leaves those).
	A bone's parent is its wired input COMP, else the bone COMP holding it (a Blender Armature)."""
	fbx = scriptOp.par.Rig.eval()
	bones = {}
	for bone in (c for c in fbx.findChildren() if c.OPType == 'nullCOMP') if fbx else ():
		holder = bone.parent()
		parent = bone.inputCOMPs[0] if bone.inputCOMPs else (holder if holder != fbx and holder.OPType == 'nullCOMP' else None)
		bones[bone.name] = {
			'parent': parent.name if parent else None,
			't': [bone.par.tx.val, bone.par.ty.val, bone.par.tz.val],
			'r': [bone.par.rx.val, bone.par.ry.val, bone.par.rz.val],
			's': bone.par.sx.val * bone.par.scale.val,
		}
	scriptOp.store('rest_bones', bones)
	_rigs.pop(scriptOp.path, None)
	_last.pop(scriptOp.path, None)
	_keys.pop(scriptOp.path, None)
	return bones


def rig(scriptOp):
	"""(Rig, HandSpec or None) for the Script CHOP's FBX COMP, or (None, None)."""
	if scriptOp.path not in _rigs:
		bones = scriptOp.fetch('rest_bones', None) or capture(scriptOp)
		r = mixamo_retarget.Rig(bones) if bones else None
		_rigs[scriptOp.path] = (r, hand_retarget.spec(r) if r else None)
	return _rigs[scriptOp.path]


# ---------------------------------------------------------------- tracks table

def _layout(tracks):
	"""Column layout: per landmark its world (_wx) and image (_x) columns, and the row columns."""
	key = (tracks.numCols, tracks[0, 0].val, tracks[0, tracks.numCols - 1].val)
	layout = _columns.get(key)
	if layout is None:
		header = [c.val for c in tracks.row(0)]
		index = {name: i for i, name in enumerate(header)}
		world, image = {}, {}
		for i, name in enumerate(header):
			if name.endswith('_wx'):
				world[name[:-3]] = i
			elif name.endswith('_x') and name[:-2] + '_y' in index:
				image[name[:-2]] = (i, index[name[:-2] + '_y'], index.get(name[:-2] + '_conf'))
		layout = _columns[key] = {
			'world': world, 'image': image, 'conf': {n: index.get(n + '_conf') for n in world},
			'lost': index.get('lost_frames'), 'total': index.get('total_frames'),
			'handedness': index.get('handedness'),
		}
	return layout


def _pick_row(tracks, layout, rows):
	"""With several tracks (one held after a dropout), the one matched this frame, longest-lived first."""
	lost, total = layout['lost'], layout['total']
	if lost is None or total is None or len(rows) < 2:
		return rows[0]
	return min(rows, key=lambda i: (float(tracks[i, lost]), -float(tracks[i, total])))


def read(tracks, hand=None):
	"""(world landmarks {name: [x, y, z]}, image points {name: (u, v, conf)}, hand side, the row's
	text) for the
	current person, or the current `hand` ('left' / 'right' / 'any'), names without their
	'pose_' / '<side>_hand_' prefix; a landmark whose keypoint is absent (conf 0: an unseen hand)
	left out. (None, None, None, None) when there's nothing to read.

	Sides are the person's own: MediaPipe Hands' handedness (0 = Left, 1 = Right; checked against
	the hand's 3D shape, whose fingers curl toward the palm) and Holistic's hand columns."""
	if tracks is None or tracks.numRows < 2:
		return None, None, None, None
	layout = _layout(tracks)
	if not layout['world']:
		return None, None, None, None	# World Landmark Columns off
	rows = list(range(1, tracks.numRows))
	prefix, side = 'pose_', None
	if hand is not None and 'right_hand_wrist' in layout['world']:	# Holistic: a column prefix per side
		row_index = _pick_row(tracks, layout, rows)
		side = hand
		if hand == 'any':	# the hand seen, the right one first
			seen = lambda s: (layout['conf'][f'{s}_hand_wrist'] is None	# noqa: E731
				or float(tracks[row_index, layout['conf'][f'{s}_hand_wrist']]) > 0.0)
			side = 'left' if not seen('right') and seen('left') else 'right'
		prefix = f'{side}_hand_'
		rows = [row_index]
	elif hand is not None:	# a MediaPipe Hands table: one row per hand, by handedness
		prefix, side = '', 'right' if hand == 'any' else hand
		col = layout['handedness']
		if col is not None:
			sides = {r: 'left' if round(float(tracks[r, col])) == 0 else 'right' for r in rows}
			if hand != 'any':
				rows = [r for r in rows if sides[r] == hand]
			if not rows:
				return None, None, None, None
			side = sides[_pick_row(tracks, layout, rows)]
	# One row's text split once: tracks.row() would make a Cell object per column (~2500 for
	# Holistic with its face mesh), most of them never read.
	line = tracks.text.splitlines()[_pick_row(tracks, layout, rows)]
	row = line.split('\t')
	strip = len(prefix)
	world, image = {}, {}
	for name, i in layout['world'].items():
		if not name.startswith(prefix):
			continue
		conf = layout['conf'][name]
		if conf is None or float(row[conf]) > 0.0:
			world[name[strip:]] = [float(row[i]), float(row[i + 1]), float(row[i + 2])]
	for name, (x, y, conf) in layout['image'].items():
		if name.startswith(prefix) and name in layout['world']:
			c = float(row[conf]) if conf is not None else 1.0
			image[name[strip:]] = (float(row[x]), float(row[y]), c)
	return world, image, side, line


# ---------------------------------------------------------------- cook

def cook(scriptOp):
	r, hand = rig(scriptOp)
	if r is None:
		scriptOp.clear()
		_chans.pop(scriptOp.path, None)
		return
	par = scriptOp.par
	over_image = par.Placement.eval() == 'image'
	world, image, side, line = read(par.Tracks.eval(), par.Hand.eval() if hand else None)
	top = par.Image.eval()
	# The same tracked row and settings as last cook (it cooks every frame; the tracker only
	# moves on a new model result): keep the channels as they are.
	key = (line, over_image, par.Hand.eval(), par.Mirror.eval(), par.Fov.eval(), par.Placesmoothing.eval(),
		top.width if top is not None else 0, top.height if top is not None else 0)
	if line is not None and _keys.get(scriptOp.path) == key and scriptOp.path in _chans:
		return
	_keys[scriptOp.path] = key
	out = {}
	if hand and side is None:
		side = _sides.get(scriptOp.path, hand.side)	# keep the last hand's mirroring while none is seen
	_sides[scriptOp.path] = side
	flip = bool(hand) and side[0].upper() != hand.side	# the other hand mirrors onto the model
	mirror = flip if hand else (par.Mirror.eval() and not over_image)
	positions = {}
	if world:
		kw = {'targets': hand.targets, 'blends': []} if hand else {}
		rotations = mixamo_retarget.solve(r, world, mirror=mirror, positions=positions, **kw)
		_last[scriptOp.path] = rotations
	else:	# nobody tracked: hold the last pose (rest before the first)
		rotations = _last.get(scriptOp.path) or {name: r.bones[name]['r'] for name in r.order}
	for bone, values in rotations.items():
		for axis, value in zip(('rx', 'ry', 'rz'), values):
			out[f'{bone}_{axis}'] = value

	root = _roots.get(scriptOp.path)
	if not over_image:
		root = _roots[scriptOp.path] = None
		if hand and positions:	# a hand turns about its forearm: keep the palm at the origin instead
			a, b = (positions[n] for n in hand.center)
			root = [-(a[0] + b[0]) / 2, -(a[1] + b[1]) / 2, -(a[2] + b[2]) / 2]
			_center[scriptOp.path] = root
		elif hand:
			root = _center.get(scriptOp.path)
	elif world and image:
		aspect = top.width / top.height if top is not None and top.height else 16 / 9
		if flip:	# placed in the mirrored image; the render is flipped back
			image = {name: (1.0 - u, v, c) for name, (u, v, c) in image.items()}
		fit = mixamo_retarget.place(r, positions, image, aspect, par.Fov.eval(),
			joints=hand.place_joints if hand else None)
		if fit is not None:
			k = 0.0 if root is None else min(max(par.Placesmoothing.eval(), 0.0), 0.99)
			root = _roots[scriptOp.path] = [a * k + b * (1.0 - k) for a, b in zip(root or fit, fit)]
	for name, value in zip(('root_tx', 'root_ty', 'root_tz'), root or (0.0, 0.0, 0.0)):
		out[name] = value
	out['mirror'] = 1.0 if flip else 0.0
	_write(scriptOp, out)


def _write(scriptOp, values):
	"""Set the channels: in place while they're the same ones (re-creating ~200 channels every
	cook cost more than solving), else rebuilt."""
	names = tuple(values)
	cached = _chans.get(scriptOp.path)
	if cached is not None and cached[0] == names:
		try:
			for chan, value in zip(cached[1], values.values()):
				chan[0] = value
			return
		except Exception:	# channels gone (cleared elsewhere): rebuild them
			pass
	scriptOp.clear()
	scriptOp.numSamples = 1
	chans = []
	for name, value in values.items():
		chan = scriptOp.appendChan(name)
		chan[0] = value
		chans.append(chan)
	_chans[scriptOp.path] = (names, chans)
