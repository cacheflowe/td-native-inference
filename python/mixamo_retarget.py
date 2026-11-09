"""Retarget MediaPipe world landmarks onto a Mixamo skeleton: joint positions in, one local
rotation per bone out. numpy only, so it runs and tests outside TouchDesigner.

Rig: the rest pose of a Mixamo character as imported by an FBX COMP, one Null COMP per bone:
	{bone name: {'parent': name or None, 't': [tx, ty, tz], 'r': [rx, ry, rz]}}
TD composes each bone's local transform as T * Rz * Ry * Rx (rotate order xyz, transform order
srt, column vectors). Mixamo bones point along their local +Y toward their child, the character
faces +Z with its left side at +X, y up.

Landmarks: {name: [x, y, z]} in TD axes (y up, +z toward the camera), as the trackers output
<joint>_wx/_wy/_wz: MediaPipe pose names ('left_shoulder', ...) and holistic hand names
('left_hand_index_finger_mcp', ...). A person facing the camera has their left side at +x, so
facing the viewer the character copies them like a mirror image; mirror=True swaps sides (for a
mirrored webcam image). Missing landmarks (None or absent) leave their bones at rest.

Each driven bone gets a target frame from two landmark vectors: a primary direction (the bone's
own: shoulder -> elbow) and a hint that fixes its roll (the elbow's bend, the hand's width). The
same two vectors measured on the rest skeleton give the rest frame, and the bone turns by the
difference. A hint nearly parallel to the primary (a straight arm) fades to the shortest turn.

Usage:
	rig = Rig(bones)
	rotations = solve(rig, landmarks)	# {bone: (rx, ry, rz)} local degrees, every bone
"""
import math

import numpy as np

# Below this sine between hint and primary the roll is unknown (fades to swing-only above it).
HINT_FADE = (0.05, 0.3)


# ---------------------------------------------------------------- rotations

def euler_xyz(rx, ry, rz):
	"""Rotation matrix for TD's rotate order xyz (x applied first): Rz @ Ry @ Rx, degrees."""
	x, y, z = math.radians(rx), math.radians(ry), math.radians(rz)
	cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
	return np.array([
		[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
		[sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
		[-sy, cy * sx, cy * cx],
	])


def to_euler_xyz(m):
	"""Inverse of euler_xyz: (rx, ry, rz) degrees."""
	sy = -m[2, 0]
	if abs(sy) < 0.99999:
		rx = math.atan2(m[2, 1], m[2, 2])
		ry = math.asin(sy)
		rz = math.atan2(m[1, 0], m[0, 0])
	else:	# gimbal lock: put it all in x
		rx = math.atan2(-m[1, 2], m[1, 1])
		ry = math.copysign(math.pi / 2, sy)
		rz = 0.0
	return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _quat(m):
	"""Rotation matrix -> unit quaternion (w, x, y, z)."""
	t = m[0, 0] + m[1, 1] + m[2, 2]
	if t > 0:
		s = math.sqrt(t + 1.0) * 2
		q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
	elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
		s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
		q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
	elif m[1, 1] > m[2, 2]:
		s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
		q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
	else:
		s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
		q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
	q = np.array(q)
	return q / np.linalg.norm(q)


def _matrix(q):
	w, x, y, z = q
	return np.array([
		[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
		[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
		[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
	])


def slerp(a, b, t):
	"""Spherical blend between rotation matrices a (t = 0) and b (t = 1)."""
	if t <= 0.0:
		return a
	if t >= 1.0:
		return b
	qa, qb = _quat(a), _quat(b)
	d = float(np.dot(qa, qb))
	if d < 0:
		qb, d = -qb, -d
	if d > 0.9995:
		q = qa + t * (qb - qa)
	else:
		theta = math.acos(d)
		q = (math.sin((1 - t) * theta) * qa + math.sin(t * theta) * qb) / math.sin(theta)
	return _matrix(q / np.linalg.norm(q))


# numpy's cross() and norm() cost microseconds each on 3-vectors; these are a few times faster.
def _cross(a, b):
	return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _norm(v):
	return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v):
	n = _norm(v)
	return v / n if n > 1e-9 else None


def swing(a, b):
	"""Shortest rotation turning direction a onto direction b."""
	a, b = _unit(a), _unit(b)
	axis = _cross(a, b)
	s, c = _norm(axis), float(np.dot(a, b))
	if s < 1e-9:
		if c > 0:
			return np.eye(3)
		# Opposite: half a turn about any axis perpendicular to a.
		perp = _unit(_cross(a, [1.0, 0.0, 0.0])) if abs(a[0]) < 0.9 else _unit(_cross(a, [0.0, 1.0, 0.0]))
		return 2.0 * np.outer(perp, perp) - np.eye(3)
	k = axis / s
	kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
	return np.eye(3) + s * kx + (1 - c) * kx @ kx


def _frame(primary, hint):
	"""Orthonormal frame (columns x, y, z) with y along primary and x toward hint, plus how
	usable the hint is (sine of its angle to the primary)."""
	y = _unit(primary)
	h = np.asarray(hint, dtype=float)
	hn = _norm(h)
	if y is None or hn < 1e-9:
		return None, 0.0
	x = h - y * np.dot(h, y)
	s = _norm(x) / hn
	if s < 1e-6:
		return None, 0.0
	x = x / _norm(x)
	return np.column_stack([x, y, _cross(x, y)]), s


def turn(rest_primary, rest_hint, primary, hint, base=None):
	"""World rotation taking the rest (primary, hint) frame onto the target one. With a weak
	hint (nearly parallel to its primary, on either side) it fades to `base` (the parent's turn,
	so a straight arm keeps the chest's twist) plus the shortest swing onto the primary."""
	base = np.eye(3) if base is None else base
	plain = swing(base @ rest_primary, primary) @ base
	if hint is None or rest_hint is None:
		return plain
	rest_frame, rest_s = _frame(rest_primary, rest_hint)
	frame, s = _frame(primary, hint)
	if rest_frame is None or frame is None:
		return plain
	lo, hi = HINT_FADE
	weight = min(1.0, max(0.0, (min(s, rest_s) - lo) / (hi - lo)))
	return slerp(plain, frame @ rest_frame.T, weight)


# ---------------------------------------------------------------- rig

class Rig:
	"""A Mixamo rest pose: per bone its rest local rotation and offset, and by forward
	kinematics its rest world rotation and position (relative to the FBX COMP)."""

	def __init__(self, bones):
		self.bones = bones
		hips = next(name for name in bones if name.endswith('Hips') and not bones[name]['parent'])
		self.prefix = hips[:-len('Hips')]	# 'mixamorig_' (TD turns 'mixamorig:' into this)
		self.order = []
		children = {}
		for name, bone in bones.items():
			children.setdefault(bone['parent'], []).append(name)
		pending = list(children.get(None, []))
		while pending:
			name = pending.pop(0)
			self.order.append(name)
			pending += children.get(name, [])
		self.rest_local = {n: euler_xyz(*bones[n]['r']) for n in self.order}
		self.world_rot, self.world_pos = {}, {}
		for n in self.order:
			parent = bones[n]['parent']
			t = np.asarray(bones[n]['t'], dtype=float)
			if parent is None:
				self.world_rot[n] = self.rest_local[n]
				self.world_pos[n] = t
			else:
				self.world_rot[n] = self.world_rot[parent] @ self.rest_local[n]
				self.world_pos[n] = self.world_pos[parent] + self.world_rot[parent] @ t

	def name(self, short):
		"""'LeftArm' -> 'mixamorig_LeftArm', or None if the rig has no such bone."""
		full = self.prefix + short
		return full if full in self.bones else None

	def pos(self, short):
		full = self.name(short)
		return self.world_pos[full] if full else None


# ---------------------------------------------------------------- targets
# Each driven bone: (Mixamo bone, target primary, target hint, rest primary, rest hint).
# Target vectors are (from, to) landmark pairs; a landmark is a name or ('mid', a, b). Rest
# vectors are (from, to) bone pairs, or a fixed direction in the rig's space.

_LEFT, _UP, _FORWARD = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
_SHOULDERS = ('mid', 'right_shoulder', 'left_shoulder')
_HIPS = ('mid', 'right_hip', 'left_hip')
# The same points on the rig: MediaPipe's shoulders and hips sit at the Arm and UpLeg bones.
_REST_SHOULDERS = ('mid', 'RightArm', 'LeftArm')
_REST_HIPS = ('mid', 'RightUpLeg', 'LeftUpLeg')


def _targets():
	t = [
		# Torso: hips from the hip line, the chest from the shoulder line; the spine bones blend
		# between them (_BLENDS below).
		('Hips', (_HIPS, _SHOULDERS), ('right_hip', 'left_hip'), (_REST_HIPS, _REST_SHOULDERS), ('RightUpLeg', 'LeftUpLeg')),
		('Spine2', (_HIPS, _SHOULDERS), ('right_shoulder', 'left_shoulder'), (_REST_HIPS, _REST_SHOULDERS), ('RightArm', 'LeftArm')),
		# Head: up = across the ears x nose direction, hint = the ear line.
		('Head', 'head', ('right_ear', 'left_ear'), _UP, _LEFT),
	]
	for side, mp in (('Left', 'left'), ('Right', 'right')):
		hand = f'{mp}_hand_'
		t += [
			# Upper arm: roll from the elbow's bend (forward in the T-pose rest).
			(f'{side}Arm', (f'{mp}_shoulder', f'{mp}_elbow'), (f'{mp}_elbow', f'{mp}_wrist'),
				(f'{side}Arm', f'{side}ForeArm'), _FORWARD),
			# Forearm: roll from the hand's width (pronation).
			(f'{side}ForeArm', (f'{mp}_elbow', f'{mp}_wrist'), 'hand_width:' + mp,
				(f'{side}ForeArm', f'{side}Hand'), (f'{side}HandPinky1', f'{side}HandIndex1')),
			(f'{side}Hand', 'hand_dir:' + mp, 'hand_width:' + mp,
				(f'{side}Hand', f'{side}HandMiddle1'), (f'{side}HandPinky1', f'{side}HandIndex1')),
			# Legs: roll from the foot's forward direction (heel -> toes).
			(f'{side}UpLeg', (f'{mp}_hip', f'{mp}_knee'), (f'{mp}_heel', f'{mp}_foot_index'),
				(f'{side}UpLeg', f'{side}Leg'), (f'{side}Foot', f'{side}ToeBase')),
			(f'{side}Leg', (f'{mp}_knee', f'{mp}_ankle'), (f'{mp}_heel', f'{mp}_foot_index'),
				(f'{side}Leg', f'{side}Foot'), (f'{side}Foot', f'{side}ToeBase')),
			(f'{side}Foot', (f'{mp}_ankle', f'{mp}_foot_index'), ('right_hip', 'left_hip'),
				(f'{side}Foot', f'{side}ToeBase'), ('RightUpLeg', 'LeftUpLeg')),
		]
		# Fingers (holistic hands): each segment from its joint to the next, rolled by the
		# palm's width; the thumb, which runs along the width, by the palm's length instead.
		width = (hand + 'pinky_mcp', hand + 'index_finger_mcp')
		rest_width = (f'{side}HandPinky1', f'{side}HandIndex1')
		for finger, mp_finger in (('Index', 'index_finger'), ('Middle', 'middle_finger'),
				('Ring', 'ring_finger'), ('Pinky', 'pinky')):
			joints = [f'{hand}{mp_finger}_{j}' for j in ('mcp', 'pip', 'dip', 'tip')]
			for i in range(3):
				bone = f'{side}Hand{finger}{i + 1}'
				t.append((bone, (joints[i], joints[i + 1]), width,
					(bone, f'{side}Hand{finger}{i + 2}'), rest_width))
		joints = [f'{hand}thumb_{j}' for j in ('cmc', 'mcp', 'ip', 'tip')]
		for i in range(3):
			bone = f'{side}HandThumb{i + 1}'
			t.append((bone, (joints[i], joints[i + 1]), (hand + 'wrist', hand + 'middle_finger_mcp'),
				(bone, f'{side}HandThumb{i + 2}'), (f'{side}Hand', f'{side}HandMiddle1')))
	return t


TARGETS = _targets()
# Bones between two driven ones: (bone, from, to, fraction) of the way from one turn to the other.
_BLENDS = [('Spine', 'Hips', 'Spine2', 1 / 3), ('Spine1', 'Hips', 'Spine2', 2 / 3), ('Neck', 'Spine2', 'Head', 0.5)]


def _mirror(landmarks):
	"""Swap left and right and flip x (for a mirrored image)."""
	out = {}
	for name, p in landmarks.items():
		if name.startswith('left_'):
			name = 'right_' + name[5:]
		elif name.startswith('right_'):
			name = 'left_' + name[6:]
		out[name] = None if p is None else [-p[0], p[1], p[2]]
	return out


class _Points:
	def __init__(self, landmarks):
		self.lm = landmarks

	def get(self, key):
		if isinstance(key, tuple):	# ('mid', a, b)
			a, b = self.get(key[1]), self.get(key[2])
			return None if a is None or b is None else (a + b) / 2
		p = self.lm.get(key)
		return None if p is None else np.asarray(p, dtype=float)

	def first(self, *keys):
		for key in keys:
			p = self.get(key)
			if p is not None:
				return p
		return None

	def vector(self, spec):
		"""A target vector: (from, to), or a named special."""
		if isinstance(spec, str):
			kind, _, side = spec.partition(':')
			if kind == 'head':
				ears = self.vector(('right_ear', 'left_ear'))
				nose, mid = self.get('nose'), self.get(('mid', 'right_ear', 'left_ear'))
				return None if ears is None or nose is None or mid is None else _cross(nose - mid, ears)
			hand = f'{side}_hand_'
			if kind == 'hand_dir':	# wrist -> middle knuckle (holistic), else -> between index and pinky (pose)
				wrist = self.first(hand + 'wrist', f'{side}_wrist')
				tip = self.first(hand + 'middle_finger_mcp', ('mid', f'{side}_index', f'{side}_pinky'))
				return None if wrist is None or tip is None else tip - wrist
			if kind == 'hand_width':	# pinky -> index knuckle
				a = self.first(hand + 'pinky_mcp', f'{side}_pinky')
				b = self.first(hand + 'index_finger_mcp', f'{side}_index')
				return None if a is None or b is None else b - a
			raise ValueError(spec)
		a, b = self.get(spec[0]), self.get(spec[1])
		return None if a is None or b is None else b - a


def _rest_point(rig, key):
	if isinstance(key, tuple):	# ('mid', a, b)
		a, b = rig.pos(key[1]), rig.pos(key[2])
		return None if a is None or b is None else (a + b) / 2
	return rig.pos(key)


def _rest_vector(rig, spec):
	if isinstance(spec[0], float):
		return np.asarray(spec)
	a, b = _rest_point(rig, spec[0]), _rest_point(rig, spec[1])
	return None if a is None or b is None else b - a


def _parent_turn(rig, bone, turns):
	"""The turn of the nearest driven ancestor (undriven bones ride along with it)."""
	parent = rig.bones[bone]['parent']
	while parent:
		short = parent[len(rig.prefix):]
		if short in turns:
			return turns[short]
		parent = rig.bones[parent]['parent']
	return None


def solve(rig, landmarks, mirror=False):
	"""{bone: (rx, ry, rz)} local Euler degrees for every bone in the rig: driven bones turned
	toward the landmarks, the rest (and bones whose landmarks are missing) at their rest pose."""
	points = _Points(_mirror(landmarks) if mirror else landmarks)
	turns = {}
	for short, primary, hint, rest_primary, rest_hint in TARGETS:
		bone = rig.name(short)
		if bone is None:
			continue
		p, h = points.vector(primary), points.vector(hint)
		rp, rh = _rest_vector(rig, rest_primary), _rest_vector(rig, rest_hint)
		if p is None or rp is None or _unit(p) is None:
			continue
		turns[short] = turn(rp, rh, p, h, _parent_turn(rig, bone, turns))
	for short, a, b, f in _BLENDS:
		if rig.name(short) and a in turns and b in turns:
			turns[short] = slerp(turns[a], turns[b], f)

	world, out = {}, {}
	for bone in rig.order:
		parent = rig.bones[bone]['parent']
		parent_world = world[parent] if parent else np.eye(3)
		short = bone[len(rig.prefix):]
		if short in turns:
			world[bone] = turns[short] @ rig.world_rot[bone]
			local = parent_world.T @ world[bone]
		else:
			local = rig.rest_local[bone]
			world[bone] = parent_world @ local
		out[bone] = to_euler_xyz(local)
	return out
