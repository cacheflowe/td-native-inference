"""Retarget MediaPipe hand landmarks onto a rigged hand: the same solver as mixamo_retarget
(solve(), place()), with a hand's bone map instead of the Mixamo body's.

Landmarks: {name: [x, y, z]} with MediaPipe's hand names ('wrist', 'thumb_cmc',
'index_finger_mcp', ...), in TD axes. Holistic's '<side>_hand_' columns or a MediaPipe Hands
track, prefix dropped (see puppet_solver).

The bone map fits Blender-style hand rigs like handRig_02.fbx: one hand, side suffix '_R' or
'_L'; a forearm root 'pulse', the wrist 'hand', one '<finger>_base' bone per metacarpal and three
phalanges '<finger>_01'..'_03' (+ '_03_end'); the thumb's metacarpal is 'thumb_base'. The rig's
control bones ('*_Ctrl*', '*_tip') don't deform the mesh and stay at rest. Each bone turns from
its own rest direction (the rig's own joint positions) to the matching landmark segment. The root
carries the whole hand, rolled by the palm's width; the metacarpals ride rigidly with it. Each
phalanx is a hinge with limits (FINGER_LIMITS) and no roll of its own.

Usage:
	rig = mixamo_retarget.Rig(bones)
	spec = hand_retarget.spec(rig)	# None if the rig isn't a hand
	rotations = mixamo_retarget.solve(rig, landmarks, mirror=..., targets=spec.targets, blends=[])
"""
from dataclasses import dataclass, field

FINGERS = (('index', 'index_finger'), ('middle', 'middle_finger'), ('ring', 'ring_finger'), ('pinky', 'pinky'))
# Per phalanx (knuckle, middle, end joint): bend (curl, + toward the palm) and sideways spread
# limits in degrees, so tracking noise in depth can't bend a finger backward or sideways.
FINGER_LIMITS = [
	{'bend': (-30, 95), 'spread': (-25, 25)},
	{'bend': (-5, 115), 'spread': (0, 0)},
	{'bend': (-5, 90), 'spread': (0, 0)},
]


@dataclass
class HandSpec:
	side: str	# 'R' or 'L': which hand the model is
	targets: list = field(default_factory=list)
	place_joints: list = field(default_factory=list)
	center: tuple = ()	# bones whose midpoint (the palm) Center placement keeps at the origin


def spec(rig):
	"""The bone map for a hand rig (bone names as above), or None."""
	for side in ('R', 'L'):
		s = f'_{side}'
		if f'hand{s}' in rig.bones and f'pulse{s}' in rig.bones:
			return HandSpec(side, _targets(s), _place_joints(s), (f'hand{s}', f'middle_01{s}'))
	return None


def _targets(s):
	width = ('pinky_mcp', 'index_finger_mcp')
	rest_width = (f'pinky_01{s}', f'index_01{s}')
	# Fingers bend about the knuckle line; across it from pinky to index on a right hand puts
	# cross(axis, finger) on the palm side, so a positive bend curls the finger in.
	axis = rest_width if s == '_R' else rest_width[::-1]
	t = [
		# The whole hand from the palm: wrist -> middle knuckle, rolled by the knuckle line. The
		# metacarpals ('<finger>_base') ride along: a palm is nearly rigid, and driving each from
		# its own (noisy) wrist -> knuckle line creased the skin.
		(f'pulse{s}', ('wrist', 'middle_finger_mcp'), width, (f'hand{s}', f'middle_01{s}'), rest_width),
	]
	for finger, mp in FINGERS:
		joints = [f'{mp}_{j}' for j in ('mcp', 'pip', 'dip', 'tip')]
		bones = [f'{finger}_{b}{s}' for b in ('01', '02', '03')]
		rest = bones + [f'{finger}_03{s}_end']
		for i, bone in enumerate(bones):
			t.append((bone, (joints[i], joints[i + 1]), width, (rest[i], rest[i + 1]), rest_width,
				{'axis': axis, **FINGER_LIMITS[i]}))
	# Thumb: metacarpal, proximal, then the distal phalanx in two bones (both ip -> tip); rolled
	# by the palm's length, since the thumb runs along its width.
	segments = [('thumb_cmc', 'thumb_mcp'), ('thumb_mcp', 'thumb_ip'), ('thumb_ip', 'thumb_tip'), ('thumb_ip', 'thumb_tip')]
	length = ('wrist', 'middle_finger_mcp')
	rest_length = (f'hand{s}', f'middle_01{s}')
	bones = [f'thumb_base{s}', f'thumb_01{s}', f'thumb_02{s}', f'thumb_03{s}']
	rest = [(f'thumb_base{s}', f'thumb_01{s}'), (f'thumb_01{s}', f'thumb_02{s}'),
		(f'thumb_02{s}', f'thumb_03{s}_end'), (f'thumb_02{s}', f'thumb_03{s}_end')]
	for i, bone in enumerate(bones):
		t.append((bone, segments[i], length, rest[i], rest_length))
	return t


def _place_joints(s):
	"""(landmark, bone) pairs sitting on the same point, for place()."""
	joints = [('wrist', f'hand{s}'), ('thumb_cmc', f'thumb_base{s}'), ('thumb_mcp', f'thumb_01{s}'),
		('thumb_ip', f'thumb_02{s}'), ('thumb_tip', f'thumb_03{s}_end')]
	for finger, mp in FINGERS:
		joints += [(f'{mp}_mcp', f'{finger}_01{s}'), (f'{mp}_pip', f'{finger}_02{s}'),
			(f'{mp}_dip', f'{finger}_03{s}'), (f'{mp}_tip', f'{finger}_03{s}_end')]
	return joints
