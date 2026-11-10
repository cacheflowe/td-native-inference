"""Offline tests for mixamo_retarget (no TD needed).

	python -m unittest discover -s tests -p test_retarget.py

A synthetic Mixamo-named T-pose rig stands in for a real character. When build/ybot_rest.json
exists (dumped from the dev project's /project1/PuppetLab FBX COMP; Mixamo files aren't
committed), the same checks also run on Y Bot and its rest pose is compared with TD's matrices.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import mixamo_retarget as mr  # noqa: E402

P = 'mixamorig_'
YBOT = ROOT / 'build' / 'ybot_rest.json'


def _side(sign, s):
	"""One side's arm and leg: (bone, parent, position) with x mirrored by sign."""
	x = lambda v: sign * v	# noqa: E731
	bones = [
		(f'{s}Shoulder', 'Spine2', (x(6), 145, 0)), (f'{s}Arm', f'{s}Shoulder', (x(18), 145, 0)),
		(f'{s}ForeArm', f'{s}Arm', (x(45), 145, 0)), (f'{s}Hand', f'{s}ForeArm', (x(72), 145, 0)),
		(f'{s}UpLeg', 'Hips', (x(9), 93, 0)), (f'{s}Leg', f'{s}UpLeg', (x(9), 53, 1)),
		(f'{s}Foot', f'{s}Leg', (x(9), 10, -1)), (f'{s}ToeBase', f'{s}Foot', (x(9), 2, 12)),
		(f'{s}Toe_End', f'{s}ToeBase', (x(9), 2, 20)),
	]
	for i in range(1, 5):
		bones.append((f'{s}HandThumb{i}', f'{s}HandThumb{i - 1}' if i > 1 else f'{s}Hand', (x(73 + 3 * i), 145 - 0.5 * i, 2 + 2 * i)))
	for finger, z in (('Index', 2.5), ('Middle', 0.5), ('Ring', -1.5), ('Pinky', -3.5)):
		for i in range(1, 5):
			bones.append((f'{s}Hand{finger}{i}', f'{s}Hand{finger}{i - 1}' if i > 1 else f'{s}Hand',
				(x(80 + 4 * i - abs(z)), 145 - 0.3 * i, z)))
	return bones


SPEC = [
	('Hips', None, (0, 100, 0)), ('Spine', 'Hips', (0, 110, -1)), ('Spine1', 'Spine', (0, 122, -1)),
	('Spine2', 'Spine1', (0, 135, 0)), ('Neck', 'Spine2', (0, 150, 1)), ('Head', 'Neck', (0, 160, 3)),
	('HeadTop_End', 'Head', (0, 178, 5)),
] + _side(1, 'Left') + _side(-1, 'Right')


def synthetic_rig():
	"""Bones as {'parent', 't', 'r'}: each bone's +Y aims at its first child (Mixamo-style), with
	a different roll per bone so nothing relies on rest rotations being simple."""
	pos = {n: np.array(p, dtype=float) for n, _, p in SPEC}
	first_child = {}
	for n, parent, _ in SPEC:
		if parent:
			first_child.setdefault(parent, n)
	first_child.update(Hips='Spine', Spine2='Neck', LeftHand='LeftHandMiddle1', RightHand='RightHandMiddle1')
	world, bones = {}, {}
	for i, (n, parent, _) in enumerate(SPEC):
		child = first_child.get(n)
		if child:
			y = mr._unit(pos[child] - pos[n])
			roll = mr.euler_xyz(*(37 * i % 90 - 45, 0, 0)) @ np.array([1.0, 0.0, 0.0])
			if abs(np.dot(roll, y)) > 0.9:
				roll = np.array([0.0, 0.0, 1.0])
			frame, _ = mr._frame(y, roll)
			world[n] = frame
		else:
			world[n] = world[parent]
		pw = world[parent] if parent else np.eye(3)
		pp = pos[parent] if parent else np.zeros(3)
		bones[P + n] = {'parent': P + parent if parent else None,
			't': list(pw.T @ (pos[n] - pp)), 'r': list(mr.to_euler_xyz(pw.T @ world[n]))}
	return bones


def landmarks_from(rig):
	"""MediaPipe-named landmarks at the rig's own rest joints: solving them must give the rest pose."""
	g = rig.pos
	lm = {'nose': g('Head') + [0, 0, 10]}
	for side, mp in (('Left', 'left'), ('Right', 'right')):
		sign = 1 if side == 'Left' else -1
		lm[f'{mp}_ear'] = g('Head') + [sign * 7, 0, 0]
		for name, bone in (('shoulder', 'Arm'), ('elbow', 'ForeArm'), ('wrist', 'Hand'), ('hip', 'UpLeg'),
				('knee', 'Leg'), ('ankle', 'Foot'), ('heel', 'Foot'), ('foot_index', 'ToeBase'),
				('index', 'HandIndex1'), ('pinky', 'HandPinky1')):
			lm[f'{mp}_{name}'] = g(side + bone)
		hand = f'{mp}_hand_'
		lm[hand + 'wrist'] = g(side + 'Hand')
		for i, j in enumerate(('cmc', 'mcp', 'ip', 'tip')):
			lm[f'{hand}thumb_{j}'] = g(f'{side}HandThumb{i + 1}')
		for finger, mp_finger in (('Index', 'index_finger'), ('Middle', 'middle_finger'), ('Ring', 'ring_finger'), ('Pinky', 'pinky')):
			for i, j in enumerate(('mcp', 'pip', 'dip', 'tip')):
				lm[f'{hand}{mp_finger}_{j}'] = g(f'{side}Hand{finger}{i + 1}')
	return {k: list(v) for k, v in lm.items()}


def world_rotations(rig, eulers):
	"""Forward kinematics of solve()'s output."""
	world = {}
	for bone in rig.order:
		parent = rig.bones[bone]['parent']
		local = mr.euler_xyz(*eulers[bone])
		world[bone] = (world[parent] if parent else np.eye(3)) @ local
	return world


def turned(lm, rotation, names, pivot):
	"""lm with the named points turned by rotation about pivot."""
	out = dict(lm)
	for n in names:
		out[n] = list(pivot + rotation @ (np.asarray(lm[n]) - pivot))
	return out


class RetargetTests(unittest.TestCase):
	def rigs(self):
		rigs = [('synthetic', mr.Rig(synthetic_rig()))]
		if YBOT.is_file():
			rigs.append(('ybot', mr.Rig(json.loads(YBOT.read_text())['bones'])))
		return rigs

	def assertRotationsClose(self, a, b, places=4, msg=None):
		np.testing.assert_allclose(a, b, atol=10 ** -places, err_msg=msg or '')

	def test_euler_round_trip(self):
		for r in ((10, 20, 30), (-170, 45, 89), (0, 90, 0), (5, -90, 12)):
			m = mr.euler_xyz(*r)
			self.assertRotationsClose(mr.euler_xyz(*mr.to_euler_xyz(m)), m)

	def test_rest_landmarks_give_the_rest_pose(self):
		for label, rig in self.rigs():
			out = solve_world = world_rotations(rig, mr.solve(rig, landmarks_from(rig)))
			for bone in rig.order:
				self.assertRotationsClose(out[bone], rig.world_rot[bone], 3, f'{label} {bone}')
			del solve_world

	def test_whole_body_turn_only_turns_the_hips(self):
		turn = mr.euler_xyz(0, 35, 0) @ mr.euler_xyz(10, 0, 0)
		for label, rig in self.rigs():
			lm = landmarks_from(rig)
			lm = turned(lm, turn, lm.keys(), np.zeros(3))
			out = mr.solve(rig, lm)
			hips = rig.name('Hips')
			self.assertRotationsClose(mr.euler_xyz(*out[hips]), turn @ rig.rest_local[hips], 3, label)
			for bone in rig.order:
				if bone != hips:
					self.assertRotationsClose(mr.euler_xyz(*out[bone]), rig.rest_local[bone], 3, f'{label} {bone}')

	def test_raised_left_arm(self):
		up = mr.euler_xyz(0, 0, 80)	# the T-pose arm (+x) turned to nearly straight up
		for label, rig in self.rigs():
			lm = landmarks_from(rig)
			arm = [k for k in lm if k.startswith(('left_elbow', 'left_wrist', 'left_index', 'left_pinky', 'left_hand_'))]
			lm = turned(lm, up, arm, np.asarray(lm['left_shoulder']))
			world = world_rotations(rig, mr.solve(rig, lm))
			bone = rig.name('LeftArm')
			rest_dir = rig.pos('LeftForeArm') - rig.pos('LeftArm')
			# The upper arm now points where the turned elbow is.
			want = mr._unit(up @ rest_dir)
			got = world[bone] @ np.linalg.solve(rig.world_rot[bone], mr._unit(rest_dir))
			self.assertRotationsClose(got, want, 3, label)
			# The right arm didn't move.
			right = rig.name('RightArm')
			self.assertRotationsClose(world[right], rig.world_rot[right], 3, label)

	def test_bent_elbow(self):
		bend = mr.euler_xyz(0, -70, 0)	# forearm swung 70 degrees forward about the elbow
		for label, rig in self.rigs():
			lm = landmarks_from(rig)
			forearm = [k for k in lm if k.startswith(('left_wrist', 'left_index', 'left_pinky', 'left_hand_'))]
			lm = turned(lm, bend, forearm, np.asarray(lm['left_elbow']))
			world = world_rotations(rig, mr.solve(rig, lm))
			y = np.array([0.0, 1.0, 0.0])
			upper = world[rig.name('LeftArm')] @ y
			lower = world[rig.name('LeftForeArm')] @ y
			self.assertAlmostEqual(math.degrees(math.acos(np.clip(np.dot(upper, lower), -1, 1))), 70, 1, label)

	def test_missing_hand_stays_at_rest(self):
		for label, rig in self.rigs():
			lm = {k: v for k, v in landmarks_from(rig).items() if not k.startswith('left_hand_')}
			lm = turned(lm, mr.euler_xyz(0, 0, 40), [k for k in lm if k.startswith('left_')], np.zeros(3))
			out = mr.solve(rig, lm)
			index = rig.name('LeftHandIndex2')
			self.assertRotationsClose(mr.euler_xyz(*out[index]), rig.rest_local[index], 4, label)

	def test_mirror_swaps_sides(self):
		for label, rig in self.rigs():
			lm = landmarks_from(rig)
			arm = [k for k in lm if k.startswith(('left_elbow', 'left_wrist', 'left_index', 'left_pinky', 'left_hand_'))]
			lm = turned(lm, mr.euler_xyz(0, 0, 80), arm, np.asarray(lm['left_shoulder']))
			world = world_rotations(rig, mr.solve(rig, lm, mirror=True))
			left, right = rig.name('LeftArm'), rig.name('RightArm')
			self.assertRotationsClose(world[left], rig.world_rot[left], 3, label)
			self.assertFalse(np.allclose(world[right], rig.world_rot[right], atol=1e-2), label)

	def test_posed_positions_follow_the_pose(self):
		for label, rig in self.rigs():
			lm = landmarks_from(rig)
			positions = {}
			mr.solve(rig, lm, positions=positions)
			for bone in rig.order:	# rest landmarks: the rest skeleton
				np.testing.assert_allclose(positions[bone], rig.world_pos[bone], atol=1e-3, err_msg=f'{label} {bone}')
			arm = [k for k in lm if k.startswith(('left_elbow', 'left_wrist', 'left_index', 'left_pinky', 'left_hand_'))]
			lm = turned(lm, mr.euler_xyz(0, 0, 80), arm, np.asarray(lm['left_shoulder']))
			mr.solve(rig, lm, positions=positions)
			hand, elbow = positions[rig.name('LeftHand')], positions[rig.name('LeftForeArm')]
			self.assertGreater(hand[1] - elbow[1], 10, label)	# the raised forearm points up

	def test_place_recovers_the_root_translation(self):
		aspect, fov = 16 / 9, 50.0
		tan_h = math.tan(math.radians(fov) / 2)
		for label, rig in self.rigs():
			positions = {}
			mr.solve(rig, landmarks_from(rig), positions=positions)
			root = np.array([40.0, -120.0, -600.0])
			points = {}
			for name, short in mr.PLACE_JOINTS:
				p = positions[rig.name(short)] + root
				u = 0.5 + 0.5 * p[0] / (-p[2] * tan_h)
				v = 0.5 + 0.5 * p[1] / (-p[2] * tan_h / aspect)
				points[name] = (u, v, 1.0)
			np.testing.assert_allclose(mr.place(rig, positions, points, aspect, fov), root, atol=1e-6, err_msg=label)
			# An unseen joint is left out; too few joints give no fit.
			points['left_wrist'] = (0.0, 0.0, 0.0)
			np.testing.assert_allclose(mr.place(rig, positions, points, aspect, fov), root, atol=1e-6, err_msg=label)
			self.assertIsNone(mr.place(rig, positions, {'left_hip': (0.5, 0.5, 1.0)}, aspect, fov))

	@unittest.skipUnless(YBOT.is_file(), 'no build/ybot_rest.json')
	def test_ybot_rest_matches_td(self):
		data = json.loads(YBOT.read_text())
		rig = mr.Rig(data['bones'])
		for bone, b in data['bones'].items():
			m = np.array(b['world'])
			self.assertRotationsClose(rig.world_rot[bone], m[:3, :3], 4, bone)
			self.assertRotationsClose(rig.world_pos[bone], m[:3, 3], 2, bone)


if __name__ == '__main__':
	unittest.main()
