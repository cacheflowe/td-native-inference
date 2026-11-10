"""Offline tests for hand_retarget (no TD needed).

	python -m unittest discover -s tests -p test_hand_retarget.py

Runs on build/hand_rest.json, the rest pose dumped from the dev project's hand FBX COMP
(handRig_02.fbx; models aren't committed), and skips without it.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import hand_retarget as hr  # noqa: E402
import mixamo_retarget as mr  # noqa: E402

HAND = ROOT / 'build' / 'hand_rest.json'


def rest_landmarks(rig, spec):
	"""MediaPipe hand landmarks at the rig's own rest joints."""
	return {name: list(rig.world_pos[bone]) for name, bone in spec.place_joints}


def turned(lm, rotation, names, pivot):
	out = dict(lm)
	for n in names:
		out[n] = list(pivot + rotation @ (np.asarray(lm[n]) - pivot))
	return out


@unittest.skipUnless(HAND.is_file(), 'no build/hand_rest.json')
class HandRetargetTests(unittest.TestCase):
	def setUp(self):
		data = json.loads(HAND.read_text())['bones']
		self.rig = mr.Rig(data)
		self.spec = hr.spec(self.rig)
		self.data = data

	def solve(self, lm, **kw):
		positions = {}
		out = mr.solve(self.rig, lm, positions=positions, targets=self.spec.targets, blends=[], **kw)
		return out, positions

	def test_fk_matches_td(self):
		for bone in self.rig.order:
			m = np.array(self.data[bone]['world'])
			np.testing.assert_allclose(self.rig.world_pos[bone], m[:3, 3], atol=1e-3, err_msg=bone)

	def test_spec_found(self):
		self.assertEqual(self.spec.side, 'R')
		for bone, *_ in self.spec.targets:
			self.assertIn(bone, self.rig.bones)

	def test_rest_landmarks_give_the_rest_pose(self):
		out, positions = self.solve(rest_landmarks(self.rig, self.spec))
		for bone in self.rig.order:
			np.testing.assert_allclose(mr.euler_xyz(*out[bone]), self.rig.rest_local[bone], atol=1e-3, err_msg=bone)
			np.testing.assert_allclose(positions[bone], self.rig.world_pos[bone], atol=1e-2, err_msg=bone)

	def test_whole_hand_turns(self):
		lm = rest_landmarks(self.rig, self.spec)
		wrist = np.asarray(lm['wrist'])
		rotation = mr.euler_xyz(30, -50, 70)
		lm = turned(lm, rotation, list(lm), wrist)
		out, positions = self.solve(lm)
		# Rotations only: the hand turns about its root, so compare the shape from the wrist.
		for name, bone in self.spec.place_joints:
			np.testing.assert_allclose(positions[bone] - positions['hand_R'], np.asarray(lm[name]) - wrist, atol=0.5, err_msg=bone)

	def finger_turned(self, joint, axis, degrees):
		"""Rest landmarks with the index finger beyond `joint` turned about `axis`."""
		lm = rest_landmarks(self.rig, self.spec)
		names = ['index_finger_pip', 'index_finger_dip', 'index_finger_tip']
		names = names[names.index(joint) + 1:]
		k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
		a = math.radians(degrees)
		rotation = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k
		return turned(lm, rotation, names, np.asarray(lm[joint]))

	def knuckle_line(self):
		lm = rest_landmarks(self.rig, self.spec)
		return mr._unit(np.asarray(lm['index_finger_mcp']) - np.asarray(lm['pinky_mcp']))

	def test_bent_finger_follows(self):
		# Curl the index finger 80 degrees (toward the palm) at its middle joint, about the
		# joint's hinge axis (the knuckle line made square to the finger).
		lm = rest_landmarks(self.rig, self.spec)
		r = mr._unit(np.asarray(lm['index_finger_dip']) - np.asarray(lm['index_finger_pip']))
		hinge = np.cross(r, mr._unit(np.cross(self.knuckle_line(), r)))
		lm = self.finger_turned('index_finger_pip', hinge, 80)
		_, positions = self.solve(lm)
		pip = np.asarray(lm['index_finger_pip'])
		np.testing.assert_allclose(positions['index_03_R_end'] - positions['index_02_R'], np.asarray(lm['index_finger_tip']) - pip, atol=0.5)
		np.testing.assert_allclose(positions['middle_03_R_end'], self.rig.world_pos['middle_03_R_end'], atol=1e-2)

	def test_finger_cannot_bend_backward(self):
		_, positions = self.solve(self.finger_turned('index_finger_pip', self.knuckle_line(), -60))
		u = positions['index_03_R'] - positions['index_02_R']
		v = self.rig.world_pos['index_03_R'] - self.rig.world_pos['index_02_R']
		angle = math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
		self.assertAlmostEqual(angle, 5.0, delta=0.5)	# clamped at 5 degrees back

	def test_finger_cannot_bend_sideways_at_its_middle_joint(self):
		lm = rest_landmarks(self.rig, self.spec)
		finger = mr._unit(np.asarray(lm['index_finger_pip']) - np.asarray(lm['index_finger_mcp']))
		palm_normal = mr._unit(np.cross(self.knuckle_line(), finger))
		_, positions = self.solve(self.finger_turned('index_finger_pip', palm_normal, 40))
		for bone in ('index_03_R', 'index_03_R_end'):
			np.testing.assert_allclose(positions[bone], self.rig.world_pos[bone], atol=0.5, err_msg=bone)

	def test_mirrored_left_hand_lands_on_the_image(self):
		"""A left hand drives the right-hand model through mirror=True, shown with sx = -1."""
		lm = rest_landmarks(self.rig, self.spec)
		left = {k: [-v[0], v[1], v[2]] for k, v in lm.items()}	# the same hand, mirrored
		_, positions = self.solve(left, mirror=True)
		for name, bone in self.spec.place_joints:
			shown = positions[bone] * [-1, 1, 1]
			np.testing.assert_allclose(shown, left[name], atol=1e-2, err_msg=bone)

	def test_place_recovers_the_root_translation(self):
		aspect, fov = 4 / 3, 40.0
		tan_h = math.tan(math.radians(fov) / 2)
		_, positions = self.solve(rest_landmarks(self.rig, self.spec))
		for flip in (False, True):
			root = np.array([-30.0, 20.0, -900.0])
			points = {}
			for name, bone in self.spec.place_joints:
				p = positions[bone] * ([-1, 1, 1] if flip else 1) + root
				points[name] = (0.5 + 0.5 * p[0] / (-p[2] * tan_h), 0.5 + 0.5 * p[1] / (-p[2] * tan_h / aspect), 1.0)
			t = mr.place(self.rig, positions, points, aspect, fov, joints=self.spec.place_joints, flip_x=flip)
			np.testing.assert_allclose(t, root, atol=1e-6)


if __name__ == '__main__':
	unittest.main()
