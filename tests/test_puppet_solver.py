"""Offline tests for puppet_solver.read() (no TD needed).

	python -m unittest discover -s tests -p test_puppet_solver.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import puppet_solver  # noqa: E402


class Tracks:
	"""Just enough of a tracks Table DAT: numRows/numCols, [r, c] and row(0) cells with .val, text."""

	class Cell:
		def __init__(self, val):
			self.val = str(val)

		def __float__(self):
			return float(self.val)

	def __init__(self, rows):
		self.rows = [[self.Cell(v) for v in row] for row in rows]
		self.numRows = len(rows)
		self.numCols = len(rows[0])
		self.text = '\r\n'.join('\t'.join(c.val for c in row) for row in self.rows)

	def row(self, i):
		return self.rows[i]

	def __getitem__(self, rc):
		return self.rows[rc[0]][rc[1]]


def holistic_tracks(left_hand_seen=True):
	"""One person: a pose wrist and both holistic hand wrists, image and world columns."""
	header, row = ['track_id', 'lost_frames', 'total_frames'], [1, 0, 10]
	for name, conf in (('pose_left_wrist', 1.0), ('left_hand_wrist', 1.0 if left_hand_seen else 0.0),
			('right_hand_wrist', 1.0)):
		header += [f'{name}_x', f'{name}_y', f'{name}_conf', f'{name}_z']
		row += [0.4, 0.5, conf, 0.0]
	for name, x in (('pose_left_wrist', 0.1), ('left_hand_wrist', 0.2), ('right_hand_wrist', 0.3)):
		header += [f'{name}_wx', f'{name}_wy', f'{name}_wz']
		row += [x, 0.2, 0.3]
	return Tracks([header, row])


class ReadTests(unittest.TestCase):
	def setUp(self):
		puppet_solver._columns.clear()

	def test_body_gets_pose_and_hand_landmarks(self):
		world, image, side, line = puppet_solver.read(holistic_tracks(), None)
		# The pose prefix is dropped; Holistic's hands keep their names (the fingers use them).
		self.assertEqual(set(world), {'left_wrist', 'left_hand_wrist', 'right_hand_wrist'})
		self.assertEqual(world['left_hand_wrist'], [0.2, 0.2, 0.3])
		self.assertIn('left_wrist', image)
		self.assertIsNone(side)

	def test_body_leaves_out_an_unseen_hand(self):
		world, *_ = puppet_solver.read(holistic_tracks(left_hand_seen=False), None)
		self.assertNotIn('left_hand_wrist', world)
		self.assertIn('right_hand_wrist', world)

	def test_hand_gets_only_its_side(self):
		world, image, side, line = puppet_solver.read(holistic_tracks(), 'right')
		self.assertEqual(set(world), {'wrist'})
		self.assertEqual(world['wrist'], [0.3, 0.2, 0.3])
		self.assertEqual(side, 'right')


class HoldFingersTests(unittest.TestCase):
	class Rig:
		prefix = 'mixamorig_'

	def test_unseen_hand_keeps_its_last_finger_pose(self):
		rest, last = (0.0, 0.0, 0.0), (10.0, 20.0, 30.0)
		bones = ['mixamorig_LeftHandIndex1', 'mixamorig_LeftHand', 'mixamorig_RightHandIndex1']
		rotations = {b: rest for b in bones}
		world = {'right_hand_wrist': [0, 0, 0]}	# only the right hand seen
		puppet_solver._hold_fingers(self.Rig(), world, rotations, {b: last for b in bones}, mirror=False)
		self.assertEqual(rotations['mixamorig_LeftHandIndex1'], last)	# held
		self.assertEqual(rotations['mixamorig_LeftHand'], rest)		# the hand itself follows the pose
		self.assertEqual(rotations['mixamorig_RightHandIndex1'], rest)	# seen: solved

	def test_mirror_holds_the_other_side(self):
		rest, last = (0.0, 0.0, 0.0), (10.0, 20.0, 30.0)
		rotations = {'mixamorig_RightHandIndex1': rest}
		puppet_solver._hold_fingers(self.Rig(), {'right_hand_wrist': [0, 0, 0]}, rotations,
			{'mixamorig_RightHandIndex1': last}, mirror=True)
		self.assertEqual(rotations['mixamorig_RightHandIndex1'], last)	# mirrored: the right bones use the left hand


if __name__ == '__main__':
	unittest.main()
