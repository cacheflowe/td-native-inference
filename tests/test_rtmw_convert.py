"""rtmw_convert: RTMW3D's 133 keypoints -> the Holistic-shaped table, on a saved decode of the warrior photo
(tests/data/rtmw3d_warrior.json: docs/images/sources/body.jpg through RTMW3D-X, units as OnnxTOP's Pose Crops
Info DAT writes them). Run: python tests/test_rtmw_convert.py"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'python'))
import mediapipe_skeletons  # noqa: E402
import rtmw_convert  # noqa: E402


class FakeTable:
	def __init__(self, rows=()):
		self.rows = [list(r) for r in rows]

	def clear(self):
		self.rows = []

	def appendRow(self, row):
		self.rows.append([str(c) for c in row])

	@property
	def numRows(self):
		return len(self.rows)

	@property
	def text(self):
		return '\n'.join('\t'.join(str(c) for c in r) for r in self.rows)


class Image:
	def __init__(self, width, height):
		self.width, self.height = width, height


def info_table(fixture):
	header = ['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2'] + [f'kp{i}_{s}' for i in range(133) for s in 'uvc'] \
		+ [f'kp{i}_z' for i in range(133)] + ['crop_w', 'crop_h']
	row = ['0', '0', '0.9', '0.2', '0.1', '0.8', '0.9']
	for i in range(133):
		row += [fixture['u'][i], fixture['v'][i], fixture['c'][i]]
	row += fixture['z'] + [fixture['crop_w'], fixture['crop_h']]
	return FakeTable([header, row])


class ConvertTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.fixture = json.loads((ROOT / 'tests' / 'data' / 'rtmw3d_warrior.json').read_text())
		out = FakeTable()
		rtmw_convert.convert(out, info_table(cls.fixture), Image(*cls.fixture['image']))
		cls.header, cls.row = out.rows[0], out.rows[1]
		cls.cols = {name: i for i, name in enumerate(cls.header)}

	def value(self, name):
		return float(self.row[self.cols[name]])

	def world(self, name):
		i = mediapipe_skeletons.RTMW_BODY.names.index(name)
		return [self.value(f'w{i}_{a}') for a in 'xyz']

	def test_shape_matches_the_skeleton(self):
		self.assertEqual(rtmw_convert.COUNT, mediapipe_skeletons.RTMW_BODY.count)
		self.assertEqual(len(self.header), len(self.row))
		self.assertEqual(self.header[8:11], ['kp0_u', 'kp0_v', 'kp0_c'])

	def test_hips_are_the_origin(self):
		l, r = self.world('pose_left_hip'), self.world('pose_right_hip')
		for axis in range(3):
			self.assertAlmostEqual((l[axis] + r[axis]) / 2, 0.0, places=3)

	def test_bones_are_human_sized(self):
		def dist(a, b):
			return sum((x - y) ** 2 for x, y in zip(self.world(a), self.world(b))) ** 0.5
		self.assertTrue(0.2 < dist('pose_left_shoulder', 'pose_left_elbow') < 0.4)
		self.assertTrue(0.3 < dist('pose_left_hip', 'pose_left_knee') < 0.55)
		self.assertTrue(0.25 < dist('pose_left_shoulder', 'pose_right_shoulder') < 0.55)

	def test_axes_are_mediapipe_raw(self):
		# y down: the head is above (negative y) and the knees below (positive y) the hips
		self.assertLess(self.world('pose_nose')[1], -0.3)
		self.assertGreater(self.world('pose_left_knee')[1], 0.1)
		# x right in the image: the subject's left wrist is on the image's right
		self.assertGreater(self.world('pose_left_wrist')[0], self.world('pose_right_wrist')[0] + 0.5)

	def test_unseen_hand_is_absent(self):
		low = dict(self.fixture, c=[c if not 91 <= i < 112 else 0.05 for i, c in enumerate(self.fixture['c'])])
		out = FakeTable()
		rtmw_convert.convert(out, info_table(low), Image(*low['image']))
		cols = {name: i for i, name in enumerate(out.rows[0])}
		self.assertEqual(float(out.rows[1][cols['kp33_c']]), 0.0)	# left hand wrist
		self.assertGreater(float(out.rows[1][cols['kp54_c']]), 0.0)	# right hand still there

	def test_no_people_gives_only_the_header(self):
		out = FakeTable()
		rtmw_convert.convert(out, FakeTable([['id', 'conf']]), Image(100, 100))
		self.assertEqual(len(out.rows), 1)


if __name__ == '__main__':
	unittest.main()
