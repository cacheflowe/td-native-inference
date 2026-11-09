"""Offline tests for object_tracker.KeypointTracker's scale-relative matching (no TD needed).

	python -m unittest discover -s tests -p test_tracking.py
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import mediapipe_skeletons  # noqa: E402
import object_tracker  # noqa: E402

SKELETON = mediapipe_skeletons.COCO_17
_rng = random.Random(7)
# A fixed body layout: each keypoint at a fraction of the person's box.
LAYOUT = [(_rng.uniform(0.2, 0.8), _rng.uniform(0.0, 1.0)) for _ in SKELETON.names]


def person(x, y, height):
	"""Detection dict for a person whose box's bottom-left is (x, y), 0.4 x height wide."""
	width = height * 0.4
	return {'box': [x, y, x + width, y + height], 'score': 0.9,
		'keypoints': [[x + fx * width, y + fy * height, 1.0] for fx, fy in LAYOUT]}


def tracker(relative):
	return object_tracker.KeypointTracker(
		max_match_dist=SKELETON.max_match_dist, distance_keypoint_indices=SKELETON.distance_indices,
		relative_match_dist=SKELETON.rel_match_dist if relative else None, track_buffer=0)


def ids(tracks):
	return sorted(track.track_id for track in tracks)


class ScaleRelativeMatchingTests(unittest.TestCase):
	def test_large_fast_skeleton_keeps_its_id(self):
		# Close to the camera (0.9 of the frame tall) and moving 0.08 per frame.
		for relative, keeps in ((False, False), (True, True)):
			with self.subTest(relative=relative):
				subject = tracker(relative)
				first = ids(subject.update([person(0.1, 0.05, 0.9)]))
				second = ids(subject.update([person(0.18, 0.05, 0.9)]))
				self.assertEqual(first == second, keeps)

	def test_small_neighbour_is_not_absorbed(self):
		# Far away (0.1 tall): person A leaves, B appears 0.04 to the right (under a body width).
		for relative, absorbed in ((False, True), (True, False)):
			with self.subTest(relative=relative):
				subject = tracker(relative)
				first = ids(subject.update([person(0.5, 0.5, 0.1)]))
				second = ids(subject.update([person(0.54, 0.5, 0.1)]))
				self.assertEqual(first == second, absorbed)

	def test_relative_matching_is_scale_invariant(self):
		def scene(scale):
			subject = tracker(True)
			frames = [
				[person(0.1 * scale, 0.1 * scale, 0.2 * scale), person(0.2 * scale, 0.1 * scale, 0.2 * scale)],
				[person(0.12 * scale, 0.1 * scale, 0.2 * scale), person(0.21 * scale, 0.1 * scale, 0.2 * scale)],
				[person(0.13 * scale, 0.1 * scale, 0.2 * scale)],
			]
			object_tracker.KeypointTrack._next_id = 1
			return [ids(subject.update(frame)) for frame in frames]
		self.assertEqual(scene(1.0), scene(3.0))


if __name__ == '__main__':
	unittest.main()


import pose_tracker_pipeline  # noqa: E402

HOLISTIC = mediapipe_skeletons.MP_HOLISTIC
LEFT_HAND = range(mediapipe_skeletons.HOLISTIC_LEFT_HAND, mediapipe_skeletons.HOLISTIC_RIGHT_HAND)


def holistic(world_x, left_hand=True):
	"""One holistic detection: every keypoint at the frame centre, world landmarks all at
	(world_x, 1, 2); the left hand's keypoints get conf 0 (unseen) unless left_hand."""
	kps = [[0.5, 0.5, 0.0 if k in LEFT_HAND and not left_hand else 1.0] for k in range(HOLISTIC.count)]
	return {'box': [0.3, 0.1, 0.7, 0.9], 'score': 0.9, 'keypoints': kps,
		'world': [[world_x, 1.0, 2.0] for _ in range(HOLISTIC.world_count)]}


def run(pipeline, detections, smoothing):
	for det in detections:
		objs = pipeline.update([det], conf_threshold=0.1, smoothing=smoothing, min_hits=1)
	return objs


class WorldLandmarkTests(unittest.TestCase):
	def test_world_is_smoothed_and_written(self):
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		objs = run(pipeline, [holistic(0.0), holistic(1.0)], smoothing=0.5)
		self.assertAlmostEqual(objs[0]['world'][0][0], 0.5)

		class Table:
			def __init__(self):
				self.rows = []

			def clear(self):
				self.rows = []

			def appendRow(self, row):
				self.rows.append(row)

		table = Table()
		pipeline.write_tracks_table(table)
		header, row = table.rows
		# Body and hands only (75 joints x 3); the face has no world landmarks.
		world_cols = [h for h in header if h.endswith(('_wx', '_wy', '_wz'))]
		self.assertEqual(len(world_cols), HOLISTIC.world_count * 3)
		self.assertEqual(world_cols[0], 'pose_nose_wx')
		self.assertEqual(row[header.index('pose_nose_wx')], '0.5000')

	def test_unseen_hand_holds_its_last_position(self):
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		hand = LEFT_HAND.start
		# Never seen yet: no value rather than easing in from the origin.
		objs = run(pipeline, [holistic(3.0, left_hand=False)], smoothing=0.5)
		self.assertIsNone(objs[0]['world'][hand])
		# First sighting is taken as-is, then held while the hand is lost.
		objs = run(pipeline, [holistic(1.0), holistic(5.0, left_hand=False)], smoothing=0.5)
		self.assertEqual(objs[0]['world'][hand], [1.0, 1.0, 2.0])
		self.assertNotEqual(objs[0]['world'][0][0], 1.0)	# the body kept updating

	def test_no_world_columns_without_world_data(self):
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		det = holistic(0.0)
		del det['world']
		objs = run(pipeline, [det], smoothing=0.5)
		self.assertIsNone(objs[0]['world'])


import onnxtop_tables  # noqa: E402


class FakeDat:
	"""Just enough of a Table DAT for read_detections: numRows and row(i) of cells with .val."""

	class Cell:
		def __init__(self, val):
			self.val = str(val)

	def __init__(self, rows):
		self.rows = [[self.Cell(v) for v in row] for row in rows]
		self.numRows = len(self.rows)

	def row(self, i):
		return self.rows[i]


class ReadWorldTests(unittest.TestCase):
	def test_world_columns_turn_to_td_axes(self):
		pose = mediapipe_skeletons.MP_POSE
		header = ['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label']
		header += [f'kp{k}_{s}' for k in range(pose.count) for s in 'uvc']
		header += [f'w{k}_{s}' for k in range(pose.count) for s in 'xyz']
		row = [0, 0, 0.9, 0.1, 0.1, 0.9, 0.9, '']
		row += [0.5] * (pose.count * 3)
		row += [v for k in range(pose.count) for v in (0.1 * k, 0.2, 0.3)]
		det, = onnxtop_tables.read_detections(FakeDat([header, row]), 'mp_pose')
		self.assertEqual(len(det['world']), 33)
		# x kept; MediaPipe's y-down / z-away flipped to TD's y-up / z-toward-camera.
		self.assertEqual(det['world'][2], [0.2, -0.2, -0.3])
		self.assertNotIn('w0_x', det['extras'])
