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
