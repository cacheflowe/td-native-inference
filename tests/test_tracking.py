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


class Chan:
	def __init__(self, v):
		self.v = v

	def eval(self):
		return self.v


class Stats:
	"""A MediaPipeTOP wrapper's inference_stats Info CHOP: results and result_delay."""
	isCHOP = True

	def __init__(self, results, delay):
		self.chans_ = {'results': Chan(results), 'result_delay': Chan(delay)}

	def chan(self, name):
		return self.chans_.get(name)


class Comp:
	def __init__(self, stats, path='/fake/tracker'):
		self.stats, self.path = stats, path

	def op(self, name):
		return self.stats if name == 'inference_stats' else None

	def parent(self):
		return None


class Table:
	numRows = 2
	path = '/fake/out'


class SkipRepeatedResultTests(unittest.TestCase):
	def setUp(self):
		onnxtop_tables._written.clear()

	def fresh(self, results, delay):
		comp = Comp(Stats(results, delay))
		return onnxtop_tables._fresh(Table(), comp, ('mp_holistic', onnxtop_tables._result_number(comp)))

	def test_repeat_skipped_while_a_newer_result_is_coming(self):
		self.assertTrue(self.fresh(5, 2))
		self.assertFalse(self.fresh(5, 2))
		self.assertTrue(self.fresh(6, 2))

	def test_still_image_runs_every_cook(self):
		# The result is for the newest input frame: the tracker needs repeats to confirm tracks.
		self.assertTrue(self.fresh(5, 0))
		self.assertTrue(self.fresh(5, 0))

	def test_no_stats_runs_every_cook(self):
		comp = Comp(None)
		for _ in range(2):
			self.assertTrue(onnxtop_tables._fresh(Table(), comp, ('detect', onnxtop_tables._result_number(comp))))


import box_tracker_pipeline  # noqa: E402


class ReacquireSnapTests(unittest.TestCase):
	def test_reacquired_after_a_gap(self):
		gaps = {}
		seq = [0, 1, 2, 0, 1, 0]	# lost 2 frames -> back (snap); lost 1 -> back (no snap)
		self.assertEqual([object_tracker.reacquired(gaps, 7, n) for n in seq], [False, False, False, True, False, False])

	def test_box_smooth_snap(self):
		state = {}
		object_tracker.box_smooth(state, 1, [0, 0, 1, 1], 0.9)
		self.assertAlmostEqual(object_tracker.box_smooth(state, 1, [1, 1, 2, 2], 0.9)[0], 0.1)
		self.assertEqual(object_tracker.box_smooth(state, 1, [3, 3, 4, 4], 0.9, snap=True), [3, 3, 4, 4])

	def test_box_track_back_after_a_gap_jumps_to_it(self):
		def det(x):
			return {'box': [x, 0.3, x + 0.3, 0.7], 'score': 0.9, 'class_id': 0, 'class_name': 'person'}

		def cx(gap):
			p = box_tracker_pipeline.BoxTrackerPipeline()
			kw = dict(conf_threshold=0.3, low_conf_threshold=0.1, track_iou_threshold=0.3, track_buffer=30,
				min_hits=1, smoothing=0.9)
			for _ in range(5):
				objs = p.update([det(0.2)], **kw)
			for _ in range(gap):
				p.update([], **kw)
			objs = p.update([det(0.26)], **kw)
			return objs[0]['cx']

		start = 0.2 + 0.15
		snapped, smoothed = cx(3) - start, cx(1) - start
		self.assertGreater(snapped, 0.03)	# most of the 0.06 jump at once (the Kalman estimate)
		self.assertLess(smoothed, 0.02)	# a one-frame dropout still eases in


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

	def test_track_back_after_a_gap_snaps(self):
		def shifted(dx):
			det = holistic(0.0)
			for kp in det['keypoints']:
				kp[0] += dx
			return det

		for gap, expect_snap in ((3, True), (1, False)):
			pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
			run(pipeline, [shifted(0.0)] * 3, smoothing=0.9)
			for _ in range(gap):
				pipeline.update([], conf_threshold=0.1, smoothing=0.9, min_hits=1)
			objs = run(pipeline, [shifted(0.02)], smoothing=0.9)
			x = objs[0]['keypoints'][0][0]
			if expect_snap:
				self.assertAlmostEqual(x, 0.52)
			else:
				self.assertAlmostEqual(x, 0.5 * 0.9 + 0.52 * 0.1)

	def test_unseen_hand_does_not_fly_in_on_screen(self):
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		hand = LEFT_HAND.start
		absent = holistic(0.0, left_hand=False)
		for kp in absent['keypoints'][LEFT_HAND.start:LEFT_HAND.stop]:
			kp[0] = kp[1] = 0.0	# what the plugin writes for a missing part
		# Seen, lost for a while (held, hidden), then back: no easing through the corner.
		objs = run(pipeline, [holistic(0.0)] + [absent] * 5, smoothing=0.9)
		self.assertEqual(objs[0]['keypoints'][hand][:3], [0.5, 0.5, 0.0])
		# Back: hidden, but already at its new reading, until confirmed.
		for _ in range(pose_tracker_pipeline.PART_CONFIRM_FRAMES - 1):
			objs = run(pipeline, [holistic(0.0)], smoothing=0.9)
			self.assertEqual(objs[0]['keypoints'][hand][:3], [0.5, 0.5, 0.0])
		objs = run(pipeline, [holistic(0.0)], smoothing=0.9)
		self.assertEqual(objs[0]['keypoints'][hand][:3], [0.5, 0.5, 1.0])
		# A track that starts without the hand takes its first sighting as-is.
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		objs = run(pipeline, [absent] + [holistic(0.0)] * pose_tracker_pipeline.PART_CONFIRM_FRAMES, smoothing=0.9)
		self.assertEqual(objs[0]['keypoints'][hand][:3], [0.5, 0.5, 1.0])
		# The rest of the body shows from the track's first frame.
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		objs = run(pipeline, [absent], smoothing=0.9)
		self.assertEqual(objs[0]['keypoints'][0][2], 1.0)

	def test_no_world_columns_without_world_data(self):
		pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=HOLISTIC)
		det = holistic(0.0)
		del det['world']
		objs = run(pipeline, [det], smoothing=0.5)
		self.assertIsNone(objs[0]['world'])


import onnxtop_tables  # noqa: E402


class FakeDat:
	"""Just enough of a Table DAT for read_detections: numRows, row(i) of cells with .val, and
	text (tab-separated rows, as TD gives it)."""

	class Cell:
		def __init__(self, val):
			self.val = str(val)

	def __init__(self, rows, path='/fake/info'):
		self.rows = [[self.Cell(v) for v in row] for row in rows]
		self.numRows = len(self.rows)
		self.numCols = len(self.rows[0]) if self.rows else 0
		self.path = path
		self.text = '\r\n'.join('\t'.join(c.val for c in row) for row in self.rows)

	def row(self, i):
		return self.rows[i]

	def __getitem__(self, rc):
		return self.rows[rc[0]][rc[1]]


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

	def test_holistic_without_face_mesh(self):
		class Par:
			def __init__(self, value):
				self.value = value

			def eval(self):
				return self.value

		class Comp:
			def __init__(self, path, face_mesh):
				self.path = path
				self.par = type('Pars', (), {'Trackermode': Par('mp_holistic'), 'Facemesh': Par(face_mesh)})()

			def op(self, name):
				return None

			def parent(self):
				return None

		class Table:
			def __init__(self):
				self.rows, self.path = [], '/fake/out'

			@property
			def numRows(self):
				return len(self.rows)

			def clear(self):
				self.rows = []

			def appendRow(self, row):
				self.rows.append(row)

		skeleton = mediapipe_skeletons.MP_HOLISTIC
		header = ['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label']
		header += [f'kp{k}_{s}' for k in range(skeleton.count) for s in 'uvc']
		row = [0, 0, 0.9, 0.3, 0.2, 0.7, 0.9, ''] + [0.5, 0.5, 1.0] * skeleton.count
		info = FakeDat([header, row], path='/fake/holistic')
		for face_mesh, joints in ((True, 553), (False, 75)):
			out = Table()
			onnxtop_tables.write_tracks(out, Comp(f'/fake/tracker_{face_mesh}', face_mesh), info)
			columns = out.rows[0]
			self.assertEqual(sum(1 for h in columns if h.endswith('_conf')), joints)
			self.assertEqual(len(out.rows), 2)
			self.assertEqual(any(h.startswith('face_') for h in columns), face_mesh)

	def test_header_cache_follows_a_changed_header(self):
		a = FakeDat([['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label']], path='/fake/cache')
		self.assertEqual(onnxtop_tables._columns(a)[1][-1], 'label')
		b = FakeDat([['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label', 'kp0_u', 'kp0_v', 'kp0_c']], path='/fake/cache')
		col, header = onnxtop_tables._columns(b)
		self.assertEqual(col['kp0_c'], 10)
