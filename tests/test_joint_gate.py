"""The C++ tracker's joint gate (Joint Confidence Threshold, Hide Off-Image Joints; not in the Python
tracker), replayed on recorded Info DATs with the per-mode defaults.

	python -m unittest tests.test_joint_gate

Chest-up recordings (the dancer's close-up zoomed so the hips and legs leave the image; a desk
webcam): the models still report knees and ankles, which the gate must output as unseen. Full-body
recordings: legs must stay visible. Uses build/tracking/ (full recordings) when present, else the
trimmed copies in tests/data/tracking/.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import tracking_replay as tr  # noqa: E402

FULL = ROOT / 'build' / 'tracking'
FIXTURES = ROOT / 'tests' / 'data' / 'tracking'
LEGS = ('left_knee', 'right_knee', 'left_ankle', 'right_ankle')
UPPER = ('nose', 'left_shoulder', 'right_shoulder')


def fixture(name):
	path = FULL / f'{name}.jsonl.gz'
	return path if path.exists() else FIXTURES / f'{name}.jsonl.gz'


def seen_rates(name, gate=True, window=None):
	"""Per joint: in frames with a tracked person, how often it's seen in the tracks table (conf > 0)
	and present in the joints table. gate False: the gate off (the Python tracker's output).
	window: (first, last) recorded frames counted (all are replayed)."""
	meta, frames = tr.load(fixture(name))
	if window is None and meta.get('start') == 240:	# the first chest-up recordings: frames 30-95 are the clean framing
		window = (30, 95)
	engine = tr.CppEngine(meta, cpp_only={} if gate else None)
	prefix = 'pose_' if meta['mode'] == 'mp_holistic' else ''
	info = None
	counts = {j: [0, 0] for j in LEGS + UPPER}	# tracks conf > 0, in joints table
	people = 0
	for frame in frames:
		if frame['info'] is not None:
			info = tr.InfoDat(tr.parse(frame['info']), '/replay/info')
		got = {}
		for t, cooks in zip(tr.TABLES, frame['cooked']):
			for _ in range(int(cooks)):
				got[t] = engine.cook(t, info, frame['results'], frame['result_delay'])
		tracks, joints = got.get('tracks'), got.get('joints')
		if not tracks or len(tracks) < 2 or joints is None:
			continue
		if window and not window[0] <= frame['f'] <= window[1]:
			continue
		header = tracks[0]
		row = dict(zip(header, tracks[1]))
		# Legs out of the picture means hips at the image's bottom edge; when the dancer crouches
		# (hips at mid-image) his bent knees come into view.
		if window and (float(row[f'{prefix}left_hip_y']) + float(row[f'{prefix}right_hip_y'])) / 2 > 0.35:
			continue
		people += 1
		first_id = tracks[1][0]
		names = {r[1] for r in joints[1:] if r[0] == first_id}
		for j in counts:
			counts[j][0] += float(row.get(f'{prefix}{j}_conf', 0)) > 0
			counts[j][1] += (prefix + j) in names
	return people, {j: (a / max(people, 1), b / max(people, 1)) for j, (a, b) in counts.items()}


class JointGate(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		if not tr.cpp_lib():
			raise unittest.SkipTest('tracker_replay.dll not built (cpp\\Tracking\\build.cmd)')

	def check_chestup(self, name):
		if not fixture(name).exists():
			self.skipTest(f'no {name} recording')
		people, gated = seen_rates(name)
		_, raw = seen_rates(name, gate=False)
		print(f'\n  {name}: {people} frames; seen (tracks) / drawn (joints), raw -> gated:')
		for j in LEGS + UPPER:
			print(f'    {j:15s} {raw[j][0]:.2f}/{raw[j][1]:.2f} -> {gated[j][0]:.2f}/{gated[j][1]:.2f}')
		self.assertGreater(people, 0)
		for j in LEGS:
			self.assertLess(gated[j][1], 0.15, f'{name}: {j} still drawn')
			self.assertLess(gated[j][0], 0.15, f'{name}: {j} still seen in the tracks table')
		for j in UPPER:
			self.assertGreater(gated[j][1], 0.85, f'{name}: {j} hidden')

	def check_full_body(self, name):
		if not fixture(name).exists():
			self.skipTest(f'no {name} recording')
		people, gated = seen_rates(name)
		print(f'\n  {name}: {people} frames; legs drawn: ' + ', '.join(f'{j} {gated[j][1]:.2f}' for j in LEGS))
		for j in LEGS:
			self.assertGreater(gated[j][1], 0.85, f'{name}: {j} hidden on a full-body clip')

	def test_holistic_gpu_chestup(self):
		self.check_chestup('holistic_gpu_chestup')

	def test_mediapipe_pose_chestup(self):
		self.check_chestup('pose_chestup')

	def test_movenet_chestup(self):
		self.check_chestup('movenet_chestup')

	def test_yolopose_chestup(self):
		meta, _ = tr.load(fixture('yolopose_chestup')) if fixture('yolopose_chestup').exists() else ({}, None)
		if not meta.get('yolo_visibility_fixed'):
			self.skipTest('recorded before the YOLO26 visibility fix (re-record after rebuilding OnnxTOP)')
		self.check_chestup('yolopose_chestup')

	def test_full_body_clips_keep_their_legs(self):
		for name in ('pose_twins', 'movenet_walking'):
			with self.subTest(name):
				self.check_full_body(name)


if __name__ == '__main__':
	unittest.main()
