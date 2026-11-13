"""Replays the recorded tracker fixtures in tests/data/tracking (see tests/tracking_replay.py): the C++
tracker (cpp/Tracking, tracker_replay.dll) must write exactly the tables stored in them, its own
output (tracking_replay.golden). After a deliberate change to the output, regenerate and review:

	python tests/tracking_replay.py tests/data/tracking/*.jsonl.gz --golden
	python -m unittest tests.test_tracking_replay
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking_replay  # noqa: E402

FIXTURES = sorted((Path(__file__).resolve().parent / 'data' / 'tracking').glob('*.jsonl.gz'))


@unittest.skipUnless(tracking_replay.CPP_DLL.exists(), 'cpp/Tracking not built (build.cmd)')
class ReplayTests(unittest.TestCase):
	def test_cpp_tracker(self):
		self.assertTrue(FIXTURES, 'no fixtures in tests/data/tracking')
		for path in FIXTURES:
			with self.subTest(fixture=path.name):
				meta, _ = tracking_replay.load(path)
				self.assertTrue(tracking_replay.is_cpp_reference(meta), 'not a C++ reference: run --golden')
				report = tracking_replay.replay(str(path), 'cpp', show=3)
				self.assertFalse(report.get('skipped'), report)
				self.assertGreater(report['compared'], 0)
				self.assertEqual(report['differing_frames'], 0, report['diffs'])


if __name__ == '__main__':
	unittest.main()
