"""Replays the recorded tracker fixtures in tests/data/tracking (see tests/tracking_replay.py):
the Python tracker must still write exactly the tables it wrote in TouchDesigner, and the C++ port
(cpp/Tracking, when tracker_replay.dll is built) exactly the same tables.

	python -m unittest tests.test_tracking_replay
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking_replay  # noqa: E402

FIXTURES = sorted((Path(__file__).resolve().parent / 'data' / 'tracking').glob('*.jsonl.gz'))


class ReplayTests(unittest.TestCase):
	def check(self, engine):
		self.assertTrue(FIXTURES, 'no fixtures in tests/data/tracking')
		for path in FIXTURES:
			with self.subTest(fixture=path.name):
				report = tracking_replay.replay(str(path), engine, show=3)
				if report.get('skipped'):
					continue
				self.assertGreater(report['compared'], 0)
				self.assertEqual(report['differing_frames'], 0, report['diffs'])

	def test_python_tracker(self):
		self.check('python')

	@unittest.skipUnless(tracking_replay.CPP_DLL.exists(), 'cpp/Tracking not built (build.cmd)')
	def test_cpp_tracker(self):
		self.check('cpp')


if __name__ == '__main__':
	unittest.main()
