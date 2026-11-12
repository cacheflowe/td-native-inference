"""The C++ tracker's math against Python's (needs cpp/Tracking/build.cmd; skipped without the DLL).

	python -m unittest tests.test_tracking_cpp
"""
import ctypes
import math
import random
import unittest
from pathlib import Path

DLL = Path(__file__).resolve().parents[1] / 'cpp' / 'Tracking' / 'build' / 'tracker_replay.dll'


def load():
	lib = ctypes.CDLL(str(DLL))
	lib.trk_hypot.restype = ctypes.c_double
	lib.trk_hypot.argtypes = [ctypes.c_double, ctypes.c_double]
	lib.trk_fixed.restype = ctypes.c_char_p
	lib.trk_fixed.argtypes = [ctypes.c_double, ctypes.c_int]
	return lib


@unittest.skipUnless(DLL.exists(), 'tracker_replay.dll not built')
class MathTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.lib = load()

	def test_hypot_matches_python(self):
		rng = random.Random(3)
		values = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(200000)]
		values += [(rng.uniform(-1e-3, 1e-3), rng.uniform(-1, 1)) for _ in range(100000)]
		values += [(0.0, 0.0), (0.0, 0.5), (3.0, 4.0), (-0.0, 1e-300)]
		bad = [(a, b) for a, b in values if self.lib.trk_hypot(a, b) != math.hypot(a, b)]
		self.assertEqual(bad[:5], [])

	def test_fixed_matches_python(self):
		rng = random.Random(5)
		values = [rng.uniform(-2, 2) for _ in range(100000)] + [-0.00001, -0.0, 0.0, 0.5, 2.5, 0.125, 1e-9, -1e-9]
		values += [round(rng.uniform(-1, 1), 5) for _ in range(100000)]	# halfway cases at 4 digits
		for digits in (3, 4, 5):
			bad = [v for v in values if self.lib.trk_fixed(v, digits).decode() != f'{v:.{digits}f}']
			self.assertEqual(bad[:5], [], f'.{digits}f')


if __name__ == '__main__':
	unittest.main()
