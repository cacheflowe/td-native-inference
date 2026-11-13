"""The C++ port of wrapper_runtime.write_channels (cpp/Tracking/src/Extras.cpp, trackChannels) against
the Python on every recorded tracks table in tests/data/tracking: the same channel names in the same
order, and the same float32 values. (Class Scores and Convert RTMW3D are compared live in TouchDesigner
by tests/td_extras_lab.py: their inputs aren't in the fixtures.)

	python -m unittest tests.test_tracking_extras
"""
import ctypes
import struct
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'python'))
import tracking_replay  # noqa: E402

sys.modules.setdefault('td', types.ModuleType('td'))	# wrapper_runtime imports TouchDesigner's module
import wrapper_runtime  # noqa: E402

FIXTURES = sorted((HERE / 'data' / 'tracking').glob('*.jsonl.gz'))


def f32(value):
	return struct.unpack('f', struct.pack('f', value))[0]


class Cell:
	def __init__(self, val):
		self.val = val


class TracksDat:
	def __init__(self, rows):
		self._rows = rows
		self.numRows = len(rows)

	def cook(self):
		pass

	def row(self, i):
		return [Cell(v) for v in self._rows[i]]

	def rows(self):
		return [self.row(i) for i in range(self.numRows)]


class NoStats:
	def cook(self, force=False):
		pass

	def chans(self):
		return []


class Wrapper:
	def __init__(self, rows):
		self.ops = {'tracks': TracksDat(rows), 'inference_stats': NoStats()}

	def op(self, name):
		return self.ops[name]


class ScriptChop:
	"""Channels as TouchDesigner stores them: one float32 sample each."""

	def __init__(self):
		self.channels = []

	def clear(self):
		self.channels = []

	def appendChan(self, name):
		channel = [name, 0.0]
		self.channels.append(channel)

		class Sample:
			def __setitem__(self, index, value):
				channel[1] = f32(value)
		return Sample()


@unittest.skipUnless(tracking_replay.CPP_DLL.exists(), 'cpp/Tracking not built (build.cmd)')
class ChannelTests(unittest.TestCase):
	def test_channels_match_write_channels(self):
		lib = ctypes.CDLL(str(tracking_replay.CPP_DLL))
		lib.trk_channels.restype = ctypes.c_char_p
		lib.trk_channels.argtypes = [ctypes.c_char_p]
		tables = 0
		for path in FIXTURES:
			_, frames = tracking_replay.load(str(path))
			for frame in frames:
				text = frame['tracks']
				if not text:
					continue
				rows = tracking_replay.parse(text)
				width = max(len(r) for r in rows)
				rows = [r + [''] * (width - len(r)) for r in rows]	# a Table DAT is rectangular
				chop = ScriptChop()
				wrapper_runtime.write_channels(chop, Wrapper(rows))
				expected = [(name, value) for name, value in chop.channels]
				got = []
				for line in lib.trk_channels(text.encode()).decode().splitlines():
					name, value = line.split('\t')
					got.append((name, f32(float(value))))
				with self.subTest(fixture=path.name, frame=frame['f']):
					self.assertEqual([n for n, _ in expected], [n for n, _ in got])
					for (name, a), (_, b) in zip(expected, got):
						if a != a and b != b:
							continue
						self.assertEqual(a, b, name)
				tables += 1
		self.assertGreater(tables, 100)


if __name__ == '__main__':
	unittest.main()
