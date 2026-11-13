"""The Tracker DAT's track channels (cpp/Tracking/src/Extras.cpp, trackChannels: t{id}_{column} for every
numeric cell of the tracks table) on every recorded tracks table in tests/data/tracking, against the
stored reference (tests/data/tracking/channels.json: per fixture, a hash of the channels it wrote).
After a deliberate change, regenerate and review:

	python tests/test_tracking_extras.py --golden
	python -m unittest tests.test_tracking_extras
"""
import ctypes
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracking_replay  # noqa: E402

FIXTURES = sorted((HERE / 'data' / 'tracking').glob('*.jsonl.gz'))
REFERENCE = HERE / 'data' / 'tracking' / 'channels.json'


def channels(lib, path):
	"""(hash of every frame's channels, tables, names well-formed and unique on every frame)."""
	digest = hashlib.sha256()
	tables, well_formed = 0, True
	_, frames = tracking_replay.load(str(path))
	for frame in frames:
		text = frame['tracks']
		if not text:
			continue
		out = lib.trk_channels(text.encode()).decode()
		names = [line.split('\t')[0] for line in out.splitlines()]
		well_formed &= len(names) == len(set(names)) and all(re.match(r'^t\d+_\w+$', n) for n in names)
		digest.update(f"{frame['f']}\n{out}\n".encode())
		tables += 1
	return digest.hexdigest(), tables, well_formed


def library():
	lib = ctypes.CDLL(str(tracking_replay.CPP_DLL))
	lib.trk_channels.restype = ctypes.c_char_p
	lib.trk_channels.argtypes = [ctypes.c_char_p]
	return lib


@unittest.skipUnless(tracking_replay.CPP_DLL.exists(), 'cpp/Tracking not built (build.cmd)')
class ChannelTests(unittest.TestCase):
	def test_channels_match_reference(self):
		reference = json.loads(REFERENCE.read_text())
		lib = library()
		tables = 0
		for path in FIXTURES:
			with self.subTest(fixture=path.name):
				digest, count, well_formed = channels(lib, path)
				self.assertTrue(well_formed, 'channel names not t{id}_{column}, or repeated')
				self.assertEqual(digest, reference.get(path.name), 'channels changed: --golden, and review')
				tables += count
		self.assertGreater(tables, 100)


if __name__ == '__main__':
	if '--golden' in sys.argv:
		lib = library()
		REFERENCE.write_text(json.dumps({p.name: channels(lib, p)[0] for p in FIXTURES}, indent='\t') + '\n')
		print(f'wrote {REFERENCE}')
	else:
		unittest.main()
