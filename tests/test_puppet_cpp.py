"""The C++ puppet solver (cpp/Puppet, puppet_replay.dll) on the recorded Holistic GPU tracks tables in
tests/data/tracking, against its own stored output (tests/data/puppet/reference.json: per rig, clip and
settings, a hash of every solve's channels to 1e-5). Build first: cpp\\Puppet\\build.cmd. After a
deliberate change, regenerate and review:

	python tests/test_puppet_cpp.py --golden
	python -m unittest tests.test_puppet_cpp

Rigs: the synthetic Mixamo rig from test_retarget (committed); build/ybot_rest.json (Y Bot) and
build/hand_rest.json (the rigged hand) when present (Mixamo / Blender files aren't committed; their
references are kept with the others and skipped without them).
"""
import ctypes
import gzip
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tests'))
import mixamo_retarget as mr  # noqa: E402
from test_retarget import synthetic_rig  # noqa: E402

DLL = ROOT / 'cpp' / 'Puppet' / 'build' / 'puppet_replay.dll'
FIXTURES = ROOT / 'tests' / 'data' / 'tracking'
REFERENCE = ROOT / 'tests' / 'data' / 'puppet' / 'reference.json'
CLIPS = ('holistic_gpu_dancer', 'holistic_gpu_noface_dancer', 'holistic_gpu_chestup')
IMAGE = (1280, 720)
RATE = 60.0	# the recordings step one clip frame per TD frame at 60 fps

# The CHOP's defaults, and everything off (the solve alone).
DEFAULTS = {'placement': 'image', 'mirror': 0, 'hand': 'any', 'fov': 50.0, 'smoothing': 0.6, 'cutoff': 1.5,
	'speed': 4.0, 'slope': 1.0, 'twist': 2.0, 'limits': 1, 'relax': 1}
PLAIN = dict(DEFAULTS, cutoff=0.0, twist=0.0, limits=0, relax=0)
BODY_SETTINGS = {'defaults': DEFAULTS, 'center_mirror': dict(DEFAULTS, placement='center', mirror=1), 'plain': PLAIN}
HAND_SETTINGS = {f'{hand}_{placement}': dict(DEFAULTS, hand=hand, placement=placement)
	for hand in ('any', 'right', 'left') for placement in ('image', 'center')}


def load_rig(path):
	data = json.loads(path.read_text())
	bones = data.get('bones', data)
	return {n: {'parent': b['parent'], 't': b['t'], 'r': b['r'], 's': b.get('s', 1.0)} for n, b in bones.items()}


def rest_text(bones):
	lines = ['name\tparent\ttx\tty\ttz\trx\try\trz\ts']
	for name, b in bones.items():
		s = b.get('s', 1.0)
		s = s[0] if isinstance(s, (list, tuple)) else s
		lines.append('\t'.join([name, b['parent'] or ''] + [repr(float(v)) for v in list(b['t']) + list(b['r'])] + [repr(float(s))]))
	return '\n'.join(lines)


def rigs():
	"""name -> bones, the committed synthetic rig first."""
	out = {'synthetic': synthetic_rig()}
	for name, file in (('ybot', 'ybot_rest.json'), ('hand', 'hand_rest.json')):
		path = ROOT / 'build' / file
		if path.exists():
			out[name] = load_rig(path)
	return out


def frames(name):
	"""(time, tracks text or None) per recorded frame, with a gap (None) every ninth frame so holding
	the last pose is exercised too."""
	rows = [json.loads(line) for line in gzip.open(FIXTURES / f'{name}.jsonl.gz', 'rt')][1:]
	out = []
	for i, r in enumerate(rows):
		if 'tracks' not in r:
			continue
		out.append((r['f'] / RATE, r['tracks']))
		if i % 9 == 8:
			out.append((r['f'] / RATE, None))
	return out


class CppEngine:
	_lib = None

	@classmethod
	def lib(cls):
		if cls._lib is None:
			lib = ctypes.CDLL(str(DLL))
			lib.pup_create.restype = ctypes.c_void_p
			lib.pup_destroy.argtypes = [ctypes.c_void_p]
			lib.pup_set_rest.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
			lib.pup_cook.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
				ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
			lib.pup_channels.argtypes = [ctypes.c_void_p]
			lib.pup_channels.restype = ctypes.c_char_p
			lib.pup_warning.argtypes = [ctypes.c_void_p]
			lib.pup_warning.restype = ctypes.c_char_p
			lib.pup_set_extra.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
				ctypes.c_double]
			lib.pup_set_twist.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
			lib.pup_set_relax.argtypes = [ctypes.c_void_p, ctypes.c_int]
			cls._lib = lib
		return cls._lib

	def __init__(self, bones, settings):
		self.h = self.lib().pup_create()
		self.kind = self.lib().pup_set_rest(self.h, rest_text(bones).encode())
		self.s = settings
		self.lib().pup_set_twist(self.h, settings['twist'], 0.0)
		self.lib().pup_set_relax(self.h, settings['relax'])

	def close(self):
		self.lib().pup_destroy(self.h)

	def cook(self, text, time=-1.0):
		s, lib = self.s, self.lib()
		lib.pup_set_extra(self.h, s['cutoff'], s['speed'], s['slope'], s['limits'], time)
		lib.pup_cook(self.h, None if text is None else text.encode(), s['hand'].encode(), s['mirror'],
			s['placement'] == 'image', s['fov'], s['smoothing'], *IMAGE)
		out = []
		for line in lib.pup_channels(self.h).decode().splitlines():
			name, value = line.split('\t')
			out.append((name, float(value)))
		return out, lib.pup_warning(self.h).decode()


def run(bones, clip, settings):
	"""Hash of every cook's channels (names and values to 1e-5), and the number of cooks."""
	engine = CppEngine(bones, settings)
	digest = hashlib.sha256()
	cooks = 0
	try:
		for time, text in frames(clip):
			channels, warning = engine.cook(text, time)
			digest.update((warning + '\n' + '\n'.join(f'{n}\t{v:.5f}' for n, v in channels) + '\n').encode())
			cooks += 1
	finally:
		engine.close()
	return digest.hexdigest(), cooks


def cases():
	"""(key, bones, clip, settings) for every rig present."""
	for rig, bones in rigs().items():
		for clip in CLIPS:
			for label, settings in (HAND_SETTINGS if rig == 'hand' else BODY_SETTINGS).items():
				yield f'{rig}/{clip}/{label}', bones, clip, settings


def _bone_matrices(channels):
	out = {}
	for name, value in channels:
		if name[-3:] in ('_rx', '_ry', '_rz'):
			out.setdefault(name[:-3], [0.0, 0.0, 0.0])['xyz'.index(name[-1])] = value
	return {bone: mr.euler_xyz(*r) for bone, r in out.items()}


@unittest.skipUnless(DLL.exists(), f'{DLL} not built (cpp\\Puppet\\build.cmd)')
class Reference(unittest.TestCase):
	def test_matches_reference(self):
		reference = json.loads(REFERENCE.read_text())
		checked = 0
		for key, bones, clip, settings in cases():
			with self.subTest(key):
				self.assertIn(key, reference, 'no reference: --golden, and review')
				digest, cooks = run(bones, clip, settings)
				self.assertGreater(cooks, 0)
				self.assertEqual(digest, reference[key], 'output changed: --golden, and review')
				checked += 1
		self.assertGreater(checked, 0)

	def test_smoothing_calms_forearms_and_hands(self):
		"""With the CHOP's default smoothing, the forearms' and hands' frame-to-frame jumps (mostly roll,
		from the noisy knuckle line) shrink."""

		def steps(settings):
			engine = CppEngine(synthetic_rig(), settings)
			out, last, prev = [], None, None
			for time, text in frames('holistic_gpu_dancer'):
				if text is None or text == last:
					continue
				last = text
				channels, _ = engine.cook(text, time)
				m = _bone_matrices([(n, v) for n, v in channels if 'ForeArm' in n or n[:-3].endswith('Hand')])
				if prev is not None:
					for bone in m:
						c = (float(np.trace(prev[bone].T @ m[bone])) - 1) / 2
						out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
				prev = m
			engine.close()
			return math.sqrt(sum(a * a for a in out) / len(out))

		raw, smoothed = steps(PLAIN), steps(DEFAULTS)
		print(f'  forearm / hand RMS step: raw {raw:.1f} deg, smoothed {smoothed:.1f} deg')
		self.assertLess(smoothed, raw * 0.6)

	def test_no_world_columns_warns(self):
		text = 'track_id\tlost_frames\ttotal_frames\tnose_x\tnose_y\n1\t0\t5\t0.5\t0.5'
		bones = synthetic_rig()
		engine = CppEngine(bones, DEFAULTS)
		try:
			channels, warning = engine.cook(text)
		finally:
			engine.close()
		self.assertIn('World Landmark Columns', warning)
		self.assertEqual(len(channels), 3 * len(bones) + 4)	# the rest pose, root and mirror


if __name__ == '__main__':
	if '--golden' in sys.argv:
		REFERENCE.parent.mkdir(parents=True, exist_ok=True)
		reference = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
		for key, bones, clip, settings in cases():
			reference[key] = run(bones, clip, settings)[0]
			print(key)
		REFERENCE.write_text(json.dumps(reference, indent='\t', sort_keys=True) + '\n')
		print(f'wrote {REFERENCE}')
	else:
		unittest.main()
