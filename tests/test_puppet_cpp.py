"""The C++ puppet solver (cpp/Puppet, puppet_replay.dll) against the Python one (puppet_solver.cook),
frame by frame on recorded Holistic tracks tables. Build first: cpp\\Puppet\\build.cmd.

	python -m unittest tests.test_puppet_cpp

Rigs: the synthetic Mixamo rig from test_retarget (committed). When present (Mixamo / Blender files
aren't committed): build/ybot_rest.json (Y Bot) and build/hand_rest.json (the rigged hand), and the
full recordings in build/tracking/.

Rotations agree to rounding, not bit for bit (numpy sums matrix products in another order), so each
bone's rx/ry/rz is compared as a rotation matrix (an Euler split can jump near +-180 / gimbal lock
while the rotation is the same); roots and the mirror flag by value.
"""
import ctypes
import gzip
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tests'))
import mixamo_retarget as mr  # noqa: E402
import puppet_solver  # noqa: E402
from test_puppet_solver import Tracks  # noqa: E402
from test_retarget import synthetic_rig  # noqa: E402

DLL = ROOT / 'cpp' / 'Puppet' / 'build' / 'puppet_replay.dll'
FIXTURES = ROOT / 'tests' / 'data' / 'tracking'
FULL = ROOT / 'build' / 'tracking'
HOLISTIC = ('holistic_gpu_dancer', 'holistic_gpu_noface_dancer')
ROT_TOL = 1e-9		# rotation matrix elements
ROOT_TOL = 1e-7		# root translation, relative to its size (rig units)
IMAGE = (1280, 720)


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


def frames(name):
	"""Tracks tables per recorded frame (full recording if there is one), with a gap (None) every
	ninth frame so holding the last pose is exercised too."""
	path = FULL / f'{name}.jsonl.gz' if (FULL / f'{name}.jsonl.gz').exists() else FIXTURES / f'{name}.jsonl.gz'
	rows = [json.loads(line) for line in gzip.open(path, 'rt')][1:]
	out = []
	for i, r in enumerate(rows):
		if 'tracks' not in r:
			continue
		out.append(r['tracks'])
		if i % 9 == 8:
			out.append(None)
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
			cls._lib = lib
		return cls._lib

	def __init__(self, bones, settings):
		self.h = self.lib().pup_create()
		self.kind = self.lib().pup_set_rest(self.h, rest_text(bones).encode())
		self.s = settings

	def close(self):
		self.lib().pup_destroy(self.h)

	def cook(self, text):
		s = self.s
		self.lib().pup_cook(self.h, None if text is None else text.encode(), s['hand'].encode(), s['mirror'],
			s['placement'] == 'image', s['fov'], s['smoothing'], *IMAGE)
		out = []
		for line in self.lib().pup_channels(self.h).decode().splitlines():
			name, value = line.split('\t')
			out.append((name, float(value)))
		return out, self.lib().pup_warning(self.h).decode()


class PyEngine:
	"""puppet_solver.cook on a fake Script CHOP."""

	class Par:
		def __init__(self, value):
			self.value = value

		def eval(self):
			return self.value

	class Chan:
		def __init__(self):
			self.v = [0.0]

		def __setitem__(self, i, value):
			self.v[i] = value

	class Image:
		width, height = IMAGE

	_count = 0

	def __init__(self, bones, settings):
		PyEngine._count += 1
		self.path = f'/fake/solver{PyEngine._count}'
		self.stored = {'rest_bones': bones}
		self.tracks = None
		self.chans, self.warnings = [], []
		self.numSamples = 1
		names = {'Placement': settings['placement'], 'Hand': settings['hand'], 'Mirror': bool(settings['mirror']),
			'Fov': settings['fov'], 'Placesmoothing': settings['smoothing'], 'Image': self.Image(), 'Rig': None}
		self.par = type('Pars', (), {})()
		for name, value in names.items():
			setattr(self.par, name, self.Par(value))
		self.par.Tracks = self

	def eval(self):	# par.Tracks.eval()
		return self.tracks

	def fetch(self, key, default=None):
		return self.stored.get(key, default)

	def store(self, key, value):
		self.stored[key] = value

	def clear(self):
		self.chans = []

	def appendChan(self, name):
		chan = self.Chan()
		self.chans.append((name, chan))
		return chan

	def addWarning(self, text):
		self.warnings.append(text)

	def close(self):
		for cache in (puppet_solver._rigs, puppet_solver._roots, puppet_solver._last, puppet_solver._center,
				puppet_solver._sides, puppet_solver._keys, puppet_solver._chans):
			cache.pop(self.path, None)

	def cook(self, text):
		self.tracks = None if text is None else Tracks([line.split('\t') for line in text.splitlines()])
		self.warnings = []
		puppet_solver.cook(self)
		return [(name, chan.v[0]) for name, chan in self.chans], self.warnings[0] if self.warnings else ''


def _bone_matrices(channels):
	out = {}
	for name, value in channels:
		if name[-3:] in ('_rx', '_ry', '_rz'):
			out.setdefault(name[:-3], [0.0, 0.0, 0.0])['xyz'.index(name[-1])] = value
	return {bone: mr.euler_xyz(*r) for bone, r in out.items()}


class Replay(unittest.TestCase):
	def run_case(self, bones, fixture, settings):
		texts = frames(fixture)
		py, cpp = PyEngine(bones, settings), CppEngine(bones, settings)
		worst_rot = worst_root = 0.0
		try:
			for i, text in enumerate(texts):
				pc, pw = py.cook(text)
				cc, cw = cpp.cook(text)
				where = f'{fixture} {settings} frame {i}'
				self.assertEqual([n for n, _ in pc], [n for n, _ in cc], where)
				self.assertEqual(pw, cw, where)
				pm, cm = _bone_matrices(pc), _bone_matrices(cc)
				for bone in pm:
					d = float(abs(pm[bone] - cm[bone]).max())
					worst_rot = max(worst_rot, d)
					self.assertLess(d, ROT_TOL, f'{where} {bone}')
				for (name, a), (_, b) in zip(pc, cc):
					if name.startswith('root_') or name == 'mirror':
						d = abs(a - b) / max(1.0, abs(a))
						worst_root = max(worst_root, d)
						self.assertLess(d, ROOT_TOL, f'{where} {name}: {a} vs {b}')
		finally:
			py.close()
			cpp.close()
		return len(texts), worst_rot, worst_root

	def body_cases(self, bones, label):
		for fixture in HOLISTIC:
			for placement, mirror in (('image', 0), ('center', 0), ('center', 1)):
				settings = {'placement': placement, 'mirror': mirror, 'hand': 'any', 'fov': 50.0, 'smoothing': 0.6}
				with self.subTest(rig=label, fixture=fixture, placement=placement, mirror=mirror):
					n, rot, root = self.run_case(bones, fixture, settings)
					print(f'  {label} {fixture} {placement} mirror={mirror}: {n} frames, worst rot {rot:.1e}, root {root:.1e}')

	@classmethod
	def setUpClass(cls):
		if not DLL.exists():
			raise unittest.SkipTest(f'{DLL} not built (cpp\\Puppet\\build.cmd)')

	def test_synthetic_body(self):
		self.body_cases(synthetic_rig(), 'synthetic')

	def test_ybot_body(self):
		path = ROOT / 'build' / 'ybot_rest.json'
		if not path.exists():
			self.skipTest('no build/ybot_rest.json')
		self.body_cases(load_rig(path), 'ybot')

	def test_hand_rig(self):
		path = ROOT / 'build' / 'hand_rest.json'
		if not path.exists():
			self.skipTest('no build/hand_rest.json')
		bones = load_rig(path)
		for fixture in HOLISTIC:
			for hand in ('any', 'right', 'left'):
				for placement in ('image', 'center'):
					settings = {'placement': placement, 'mirror': 0, 'hand': hand, 'fov': 50.0, 'smoothing': 0.6}
					with self.subTest(fixture=fixture, hand=hand, placement=placement):
						n, rot, root = self.run_case(bones, fixture, settings)
						print(f'  hand {fixture} {hand} {placement}: {n} frames, worst rot {rot:.1e}, root {root:.1e}')

	def test_no_world_columns_warns(self):
		text = 'track_id\tlost_frames\ttotal_frames\tnose_x\tnose_y\n1\t0\t5\t0.5\t0.5'
		settings = {'placement': 'image', 'mirror': 0, 'hand': 'any', 'fov': 50.0, 'smoothing': 0.6}
		py, cpp = PyEngine(synthetic_rig(), settings), CppEngine(synthetic_rig(), settings)
		try:
			pc, pw = py.cook(text)
			cc, cw = cpp.cook(text)
			self.assertTrue(pw)
			self.assertEqual(pw, cw)
			self.assertEqual([n for n, _ in pc], [n for n, _ in cc])
			for (_, a), (_, b) in zip(pc, cc):
				self.assertTrue(math.isclose(a, b, abs_tol=1e-12))	# the rest pose
		finally:
			py.close()
			cpp.close()


if __name__ == '__main__':
	unittest.main()
