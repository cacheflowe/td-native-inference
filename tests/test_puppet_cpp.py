"""The C++ puppet solver (cpp/Puppet, puppet_replay.dll) on the recorded Holistic GPU tracks tables in
tests/data/tracking, against its own stored output (tests/data/puppet/reference.json: per rig, clip and
settings, a hash of every solve's channels to 1e-5). Build first: cpp\\Puppet\\build.cmd. After a
deliberate change, regenerate and review:

	python tests/test_puppet_cpp.py --golden
	python -m unittest tests.test_puppet_cpp

Rigs: a synthetic Mixamo-named T-pose rig (below); build/ybot_rest.json (Y Bot) and
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


# ---------------------------------------------------------------- the synthetic rig

def euler_xyz(rx, ry, rz):
	"""Rotation matrix for TD's rotate order xyz (x applied first): Rz @ Ry @ Rx, degrees."""
	x, y, z = math.radians(rx), math.radians(ry), math.radians(rz)
	cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
	return np.array([
		[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
		[sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
		[-sy, cy * sx, cy * cx],
	])


def to_euler_xyz(m):
	"""Inverse of euler_xyz: (rx, ry, rz) degrees."""
	sy = -m[2, 0]
	if abs(sy) < 0.99999:
		rx, ry, rz = math.atan2(m[2, 1], m[2, 2]), math.asin(sy), math.atan2(m[1, 0], m[0, 0])
	else:
		rx, ry, rz = math.atan2(-m[1, 2], m[1, 1]), math.copysign(math.pi / 2, sy), 0.0
	return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _unit(v):
	n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
	return v / n if n > 1e-9 else None


def _frame(primary, hint):
	"""Orthonormal frame (columns x, y, z), y along primary, x toward hint."""
	y = _unit(primary)
	h = np.asarray(hint, dtype=float)
	x = h - y * np.dot(h, y)
	x = x / math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
	z = np.array([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]])
	return np.column_stack([x, y, z])


def _side(sign, s):
	"""One side's arm and leg: (bone, parent, position) with x mirrored by sign."""
	x = lambda v: sign * v	# noqa: E731
	bones = [
		(f'{s}Shoulder', 'Spine2', (x(6), 145, 0)), (f'{s}Arm', f'{s}Shoulder', (x(18), 145, 0)),
		(f'{s}ForeArm', f'{s}Arm', (x(45), 145, 0)), (f'{s}Hand', f'{s}ForeArm', (x(72), 145, 0)),
		(f'{s}UpLeg', 'Hips', (x(9), 93, 0)), (f'{s}Leg', f'{s}UpLeg', (x(9), 53, 1)),
		(f'{s}Foot', f'{s}Leg', (x(9), 10, -1)), (f'{s}ToeBase', f'{s}Foot', (x(9), 2, 12)),
		(f'{s}Toe_End', f'{s}ToeBase', (x(9), 2, 20)),
	]
	for i in range(1, 5):
		bones.append((f'{s}HandThumb{i}', f'{s}HandThumb{i - 1}' if i > 1 else f'{s}Hand', (x(73 + 3 * i), 145 - 0.5 * i, 2 + 2 * i)))
	for finger, z in (('Index', 2.5), ('Middle', 0.5), ('Ring', -1.5), ('Pinky', -3.5)):
		for i in range(1, 5):
			bones.append((f'{s}Hand{finger}{i}', f'{s}Hand{finger}{i - 1}' if i > 1 else f'{s}Hand',
				(x(80 + 4 * i - abs(z)), 145 - 0.3 * i, z)))
	return bones


SPEC = [
	('Hips', None, (0, 100, 0)), ('Spine', 'Hips', (0, 110, -1)), ('Spine1', 'Spine', (0, 122, -1)),
	('Spine2', 'Spine1', (0, 135, 0)), ('Neck', 'Spine2', (0, 150, 1)), ('Head', 'Neck', (0, 160, 3)),
	('HeadTop_End', 'Head', (0, 178, 5)),
] + _side(1, 'Left') + _side(-1, 'Right')


def synthetic_rig():
	"""A Mixamo-named T-pose rig, bones as {'parent', 't', 'r'}: each bone's +Y aims at its first
	child (Mixamo-style), with a different roll per bone so nothing relies on simple rest rotations."""
	P = 'mixamorig_'
	pos = {n: np.array(p, dtype=float) for n, _, p in SPEC}
	first_child = {}
	for n, parent, _ in SPEC:
		if parent:
			first_child.setdefault(parent, n)
	first_child.update(Hips='Spine', Spine2='Neck', LeftHand='LeftHandMiddle1', RightHand='RightHandMiddle1')
	world, bones = {}, {}
	for i, (n, parent, _) in enumerate(SPEC):
		child = first_child.get(n)
		if child:
			y = _unit(pos[child] - pos[n])
			roll = euler_xyz(*(37 * i % 90 - 45, 0, 0)) @ np.array([1.0, 0.0, 0.0])
			if abs(np.dot(roll, y)) > 0.9:
				roll = np.array([0.0, 0.0, 1.0])
			world[n] = _frame(y, roll)
		else:
			world[n] = world[parent]
		pw = world[parent] if parent else np.eye(3)
		pp = pos[parent] if parent else np.zeros(3)
		bones[P + n] = {'parent': P + parent if parent else None,
			't': list(pw.T @ (pos[n] - pp)), 'r': list(to_euler_xyz(pw.T @ world[n]))}
	return bones


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
	return {bone: euler_xyz(*r) for bone, r in out.items()}


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
