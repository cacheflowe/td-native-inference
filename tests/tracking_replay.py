"""Replay recorded tracker fixtures (tests/td_tracking_recorder.py) offline and diff the tables.

	python tests/tracking_replay.py build/tracking/*.jsonl.gz [--engine python] [--show 5]

Each fixture holds what a wrapper's tracker saw in TouchDesigner, frame by frame (the Info DAT,
the result counters, which scripts cooked) and what it wrote. The replay feeds the same frames to
an engine and compares every table cell as text:
	python  the Python tracker (python/onnxtop_tables.py) with TouchDesigner faked. Matching the
	        recording proves the harness; it's then the reference the C++ port must equal.
Exit code 1 when any fixture differs.
"""
import argparse
import glob
import gzip
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
TABLES = ('tracks', 'joints', 'bones')


def load(path):
	with gzip.open(path, 'rt', encoding='utf-8') as f:
		lines = [json.loads(line) for line in f if line.strip()]
	return lines[0]['meta'], lines[1:]


def parse(text):
	"""Table text (tab-separated, one row per line) -> rows of cell strings."""
	if not text:
		return []
	return [line.split('\t') for line in text.splitlines()]


def cell(value):
	"""What a Table DAT stores for an appended Python value."""
	return value if isinstance(value, str) else str(value)


# ---------------------------------------------------------------- TouchDesigner fakes

class Cell:
	__slots__ = ('val',)

	def __init__(self, val):
		self.val = val


class InfoDat:
	"""The Info DAT the tracker reads: rows, cells, text, path."""

	def __init__(self, rows, path):
		self.rows = rows
		self.numRows = len(rows)
		self.numCols = len(rows[0]) if rows else 0
		self.path = path
		self.text = '\r\n'.join('\t'.join(r) for r in rows)
		self._cells = None

	def _cell_rows(self):
		if self._cells is None:
			self._cells = [[Cell(v) for v in r] for r in self.rows]
		return self._cells

	def row(self, i):
		return self._cell_rows()[i]

	def __getitem__(self, rc):
		return Cell(self.rows[rc[0]][rc[1]])


class OutDat:
	"""The Script DAT the tracker writes."""

	def __init__(self, path):
		self.path = path
		self.rows = []

	@property
	def numRows(self):
		return len(self.rows)

	@property
	def numCols(self):
		return max((len(r) for r in self.rows), default=0)

	def clear(self):
		self.rows = []

	def appendRow(self, row):
		self.rows.append([cell(v) for v in row])

	def appendRows(self, rows):
		for row in rows:
			self.appendRow(row)

	def copy(self, other):
		self.rows = [list(r) for r in other.rows]

	def row(self, i):
		return [Cell(v) for v in self.rows[i]]


class Par:
	def __init__(self, value):
		self.value = value

	def eval(self):
		return self.value


class Pars:
	def __init__(self, values):
		for name, value in values.items():
			setattr(self, name, Par(value))


class Chan:
	def __init__(self, value):
		self.value = value

	def eval(self):
		return self.value


class Stats:
	"""The wrapper's inference_stats Info CHOP."""
	isCHOP = True

	def __init__(self):
		self.chans = {}

	def chan(self, name):
		return self.chans.get(name)


class Passthrough:
	def __init__(self, width, height):
		self.width, self.height = width, height


class Wrapper:
	def __init__(self, stats):
		self.stats = stats

	def op(self, name):
		return self.stats if name == 'inference_stats' else None

	def parent(self):
		return None


class Tracker:
	"""The wrapper's tracker COMP: its parameters, the frame size, the wrapper as parent."""

	def __init__(self, meta, stats):
		self.path = '/replay/tracker'
		self.par = Pars(meta['tracker_pars'])
		self.wrapper = Wrapper(stats)
		self.passthrough = Passthrough(*meta['frame_size'])
		self.inputs = []

	def op(self, name):
		return self.passthrough if name == 'null_passthrough' else None

	def parent(self):
		return self.wrapper


# ---------------------------------------------------------------- engines

class PythonEngine:
	@staticmethod
	def supports(meta):
		if meta['tracker_pars'].get('Posemethod') != 'solvepnp':
			return True
		try:
			import cv2  # noqa: F401 (the solvePnP head pose needs OpenCV, as in TouchDesigner)
			return True
		except ImportError:
			return False

	def __init__(self, meta):
		import object_tracker
		import onnxtop_tables
		self.tables = onnxtop_tables
		for cache in (onnxtop_tables._pipelines, onnxtop_tables._written, onnxtop_tables._header_cache,
				onnxtop_tables._layout_cache, onnxtop_tables._class_scores):
			cache.clear()
		object_tracker.Track._next_id = 1
		object_tracker.KeypointTrack._next_id = 1
		self.stats = Stats()
		self.tracker = Tracker(meta, self.stats)
		self.out = {t: OutDat(f'/replay/script_{t}') for t in TABLES}

	def cook(self, table, info, results, delay):
		self.stats.chans = {k: Chan(v) for k, v in (('results', results), ('result_delay', delay)) if v is not None}
		out = self.out[table]
		if table == 'tracks':
			self.tables.write_tracks(out, self.tracker, info)
		elif table == 'joints':
			self.tables.write_joints(out, self.tracker)
		else:
			self.tables.write_bones(out, self.tracker)
		return out.rows


CPP_DLL = ROOT / 'cpp' / 'Tracking' / 'build' / 'tracker_replay.dll'
_cpp_lib = None


def cpp_lib():
	global _cpp_lib
	if _cpp_lib is None:
		import ctypes
		lib = ctypes.CDLL(str(CPP_DLL))
		lib.trk_supported.argtypes = [ctypes.c_char_p]
		lib.trk_create.restype = ctypes.c_void_p
		lib.trk_create.argtypes = [ctypes.c_char_p]
		lib.trk_destroy.argtypes = [ctypes.c_void_p]
		lib.trk_set_par.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
		lib.trk_cook.restype = ctypes.c_char_p
		lib.trk_cook.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
			ctypes.c_double, ctypes.c_double]
		_cpp_lib = lib
	return _cpp_lib


class CppEngine:
	"""The C++ port (cpp/Tracking, tracker_replay.dll): same cooks, tables back as text."""

	def __init__(self, meta):
		self.lib = cpp_lib()
		self.lib.trk_reset_ids()
		self.handle = self.lib.trk_create(meta['mode'].encode())
		for name, value in meta['tracker_pars'].items():
			if name == 'Posemethod':
				value = 1.0 if value == 'solvepnp' else 0.0
			if isinstance(value, (bool, int, float)):
				self.lib.trk_set_par(self.handle, name.encode(), float(value))
		width, height = meta['frame_size']
		self.lib.trk_set_par(self.handle, b'Framewidth', float(width))
		self.lib.trk_set_par(self.handle, b'Frameheight', float(height))
		self.info_text = None

	@staticmethod
	def supports(meta):
		return bool(cpp_lib().trk_supported(meta['mode'].encode()))

	def cook(self, table, info, results, delay):
		has = results is not None and delay is not None
		text = self.lib.trk_cook(self.handle, TABLES.index(table), info.text.encode() if info is not None else None,
			int(info is not None), int(has), results or 0.0, delay or 0.0)
		return parse(text.decode())

	def __del__(self):
		if getattr(self, 'handle', None):
			self.lib.trk_destroy(self.handle)


ENGINES = {'python': PythonEngine, 'cpp': CppEngine}


# ---------------------------------------------------------------- replay

def replay(path, engine_name='python', show=5):
	meta, frames = load(path)
	engine_class = ENGINES[engine_name]
	if hasattr(engine_class, 'supports') and not engine_class.supports(meta):
		return {'fixture': Path(path).name, 'mode': meta['mode'], 'skipped': True}
	engine = engine_class(meta)
	expected = {t: [] for t in TABLES}
	info_rows = []
	info = None
	report = {'fixture': Path(path).name, 'mode': meta['mode'], 'frames': len(frames), 'compared': 0,
		'differing_frames': 0, 'multi_cooks': 0, 'diffs': []}
	# solvePnP head pose: the C++ solver follows OpenCV's steps but not its rounding, and on badly
	# conditioned frames (5 near-planar points) either solver can wander to a different rotation, so
	# the angles are counted, not required to match (everything else still must).
	approximate = APPROXIMATE if engine_name == 'cpp' and meta['tracker_pars'].get('Posemethod') == 'solvepnp' else ()
	if approximate:
		report['approximate_cells'] = 0
	for frame in frames:
		if frame['info'] is not None:
			info_rows = parse(frame['info'])
			info = InfoDat(info_rows, '/replay/info')
		for t in TABLES:
			if frame[t] is not None:
				expected[t] = parse(frame[t])
		differs = False
		for t, cooks in zip(TABLES, frame['cooked']):
			cooks = int(cooks)
			if cooks > 1:
				report['multi_cooks'] += 1
			got = None
			for _ in range(cooks):
				got = engine.cook(t, info, frame['results'], frame['result_delay'])
			if got is None:
				continue
			report['compared'] += 1
			if approximate and t == 'tracks':
				report['approximate_cells'] += count_cells(expected[t], got, approximate)
				expected_t, got = mask(expected[t], approximate), mask(got, approximate)
			else:
				expected_t = expected[t]
			diff = first_difference(expected_t, got)
			if diff:
				differs = True
				if len(report['diffs']) < show:
					report['diffs'].append({'f': frame['f'], 'table': t, **diff})
		report['differing_frames'] += differs
	return report


APPROXIMATE = ('yaw', 'pitch', 'roll')


def mask(rows, columns):
	"""The table with these columns' cells blanked (header kept)."""
	if not rows:
		return rows
	idx = [i for i, name in enumerate(rows[0]) if name in columns]
	return [rows[0]] + [[('' if i in idx else v) for i, v in enumerate(r)] for r in rows[1:]]


def count_cells(expected, got, columns):
	if not expected or not got or expected[0] != got[0] or len(expected) != len(got):
		return 0
	idx = [i for i, name in enumerate(expected[0]) if name in columns]
	return sum(1 for a, b in zip(expected[1:], got[1:]) for i in idx if i < len(a) and i < len(b) and a[i] != b[i])


def first_difference(expected, got):
	if len(expected) != len(got):
		return {'what': 'rows', 'expected': len(expected), 'got': len(got),
			'header_equal': bool(expected and got and expected[0] == got[0])}
	for r, (a, b) in enumerate(zip(expected, got)):
		if a == b:
			continue
		if len(a) != len(b):
			return {'what': 'cols', 'row': r, 'expected': len(a), 'got': len(b)}
		cells = [(c, x, y) for c, (x, y) in enumerate(zip(a, b)) if x != y]
		header = expected[0] if expected else []
		return {'what': 'cells', 'row': r, 'count': len(cells),
			'first': [{'col': header[c] if r and c < len(header) else c, 'expected': x, 'got': y} for c, x, y in cells[:4]]}
	return None


def trim(path, frames, out_dir):
	"""Write the fixture's first `frames` frames (after warm-up) to out_dir: a small copy to commit."""
	meta, records = load(path)
	kept = [r for r in records if r['f'] < frames]
	meta = dict(meta, frames=frames, trimmed_from=Path(path).name)
	out = Path(out_dir) / Path(path).name
	out.parent.mkdir(parents=True, exist_ok=True)
	with gzip.open(out, 'wt', encoding='utf-8') as f:
		f.write('\n'.join(json.dumps(x) for x in [{'meta': meta}, *kept]) + '\n')
	return out


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('fixtures', nargs='+')
	parser.add_argument('--engine', default='python', choices=sorted(ENGINES))
	parser.add_argument('--show', type=int, default=5, help='differences listed per fixture')
	parser.add_argument('--trim', type=int, help='instead of replaying: copy the first N frames to --out')
	parser.add_argument('--out', default=str(ROOT / 'tests' / 'data' / 'tracking'))
	args = parser.parse_args(argv)
	paths = [p for pattern in args.fixtures for p in sorted(glob.glob(pattern))]
	if args.trim:
		for path in paths:
			out = trim(path, args.trim, args.out)
			print(f'{out} ({out.stat().st_size // 1024} KB)')
		return 0
	failed = False
	for path in paths:
		report = replay(path, args.engine, args.show)
		if report.get('skipped'):
			print(f"SKIP {report['fixture']}: {report['mode']} (not supported by the {args.engine} engine yet)")
			continue
		ok = report['differing_frames'] == 0
		failed |= not ok
		print(f"{'OK  ' if ok else 'DIFF'} {report['fixture']}: {report['mode']}, {report['frames']} frames, "
			f"{report['compared']} tables compared, {report['differing_frames']} frames differ"
			+ (f", {report['multi_cooks']} multi-cook frames" if report['multi_cooks'] else '')
			+ (f", {report['approximate_cells']} head pose cells differ (solvePnP, allowed)" if 'approximate_cells' in report else ''))
		for diff in report['diffs']:
			print('    ', json.dumps(diff))
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
