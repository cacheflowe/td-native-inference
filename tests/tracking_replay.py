"""Replay recorded tracker fixtures (tests/td_tracking_recorder.py) offline through the C++ tracker
(cpp/Tracking, tracker_replay.dll) and diff its tables against the ones stored in the fixtures.

	python tests/tracking_replay.py tests/data/tracking/*.jsonl.gz [--show 5]
	python tests/tracking_replay.py build/tracking/<session>.jsonl.gz --golden	# store its tables

Each fixture holds what a wrapper's tracker read in TouchDesigner, frame by frame (the Info DAT, the
result counters, how many times each table cooked), and the reference tables: the C++ tracker's own
output with its per-mode defaults (golden()), checked by tests/test_tracking_replay.py. Exit code 1
when any fixture differs.
"""
import argparse
import ctypes
import glob
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ('tracks', 'joints', 'bones')
CPP_DLL = ROOT / 'cpp' / 'Tracking' / 'build' / 'tracker_replay.dll'
# The joint gate off (tests/test_joint_gate.py compares against the ungated tables).
GATE_OFF = {'Jointthreshold': 0.0, 'Hideoffimage': 0.0}


def load(path):
	with gzip.open(path, 'rt', encoding='utf-8') as f:
		lines = [json.loads(line) for line in f if line.strip()]
	return lines[0]['meta'], lines[1:]


def parse(text):
	"""Table text (tab-separated, one row per line) -> rows of cell strings."""
	if not text:
		return []
	return [line.split('\t') for line in text.splitlines()]


class InfoDat:
	"""A recorded Info DAT: its rows and its text (what the C++ tracker reads)."""

	def __init__(self, rows, path):
		self.rows = rows
		self.numRows = len(rows)
		self.numCols = len(rows[0]) if rows else 0
		self.path = path
		self.text = '\r\n'.join('\t'.join(r) for r in rows)


_cpp_lib = None


def cpp_lib():
	global _cpp_lib
	if _cpp_lib is None:
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
	"""The C++ tracker (tracker_replay.dll): the recording's tracker parameters over the mode's
	defaults, then `overrides`; the same cooks as in TD, tables back as rows."""

	def __init__(self, meta, overrides=None):
		self.lib = cpp_lib()
		self.lib.trk_reset_ids()
		self.handle = self.lib.trk_create(meta['mode'].encode())
		for name, value in {**meta['tracker_pars'], **(overrides or {})}.items():
			if name == 'Posemethod':
				value = 1.0 if value == 'solvepnp' else 0.0
			if isinstance(value, (bool, int, float)):
				self.lib.trk_set_par(self.handle, name.encode(), float(value))
		width, height = meta['frame_size']
		self.lib.trk_set_par(self.handle, b'Framewidth', float(width))
		self.lib.trk_set_par(self.handle, b'Frameheight', float(height))

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


def is_cpp_reference(meta):
	"""Whether the fixture's tables are the C++ tracker's own (golden())."""
	return meta.get('reference') == 'cpp'


def golden(path, out=None):
	"""Make a recording a fixture: replay its Info DATs through the C++ tracker (the recording's
	parameters over the mode's defaults) and store its tables in place of the recorded ones. Re-run
	after a deliberate change to the tracker's output, and review the diff."""
	meta, frames = load(path)
	engine = CppEngine(meta)
	info = None
	for frame in frames:
		if frame['info'] is not None:
			info = InfoDat(parse(frame['info']), '/replay/info')
		for t, cooks in zip(TABLES, frame['cooked']):
			got = None
			for _ in range(int(cooks)):
				got = engine.cook(t, info, frame['results'], frame['result_delay'])
			frame[t] = '\r\n'.join('\t'.join(row) for row in got) if got is not None else None
	meta = dict(meta, reference='cpp')
	out = Path(out or path)
	with gzip.open(out, 'wt', encoding='utf-8') as f:
		f.write('\n'.join(json.dumps(x) for x in [{'meta': meta}, *frames]) + '\n')
	return out


def replay(path, show=5):
	meta, frames = load(path)
	if not CppEngine.supports(meta):
		return {'fixture': Path(path).name, 'mode': meta['mode'], 'skipped': True}
	engine = CppEngine(meta)
	expected = {t: [] for t in TABLES}
	info_rows = []
	info = None
	report = {'fixture': Path(path).name, 'mode': meta['mode'], 'frames': len(frames), 'compared': 0,
		'differing_frames': 0, 'multi_cooks': 0, 'diffs': []}
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
			diff = first_difference(expected[t], got)
			if diff:
				differs = True
				if len(report['diffs']) < show:
					report['diffs'].append({'f': frame['f'], 'table': t, **diff})
		report['differing_frames'] += differs
	return report


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
	parser.add_argument('--show', type=int, default=5, help='differences listed per fixture')
	parser.add_argument('--trim', type=int, help='instead of replaying: copy the first N frames to --out')
	parser.add_argument('--golden', action='store_true',
		help="instead of replaying: store the C++ tracker's tables in the fixtures (in place); review the diff")
	parser.add_argument('--out', default=str(ROOT / 'tests' / 'data' / 'tracking'))
	args = parser.parse_args(argv)
	paths = [p for pattern in args.fixtures for p in sorted(glob.glob(pattern))]
	if args.golden:
		for path in paths:
			print(golden(path))
		return 0
	if args.trim:
		for path in paths:
			out = trim(path, args.trim, args.out)
			print(f'{out} ({out.stat().st_size // 1024} KB)')
		return 0
	failed = False
	for path in paths:
		report = replay(path, args.show)
		if report.get('skipped'):
			print(f"SKIP {report['fixture']}: {report['mode']} (not supported by the C++ tracker)")
			continue
		ok = report['differing_frames'] == 0
		failed |= not ok
		print(f"{'OK  ' if ok else 'DIFF'} {report['fixture']}: {report['mode']}, {report['frames']} frames, "
			f"{report['compared']} tables compared, {report['differing_frames']} frames differ"
			+ (f", {report['multi_cooks']} multi-cook frames" if report['multi_cooks'] else ''))
		for diff in report['diffs']:
			print('    ', json.dumps(diff))
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
