"""The Puppet CHOP (C++) vs the Python puppet solver, live in TouchDesigner. POST to the bridge's /run:

	curl -X POST --data-binary @tests/td_puppet_lab.py http://127.0.0.1:3031/run

In /project1/PuppetLab, next to each Python solver (`solver_py`: Y Bot; `hand_solver_py`: the
rigged hand from MediaPipe Hands) it builds a Puppet CHOP ('_cmp_*', additions off) on the same tracks,
image and settings, with a rest-pose table captured from the same FBX COMP
(puppet_runtime.rest_rows). For FRAMES frames it pulls both and compares every bone's rotation (as a
matrix: an Euler split can jump at +-180) and the root, and reads both cook times. Returns at once;
the report lands in build/puppet-lab.json. The lab's Cooking flag is restored afterwards.

Before TD has registered PuppetCHOP.dll (a restart after its first build) the C++ solvers are
CPlusPlus CHOPs loading it by path: TD asks to Allow the DLL on the first one.
"""
import json
import sys
from pathlib import Path

ROOT = Path(project.folder)
LAB = '/project1/PuppetLab'
REPORT = ROOT / 'build' / 'puppet-lab.json'
FRAMES = 300
PAIRS = [('solver_py', '_cmp_solver', '_cmp_rest'), ('hand_solver_py', '_cmp_hand_solver', '_cmp_hand_rest')]
ROT_TOL = 1e-4		# rotation matrix elements (channels are float32)
ROOT_TOL = 1e-3		# root translation, rig units (centimetres)

python_dir = (ROOT / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import importlib  # noqa: E402

import mixamo_retarget  # noqa: E402
import puppet_builder  # noqa: E402
import puppet_runtime  # noqa: E402

importlib.reload(puppet_runtime)
importlib.reload(puppet_builder)

lab = op(LAB)
state = {'frame': 0, 'cooking': lab.allowCooking, 'handinput': lab.par.Handinput.eval(), 'pairs': {}}
HAND_INPUT = 'clip'	# the hand clip (the lab may be on its webcam)


def setup():
	for i, (py_name, cpp_name, rest_name) in enumerate(PAIRS):
		py = lab.op(py_name)
		# Placement smoothing off for the run: the Script CHOP also solves when the render pulls it
		# (sometimes a newer row than this test's pull), so its smoothed root would follow another
		# history than the Puppet CHOP's. Unsmoothed, the root depends on the row alone.
		state.setdefault('smoothing', {})[py_name] = (py.par.Placesmoothing.mode, py.par.Placesmoothing.expr,
			py.par.Placesmoothing.val)
		py.par.Placesmoothing.mode = ParMode.CONSTANT
		py.par.Placesmoothing.val = 0.0
		rest = lab.op(rest_name) or lab.create(tableDAT, rest_name)
		rest.clear()
		for row in puppet_runtime.rest_rows(py.par.Rig.eval()):
			rest.appendRow(row)
		cpp = lab.op(cpp_name) or puppet_builder._puppet_chop(lab, cpp_name)
		cpp.par.Restpose = rest.name
		cpp.par.Tracks = py.par.Tracks.eval().path
		cpp.par.Image = py.par.Image.eval().path
		for name in ('Hand', 'Mirror', 'Placement', 'Fov', 'Placesmoothing'):
			cpp.par[name].expr = f"op('{py_name}').par.{name}"
		# The C++ solver's additions (smoothing, knee limits) off: the Python solver has none.
		for name in ('Smoothcutoff', 'Twistcutoff', 'Kneelimits', 'Relaxunseen'):
			cpp.par[name].mode = ParMode.CONSTANT
			cpp.par[name].val = 0
		cpp.nodeX, cpp.nodeY = py.nodeX, py.nodeY - 150
		rest.nodeX, rest.nodeY = py.nodeX - 150, py.nodeY - 150
		state['pairs'][py_name] = {'cpp': cpp.path, 'type': cpp.OPType, 'frames': 0, 'compared': 0, 'differing': 0,
			'first_difference': None, 'worst_rot': 0.0, 'worst_root': 0.0, 'py_ms': [], 'cpp_ms': [], 'solve_ms': []}


def channels(chop):
	return [(c.name, c.eval()) for c in chop.chans()]


def matrices(chans):
	out = {}
	for name, value in chans:
		if name[-3:] in ('_rx', '_ry', '_rz'):
			out.setdefault(name[:-3], [0.0, 0.0, 0.0])['xyz'.index(name[-1])] = value
	return {bone: mixamo_retarget.euler_xyz(*r) for bone, r in out.items()}


def compare(py_name):
	rec = state['pairs'][py_name]
	py, cpp = lab.op(py_name), op(rec['cpp'])
	# Both cooked here, from the same tracks row: the render pulls the Script CHOP at another moment
	# of the frame than this callback would pull the Puppet CHOP (a frame apart on a playing clip).
	# Each solves a new row once either way (an unchanged row keeps its channels).
	py.cook(force=True)
	cpp.cook(force=True)
	a, b = channels(py), channels(cpp)
	frame = absTime.frame
	rec['frames'] += 1
	tracks = py.par.Tracks.eval()
	if tracks is not None and tracks.numRows > 1:
		rec['tracked'] = rec.get('tracked', 0) + 1
	if py.cookAbsFrame == frame:
		rec['py_ms'].append(py.cpuCookTime)
	if cpp.cookAbsFrame == frame:
		rec['cpp_ms'].append(cpp.cpuCookTime)
	if not a:
		return
	rec['compared'] += 1
	problem = None
	if [n for n, _ in a] != [n for n, _ in b]:
		problem = f'channel names differ ({len(a)} vs {len(b)})'
	else:
		ma, mb = matrices(a), matrices(b)
		for bone in ma:
			d = float(abs(ma[bone] - mb[bone]).max())
			rec['worst_rot'] = max(rec['worst_rot'], d)
			if d > ROT_TOL and problem is None:
				problem = f'{bone}: rotation differs by {d:.2e}'
		for (name, x), (_, y) in zip(a, b):
			if name.startswith('root_') or name == 'mirror':
				d = abs(x - y)
				rec['worst_root'] = max(rec['worst_root'], d)
				if d > ROOT_TOL and problem is None:
					problem = f'{name}: {x} vs {y}'
	if problem:
		rec['differing'] += 1
		if rec['first_difference'] is None:
			rec['first_difference'] = {'frame': rec['frames'], 'problem': problem}


def step():
	for py_name in state['pairs']:
		compare(py_name)
	state['frame'] += 1
	if state['frame'] < FRAMES:
		run('args[0]()', step, delayFrames=1)
	else:
		finish()


def finish():
	lab.allowCooking = state['cooking']
	lab.par.Handinput = state['handinput']
	for py_name, (mode, expr, val) in state['smoothing'].items():
		par = lab.op(py_name).par.Placesmoothing
		par.val, par.expr, par.mode = val, expr, mode

	def mean(values):
		return round(sum(values) / len(values), 4) if values else None

	report = {}
	for py_name, rec in state['pairs'].items():
		report[py_name] = {k: v for k, v in rec.items() if not k.endswith('_ms')}
		report[py_name].update({'py_cook_ms': mean(rec['py_ms']), 'cpp_cook_ms': mean(rec['cpp_ms']),
			'py_cooks': len(rec['py_ms']), 'cpp_cooks': len(rec['cpp_ms'])})
	REPORT.write_text(json.dumps(report, indent=1))


setup()
lab.allowCooking = True
if HAND_INPUT in lab.par.Handinput.menuNames:
	lab.par.Handinput = HAND_INPUT
run('args[0]()', step, delayFrames=2)
print({'started': [p for p in state['pairs']], 'types': {k: v['type'] for k, v in state['pairs'].items()}, 'frames': FRAMES})
