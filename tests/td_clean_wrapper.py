"""Clean-process wrapper test: loads an exported wrapper tox into a fresh TD process.

Build the driver tox from the dev project (POST this file to the bridge's /run), then launch
TouchDesigner.exe on build/clean-wrapper/RunCleanWrapper.tox with:
	TDNI_WRAPPER_TOX  exported wrapper to load
	TDNI_IMAGE        test image fed to its input
	TDNI_RESULT       JSON report path (TD quits after writing it)
The report records the inner node type, errors/warnings, tracks, joints and overlay pixels.
"""

import json
import os
from pathlib import Path

MAX_FRAMES = 900


def _issues(nodes, method):
	return {node.path: getattr(node, method)() for node in nodes if getattr(node, method)()}


def start():
	container = me.parent()
	wrapper = container.loadTox(os.environ['TDNI_WRAPPER_TOX'])
	source = container.create('moviefileinTOP', 'source')
	source.par.file = os.environ['TDNI_IMAGE']
	wrapper.inputConnectors[0].connect(source)
	container.store('frames', 0)
	run('args[0]()', poll, delayFrames=1)


def poll():
	container = me.parent()
	wrapper = next(child for child in container.children if child.family == 'COMP' and child.name != container.name
		and child.op('inference') is not None)
	frames = container.fetch('frames') + 1
	container.store('frames', frames)
	for name in ('debug', 'out_channels'):
		node = wrapper.op(name)
		if node is not None:
			node.cook(force=True)
	tracks = wrapper.op('tracks')
	if frames < MAX_FRAMES and (tracks is None or tracks.numRows < 2):
		run('args[0]()', poll, delayFrames=1)
		return
	finish(wrapper, frames)


def finish(wrapper, frames):
	import numpy as np
	nodes = [wrapper, *wrapper.findChildren()]
	inference = wrapper.op('inference')
	tracks = wrapper.op('tracks')
	header = [cell.val for cell in tracks.row(0)] if tracks is not None and tracks.numRows else []
	rows = [[cell.val for cell in row] for row in tracks.rows()[1:]] if tracks is not None else []
	debug = wrapper.op('debug')
	report = {
		'wrapper': wrapper.path, 'frames': frames, 'project_folder': project.folder,
		'inference_type': inference.OPType if inference is not None else None,
		'errors': _issues(nodes, 'errors'), 'warnings': _issues(nodes, 'warnings'),
		'tracks': len(rows),
		'gestures': [row[header.index('gesture')] for row in rows] if 'gesture' in header else [],
		'joints': wrapper.op('joints').numRows - 1 if wrapper.op('joints') is not None else None,
		'overlay_pixels': int((debug.numpyArray()[..., 3] > 0).sum()) if debug is not None else None,
		'pages': {page.name: [par.name for par in page.pars] for page in wrapper.customPages},
	}
	Path(os.environ['TDNI_RESULT']).write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
	project.quit(force=True)


def build_driver():
	"""Run in the dev project: save build/clean-wrapper/RunCleanWrapper.tox."""
	destination = Path(project.folder) / 'build' / 'clean-wrapper' / 'RunCleanWrapper.tox'
	container = op('/project1').create('baseCOMP', 'RunCleanWrapper')
	try:
		driver = container.create('textDAT', 'test_driver')
		driver.text = (Path(project.folder) / 'tests' / 'td_clean_wrapper.py').read_text(encoding='utf-8')
		startup = container.create('executeDAT', 'startup')
		startup.par.create = True
		startup.par.start = True
		startup.text = ("import os\ndef onStart():\n\tif os.environ.get('TDNI_RESULT') and not parent().fetch('started', False):\n"
			"\t\tparent().store('started', True)\n\t\trun(\"op(args[0]).module.start()\", parent().op('test_driver').path, delayFrames=30)\n"
			"def onCreate():\n\tonStart()\n")
		container.save(destination.as_posix(), createFolders=True)
	finally:
		container.destroy()
	print(json.dumps({'driver': destination.as_posix()}))


if me.name != 'test_driver':
	build_driver()
