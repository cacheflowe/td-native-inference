"""Build model wrappers from wrappers/*.json and export tox/<Name>.tox. Run inside TD:

	curl -X POST --data-binary @scripts/build_wrappers.py http://127.0.0.1:3031/run

Edit NAMES to choose wrappers (empty = every spec). Wrappers are rebuilt in /project1/WrapperLab,
keeping their position and wires. The project is not saved.
"""

import importlib
import json
import sys
from pathlib import Path

NAMES = ['MediaPipeHands']
EXPORT = True
LAB = '/project1/WrapperLab'
TEST_IMAGE = 'build/thumbs_up.jpg'

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
importlib.reload(wrapper_builder)

lab = op(LAB)
if lab is None:
	lab = op('/project1').create('baseCOMP', LAB.rsplit('/', 1)[1])
	lab.nodeX, lab.nodeY = 0, -600
source = lab.op('source')
if source is None:
	source = lab.create('moviefileinTOP', 'source')
	source.par.file = TEST_IMAGE
	source.nodeX, source.nodeY = 0, 0

report = {}
for index, name in enumerate(NAMES or wrapper_builder.spec_names()):
	spec = wrapper_builder.load_spec(name)
	existed = lab.op(name) is not None
	comp = wrapper_builder.build(spec, lab)
	if not existed:
		comp.nodeX, comp.nodeY = 300, -index * 200
		comp.inputConnectors[0].connect(source)
	entry = {'path': comp.path, 'nodes': len(comp.findChildren())}
	if EXPORT:
		destination = Path(project.folder) / 'tox' / f'{name}.tox'
		entry['export'] = wrapper_builder.export(comp, destination)
		entry['export_problems'] = wrapper_builder.check_export(destination)
	report[name] = entry
print(json.dumps(report, indent=1))
