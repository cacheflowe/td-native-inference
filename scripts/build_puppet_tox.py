"""Build the MixamoPuppet COMP in /project1/WrapperLab and export tox/components/MixamoPuppet.tox. Run inside TD:

	curl -X POST --data-binary @scripts/build_puppet_tox.py http://127.0.0.1:3031/run

FBX loads a character into the lab copy (the exported tox carries none). The project is not saved."""

import importlib
import sys
from pathlib import Path

LAB = '/project1/WrapperLab'
FBX = 'assets/models/y-bot.fbx'
EXPORT = True

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
import puppet_builder
importlib.reload(wrapper_builder)
importlib.reload(puppet_builder)

comp = puppet_builder.build(op(LAB))
comp.par.Fbxfile = FBX
report = {'built': comp.path, 'status': comp.par.Status.eval()}
if EXPORT:
	destination = Path(project.folder) / 'tox' / 'components' / 'MixamoPuppet.tox'
	puppet_builder.export(comp, destination)
	report['export_problems'] = wrapper_builder.check_export(destination)
print(report)
