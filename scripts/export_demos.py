"""Export the hand-built demos in /project1/Demos to tox/<Name>.tox. Run inside TD:

	curl -X POST --data-binary @scripts/export_demos.py http://127.0.0.1:3031/run

Build a demo however you like inside /project1/Demos/<Name> (a Base COMP; wrappers dropped in from
tox/ or the lab). The export makes the saved tox self-contained: DAT file syncs and external-tox
links (the wrappers' shared debug overlays) are cut in the saved copy only. It also loads the node
icon, if docs/images/icons/<Name>.png exists (add the demo to DEMOS in scripts/build_icons.py).

REFRESH_WRAPPERS rebuilds the model wrappers inside each demo from wrappers/*.json first (after a
wrapper change), keeping their position, wires and the parameters you changed on them. The project
is not saved: save it after a refresh to keep the rebuilt wrappers.
VisionDemo and AudioDemo are generated instead: scripts/build_wrappers.py with DEMO = True.
"""

import importlib
import json
import sys
from pathlib import Path

NAMES = []	# demos to export (empty = every COMP in DEMOS)
REFRESH_WRAPPERS = False
DEMOS = '/project1/Demos'

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
importlib.reload(wrapper_builder)

report = {}
for demo in sorted(op(DEMOS).findChildren(type=COMP, depth=1), key=lambda c: c.name):
	if NAMES and demo.name not in NAMES:
		continue
	entry = {'wrappers': [w.name for w in wrapper_builder.demo_wrappers(demo)]}
	if REFRESH_WRAPPERS:
		entry['refreshed'] = wrapper_builder.refresh_wrappers(demo)
	entry['icon'] = wrapper_builder.set_demo_icon(demo) is not None
	destination = Path(project.folder) / 'tox' / f'{demo.name}.tox'
	wrapper_builder.export(demo, destination)
	entry['export_problems'] = wrapper_builder.check_export(destination)
	report[demo.name] = entry
print(json.dumps(report, indent=1))
