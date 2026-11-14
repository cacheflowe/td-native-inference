"""Export the hand-built components and demos. Run inside TD:

	curl -X POST --data-binary @scripts/export_hand_built.py http://127.0.0.1:3031/run

Three kinds of tox (see CLAUDE.md):
- models, tox/: generated from wrappers/*.json (scripts/build_wrappers.py);
- components, tox/components/: take models' outputs, no model inside. Open source and tracked.
  The hand-built ones live in /project1/Components/<Name>; MixamoPuppet and HandsGeometry have
  their own build scripts;
- demos, tox/demos/: models and components wired together. VisionDemo and AudioDemo are generated
  (build_wrappers.py, DEMO = True) and tracked; the hand-built ones in /project1/Demos/<Name> are
  private: their sources save to demos/<Name>.tox with the project (Demos/save_demos), and their
  exports here are untracked too.

The export makes each tox self-contained: DAT file syncs and external-tox links (the wrappers'
shared debug overlays, a demo's link to a component) are cut in the saved copy only. It also
loads the node icon if docs/images/icons/<Name>.png exists (scripts/build_icons.py, DEMOS).

REFRESH_WRAPPERS rebuilds the model wrappers inside each one from wrappers/*.json first (after a
wrapper change), keeping their position, wires and the parameters you changed on them. Save the
project afterwards to keep the rebuilt wrappers.
"""

import importlib
import json
import sys
from pathlib import Path

NAMES = []	# to export (empty = everything in SOURCES)
REFRESH_WRAPPERS = False
SOURCES = {'/project1/Components': 'tox/components', '/project1/Demos': 'tox/demos'}

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
importlib.reload(wrapper_builder)

report = {}
for source, folder in SOURCES.items():
	parent_comp = op(source)
	if parent_comp is None:
		continue
	for comp in sorted(parent_comp.findChildren(type=COMP, depth=1), key=lambda c: c.name):
		if NAMES and comp.name not in NAMES:
			continue
		entry = {'wrappers': [w.name for w in wrapper_builder.demo_wrappers(comp)]}
		if REFRESH_WRAPPERS:
			entry['refreshed'] = wrapper_builder.refresh_wrappers(comp)
		entry['icon'] = wrapper_builder.set_demo_icon(comp) is not None
		destination = Path(project.folder) / folder / f'{comp.name}.tox'
		wrapper_builder.export(comp, destination)
		entry['tox'] = destination.relative_to(Path(project.folder)).as_posix()
		entry['export_problems'] = wrapper_builder.check_export(destination)
		report[comp.name] = entry
print(json.dumps(report, indent=1))
