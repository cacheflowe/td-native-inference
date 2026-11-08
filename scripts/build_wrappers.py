"""Build model wrappers from wrappers/*.json and export tox/<Name>.tox. Run inside TD:

	curl -X POST --data-binary @scripts/build_wrappers.py http://127.0.0.1:3031/run

Edit the settings below. NAMES empty = every spec. Wrappers are rebuilt in /project1/WrapperLab,
keeping their position and wires. Audio wrappers are wired to the lab's audio_source (TEST_AUDIO). DEMO rebuilds
/project1/VisionDemo (every video spec, all models off,
keeping its input wire) and exports tox/VisionDemo.tox. The project is not saved.
"""

import importlib
import json
import sys
from pathlib import Path

NAMES = []
WRAPPERS = True
DEMO = False
EXPORT = True
LAB = '/project1/WrapperLab'
TEST_IMAGE = 'docs/images/sources/hand.jpg'
TEST_AUDIO = 'build/speech_pause.wav'	# local test clip (build/ is gitignored); any speech file works

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
importlib.reload(wrapper_builder)

names = NAMES or wrapper_builder.spec_names()
report = {}
if WRAPPERS:
	lab = op(LAB)
	if lab is None:
		lab = op('/project1').create('baseCOMP', LAB.rsplit('/', 1)[1])
		lab.nodeX, lab.nodeY = 0, -600
	source = lab.op('source')
	if source is None:
		source = lab.create('moviefileinTOP', 'source')
		source.par.file = TEST_IMAGE
		source.nodeX, source.nodeY = 0, 0
	audio_source = lab.op('audio_source')
	if audio_source is None:
		audio_source = lab.create('audiofileinCHOP', 'audio_source')
		audio_source.par.file = TEST_AUDIO
		audio_source.nodeX, audio_source.nodeY = 0, -400
	for name in names:
		spec = wrapper_builder.load_spec(name)
		existed = lab.op(name) is not None
		comp = wrapper_builder.build(spec, lab)
		if not existed:
			comp.nodeX, comp.nodeY = 300, -wrapper_builder.spec_names().index(name) * 200
			comp.inputConnectors[0].connect(audio_source if spec.get('input') == 'audio' else source)
		entry = {'nodes': len(comp.findChildren())}
		if EXPORT:
			destination = Path(project.folder) / 'tox' / f'{name}.tox'
			wrapper_builder.export(comp, destination)
			entry['export_problems'] = wrapper_builder.check_export(destination)
		report[name] = entry
if DEMO:
	demo = wrapper_builder.build_demo(op('/project1'), wrapper_builder.spec_names('video'))
	entry = {'path': demo.path, 'nodes': len(demo.findChildren())}
	if EXPORT:
		destination = Path(project.folder) / 'tox' / 'VisionDemo.tox'
		wrapper_builder.export_demo(demo, destination)
		entry['export_problems'] = wrapper_builder.check_export(destination)
	report['VisionDemo'] = entry
print(json.dumps(report, indent=1))
