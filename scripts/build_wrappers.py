"""Build model wrappers from wrappers/*.json and export tox/<Name>.tox. Run inside TD:

	curl -X POST --data-binary @scripts/build_wrappers.py http://127.0.0.1:3031/run

Edit the settings below. NAMES empty = every spec. Wrappers are rebuilt in /project1/WrapperLab,
keeping their position and wires; audio wrappers are wired to the lab's audio_source (TEST_AUDIO),
text wrappers to text_source. DEMO rebuilds /project1/VisionDemo (every video spec) and
/project1/AudioDemo (the audio wrappers), all models off, keeping their wires, and exports
tox/VisionDemo.tox and tox/AudioDemo.tox. The project is not saved. The other demos are built by
hand in /project1/Demos: scripts/export_demos.py.
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
	text_source = lab.op('text_source')
	if text_source is None:
		text_source = lab.create('textDAT', 'text_source')
		text_source.text = 'Hello from the wrapper lab.'
		text_source.nodeX, text_source.nodeY = 0, -600
	# Optical flow needs motion: the test image swaying (a few percent of the frame, every frame).
	moving = lab.op('moving_source')
	if moving is None:
		moving = lab.create('transformTOP', 'moving_source')
		moving.inputConnectors[0].connect(source)
		moving.par.extend = 'mirror'
		moving.par.tx.expr = '0.04 * math.sin(absTime.frame * 0.15)'
		moving.par.ty.expr = '0.02 * math.cos(absTime.frame * 0.11)'
		moving.nodeX, moving.nodeY = 150, 150
	sources = {'video': source, 'audio': audio_source, 'text': text_source}
	for name in names:
		spec = wrapper_builder.load_spec(name)
		existed = lab.op(name) is not None
		comp = wrapper_builder.build(spec, lab)
		if not existed:
			comp.nodeX, comp.nodeY = 300, -wrapper_builder.spec_names().index(name) * 200
			comp.inputConnectors[0].connect(moving if spec.get('flow') else sources[spec.get('input', 'video')])
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
	existed = op('/project1/AudioDemo') is not None
	audio_demo = wrapper_builder.build_audio_demo(op('/project1'))
	if not existed:
		audio_demo.nodeX, audio_demo.nodeY = demo.nodeX, demo.nodeY - 200
	entry = {'path': audio_demo.path, 'nodes': len(audio_demo.findChildren())}
	if EXPORT:
		destination = Path(project.folder) / 'tox' / 'AudioDemo.tox'
		wrapper_builder.export_demo(audio_demo, destination)
		entry['export_problems'] = wrapper_builder.check_export(destination)
	report['AudioDemo'] = entry
if WRAPPERS:
	# With the lab's Cooking flag off, wrappers just rebuilt (and force-cooked) inside it can keep
	# cooking every frame anyway; switching the flag on and off again stops them.
	wrapper_builder.refresh_cooking(op(LAB))
print(json.dumps(report, indent=1))
