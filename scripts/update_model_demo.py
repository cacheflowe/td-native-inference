"""Add the combined active-model debug output without rebuilding or resetting models."""

import json
from pathlib import Path


demo = op('/project1/ModelDemo') or op('/project1/ONNX_Native/ModelDemo')
assert demo is not None, 'ModelDemo not found'
names = demo.fetch('completed_wrappers')
before = {name: demo.par[name.capitalize()].eval() for name in names}
source = list(demo.inputs)
references = []
for node in [demo, *demo.findChildren()]:
	for parameter in node.pars():
		if '/ONNX_Native' in (parameter.expr or '') or (parameter.isOP and '/ONNX_Native' in str(parameter.eval())):
			references.append(f'{node.path}.{parameter.name}')
assert not references, f'Demo still references legacy nodes: {references}'
counts = {name: demo.op(name + '/inference').totalCooks for name in names if not before[name]}
order = demo.op('component').module.setup_demo_composite(demo)
demo.op('out_debug').cook(force=True)
assert not demo.op('debug_composite').errors(), demo.op('debug_composite').errors()
assert demo.op('debug_composite').par.operand.eval() == 'over'
assert len(demo.op('debug_composite').inputs) == len(names)
assert (demo.op('out_debug').width, demo.op('out_debug').height) == (demo.op('input_frame').width, demo.op('input_frame').height)
assert before == {name: demo.par[name.capitalize()].eval() for name in names}
assert source == list(demo.inputs)
for name, count in counts.items():
	assert demo.op(name + '/inference').totalCooks == count, f'Composite woke inactive {name}'
legacy = op('/project1/ONNX_Native')
native_sources = [node.name for node in legacy.children if node.name.startswith(('Onnx_', 'MP_'))] if legacy is not None else []
wrapped_sources = {demo.op(name).fetch('model_wrapper_spec')['source'] for name in names}
assert not set(native_sources) - wrapped_sources, 'Some legacy models are missing wrappers'
container = demo.parent().create('baseCOMP', '_debug_composite_validation')
try:
	driver = container.create('textDAT', 'composite_driver')
	driver.text = (Path(project.folder) / 'tests/td_model_wrappers.py').read_text(encoding='utf-8')
	pixel_checks = driver.module.check_demo_composite(demo)
	demo.op('component').module.export_demo(demo, Path(project.folder) / 'tox/ModelDemo.tox')
	subject = container.loadTox(str(Path(project.folder) / 'tox/ModelDemo.tox'))
	subject.par.Packageroot = project.folder
	subject.inputConnectors[0].connect(container.op('input_frame').outputConnectors[0])
	import numpy as np
	for name in names:
		assert not subject.par[name.capitalize()].eval(), f'Exported {name} is enabled'
		assert not subject.op(name).allowCooking, f'Exported {name} is cooking'
	idle_counts = {name: subject.op(name + '/inference').totalCooks for name in names}
	subject.op('out_debug').cook(force=True)
	assert np.allclose(subject.op('out_debug').numpyArray(), 0), 'Exported demo is not transparent by default'
	assert idle_counts == {name: subject.op(name + '/inference').totalCooks for name in names}
	for node in [subject, *subject.findChildren()]:
		for parameter_name in ('syncfile', 'loadonstart', 'externaltox', 'enableexternaltox'):
			parameter = getattr(node.par, parameter_name, None)
			assert parameter is None or not parameter.eval(), f'Remaining source link: {node.path}.{parameter_name}'
finally:
	run('args[0].destroy()', container, delayFrames=2)
assert before == {name: demo.par[name.capitalize()].eval() for name in names}, 'Export changed live controls'
assert source == list(demo.inputs), 'Export changed live input'
print(json.dumps({'demo': demo.path, 'wrappers': names, 'layer_order_front_to_back': order,
	'legacy_native_sources': native_sources, 'legacy_references': references,
	'controls_preserved': True, 'inactive_models_stayed_idle': True,
	'pixel_checks': pixel_checks, 'export': 'tox/ModelDemo.tox', 'export_default_off_and_embedded': True,
	'debug_output': demo.op('out_debug').path}, indent=2))