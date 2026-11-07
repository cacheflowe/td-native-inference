"""Route YOLO segmentation's native matte to its primary result output."""

import json
from pathlib import Path


demo = op('/project1/ModelDemo') or op('/project1/ONNX_Native/ModelDemo')
assert demo is not None, 'ModelDemo not found'
wrapper = demo.op('Yolo26Seg')
controls = {name: demo.par[name.capitalize()].eval() for name in demo.fetch('completed_wrappers')}
source = list(demo.inputs)
counts = {name: demo.op(name + '/inference').totalCooks for name, active in controls.items() if not active}
demo.op('component').module.setup_segmentation_output(wrapper)
order = demo.op('component').module.setup_demo_composite(demo)
assert wrapper.op('preview').inputs[1] == wrapper.op('inference')
assert wrapper.op('null_passthrough').inputs[0] == wrapper.op('input')
assert wrapper.op('out_boxes').inputs[0] == wrapper.op('overlay')
assert wrapper.op('out_boxes').par.connectorder.eval() == 3
assert controls == {name: demo.par[name.capitalize()].eval() for name in controls}
assert source == list(demo.inputs)
demo.op('out_debug').cook(force=True)
assert counts == {name: demo.op(name + '/inference').totalCooks for name in counts}
container = demo.parent().create('baseCOMP', '_segmentation_output_validation')
try:
	driver = container.create('textDAT', 'composite_driver')
	driver.text = (Path(project.folder) / 'tests/td_model_wrappers.py').read_text(encoding='utf-8')
	pixel_checks = driver.module.check_segmentation_output(demo)
	helper = demo.op('component').module
	subject = container.copy(wrapper, name='Yolo26Seg')
	subject.par.Active.bindExpr = ''
	subject.par.Active = False
	subject.par.Packageroot = project.folder
	subject.allowCooking = True
	helper.export(subject, Path(project.folder) / 'tox/Yolo26Seg.tox')
	helper.export_demo(demo, Path(project.folder) / 'tox/ModelDemo.tox')
	assert controls == {name: demo.par[name.capitalize()].eval() for name in controls}
	assert source == list(demo.inputs)
finally:
	run('args[0].destroy()', container, delayFrames=2)
print(json.dumps({'primary_output': 'native segmentation matte', 'second_output': 'input image',
	'third_output': 'bounding boxes', 'data_preserved': True, 'controls_preserved': True,
	'inactive_models_stayed_idle': True, 'layer_order': order, 'pixel_checks': pixel_checks,
	'exports': ['tox/Yolo26Seg.tox', 'tox/ModelDemo.tox']}, indent=2))