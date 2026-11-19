"""Build /project1/Demos/PersonSplitDemo (run inside TD through the bridge's /run): YOLO26
Segmentation -> PersonSplit, and three ways to use the split.

  Line-up    every person's cutout on a card, side by side, one card per slot: Geometry COMP
             instancing, the textures from PersonSplit's Cutouts (a 2D texture array, layer =
             slot - 1), the cards from its slots CHOP (`present` hides an empty slot).
  Spotlight  one person (PersonSplit's Person output, Pick Slot) sharp over the dimmed, blurred frame.
  Preview    PersonSplit's own view: each person tinted in their track color.

A public example (tox/demos/PersonSplitDemo.tox, tracked). After this first build, edit it in TD like
any demo; rerunning this script rebuilds it from scratch.

	curl -X POST --data-binary @scripts/lab/build_person_split_demo.py http://127.0.0.1:3031/run
"""
import importlib
import sys
from pathlib import Path

PARENT = '/project1/Demos'
NAME = 'PersonSplitDemo'
COMPONENT = '/project1/Components/PersonSplit'

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder	# noqa: E402
importlib.reload(wrapper_builder)

LINEUP = '''# Line-up: one card per PersonSplit slot, in a row; empty slots get scale 0.
def onCook(scriptOp):
	slots = scriptOp.inputs[0] if scriptOp.inputs else None	# PersonSplit's slots CHOP
	n = slots.numSamples if slots is not None else 0
	spacing = float(parent().par.Cardspacing)
	scriptOp.clear()
	scriptOp.numSamples = n
	tx, scale, index = (scriptOp.appendChan(name) for name in ('tx', 'scale', 'texindex'))
	for i in range(n):
		tx[i] = (i - (n - 1) / 2) * spacing
		scale[i] = slots['present'][i]
		index[i] = i
	return
'''


def connector(comp, name, outputs=False):
	for c in (comp.outputConnectors if outputs else comp.inputConnectors):
		if (c.outOP if outputs else c.inOP).name == name:
			return c
	raise KeyError(f'{comp.path}: no connector for {name}')


def wire(source, out_name, destination, in_name):
	connector(destination, in_name).connect(connector(source, out_name, outputs=True))


def build():
	parent_comp = op(PARENT)
	position, inputs, outputs = wrapper_builder._take_place(parent_comp, NAME)
	demo = parent_comp.create('baseCOMP', NAME)
	demo.nodeX, demo.nodeY = position if position is not None else (300, -400)

	page = demo.appendCustomPage('Demo')
	view = page.appendMenu('View', label='View')[0]
	view.menuNames = ['lineup', 'spotlight', 'preview']
	view.menuLabels = ['Line-up', 'Spotlight', 'Preview']
	view.val = view.default = 'lineup'
	pick = page.appendInt('Pickslot', label='Spotlight Slot')[0]
	pick.val = pick.default = 1
	pick.min = pick.normMin = 1
	pick.clampMin = True
	pick.normMax = 6
	spacing = page.appendFloat('Cardspacing', label='Card Spacing')[0]
	spacing.val = spacing.default = 1.1
	about = demo.appendCustomPage('About')
	release = about.appendStr('Version', label='Version')[0]
	release.val = release.default = wrapper_builder.version()
	release.readOnly = True

	source = demo.create('inTOP', 'input')
	seg = wrapper_builder.build(wrapper_builder.load_spec('Yolo26Seg'), demo)
	seg.par.Matteencoding = 'instances'
	seg.par.Matteconf.expr = 'me.par.Confthreshold'	# every tracked person in the matte, or their slot has no pixels
	seg.inputConnectors[0].connect(source)
	split = demo.copy(op(COMPONENT))
	split.par.externaltox = 'tox/components/PersonSplit.tox'
	split.par.Pickslot.expr = "parent().par.Pickslot"
	wire(seg, 'debug', split, 'in_matte')	# TOP 1: the instance-encoded matte (result = inference)
	wire(seg, 'passthrough', split, 'in_image')
	wire(seg, 'out_tracks', split, 'in_tracks')

	nodes = {}
	for out_name in ('out_preview', 'out_person', 'out_cutouts', 'out_slots'):
		kind = 'nullCHOP' if out_name == 'out_slots' else 'nullTOP'
		node = demo.create(kind, out_name.replace('out_', ''))
		node.inputConnectors[0].connect(connector(split, out_name, outputs=True))
		nodes[node.name] = node

	# Line-up
	lineup = demo.create('scriptCHOP', 'lineup')
	callbacks = demo.op('lineup_callbacks') or demo.create('textDAT', 'lineup_callbacks')	# TD makes one with the Script CHOP
	callbacks.text = LINEUP
	lineup.par.callbacks = callbacks.name
	lineup.inputConnectors[0].connect(nodes['slots'])
	geo = demo.create('geometryCOMP', 'cards')
	for child in list(geo.children):
		child.destroy()
	card = geo.create('rectangleSOP', 'card')
	card.render = card.display = True
	geo.par.instancing = True
	geo.par.instanceop = lineup
	geo.par.instancetx = 'tx'
	geo.par.instancesx = 'scale'
	geo.par.instancesy = 'scale'
	geo.par.instancetexmode = 'transform'	# W picks the array layer (= slot - 1)
	geo.par.instancew = 'texindex'
	material = demo.create('constantMAT', 'card_mat')
	material.par.colormap = nodes['cutouts']
	material.par.blending = True
	geo.par.material = material
	camera = demo.create('cameraCOMP', 'lineup_camera')
	camera.par.projection = 'ortho'
	camera.par.orthowidth.expr = "max(3.0, op('slots').numSamples * parent().par.Cardspacing)"
	camera.par.tz = 5
	render = demo.create('renderTOP', 'lineup_render')
	render.par.camera = camera.name
	render.par.geometry = geo.name
	render.par.lights = ''
	render.par.outputresolution = 'custom'
	render.par.resolutionw.expr = "op('input').width"
	render.par.resolutionh.expr = "op('input').width * 0.45"
	render.par.bgcolora = 0
	backdrop = demo.create('constantTOP', 'lineup_backdrop')
	backdrop.par.colorr, backdrop.par.colorg, backdrop.par.colorb = 0.08, 0.08, 0.1
	backdrop.par.outputresolution = 'custom'
	backdrop.par.resolutionw.expr = "op('lineup_render').width"
	backdrop.par.resolutionh.expr = "op('lineup_render').height"
	lineup_over = demo.create('overTOP', 'lineup_view')
	lineup_over.inputConnectors[0].connect(render)
	lineup_over.inputConnectors[1].connect(backdrop)

	# Spotlight
	blur = demo.create('blurTOP', 'spot_blur')
	blur.par.size = 24
	blur.inputConnectors[0].connect(source)
	dim = demo.create('levelTOP', 'spot_dim')
	dim.par.brightness1 = 0.35
	dim.inputConnectors[0].connect(blur)
	spot = demo.create('overTOP', 'spotlight_view')
	spot.inputConnectors[0].connect(nodes['person'])
	spot.inputConnectors[1].connect(dim)

	switch = demo.create('switchTOP', 'view')
	for i, node in enumerate((lineup_over, spot, nodes['preview'])):
		switch.inputConnectors[i].connect(node)
	switch.par.index.expr = "parent().par.View.menuIndex"
	out = demo.create('outTOP', 'out_view')
	out.inputConnectors[0].connect(switch)
	slots_out = demo.create('outCHOP', 'out_slots')
	slots_out.inputConnectors[0].connect(nodes['slots'])

	grid = {'input': (0, 0), seg.name: (150, 0), split.name: (300, 0), 'preview': (450, 100), 'person': (450, 0),
		'cutouts': (450, -100), 'slots': (450, -200), 'lineup': (600, -200), 'lineup_callbacks': (600, -300),
		'card_mat': (600, -100), 'cards': (750, -200), 'lineup_camera': (750, -300), 'lineup_render': (900, -200),
		'lineup_backdrop': (900, -300), 'lineup_view': (1050, -200), 'spot_blur': (600, 100), 'spot_dim': (750, 100),
		'spotlight_view': (900, 50), 'view': (1200, 0), 'out_view': (1350, 0), 'out_slots': (1350, -200)}
	for name, (x, y) in grid.items():
		demo.op(name).nodeX, demo.op(name).nodeY = x, y
	wrapper_builder._restore_wires(demo, inputs, outputs)
	if not demo.inputs and op(PARENT).op('camera') is not None:
		demo.inputConnectors[0].connect(op(PARENT).op('camera'))
	return demo


existing = op(PARENT).op(NAME)
running = wrapper_builder.demo_wrappers(existing) if existing is not None else []
if running:	# destroying a running ONNX wrapper crashes TD: unload them first, rebuild on the next run
	for wrapper in running:
		wrapper_builder.remove(wrapper)
	print('removing', [w.name for w in running], '- run again in 2 s')
else:
	demo = build()
	print('built', demo.path, demo.errors(recurse=True)[:400])
