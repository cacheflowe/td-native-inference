"""Build tox/HandsGeometry.tox from the lab's authored /project1/PuppetLab/HandsGeometry (joints DAT
in -> sphere / cylinder hand geometry as a POP), adding a render rig so the tox shows the hands on
its own. Run inside TD:

	curl -X POST --data-binary @scripts/build_hands_tox.py http://127.0.0.1:3031/run

The lab COMP stays the source of the geometry network (edit it there, then rebuild). This script
adds, on a copy: Render page (Resolution, Camera Distance, Field of View), a Geometry COMP that
renders the POP, camera, light and Render TOP, and the outputs: TOP out_render (transparent) and
POP out_hands. The project is not saved."""

import importlib
import sys
from pathlib import Path

LAB = '/project1/PuppetLab'
SOURCE = 'HandsGeometry'
NAME = 'HandsGeometry'
HOLDER = '_hands_tox_build'
EXPORT = True

python_dir = (Path(project.folder) / 'python').as_posix()
if python_dir not in sys.path:
	sys.path.insert(0, python_dir)
import wrapper_builder
importlib.reload(wrapper_builder)

README = """Hand geometry

Input: the joint DAT of a MediaPipe Hands tox (out_joints: track_id, name, x, y, z, conf, r, g, b).
Each tracked joint becomes a sphere and the 21 hand connections become cylinders (a palm edge joins
the thumb base to the index knuckle). The two branches merge into out_hands, the POP output, and
out_render is that geometry rendered with a camera and light on a transparent background. Tracking
loss produces no geometry. Joints are grouped by track_id, so separate hands never connect.

Coordinates: image-normalized X/Y centred on zero, Y up. Aspect Ratio adjusts X and Z relative to Y
(set it to the source image width / height). Z is relative landmark depth, flipped so closer points
are positive; it is not metric depth, and Depth Scale changes its influence. Joint Radius and Bone
Radius set the mesh thickness. Colors come from the joint table.

Needs a TouchDesigner build with POPs (2025 and later).
"""


def expr(par, text):
	par.expr = text
	par.mode = ParMode.EXPRESSION


lab = op(LAB)
holder = lab.op(HOLDER) or lab.create(baseCOMP, HOLDER)
for child in list(holder.children):
	child.destroy()
comp = holder.copyOPs([lab.op(SOURCE)])[0]
comp.name = NAME

# One input, clearly named.
joints_in = comp.op('in2')
joints_in.name = 'in_joints'
joints_in.par.connectorder = 1
comp.op('readMe').text = README

# Render rig. The Geometry COMP draws whatever carries its render flag: a Select POP on the output.
geo = comp.create(geometryCOMP, 'hands_geo')
for child in list(geo.children):
	child.destroy()
select = geo.create(selectPOP, 'select_hands')
select.par.pop = '../out_hands'
select.render = True
select.display = True
camera = comp.create(cameraCOMP, 'camera')
camera.par.tx = camera.par.ty = 0
light = comp.create(lightCOMP, 'light')
light.par.tx, light.par.ty, light.par.tz = 1, 2, 3
render = comp.create(renderTOP, 'render')
render.par.geometry = 'hands_geo'
render.par.camera = 'camera'
render.par.lights = 'light'
render.par.outputresolution = 'custom'
render.par.bgcolora = 0
out = comp.create(outTOP, 'out_render')
out.inputConnectors[0].connect(render)

# Render page.
page = comp.appendCustomPage('Render')
wh = page.appendWH('Resolution', label='Resolution')
wh[0].default = wh[0].val = 1024
wh[1].default = wh[1].val = 576
distance = page.appendFloat('Cameradistance', label='Camera Distance')[0]
distance.default = distance.val = 3.0
distance.normMin, distance.normMax = 1.0, 10.0
distance.help = 'How far the camera sits from the hands (the geometry is about 1 unit across).'
fov = page.appendFloat('Fov', label='Field of View')[0]
fov.default = fov.val = 40.0
fov.normMin, fov.normMax = 10.0, 90.0
expr(camera.par.tz, 'parent().par.Cameradistance')
expr(camera.par.fov, 'parent().par.Fov')
expr(render.par.resolutionw, 'parent().par.Resolutionw')
expr(render.par.resolutionh, 'parent().par.Resolutionh')
about = comp.appendCustomPage('About')
version = about.appendStr('Version', label='Version')[0]
version.val = version.default = wrapper_builder.version()
version.readOnly = True

# Layout: source network on the left, render rig to the right of it.
for name, (x, y) in {'hands_geo': (1000, 300), 'camera': (1000, 150), 'light': (1000, 0), 'render': (1200, 200), 'out_render': (1400, 200)}.items():
	comp.op(name).nodeX, comp.op(name).nodeY = x, y
for node in comp.findChildren():
	node.viewer = False
comp.op('out_render').viewer = False
wrapper_builder._add_icon(comp, NAME, './render')
comp.allowCooking = True

report = {'built': comp.path, 'children': len(comp.children), 'errors': {c.path: c.errors() for c in comp.findChildren() if c.errors()}}
if EXPORT:
	destination = Path(project.folder) / 'tox' / f'{NAME}.tox'
	wrapper_builder.export(comp, destination)
	report['export_problems'] = wrapper_builder.check_export(destination)
print(report)
