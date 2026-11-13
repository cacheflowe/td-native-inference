"""Runtime support embedded in MixamoPuppet (its `component` DAT): importing the character FBX,
binding its bones to the solver's channels, and the render's geometry. Building and exporting
live in puppet_builder.py.

The tox ships without a character (Mixamo's license allows use, not redistribution): Load
Character imports the FBX File par's file inside `fbx`, captures the rest pose from the bones'
constant values, then binds every bone's rotation (and the COMP's translate) to `rotations`.
"""

import os

import td	# imported modules don't get TD's globals (project, run)

MAX_RETRIES = 10


def _status(comp, text):
	comp.par.Status = text


def _bones(fbx):
	return [c for c in fbx.findChildren(type=td.nullCOMP) if c.OPType == 'nullCOMP']


def load_character(comp, attempt=0):
	"""(Re)import the FBX File and bind it. Safe to call any time; with no file it just clears."""
	fbx, render = comp.op('fbx'), comp.op('render')
	path = comp.par.Fbxfile.eval()
	if not path:
		fbx.par.file = ''
		render.par.geometry = ''
		_status(comp, 'Set FBX File to a Mixamo character')
		return
	full = path if os.path.isabs(path) else os.path.join(td.project.folder, path)
	if not os.path.exists(full):
		render.par.geometry = ''
		_status(comp, f'FBX not found: {path}')
		return
	if attempt == 0:
		_status(comp, 'Importing...')
		fbx.par.file = path
		fbx.par.imp.pulse()
	if not _bones(fbx):
		if attempt >= MAX_RETRIES:
			_status(comp, 'The FBX has no skeleton')
			return
		td.run("args[0].module.load_character(parent(), args[1])", comp.op('component'), attempt + 1, delayFrames=3)
		return
	bind(comp)


def _expr(par, text):
	par.expr = text
	par.mode = type(par.mode).EXPRESSION


REST_COLUMNS = ['name', 'parent', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 's']


def rest_rows(fbx):
	"""The rig's rest pose from the bones' constant values (.val: binding them to the solver leaves
	those), as the Puppet CHOP's Rest Pose table: a bone's parent is its wired input COMP, else the
	bone COMP holding it (a Blender Armature); s is its uniform scale."""
	rows = [REST_COLUMNS]
	for bone in _bones(fbx) if fbx else ():
		holder = bone.parent()
		parent = bone.inputCOMPs[0] if bone.inputCOMPs else (holder if holder != fbx and holder.OPType == 'nullCOMP' else None)
		p = bone.par
		values = (p.tx.val, p.ty.val, p.tz.val, p.rx.val, p.ry.val, p.rz.val, p.sx.val * p.scale.val)
		rows.append([bone.name, parent.name if parent else ''] + [repr(float(v)) for v in values])
	return rows


def capture_rest(comp):
	"""Re-read the character's rest pose into `rest_pose`, the Puppet CHOP's rig."""
	table = comp.op('rest_pose')
	table.clear()
	for row in rest_rows(comp.op('fbx')):
		table.appendRow(row)


def bind(comp):
	fbx, solver, render = comp.op('fbx'), comp.op('solver'), comp.op('render')
	# The import exports the FBX's own animation onto some bones, which would override the binding.
	for chop in fbx.findChildren(type=td.CHOP):
		chop.export = False
	capture_rest(comp)	# from the bones' constant values: before they're bound
	solver.cook(force=True)
	if not solver.numChans:
		_status(comp, 'No Mixamo skeleton found in that FBX (needs mixamorig_ bones)')
		return
	depth = len(comp.path.split('/'))
	bones = _bones(fbx)
	for bone in bones:
		up = '../' * (len(bone.path.split('/')) - depth - 1)	# from the bone's network up to this COMP
		for axis in ('rx', 'ry', 'rz'):
			_expr(bone.par[axis], f"op('{up}rotations')['{bone.name}_{axis}']")
	for axis in ('tx', 'ty', 'tz'):
		_expr(fbx.par[axis], f"op('rotations')['root_{axis}']")
	fbx.par.sx.mode = type(fbx.par.sx.mode).CONSTANT
	fbx.par.sx = 1
	meshes = [c.path for c in fbx.children if c.OPType == 'geometryCOMP']
	render.par.geometry = ' '.join([fbx.path] + meshes)
	_status(comp, f'{len(bones)} bones bound')
