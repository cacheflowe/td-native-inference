"""Tracking adapter for the C++ OnnxTOP: Info DAT in, the Python scripts' tracked tables out.

OnnxTOP does inference + YOLO decode (+ seg masks) on the GPU and publishes raw per-frame
detections in its Info DAT (see cpp/OnnxTOP/README.md): header row, then one row per detection:
	id, class, conf, u1, v1, u2, v2[, mask_fill (seg)][, kp{i}_u, kp{i}_v, kp{i}_c ... (pose)]
in TD UV space (0-1, origin bottom-left). This module turns those into the detection dicts the
shared pipelines expect and runs the SAME tracker the Python scripts use:

	Tracker Mode 'pose'    -> pose_tracker_pipeline.PoseTrackerPipeline  (onnx_yolo26_pose.py)
	Tracker Mode 'movenet' -> pose_tracker_pipeline.PoseTrackerPipeline  (MoveNet's tuned defaults)
	Tracker Mode 'face'    -> face_tracker_pipeline.FaceTrackerPipeline  (onnx_yunet.py)
	Tracker Mode 'detect'  -> box_tracker_pipeline.BoxTrackerPipeline    (onnx_yolo26_obj_det.py)
	Tracker Mode 'segment' -> box_tracker_pipeline.BoxTrackerPipeline    (onnx_yolo26_seg.py)
	Tracker Mode 'mp_pose' / 'mp_hands' / 'mp_face'
	                       -> pose_tracker_pipeline.PoseTrackerPipeline with a MediaPipe skeleton
	                          (mediapipe_skeletons), reading a MediaPipeTOP Info DAT

so DebugBB / DebugKeypoints work unchanged and get stable track ids.

Used from the OnnxTopTables COMP (one pipeline per COMP, kept here across cooks):
	script_tracks (input: Info DAT)      -> onnxtop_tables.write_tracks(scriptOp, parent(), scriptOp.inputs[0])
	script_joints (input: script_tracks) -> onnxtop_tables.write_joints(scriptOp, parent())
	script_bones  (input: script_tracks) -> onnxtop_tables.write_bones(scriptOp, parent())
	parexec (Trackermode changes)        -> onnxtop_tables.setup_parameters(parent())
script_tracks is the only one that advances the tracker; joints/bones are wired after it so
they cook after it and just format the same frame's state (joints: pose + face modes; bones: pose only; header-only otherwise).

Optional second stage (e.g. HSEmotion on YuNet faces): a Classify Crops OnnxTOP reads
script_tracks as its Crop DAT, and its Info DAT comes back in as the COMP's 2nd input:
	script_scores (inputs: script_tracks, in_scores) -> onnxtop_tables.write_scores(scriptOp, parent(),
	                                                     scriptOp.inputs[0], scriptOp.inputs[1])
which joins the per-track class scores (smoothed) onto the tracks table for out_tracks. The
classifier reads script_tracks, not out_tracks, so there's no dependency loop.
"""
import numpy as np

import re

import box_tracker_pipeline
import face_tracker_pipeline
import mediapipe_skeletons
import object_tracker
import pose_tracker_pipeline

MODES = ['pose', 'movenet', 'face', 'detect', 'segment', 'mp_pose', 'mp_hands', 'mp_face', 'mp_holistic', 'mp_facedet']
MODE_LABELS = ['Pose (YOLO26-pose)', 'Pose (MoveNet multipose)', 'Face (YuNet)', 'Detect (YOLO26)', 'Segment (YOLO26-seg)',
	'MediaPipe Pose (33)', 'MediaPipe Hands / Gesture (21)', 'MediaPipe Face (478)', 'MediaPipe Holistic (553)',
	'MediaPipe Face Detector (6)']
# MediaPipeTOP Task that produces each MediaPipe mode's Info DAT.
MP_SKELETONS = {
	'mp_pose': mediapipe_skeletons.MP_POSE,
	'mp_hands': mediapipe_skeletons.MP_HANDS,
	'mp_face': mediapipe_skeletons.MP_FACE,
	'mp_holistic': mediapipe_skeletons.MP_HOLISTIC,
	'mp_facedet': mediapipe_skeletons.MP_FACE_DET,
}
MP_TASKS = {'mp_pose': 'Pose', 'mp_hands': 'Hands or Gesture', 'mp_face': 'Face', 'mp_holistic': 'Holistic',
	'mp_facedet': 'Face Detector'}
POSE_MODES = ('pose', 'movenet', *MP_SKELETONS)

# MediaPipe already detects, tracks and smooths, so the Python side mostly assigns ids:
# no extra smoothing, confirm on the first frame, and normal 0-1 confidences (face = 1, hands =
# handedness score, pose = mean visibility).
MEDIAPIPE_DEFAULTS = {
	'Confthreshold': 0.1,
	'Outputsmoothing': 0.0,
	'Trackconfirmframes': 1,
	# A lost MediaPipe track means MediaPipe itself dropped it (e.g. num_poses=2 finding only one
	# of two dancers, alternating per frame), so only bridge a couple of frames: the 15-frame
	# YOLO hold left frozen "ghost" skeletons on screen.
	'Tracklossframes': 3,
	'Minboxwidth': 0.01,
	'Minboxheight': 0.01,
}

# Info DAT columns that aren't keypoints or extras (MediaPipeTOP adds world landmarks w{i}_* and
# the face matrix m{r}{c}; anything else, e.g. blendshape names, becomes a tracks-table extra).
_KNOWN_COLS = {'id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label', 'mask_fill'}
_NOT_EXTRA = re.compile(r'^(kp\d+_[uvcz]|w\d+_[xyz]|m\d\d)$')


# Hands keep the last position longer and glide through dropouts: MediaPipe loses a fast hand
# for a frame or two, and a hand doesn't swap between people the way Pose (Max 2) does.
MEDIAPIPE_HANDS_DEFAULTS = {
	'Tracklossframes': 6,
	'Coastwhilelost': 0.8,
	'Presencefall': 6,
}


def pose_defaults(mode):
	if mode == 'movenet':
		return pose_tracker_pipeline.MOVENET_DEFAULTS
	if mode in MP_SKELETONS:
		d = dict(MEDIAPIPE_DEFAULTS, Maxmatchdist=MP_SKELETONS[mode].max_match_dist)
		if mode == 'mp_hands':
			d.update(MEDIAPIPE_HANDS_DEFAULTS)
		return d
	return None


def pose_extra_parameters(mode):
	"""Par lists added to PoseTrackerPipeline's Tracking page for this mode."""
	if mode == 'mp_hands':
		return (*pose_tracker_pipeline.PERSISTENCE_PARAMETERS, *pose_tracker_pipeline.LABEL_PARAMETERS)
	if mode in MP_SKELETONS:
		return tuple(pose_tracker_pipeline.PERSISTENCE_PARAMETERS)
	return ()


def expected_keypoints(mode):
	return MP_SKELETONS[mode].count if mode in MP_SKELETONS else pose_tracker_pipeline.NUM_KEYPOINTS
TRACKING_PAGE = 'Tracking'
STORE_KEY = 'onnxtop_tracking_values'		# {mode: {par name: value}}
STORE_MODE_KEY = 'onnxtop_tracking_mode'	# mode the Tracking page was last built for

# OnnxTopTables COMP path -> (mode, pipeline). Module-level so tracker state survives cooks;
# a module reload (editing this file) simply restarts tracking.
_pipelines = {}


def get_mode(comp):
	par = getattr(comp.par, 'Trackermode', None)
	return par.eval() if par is not None else 'pose'


def get_pipeline(comp):
	mode = get_mode(comp)
	entry = _pipelines.get(comp.path)
	if entry is None or entry[0] != mode:
		if mode in POSE_MODES:
			pipeline = pose_tracker_pipeline.PoseTrackerPipeline(skeleton=MP_SKELETONS.get(mode))
		elif mode == 'face':
			pipeline = face_tracker_pipeline.FaceTrackerPipeline()
		else:
			pipeline = box_tracker_pipeline.BoxTrackerPipeline(segmentation=mode == 'segment')
		entry = (mode, pipeline)
		_pipelines[comp.path] = entry
	return entry[1]


def reset(comp):
	"""Drop tracker state (e.g. after a mode change)."""
	_pipelines.pop(comp.path, None)


def setup_parameters(comp):
	"""Create the Setup page (Tracker Mode) if needed, and rebuild the Tracking page for the
	current mode. Safe to call repeatedly; call it when Trackermode changes."""
	setup = next((p for p in comp.customPages if p.name == 'Setup'), None) or comp.appendCustomPage('Setup')
	if getattr(comp.par, 'Trackermode', None) is None:
		p = setup.appendMenu('Trackermode', label='Tracker Mode')
		p[0].menuNames = MODES
		p[0].menuLabels = MODE_LABELS
		p[0].default = 'pose'
		p[0].help = ("Which shared Python tracker to run on OnnxTOP's detections. Must match the model: "
			"pose models need Pose (YOLO26-pose) or MoveNet, YuNet needs Face, seg models (Info DAT has a mask_fill "
			"column) need Segment. MediaPipe modes read a MediaPipeTOP Info DAT (Task Pose / Hands or Gesture / Face).")
	else:
		p = comp.par.Trackermode
		p.menuNames = MODES	# older COMPs: pick up modes added since
		p.menuLabels = MODE_LABELS
	if getattr(comp.par, 'Classprefix', None) is None:
		p = setup.appendStr('Classprefix', label='Class Column Prefix')
		p[0].default = 'class'
		p[0].val = 'class'
		p[0].help = ("Only used with a classifier on the 2nd input (see write_scores): columns become "
			"<prefix>_label and <prefix>_<class name>. 'emotion' matches onnx_hsemotion.py's table.")
	if getattr(comp.par, 'Classsmoothing', None) is None:
		p = setup.appendFloat('Classsmoothing', label='Class Score Smoothing')
		p[0].default = 0.5
		p[0].val = 0.5
		p[0].min, p[0].max = 0.0, 0.99
		p[0].clampMin = p[0].clampMax = True
		p[0].help = ("Per-track lerp on classifier scores (0 = raw, higher = steadier), like "
			"onnx_hsemotion.py's Outputsmoothing on emotion scores.")
	mode = get_mode(comp)
	# Pars share names across modes but not meaning (pose Confthreshold ~0.005, seg 0.35), so
	# each mode's tuned values are stored per mode and restored when switching back.
	saved = dict(comp.fetch(STORE_KEY, {}, search=False))
	page = next((p for p in comp.customPages if p.name == TRACKING_PAGE), None)
	if page is not None:
		# No record yet = a page from before modes existed, built for the current mode.
		built_mode = comp.fetch(STORE_MODE_KEY, None, search=False) or mode
		saved[built_mode] = {par.name: par.eval() for par in page.pars}
		page.destroy()
	page = comp.appendCustomPage(TRACKING_PAGE)
	if mode in POSE_MODES:
		pose_tracker_pipeline.append_parameters(page, comp, defaults=pose_defaults(mode), extra=pose_extra_parameters(mode))
	elif mode == 'face':
		face_tracker_pipeline.append_parameters(page, comp)
	else:
		box_tracker_pipeline.append_parameters(page, comp, segmentation=mode == 'segment')
	for name, value in saved.get(mode, {}).items():
		par = getattr(comp.par, name, None)
		if par is not None:
			par.val = value
	comp.store(STORE_KEY, saved)
	comp.store(STORE_MODE_KEY, mode)
	reset(comp)


def _columns(info_dat):
	header = [c.val for c in info_dat.row(0)]
	return {name: i for i, name in enumerate(header)}, header


def read_detections(info_dat, mode):
	"""Info DAT rows -> pipeline detection dicts (box = [x_left, y_bottom, x_right, y_top])."""
	if info_dat is None or info_dat.numRows < 2:
		return []
	col, header = _columns(info_dat)
	num_kp = sum(1 for h in header if h.startswith('kp') and h.endswith('_u'))
	has_fill = 'mask_fill' in col
	mp = mode in MP_SKELETONS
	has_z = mp and 'kp0_z' in col
	label_col = col.get('label', -1)
	extra_cols = [(h, i) for i, h in enumerate(header) if h not in _KNOWN_COLS and not _NOT_EXTRA.match(h)] if mp else []
	dets = []
	for r in range(1, info_dat.numRows):
		row = info_dat.row(r)
		det = {
			'box': [float(row[col[c]].val) for c in ('u1', 'v1', 'u2', 'v2')],
			'score': float(row[col['conf']].val),
		}
		if mode in POSE_MODES:
			kps = []
			for k in range(num_kp):
				base = col[f'kp{k}_u']
				kp = [float(row[base].val), float(row[base + 1].val), float(row[base + 2].val)]
				if has_z:
					kp.append(float(row[col[f'kp{k}_z']].val))
				kps.append(kp)
			det['keypoints'] = kps
			if mp:
				if label_col >= 0 and row[label_col].val:
					det['label'] = row[label_col].val
				extras = {name: float(row[i].val) for name, i in extra_cols}
				if mode == 'mp_hands':
					extras['handedness'] = float(row[col['class']].val)	# 0 = Left, 1 = Right
				det['extras'] = extras
		elif mode == 'face':
			det['keypoints'] = [[float(row[col[f'kp{k}_u']].val), float(row[col[f'kp{k}_v']].val)] for k in range(num_kp)]
		else:
			class_id = int(row[col['class']].val)
			det['class_id'] = class_id
			det['class_name'] = box_tracker_pipeline.COCO_CLASSES.get(class_id, 'unknown')
			if has_fill:
				det['mask_area_ratio'] = float(row[col['mask_fill']].val)
		dets.append(det)
	return dets


def _mode_error(info_dat, mode):
	"""Explain a Tracker Mode / model mismatch, or None if they agree."""
	if info_dat is None or info_dat.numRows < 1:
		return None
	col, header = _columns(info_dat)
	num_kp = sum(1 for h in header if h.startswith('kp') and h.endswith('_u'))
	if mode in MP_SKELETONS and num_kp != expected_keypoints(mode):
		return (f"Tracker Mode {mode} needs a MediaPipeTOP Info DAT with {expected_keypoints(mode)} keypoints, got {num_kp} "
			f"-- set the MediaPipeTOP's Task to {MP_TASKS[mode]}")
	if mode in POSE_MODES and num_kp != expected_keypoints(mode) and info_dat.numRows > 1:
		return f"Tracker Mode is Pose but the model reports {num_kp} keypoints (needs 17) -- set Tracker Mode to Face (5 keypoints), Detect or Segment"
	if mode == 'segment' and 'mask_fill' not in col:
		return "Tracker Mode is Segment but the Info DAT has no mask_fill column (not a seg model) -- set Tracker Mode to Detect or Pose"
	if mode == 'face' and num_kp != face_tracker_pipeline.NUM_KEYPOINTS:
		return f"Tracker Mode is Face but the model reports {num_kp} keypoints (needs 5) -- set the node's Output Mode to YuNet Faces"
	if mode in ('detect', 'segment') and num_kp:
		return "This model has keypoints -- set Tracker Mode to Pose, MoveNet or Face"
	return None


def _frame_size(comp):
	"""Source frame size (for solvePnP head pose): the COMP's null_passthrough TOP (the
	visualizer contract), else the input of the OnnxTOP its Info DAT reads, else its
	Inputaspectratio par, else square."""
	npt = comp.op('null_passthrough')
	if npt is not None and npt.height > 0:
		return npt.width, npt.height
	src = comp.inputs[0] if comp.inputs else None	# Info DAT wired into the COMP
	node = getattr(src.par, 'op', None) if src is not None else None
	node = node.eval() if node is not None else None
	if node is not None and getattr(node, 'inputs', None) and node.inputs[0].height > 0:
		return node.inputs[0].width, node.inputs[0].height
	ratio = getattr(comp.par, 'Inputaspectratio', None)
	return ((ratio.eval() or 1.0) * 1000.0, 1000.0) if ratio is not None else (1000.0, 1000.0)


def write_tracks(out, comp, info_dat):
	"""Advance the tracker one frame from the Info DAT and write table_output."""
	mode = get_mode(comp)
	err = _mode_error(info_dat, mode)
	if err:
		out.clear()
		out.appendRow(['error'])
		out.appendRow([err])
		return
	pipeline = get_pipeline(comp)
	dets = read_detections(info_dat, mode)
	if mode in POSE_MODES:
		extra = pose_extra_parameters(mode)
		p = pose_tracker_pipeline.read_parameters(comp, defaults=pose_defaults(mode), extra=extra)
		dets = pose_tracker_pipeline.filter_detections(
			dets, p['Confthreshold'], p['Minboxwidth'], p['Minboxheight'])
		persist = {}
		if extra:
			persist = {'coast': p['Coastwhilelost'], 'presence': (p['Presencerise'], p['Presencefall'])}
		if mode == 'mp_hands':
			persist.update(label_hold=int(p['Gestureholdframes']), label_min_score=p['Gestureminscore'])
		pipeline.update(
			dets,
			conf_threshold=p['Confthreshold'],
			smoothing=p['Outputsmoothing'],
			max_match_dist=p['Maxmatchdist'],
			track_buffer=p['Tracklossframes'],
			min_hits=p['Trackconfirmframes'],
			**persist,
		)
	elif mode == 'face':
		p = face_tracker_pipeline.read_parameters(comp)
		dets = face_tracker_pipeline.filter_detections(
			dets, p['Lowconfthreshold'], p['Minboxwidth'], p['Minboxheight'])
		frame_w, frame_h = _frame_size(comp)
		pipeline.update(
			dets,
			conf_threshold=p['Confthreshold'],
			low_conf_threshold=p['Lowconfthreshold'],
			track_iou_threshold=p['Trackiouthreshold'],
			track_buffer=p['Tracklossframes'],
			min_hits=p['Trackconfirmframes'],
			smoothing=p['Outputsmoothing'],
			pose_method=p['Posemethod'],
			yaw_scale=p['Poseyawscale'],
			pitch_scale=p['Posepitchscale'],
			focal_scale=p['Posefocalscale'],
			pose_smoothing=p['Posesmoothing'],
			frame_w=frame_w,
			frame_h=frame_h,
			size_aspect=1.0,	# OnnxTOP letterboxes, so boxes keep their true aspect
		)
	else:
		seg = mode == 'segment'
		p = box_tracker_pipeline.read_parameters(comp, segmentation=seg)
		dets = box_tracker_pipeline.filter_detections(
			dets, p['Lowconfthreshold'], p['Minboxwidth'], p['Minboxheight'], p['Nmsiouthreshold'],
			min_mask_area=p['Minmaskarea'] if seg else None)
		pipeline.update(
			dets,
			conf_threshold=p['Confthreshold'],
			low_conf_threshold=p['Lowconfthreshold'],
			track_iou_threshold=p['Trackiouthreshold'],
			track_buffer=p['Tracklossframes'],
			min_hits=p['Trackconfirmframes'],
			smoothing=p['Outputsmoothing'],
			presence_rise=p.get('Presencerise', 5),
			presence_fall=p.get('Presencefall', 15),
		)
	pipeline.write_tracks_table(out)


# OnnxTopTables COMP path -> {track_id: smoothed class scores}
_class_scores = {}


def write_scores(out, comp, tracks_dat, scores_dat):
	"""out = tracks_dat plus per-track classifier columns: <prefix>_label and <prefix>_<name>
	for each class (onnx_hsemotion.py's table layout with prefix 'emotion'), and the label
	cell shows "<id> <LABEL>" for DebugBB. scores_dat is a Classify Crops OnnxTOP's Info DAT
	(id, class, conf, <class names>...), ids = track ids. Scores are smoothed per track
	(Classsmoothing) and held while a track has no fresh crop; without a classifier this is
	a plain copy."""
	if tracks_dat is None:
		out.clear()
		return
	out.copy(tracks_dat)
	if scores_dat is None or scores_dat.numRows < 1 or out.numRows < 1:
		return
	sh = [c.val for c in scores_dat.row(0)]
	if sh[:3] != ['id', 'class', 'conf'] or len(sh) < 4:
		return	# not a Classify Crops Info DAT
	names = sh[3:]
	th = [c.val for c in out.row(0)]
	if 'track_id' not in th:
		return	# e.g. the error row
	prefix_par = getattr(comp.par, 'Classprefix', None)
	prefix = (prefix_par.eval() if prefix_par is not None else '') or 'class'
	smooth_par = getattr(comp.par, 'Classsmoothing', None)
	smoothing = smooth_par.eval() if smooth_par is not None else 0.5

	state = _class_scores.setdefault(comp.path, {})
	for r in range(1, scores_dat.numRows):
		row = scores_dat.row(r)
		probs = np.array([float(c.val) for c in row[3:3 + len(names)]], dtype=np.float32)
		tid = row[0].val
		prev = state.get(tid)
		state[tid] = probs if prev is None or len(prev) != len(probs) else prev * smoothing + probs * (1.0 - smoothing)

	id_col = th.index('track_id')
	label_col = th.index('label') if 'label' in th else -1
	present = set()
	rows = []
	for r in range(1, out.numRows):
		row = [c.val for c in out.row(r)]
		tid = row[id_col]
		present.add(tid)
		probs = state.get(tid)
		if probs is not None:
			label = names[int(np.argmax(probs))].upper()
			if label_col >= 0:
				cell = row[label_col]
				head = cell[:cell.rfind('}') + 1] if '}' in cell else ''	# keep the {#color(...);} tag
				row[label_col] = f"{head}{tid} {label}"
			extra = [label] + [f'{v:.4f}' for v in probs]
		else:
			extra = [''] * (1 + len(names))
		rows.append((row, extra))
	for tid in [t for t in state if t not in present]:
		del state[tid]

	# Insert before the trailing r, g, b color columns, like onnx_hsemotion.py.
	at = len(th) - 3 if th[-3:] == ['r', 'g', 'b'] else len(th)
	header = th[:at] + [f'{prefix}_label'] + [f'{prefix}_{n.lower()}' for n in names] + th[at:]
	out.clear()
	out.appendRow(header)
	for row, extra in rows:
		out.appendRow(row[:at] + extra + row[at:])


def write_joints(out, comp):
	pipeline = get_pipeline(comp)
	if hasattr(pipeline, 'write_joints_table'):
		pipeline.write_joints_table(out)
	else:
		out.clear()
		out.appendRow(object_tracker.joints_header())


def write_bones(out, comp):
	pipeline = get_pipeline(comp)
	if isinstance(pipeline, pose_tracker_pipeline.PoseTrackerPipeline):
		pipeline.write_bones_table(out)
	else:
		out.clear()
		out.appendRow(object_tracker.bones_header())
