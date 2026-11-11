"""Backend-agnostic box tracking (YOLO detect + seg): detections in, tracked table out.

Everything YOLO26 detect/seg post-processing does AFTER the model-specific decode --
ByteTracker (Kalman + two-stage high/low association), box smoothing, the display gate,
seg's held masks + presence ramp, and the table_output format -- lives here so every
backend shares one implementation:

	- python/scripts/onnx_yolo26_obj_det.py  (Python ONNX Runtime, detect)
	- python/scripts/onnx_yolo26_seg.py      (Python ONNX Runtime, seg)
	- python/onnxtop_tables.py          (C++ OnnxTOP, decode + masks on the GPU -> Info DAT)

The pose equivalent (KeypointTracker-based) is pose_tracker_pipeline.py.

Input detections are dicts in TD space (0-1, origin bottom-left):
	{'box': [x_left, y_bottom, x_right, y_top], 'score': float, 'class_id': int,
	 'class_name': str, ['mask': ndarray, 'mask_area_ratio': float]  (seg)}
"""
import numpy as np

import object_tracker

COCO_CLASSES = {
	0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
	7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
	12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
	19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
	25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
	31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
	36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
	41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
	48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
	54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
	60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
	66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
	72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
	78: 'hair drier', 79: 'toothbrush',
}

# Tuned defaults of the two Python scripts (see the comments next to the same constants in
# onnx_yolo26_obj_det.py / onnx_yolo26_seg.py for the reasoning behind each value).
DETECT_DEFAULTS = {
	'Confthreshold': 0.5, 'Lowconfthreshold': 0.1, 'Nmsiouthreshold': 0.45,
	'Minboxwidth': 0.02, 'Minboxheight': 0.02,
	'Tracklossframes': 30, 'Trackiouthreshold': 0.3, 'Trackconfirmframes': 3,
	'Outputsmoothing': 0.5,
}
SEGMENT_DEFAULTS = {
	'Confthreshold': 0.35, 'Lowconfthreshold': 0.1, 'Nmsiouthreshold': 0.5,
	'Minboxwidth': 0.02, 'Minboxheight': 0.02, 'Minmaskarea': 0.15,
	'Tracklossframes': 30, 'Trackiouthreshold': 0.3, 'Trackconfirmframes': 3,
	'Outputsmoothing': 0.5, 'Presencerise': 5, 'Presencefall': 15,
}


def parameters(segmentation, defaults=None):
	"""(name, label, default, min, max, clampMax, help) per tracking par, in page order."""
	d = dict(SEGMENT_DEFAULTS if segmentation else DETECT_DEFAULTS)
	d.update(defaults or {})
	who = {'subject': 'person', 'subject_plural': 'people'} if segmentation else {}
	shape_note = (
		"a standing person is naturally much narrower than tall, so one shared threshold high "
		"enough to reject noise ends up rejecting real, legitimately tall-but-narrow people."
	) if segmentation else (
		"an object is rarely square, so one shared threshold high enough to reject noise ends up "
		"rejecting real objects that are just proportionally narrow on one axis."
	)
	pars = [
		('Confthreshold', 'Confidence Threshold', d['Confthreshold'], 0.0, 1.0, True,
			object_tracker.par_help('Confthreshold')),
		('Lowconfthreshold', 'Low Confidence Threshold (Recovery)', d['Lowconfthreshold'], 0.0, 1.0, True,
			object_tracker.par_help('Lowconfthreshold')),
		('Nmsiouthreshold', 'NMS IoU Threshold (Dedup)', d['Nmsiouthreshold'], 0.0, 1.0, True,
			object_tracker.par_help('Nmsiouthreshold', **who)),
		('Minboxwidth', 'Min Box Width', d['Minboxwidth'], 0.0, 0.2, False,
			object_tracker.par_help('Minboxwidth', shape_note=shape_note)),
		('Minboxheight', 'Min Box Height', d['Minboxheight'], 0.0, 0.2, False,
			object_tracker.par_help('Minboxheight')),
	]
	if segmentation:
		pars.append(('Minmaskarea', 'Min Mask Fill Ratio (Of Own Box)', d['Minmaskarea'], 0.0, 1.0, True,
			"Rejects a decoded mask if it fills less than this fraction of ITS OWN detection "
			"box (not a fraction of the whole frame -- scale-invariant regardless of how near/far a "
			"person is). A real silhouette typically fills ~35-70% of its own box (limb gaps keep it "
			"under 100%); much lower usually means a degenerate/noisy mask."))
	pars += [
		('Tracklossframes', 'Track Loss Frames', d['Tracklossframes'], 0.0, 90.0, False,
			object_tracker.par_help('Tracklossframes')),
		('Trackiouthreshold', 'Track IoU Threshold', d['Trackiouthreshold'], 0.0, 1.0, True,
			object_tracker.par_help('Trackiouthreshold', **({'subject': 'person'} if segmentation else {}))),
		('Trackconfirmframes', 'Track Confirm Frames', d['Trackconfirmframes'], 1.0, 30.0, False,
			object_tracker.par_help('Trackconfirmframes', **({'subject': 'person'} if segmentation else {}))),
		('Outputsmoothing', 'Output Smoothing (Box)' if segmentation else 'Output Smoothing',
			d['Outputsmoothing'], 0.0, 1.0, True,
			object_tracker.par_help('Outputsmoothing', extra=(
				"Masks themselves are NOT smoothed by this -- see class docstring." if segmentation else None))),
	]
	if segmentation:
		pars += [
			('Presencerise', 'Presence Rise Frames', d['Presencerise'], 1.0, 60.0, False,
				object_tracker.par_help('Presencerise', subject='person')),
			('Presencefall', 'Presence Fall Frames', d['Presencefall'], 1.0, 90.0, False,
				object_tracker.par_help('Presencefall', subject='person')),
		]
	return pars


def append_parameters(page, owner, segmentation, defaults=None):
	"""Create the tracking pars on `page` (replacing existing ones) and set them to defaults."""
	for name, label, default, vmin, vmax, clamp_max, help_text in parameters(segmentation, defaults):
		p = page.appendFloat(name, label=label, size=1)
		p[0].default = default
		p[0].min = vmin
		p[0].max = vmax
		p[0].clampMin = True
		p[0].clampMax = clamp_max
		p[0].help = help_text
		setattr(owner.par, name, default)


def read_parameters(owner, segmentation, defaults=None):
	"""Current par values from `owner` (falling back to defaults), keyed by par name."""
	values = {}
	for name, _, default, *_ in parameters(segmentation, defaults):
		par = getattr(owner.par, name, None)
		values[name] = par.eval() if par is not None else default
	return values


def filter_detections(detections, low_conf_threshold, min_box_width, min_box_height,
		nms_iou_threshold, min_mask_area=None):
	"""Admission gate for backends that hand over unfiltered boxes, in the Python scripts'
	order: low-confidence floor + min box size, NMS, then (seg) min mask fill ratio."""
	dets = [
		d for d in detections
		if d['score'] > low_conf_threshold
		and d['box'][2] - d['box'][0] >= min_box_width
		and d['box'][3] - d['box'][1] >= min_box_height
	]
	if dets:
		boxes = np.array([d['box'] for d in dets], dtype=np.float32)
		scores = np.array([d['score'] for d in dets], dtype=np.float32)
		dets = [dets[i] for i in object_tracker.nms(boxes, scores, nms_iou_threshold)]
	if min_mask_area is not None:
		dets = [d for d in dets if d.get('mask_area_ratio', 1.0) > min_mask_area]
	return dets


class BoxTrackerPipeline:
	"""Owns the ByteTracker plus per-track smoothing/mask/presence state. One per stream.

	Detect mode shows a track while it's confirmed and its (decaying) score clears
	Confthreshold. Seg mode deliberately drops the score gate: it reacts to noisy raw
	per-frame confidence even while the track is perfectly stable, which made masks flicker;
	the presence ramp (driven by lost_frames, see object_tracker.presence_ramp()) is the
	smooth replacement. Seg tracks also hold their last real decoded mask across frames
	without a fresh detection rather than smoothing or re-decoding it.
	"""

	def __init__(self, segmentation=False, defaults=None):
		self.segmentation = segmentation
		d = dict(SEGMENT_DEFAULTS if segmentation else DETECT_DEFAULTS)
		d.update(defaults or {})
		self.tracker = object_tracker.ByteTracker(
			high_thresh=d['Confthreshold'], low_thresh=d['Lowconfthreshold'],
			match_thresh=d['Trackiouthreshold'], track_buffer=d['Tracklossframes'],
			min_hits=d['Trackconfirmframes'],
		)
		self.conf_threshold = d['Confthreshold']
		self.low_conf_threshold = d['Lowconfthreshold']
		self._box_state = {}
		self._gap_state = {}	# track_id -> last lost_frames (object_tracker.reacquired)
		self._mask_state = {}
		self._presence_state = {}
		# Structured tracking data for CHOP/table consumption; see update() for the dict keys.
		self.tracked_objects = []

	def update(self, detections, conf_threshold, low_conf_threshold, track_iou_threshold,
			track_buffer, min_hits, smoothing, presence_rise=5, presence_fall=15):
		"""Advance the tracker one frame. `detections` must already be admission-filtered
		(see filter_detections()). Returns (and stores) the displayable tracked objects."""
		self.conf_threshold = conf_threshold
		self.low_conf_threshold = low_conf_threshold
		self.tracker.high_thresh = conf_threshold
		self.tracker.low_thresh = low_conf_threshold
		self.tracker.match_thresh = track_iou_threshold
		self.tracker.track_buffer = track_buffer
		self.tracker.min_hits = int(min_hits)

		# Update tracker (runs on main thread, no lock needed)
		active_tracks = self.tracker.update(detections)

		active_ids = {t.track_id for t in active_tracks}
		self.tracked_objects = []
		for t in active_tracks:
			snap = object_tracker.reacquired(self._gap_state, t.track_id, t.lost_frames)
			if not t.confirmed:
				continue
			if not self.segmentation and t.score < conf_threshold:
				continue
			smoothed = object_tracker.box_smooth(self._box_state, t.track_id, t.box, smoothing, snap)  # t.box = Kalman estimate
			obj = {
				'track_id': t.track_id,
				'class_id': t.payload.get('class_id'),
				'class_name': t.payload.get('class_name', 'unknown'),
				'score': t.score,
				'cx': (smoothed[0] + smoothed[2]) / 2, 'cy': (smoothed[1] + smoothed[3]) / 2,
				'w': smoothed[2] - smoothed[0], 'h': smoothed[3] - smoothed[1],
				'x_left': smoothed[0],
				'x_right': smoothed[2],
				'y_top': smoothed[3],     # top edge of bbox (TD coords)
				'y_bottom': smoothed[1],  # bottom edge of bbox (TD coords)
				'vx': float(t.mean[4]), 'vy': float(t.mean[5]),  # Kalman-estimated box-center velocity
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
			}
			if self.segmentation:
				obj['presence'] = object_tracker.presence_ramp(
					self._presence_state, t.track_id, t.lost_frames == 0, presence_rise, presence_fall,
				)
				# Masks aren't smoothed pixel-wise -- hold the last real decoded mask across
				# any frame this track didn't get a fresh detection.
				new_mask = t.payload.get('mask')
				if new_mask is not None:
					self._mask_state[t.track_id] = new_mask
				obj['mask'] = self._mask_state.get(t.track_id)
				obj['mask_area_ratio'] = t.payload.get('mask_area_ratio', 0.0)
				# This frame's detection for the track (its pixels in an Instances matte), -1 while coasting.
				obj['det_index'] = t.payload.get('det_index', -1) if t.lost_frames == 0 else -1
			self.tracked_objects.append(obj)

		# Prune box/mask/presence state for tracks the tracker has dropped entirely.
		object_tracker.prune_stale(active_ids, self._box_state, self._mask_state, self._presence_state, self._gap_state)
		return self.tracked_objects

	def write_tracks_table(self, tbl):
		"""table_output: label ("<id> <class name> <score>%") + class + box columns (+ seg's
		mask_area_ratio, presence, det_index) + color."""
		tbl.clear()
		extra = ['mask_area_ratio', 'presence', 'det_index'] if self.segmentation else []
		tbl.appendRow([
			*object_tracker.label_header(),
			'class_id', 'class_name',
			*object_tracker.box_header(),
			*extra,
			*object_tracker.color_header(),
		])
		for obj in self.tracked_objects:
			extra_vals = ([f"{obj['mask_area_ratio']:.4f}", f"{obj['presence']:.4f}", obj.get('det_index', -1)]
				if self.segmentation else [])
			# Label names the class too ("161 person 94%"), so DebugBB shows what each box is.
			label_text = f"{obj['track_id']} {obj['class_name']} {round(obj['score'] * 100)}%"
			tbl.appendRow([
				*object_tracker.label_row(obj['track_id'], obj['score'], label_text=label_text),
				obj['class_id'], obj['class_name'],
				*object_tracker.box_row(obj),
				*extra_vals,
				*object_tracker.color_row(obj['track_id']),
			])
