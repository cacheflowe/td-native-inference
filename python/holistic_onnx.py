"""MediaPipe Holistic on the GPU: Holistic's own landmark models (converted to ONNX by
scripts/convert_holistic_onnx.py) run by OnnxTOP's Landmark Crops mode, with MediaPipe's ROI logic
reimplemented here. The output table matches MediaPipeTOP's Holistic Info DAT (pose 0-32, left hand
33-53, right hand 54-74, face 75-552; kp{i}_u/v/c, kp{i}_z, w{i}_x/y/z for the first 75), so the
tracker's `mp_holistic` mode and MixamoPuppet read it unchanged.

Per frame (one person per ROI, several people at once):

	tracks (YOLO26 people) --+
	                         +--> pose_rois --> OnnxTOP pose_landmarks_detector --> pose_decode
	roi_state (Table DAT) ---+                                                      | writes roi_state
	                                                                                 v
	                                                                         holistic table

pose_rois: last frame's landmark ROIs (MediaPipe's tracking: the pose's own alignment points give
the next crop), plus a box ROI for each detected person no ROI covers yet. pose_decode: decodes the
model outputs, projects them back through the rotated crop, and writes the next frame's ROIs to
roi_state (a Table DAT, so pose_rois recooks). A person whose pose presence drops below 0.5 loses
the ROI and is picked up again from the detector.

The math follows MediaPipe's calculators (google-ai-edge/mediapipe, mediapipe/calculators/util and
mediapipe/modules/pose_landmark): TensorsToLandmarks, LandmarkProjection, WorldLandmarkProjection,
AlignmentPointsRects, RectTransformation. Not reproduced (yet): the heatmap refinement of the
landmark positions, and MediaPipe's landmark smoothing (the tracker smooths). Pure Python.
"""
import math

POSE_COUNT, HAND_COUNT, FACE_COUNT = 33, 21, 478
LEFT_HAND0, RIGHT_HAND0, FACE0 = 33, 54, 75
TOTAL = FACE0 + FACE_COUNT	# 553
WORLD_COUNT = FACE0			# world landmarks for pose + hands
POSE_INPUT = 256
POSE_PRESENCE = 0.5			# ThresholdingCalculator in tensors_to_pose_landmarks_and_segmentation
ROI_SCALE = 1.25			# pose_landmarks_to_roi / pose_detection_to_roi
BOX_ROI_SCALE = 1.25		# a detector box -> first ROI (the pose's own points take over next frame)
DUPLICATE_IOU = 0.5			# two ROIs whose landmark boxes overlap this much are one person
COVERED_IOU = 0.3			# a detector box overlapping a tracked person this much is that person
MAX_PEOPLE = 4

HEADER = (['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label']
	+ [f'kp{i}_{s}' for i in range(TOTAL) for s in ('u', 'v', 'c')]
	+ [f'kp{i}_z' for i in range(TOTAL)]
	+ [f'w{i}_{s}' for i in range(WORLD_COUNT) for s in 'xyz'])
ROI_HEADER = ['id', 'roi_x', 'roi_y', 'roi_w', 'roi_h', 'roi_angle', 'box_u1', 'box_v1', 'box_u2', 'box_v2']


def _sigmoid(v):
	return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v))))


def _normalize_radians(a):
	return a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))


def alignment_roi(x0, y0, x1, y1, width, height, target_degrees=90.0, scale=1.0, shift_y=0.0):
	"""MediaPipe's AlignmentPointsRectsCalculator + RectTransformationCalculator (square_long):
	center (x0, y0), scale point (x1, y1), normalized top-down -> (x, y, w, h, angle)."""
	px0, py0, px1, py1 = x0 * width, y0 * height, x1 * width, y1 * height
	size = 2.0 * math.hypot(px1 - px0, py1 - py0)
	angle = _normalize_radians(math.radians(target_degrees) - math.atan2(-(py1 - py0), px1 - px0))
	return transform_roi(x0, y0, size / width, size / height, angle, width, height, scale, shift_y)


def transform_roi(x, y, w, h, angle, width, height, scale=1.0, shift_y=0.0, shift_x=0.0):
	"""RectTransformationCalculator with square_long."""
	if angle == 0.0:
		x += w * shift_x
		y += h * shift_y
	else:
		c, s = math.cos(angle), math.sin(angle)
		x += (width * w * shift_x * c - height * h * shift_y * s) / width
		y += (width * w * shift_x * s + height * h * shift_y * c) / height
	side = max(w * width, h * height)
	return x, y, side / width * scale, side / height * scale, angle


def project(nx, ny, roi):
	"""LandmarkProjectionCalculator: crop-normalized (0-1) -> image-normalized, top-down."""
	x, y = nx - 0.5, ny - 0.5
	c, s = math.cos(roi[4]), math.sin(roi[4])
	return (c * x - s * y) * roi[2] + roi[0], (s * x + c * y) * roi[3] + roi[1]


def _iou(a, b):
	ix = min(a[2], b[2]) - max(a[0], b[0])
	iy = min(a[3], b[3]) - max(a[1], b[1])
	if ix <= 0 or iy <= 0:
		return 0.0
	inter = ix * iy
	return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _rows(dat):
	"""Table DAT -> list of {column: string} (header row 0)."""
	if dat is None or dat.numRows < 2:
		return []
	names = [c.val for c in dat.row(0)]
	return [dict(zip(names, (c.val for c in dat.row(r)))) for r in range(1, dat.numRows)]


def _columns(info, prefix, count):
	"""Column indices of o{k}_0..count-1 for output k = prefix, or None."""
	header = [c.val for c in info.row(0)]
	try:
		first = header.index(f'{prefix}_0')
	except ValueError:
		return None
	return first if first + count <= len(header) else None


# ---------------------------------------------------------------- pose stage

def pose_rois(script_op, state, tracks, image, max_people=MAX_PEOPLE):
	"""Script DAT: this frame's pose ROIs. state: roi_state (pose_decode's next ROIs, with each
	person's landmark box in TD UV); tracks: the detector's script_tracks (track_id, x_left, y_bottom,
	x_right, y_top in TD UV); image: the source TOP (for its size)."""
	script_op.clear()
	script_op.appendRow(ROI_HEADER[:6])
	width, height = float(image.width), float(image.height)
	kept = []	# (id, roi, box in TD UV)
	for row in _rows(state)[:max_people]:
		roi = [float(row[k]) for k in ROI_HEADER[1:6]]
		box = [float(row[k]) for k in ROI_HEADER[6:10]]
		kept.append((row['id'], roi, box))
	used = {k[0] for k in kept}
	for row in _rows(tracks):
		if len(kept) >= max_people:
			break
		box = [float(row['x_left']), float(row['y_bottom']), float(row['x_right']), float(row['y_top'])]
		if any(_iou(box, k[2]) > COVERED_IOU for k in kept):
			continue
		ident = row['track_id']	# numeric: the tracker reads id as a number
		if ident in used:
			continue
		# Box center, square on its long side (pixels), upright; top-down normalized.
		cx, cy = (box[0] + box[2]) * 0.5, 1.0 - (box[1] + box[3]) * 0.5
		roi = transform_roi(cx, cy, box[2] - box[0], box[3] - box[1], 0.0, width, height, BOX_ROI_SCALE)
		kept.append((ident, list(roi), box))
		used.add(ident)
	for ident, roi, _ in kept:
		script_op.appendRow([ident] + [f'{v:.6f}' for v in roi])


def _decode_pose(info, r, cols, roi):
	"""One Info DAT row -> (landmarks [33 x (x, y, z, vis, pres)] image-normalized top-down,
	aux [2 x (x, y)], world [33 x (x, y, z)], presence)."""
	lm_col, flag_col, world_col = cols
	cells = info.row(r)
	presence = float(cells[flag_col].val)
	c, s = math.cos(roi[4]), math.sin(roi[4])
	landmarks, aux = [], []
	for k in range(POSE_COUNT + 2):
		at = lm_col + k * 5
		nx = float(cells[at].val) / POSE_INPUT
		ny = float(cells[at + 1].val) / POSE_INPUT
		x, y = project(nx, ny, roi)
		if k >= POSE_COUNT:
			aux.append((x, y))
			continue
		z = float(cells[at + 2].val) / POSE_INPUT * roi[2]
		landmarks.append((x, y, z, _sigmoid(float(cells[at + 3].val)), _sigmoid(float(cells[at + 4].val))))
	world = []
	for k in range(POSE_COUNT):
		at = world_col + k * 3
		wx, wy, wz = float(cells[at].val), float(cells[at + 1].val), float(cells[at + 2].val)
		world.append((c * wx - s * wy, s * wx + c * wy, wz))
	return landmarks, aux, world, presence


def pose_decode(script_op, info, image, state):
	"""Script DAT: pose OnnxTOP Info DAT -> the Holistic table (pose filled in); writes the next
	frame's ROIs into `state` (a Table DAT)."""
	script_op.clear()
	script_op.appendRow(HEADER)
	width, height = float(image.width), float(image.height)
	people = []
	if info is not None and info.numRows >= 2:
		cols = (_columns(info, 'o0', (POSE_COUNT + 6) * 5), _columns(info, 'o1', 1), _columns(info, 'o3', (POSE_COUNT + 6) * 3))
		if None not in cols:
			for r in range(1, info.numRows):
				cells = info.row(r)
				roi = [float(cells[i].val) for i in range(1, 6)]
				landmarks, aux, world, presence = _decode_pose(info, r, cols, roi)
				if presence < POSE_PRESENCE:
					continue
				xs = [p[0] for p in landmarks]
				ys = [p[1] for p in landmarks]
				box = [min(xs), 1.0 - max(ys), max(xs), 1.0 - min(ys)]	# TD UV
				score = sum(p[3] for p in landmarks) / POSE_COUNT
				people.append({'id': cells[0].val, 'landmarks': landmarks, 'world': world, 'box': box, 'score': score,
					'next': alignment_roi(aux[0][0], aux[0][1], aux[1][0], aux[1][1], width, height, 90.0, ROI_SCALE)})
	# Two ROIs that converged on one person: keep the more confident.
	people.sort(key=lambda p: -p['score'])
	unique = []
	for p in people:
		if all(_iou(p['box'], q['box']) < DUPLICATE_IOU for q in unique):
			unique.append(p)
	state.clear()
	state.appendRow(ROI_HEADER)
	for p in unique:
		state.appendRow([p['id']] + [f'{v:.6f}' for v in p['next']] + [f'{v:.5f}' for v in p['box']])
		script_op.appendRow(_holistic_row(p))


# ---------------------------------------------------------------- hands and face

HAND_INPUT, RECROP_INPUT, FACE_INPUT = 224, 256, 192
HAND_PRESENCE, FACE_PRESENCE = 0.5, 0.5
WRIST_VISIBILITY = 0.1		# hand_visibility_from_hand_landmarks_from_pose
HAND_POSE_POINTS = {'L': (15, 17, 19), 'R': (16, 18, 20)}	# wrist, pinky, index
HAND_BASE = {'L': LEFT_HAND0, 'R': RIGHT_HAND0}
FACE_POSE_POINTS = (0, 2, 5, 7, 8, 9, 10)	# nose, eyes, ears, mouth: BlazeFace's six points, roughly
# (rotation degrees, translation, scale, landmark margin): RoiTrackingCalculator options
HAND_TRACKING = (40.0, 0.2, 0.4, -0.1)
FACE_TRACKING = (15.0, 0.1, 0.3, -0.2)
_PALM = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)	# HandLandmarksToRectCalculator's partial landmarks
_prev = {}	# (merge op path, part id) -> last frame's landmarks [(x, y)], image-normalized top-down
KP0, Z0 = 8, 8 + TOTAL * 3
W0 = Z0 + TOTAL


def _pose_point(row, k):
	"""Holistic table row (list of str) -> pose landmark k as (x, y top-down, visibility)."""
	return float(row[KP0 + k * 3]), 1.0 - float(row[KP0 + k * 3 + 1]), float(row[KP0 + k * 3 + 2])


def _people(table):
	return [[c.val for c in table.row(r)] for r in range(1, table.numRows)] if table is not None else []


def hand_recrop_rois(script_op, pose_table, image):
	"""Script DAT: for each person and side whose pose wrist is visible, the hand ROI from the pose
	(HandLandmarksFromPoseToRecropRoi) for the ROI-refinement model. id = '<person>:L' / ':R'."""
	script_op.clear()
	script_op.appendRow(ROI_HEADER[:6])
	width, height = float(image.width), float(image.height)
	for row in _people(pose_table):
		for side, (wrist, pinky, index) in HAND_POSE_POINTS.items():
			wx, wy, vis = _pose_point(row, wrist)
			if vis <= WRIST_VISIBILITY:
				continue
			px, py, _ = _pose_point(row, pinky)
			ix, iy, _ = _pose_point(row, index)
			mx, my = (2.0 * ix + px) / 3.0, (2.0 * iy + py) / 3.0	# estimated middle finger
			roi = alignment_roi(mx, my, wx, wy, width, height, 90.0)	# size: 2 x middle-wrist
			# HandDetectionsFromPoseToRects: rotation from wrist -> middle finger, not center -> scale point.
			angle = _normalize_radians(math.pi / 2 - math.atan2(-(my - wy) * height, (mx - wx) * width))
			roi = transform_roi(mx, my, roi[2], roi[3], angle, width, height, 2.7, -0.1)
			script_op.appendRow([f'{row[0]}:{side}'] + [f'{v:.6f}' for v in roi])


def _palm_roi(points, width, height):
	"""HandLandmarksToRoi (holistic): last frame's 21 hand points -> ROI."""
	pts = [points[i] for i in _PALM]
	x0, y0 = pts[0][0] * width, pts[0][1] * height
	x1 = ((pts[4][0] + pts[8][0]) / 2.0 + pts[6][0]) / 2.0 * width
	y1 = ((pts[4][1] + pts[8][1]) / 2.0 + pts[6][1]) / 2.0 * height
	angle = _normalize_radians(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
	return _rotated_bounds(pts, angle, width, height, 2.0, -0.1)


def _rotated_bounds(pts, angle, width, height, scale, shift_y):
	"""The bounds of `pts` in the frame rotated by `angle`, as a ROI (HandLandmarksToRectCalculator's
	box), then RectTransformation."""
	cx = (max(p[0] for p in pts) + min(p[0] for p in pts)) / 2.0
	cy = (max(p[1] for p in pts) + min(p[1] for p in pts)) / 2.0
	c, s = math.cos(-angle), math.sin(-angle)
	xs, ys = [], []
	for x, y in pts:
		ox, oy = (x - cx) * width, (y - cy) * height
		xs.append(ox * c - oy * s)
		ys.append(ox * s + oy * c)
	pcx, pcy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
	ca, sa = math.cos(angle), math.sin(angle)
	x = (pcx * ca - pcy * sa + width * cx) / width
	y = (pcx * sa + pcy * ca + height * cy) / height
	return transform_roi(x, y, (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height, angle, width, height,
		scale, shift_y)


def _keep_tracking(prev_roi, prev_points, recrop, width, height, req):
	"""RoiTrackingCalculator: keep last frame's landmark ROI while it agrees with the recrop ROI."""
	rot_deg, translation, scale, margin = req
	c, s = math.cos(-recrop[4]), math.sin(-recrop[4])
	rx = lambda x, y: x * width * c - y * height * s
	ry = lambda x, y: x * width * s + y * height * c
	diff = math.degrees(prev_roi[4] - recrop[4])
	diff = (diff + 180.0) % 360.0 - 180.0
	if abs(diff) > rot_deg:
		return False
	if abs(rx(prev_roi[0], prev_roi[1]) - rx(recrop[0], recrop[1])) > recrop[2] * width * translation:
		return False
	if abs(ry(prev_roi[0], prev_roi[1]) - ry(recrop[0], recrop[1])) > recrop[3] * height * translation:
		return False
	if abs(prev_roi[2] - recrop[2]) * width > recrop[2] * width * scale or abs(prev_roi[3] - recrop[3]) * height > recrop[3] * height * scale:
		return False
	cx, cy = rx(recrop[0], recrop[1]), ry(recrop[0], recrop[1])
	hw, hh = recrop[2] * width * (1.0 + margin) / 2.0, recrop[3] * height * (1.0 + margin) / 2.0
	return all(abs(rx(x, y) - cx) < hw and abs(ry(x, y) - cy) < hh for x, y in prev_points)


def hand_rois(script_op, recrop_info, image, merge_op):
	"""Script DAT: the refinement model's two points -> the recrop ROI (HandRecropByRoi), then
	HandTracking: last frame's hand ROI while it still agrees, else the recrop ROI."""
	script_op.clear()
	script_op.appendRow(ROI_HEADER[:6])
	if recrop_info is None or recrop_info.numRows < 2:
		return
	col = _columns(recrop_info, 'o0', 4)
	if col is None:
		return
	width, height = float(image.width), float(image.height)
	for r in range(1, recrop_info.numRows):
		cells = recrop_info.row(r)
		ident = cells[0].val
		roi = [float(cells[i].val) for i in range(1, 6)]
		pts = [project(float(cells[col + 2 * k].val) / RECROP_INPUT, float(cells[col + 2 * k + 1].val) / RECROP_INPUT, roi)
			for k in range(2)]
		recrop = alignment_roi(pts[0][0], pts[0][1], pts[1][0], pts[1][1], width, height, -90.0, 1.0, -0.1)
		prev = _prev.get((merge_op.path, ident))
		if prev is not None:
			prev_roi = _palm_roi(prev, width, height)
			if _keep_tracking(prev_roi, prev, recrop, width, height, HAND_TRACKING):
				recrop = prev_roi
		script_op.appendRow([ident] + [f'{v:.6f}' for v in recrop])


def face_rois(script_op, pose_table, image, merge_op):
	"""Script DAT: a face ROI per person (FaceLandmarksFromPose): last frame's face-mesh ROI while
	it agrees with the ROI from the pose's face points, else that one. MediaPipe runs its face
	detector inside the pose's face crop for the latter; the pose's points stand in for its six
	keypoints here (the same kind of points)."""
	script_op.clear()
	script_op.appendRow(ROI_HEADER[:6])
	width, height = float(image.width), float(image.height)
	for row in _people(pose_table):
		pts = [_pose_point(row, k) for k in FACE_POSE_POINTS]
		if min(p[2] for p in pts[:3]) < 0.5:	# nose and eyes
			continue
		xs, ys = [p[0] for p in pts], [p[1] for p in pts]
		# DetectionsToRects (box from the keypoints, rotation right eye -> left eye) + scale 2, shift -0.1
		rex, rey, _ = _pose_point(row, 5)
		lex, ley, _ = _pose_point(row, 2)
		angle = _normalize_radians(-math.atan2(-(ley - rey) * height, (lex - rex) * width))
		recrop = transform_roi((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, max(xs) - min(xs), max(ys) - min(ys),
			angle, width, height, 2.0, -0.1)
		ident = f'{row[0]}:F'
		prev = _prev.get((merge_op.path, ident))
		if prev is not None:
			# FaceLandmarkLandmarksToRoi: box of the mesh, rotation from points 33 -> 263, scale 1.5.
			mxs, mys = [p[0] for p in prev], [p[1] for p in prev]
			a = _normalize_radians(-math.atan2(-(prev[263][1] - prev[33][1]) * height, (prev[263][0] - prev[33][0]) * width))
			prev_roi = transform_roi((min(mxs) + max(mxs)) / 2.0, (min(mys) + max(mys)) / 2.0, max(mxs) - min(mxs),
				max(mys) - min(mys), a, width, height, 1.5)
			if _keep_tracking(prev_roi, prev, recrop, width, height, FACE_TRACKING):
				recrop = prev_roi
		script_op.appendRow([ident] + [f'{v:.6f}' for v in recrop])


def _decode_part(info, r, lm_col, count, size, normalize_z, roi):
	"""Landmark rows of a hand / face model -> [(x, y, z)] image-normalized top-down."""
	cells = info.row(r)
	out = []
	for k in range(count):
		at = lm_col + k * 3
		x, y = project(float(cells[at].val) / size, float(cells[at + 1].val) / size, roi)
		out.append((x, y, float(cells[at + 2].val) / size / normalize_z * roi[2]))
	return out


def merge(script_op, pose_table, hand_info, face_info):
	"""Script DAT: the pose table with each person's hands (pose-side assignment, as Holistic does)
	and face mesh filled in; remembers them for the next frame's hand / face tracking."""
	script_op.clear()
	script_op.appendRow(HEADER)
	rows = {row[0]: row for row in _people(pose_table)}
	seen = set()
	if hand_info is not None and hand_info.numRows >= 2:
		cols = (_columns(hand_info, 'o0', 63), _columns(hand_info, 'o1', 1), _columns(hand_info, 'o3', 63))
		if None not in cols:
			for r in range(1, hand_info.numRows):
				cells = hand_info.row(r)
				ident = cells[0].val
				person, side = ident.rsplit(':', 1)
				if person not in rows or float(cells[cols[1]].val) < HAND_PRESENCE:
					continue
				roi = [float(cells[i].val) for i in range(1, 6)]
				pts = _decode_part(hand_info, r, cols[0], HAND_COUNT, HAND_INPUT, 0.4, roi)
				_prev[(script_op.path, ident)] = [(p[0], p[1]) for p in pts]
				seen.add(ident)
				row, base = rows[person], HAND_BASE[side]
				c, s = math.cos(roi[4]), math.sin(roi[4])
				world = []
				for k in range(HAND_COUNT):
					at = cols[2] + k * 3
					wx, wy, wz = (float(cells[at + j].val) for j in range(3))
					world.append((c * wx - s * wy, s * wx + c * wy, wz))
				# Holistic puts the hand in the body's space: its wrist on the pose's wrist.
				wrist = HAND_POSE_POINTS[side][0]
				offset = [float(row[W0 + wrist * 3 + j]) - world[0][j] for j in range(3)]
				for k, (x, y, z) in enumerate(pts):
					i = base + k
					row[KP0 + i * 3:KP0 + i * 3 + 3] = [f'{x:.5f}', f'{1.0 - y:.5f}', '1']
					row[Z0 + i] = f'{z:.5f}'
					row[W0 + i * 3:W0 + i * 3 + 3] = [f'{world[k][j] + offset[j]:.5f}' for j in range(3)]
	if face_info is not None and face_info.numRows >= 2:
		cols = (_columns(face_info, 'o0', FACE_COUNT * 3), _columns(face_info, 'o1', 1))
		if None not in cols:
			for r in range(1, face_info.numRows):
				cells = face_info.row(r)
				ident = cells[0].val
				person = ident.rsplit(':', 1)[0]
				if person not in rows or _sigmoid(float(cells[cols[1]].val)) < FACE_PRESENCE:
					continue
				roi = [float(cells[i].val) for i in range(1, 6)]
				pts = _decode_part(face_info, r, cols[0], FACE_COUNT, FACE_INPUT, 1.0, roi)
				_prev[(script_op.path, ident)] = [(p[0], p[1]) for p in pts]
				seen.add(ident)
				row = rows[person]
				for k, (x, y, z) in enumerate(pts):
					i = FACE0 + k
					row[KP0 + i * 3:KP0 + i * 3 + 3] = [f'{x:.5f}', f'{1.0 - y:.5f}', '1']
					row[Z0 + i] = f'{z:.5f}'
	for key in [k for k in _prev if k[0] == script_op.path and k[1] not in seen]:
		del _prev[key]
	for row in rows.values():
		script_op.appendRow(row)


def _holistic_row(p):
	kp, z = [], []
	for i in range(TOTAL):
		if i < POSE_COUNT:
			x, y, depth, vis, _ = p['landmarks'][i]
			kp += [f'{x:.5f}', f'{1.0 - y:.5f}', f'{vis:.4f}']
			z.append(f'{depth:.5f}')
		else:
			kp += ['0', '0', '0']
			z.append('0')
	world = []
	for i in range(WORLD_COUNT):
		if i < POSE_COUNT:
			world += [f'{v:.5f}' for v in p['world'][i]]
		else:
			world += ['0', '0', '0']
	box = [f'{min(max(v, 0.0), 1.0):.5f}' for v in p['box']]
	return [p['id'], '0', f"{p['score']:.4f}"] + box + [''] + kp + z + world
