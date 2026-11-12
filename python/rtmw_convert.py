"""RTMW3D (OnnxTOP Pose Crops, 133 COCO-WholeBody keypoints) -> a MediaPipe Holistic-shaped table
for the tracker's `rtmw` mode: pose 0-32, left hand 33-53, right hand 54-74 (no face mesh), per
keypoint u, v, conf, z and world landmarks w{i}_x/y/z in metres, hip origin, MediaPipe's raw axes
(y down, z away from the camera; the tracker turns them to TD's).

Runs in a Script DAT, one table row per person:

	import rtmw_convert
	def onCook(scriptOp):
		rtmw_convert.convert(scriptOp, scriptOp.inputs[0], parent().op('input'))

RTMW3D gives x, y in the source image and z in its own relative units (about half a crop width
across the model's range), so the metric world comes from scaling the whole body by the median
ratio of typical human bone lengths to the measured ones, per person per frame. It is a good
direction estimate, not an absolute size. No numpy: it runs inside TouchDesigner's main thread.
"""
import statistics

import mediapipe_skeletons as skeletons

FACE0, LEFT_HAND0, RIGHT_HAND0 = 23, 91, 112	# COCO-WholeBody: body 0-16, feet 17-22, face 23-90, hands 91-132
# Source RTMW keypoint for every MediaPipe pose landmark (see POSE_NAMES): the eyes and mouth come
# from the 68-point face, the hand-side landmarks from the finger joints.
POSE_SOURCE = [
	0, FACE0 + 42, 1, FACE0 + 45, FACE0 + 39, 2, FACE0 + 36, 3, 4, FACE0 + 54, FACE0 + 48,
	5, 6, 7, 8, 9, 10,
	LEFT_HAND0 + 17, RIGHT_HAND0 + 17, LEFT_HAND0 + 5, RIGHT_HAND0 + 5, LEFT_HAND0 + 4, RIGHT_HAND0 + 4,
	11, 12, 13, 14, 15, 16, 19, 22, 17, 20,
]
SOURCE = POSE_SOURCE + list(range(LEFT_HAND0, LEFT_HAND0 + 21)) + list(range(RIGHT_HAND0, RIGHT_HAND0 + 21))
COUNT = len(SOURCE)	# 75
HAND_RANGES = ((33, 54), (54, 75))
HAND_MIN_SCORE = 0.3	# a hand whose mean score is below this is reported absent (conf 0)
BONE_MIN_SCORE = 0.3
# (a, b, metres) between RTMW keypoints: the bones used for the body's scale.
BONES = [(5, 7, 0.29), (7, 9, 0.26), (6, 8, 0.29), (8, 10, 0.26), (11, 13, 0.42), (13, 15, 0.41), (12, 14, 0.42),
	(14, 16, 0.41), (5, 6, 0.38), (11, 12, 0.27), (5, 11, 0.50), (6, 12, 0.50)]
HEADER = (['id', 'class', 'conf', 'u1', 'v1', 'u2', 'v2', 'label']
	+ [f'kp{i}_{s}' for i in range(COUNT) for s in ('u', 'v', 'c')]
	+ [f'kp{i}_z' for i in range(COUNT)]
	+ [f'w{i}_{s}' for i in range(COUNT) for s in 'xyz'])


def _columns(header):
	index = {name: i for i, name in enumerate(header)}
	kp = [index.get(f'kp{k}_u') for k in range(133)]
	if None in kp or 'kp0_z' not in index or 'crop_w' not in index:
		return None
	return {'kp': kp, 'z': [index[f'kp{k}_z'] for k in range(133)], 'crop_w': index['crop_w'],
		'id': index.get('id'), 'conf': index['conf'], 'box': [index[c] for c in ('u1', 'v1', 'u2', 'v2')]}


def _world(points, conf, crop_w):
	"""Hip-centered metric points (MediaPipe raw axes) for all 133 keypoints, or None."""
	if conf[11] < BONE_MIN_SCORE and conf[12] < BONE_MIN_SCORE:
		return None
	# points: (X px, Y px down, Z px away); z units are about half a crop width
	ratios = []
	for a, b, metres in BONES:
		if conf[a] < BONE_MIN_SCORE or conf[b] < BONE_MIN_SCORE:
			continue
		length = sum((points[a][i] - points[b][i]) ** 2 for i in range(3)) ** 0.5
		if length > 1e-6:
			ratios.append(metres / length)
	if not ratios:
		return None
	scale = statistics.median(ratios)
	hips = [i for i in (11, 12) if conf[i] >= BONE_MIN_SCORE]
	root = [sum(points[i][c] for i in hips) / len(hips) for c in range(3)]
	return [[(p[c] - root[c]) * scale for c in range(3)] for p in points]


def convert(script_op, info, image):
	"""Fill script_op with the Holistic-shaped table for `info` (OnnxTOP Pose Crops Info DAT)."""
	script_op.clear()
	script_op.appendRow(HEADER)
	if info.numRows < 2:
		return
	lines = info.text.splitlines()
	cols = _columns(lines[0].split('\t'))
	if cols is None:
		return
	width, height = float(image.width), float(image.height)
	for line in lines[1:info.numRows]:
		row = line.split('\t')
		u = [float(row[c]) for c in cols['kp']]
		v = [float(row[c + 1]) for c in cols['kp']]
		conf = [float(row[c + 2]) for c in cols['kp']]
		z = [float(row[c]) for c in cols['z']]
		z_px = float(row[cols['crop_w']]) / 2.0
		points = [(u[k] * width, (1.0 - v[k]) * height, z[k] * z_px) for k in range(133)]
		world = _world(points, conf, z_px * 2.0)
		present = [True] * COUNT
		for start, end in HAND_RANGES:
			scores = [conf[SOURCE[i]] for i in range(start, end)]
			if sum(scores) / len(scores) < HAND_MIN_SCORE:
				for i in range(start, end):
					present[i] = False
		out = [row[cols['id']] if cols['id'] is not None else '0', '0', row[cols['conf']]] + [row[c] for c in cols['box']] + ['']
		for i in range(COUNT):
			k = SOURCE[i]
			out += [f'{u[k]:.5f}', f'{v[k]:.5f}', f'{conf[k] if present[i] else 0.0:.4f}']
		out += [f'{z[SOURCE[i]]:.5f}' if present[i] else '0' for i in range(COUNT)]
		for i in range(COUNT):
			p = world[SOURCE[i]] if world is not None and present[i] else (0.0, 0.0, 0.0)
			out += [f'{p[0]:.5f}', f'{p[1]:.5f}', f'{p[2]:.5f}']
		script_op.appendRow(out)
