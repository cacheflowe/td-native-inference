"""Backend-agnostic face tracking (YuNet): detections in, tracked table out.

Everything onnx_yunet.py does AFTER the model-specific decode -- ByteTracker (Kalman +
two-stage high/low association), box + keypoint smoothing, head direction (yaw/pitch/roll,
geometric or solvePnP), the display gate, and the table_output / table_joints formats --
lives here so every backend shares one implementation:

	- python/scripts/onnx_yunet.py     (cv2.FaceDetectorYN decode on the CPU)
	- python/onnxtop_tables.py    (C++ OnnxTOP decode on the GPU -> Info DAT, Tracker Mode 'face')

Input detections are dicts in TD space (0-1, origin bottom-left):
	{'box': [x_left, y_bottom, x_right, y_top], 'score': float,
	 'keypoints': [[x, y], ...] (5: eye_r, eye_l, nose, mouth_r, mouth_l)}
"""
import math

import numpy as np

import object_tracker

ByteTracker = object_tracker.ByteTracker

# ==================== CONFIGURATION ====================
# YuNet's own facial keypoints (fixed by the model, not configurable): right eye, left
# eye, nose tip, right mouth corner, left mouth corner.
NUM_KEYPOINTS = 5
KEYPOINT_NAMES = ['eye_r', 'eye_l', 'nose', 'mouth_r', 'mouth_l']

# Confidence threshold for a detected face to be shown/tracked (0.0 - 1.0). This is
# ByteTracker's "high confidence" threshold -- detections at or above this score are
# matched first and can start brand-new tracks. YuNet's own scores run fairly high for
# real faces, unlike e.g. onnx_yolo26_pose.py's degenerate low-confidence scene.
CONF_THRESHOLD = 0.7

# ByteTracker's "low confidence" threshold. Detections scoring between this and
# CONF_THRESHOLD are never used to start a new track, but ARE used in a second
# association pass to recover existing tracks that a plain confidence-thresholded
# detector would otherwise drop (occlusion, motion blur, partial visibility, head turn).
LOW_CONF_THRESHOLD = 0.3

# Minimum box width/height (normalized 0-1, fraction of frame dimension) for a detection
# to be kept at all -- applied alongside Confthreshold, before NMS/tracking. Separate
# width/height floors, NOT one shared value -- see the identical comment in
# onnx_yolo26_pose.py/onnx_yolo26_obj_det.py for why a single shared threshold ends up
# rejecting real, legitimately-large-but-narrow detections once tuned high enough to
# reject noise.
MIN_BOX_WIDTH = 0.02
MIN_BOX_HEIGHT = 0.02

# Tracker: max frames to keep a lost track alive. Faces are prone to brief full
# occlusion (head turn, hand pass, walking behind something) more than a general object
# detector's targets, hence the longer buffer than onnx_yolo26_obj_det.py's default.
TRACKER_MAX_AGE = 45

# Tracker: min IoU to accept a match between a track and a detection.
TRACKER_IOU_THRESHOLD = 0.2

# Tracker: total matched frames (not necessarily consecutive -- see object_tracker.Track's
# confirmed/hits) a brand-new track needs before it's confirmed and shown/output at all.
# Cuts down on single-frame noise "detections" registering as a real face -- almost none
# of that noise ever gets a second real match at all before track_buffer prunes it.
TRACKER_MIN_HITS = 3

# Smoothing factor for box position/size lerp (0 = no smoothing, 1 = frozen). Same role
# as Outputsmoothing in onnx_yolo26_obj_det.py -- ByteTracker's Kalman filter already
# smooths motion prediction, this adds an extra lerp on top of the matched-detection
# snap. Kept lower than onnx_yolo26_obj_det.py's default since a face's own motion
# (head turns) reads as less natural with heavy smoothing.
OUTPUT_SMOOTHING = 0.3

# Which head-pose method to use -- see _compute_head_direction_geometric() (default,
# keypoint-distance-ratio heuristic, no solver) vs _compute_head_direction_solvepnp()
# (3D model + cv2.solvePnP, kept available behind this flag). Geometric is the default:
# far less jittery, since it's bounded distance ratios rather than an ill-conditioned
# 6-DOF solve over 5 sparse near-planar points.
POSE_METHOD = 'geometric'  # or 'solvepnp'

# Degrees-per-ratio scale for the geometric method's yaw/pitch, SEPARATE for each axis
# -- see _compute_head_direction_geometric()'s docstring for why one shared scale
# doesn't work (yaw's and pitch's ratios cover very different natural ranges for the
# same real rotation angle). Tune live: calibrated against each ratio's own observed
# range for a realistic head turn, not against a theoretical +-1 bound.
POSE_YAW_SCALE = 340.0
POSE_PITCH_SCALE = 100.0

# Multiplier on the assumed camera focal length used only by the solvepnp method -- see
# _compute_head_direction_solvepnp's focal_scale doc. No real camera calibration
# available, so this is a stand-in meant to be tuned live: it's the single biggest
# source of scale error in that method's derived rotation (wrong focal length directly
# biases how large a rotation the solve thinks is needed to explain a given 2D keypoint
# displacement).
POSE_FOCAL_SCALE = 1.0

# Smoothing factor for yaw/pitch/roll specifically -- separate from OUTPUT_SMOOTHING,
# which only smooths box position/size. Applies regardless of which Posemethod is
# active -- both methods' raw per-frame angles can be visibly jittery even when
# genuinely holding still (the geometric method less so, but not immune -- keypoint
# noise is keypoint noise). Higher = smoother but more lag following a real head turn.
POSE_SMOOTHING = 0.5


# Rough canonical 3D face model (average adult proportions, NOT subject-specific --
# solvePnP only needs plausible relative proportions to recover orientation, not exact
# measurements) matched 1:1 against YuNet's 5 keypoints. Face-centered coordinate frame
# deliberately matches OpenCV's camera-space convention (X = right, Y = DOWN, Z = away
# from camera), not a "natural" Y-up convention: with Y-up, the (0,0,0) "facing camera"
# initial guess below would actually correspond to upside-down (identity rotation maps
# model axes straight onto camera axes), pulling every solve toward a ~180-degree-pitch
# local optimum. With Y-down, identity rotation genuinely means "upright, facing camera."
#
# Depth (Z) values are intentionally modest relative to the eye/mouth spread (~63mm
# interpupillary distance, nose protruding only ~10-15mm forward of the eye/mouth plane).
# Exaggerating them makes yaw specifically (the axis most dependent on correctly
# interpreting depth from a near-planar point set) badly over/under-sensitive, while
# roll (nearly depth-independent -- basically just the 2D eye-line slope) is unaffected.
_HEAD_MODEL_POINTS = np.array([
	(-31.5, -20.0, -10.0),  # right eye
	( 31.5, -20.0, -10.0),  # left eye
	(  0.0,   0.0,   0.0),  # nose tip (most forward point -- the origin)
	(-25.0,  25.0, -12.0),  # right mouth corner
	( 25.0,  25.0, -12.0),  # left mouth corner
], dtype=np.float64)


def _rotation_matrix_to_euler(R):
	"""Rotation matrix -> (pitch, yaw, roll) in radians, extrinsic X-Y-Z (Tait-Bryan)
	decomposition -- the standard convention used across OpenCV head-pose tutorials."""
	sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
	if sy >= 1e-6:
		pitch = math.atan2(R[2, 1], R[2, 2])
		yaw = math.atan2(-R[2, 0], sy)
		roll = math.atan2(R[1, 0], R[0, 0])
	else:
		pitch = math.atan2(-R[1, 2], R[1, 1])
		yaw = math.atan2(-R[2, 0], sy)
		roll = 0.0
	return pitch, yaw, roll


def _compute_head_direction_geometric(keypoints, yaw_scale=340.0, pitch_scale=100.0):
	"""Estimate head yaw/pitch/roll from keypoint GEOMETRY directly, no 3D model or
	solver involved -- the alternative to _compute_head_direction_solvepnp() below,
	switchable via the Posemethod par (this is the default). Uses bounded distance
	RATIOS between the nose and the other 4 keypoints rather than solving a 6-DOF pose
	from 5 sparse points, so it's far less sensitive to keypoint noise (no ill-
	conditioned optimization, no near-planar flip ambiguity). Not a calibrated
	real-world angle, so each ratio is mapped to degrees via its own tunable linear
	scale (Poseyawscale/Posepitchscale) rather than a single shared one: yaw's and
	pitch's ratios cover very different natural ranges for the same real rotation
	(yaw barely reaches ~0.13 at a true 45 degree turn, pitch reaches ~0.45-0.56 at a
	true 45 degree nod), so one shared scale would badly under-drive yaw specifically.

	Yaw: as the head turns, the nose visually moves toward the near-side eye/mouth
	  corner and away from the far side -- (nose-to-right distances) vs (nose-to-left
	  distances), summed across both eye and mouth for noise-averaging, turned into a
	  ratio and scaled to degrees. The ratio saturates well before +-1 in practice (a
	  real face is only reliably detectable up to some self-occlusion limit, the far
	  corner vanishing, well before the mathematical 90 degrees) -- Poseyawscale is
	  calibrated against the ratio's actual observed range, not a theoretical +-1 bound.
	Pitch: identical idea, vertically -- nose distance to the eye-line vs the mouth-line.
	  Has a small inherent baseline offset even at a neutral pose (real faces aren't
	  perfectly symmetric top/bottom around the nose -- the mouth typically sits farther
	  from the nose than the eyes do), left uncorrected since the right baseline varies
	  per-face/per-camera-setup and hardcoding one measured against a synthetic model
	  risks being wrong for a real face.
	Roll: eye-line tilt -- already a genuine, robust real angle, same as the solvePnP
	  version's roll below.

	Returns (0.0, 0.0, 0.0) if keypoints are missing/degenerate.
	"""
	if not keypoints or len(keypoints) < NUM_KEYPOINTS:
		return 0.0, 0.0, 0.0

	eye_r = np.array(keypoints[0])
	eye_l = np.array(keypoints[1])
	nose = np.array(keypoints[2])
	mouth_r = np.array(keypoints[3])
	mouth_l = np.array(keypoints[4])

	dist_r = np.linalg.norm(nose - eye_r) + np.linalg.norm(nose - mouth_r)
	dist_l = np.linalg.norm(nose - eye_l) + np.linalg.norm(nose - mouth_l)
	# (dist_r - dist_l), NOT (dist_l - dist_r) -- a positive real yaw must produce a
	# positive ratio here; the reversed order gives the opposite sign.
	yaw_ratio = (dist_r - dist_l) / max(dist_l + dist_r, 1e-6)

	eye_mid = (eye_r + eye_l) * 0.5
	mouth_mid = (mouth_r + mouth_l) * 0.5
	dist_eyes = np.linalg.norm(nose - eye_mid)
	dist_mouth = np.linalg.norm(nose - mouth_mid)
	# (dist_eyes - dist_mouth), NOT (dist_mouth - dist_eyes) -- sign matters here the
	# same way it does for yaw_ratio above.
	pitch_ratio = (dist_eyes - dist_mouth) / max(dist_mouth + dist_eyes, 1e-6)

	dx = eye_l[0] - eye_r[0]
	dy = eye_l[1] - eye_r[1]
	roll = math.degrees(math.atan2(dy, dx))

	yaw_ratio = max(-1.0, min(1.0, yaw_ratio))
	pitch_ratio = max(-1.0, min(1.0, pitch_ratio))
	return yaw_ratio * yaw_scale, pitch_ratio * pitch_scale, roll


def _compute_head_direction_solvepnp(keypoints, width, height, focal_scale=1.0):
	"""Estimate real 3D head rotation -- yaw, pitch, roll, all in DEGREES -- from the 5
	facial keypoints (TD-normalized coords) via cv2.solvePnP. Matches the 5 detected
	keypoints against _HEAD_MODEL_POINTS (a rough canonical 3D face, not subject-specific)
	using an approximate camera intrinsic matrix -- a genuine, if approximate, 3D
	orientation estimate, directly usable for driving an actual 3D rotation (e.g. a
	cube's ry to visualize head turn), unlike _compute_head_direction_geometric()'s
	unitless ratios. cv2.solvePnP on this few points is a cheap numerical solve (not a
	model inference) -- negligible cost next to the detector itself.

	focal_scale: multiplier on the assumed focal length (focal_length = width *
	  focal_scale) -- there's no real camera calibration available, so this is a
	  standard stand-in, and the single biggest source of scale error (wrong focal
	  length directly biases how large a rotation the solve thinks is needed to explain
	  a given 2D keypoint displacement). Exposed as a tunable par (Posefocalscale)
	  specifically so it can be calibrated live against your actual camera/lens by
	  comparing a real head turn to the displayed rotation, rather than guessed once
	  and left wrong.

	yaw = rotation about the vertical axis -- left/right head turn (this is what you
	  want for e.g. a cube's Y-rotation).
	pitch = rotation about the horizontal axis -- up/down head tilt/nod.
	roll = rotation about the depth axis -- ear-to-shoulder tilt (matches the old
	  heuristic's roll closely, since that one was already a real angle).

	Returns (0.0, 0.0, 0.0) if keypoints are missing or the solve fails.
	"""
	if not keypoints or len(keypoints) < NUM_KEYPOINTS:
		return 0.0, 0.0, 0.0

	# Undo TD's normalization/Y-flip to get back to standard image-space pixel coords
	# (origin top-left, Y down) that cv2.solvePnP expects.
	image_points = np.array([
		(kx * width, (1.0 - ky) * height) for kx, ky in keypoints[:NUM_KEYPOINTS]
	], dtype=np.float64)

	focal_length = width * focal_scale
	center = (width / 2.0, height / 2.0)
	camera_matrix = np.array([
		[focal_length, 0.0, center[0]],
		[0.0, focal_length, center[1]],
		[0.0, 0.0, 1.0],
	], dtype=np.float64)
	dist_coeffs = np.zeros((4, 1))

	# ITERATIVE with an explicit extrinsic guess, NOT the default DLT-based init (needs
	# >=6 points, fails on YuNet's 5) and NOT a bare EPNP call either -- EPNP alone can
	# pick a wildly wrong solution (e.g. pitch ~162 degrees, "facing away and upside
	# down"), a real flip ambiguity inherent to near-planar point sets like a face (two
	# very different rotations can project to nearly the same 2D points). Seeding with
	# a plausible "facing the camera, at a reasonable distance" guess and refining via
	# Levenberg-Marquardt converges to the physically sensible solution instead, and
	# incidentally sidesteps ITERATIVE's 6-point minimum (which only applies to its own
	# internal DLT initialization, bypassed by the explicit guess).
	rvec_guess = np.zeros((3, 1), dtype=np.float64)
	tvec_guess = np.array([[0.0], [0.0], [focal_length]], dtype=np.float64)
	import cv2
	success, rotation_vector, _ = cv2.solvePnP(
		_HEAD_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
		rvec_guess, tvec_guess, useExtrinsicGuess=True,
		flags=cv2.SOLVEPNP_ITERATIVE,
	)
	if not success:
		return 0.0, 0.0, 0.0

	rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
	pitch, yaw, roll = _rotation_matrix_to_euler(rotation_matrix)
	return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


# ==================== CUSTOM PARAMETERS ====================
# (name, label, default, min, max, clampMax, help) -- one table so the Python script and the
# OnnxTopTables COMP expose the identical tuning surface. 'Posemethod' is a menu (see below).
PARAMETERS = [
	('Confthreshold', 'Confidence Threshold', CONF_THRESHOLD, 0.0, 1.0, True,
		object_tracker.par_help('Confthreshold', extra=(
			"YuNet's own scores for real faces run fairly high (0.7-0.9+ typical)."
		))),
	('Lowconfthreshold', 'Low Confidence Threshold (Recovery)', LOW_CONF_THRESHOLD, 0.0, 1.0, True,
		object_tracker.par_help('Lowconfthreshold')),
	('Minboxwidth', 'Min Box Width', MIN_BOX_WIDTH, 0.0, 0.2, False,
		object_tracker.par_help('Minboxwidth', shape_note=(
			"a face is rarely square, so one shared threshold high enough to reject noise ends up "
			"rejecting real faces that are just proportionally narrow (e.g. a turned head)."
		))),
	('Minboxheight', 'Min Box Height', MIN_BOX_HEIGHT, 0.0, 0.2, False,
		object_tracker.par_help('Minboxheight')),
	('Tracklossframes', 'Track Loss Frames', TRACKER_MAX_AGE, 0.0, 90.0, False,
		object_tracker.par_help('Tracklossframes', extra=(
			"Kept longer than the other scripts' default here since faces are especially prone to "
			"brief full occlusion (head turn, hand pass) where NO detection fires at all, unlike a "
			"body/box detector that often still gets a degraded read through partial occlusion."
		))),
	('Trackiouthreshold', 'Track IoU Threshold', TRACKER_IOU_THRESHOLD, 0.0, 1.0, True,
		object_tracker.par_help('Trackiouthreshold', subject='face')),
	('Trackconfirmframes', 'Track Confirm Frames', TRACKER_MIN_HITS, 1.0, 30.0, False,
		object_tracker.par_help('Trackconfirmframes', subject='face')),
	('Outputsmoothing', 'Output Smoothing', OUTPUT_SMOOTHING, 0.0, 1.0, True,
		object_tracker.par_help('Outputsmoothing', extra=(
			"Keypoints/head direction have their OWN separate smoothing (Pose Smoothing below) -- "
			"they're held at their last real value across occlusion regardless, see the pipeline docstring. "
			"Kept lower than the other scripts' default since heavy smoothing reads as unnatural on a "
			"face's own head-turn motion."
		))),
	('Posemethod', 'Head Pose Method', POSE_METHOD, None, None, None,
		("Which method estimates yaw/pitch/roll. Geometric: bounded distance ratios "
		"between the nose and the other 4 keypoints, no solver -- far less sensitive to keypoint "
		"noise, not a calibrated real angle (see Poseyawscale/Posepitchscale). SolvePnP: matches "
		"keypoints against a rough 3D face model via cv2.solvePnP -- a genuine 3D pose estimate, "
		"but sensitive to noise on only 5 sparse near-planar points (see Posefocalscale). Roll is "
		"identical either way -- it was never the problem, just the eye-line's 2D tilt angle.")),
	('Poseyawscale', 'Head Pose Yaw Scale (Geometric)', POSE_YAW_SCALE, 1.0, 1000.0, False,
		("Degrees-per-ratio scale for yaw specifically (only used when Head Pose Method "
		"= Geometric) -- kept separate from Head Pose Pitch Scale since yaw's underlying ratio "
		"covers a much smaller natural range than pitch's for the same real rotation. Tune live: "
		"if head turns look too subtle, increase; too exaggerated, decrease.")),
	('Posepitchscale', 'Head Pose Pitch Scale (Geometric)', POSE_PITCH_SCALE, 1.0, 1000.0, False,
		("Degrees-per-ratio scale for pitch specifically (only used when Head Pose Method "
		"= Geometric) -- see Head Pose Yaw Scale for why this is separate. Tune live: if head nods "
		"look too subtle, increase; too exaggerated, decrease.")),
	('Posefocalscale', 'Head Pose Focal Scale (SolvePnP)', POSE_FOCAL_SCALE, 0.1, 5.0, False,
		("Multiplier on the assumed camera focal length (only used when Head Pose Method "
		"= SolvePnP) -- no real camera calibration available, so this is a stand-in you can "
		"calibrate live: if head-turn rotation looks too subtle for the real motion, increase; "
		"if it looks exaggerated, decrease. Does not affect box/keypoint detection at all, only "
		"the derived rotation angles.")),
	('Posesmoothing', 'Head Pose Smoothing', POSE_SMOOTHING, 0.0, 1.0, True,
		("Extra lerp on yaw/pitch/roll specifically (separate from Output Smoothing, "
		"which only smooths box position/size), applied regardless of Head Pose Method -- "
		"keypoint detection noise on just 5 sparse points can make raw per-frame angles look "
		"jittery even when not moving. 0 = raw/no smoothing, 1 = frozen in place.")),
]
POSE_METHODS = (['geometric', 'solvepnp'], ['Geometric (Keypoint Ratios)', 'SolvePnP (3D Model)'])


def append_parameters(page, owner, defaults=None):
	"""Create the tracking pars on `page` (replacing existing ones) and set them to defaults."""
	defaults = defaults or {}
	for name, label, default, vmin, vmax, clamp_max, help_text in PARAMETERS:
		default = defaults.get(name, default)
		if name == 'Posemethod':
			p = page.appendMenu(name, label=label, size=1)
			p[0].menuNames, p[0].menuLabels = POSE_METHODS
		else:
			p = page.appendFloat(name, label=label, size=1)
			p[0].min = vmin
			p[0].max = vmax
			p[0].clampMin = True
			p[0].clampMax = clamp_max
		p[0].default = default
		p[0].help = help_text
		setattr(owner.par, name, default)


def read_parameters(owner, defaults=None):
	"""Current par values from `owner` (falling back to defaults), keyed by par name."""
	defaults = defaults or {}
	values = {}
	for name, _, default, *_ in PARAMETERS:
		par = getattr(owner.par, name, None)
		values[name] = par.eval() if par is not None else defaults.get(name, default)
	return values


def filter_detections(detections, low_conf_threshold, min_box_width, min_box_height):
	"""Admission gate: keep everything down to the LOW threshold (ByteTracker splits high/low
	itself for its recovery pass) that clears the min box size. No NMS here -- both backends'
	decoders already ran one (cv2.FaceDetectorYN internally, OnnxTOP's NMS IoU Threshold)."""
	return [
		d for d in detections
		if d['score'] > low_conf_threshold
		and d['box'][2] - d['box'][0] >= min_box_width
		and d['box'][3] - d['box'][1] >= min_box_height
	]


class FaceTrackerPipeline:
	"""Owns the ByteTracker plus per-track box/keypoint/head-pose smoothing state. One
	instance per video stream.

	Box position/size is smoothed on top of the tracker's own Kalman estimate
	(self._box_state). Facial keypoints and head direction are the face-specific extras:
	keypoints are positions, same as the box, so they're smoothed the same way
	(Outputsmoothing), and held at their last smoothed value across lost frames rather than
	reset. Head direction gets its OWN separate smoothing on top (Posesmoothing) -- deriving
	an angle from just 5 sparse keypoints amplifies real detection noise more than the
	(already-smoothed) keypoints themselves visibly show.
	"""

	def __init__(self):
		self.tracker = ByteTracker(
			high_thresh=CONF_THRESHOLD, low_thresh=LOW_CONF_THRESHOLD,
			match_thresh=TRACKER_IOU_THRESHOLD, track_buffer=TRACKER_MAX_AGE,
			min_hits=TRACKER_MIN_HITS,
		)
		self.conf_threshold = CONF_THRESHOLD
		self.low_conf_threshold = LOW_CONF_THRESHOLD
		self._box_state = {}
		self._kpt_state = {}
		# Structured tracking data for CHOP/table consumption; see update() for the dict keys.
		self.tracked_objects = []

	def update(self, detections, conf_threshold, low_conf_threshold, track_iou_threshold, track_buffer,
			min_hits, smoothing, pose_method=POSE_METHOD, yaw_scale=POSE_YAW_SCALE,
			pitch_scale=POSE_PITCH_SCALE, focal_scale=POSE_FOCAL_SCALE, pose_smoothing=POSE_SMOOTHING,
			frame_w=1.0, frame_h=1.0, size_aspect=1.0):
		"""Advance the tracker one frame. `detections` must already be admission-filtered
		(see filter_detections()). frame_w/frame_h: the source frame's size (solvePnP's
		camera model). size_aspect: the source aspect (w/h) when the detector saw the frame
		squeezed to a square (onnx_yunet.py's fit_square 'fill' input), else 1.0 -- see
		docs/learnings/debug-comp-camera-aspect.md "Bug 3". Returns the displayable tracks."""
		self.conf_threshold = conf_threshold
		self.low_conf_threshold = low_conf_threshold
		self.tracker.high_thresh = conf_threshold
		self.tracker.low_thresh = low_conf_threshold
		self.tracker.match_thresh = track_iou_threshold
		self.tracker.track_buffer = track_buffer
		self.tracker.min_hits = int(min_hits)

		active_tracks = self.tracker.update(detections)
		active_ids = {t.track_id for t in active_tracks}

		self.tracked_objects = []
		for t in active_tracks:
			# t.confirmed gates display, same reasoning as the score check -- a track still
			# in its min_hits confirmation window is genuinely "alive" (kept in active_ids,
			# keeps its tracker-side Kalman/box/keypoint state) but not yet trusted as a real
			# face, exactly like a track kept alive only by low-confidence recovery.
			if t.score < conf_threshold or not t.confirmed:
				continue
			box = t.box  # Kalman estimate
			smoothed = object_tracker.box_smooth(self._box_state, t.track_id, box, smoothing)

			# Keypoints/head direction: only advance on a frame where this track actually
			# matched a real detection -- a Kalman-predicted-only frame (lost_frames > 0) has
			# no new keypoint data at all. Held state persists across occlusion.
			raw_kpts = t.payload.get('keypoints')
			if t.lost_frames == 0 and raw_kpts is not None:
				# Smoothed BEFORE head direction is derived from them, so it's a genuine noise
				# reduction on the pose input -- complements Posesmoothing's separate lerp on
				# the resulting yaw/pitch/roll.
				prev_pose = self._kpt_state.get(t.track_id)
				if prev_pose is not None and prev_pose.get('keypoints') is not None:
					prev_kpts = prev_pose['keypoints']
					smoothed_kpts = [
						[prev_kpts[k][0] * smoothing + raw_kpts[k][0] * (1.0 - smoothing),
						 prev_kpts[k][1] * smoothing + raw_kpts[k][1] * (1.0 - smoothing)]
						for k in range(NUM_KEYPOINTS)
					]
				else:
					smoothed_kpts = [list(kp) for kp in raw_kpts]

				if pose_method == 'solvepnp':
					yaw, pitch, roll = _compute_head_direction_solvepnp(smoothed_kpts, frame_w, frame_h, focal_scale)
				else:
					yaw, pitch, roll = _compute_head_direction_geometric(smoothed_kpts, yaw_scale, pitch_scale)
				if prev_pose is not None:
					yaw = prev_pose['yaw'] * pose_smoothing + yaw * (1.0 - pose_smoothing)
					pitch = prev_pose['pitch'] * pose_smoothing + pitch * (1.0 - pose_smoothing)
					roll = prev_pose['roll'] * pose_smoothing + roll * (1.0 - pose_smoothing)
				self._kpt_state[t.track_id] = {'keypoints': smoothed_kpts, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
			state = self._kpt_state.get(t.track_id, {'keypoints': [[0.0, 0.0]] * NUM_KEYPOINTS, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0})

			cx = (smoothed[0] + smoothed[2]) / 2
			cy = (smoothed[1] + smoothed[3]) / 2
			w_ = (smoothed[2] - smoothed[0]) / math.sqrt(size_aspect)
			h_ = (smoothed[3] - smoothed[1]) * math.sqrt(size_aspect)
			self.tracked_objects.append({
				'track_id': t.track_id,
				'score': t.score,
				'cx': cx, 'cy': cy, 'w': w_, 'h': h_,
				'x_left': smoothed[0],
				'x_right': smoothed[2],
				'y_top': smoothed[3],     # top edge of bbox (TD coords)
				'y_bottom': smoothed[1],  # bottom edge of bbox (TD coords)
				'vx': float(t.mean[4]), 'vy': float(t.mean[5]),  # Kalman-estimated box-center velocity
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
				'keypoints': state['keypoints'],
				'yaw': state['yaw'], 'pitch': state['pitch'], 'roll': state['roll'],
			})

		# Prune box/keypoint state for tracks the tracker has dropped entirely.
		object_tracker.prune_stale(active_ids, self._box_state, self._kpt_state)
		return self.tracked_objects

	def write_tracks_table(self, tbl):
		"""table_output: onnx_yolo26_obj_det.py's core columns (minus class_id/class_name -- a
		face detector has no classes), plus yaw/pitch/roll and keypoints, so consumers that
		read by column NAME (Debug's geo/select nodes, DebugBB) work unchanged."""
		if tbl is None:
			return
		kpt_header = []
		for name in KEYPOINT_NAMES:
			kpt_header += [f'{name}_x', f'{name}_y']
		tbl.clear()
		# yaw/pitch/roll are all in DEGREES, but exactly how "real" depends on Posemethod --
		# see _compute_head_direction_geometric()/_compute_head_direction_solvepnp().
		tbl.appendRow([
			*object_tracker.label_header(),
			*object_tracker.box_header(),
			'yaw', 'pitch', 'roll',
			*kpt_header,
			*object_tracker.color_header(),
		])
		for obj in self.tracked_objects:
			flat_kpts = [v for kp in obj['keypoints'] for v in kp]  # 5*2 flat list
			tbl.appendRow([
				*object_tracker.label_row(obj['track_id'], obj['score']),
				*object_tracker.box_row(obj),
				f"{obj['yaw']:.4f}", f"{obj['pitch']:.4f}", f"{obj['roll']:.4f}",
				*[f"{v:.4f}" for v in flat_kpts],
				*object_tracker.color_row(obj['track_id']),
			])

	def write_joints_table(self, tbl):
		"""Flat per-keypoint table_joints (object_tracker.joints_header()). YuNet's 5 keypoints
		are 2D-only (no z) and have no per-point confidence, so both are constant stand-ins
		(0.0 / 1.0). No bones -- eyes/nose/mouth don't form a skeleton."""
		if tbl is None:
			return
		tbl.clear()
		tbl.appendRow(object_tracker.joints_header())
		for obj in self.tracked_objects:
			track_id = obj['track_id']
			for name, (kx, ky) in zip(KEYPOINT_NAMES, obj['keypoints']):
				tbl.appendRow(object_tracker.joints_row(track_id, name, kx, ky, 0.0, 1.0))
