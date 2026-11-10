"""Backend-agnostic pose tracking: detections in, tracked/smoothed tables out.

Everything YOLO26 pose post-processing does AFTER the model-specific decode -- KeypointTracker
matching, per-keypoint + box smoothing, visibility hysteresis, display gating, and the
table_output / table_joints / table_bones formats -- lives here so every backend shares one
implementation:

	- python/scripts/onnx_yolo26_pose.py  (Python ONNX Runtime, numpy decode)
	- python/onnxtop_tables.py       (C++ OnnxTOP, decode done on the GPU -> Info DAT)

Input detections are dicts in TD space (0-1, origin bottom-left):
	{'box': [x_left, y_bottom, x_right, y_top], 'score': float, 'keypoints': [[x, y, conf], ...]}
optionally with [x, y, conf, z] keypoints, a 'label' (e.g. MediaPipe handedness / gesture, shown
as "<id> <label>") and 'extras' ({column name: value}, e.g. face blendshapes; smoothed like the
keypoints and written as extra table_output columns), and 'world' ([[x, y, z], ...] in metres,
TD axes: MediaPipe world landmarks for the skeleton's first world_count keypoints; smoothed like
the keypoints and written as <name>_wx/_wy/_wz columns).

The keypoint layout is a mediapipe_skeletons.Skeleton: COCO-17 by default (YOLO26-pose, MoveNet),
or MP_POSE / MP_HANDS / MP_FACE for MediaPipeTOP.

Usage:
	pipeline = PoseTrackerPipeline()
	tracked = pipeline.update(detections, **pipeline_params)   # once per frame
	pipeline.write_tracks_table(dat); pipeline.write_joints_table(dat); pipeline.write_bones_table(dat)
"""
import math

import mediapipe_skeletons
import object_tracker

# Standard COCO-17 layout (matches the YOLO26 pose export's 'kpt_names' metadata), kept as
# module constants for scripts that import them. Defined in mediapipe_skeletons.COCO_17.
KEYPOINT_NAMES = mediapipe_skeletons.COCO_17.names
NUM_KEYPOINTS = mediapipe_skeletons.COCO_17.count  # 17
SKELETON_EDGES = mediapipe_skeletons.COCO_17.edges

# Keypoint indices used for KeypointTracker's matching distance -- the STABLE core
# (face, shoulders, hips), excluding elbows/wrists/knees/ankles, which move the most
# frame-to-frame and carry the most jitter. See onnx_movenet.py's identical constant
# for the full rationale (that script's KeypointTracker port is the reference).
DISTANCE_KEYPOINT_INDICES = mediapipe_skeletons.COCO_17.distance_indices

# ==================== DEFAULTS (tuned live -- see comments) ====================

# Confidence threshold for a detected person to be shown/tracked (0.0 - 1.0). Tuned live
# for this camera/scene, where real-signal confidence runs extremely low (fractions of a
# percent) -- not a typical detector threshold. KeypointTracker has no ByteTrack-style
# two-stage high/low recovery band, so this is a single admission/display gate.
CONF_THRESHOLD = 0.005

# Tracker: max frames to keep a lost track alive. Tuned live: ~0.25s of grace at 60fps.
TRACKER_MAX_AGE = 15

# Tracker: total matched frames (not necessarily consecutive -- see object_tracker.Track's
# confirmed/hits) a brand-new track needs before it's confirmed and shown/output at all.
# Cuts down on both overlap with existing boxes and single-frame noise "detections"
# registering as a real person -- almost none of that noise ever gets a second real match
# at all before track_buffer prunes it. Costs a few frames of extra latency on every
# genuinely new person's first appearance; once confirmed a track stays confirmed through
# brief occlusion (that's Tracklossframes' job, not this one).
TRACKER_MIN_HITS = 3

# Tracker: max summed keypoint distance (object_tracker.KeypointTracker, restricted to
# DISTANCE_KEYPOINT_INDICES above) to accept a match between a track and a detection --
# NOT a box IoU threshold. This model uses KeypointTracker (keypoint-distance matching)
# rather than box-IoU + Kalman tracking; see Round 9 in td-threaded-inference-optimization.md
# and object_tracker.py's "POSE-SPECIFIC TRACKER" section for why. Starting value carried
# over from onnx_movenet.py's own tuning; re-tune live against this scene.
MAX_MATCH_DIST = 0.5

# Scale-relative matching (the default): mean stable-keypoint distance in units of the skeleton's
# box diagonal, so one value works for small distant and large close people. 0.15 matches
# MAX_MATCH_DIST for a person about half the frame tall. Off = MAX_MATCH_DIST as before.
SCALE_MATCH = True
REL_MATCH_DIST = 0.15

# Fraction of MAX_MATCH_DIST used as the duplicate-detection suppression threshold --
# see KeypointTracker.update()'s docstring for why this needs to be tighter than the
# match threshold itself.
DUP_DIST_FACTOR = 0.5

# Minimum box width/height (normalized 0-1, fraction of frame dimension) for a detection
# to be kept at all -- applied alongside Confthreshold, before tracking. A tiny,
# degenerate box is almost never a real person regardless of confidence, so a size floor
# catches noise a confidence threshold alone lets through. Separate width/height floors,
# NOT one shared value: a standing person's box is naturally much narrower than tall
# (~0.4-0.6 width/height is typical), so a single threshold high enough to reject
# small-on-both-axes noise blobs also rejects real, legitimately-tall-but-narrow people.
# Tune independently.
MIN_BOX_WIDTH = 0.02
MIN_BOX_HEIGHT = 0.02

# Smoothing factor for keypoint AND box position lerp -- used for both the debug draw
# and the output table (0 = no smoothing, 1 = frozen). KeypointTracker holds the box at
# its last-matched value with no damping of its own (see object_tracker.KeypointTrack's
# docstring), so this is the ONLY smoothing applied to box position. Tuned for the
# keypoint case originally -- re-verify it still suits box motion too.
OUTPUT_SMOOTHING = 0.23

# Skeletons whose parts can go missing (hide_missing: holistic's hands and face): a part that
# comes back is seeded from its first reading, then smoothed but kept hidden for this many
# matched frames, so it shows up already settled rather than snapping or gliding in.
PART_CONFIRM_FRAMES = 3
# A track matched again after at least this many lost frames snaps to its fresh detection instead
# of gliding there from where it was held (or coasted) while lost.
REACQUIRE_SNAP_FRAMES = 2


# ==================== CUSTOM PARAMETERS ====================
# (name, label, default, min, max, clampMax, help) -- one table so the Python script and the
# OnnxTopTables COMP expose the identical tuning surface.
PARAMETERS = [
	('Confthreshold', 'Confidence Threshold', CONF_THRESHOLD, 0.0, 0.05, False,
		# real signal in this scene lives near ~0.005 -- 0-1 made the slider unusably coarse
		"Minimum detection confidence for a person to be admitted to the tracker at all, "
		"and to stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage "
		"high/low recovery band -- ages out purely via Track Loss Frames/score decay instead)."),
	('Outputsmoothing', 'Output Smoothing', OUTPUT_SMOOTHING, 0.0, 1.0, True,
		object_tracker.par_help('Outputsmoothing', what='keypoint position')),
	('Tracklossframes', 'Track Loss Frames', TRACKER_MAX_AGE, 0.0, 90.0, False,
		object_tracker.par_help('Tracklossframes')),
	('Scalematch', 'Scale Match To Skeleton', SCALE_MATCH, 0.0, 1.0, True,
		"Match tracks by how far keypoints moved relative to the skeleton's size (Relative Match "
		"Distance), so small distant and large close-up skeletons need no re-tuning. Off = the "
		"absolute Max Match Distance."),
	('Relmatchdist', 'Relative Match Distance', REL_MATCH_DIST, 0.0, 1.0, False,
		"With Scale Match To Skeleton on: how far, on average, the stable keypoints may move between "
		"frames and still be the same track, as a fraction of the skeleton's size (its box diagonal). "
		"Too low breaks tracks on fast movement; too high lets neighbours swap ids."),
	('Maxmatchdist', 'Max Match Distance', MAX_MATCH_DIST, 0.0, 2.0, False,
		object_tracker.par_help('Maxmatchdist', subject='person') +
		' Used only with Scale Match To Skeleton off.'),
	('Trackconfirmframes', 'Track Confirm Frames', TRACKER_MIN_HITS, 1.0, 30.0, False,
		object_tracker.par_help('Trackconfirmframes', subject='person')),
	('Minboxwidth', 'Min Box Width', MIN_BOX_WIDTH, 0.0, 0.2, False,
		object_tracker.par_help('Minboxwidth', shape_note=(
			"a standing person is naturally much narrower than tall, so one shared threshold high "
			"enough to reject noise ends up rejecting real, legitimately tall-but-narrow people."
		))),
	('Minboxheight', 'Min Box Height', MIN_BOX_HEIGHT, 0.0, 0.2, False,
		object_tracker.par_help('Minboxheight')),
]


# Tuned defaults for MoveNet multipose (from onnx_movenet.py). Its person scores live in a
# normal 0-1 range, unlike YOLO26-pose's ~0.005, and its box is derived from keypoints.
MOVENET_DEFAULTS = {
	'Confthreshold': 0.1,
	'Outputsmoothing': 0.4,
	'Minboxwidth': 0.05,
	'Minboxheight': 0.05,
}


# ByteTrack-style persistence for MediaPipe modes, on top of PARAMETERS. MediaPipe reports a
# hand/pose/face or nothing (no low-score detections to recover a track from), so instead of
# ByteTrack's low-confidence band these carry a lost track along its last motion and give it
# a smooth presence ramp.
PERSISTENCE_PARAMETERS = [
	('Coastwhilelost', 'Coast While Lost', 0.0, 0.0, 1.0, True,
		"While a track is lost (up to Track Loss Frames), keep moving its keypoints and box along "
		"its last frame-to-frame velocity, multiplied by this factor every lost frame (0 = freeze in "
		"place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman prediction, it bridges "
		"a fast-moving hand that MediaPipe drops for a frame or two."),
	('Presencerise', 'Presence Rise Frames', 3.0, 1.0, 60.0, False,
		object_tracker.par_help('Presencerise')),
	('Presencefall', 'Presence Fall Frames', 3.0, 1.0, 90.0, False,
		object_tracker.par_help('Presencefall')),
]

# Debounced labels: MediaPipe's gesture label flickers between gestures (and None) on
# transition frames. Only used where the detections carry a label (MediaPipe Hands / Gesture).
LABEL_PARAMETERS = [
	('Gestureholdframes', 'Gesture Hold Frames', 3.0, 1.0, 30.0, False,
		"A new gesture must be the top result for this many consecutive frames before the track's "
		"gesture column switches to it. Higher = steadier, but triggers later. 1 = raw MediaPipe "
		"output."),
	('Gestureminscore', 'Gesture Min Score', 0.5, 0.0, 1.0, True,
		"A gesture scoring below this counts as None (no gesture). Raise it to cut false triggers."),
]


def append_parameters(page, owner, defaults=None, extra=()):
	"""Create the tracking pars on `page` (replacing existing ones) and set them to defaults.
	`defaults` overrides per-par defaults (e.g. MOVENET_DEFAULTS); `extra` adds par lists such
	as PERSISTENCE_PARAMETERS."""
	defaults = defaults or {}
	for name, label, default, vmin, vmax, clamp_max, help_text in [*PARAMETERS, *extra]:
		default = defaults.get(name, default)
		if isinstance(default, bool):
			p = page.appendToggle(name, label=label)
			p[0].default = default
			p[0].help = help_text
			setattr(owner.par, name, default)
			continue
		p = page.appendFloat(name, label=label, size=1)
		p[0].default = default
		p[0].min = vmin
		p[0].max = max(vmax, 1.0) if default > vmax else vmax	# slider range fits the default
		p[0].clampMin = True
		p[0].clampMax = clamp_max
		p[0].help = help_text
		setattr(owner.par, name, default)


def read_parameters(owner, defaults=None, extra=()):
	"""Current par values from `owner` (falling back to defaults), keyed by par name."""
	defaults = defaults or {}
	values = {}
	for name, _, default, *_ in [*PARAMETERS, *extra]:
		par = getattr(owner.par, name, None)
		values[name] = par.eval() if par is not None else defaults.get(name, default)
	return values


def filter_detections(detections, conf_threshold, min_box_width, min_box_height):
	"""Confidence + min-box-size admission gate, for backends that hand over unfiltered boxes."""
	return [
		d for d in detections
		if d['score'] > conf_threshold
		and d['box'][2] - d['box'][0] >= min_box_width
		and d['box'][3] - d['box'][1] >= min_box_height
	]


def _write_rows(tbl, rows):
	"""Replace a table's rows in one call (hundreds of appendRow calls cost more)."""
	tbl.clear()
	if hasattr(tbl, 'appendRows'):
		tbl.appendRows(rows)
	else:
		for row in rows:
			tbl.appendRow(row)


class PoseTrackerPipeline:
	"""Owns the tracker plus per-track smoothing state. One instance per video stream.

	`object_tracker.KeypointTracker` owns the generic tracking lifecycle: greedy keypoint-
	distance matching, track lifecycle, confidence decay. This class owns what's pose-
	specific: per-keypoint smoothing and visibility hysteresis, keyed by track_id in
	self._kpt_state since KeypointTracker itself doesn't smooth anything. Keypoint visibility
	uses the same hold window as track loss (Tracklossframes/track_buffer) rather than its
	own par, since a pose can't outlive the track it belongs to.
	"""

	def __init__(self, skeleton=None):
		self.skeleton = skeleton or mediapipe_skeletons.COCO_17
		self.tracker = object_tracker.KeypointTracker(
			max_match_dist=self.skeleton.max_match_dist, distance_keypoint_indices=self.skeleton.distance_indices,
			relative_match_dist=self.skeleton.rel_match_dist if SCALE_MATCH else None,
			dup_dist_factor=DUP_DIST_FACTOR, track_buffer=TRACKER_MAX_AGE,
			min_hits=TRACKER_MIN_HITS,
		)
		self.conf_threshold = CONF_THRESHOLD
		self._kpt_state = {}
		self._presence_state = {}
		self._label_state = {}
		self._presence_on = False
		self._labels_on = False
		# Structured tracking data for CHOP/table consumption; see update() for the dict keys.
		self.tracked_objects = []

	def update(self, detections, conf_threshold=CONF_THRESHOLD, smoothing=OUTPUT_SMOOTHING,
			max_match_dist=MAX_MATCH_DIST, track_buffer=TRACKER_MAX_AGE, min_hits=TRACKER_MIN_HITS,
			coast=0.0, relative_match_dist=None, presence=None, label_hold=None, label_min_score=0.0):
		"""Advance the tracker one frame. `detections` must already be admission-filtered
		(see filter_detections()). Returns (and stores) the displayable tracked objects.

		Optional persistence (PERSISTENCE_PARAMETERS / LABEL_PARAMETERS):
			coast        -- per-lost-frame velocity factor for carrying a lost track along (0 = hold still)
			presence     -- (rise_frames, fall_frames): adds a 0-1 'presence' ramp column
			label_hold   -- debounce detection labels (gestures): adds gesture / gesture_score /
			                gesture_frames / gesture_onset columns
			label_min_score -- a label scoring below this counts as 'None'
		"""
		self.conf_threshold = conf_threshold
		self._presence_on = presence is not None
		self._labels_on = label_hold is not None
		self.tracker.max_match_dist = max_match_dist
		self.tracker.relative_match_dist = relative_match_dist
		self.tracker.track_buffer = track_buffer
		self.tracker.min_hits = int(min_hits)
		self.tracker.coast = coast

		# Update tracker (runs on main thread, no lock needed)
		active_tracks = self.tracker.update(detections)

		# Build structured data for CHOP/table output (filter out decayed tracks).
		# active_ids covers every track the tracker is still maintaining, not just the
		# ones cleared for display below -- a track kept alive via low-confidence recovery
		# (score hovering just under Confthreshold) must keep its keypoint-smoothing state,
		# so scoping this to the display-filtered subset would wipe and restart smoothing
		# every frame for any such track.
		active_ids = {t.track_id for t in active_tracks}
		num_kp = self.skeleton.count
		self.tracked_objects = []
		for t in active_tracks:
			# t.confirmed gates display same as the score check: a track still in its
			# min_hits confirmation window is kept in active_ids but not yet shown.
			if t.score < conf_threshold or not t.confirmed:
				continue
			# Last-matched box, held as-is during occlusion (no Kalman filter/prediction --
			# see object_tracker.KeypointTrack's docstring). KeypointTracker applies no
			# damping of its own, so the smoothing lerp below is what keeps this from
			# jumping straight to each new detection's raw box every matched frame.
			box_raw = t.box

			state = self._kpt_state.get(t.track_id)
			raw_kpts = t.payload.get('keypoints')
			raw_extras = t.payload.get('extras') or {}
			raw_world = t.payload.get('world')
			if state is None:
				# Brand-new track: seed smoothed state directly from its first detection.
				state = {
					'smoothed': [list(kp) for kp in raw_kpts] if raw_kpts else [[0.0, 0.0, 0.0] for _ in range(num_kp)],
					'lost_frames': [0] * num_kp,
					'box': list(box_raw),
					'extras': dict(raw_extras),
					'world': None,
					# Matched frames each part has been seen in a row (PART_CONFIRM_FRAMES).
					'seen': [PART_CONFIRM_FRAMES if not raw_kpts or kp[2] > 0.0 else 0 for kp in raw_kpts or range(num_kp)],
				}
				self._kpt_state[t.track_id] = state
				if raw_world:
					self._smooth_world(state, raw_world, raw_kpts, smoothing)
			elif t.lost_frames == 0 and raw_kpts is not None:
				gap = state.get('gap', 0)
				if gap >= REACQUIRE_SNAP_FRAMES:
					# Back after a gap: start from the fresh detection, like a new track.
					state['box'] = list(box_raw)
					state['extras'] = dict(raw_extras)
					state['snap'] = True	# _smooth_world takes this frame's world as-is
					for k, kp in enumerate(raw_kpts):
						if not (self.skeleton.hide_missing and kp[2] <= 0.0):
							state['smoothed'][k] = list(kp)
				# Only advance keypoint smoothing on a frame where this track actually
				# matched a real detection -- a frame with no fresh match (lost_frames > 0)
				# has no new keypoint data at all, so just let the hold counters age below.
				# No per-keypoint confidence gate: this model's per-keypoint confidence
				# values run far too low to be a meaningful signal on their own, so if the
				# box cleared Confthreshold, its keypoints are used as-is.
				hide = self.skeleton.hide_missing
				for k, kp in enumerate(raw_kpts):
					prev = state['smoothed'][k]
					seen = state['seen']
					if hide and kp[2] <= 0.0:
						# Absent part (holistic's unseen hand/face sits at 0, 0): keep the
						# last position, hidden, instead of easing towards the corner.
						prev[2] = kp[2]
						seen[k] = 0
						continue
					seen[k] += 1
					if hide and seen[k] == 1:
						# Back after being absent: start here (hidden until confirmed), don't fly in.
						state['smoothed'][k] = list(kp)
					else:
						# x, y (and z when present) are smoothed; conf is taken as-is.
						state['smoothed'][k] = [
							prev[i] * smoothing + kp[i] * (1.0 - smoothing) if i != 2 and i < len(prev) else kp[i]
							for i in range(len(kp))
						]
					state['lost_frames'][k] = 0
				for name, value in raw_extras.items():
					prev = state['extras'].get(name, value)
					state['extras'][name] = prev * smoothing + value * (1.0 - smoothing)
				if raw_world:
					self._smooth_world(state, raw_world, raw_kpts, smoothing)
				# Same lerp, same "only advance on a frame that actually matched a real
				# detection" cadence as the keypoints above.
				state['box'] = [
					state['box'][i] * smoothing + box_raw[i] * (1.0 - smoothing)
					for i in range(4)
				]
			state['gap'] = t.lost_frames
			if t.lost_frames > 0:
				for k in range(num_kp):
					state['lost_frames'][k] += 1
				if coast > 0.0:
					# Carry the lost track along its last box-center velocity, decaying each frame.
					f = coast ** t.lost_frames
					dx, dy = t.mean[4] * f, t.mean[5] * f
					for kp in state['smoothed']:
						kp[0] += dx
						kp[1] += dy
					state['box'] = [state['box'][0] + dx, state['box'][1] + dy, state['box'][2] + dx, state['box'][3] + dy]

			obj_extra = {}
			if presence is not None:
				obj_extra['presence'] = object_tracker.presence_ramp(
					self._presence_state, t.track_id, t.lost_frames == 0, presence[0], presence[1])
			label = t.payload.get('label')
			if label_hold is not None:
				label = self._stable_label(t, label, label_hold, label_min_score, smoothing)
				obj_extra['gesture'] = label
				obj_extra.update(self._label_state[t.track_id])

			box = state['box']
			self.tracked_objects.append({
				**obj_extra,
				'track_id': t.track_id,
				'score': t.score,
				'cx': (box[0] + box[2]) / 2, 'cy': (box[1] + box[3]) / 2,
				'w': box[2] - box[0], 'h': box[3] - box[1],
				'x_left': box[0],
				'x_right': box[2],
				'y_top': box[3],     # top edge of bbox (TD coords)
				'y_bottom': box[1],  # bottom edge of bbox (TD coords)
				'keypoints': self._confirmed(state),  # skeleton.count x [x, y, conf(, z)], smoothed, TD coords
				'label': label,
				'extras': state['extras'],
				'world': state['world'],  # world_count x [x, y, z] metres, TD axes, or None
				'keypoints_visible': [lost <= self.tracker.track_buffer for lost in state['lost_frames']],
				'vx': float(t.mean[4]), 'vy': float(t.mean[5]),  # frame-to-frame box-center delta (see KeypointTrack.mean)
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
			})

		# Prune keypoint smoothing state for tracks the tracker has dropped entirely.
		object_tracker.prune_stale(active_ids, self._kpt_state, self._presence_state, self._label_state)
		return self.tracked_objects

	def _confirmed(self, state):
		"""The smoothed keypoints, with a part that's back but not yet confirmed hidden (conf 0)."""
		kpts = state['smoothed']
		if not self.skeleton.hide_missing:
			return kpts
		return [kp if n >= PART_CONFIRM_FRAMES else [kp[0], kp[1], 0.0, *kp[3:]] for kp, n in zip(kpts, state['seen'])]

	def _smooth_world(self, state, raw_world, raw_kpts, smoothing):
		"""Advance the world landmarks one matched frame. A joint whose keypoint is absent
		(conf 0: holistic's unseen hand) keeps its last value instead of easing to the origin,
		and one seen for the first time is taken as-is."""
		world = state['world']
		snap = state.pop('snap', False)
		if world is None or len(world) != len(raw_world):
			state['world'] = world = [None] * len(raw_world)
		for k, p in enumerate(raw_world):
			if self.skeleton.hide_missing and raw_kpts and raw_kpts[k][2] <= 0.0:
				continue
			prev = world[k]
			back = snap or (self.skeleton.hide_missing and 'seen' in state and state['seen'][k] == 1)
			world[k] = list(p) if prev is None or back else [prev[i] * smoothing + p[i] * (1.0 - smoothing) for i in range(3)]

	def _stable_label(self, t, raw, hold_frames, min_score, smoothing):
		"""Debounce a track's label: a new label must win `hold_frames` consecutive matched
		frames before it replaces the current one. Updates self._label_state[track_id]
		(gesture_score, gesture_frames, gesture_onset) and returns the stable label."""
		st = self._label_state.get(t.track_id)
		if t.lost_frames == 0:
			score = t.score
			if not raw or score < min_score:
				raw = 'None'
		if st is None:
			# A new track takes its first label immediately (onset = 1 on this frame).
			self._label_state[t.track_id] = {
				'_stable': raw if t.lost_frames == 0 else 'None', '_cand': None, '_cand_n': 0,
				'gesture_score': t.score, 'gesture_frames': 1, 'gesture_onset': 1,
			}
			return self._label_state[t.track_id]['_stable']
		st['gesture_onset'] = 0
		st['gesture_frames'] += 1
		if t.lost_frames == 0:
			if raw == st['_stable']:
				st['_cand'], st['_cand_n'] = None, 0
				st['gesture_score'] = st['gesture_score'] * smoothing + score * (1.0 - smoothing)
			else:
				st['_cand_n'] = st['_cand_n'] + 1 if raw == st['_cand'] else 1
				st['_cand'] = raw
				if st['_cand_n'] >= hold_frames:
					st['_stable'], st['_cand'], st['_cand_n'] = raw, None, 0
					st['gesture_score'] = score
					st['gesture_frames'] = 1
					st['gesture_onset'] = 1
		return st['_stable']

	# ==================== TABLE OUTPUT ====================
	# tracked_objects already holds smoothed values, so these just format them.

	def write_tracks_table(self, tbl):
		"""table_output: label + box columns, flattened keypoints (+ z when the detections
		have it), extras (e.g. blendshapes), track color."""
		has_z = any(len(obj['keypoints'][0]) > 3 for obj in self.tracked_objects if obj['keypoints'])
		suffixes = ('x', 'y', 'conf', 'z') if has_z else ('x', 'y', 'conf')
		extra_names = list(self.tracked_objects[0]['extras']) if self.tracked_objects else []
		world_count = max((len(obj['world']) for obj in self.tracked_objects if obj.get('world')), default=0)
		# Persistence columns (only when enabled, so COCO pose tables keep their layout).
		state_names = (['presence'] if self._presence_on else []) + \
			(['gesture', 'gesture_score', 'gesture_frames', 'gesture_onset'] if self._labels_on else [])

		# The header (thousands of names for Holistic) only changes with its inputs.
		key = (has_z, world_count, tuple(extra_names), tuple(state_names))
		if getattr(self, '_header_key', None) != key:
			self._header_key = key
			self._header = [
				*object_tracker.label_header(),
				*object_tracker.box_header(),
				*state_names,
				*[f'{name}_{s}' for name in self.skeleton.names for s in suffixes],
				*[f'{name}_w{s}' for name in self.skeleton.names[:world_count] for s in 'xyz'],
				*extra_names,
				*object_tracker.color_header(),
			]
		tbl.clear()
		tbl.appendRow(self._header)
		n = len(suffixes)
		for obj in self.tracked_objects:
			flat_kpts = [(kp[i] if i < len(kp) else 0.0) for kp in obj['keypoints'] for i in range(n)]
			label = obj.get('label')
			tbl.appendRow([
				*object_tracker.label_row(obj['track_id'], obj['score'],
					label_text=f"{obj['track_id']} {label}" if label else None),
				*object_tracker.box_row(obj),
				*[f"{v:.4f}" if isinstance(v, float) else v for v in (obj.get(name, '') for name in state_names)],
				*[f"{v:.4f}" for v in flat_kpts],
				*self._world_cells(obj, world_count),
				*[f"{obj['extras'].get(name, 0.0):.4f}" for name in extra_names],
				*object_tracker.color_row(obj['track_id']),
			])

	@staticmethod
	def _world_cells(obj, world_count):
		world = obj.get('world') or []
		cells = []
		for k in range(world_count):
			p = world[k] if k < len(world) and world[k] is not None else (0.0, 0.0, 0.0)
			cells += [f'{v:.4f}' for v in p]
		return cells

	def write_joints_table(self, tbl):
		"""table_joints: one row per visible keypoint across all tracked people."""
		rows = [object_tracker.joints_header()]
		for obj in self.tracked_objects:
			track_id = obj['track_id']
			for name, kp, vis in zip(self.skeleton.names, obj['keypoints'], obj['keypoints_visible']):
				if vis and not (self.skeleton.hide_missing and kp[2] <= 0.0):
					rows.append(object_tracker.joints_row(track_id, name, kp[0], kp[1], kp[3] if len(kp) > 3 else 0.0, kp[2]))
		_write_rows(tbl, rows)

	def write_bones_table(self, tbl):
		"""table_bones: one row per bone whose two keypoints are both visible."""
		rows = [object_tracker.bones_header()]
		for obj in self.tracked_objects:
			track_id = obj['track_id']
			kpts = obj['keypoints']
			vis = obj['keypoints_visible']
			hide = self.skeleton.hide_missing
			for a, b in self.skeleton.edges:
				if vis[a] and vis[b] and not (hide and (kpts[a][2] <= 0.0 or kpts[b][2] <= 0.0)):
					ax, ay, aconf = kpts[a][:3]
					bx, by, bconf = kpts[b][:3]
					dx = bx - ax
					dy = by - ay
					rows.append(object_tracker.bones_row(
						track_id, (ax + bx) / 2.0, (ay + by) / 2.0,
						math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy), min(aconf, bconf)))
		_write_rows(tbl, rows)
