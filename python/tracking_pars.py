"""The tracker COMP's pages in every wrapper: Setup (Tracker Mode, the class score columns) and the
mode's Tracking page, whose parameters the Tracker DAT (cpp/Tracking) is bound to. Data only: the
tables were generated once from the retired Python tracker's own definitions (Nov 13, 2026), so
the pages are the ones the toxes had. A new tracker parameter: add it to cpp/Tracking (TrackerDAT's
kPars, Tracker::par's per-mode default) and here, with the same name and default.

	build_pages(tracker_comp, 'mp_holistic')
"""

STYLES = ('Float', 'Toggle', 'Menu', 'Str')
ATTRS = ('menuNames', 'menuLabels', 'default', 'min', 'max', 'clampMin', 'clampMax', 'normMin', 'normMax', 'help')


def build_pages(comp, mode):
	"""(Re)build the Setup and Tracking pages for `mode` on `comp`, every parameter at its default."""
	for page in [p for p in comp.customPages if p.name in ('Setup', 'Tracking')]:
		page.destroy()
	for page_name, records in (('Setup', SETUP), ('Tracking', TRACKING[mode])):
		page = comp.appendCustomPage(page_name)
		for r in records:
			par = getattr(page, 'append' + r['style'])(r['name'], label=r['label'])[0]
			for key in ATTRS:
				if key in r:
					setattr(par, key, r[key])
			par.val = r['default']
	comp.par.Trackermode = mode


MODES = ['pose', 'movenet', 'face', 'detect', 'segment', 'mp_pose', 'mp_hands', 'mp_face', 'mp_holistic', 'mp_facedet', 'rtmw']

SETUP = [{'name': 'Trackermode',
  'style': 'Menu',
  'label': 'Tracker Mode',
  'menuNames': ['pose',
                'movenet',
                'face',
                'detect',
                'segment',
                'mp_pose',
                'mp_hands',
                'mp_face',
                'mp_holistic',
                'mp_facedet',
                'rtmw'],
  'menuLabels': ['Pose (YOLO26-pose)',
                 'Pose (MoveNet multipose)',
                 'Face (YuNet)',
                 'Detect (YOLO26)',
                 'Segment (YOLO26-seg)',
                 'MediaPipe Pose (33)',
                 'MediaPipe Hands / Gesture (21)',
                 'MediaPipe Face (478)',
                 'MediaPipe Holistic (553)',
                 'MediaPipe Face Detector (6)',
                 'RTMW3D Whole Body (75)'],
  'default': 'pose',
  'help': "The tracker's mode, matching the model: pose models need Pose (YOLO26-pose) or MoveNet, YuNet "
          'needs Face, seg models need Segment, MediaPipe models their MediaPipe mode, RTMW3D its own.'},
 {'name': 'Classprefix',
  'style': 'Str',
  'label': 'Class Column Prefix',
  'default': 'class',
  'help': 'Class Scores tables (HSEmotion): the columns become <prefix>_label and <prefix>_<class name>. '
          "'emotion' in HSEmotion."},
 {'name': 'Classsmoothing',
  'style': 'Float',
  'label': 'Class Score Smoothing',
  'default': 0.5,
  'min': 0.0,
  'max': 0.99,
  'clampMin': True,
  'clampMax': True,
  'help': 'Class Scores tables: per-track smoothing of the classifier scores (0 = raw, higher = steadier).'}]

TRACKING = {'pose': [{'name': 'Confthreshold',
           'style': 'Float',
           'label': 'Confidence Threshold',
           'default': 0.005,
           'min': 0.0,
           'max': 0.05,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and to '
                   'stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage high/low '
                   'recovery band -- ages out purely via Track Loss Frames/score decay instead).'},
          {'name': 'Outputsmoothing',
           'style': 'Float',
           'label': 'Output Smoothing',
           'default': 0.23,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                   'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
          {'name': 'Tracklossframes',
           'style': 'Float',
           'label': 'Track Loss Frames',
           'default': 15,
           'min': 0.0,
           'max': 90.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                   'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                   'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
          {'name': 'Scalematch',
           'style': 'Toggle',
           'label': 'Scale Match To Skeleton',
           'default': True,
           'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative Match "
                   'Distance), so small distant and large close-up skeletons need no re-tuning. Off = the '
                   'absolute Max Match Distance.'},
          {'name': 'Relmatchdist',
           'style': 'Float',
           'label': 'Relative Match Distance',
           'default': 0.15,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                   "between frames and still be the same track, as a fraction of the skeleton's size (its "
                   'box diagonal). Too low breaks tracks on fast movement; too high lets neighbours swap '
                   'ids.'},
          {'name': 'Maxmatchdist',
           'style': 'Float',
           'label': 'Max Match Distance',
           'default': 0.5,
           'min': 0.0,
           'max': 2.0,
           'clampMin': True,
           'clampMax': False,
           'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to accept "
                   'a match between an existing track and a new detection. Too low breaks continuous '
                   "matching on fast movement (constant track-ID respawning); too high risks a new person's "
                   'detection getting absorbed into an unrelated existing track. Used only with Scale Match '
                   'To Skeleton off.'},
          {'name': 'Trackconfirmframes',
           'style': 'Float',
           'label': 'Track Confirm Frames',
           'default': 3,
           'min': 1.0,
           'max': 30.0,
           'clampMin': True,
           'clampMax': False,
           'help': "Total matched frames (not necessarily consecutive) a brand-new track needs before it's "
                   'trusted and shown at all. Filters out single-frame noise "detections" that almost never '
                   'get a second real match. Higher = slower to show a genuinely new person, but cleaner.'},
          {'name': 'Minboxwidth',
           'style': 'Float',
           'label': 'Min Box Width',
           'default': 0.02,
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                   'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: a '
                   'standing person is naturally much narrower than tall, so one shared threshold high '
                   'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
          {'name': 'Minboxheight',
           'style': 'Float',
           'label': 'Min Box Height',
           'default': 0.02,
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                   'Width for why this is a separate threshold rather than one shared value.'},
          {'name': 'Jointthreshold',
           'style': 'Float',
           'label': 'Joint Confidence Threshold',
           'default': 0.3,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'help': 'Body joints scoring under this are output as unseen (confidence 0 in the tracks table, '
                   'left out of the joints and bones tables), and so are the joints further down the limb '
                   '(no knee without a hip). For people framed from the chest up: the models still report '
                   'their legs. 0 = off.'},
          {'name': 'Hideoffimage',
           'style': 'Toggle',
           'label': 'Hide Off-Image Joints',
           'default': True,
           'help': "Body joints outside the image, or squashed onto its edge (YOLO and MoveNet can't place a "
                   'point outside it), are output as unseen, and so are the joints further down the limb.'}],
 'movenet': [{'name': 'Confthreshold',
              'style': 'Float',
              'label': 'Confidence Threshold',
              'default': 0.1,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and '
                      'to stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage '
                      'high/low recovery band -- ages out purely via Track Loss Frames/score decay '
                      'instead).'},
             {'name': 'Outputsmoothing',
              'style': 'Float',
              'label': 'Output Smoothing',
              'default': 0.4,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                      'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
             {'name': 'Tracklossframes',
              'style': 'Float',
              'label': 'Track Loss Frames',
              'default': 15,
              'min': 0.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                      'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                      'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
             {'name': 'Scalematch',
              'style': 'Toggle',
              'label': 'Scale Match To Skeleton',
              'default': True,
              'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                      'Match Distance), so small distant and large close-up skeletons need no re-tuning. Off '
                      '= the absolute Max Match Distance.'},
             {'name': 'Relmatchdist',
              'style': 'Float',
              'label': 'Relative Match Distance',
              'default': 0.15,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                      "between frames and still be the same track, as a fraction of the skeleton's size (its "
                      'box diagonal). Too low breaks tracks on fast movement; too high lets neighbours swap '
                      'ids.'},
             {'name': 'Maxmatchdist',
              'style': 'Float',
              'label': 'Max Match Distance',
              'default': 0.5,
              'min': 0.0,
              'max': 2.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                      'accept a match between an existing track and a new detection. Too low breaks '
                      'continuous matching on fast movement (constant track-ID respawning); too high risks a '
                      "new person's detection getting absorbed into an unrelated existing track. Used only "
                      'with Scale Match To Skeleton off.'},
             {'name': 'Trackconfirmframes',
              'style': 'Float',
              'label': 'Track Confirm Frames',
              'default': 3,
              'min': 1.0,
              'max': 30.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                      'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                      'almost never get a second real match. Higher = slower to show a genuinely new person, '
                      'but cleaner.'},
             {'name': 'Minboxwidth',
              'style': 'Float',
              'label': 'Min Box Width',
              'default': 0.05,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                      'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: '
                      'a standing person is naturally much narrower than tall, so one shared threshold high '
                      'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
             {'name': 'Minboxheight',
              'style': 'Float',
              'label': 'Min Box Height',
              'default': 0.05,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                      'Width for why this is a separate threshold rather than one shared value.'},
             {'name': 'Jointthreshold',
              'style': 'Float',
              'label': 'Joint Confidence Threshold',
              'default': 0.2,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'Body joints scoring under this are output as unseen (confidence 0 in the tracks '
                      'table, left out of the joints and bones tables), and so are the joints further down '
                      'the limb (no knee without a hip). For people framed from the chest up: the models '
                      'still report their legs. 0 = off.'},
             {'name': 'Hideoffimage',
              'style': 'Toggle',
              'label': 'Hide Off-Image Joints',
              'default': True,
              'help': "Body joints outside the image, or squashed onto its edge (YOLO and MoveNet can't "
                      'place a point outside it), are output as unseen, and so are the joints further down '
                      'the limb.'}],
 'face': [{'name': 'Confthreshold',
           'style': 'Float',
           'label': 'Confidence Threshold',
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'default': 0.7,
           'help': "Minimum detection confidence to start/confirm a new track (ByteTracker's high-confidence "
                   'threshold). Detections below this can still recover an EXISTING track down to Low '
                   "Confidence Threshold, but never start a brand-new one. YuNet's own scores for real faces "
                   'run fairly high (0.7-0.9+ typical).'},
          {'name': 'Lowconfthreshold',
           'style': 'Float',
           'label': 'Low Confidence Threshold (Recovery)',
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'default': 0.3,
           'help': "ByteTracker's second-stage recovery threshold. Detections scoring between this and "
                   "Confidence Threshold can't start a new track, but CAN re-match an already-existing track "
                   'through a bad frame (occlusion, motion blur) that a single-threshold detector would just '
                   'drop. Detections below this are discarded as background.'},
          {'name': 'Minboxwidth',
           'style': 'Float',
           'label': 'Min Box Width',
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'default': 0.02,
           'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                   'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: a '
                   'face is rarely square, so one shared threshold high enough to reject noise ends up '
                   'rejecting real faces that are just proportionally narrow (e.g. a turned head).'},
          {'name': 'Minboxheight',
           'style': 'Float',
           'label': 'Min Box Height',
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'default': 0.02,
           'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                   'Width for why this is a separate threshold rather than one shared value.'},
          {'name': 'Tracklossframes',
           'style': 'Float',
           'label': 'Track Loss Frames',
           'min': 0.0,
           'max': 90.0,
           'clampMin': True,
           'clampMax': False,
           'default': 45,
           'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                   'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                   'but also more frames of a possibly-wrong predicted position/scale before giving up. Kept '
                   "longer than the other scripts' default here since faces are especially prone to brief "
                   'full occlusion (head turn, hand pass) where NO detection fires at all, unlike a body/box '
                   'detector that often still gets a degraded read through partial occlusion.'},
          {'name': 'Trackiouthreshold',
           'style': 'Float',
           'label': 'Track IoU Threshold',
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'default': 0.2,
           'help': 'Minimum box overlap (IoU) required to accept a match between an existing track and a new '
                   "detection. Too low risks a new face's detection getting absorbed into an unrelated "
                   'existing track; too high breaks continuous matching and causes constant track-ID '
                   'respawning.'},
          {'name': 'Trackconfirmframes',
           'style': 'Float',
           'label': 'Track Confirm Frames',
           'min': 1.0,
           'max': 30.0,
           'clampMin': True,
           'clampMax': False,
           'default': 3,
           'help': "Total matched frames (not necessarily consecutive) a brand-new track needs before it's "
                   'trusted and shown at all. Filters out single-frame noise "detections" that almost never '
                   'get a second real match. Higher = slower to show a genuinely new face, but cleaner.'},
          {'name': 'Outputsmoothing',
           'style': 'Float',
           'label': 'Output Smoothing',
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'default': 0.3,
           'help': "Extra lerp on box position/size on top of the tracker's own Kalman estimate -- pure "
                   'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place. Keypoints/head '
                   "direction have their OWN separate smoothing (Pose Smoothing below) -- they're held at "
                   'their last real value across occlusion regardless, see the pipeline docstring. Kept '
                   "lower than the other scripts' default since heavy smoothing reads as unnatural on a "
                   "face's own head-turn motion."},
          {'name': 'Posemethod',
           'style': 'Menu',
           'label': 'Head Pose Method',
           'menuNames': ['geometric', 'solvepnp'],
           'menuLabels': ['Geometric (Keypoint Ratios)', 'SolvePnP (3D Model)'],
           'default': 'geometric',
           'help': 'Which method estimates yaw/pitch/roll. Geometric: bounded distance ratios between the '
                   'nose and the other 4 keypoints, no solver -- far less sensitive to keypoint noise, not a '
                   'calibrated real angle (see Poseyawscale/Posepitchscale). SolvePnP: matches keypoints '
                   'against a rough 3D face model via cv2.solvePnP -- a genuine 3D pose estimate, but '
                   'sensitive to noise on only 5 sparse near-planar points (see Posefocalscale). Roll is '
                   "identical either way -- it was never the problem, just the eye-line's 2D tilt angle."},
          {'name': 'Poseyawscale',
           'style': 'Float',
           'label': 'Head Pose Yaw Scale (Geometric)',
           'min': 1.0,
           'max': 1000.0,
           'clampMin': True,
           'clampMax': False,
           'default': 340.0,
           'help': 'Degrees-per-ratio scale for yaw specifically (only used when Head Pose Method = '
                   "Geometric) -- kept separate from Head Pose Pitch Scale since yaw's underlying ratio "
                   "covers a much smaller natural range than pitch's for the same real rotation. Tune live: "
                   'if head turns look too subtle, increase; too exaggerated, decrease.'},
          {'name': 'Posepitchscale',
           'style': 'Float',
           'label': 'Head Pose Pitch Scale (Geometric)',
           'min': 1.0,
           'max': 1000.0,
           'clampMin': True,
           'clampMax': False,
           'default': 100.0,
           'help': 'Degrees-per-ratio scale for pitch specifically (only used when Head Pose Method = '
                   'Geometric) -- see Head Pose Yaw Scale for why this is separate. Tune live: if head nods '
                   'look too subtle, increase; too exaggerated, decrease.'},
          {'name': 'Posefocalscale',
           'style': 'Float',
           'label': 'Head Pose Focal Scale (SolvePnP)',
           'min': 0.1,
           'max': 5.0,
           'clampMin': True,
           'clampMax': False,
           'default': 1.0,
           'help': 'Multiplier on the assumed camera focal length (only used when Head Pose Method = '
                   'SolvePnP) -- no real camera calibration available, so this is a stand-in you can '
                   'calibrate live: if head-turn rotation looks too subtle for the real motion, increase; if '
                   'it looks exaggerated, decrease. Does not affect box/keypoint detection at all, only the '
                   'derived rotation angles.'},
          {'name': 'Posesmoothing',
           'style': 'Float',
           'label': 'Head Pose Smoothing',
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'default': 0.5,
           'help': 'Extra lerp on yaw/pitch/roll specifically (separate from Output Smoothing, which only '
                   'smooths box position/size), applied regardless of Head Pose Method -- keypoint detection '
                   'noise on just 5 sparse points can make raw per-frame angles look jittery even when not '
                   'moving. 0 = raw/no smoothing, 1 = frozen in place.'}],
 'detect': [{'name': 'Confthreshold',
             'style': 'Float',
             'label': 'Confidence Threshold',
             'default': 0.5,
             'min': 0.0,
             'max': 1.0,
             'clampMin': True,
             'clampMax': True,
             'help': "Minimum detection confidence to start/confirm a new track (ByteTracker's "
                     'high-confidence threshold). Detections below this can still recover an EXISTING track '
                     'down to Low Confidence Threshold, but never start a brand-new one.'},
            {'name': 'Lowconfthreshold',
             'style': 'Float',
             'label': 'Low Confidence Threshold (Recovery)',
             'default': 0.1,
             'min': 0.0,
             'max': 1.0,
             'clampMin': True,
             'clampMax': True,
             'help': "ByteTracker's second-stage recovery threshold. Detections scoring between this and "
                     "Confidence Threshold can't start a new track, but CAN re-match an already-existing "
                     'track through a bad frame (occlusion, motion blur) that a single-threshold detector '
                     'would just drop. Detections below this are discarded as background.'},
            {'name': 'Nmsiouthreshold',
             'style': 'Float',
             'label': 'NMS IoU Threshold (Dedup)',
             'default': 0.45,
             'min': 0.0,
             'max': 1.0,
             'clampMin': True,
             'clampMax': True,
             'help': 'IoU threshold for collapsing near-duplicate raw detections of the same object before '
                     'they ever reach the tracker. Lower = more aggressive merging (risks merging two '
                     'close-together objects); higher = more duplicates survive to become phantom tracks.'},
            {'name': 'Minboxwidth',
             'style': 'Float',
             'label': 'Min Box Width',
             'default': 0.02,
             'min': 0.0,
             'max': 0.2,
             'clampMin': True,
             'clampMax': False,
             'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                     'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: '
                     'an object is rarely square, so one shared threshold high enough to reject noise ends '
                     'up rejecting real objects that are just proportionally narrow on one axis.'},
            {'name': 'Minboxheight',
             'style': 'Float',
             'label': 'Min Box Height',
             'default': 0.02,
             'min': 0.0,
             'max': 0.2,
             'clampMin': True,
             'clampMax': False,
             'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                     'Width for why this is a separate threshold rather than one shared value.'},
            {'name': 'Tracklossframes',
             'style': 'Float',
             'label': 'Track Loss Frames',
             'default': 30,
             'min': 0.0,
             'max': 90.0,
             'clampMin': True,
             'clampMax': False,
             'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                     'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                     'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
            {'name': 'Trackiouthreshold',
             'style': 'Float',
             'label': 'Track IoU Threshold',
             'default': 0.3,
             'min': 0.0,
             'max': 1.0,
             'clampMin': True,
             'clampMax': True,
             'help': 'Minimum box overlap (IoU) required to accept a match between an existing track and a '
                     "new detection. Too low risks a new object's detection getting absorbed into an "
                     'unrelated existing track; too high breaks continuous matching and causes constant '
                     'track-ID respawning.'},
            {'name': 'Trackconfirmframes',
             'style': 'Float',
             'label': 'Track Confirm Frames',
             'default': 3,
             'min': 1.0,
             'max': 30.0,
             'clampMin': True,
             'clampMax': False,
             'help': "Total matched frames (not necessarily consecutive) a brand-new track needs before it's "
                     'trusted and shown at all. Filters out single-frame noise "detections" that almost '
                     'never get a second real match. Higher = slower to show a genuinely new object, but '
                     'cleaner.'},
            {'name': 'Outputsmoothing',
             'style': 'Float',
             'label': 'Output Smoothing',
             'default': 0.5,
             'min': 0.0,
             'max': 1.0,
             'clampMin': True,
             'clampMax': True,
             'help': "Extra lerp on box position/size on top of the tracker's own Kalman estimate -- pure "
                     'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'}],
 'segment': [{'name': 'Confthreshold',
              'style': 'Float',
              'label': 'Confidence Threshold',
              'default': 0.35,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "Minimum detection confidence to start/confirm a new track (ByteTracker's "
                      'high-confidence threshold). Detections below this can still recover an EXISTING track '
                      'down to Low Confidence Threshold, but never start a brand-new one.'},
             {'name': 'Lowconfthreshold',
              'style': 'Float',
              'label': 'Low Confidence Threshold (Recovery)',
              'default': 0.1,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "ByteTracker's second-stage recovery threshold. Detections scoring between this and "
                      "Confidence Threshold can't start a new track, but CAN re-match an already-existing "
                      'track through a bad frame (occlusion, motion blur) that a single-threshold detector '
                      'would just drop. Detections below this are discarded as background.'},
             {'name': 'Nmsiouthreshold',
              'style': 'Float',
              'label': 'NMS IoU Threshold (Dedup)',
              'default': 0.5,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'IoU threshold for collapsing near-duplicate raw detections of the same person before '
                      'they ever reach the tracker. Lower = more aggressive merging (risks merging two '
                      'close-together people); higher = more duplicates survive to become phantom tracks.'},
             {'name': 'Minboxwidth',
              'style': 'Float',
              'label': 'Min Box Width',
              'default': 0.02,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                      'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: '
                      'a standing person is naturally much narrower than tall, so one shared threshold high '
                      'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
             {'name': 'Minboxheight',
              'style': 'Float',
              'label': 'Min Box Height',
              'default': 0.02,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                      'Width for why this is a separate threshold rather than one shared value.'},
             {'name': 'Minmaskarea',
              'style': 'Float',
              'label': 'Min Mask Fill Ratio (Of Own Box)',
              'default': 0.15,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'Rejects a decoded mask if it fills less than this fraction of ITS OWN detection box '
                      '(not a fraction of the whole frame -- scale-invariant regardless of how near/far a '
                      'person is). A real silhouette typically fills ~35-70% of its own box (limb gaps keep '
                      'it under 100%); much lower usually means a degenerate/noisy mask.'},
             {'name': 'Tracklossframes',
              'style': 'Float',
              'label': 'Track Loss Frames',
              'default': 30,
              'min': 0.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                      'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                      'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
             {'name': 'Trackiouthreshold',
              'style': 'Float',
              'label': 'Track IoU Threshold',
              'default': 0.3,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'Minimum box overlap (IoU) required to accept a match between an existing track and a '
                      "new detection. Too low risks a new person's detection getting absorbed into an "
                      'unrelated existing track; too high breaks continuous matching and causes constant '
                      'track-ID respawning.'},
             {'name': 'Trackconfirmframes',
              'style': 'Float',
              'label': 'Track Confirm Frames',
              'default': 3,
              'min': 1.0,
              'max': 30.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                      'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                      'almost never get a second real match. Higher = slower to show a genuinely new person, '
                      'but cleaner.'},
             {'name': 'Outputsmoothing',
              'style': 'Float',
              'label': 'Output Smoothing (Box)',
              'default': 0.5,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "Extra lerp on box position/size on top of the tracker's own Kalman estimate -- pure "
                      'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place. Masks themselves '
                      'are NOT smoothed by this -- see class docstring.'},
             {'name': 'Presencerise',
              'style': 'Float',
              'label': 'Presence Rise Frames',
              'default': 5,
              'min': 1.0,
              'max': 60.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Frames to linearly ramp a person's Presence value from 0 to 1 after it's first "
                      'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                      'smoothly instead of a hard on/off cut.'},
             {'name': 'Presencefall',
              'style': 'Float',
              'label': 'Presence Fall Frames',
              'default': 15,
              'min': 1.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Frames to linearly ramp Presence back down to 0 once a person stops being freshly '
                      'detected (independent of Track Loss Frames, which governs how long the tracker keeps '
                      'hunting for a re-match before dropping the track entirely -- keep this at or below '
                      'Track Loss Frames so Presence reaches 0 before the track itself is pruned).'}],
 'mp_pose': [{'name': 'Confthreshold',
              'style': 'Float',
              'label': 'Confidence Threshold',
              'default': 0.1,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and '
                      'to stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage '
                      'high/low recovery band -- ages out purely via Track Loss Frames/score decay '
                      'instead).'},
             {'name': 'Outputsmoothing',
              'style': 'Float',
              'label': 'Output Smoothing',
              'default': 0.0,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                      'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
             {'name': 'Tracklossframes',
              'style': 'Float',
              'label': 'Track Loss Frames',
              'default': 3,
              'min': 0.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                      'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                      'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
             {'name': 'Scalematch',
              'style': 'Toggle',
              'label': 'Scale Match To Skeleton',
              'default': True,
              'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                      'Match Distance), so small distant and large close-up skeletons need no re-tuning. Off '
                      '= the absolute Max Match Distance.'},
             {'name': 'Relmatchdist',
              'style': 'Float',
              'label': 'Relative Match Distance',
              'default': 0.15,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                      "between frames and still be the same track, as a fraction of the skeleton's size (its "
                      'box diagonal). Too low breaks tracks on fast movement; too high lets neighbours swap '
                      'ids.'},
             {'name': 'Maxmatchdist',
              'style': 'Float',
              'label': 'Max Match Distance',
              'default': 0.8333333333333334,
              'min': 0.0,
              'max': 2.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                      'accept a match between an existing track and a new detection. Too low breaks '
                      'continuous matching on fast movement (constant track-ID respawning); too high risks a '
                      "new person's detection getting absorbed into an unrelated existing track. Used only "
                      'with Scale Match To Skeleton off.'},
             {'name': 'Trackconfirmframes',
              'style': 'Float',
              'label': 'Track Confirm Frames',
              'default': 1,
              'min': 1.0,
              'max': 30.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                      'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                      'almost never get a second real match. Higher = slower to show a genuinely new person, '
                      'but cleaner.'},
             {'name': 'Minboxwidth',
              'style': 'Float',
              'label': 'Min Box Width',
              'default': 0.01,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                      'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: '
                      'a standing person is naturally much narrower than tall, so one shared threshold high '
                      'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
             {'name': 'Minboxheight',
              'style': 'Float',
              'label': 'Min Box Height',
              'default': 0.01,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                      'Width for why this is a separate threshold rather than one shared value.'},
             {'name': 'Coastwhilelost',
              'style': 'Float',
              'label': 'Coast While Lost',
              'default': 0.0,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and box '
                      'along its last frame-to-frame velocity, multiplied by this factor every lost frame (0 '
                      "= freeze in place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman "
                      'prediction, it bridges a fast-moving hand that MediaPipe drops for a frame or two.'},
             {'name': 'Presencerise',
              'style': 'Float',
              'label': 'Presence Rise Frames',
              'default': 3.0,
              'min': 1.0,
              'max': 60.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first "
                      'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                      'smoothly instead of a hard on/off cut.'},
             {'name': 'Presencefall',
              'style': 'Float',
              'label': 'Presence Fall Frames',
              'default': 3.0,
              'min': 1.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                      'detected (independent of Track Loss Frames, which governs how long the tracker keeps '
                      'hunting for a re-match before dropping the track entirely -- keep this at or below '
                      'Track Loss Frames so Presence reaches 0 before the track itself is pruned).'},
             {'name': 'Jointthreshold',
              'style': 'Float',
              'label': 'Joint Confidence Threshold',
              'default': 0.5,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'Body joints scoring under this are output as unseen (confidence 0 in the tracks '
                      'table, left out of the joints and bones tables), and so are the joints further down '
                      'the limb (no knee without a hip). For people framed from the chest up: the models '
                      'still report their legs. 0 = off.'},
             {'name': 'Hideoffimage',
              'style': 'Toggle',
              'label': 'Hide Off-Image Joints',
              'default': True,
              'help': "Body joints outside the image, or squashed onto its edge (YOLO and MoveNet can't "
                      'place a point outside it), are output as unseen, and so are the joints further down '
                      'the limb.'}],
 'mp_hands': [{'name': 'Confthreshold',
               'style': 'Float',
               'label': 'Confidence Threshold',
               'default': 0.1,
               'min': 0.0,
               'max': 1.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and '
                       'to stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage '
                       'high/low recovery band -- ages out purely via Track Loss Frames/score decay '
                       'instead).'},
              {'name': 'Outputsmoothing',
               'style': 'Float',
               'label': 'Output Smoothing',
               'default': 0.0,
               'min': 0.0,
               'max': 1.0,
               'clampMin': True,
               'clampMax': True,
               'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                       'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
              {'name': 'Tracklossframes',
               'style': 'Float',
               'label': 'Track Loss Frames',
               'default': 6,
               'min': 0.0,
               'max': 90.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, '
                       'no real detection) before being dropped entirely. Higher = more grace through '
                       'occlusion, but also more frames of a possibly-wrong predicted position/scale before '
                       'giving up.'},
              {'name': 'Scalematch',
               'style': 'Toggle',
               'label': 'Scale Match To Skeleton',
               'default': True,
               'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                       'Match Distance), so small distant and large close-up skeletons need no re-tuning. '
                       'Off = the absolute Max Match Distance.'},
              {'name': 'Relmatchdist',
               'style': 'Float',
               'label': 'Relative Match Distance',
               'default': 0.35,
               'min': 0.0,
               'max': 1.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                       "between frames and still be the same track, as a fraction of the skeleton's size "
                       '(its box diagonal). Too low breaks tracks on fast movement; too high lets neighbours '
                       'swap ids.'},
              {'name': 'Maxmatchdist',
               'style': 'Float',
               'label': 'Max Match Distance',
               'default': 0.3,
               'min': 0.0,
               'max': 2.0,
               'clampMin': True,
               'clampMax': False,
               'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                       'accept a match between an existing track and a new detection. Too low breaks '
                       'continuous matching on fast movement (constant track-ID respawning); too high risks '
                       "a new person's detection getting absorbed into an unrelated existing track. Used "
                       'only with Scale Match To Skeleton off.'},
              {'name': 'Trackconfirmframes',
               'style': 'Float',
               'label': 'Track Confirm Frames',
               'default': 1,
               'min': 1.0,
               'max': 30.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                       'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                       'almost never get a second real match. Higher = slower to show a genuinely new '
                       'person, but cleaner.'},
              {'name': 'Minboxwidth',
               'style': 'Float',
               'label': 'Min Box Width',
               'default': 0.01,
               'min': 0.0,
               'max': 0.2,
               'clampMin': True,
               'clampMax': False,
               'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                       'degenerate slivers regardless of confidence. Separate from Min Box Height on '
                       'purpose: a standing person is naturally much narrower than tall, so one shared '
                       'threshold high enough to reject noise ends up rejecting real, legitimately '
                       'tall-but-narrow people.'},
              {'name': 'Minboxheight',
               'style': 'Float',
               'label': 'Min Box Height',
               'default': 0.01,
               'min': 0.0,
               'max': 0.2,
               'clampMin': True,
               'clampMax': False,
               'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min '
                       'Box Width for why this is a separate threshold rather than one shared value.'},
              {'name': 'Coastwhilelost',
               'style': 'Float',
               'label': 'Coast While Lost',
               'default': 0.8,
               'min': 0.0,
               'max': 1.0,
               'clampMin': True,
               'clampMax': True,
               'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and box '
                       'along its last frame-to-frame velocity, multiplied by this factor every lost frame '
                       "(0 = freeze in place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman "
                       'prediction, it bridges a fast-moving hand that MediaPipe drops for a frame or two.'},
              {'name': 'Presencerise',
               'style': 'Float',
               'label': 'Presence Rise Frames',
               'default': 3.0,
               'min': 1.0,
               'max': 60.0,
               'clampMin': True,
               'clampMax': False,
               'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first "
                       'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                       'smoothly instead of a hard on/off cut.'},
              {'name': 'Presencefall',
               'style': 'Float',
               'label': 'Presence Fall Frames',
               'default': 6,
               'min': 1.0,
               'max': 90.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                       'detected (independent of Track Loss Frames, which governs how long the tracker keeps '
                       'hunting for a re-match before dropping the track entirely -- keep this at or below '
                       'Track Loss Frames so Presence reaches 0 before the track itself is pruned).'},
              {'name': 'Gestureholdframes',
               'style': 'Float',
               'label': 'Gesture Hold Frames',
               'default': 3.0,
               'min': 1.0,
               'max': 30.0,
               'clampMin': True,
               'clampMax': False,
               'help': 'A new gesture must be the top result for this many consecutive frames before the '
                       "track's gesture column switches to it. Higher = steadier, but triggers later. 1 = "
                       'raw MediaPipe output.'},
              {'name': 'Gestureminscore',
               'style': 'Float',
               'label': 'Gesture Min Score',
               'default': 0.5,
               'min': 0.0,
               'max': 1.0,
               'clampMin': True,
               'clampMax': True,
               'help': 'A gesture scoring below this counts as None (no gesture). Raise it to cut false '
                       'triggers.'}],
 'mp_face': [{'name': 'Confthreshold',
              'style': 'Float',
              'label': 'Confidence Threshold',
              'default': 0.1,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and '
                      'to stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage '
                      'high/low recovery band -- ages out purely via Track Loss Frames/score decay '
                      'instead).'},
             {'name': 'Outputsmoothing',
              'style': 'Float',
              'label': 'Output Smoothing',
              'default': 0.0,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                      'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
             {'name': 'Tracklossframes',
              'style': 'Float',
              'label': 'Track Loss Frames',
              'default': 3,
              'min': 0.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                      'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                      'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
             {'name': 'Scalematch',
              'style': 'Toggle',
              'label': 'Scale Match To Skeleton',
              'default': True,
              'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                      'Match Distance), so small distant and large close-up skeletons need no re-tuning. Off '
                      '= the absolute Max Match Distance.'},
             {'name': 'Relmatchdist',
              'style': 'Float',
              'label': 'Relative Match Distance',
              'default': 0.2,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                      "between frames and still be the same track, as a fraction of the skeleton's size (its "
                      'box diagonal). Too low breaks tracks on fast movement; too high lets neighbours swap '
                      'ids.'},
             {'name': 'Maxmatchdist',
              'style': 'Float',
              'label': 'Max Match Distance',
              'default': 0.3,
              'min': 0.0,
              'max': 2.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                      'accept a match between an existing track and a new detection. Too low breaks '
                      'continuous matching on fast movement (constant track-ID respawning); too high risks a '
                      "new person's detection getting absorbed into an unrelated existing track. Used only "
                      'with Scale Match To Skeleton off.'},
             {'name': 'Trackconfirmframes',
              'style': 'Float',
              'label': 'Track Confirm Frames',
              'default': 1,
              'min': 1.0,
              'max': 30.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                      'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                      'almost never get a second real match. Higher = slower to show a genuinely new person, '
                      'but cleaner.'},
             {'name': 'Minboxwidth',
              'style': 'Float',
              'label': 'Min Box Width',
              'default': 0.01,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                      'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: '
                      'a standing person is naturally much narrower than tall, so one shared threshold high '
                      'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
             {'name': 'Minboxheight',
              'style': 'Float',
              'label': 'Min Box Height',
              'default': 0.01,
              'min': 0.0,
              'max': 0.2,
              'clampMin': True,
              'clampMax': False,
              'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                      'Width for why this is a separate threshold rather than one shared value.'},
             {'name': 'Coastwhilelost',
              'style': 'Float',
              'label': 'Coast While Lost',
              'default': 0.0,
              'min': 0.0,
              'max': 1.0,
              'clampMin': True,
              'clampMax': True,
              'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and box '
                      'along its last frame-to-frame velocity, multiplied by this factor every lost frame (0 '
                      "= freeze in place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman "
                      'prediction, it bridges a fast-moving hand that MediaPipe drops for a frame or two.'},
             {'name': 'Presencerise',
              'style': 'Float',
              'label': 'Presence Rise Frames',
              'default': 3.0,
              'min': 1.0,
              'max': 60.0,
              'clampMin': True,
              'clampMax': False,
              'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first "
                      'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                      'smoothly instead of a hard on/off cut.'},
             {'name': 'Presencefall',
              'style': 'Float',
              'label': 'Presence Fall Frames',
              'default': 3.0,
              'min': 1.0,
              'max': 90.0,
              'clampMin': True,
              'clampMax': False,
              'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                      'detected (independent of Track Loss Frames, which governs how long the tracker keeps '
                      'hunting for a re-match before dropping the track entirely -- keep this at or below '
                      'Track Loss Frames so Presence reaches 0 before the track itself is pruned).'}],
 'mp_holistic': [{'name': 'Confthreshold',
                  'style': 'Float',
                  'label': 'Confidence Threshold',
                  'default': 0.1,
                  'min': 0.0,
                  'max': 1.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, '
                          'and to stay displayed once tracked (KeypointTracker has no ByteTrack-style '
                          'two-stage high/low recovery band -- ages out purely via Track Loss Frames/score '
                          'decay instead).'},
                 {'name': 'Outputsmoothing',
                  'style': 'Float',
                  'label': 'Output Smoothing',
                  'default': 0.0,
                  'min': 0.0,
                  'max': 1.0,
                  'clampMin': True,
                  'clampMax': True,
                  'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- "
                          'pure cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
                 {'name': 'Tracklossframes',
                  'style': 'Float',
                  'label': 'Track Loss Frames',
                  'default': 3,
                  'min': 0.0,
                  'max': 90.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted '
                          'position, no real detection) before being dropped entirely. Higher = more grace '
                          'through occlusion, but also more frames of a possibly-wrong predicted '
                          'position/scale before giving up.'},
                 {'name': 'Scalematch',
                  'style': 'Toggle',
                  'label': 'Scale Match To Skeleton',
                  'default': True,
                  'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                          'Match Distance), so small distant and large close-up skeletons need no re-tuning. '
                          'Off = the absolute Max Match Distance.'},
                 {'name': 'Relmatchdist',
                  'style': 'Float',
                  'label': 'Relative Match Distance',
                  'default': 0.15,
                  'min': 0.0,
                  'max': 1.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may '
                          "move between frames and still be the same track, as a fraction of the skeleton's "
                          'size (its box diagonal). Too low breaks tracks on fast movement; too high lets '
                          'neighbours swap ids.'},
                 {'name': 'Maxmatchdist',
                  'style': 'Float',
                  'label': 'Max Match Distance',
                  'default': 0.8333333333333334,
                  'min': 0.0,
                  'max': 2.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                          'accept a match between an existing track and a new detection. Too low breaks '
                          'continuous matching on fast movement (constant track-ID respawning); too high '
                          "risks a new person's detection getting absorbed into an unrelated existing track. "
                          'Used only with Scale Match To Skeleton off.'},
                 {'name': 'Trackconfirmframes',
                  'style': 'Float',
                  'label': 'Track Confirm Frames',
                  'default': 1,
                  'min': 1.0,
                  'max': 30.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                          'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                          'almost never get a second real match. Higher = slower to show a genuinely new '
                          'person, but cleaner.'},
                 {'name': 'Minboxwidth',
                  'style': 'Float',
                  'label': 'Min Box Width',
                  'default': 0.01,
                  'min': 0.0,
                  'max': 0.2,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                          'degenerate slivers regardless of confidence. Separate from Min Box Height on '
                          'purpose: a standing person is naturally much narrower than tall, so one shared '
                          'threshold high enough to reject noise ends up rejecting real, legitimately '
                          'tall-but-narrow people.'},
                 {'name': 'Minboxheight',
                  'style': 'Float',
                  'label': 'Min Box Height',
                  'default': 0.01,
                  'min': 0.0,
                  'max': 0.2,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min '
                          'Box Width for why this is a separate threshold rather than one shared value.'},
                 {'name': 'Coastwhilelost',
                  'style': 'Float',
                  'label': 'Coast While Lost',
                  'default': 0.0,
                  'min': 0.0,
                  'max': 1.0,
                  'clampMin': True,
                  'clampMax': True,
                  'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and '
                          'box along its last frame-to-frame velocity, multiplied by this factor every lost '
                          "frame (0 = freeze in place, as before; 0.8 = glide to a stop). Like ByteTrack's "
                          'Kalman prediction, it bridges a fast-moving hand that MediaPipe drops for a frame '
                          'or two.'},
                 {'name': 'Presencerise',
                  'style': 'Float',
                  'label': 'Presence Rise Frames',
                  'default': 3.0,
                  'min': 1.0,
                  'max': 60.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first "
                          'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                          'smoothly instead of a hard on/off cut.'},
                 {'name': 'Presencefall',
                  'style': 'Float',
                  'label': 'Presence Fall Frames',
                  'default': 3.0,
                  'min': 1.0,
                  'max': 90.0,
                  'clampMin': True,
                  'clampMax': False,
                  'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                          'detected (independent of Track Loss Frames, which governs how long the tracker '
                          'keeps hunting for a re-match before dropping the track entirely -- keep this at '
                          'or below Track Loss Frames so Presence reaches 0 before the track itself is '
                          'pruned).'},
                 {'name': 'Facemesh',
                  'style': 'Toggle',
                  'label': 'Face Mesh',
                  'default': True,
                  'help': "Track and output Holistic's 478 face mesh points. Off: pose and hands only (75 "
                          'points, the face columns and joints are left out), much cheaper; the Mixamo '
                          "Puppet doesn't need the face."},
                 {'name': 'Jointthreshold',
                  'style': 'Float',
                  'label': 'Joint Confidence Threshold',
                  'default': 0.5,
                  'min': 0.0,
                  'max': 1.0,
                  'clampMin': True,
                  'clampMax': True,
                  'help': 'Body joints scoring under this are output as unseen (confidence 0 in the tracks '
                          'table, left out of the joints and bones tables), and so are the joints further '
                          'down the limb (no knee without a hip). For people framed from the chest up: the '
                          'models still report their legs. 0 = off.'},
                 {'name': 'Hideoffimage',
                  'style': 'Toggle',
                  'label': 'Hide Off-Image Joints',
                  'default': True,
                  'help': "Body joints outside the image, or squashed onto its edge (YOLO and MoveNet can't "
                          'place a point outside it), are output as unseen, and so are the joints further '
                          'down the limb.'}],
 'mp_facedet': [{'name': 'Confthreshold',
                 'style': 'Float',
                 'label': 'Confidence Threshold',
                 'default': 0.1,
                 'min': 0.0,
                 'max': 1.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, '
                         'and to stay displayed once tracked (KeypointTracker has no ByteTrack-style '
                         'two-stage high/low recovery band -- ages out purely via Track Loss Frames/score '
                         'decay instead).'},
                {'name': 'Outputsmoothing',
                 'style': 'Float',
                 'label': 'Output Smoothing',
                 'default': 0.0,
                 'min': 0.0,
                 'max': 1.0,
                 'clampMin': True,
                 'clampMax': True,
                 'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- "
                         'pure cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
                {'name': 'Tracklossframes',
                 'style': 'Float',
                 'label': 'Track Loss Frames',
                 'default': 3,
                 'min': 0.0,
                 'max': 90.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, '
                         'no real detection) before being dropped entirely. Higher = more grace through '
                         'occlusion, but also more frames of a possibly-wrong predicted position/scale '
                         'before giving up.'},
                {'name': 'Scalematch',
                 'style': 'Toggle',
                 'label': 'Scale Match To Skeleton',
                 'default': True,
                 'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative "
                         'Match Distance), so small distant and large close-up skeletons need no re-tuning. '
                         'Off = the absolute Max Match Distance.'},
                {'name': 'Relmatchdist',
                 'style': 'Float',
                 'label': 'Relative Match Distance',
                 'default': 0.3,
                 'min': 0.0,
                 'max': 1.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may '
                         "move between frames and still be the same track, as a fraction of the skeleton's "
                         'size (its box diagonal). Too low breaks tracks on fast movement; too high lets '
                         'neighbours swap ids.'},
                {'name': 'Maxmatchdist',
                 'style': 'Float',
                 'label': 'Max Match Distance',
                 'default': 0.3,
                 'min': 0.0,
                 'max': 2.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to "
                         'accept a match between an existing track and a new detection. Too low breaks '
                         'continuous matching on fast movement (constant track-ID respawning); too high '
                         "risks a new person's detection getting absorbed into an unrelated existing track. "
                         'Used only with Scale Match To Skeleton off.'},
                {'name': 'Trackconfirmframes',
                 'style': 'Float',
                 'label': 'Track Confirm Frames',
                 'default': 1,
                 'min': 1.0,
                 'max': 30.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'Total matched frames (not necessarily consecutive) a brand-new track needs before '
                         'it\'s trusted and shown at all. Filters out single-frame noise "detections" that '
                         'almost never get a second real match. Higher = slower to show a genuinely new '
                         'person, but cleaner.'},
                {'name': 'Minboxwidth',
                 'style': 'Float',
                 'label': 'Min Box Width',
                 'default': 0.01,
                 'min': 0.0,
                 'max': 0.2,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                         'degenerate slivers regardless of confidence. Separate from Min Box Height on '
                         'purpose: a standing person is naturally much narrower than tall, so one shared '
                         'threshold high enough to reject noise ends up rejecting real, legitimately '
                         'tall-but-narrow people.'},
                {'name': 'Minboxheight',
                 'style': 'Float',
                 'label': 'Min Box Height',
                 'default': 0.01,
                 'min': 0.0,
                 'max': 0.2,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min '
                         'Box Width for why this is a separate threshold rather than one shared value.'},
                {'name': 'Coastwhilelost',
                 'style': 'Float',
                 'label': 'Coast While Lost',
                 'default': 0.0,
                 'min': 0.0,
                 'max': 1.0,
                 'clampMin': True,
                 'clampMax': True,
                 'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and box '
                         'along its last frame-to-frame velocity, multiplied by this factor every lost frame '
                         "(0 = freeze in place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman "
                         'prediction, it bridges a fast-moving hand that MediaPipe drops for a frame or '
                         'two.'},
                {'name': 'Presencerise',
                 'style': 'Float',
                 'label': 'Presence Rise Frames',
                 'default': 3.0,
                 'min': 1.0,
                 'max': 60.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first "
                         'freshly detected -- drives Presence in table_output, for scaling an effect in/out '
                         'smoothly instead of a hard on/off cut.'},
                {'name': 'Presencefall',
                 'style': 'Float',
                 'label': 'Presence Fall Frames',
                 'default': 3.0,
                 'min': 1.0,
                 'max': 90.0,
                 'clampMin': True,
                 'clampMax': False,
                 'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                         'detected (independent of Track Loss Frames, which governs how long the tracker '
                         'keeps hunting for a re-match before dropping the track entirely -- keep this at or '
                         'below Track Loss Frames so Presence reaches 0 before the track itself is '
                         'pruned).'}],
 'rtmw': [{'name': 'Confthreshold',
           'style': 'Float',
           'label': 'Confidence Threshold',
           'default': 0.1,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection confidence for a person to be admitted to the tracker at all, and to '
                   'stay displayed once tracked (KeypointTracker has no ByteTrack-style two-stage high/low '
                   'recovery band -- ages out purely via Track Loss Frames/score decay instead).'},
          {'name': 'Outputsmoothing',
           'style': 'Float',
           'label': 'Output Smoothing',
           'default': 0.0,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'help': "Extra lerp on keypoint position on top of the tracker's own Kalman estimate -- pure "
                   'cosmetic jitter damping. 0 = raw/no smoothing, 1 = frozen in place.'},
          {'name': 'Tracklossframes',
           'style': 'Float',
           'label': 'Track Loss Frames',
           'default': 3,
           'min': 0.0,
           'max': 90.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'How many consecutive unmatched frames a track survives (Kalman-predicted position, no '
                   'real detection) before being dropped entirely. Higher = more grace through occlusion, '
                   'but also more frames of a possibly-wrong predicted position/scale before giving up.'},
          {'name': 'Scalematch',
           'style': 'Toggle',
           'label': 'Scale Match To Skeleton',
           'default': True,
           'help': "Match tracks by how far keypoints moved relative to the skeleton's size (Relative Match "
                   'Distance), so small distant and large close-up skeletons need no re-tuning. Off = the '
                   'absolute Max Match Distance.'},
          {'name': 'Relmatchdist',
           'style': 'Float',
           'label': 'Relative Match Distance',
           'default': 0.15,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'With Scale Match To Skeleton on: how far, on average, the stable keypoints may move '
                   "between frames and still be the same track, as a fraction of the skeleton's size (its "
                   'box diagonal). Too low breaks tracks on fast movement; too high lets neighbours swap '
                   'ids.'},
          {'name': 'Maxmatchdist',
           'style': 'Float',
           'label': 'Max Match Distance',
           'default': 0.8333333333333334,
           'min': 0.0,
           'max': 2.0,
           'clampMin': True,
           'clampMax': False,
           'help': "Maximum summed keypoint distance (KeypointTracker, not ByteTracker's box IoU) to accept "
                   'a match between an existing track and a new detection. Too low breaks continuous '
                   "matching on fast movement (constant track-ID respawning); too high risks a new person's "
                   'detection getting absorbed into an unrelated existing track. Used only with Scale Match '
                   'To Skeleton off.'},
          {'name': 'Trackconfirmframes',
           'style': 'Float',
           'label': 'Track Confirm Frames',
           'default': 1,
           'min': 1.0,
           'max': 30.0,
           'clampMin': True,
           'clampMax': False,
           'help': "Total matched frames (not necessarily consecutive) a brand-new track needs before it's "
                   'trusted and shown at all. Filters out single-frame noise "detections" that almost never '
                   'get a second real match. Higher = slower to show a genuinely new person, but cleaner.'},
          {'name': 'Minboxwidth',
           'style': 'Float',
           'label': 'Min Box Width',
           'default': 0.01,
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection box width (fraction of frame width) to keep at all -- rejects '
                   'degenerate slivers regardless of confidence. Separate from Min Box Height on purpose: a '
                   'standing person is naturally much narrower than tall, so one shared threshold high '
                   'enough to reject noise ends up rejecting real, legitimately tall-but-narrow people.'},
          {'name': 'Minboxheight',
           'style': 'Float',
           'label': 'Min Box Height',
           'default': 0.01,
           'min': 0.0,
           'max': 0.2,
           'clampMin': True,
           'clampMax': False,
           'help': 'Minimum detection box height (fraction of frame height) to keep at all -- see Min Box '
                   'Width for why this is a separate threshold rather than one shared value.'},
          {'name': 'Coastwhilelost',
           'style': 'Float',
           'label': 'Coast While Lost',
           'default': 0.0,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'help': 'While a track is lost (up to Track Loss Frames), keep moving its keypoints and box along '
                   'its last frame-to-frame velocity, multiplied by this factor every lost frame (0 = freeze '
                   "in place, as before; 0.8 = glide to a stop). Like ByteTrack's Kalman prediction, it "
                   'bridges a fast-moving hand that MediaPipe drops for a frame or two.'},
          {'name': 'Presencerise',
           'style': 'Float',
           'label': 'Presence Rise Frames',
           'default': 3.0,
           'min': 1.0,
           'max': 60.0,
           'clampMin': True,
           'clampMax': False,
           'help': "Frames to linearly ramp a object's Presence value from 0 to 1 after it's first freshly "
                   'detected -- drives Presence in table_output, for scaling an effect in/out smoothly '
                   'instead of a hard on/off cut.'},
          {'name': 'Presencefall',
           'style': 'Float',
           'label': 'Presence Fall Frames',
           'default': 3.0,
           'min': 1.0,
           'max': 90.0,
           'clampMin': True,
           'clampMax': False,
           'help': 'Frames to linearly ramp Presence back down to 0 once a object stops being freshly '
                   'detected (independent of Track Loss Frames, which governs how long the tracker keeps '
                   'hunting for a re-match before dropping the track entirely -- keep this at or below Track '
                   'Loss Frames so Presence reaches 0 before the track itself is pruned).'},
          {'name': 'Jointthreshold',
           'style': 'Float',
           'label': 'Joint Confidence Threshold',
           'default': 0.0,
           'min': 0.0,
           'max': 1.0,
           'clampMin': True,
           'clampMax': True,
           'help': 'Body joints scoring under this are output as unseen (confidence 0 in the tracks table, '
                   'left out of the joints and bones tables), and so are the joints further down the limb '
                   '(no knee without a hip). For people framed from the chest up: the models still report '
                   'their legs. 0 = off.'},
          {'name': 'Hideoffimage',
           'style': 'Toggle',
           'label': 'Hide Off-Image Joints',
           'default': True,
           'help': "Body joints outside the image, or squashed onto its edge (YOLO and MoveNet can't place a "
                   'point outside it), are output as unseen, and so are the joints further down the limb.'}]}


def defaults(mode):
	"""{name: default} of the mode's Tracking page."""
	return {r['name']: r['default'] for r in TRACKING[mode]}
