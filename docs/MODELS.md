# Models

Every model is a `.tox` in `tox/`. Drag one into a network, wire a video TOP into its input, and
it runs (**Active** is on by default). All of them share the same outputs and parameter pages, so
switching models doesn't change how you wire them up.

## Shared outputs

| Output | Content |
|---|---|
| TOP 1 | The debug overlay: boxes and skeletons on a transparent background, sized like the input. Image models (segmentation, depth) output their result image here instead. |
| TOP 2 | The input image, unchanged. Composite TOP 1 over it to see the overlay in place. |
| DAT 1 `tracks` | One row per tracked person, face, hand or object (see below). |
| DAT 2 / 3 `joints` / `bones` | Keypoints and skeleton connections, in the format the debug visualizers use. Landmark models only. |
| CHOP | Per-track numbers as `t<id>_<column>` (e.g. `t1_cx`, `t1_gesture_onset`) plus the model's status and timing (`loaded`, `infer_ms`, `num_detections`, ...). |

The `tracks` table always starts with `track_id, label, score, cx, cy, w, h, x_left, x_right,
y_top, y_bottom, vx, vy, lost_frames, total_frames`, followed by model-specific columns and one
`<joint>_x, <joint>_y, <joint>_conf` triple per keypoint. Positions are 0–1 across the image with
the origin at the **bottom left** (TouchDesigner's UV convention). A track id stays with the same
person or object while it's visible and through short dropouts.

## Shared parameters

- **Model** page: **Active**, the handful of settings that matter for that model, and **Reload**.
- **Tracking** page: how detections become stable tracks. **Confidence Threshold** (ignore
  weaker detections), **Track Loss Frames** (how long a lost track is held), **Output Smoothing**
  (0 = raw), plus model-specific settings.
- **Debug** page: **Debug Overlay** on/off (or **Show Result** on image models; off outputs
  transparent black and skips the drawing work), and on landmark models **Overlay Layers** (Both,
  Bounding Boxes or Joints), **Joint Size** and **Joint Outline** (a white ring around each joint).
- **About** page: the package version.

## Tracking

The plugins report raw detections each frame; Python inside each tox (`python/object_tracker.py`
and the `*_tracker_pipeline.py` modules) turns them into tracks with stable ids. It's
ByteTrack-style, with changes that have worked well for live installations:

- **Boxes** (YOLO Detection/Segmentation, YuNet): ByteTrack. A Kalman filter predicts each
  track's box, and detections are matched by box overlap in two passes: confident detections
  first, then weaker ones that can only extend existing tracks. Segmentation adds a presence
  ramp; YuNet adds a head-pose estimate.
- **Skeletons** (every pose, hand and face-landmark model): matched by how far the stable
  keypoints (torso, wrist and palm, eyes and nose; not fast-moving extremities) moved, not by box
  overlap, because limbs make boxes jump. By default the distance is measured relative to the
  skeleton's size, so small distant and large close-up people use the same setting. Other
  additions: duplicate suppression, **Coast While Lost** (a lost track keeps drifting along its last
  motion, so a fast hand that MediaPipe drops for a frame keeps its id), presence fade in/out,
  and debounced gestures with a one-frame onset.

Tuning, if ids swap or tracks break:
- **Relative Match Distance** (default 0.15 for bodies, 0.35 for hands): how far the stable
  keypoints may move between frames, as a fraction of the skeleton's size. Lower it if
  neighbours swap ids; raise it if fast movement keeps creating new ids. Turning **Scale Match
  To Skeleton** off (inside the tox's `tracker`) restores the original absolute **Max Match
  Distance**.
- **Track Loss Frames**: longer bridges occlusions but keeps a departed person's ghost longer.
  After switching the video source, old tracks linger for this many frames.
- **Output Smoothing**: steadier keypoints at the cost of lag.

## Hardware

Both plugins need Windows and an NVIDIA GPU. **ONNX** models run on the GPU through
TouchDesigner's ONNX Runtime with TensorRT. **MediaPipe** models run Google's MediaPipe on the CPU,
on a worker thread, so their results arrive one frame late.

The first time an ONNX model runs, it starts on CUDA right away and builds a TensorRT engine in
the background (a few minutes, lower frame rate meanwhile). The engine is cached in `trt_cache/`
next to the model, so later loads take seconds.

## The models

### MediaPipe Hands (`MediaPipeHands.tox`)
Up to N hands (default 2) with 21 landmarks each. **Task** switches between Hands and Gesture
(default): Gesture adds `gesture`, `gesture_score` and a one-frame `gesture_onset` column for
Google's 7 gestures (`Thumb_Up`, `Victory`, `Open_Palm`, ...). Gestures are debounced by
**Gesture Hold Frames** / **Gesture Min Score**. MediaPipe, Apache-2.0.

### MediaPipe Pose (`MediaPipePose.tox`)
Up to N people (default 2), 33 body landmarks each (`pose_landmarker_full`). MediaPipe,
Apache-2.0.

### MediaPipe Face Landmarks (`MediaPipeFace.tox`)
Up to N faces (default 2) with the 478-point face mesh. **Blendshapes** adds 52 expression scores
(`jawOpen`, `eyeBlinkLeft`, ...) as columns; **Matrix** adds the face's 4×4 transform. MediaPipe,
Apache-2.0.

### MediaPipe Face Detector (`MediaPipeFaceDetector.tox`)
Fast face boxes with 6 keypoints (eyes, nose, mouth, ears), up to 4 faces. Use it when you need
where faces are, not their shape. MediaPipe (BlazeFace short range), Apache-2.0.

### MediaPipe Holistic (`MediaPipeHolistic.tox`)
One person: body, both hands and the face mesh in one model (75 tracked joints plus face).
**Blendshapes** adds the face expression scores. Heavier than Pose. MediaPipe, Apache-2.0.

### MediaPipe Segmentation (`MediaPipeSegment.tox`)
A person/background matte. TOP 1 is the mask at the input's resolution. **Segmenter Model**,
**Segment Output** and **Segment Class** let you load another MediaPipe segmenter (e.g. hair or multi-class).
MediaPipe (selfie segmenter), Apache-2.0.

### YOLO26 Object Detection (`Yolo26Detect.tox`)
Boxes for the 80 COCO classes (`class_name` column: person, car, dog, ...). **Class Filter** keeps
one class (-1 = all). GPU. Ultralytics YOLO26s, **AGPL-3.0**: not bundled, run
`fetch_models.ps1` once (see the README).

### YOLO26 Segmentation (`Yolo26Seg.tox`)
Like Object Detection, plus a per-object mask. TOP 1 is the native segmentation matte (at the
model's resolution), TOP 2 the input, TOP 3 the box overlay. `mask_area_ratio` and `presence`
columns per track. GPU. Ultralytics YOLO26s-seg, **AGPL-3.0**: run `fetch_models.ps1`.

### YOLO26 Pose (`Yolo26Pose.tox`)
Many people at once, 17 COCO body keypoints each. GPU, and much cheaper per person than MediaPipe
Pose in crowds. Ultralytics YOLO26n-pose, **AGPL-3.0**: run `fetch_models.ps1`.

### MoveNet MultiPose (`MoveNet.tox`)
Up to 6 people, 17 keypoints each. GPU. **Dynamic Size** sets the inference resolution (multiple
of 32; larger is more accurate and slower). Google MoveNet MultiPose Lightning, Apache-2.0.

### YuNet Face Detection (`YuNet.tox`)
Face boxes with 5 keypoints, plus a head pose estimate (`yaw`, `pitch`, `roll`; **Head Pose
Method** picks the estimator). GPU, very fast. OpenCV Zoo YuNet, MIT.

### Depth Anything V2 (`DepthAnything.tox`)
Relative depth for the whole frame (near = bright). TOP 1 is the depth image at the input's
resolution. **Dynamic Size** sets the inference resolution (multiple of 14); **Normalize** and
**Range Smoothing** keep the brightness range steady between frames. GPU. Depth Anything V2 Small, Apache-2.0.

### HSEmotion Face Emotion (`HSEmotion.tox`)
Finds faces with a built-in YuNet and adds emotion scores to each face track: `emotion_label`
plus one `emotion_<name>` score for anger, disgust, fear, happiness, neutral, sadness and
surprise. **Classify Interval** runs the classifier every Nth frame to save GPU time;
**Class Score Smoothing** steadies the scores. GPU. HSEmotion enet_b2_7, Apache-2.0.

## ModelDemo (`ModelDemo.tox`)

All 13 models in one COMP for trying them side by side. Wire a video TOP in and switch models on
its **Models** page. A model that's off doesn't cook at all. TOP 1 is every enabled model's
overlay composited together (image results behind the overlays); TOP 2 is the input.
