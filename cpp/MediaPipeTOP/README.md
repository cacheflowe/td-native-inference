# MediaPipeTOP

A CUDA-mode CPlusPlus TOP that runs Google's official **MediaPipe Tasks**: Face Landmarker, Hand
Landmarker, Gesture Recognizer, Pose Landmarker, Holistic Landmarker, Face Detector and Image
Segmenter. It uses the prebuilt MediaPipe Tasks C API
(`libmediapipe.dll`, the same engine the `mediapipe` pip package calls) with Google's own `.task`
model bundles. That means MediaPipe's full pipeline runs, not a re-implementation: detection, the
ROI loop that skips the detector while tracking, smoothing, 52 face blendshapes, the face transform
matrix, metric world landmarks, handedness, gestures and the pose segmentation mask.

```
input TOP (cudaArray) -> CUDA resize to Detect Width, RGB8 -> async pinned host copy
  -> (next cook) worker thread: MpXxxDetectForVideo (CPU / XNNPACK) -> Info DAT / Info CHOP
output TOP: the frame MediaPipe sees, or the pose segmentation mask
```

**CPU only on Windows.** Google's Windows `libmediapipe.dll` is built without GPU support: asking
for the GPU delegate fails with *"GPU processing is disabled in build flags"*, and the docs say GPU
is Ubuntu-only. MediaPipe's models are small enough that this works well; see Performance. For
GPU-heavy work (many people, big models) use [OnnxTOP](../OnnxTOP/README.md) with TensorRT.

## Setup

1. `powershell -File cpp\MediaPipeTOP\fetch_mediapipe.ps1` downloads the `mediapipe==1.2.2`
   win_amd64 wheel and extracts only `libmediapipe.dll` into `plugins/`. It also downloads the
   `.task` bundles into `models/mediapipe/`. Add `-Heavy` for `pose_landmarker_heavy.task`.
2. `cpp\MediaPipeTOP\build.cmd` builds `MediaPipeTOP.dll` (same toolchain as OnnxTOP: MSVC v142 +
   CUDA 12.3) and copies it into `plugins/`.
3. In TD, add a CPlusPlus TOP with Plugin Path `plugins/MediaPipeTOP.dll` and wire a video in.

`libmediapipe.dll` has to sit next to `MediaPipeTOP.dll`; it has no dependencies beyond Windows
system DLLs. The C headers in `third_party/mediapipe` come from the `v1.2.2` tag of
google-ai-edge/mediapipe (Apache-2.0). If you upgrade the DLL, re-vendor the headers from the
matching tag: the struct layouts must agree.

## Parameters (page MediaPipe)

| Parameter | Notes |
|---|---|
| Task | face / hands / gesture / pose / holistic / facedetector / segment (saved by name) |
| Face / Hand / Gesture / Pose / Holistic / Face Detector / Segmenter Model | Model file for each task (`.task`, or `.tflite` for the face detector and segmenters). Defaults point at `models/mediapipe/`. A relative path is looked up under the project folder, then under the folder holding `MediaPipeTOP.dll` (an installed `Plugins/<folder>/models/...`). Only the current task's parameter is enabled. |
| Active | Off: stops sending frames and clears the results. |
| Reload Model | Recreates the task. |
| Max Faces / Hands / Poses | Set this to what you really expect. While fewer are tracked, MediaPipe runs its detector every frame looking for the rest: 2 hands with 1 visible costs about 3×. |
| Detect Width | The input is resized on the GPU to this width, keeping aspect, before the copy to the CPU. 0 = input size. MediaPipe's models run at 128–256 px internally, so a larger width mostly costs copy time. |
| Min Detection / Presence / Tracking Confidence | MediaPipe's thresholds, 0.5 by default. |
| Face Blendshapes | Face: adds 52 blendshape columns, named by category (`eyeBlinkLeft`, `jawOpen`, …). |
| Face Transform Matrix | Face: adds `m00`..`m33`, row-major. This 4×4 maps MediaPipe's canonical face model to camera space. |
| Pose Segmentation Mask | Pose / Holistic: the TOP outputs the person mask (Mono 32-bit float) instead of the frame. |
| Segment Output | Segmenter: **Foreground** = the only mask (selfie models) or 1 − background (multi-class models); **One Class** = Segment Class's confidence; **Category Index** = per-pixel class index as a float. The selfie model's category map is 0 = person, 255 = background. |
| Segment Class | Class index for One Class. The Info DAT lists the classes in order. |
| Landmark Z Columns | Adds `kp{i}_z`: MediaPipe's relative depth, in roughly the same units as u. |
| World Landmark Columns | Hands / pose / holistic: adds `w{i}_x/y/z` in meters, MediaPipe's axes (y down, z away from the camera). Origin at the hand center (Hands) or the hips (Pose, Holistic). Holistic gets them for its 75 body and hand landmarks only (its hands share the body's space; the face mesh has none). The Tracker DAT turns them to TD axes (y up, +z toward the camera) as `<joint>_wx/_wy/_wz`. |
| Wait for Result | Waits in the cook (up to Max Wait) for this frame's result, so the landmarks match the image (`result_delay` 0). Worth it for fast models (Face Detector, Hands); a slow one (Holistic) costs TD frame rate. Off by default: the cook never waits. |
| Max Wait (ms) | Longest a cook waits with Wait for Result on; past it the result lands on a later cook. Default 20. |

## Outputs

**Info DAT** has one row per face, hand or pose, with the same base layout as OnnxTOP's, so
DebugBB and the Tracker DAT read it the same way:

```
id, class, conf, u1, v1, u2, v2, label, kp{i}_u, kp{i}_v, kp{i}_c ..., [kp{i}_z ...],
[w{i}_x, w{i}_y, w{i}_z ...], [<52 blendshape names>], [m00 .. m33]
```

- All coordinates are TD UV of the input image (origin bottom-left). `u1 v1 u2 v2` is the bounding
  box of the landmarks.
- **Face**: 478 keypoints. `class` = 0, `conf` = 1 (the API has no per-face score), `kp_c` =
  presence.
- **Hands**: 21 keypoints. `class` 0 = Left, 1 = Right; `label` = `Left` / `Right`; `conf` =
  handedness score. MediaPipe assumes a mirrored (selfie) image when it picks handedness, so
  unmirrored camera input reports the opposite hand.
- **Gesture**: like Hands, but `label` = the top gesture (`None`, `Closed_Fist`, `Open_Palm`,
  `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`) and `conf` = its score.
  Handedness stays in `class`.
- **Pose**: 33 keypoints (BlazePose order). `kp_c` = visibility; `conf` = mean visibility.
- **Holistic**: one row per person (one person at most) with 553 keypoints packed as pose 0–32,
  left hand 33–53, right hand 54–74, face 75–552. A part MediaPipe didn't find stays at 0 with
  `kp_c` = 0. `kp_c` is visibility for the pose part and 1 for found hand and face points. Face
  blendshapes and world landmarks (pose and hands) work as in the single tasks.
- **Face Detector**: BlazeFace short range. `u1..v2` is the model's box; `conf` is the detection
  score; there are 6 keypoints (right eye, left eye, nose tip, mouth center, right ear tragion, left
  ear tragion).
- **Segmenter**: one row per class (`class` index, `label` = the model's class name, `conf` = mean
  confidence, i.e. the fraction of the frame covered) and no keypoints. The mask is the TOP image.

**Info CHOP**: `loaded`, `loading`, `num_detections`, `infer_ms` (the worker's MediaPipe time),
`execute_ms` (TD-side cook), `frame_w`, `frame_h`, `results` (results received so far),
`dropped_frames` (frames replaced before the worker got to them), `infer_ms_avg` and
`execute_ms_avg` (running averages over about ten cooks), and `result_delay`: how many
input frames old the current result is (counted in the input TOP's cooks). To show the frame the
landmarks belong to, cache the input with a Cache TOP (Cache Size above the largest delay) and
set its Output Index to `-result_delay`.

**TOP image**: the RGB frame MediaPipe sees, at Detect Width; or, for the segmenter or with Pose
Segmentation Mask on, the mask (Mono 32-bit float).

## Tracking in Python (OnnxTopTables)

Wire the Info DAT into an OnnxTopTables COMP, as for OnnxTOP, and pick the matching Tracker
Mode. The shared keypoint tracker then gives stable track ids, plus `table_output`,
`table_joints` and `table_bones` for DebugBB and DebugKeypoints:

| MediaPipeTOP Task | Tracker Mode | Joints / bones |
|---|---|---|
| Pose | MediaPipe Pose (33) | BlazePose names, 35 bones |
| Hands, Gesture | MediaPipe Hands / Gesture (21) | `wrist`, `thumb_cmc`, … 21 bones; the gesture or handedness is in the label, plus a `handedness` column |
| Face | MediaPipe Face (478) | `face_0`…; contour + iris bones (132); blendshapes become `table_output` columns |
| Holistic | MediaPipe Holistic (553) | `pose_*`, `left_hand_*`, `right_hand_*`, `face_*`; missing parts are left out |
| Face Detector | MediaPipe Face Detector (6) | the 6 BlazeFace points |

- **Defaults** (the Tracker DAT's MediaPipe modes, `python/tracking_pars.py`): no extra smoothing, since MediaPipe
  already smooths; tracks confirm on the first frame; Confidence Threshold 0.1; **Track Loss
  Frames 3**. That last one is lower than YOLO's 15 because a lost MediaPipe track means MediaPipe
  itself dropped it. With two dancers, Pose (Max 2) and Holistic (one person) switch between people
  from frame to frame, and a 15-frame hold left frozen "ghost" skeletons on screen.
- **Persistence (all MediaPipe modes, Tracking page):** ByteTrack-style carry-through.
  MediaPipe never reports low-confidence detections, so ByteTrack's two-stage recovery band
  doesn't apply. Instead:
  - **Coast While Lost**: a lost track keeps moving along its last velocity, decaying by this
    factor each frame. Matching uses that predicted position too, so a hand MediaPipe drops for a
    frame or two keeps its id.
  - **Presence Rise / Fall Frames**: adds a 0–1 `presence` column, for fading effects in and out.
  - Together with Track Loss Frames, Max Match Distance and Track Confirm Frames.
- **Hands / Gesture defaults:** Track Loss Frames 6, Coast While Lost 0.8, Presence Fall 6.
  Pose, Holistic and Face keep coasting off and loss at 3 because of the person-swapping above.
- **Gesture debounce (Hands / Gesture mode):** adds `gesture`, `gesture_score`, `gesture_frames`
  (frames the current gesture has been held) and `gesture_onset` (1 on the frame it switched;
  use it as a trigger).
  - **Gesture Hold Frames**: a new gesture must be top for this many consecutive frames before it
    replaces the current one.
  - **Gesture Min Score**: anything scoring lower counts as `None`.
  - The track label shows the debounced gesture.
  - With Task = Hands, these columns carry the handedness label instead.
- **Depth:** with Landmark Z Columns on, z flows into `table_joints` and `table_output`.
- **Where things live:** skeletons are in `python/mediapipe_skeletons.py` (names and edges
  generated from the official package). `PoseTrackerPipeline` takes any of them, and COCO-17
  remains the default.
- **Cost:** the 478-point face costs about 3 ms per frame in Python (measured outside TD).

## Threading and latency

Each node owns one worker thread that creates the task and calls `DetectForVideo` with wall-clock
timestamps. VIDEO mode is what turns on MediaPipe's tracking between frames.

- **Frames are only sent when the input actually cooked.** The plugin checks the input's
  `totalCooks`, so a still image is processed once.
- **One frame waits at a time.** A newer frame replaces a waiting one.
- **TD never waits on inference or on the GPU** (unless **Wait for Result** is on: then the cook
  reads the frame back right away and waits up to Max Wait for its result).
- **Without waiting,** the frame's copy back to the CPU is queued with a CUDA event and handed to
  the worker on a later cook once it has landed (no `cudaStreamSynchronize` in the cook), so
  results show up about two frames behind the input; `result_delay` says exactly how many.
- **The node cooks every frame while its output is used** (`cookEveryFrameIfAsked`), so a pending
  frame or result lands even when the input is static. It doesn't force a cook of its own
  (`cookEveryFrame`, only while loading): that cooked it twice in frames where the input changed.

- **A change in frame size rebuilds the task**, and so does a graph error (at most once a
  second). MediaPipe graph errors are permanent. Holistic and pose-with-mask hit one as soon as the
  input resolution changes: SegmentationSmoothingCalculator, "current_mat->rows ==
  previous_mat->rows". The rebuild costs one model load and resets MediaPipe's tracking.

LIVE_STREAM mode isn't used: its C callback has no user-data pointer, so it can't tell two nodes
apart.

Run tasks side by side (one node each) rather than chaining them: they run in parallel on separate
threads.

## Performance

Measured on an i9-13900H laptop with 640-wide frames, tracking on real footage:

| Task | ms / frame |
|---|---|
| Face (blendshapes + matrix) | ~5.6 (detector only, no face: ~1.5) |
| Hands, Max 1 | ~7–10 |
| Pose Lite / Full / Heavy | ~8 / ~12 / ~40 |
| Pose Full + mask | ~15 |
| Face + Hands + Pose Full on 3 threads | ~19 wall |
| Holistic (with blendshapes) | ~22 |
| Face Detector | ~1.3 |
| Selfie segmenter / landscape | ~4.5 / ~2.3 |
| Hair segmenter | ~32 |
| Selfie multiclass (6 classes) | ~156: not real time on CPU (`fetch_mediapipe.ps1 -Heavy`) |

Expect up to 2× run-to-run variance on hybrid Intel CPUs, probably from scheduling between P-cores
and E-cores. The Face Landmarker's detector is short-range: it only finds faces within roughly 2 m
of the camera. For small or distant faces, use YuNet in OnnxTOP.

## Testing outside TD

`build/mp_test.exe` (built alongside the plugin) runs `MpWorker` on raw RGB frames:

```
mp_test <plugins dir> <face|hands|gesture|pose|holistic|facedetector|segment> <model> <frames.rgb> <w> <h>
        [max] [seg] [out.txt] [segoutput 0/1/2] [segclass]
```

On identical frames, every task matches the official Python API exactly (max difference 0). That
covers landmarks, blendshapes, the matrix, world landmarks, handedness, gesture, holistic parts,
face detector boxes, pose and holistic masks, and segmenter confidence and category masks.

## Custom gestures

The Gesture task's canned classifier knows 7 gestures. For your own gestures, train a
classifier head with **MediaPipe Model Maker** (`mediapipe-model-maker`, Python; Linux or Colab,
not supported on Windows). Give it a folder of example images per gesture plus a `none` folder.
It exports a `gesture_recognizer.task`; point Gesture Model at it. The C API also accepts a custom
classifier alongside the canned one (`custom_gestures_classifier_options`), which MpWorker
already fills in. For simple shapes (pinch distance, finger count, a pointing direction), rules
on the 21 tracked keypoints in `table_output` are cheaper and need no training. Motion gestures
(swipe, wave, circle) need the track's history: `vx`/`vy` and keypoint trails per `track_id`.

## Not included

- **Object Detector and Image Classifier / Embedder** are in the same DLL, but YOLO26 in OnnxTOP
  is faster and better for detection.
- **Audio** (classifier, transcriber) belongs in a CPlusPlus CHOP that reuses `MpApi`.
- **Text tasks** are easier from Python with the pip package.
