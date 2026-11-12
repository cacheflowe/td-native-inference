# OnnxTOP — GPU-resident ONNX inference as a CPlusPlus TOP

A CUDA-mode CPlusPlus TOP that runs ONNX models on **TouchDesigner's own bundled ONNX Runtime**,
using the TensorRT execution provider with CUDA as the fallback. Frames stay on the GPU the whole
way through:

``
input TOP (cudaArray, any 8/16/32-bit RGBA or Mono format)
  └─ CUDA kernel: letterbox + bilinear resize + normalize + RGB/BGR → the model's layout (NCHW or NHWC) and type (f32 / f16 / int32 / uint8)
       └─ ORT session, IoBinding on device buffers, same cudaStream as TD's interop
            ├─ YOLO seg: per-detection mask decode + composite (CUDA) → output TOP, aligned to the INPUT image
            ├─ YOLO / MoveNet / YuNet: detections (≤ 0.5 MB, the only readback) → Info DAT / Info CHOP
            ├─ Classify Crops (e.g. HSEmotion): boxes from a DAT → GPU crop + resize into a fixed batch → per-crop class scores → Info DAT
            └─ Raw mode (e.g. Depth Anything): GPU min/max normalize + map back to the input frame → output TOP
``

Measured on an RTX 4090 Laptop GPU with `yolo26s-seg` (640×640): **2.0 ms inference, 2.4 ms for the
whole pipeline**. For the same model, ORT's CUDA EP takes 5.4 ms. The Python `ONNXInferenceManager`
path pays a full readback, numpy work and the copyNumpyArray upload on top of that. See
[docs/learnings/onnx-cpp-top.md](../../docs/learnings/onnx-cpp-top.md) for design notes and findings.

**Requires TD 2025.32302 or later, installed with the Web Installer's "Public ONNX" option.** That is
the same requirement as Derivative's
[ONNXCandyStyleTOP](https://github.com/TouchDesigner/CustomOperatorSamples/tree/main/TOP/ONNXCandyStyleTOP)
sample, which also loads TD's bundled ORT.

## Build

Requirements: CUDA Toolkit 12.x, CMake ≥ 3.24, Ninja, and an MSVC toolset that your CUDA version
supports. **CUDA < 12.4 needs MSVC v142 (VS 2019 Build Tools)**, because MSVC 14.4x's STL hard-errors
on older nvcc (`STL1002`).

``bat
cpp\OnnxTOP\build.cmd
``

`build.cmd` reads `VCVARS`, `CUDA_ROOT`, `CMAKE_NINJA_ROOT` and `CONFIG` from the environment. The
defaults match this machine. Outputs are copied to `<repo>/plugins/`:

- `OnnxTOP.dll`: the plugin.
- `onnxtop_tool.exe`: the out-of-process TensorRT engine builder, plus a bench. **It must sit next to
  the DLL.**

The TD plugin SDK headers are read from the installed TD (`TD_INSTALL_DIR`). ORT headers are vendored
in `third_party/onnxruntime` (v1.23.2, MIT). Nothing links against `onnxruntime.lib`.

## Use in TouchDesigner

1. Add a **CPlusPlus TOP** and set **Plugin Path** to `plugins/OnnxTOP.dll`.
2. Wire a video TOP into it and set **Model** to an `.onnx` file, e.g. `models/onnx/yolo26/yolo26s-seg.onnx`.
3. **First use of a model:** with Execution Provider **Auto** (the default), the node runs on CUDA
   within seconds while `onnxtop_tool.exe` builds the TensorRT engine in the background (several
   minutes; frame rate is lower meanwhile), then switches to TensorRT. The Info CHOP's
   `trt_building` channel is 1 during the build. Later loads use the cached engine directly (about
   5 s). With **TensorRT** the node shows nothing until the build finishes.
4. **Rebuilding while TD is open:** `build.cmd` moves a locked `OnnxTOP.dll` aside as
   `.old-<time>` and copies the new one in. For a CPlusPlus TOP, toggle **Unload Plugin** to pick it
   up. For a Custom OP (created from the TAB menu, auto-loaded from `Plugins/`), restart TD.
5. **Nvidia Background TOPs** load the NVIDIA Video Effects SDK's own TensorRT. If one cooks before
   OnnxTOP, TensorRT is disabled for that TD session and the node runs on CUDA with a warning. See
   [docs/learnings/onnx-cpp-top.md](../../docs/learnings/onnx-cpp-top.md).

### Parameters

| Page | Parameter | Notes |
|---|---|---|
| ONNX | Model | `.onnx` with one image input `[1, 3, H, W]` or `[1, H, W, 3]`, float32 / float16 / int32 / uint8 (H, W fixed, or dynamic + Dynamic Size). A relative path is looked up under the project folder, then under the folder holding `OnnxTOP.dll` (an installed `Plugins/<folder>/models/...`) |
| ONNX | Execution Provider | **Auto** (default): CUDA now, TensorRT once its engine is built. **TensorRT**: wait for the build. **CUDA**: never build. TensorRT falls back to CUDA if session creation fails. Saved nodes keep their value |
| ONNX | TensorRT FP16 | Part of the engine cache key |
| ONNX | Reload Model | Retries after a failure, or reloads in place |
| ONNX | Scalar Input | Value for a model's extra single-number float input, e.g. RVM's `downsample_ratio` (the model works on the frame scaled by this: 0.25 for HD, 0.5 for SD). Only enabled when the model has one; changing it reloads the model |
| ONNX | Reset State | Recurrent models: forget the previous frames (after a cut). Only enabled when the model has state |
| ONNX | Active | Off = skip inference, clear the output |
| ONNX | Wait for Result | **On** (default): the cook waits for this frame's inference, so detections match the image exactly. **Off**: the cook queues the inference and returns; the next cook decodes it (`result_delay` 1), so TouchDesigner's main thread doesn't sit out the inference time. A still input runs synchronously either way. Raw mode never waits (its output stays on the GPU, same frame). |
| ONNX | Output Mode | **YOLO End2End** (Ultralytics NMS-free det/seg/pose; also DETR-style detect exports with separate `logits [1, N, C]` + `pred_boxes [1, N, 4]` outputs, like `models/onnx/yolo26/yolo26s.onnx`, decoded like `onnx_yolo26_obj_det.py`), **MoveNet Multipose**, **YuNet Faces**, **Classify Crops**, **Raw Output Tensor**, **Pose Crops** (RTMPose / RTMW / RTMW3D on person boxes), **Landmark Crops** or **MediaPipe Holistic** (both below). A model that doesn't match the mode errors instead of outputting black. Saved nodes keep their value |
| ONNX | Letterbox | Keep aspect with gray padding, like Ultralytics. Off = stretch |
| ONNX | Letterbox Padding | **Gray** (default, YOLO's training padding) or **Edge Pixels** (the image's border repeated, cv2.BORDER_REPLICATE: what DAViD's own code feeds; gray bars leave artifacts along its outputs' edges) |
| ONNX | Input Normalize | **0 to 1** (YOLO), **ImageNet mean/std** (Depth Anything, most ViTs), **−1 to 1**, **0 to 255** (MoveNet, YuNet), **Custom** (tensor = rgb01 · Custom Scale + Custom Offset, per channel) |
| ONNX | Channel Order | RGB, or BGR for OpenCV-trained models (YuNet). Applied before Input Normalize's per-channel values |
| ONNX | Dynamic Size (W, H) | Only for models whose input H/W are dynamic; greyed out otherwise. Each size is its own TensorRT engine, and only these models include the size in the cache key |
| ONNX | TensorRT Cache Folder (optional) | Leave empty: `trt_cache/` next to the model. Delete that folder to force a rebuild |
| Detection | Confidence Threshold / Class Filter / Max Detections | Class Filter −1 = all, 0 = COCO person (YOLO only). MoveNet: person score, e.g. 0.1 |
| Detection | NMS IoU Threshold | YuNet only (default 0.3, cv2.FaceDetectorYN's). YOLO end2end and MoveNet need no NMS |
| Detection | Mask Threshold / Matte Min Confidence / Matte Min Mask Fill | Seg only: which masks reach the matte (every detection is still reported) |
| Detection | Mask Box Margin | Seg only: each mask fades out over this fraction of its box's size outside the box (at least two mask cells; default 0.1). 0 = the old hard crop at the box, which cut off hair and fingertips in blocky steps when a box was tight |
| Detection | Matte NMS IoU | Seg only: a mask whose box overlaps a more confident detection's by more than this (box IoU) stays out of the matte, like the tracker's NMS (keep the two equal; default 0.5, 1 = off). The model sometimes boxes one person twice; in Instances mode the two masks would split that person's pixels |
| Detection | Matte Encoding | Seg only. **Grayscale** (default): the matte in R, G, B and A. **Instances**: R and A the matte, G the detection that owns each pixel (Info DAT `id` + 1, 0 = none; where masks overlap, the most confident one wins), B its class + 1. Half floats hold the ids exactly |
| Crops | Crop DAT | Boxes to classify: a tracks table (`track_id, x_left, y_bottom, x_right, y_top`, e.g. `OnnxTopTables_Face/script_tracks`) or an Info DAT (`id, u1, v1, u2, v2`). The node's input must be the image the boxes came from |
| Crops | Batch Size (Max Crops) | Fixed batch for a dynamic-batch model (default 16, max 64): every frame runs this many, unused slots zeroed, so the shape never changes. Part of the TensorRT cache key |
| Crops | Crop Scale / Square Crops | Grow or shrink each box around its center; Square Crops squares it (off = stretched to the model input, like `onnx_hsemotion.py`) |
| Crops | Classify Interval (Frames) | Run the classifier every Nth frame and hold the results in between (default 1; 3 = `onnx_hsemotion.py`'s Emotioninterval). Cost drops proportionally |
| Crops | Softmax | Logits → probabilities (default on) |
| Crops | Class Names | Info DAT column names in output order (empty = `c0, c1, …`) |
| Holistic | Hands / Face Mesh | MediaPipe Holistic mode: run the hand models (two per person) and the face mesh. Max Detections (Detection page) is the number of people |
| Raw | Raw Output Index | Which output to show. `[N, H, W]` / `[.., 1, H, W]` → Mono32Float, `[.., C, H, W]` → RGBA32Float (C ≤ 4) |
| Raw | Normalize | **Min / Max** (default) maps the frame's value range (letterbox padding excluded) to 0–1 on the GPU; **None** passes raw values |
| Raw | Range Smoothing | Temporal smoothing of that range (0 = per frame, default 0.9) so brightness doesn't pump |
| Raw | All Outputs (Color Buffers) | Also write every other image-shaped output to color buffers 1, 2, … in output order, raw values (no normalize), same framing; read them with a Render Select TOP (Buffer Index). One inference for a multi-output model (DAViD multi-task: depth in the main output, normals in buffer 1, matte in buffer 2) |
| Raw | Map to Input Frame | On (default): output at the input's size with the letterbox undone. Off: the tensor's own size |

#### MoveNet multipose

`models/onnx/movenet/movenet-multipose-lightning.onnx` (**not** the `_fp16` file next to it: its person scores are always 0, even in ONNX Runtime on the CPU; use TensorRT FP16 for half precision): Output Mode **MoveNet Multipose**, Input
Normalize **0 to 255**, Dynamic Size a multiple of 32 (256 × 256, the model's native size). The
input is int32 NHWC; the node handles that. Up to 6 people with 17 COCO keypoints each, in the
same Info DAT columns as YOLO pose. On the bus test image it matches ONNX Runtime run from Python
to within 0.001. With TensorRT FP16 (engine build ~2 min) the whole pipeline takes about 1.5 ms per
frame. For tracking, set `OnnxTopTables` Tracker Mode to **Pose (MoveNet multipose)**.

#### YuNet faces

`models/onnx/yunet/face_detection_yunet_2023mar.onnx`: Output Mode **YuNet Faces**, Input Normalize
**0 to 255**, Channel Order **BGR**, Confidence Threshold about 0.3–0.5 (the tracker's own
Lowconfthreshold is 0.3). The 12 per-stride outputs are decoded exactly like
`cv2.FaceDetectorYN` (score = √(cls · obj), boxes and 5 keypoints relative to grid cells), then
NMS. On the bus test image it matches `cv2.FaceDetectorYN` to three decimals; with TensorRT FP16
(engine build ~1.5 min) the whole pipeline takes about 1.2 ms per frame. Keypoints
(`eye_r, eye_l, nose, mouth_r, mouth_l`) go in the usual `kp{i}_u/v/c` columns with c = 1. For
tracking and head pose, set `OnnxTopTables` Tracker Mode to **Face (YuNet)**.

#### HSEmotion (emotion per tracked face)

A second node in **Classify Crops** mode runs `models/onnx/hsemotion/emotion.onnx` on the faces from
the YuNet node above:

1. Output Mode **Classify Crops**, Input Normalize **ImageNet**, Channel Order **RGB**, input =
   the same video as the YuNet node.
2. **Crop DAT** = `OnnxTopTables_Face/script_tracks` (tracked faces with stable IDs),
   **Class Names** = `anger disgust fear happiness neutral sadness surprise`.
3. Wire the node's Info DAT into `OnnxTopTables_Face`'s **2nd input**. Its `script_scores`
   (`onnxtop_tables.write_scores`) joins the scores onto the tracks by `track_id`, smoothed per
   track (Setup page: **Class Score Smoothing**, **Class Column Prefix** = `emotion`). `out_tracks`
   gains `emotion_label` + `emotion_anger` … `emotion_surprise` (the Python HSEmotion script's
   columns), and the label cell reads e.g. `22 NEUTRAL`, so DebugBB shows it.

The node's image output is a contact sheet of the crops it classified (128 px tiles), for checking
what the model sees. Batch 16 at 260²: about 4.2 ms per frame with TensorRT FP16 (engine build
~4.5 min, cached for batch 16), about 25 ms on CUDA. The whole batch runs every time, so size it
to the faces you expect: batch 8 is about 2.3× faster than 16 (CUDA bench, 5 faces). With
Classify Interval 3 (the Python script's cadence) the average cost drops another 3×. The crop kernel matches a Python reference with the same sampling to 0.001;
against `onnx_hsemotion.py` itself the probabilities differ by a few hundredths because that
script crops an integer box from its downscaled working frame, while the node crops the exact box
from the full-resolution input.

#### Depth Anything V2

`models/onnx/depth-anything/depth_anything_v2_vits_dynamic.onnx` (or `_vitb_indoor_`): Output Mode
**Raw** (YOLO mode errors with "doesn't look like a YOLO end2end export"), Input Normalize **ImageNet**, Dynamic Size a multiple of 14 (518 × 518, or e.g. 924 × 518
to match 16:9 input with almost no letterbox). The output is relative inverse depth (near =
bright), min/max normalized. On an RTX 4090 Laptop GPU, ViT-S at 518² takes about 5 ms per frame
with TensorRT FP16 (engine build ~3 min) and about 25 ms on CUDA.

#### Recurrent models (Robust Video Matting)

`models/onnx/rvm/rvm_mobilenetv3_fp32.onnx`: Output Mode **Raw**, Execution Provider **CUDA** (the
state's shape changes after the first frame, which TensorRT would rebuild an engine for), Input
Normalize **0 to 1**, Raw Output Index **1** (`pha`, the alpha; 0 is `fgr`, the foreground), Raw
Normalize **None**, Dynamic Size a multiple of 32, **Scalar Input** = `downsample_ratio`.

An input named `<x>i` paired with an output named `<x>o` (RVM's `r1i`..`r4i` / `r1o`..`r4o`) is
recurrent state: it starts as a 1×1×1×1 zero tensor and each run's output becomes the next run's
input, on the GPU, with no readback. **Reset State** returns it to zero. State advances once per run,
so a still input converges rather than flickers. A lone `[1]` float input gets Scalar Input. Neither
appears in the Info DAT's input list, which names the image input only.

#### DAViD (human depth, normals, matte)

Microsoft's [DAViD](https://github.com/microsoft/DAViD) models (MIT): `models/onnx/david/`
`normal-model-vitb16_384.onnx`, `depth-model-vitb16_384.onnx`,
`foreground-segmentation-model-vitb16_384.onnx` (ViT-B) and `multi-task-model-vitl16_384.onnx`
(ViT-L, all three). Output Mode **Raw**, Input Normalize **0 to 1**, Channel Order **BGR**,
Letterbox Padding **Edge Pixels**; 512 × 512 fixed input. Normals come out as -1..1 vectors
(RGBA32Float), depth as relative depth (near = small; Min / Max normalize it), the matte 0..1.
Multi-task: Raw Output Index 0 (depth) with **All Outputs** on puts normals in color buffer 1 and the
matte in buffer 2. TensorRT FP16 on an RTX 4090 Laptop GPU: about 10 ms for each ViT-B model, 21 ms
for the multi-task ViT-L (engine builds 2-4 min). The two depth outputs run opposite ways: ViT-B
depth is near = small, the multi-task one near = large. The ViT-B depth model's whole output sits in
about 0.419-0.426, which FP16 rounds into visible bands (43 ms in FP32), so the wrappers take depth
from the multi-task model. Trained on synthetic people only: the background
is meaningless in depth and normals; mask with the matte.

#### MediaPipe Holistic on the GPU

`models/onnx/mediapipe_holistic/pose_landmarks_detector.onnx`: Output Mode **MediaPipe Holistic**,
**Crop DAT** = a person detector's tracks (e.g. a nested Yolo26Detect's `tracker/script_tracks`),
**Max Detections** = people. The hand ROI refinement, hand and face models load from the same
folder (`scripts/convert_holistic_onnx.py` makes all four from `holistic_landmarker.task`); all four
build TensorRT engines and upgrade together. Input normalization is fixed (0 to 1).

Per frame, in three stages with one GPU wait each: pose on each person's crop (a new detector box,
or the crop from last frame's alignment points, as MediaPipe tracks); then the hand ROI refinement
model on crops from the pose's wrist / index / pinky, and the face mesh on a crop from the pose's
face points (or last frame's mesh while it agrees); then the hand model. `Holistic.cpp` has the
ROI logic and decoding, after MediaPipe's calculators (projection through the rotated crop, heatmap
refinement of the pose, hand ROI tracking, world landmarks rotated back, hands moved onto the pose
wrist). Differences from MediaPipe: people come from the detector instead of BlazePose's detector,
the face crop from the pose's face points instead of BlazeFace, and there's no landmark smoothing.
The Info DAT is MediaPipeTOP's Holistic table (553 points, `kp{i}_z`, `w{i}_x/y/z` for the first 75),
so `OnnxTopTables` mode **MediaPipe Holistic** reads it unchanged. The image output is a contact
sheet of every crop. Against CPU Holistic (a still photo): pose 2.5 px, hands 2–4 px, face mesh
0.7 px; world landmarks within a few cm. 5.7 ms per frame for one person (RTX 4090 Laptop GPU).

#### Landmark Crops

Runs any single-image model on rotated crops: the Crop DAT holds MediaPipe NormalizedRects
(`id, roi_x, roi_y, roi_w, roi_h, roi_angle`: center and size as fractions of the image, top-down,
radians); outside the image the edge pixels repeat, like MediaPipe's ImageToTensor. Every output
with at most 4096 values per crop goes to the Info DAT raw (`o{k}_{i}`: value i of output k), one row
per crop, for Python to decode. `python/holistic_onnx.py` drives Holistic's models this way; the
Holistic mode above replaced it for speed.

### Outputs

- **Image:**
  - Seg models: a soft matte (RGBA16Float, value in all four channels) at the Common page resolution,
    aligned to the *input* image, with the letterbox undone.
  - Det/pose models: black.
  - Raw mode: the tensor, normalized and mapped as set on the Raw page.
- **Info DAT:** one row per detection with `id, class, conf, u1, v1, u2, v2`, in TD UV space of the
  input image (origin bottom-left, same convention as `onnx_yolo26_seg.py`). Seg models add
  `mask_fill` (share of the box the mask covers at Mask Threshold, computed like the Python seg
  script's `mask_area_ratio`). Pose models add `kp{i}_u, kp{i}_v, kp{i}_c`.
- **Info CHOP:** `loaded, loading, tensorrt, num_detections, infer_ms, execute_ms, model_w, model_h,
  trt_building, result_delay, run_ms`. `infer_ms` is the inference's GPU time (CUDA events);
  `result_delay` is how many input frames old the current detections are (0 with Wait for Result
  on); `run_ms` is the CPU time ONNX Runtime takes to launch the run (paid in the cook either way).
- **Info popup** (middle-click): ORT version and path, the active provider, I/O shapes, and recent
  ORT log lines.

### Tracking and visualizers

The node itself outputs raw per-frame detections; tracking stays in Python, where it's cheap
(a dozen numbers per person) and already tuned:

- [python/pose_tracker_pipeline.py](../../python/pose_tracker_pipeline.py) holds
  `PoseTrackerPipeline`, the backend-agnostic half of YOLO26 pose post-processing:
  - `KeypointTracker` matching
  - keypoint and box smoothing
  - visibility hysteresis
  - the display gate
  - the `table_output` / `table_joints` / `table_bones` writers
  - the shared custom-parameter definitions

  Both haxlib's `onnx_yolo26_pose.py` (the Python ONNX Runtime
  backend) and this node use it.
- [python/box_tracker_pipeline.py](../../python/box_tracker_pipeline.py) holds
  `BoxTrackerPipeline`, the same thing for detect and seg models: ByteTrack, box smoothing, the
  seg presence ramp, and the tracks table (its label reads `<id> <class name> <score>%`, e.g.
  `161 person 94%`). haxlib's `onnx_yolo26_obj_det.py`
  and haxlib's `onnx_yolo26_seg.py` use it too.
- [python/face_tracker_pipeline.py](../../python/face_tracker_pipeline.py) holds
  `FaceTrackerPipeline` for YuNet: ByteTrack, box and keypoint smoothing, head direction
  (yaw/pitch/roll, geometric or solvePnP), and the tracks/joints tables.
  haxlib's `onnx_yunet.py` uses it too. OnnxTOP letterboxes, so the
  C++ path skips the Python script's square-input box-aspect correction (`size_aspect=1`), and
  solvePnP's frame size comes from the COMP's `null_passthrough` TOP, or else from the input
  of the OnnxTOP its Info DAT reads. In td-native-inference.toe, `OnnxTopTables_Face` (Tracker Mode Face, fed by
  `info3` → `Onnx_YuNet`) is the YuNet copy; `OnnxTopTables1` does the same for MoveNet.
- [python/onnxtop_tables.py](../../python/onnxtop_tables.py) turns the Info DAT into
  pipeline detections. It keeps one pipeline per COMP, so tracker state survives between cooks.
  The COMP's **Setup** page has **Tracker Mode** (Pose / MoveNet / Face / Detect / Segment;
  MoveNet is the pose pipeline with `MOVENET_DEFAULTS`), which must match the
  model; on a mismatch the tracks table shows an error row naming the mode to pick. Switching
  mode rebuilds the **Tracking** page and remembers each mode's tuned values separately. A
  Parameter Execute DAT (`parexec_mode`) calls `onnxtop_tables.setup_parameters()` on change.

In each generated model tox, the `tracker` COMP contains `in_info`, then `script_tracks`
(advances the tracker), then `script_joints` / `script_bones`. Those two are wired after
`script_tracks` so they format the same frame. The outputs are 0 tracks, 1 joints and 2 bones,
in the same formats the Python COMPs write, so `DebugBB` and `DebugKeypoints` work unchanged
(joints are written in Pose, MoveNet and Face modes; bones only for poses). The **Tracking** page has the same parameters
as the matching Python script.

`DebugKeypoints` also expects its parent to provide an `Inputaspectratio` parameter and a
`null_passthrough` TOP, the same contract the Python pose COMP satisfied.

Two thresholds apply in series. The node's **Confidence Threshold** decides what reaches the
Info DAT; for detect/seg, set it at or below the Tracking page's `Lowconfthreshold` so ByteTrack
still sees its low-confidence second-pass detections. The COMP's `Confthreshold` is the tracker's
admission and display gate.

In Segment mode, a track that lost its detection keeps coasting on its Kalman velocity for up to
`Tracklossframes` while `presence` fades out, so its box can drift past 0–1. That matches the
Python seg script. Not ported: per-track masks (the node's matte is one combined mask).

## Roadmap

Porting haxlib's Python scripts' models (`python/scripts/onnx_*.py` there), in order:

1. ~~Input options (layouts, int types, BGR, 0–255 / custom normalize) + MoveNet decoder~~ (done).
2. ~~**YuNet** decoder + `face_tracker_pipeline.py` + **Face** Tracker Mode~~ (done).
3. **HSEmotion**: ~~Classify Crops mode + `write_scores` join, running on YuNet's tracked faces~~
   (done). Still open: an SCRFD decoder for `detection.onnx` (2 anchors per cell, input
   `(rgb·255 − 127.5)/128` via Custom normalize), only needed to match the Python script's own
   face detector.

Then:

4. **MediaPipe in C++**: done as a separate plugin running Google's official MediaPipe Tasks (CPU),
   [cpp/MediaPipeTOP/](../MediaPipeTOP/README.md). The ONNX route below stays an option if
   MediaPipe ever has to run on the GPU (many people at once). Notes kept for that case (haxlib's `onnx_mediapipe_face.py`:
   `data/ml/mediapipe/face_detector.onnx` + `face_landmark_detector.onnx`; `onnx_opencv_hands.py`:
   `data/ml/opencv_zoo/palm_detection_mediapipe_2023feb.onnx` + `handpose_estimation_mediapipe_2023feb.onnx`;
   later BlazePose for 3D body). All are detector → **rotated** ROI crop → landmark model, so
   this extends Classify Crops with rotated crops and a landmark decoder that maps points back to
   image space. The
   Python scripts hold hard-won behavior that must carry over, not be reinvented: ROIs from the
   previous frame's landmarks instead of a fresh detection every frame, duplicate-hand
   suppression, and landmark smoothing between inference frames. See
   [docs/learnings/mediapipe-landmarks.md](../../docs/learnings/mediapipe-landmarks.md) and
   [docs/learnings/debug-comp-camera-aspect.md](../../docs/learnings/debug-comp-camera-aspect.md).
   Open question: one node per stage chained through DATs (simple, matches step 3), or one
   node that runs both models with the ROI loop kept on the GPU (fewer frames of latency).

Separately: an HTTP bridge improvements pass, listed in [TODO_http_bridge.md](../../TODO_http_bridge.md).

## onnxtop_tool.exe

``bat
:: bench: bare inference, then the full TOP pipeline on a raw RGBA8 image (writes PGM/PPM)
onnxtop_tool.exe "C:\Program Files\Derivative\TouchDesigner\bin" model.onnx tensorrt 300 --image img.rgba 810 1080 --out result

:: MoveNet: int32 NHWC input, 0-255
onnxtop_tool.exe "<TD bin>" movenet-multipose-lightning.onnx cuda 50 --size 256 256 --norm byte --movenet --image img.rgba 810 1080

:: YuNet: BGR 0-255
onnxtop_tool.exe "<TD bin>" face_detection_yunet_2023mar.onnx cuda 50 --norm byte --bgr --yunet --image img.rgba 810 1080

:: Classify Crops: boxes in source px, top-down (repeat --crop), fixed batch
onnxtop_tool.exe "<TD bin>" emotion.onnx cuda 50 --batch 16 --norm imagenet --crop 114 418 155 469 --image img.rgba 810 1080

:: Raw mode (min/max normalized output 0, mapped back to the image; writes <out>_raw.pgm), e.g. depth
onnxtop_tool.exe "<TD bin>" depth_anything_v2_vits_dynamic.onnx cuda 50 --size 518 518 --norm imagenet --raw --image img.rgba 810 1080 --out depth

:: what the TOP runs: build + cache a TensorRT engine, write the .built marker
:: (trailing W H only for dynamic-size models, then a batch size for dynamic-batch models)
onnxtop_tool.exe build "<TD bin>" model.onnx "<cache dir>" 1 0 [518 518 [16]]

:: run the TOP's own launcher (child process, retries, log, marker) from the command line
onnxtop_tool.exe selftest-build "<TD bin>" model.onnx
``

When the bench crashes, it prints the faulting module, the offset and a short backtrace.

## Layout

| File | Role |
|---|---|
| `src/Ort.h`, `src/OrtLoader.cpp` | Binds the Ort C++ API to TD's `onnxruntime.dll` at runtime (`ORT_API_MANUAL_INIT`) |
| `src/OnnxEngine.*` | Session creation (TensorRT → CUDA fallback), device IoBinding, warm-up, cache-key marker |
| `src/TrtBuilder.*` | Out-of-process engine build, with retry, cancel and logging |
| `src/Kernels.*` | Preprocess + normalize, crop preprocess + crop sheet, proto-mask decode, mask fill counts, matte composite, tensor range (min/max), tensor→texture, clear |
| `src/YoloDecoder.*` | Ultralytics end2end decode (det / seg / pose), MoveNet multipose, YuNet faces (+ NMS) |
| `src/OnnxTOP.*` | The TD plugin: params, loader thread (Auto: CUDA → TensorRT swap), CUDA interop, Info DAT/CHOP |
| `tools/onnxtop_tool.cpp` | Builder / bench / launcher self-test |
