# Changelog

## Unreleased

- **Performance** page on every video tox: inference, the plugin's cook (model launch and decode
  for ONNX models), the Info DAT, the tracker, the nested detector, the debug overlay and the
  tox's main-thread total, live, in milliseconds. Computed only while shown. The plugins add
  `decode_ms` (OnnxTOP) and running averages (`infer_ms_avg`, `execute_ms_avg`, ...) to their Info
  CHOPs.
- **Tracker DAT** (`TrackerDAT.dll`, a fourth plugin to allow): the tracker in C++, writing exactly
  the tables the Python tracker wrote. MediaPipe Holistic GPU uses it: one person, all three tables,
  4.5 to 1.3 ms per frame with Face Mesh on, 2.7 to 0.4 ms with it off. The other wrappers follow.
- YOLO26 Detect / Segmentation: detections with equal scores now go through NMS in detection
  order. Before, the order depended on numpy's sort internals, so which of two overlapping
  equal-score boxes survived (and which new track got the lower id) could vary.
- MediaPipe Holistic GPU: **Output Smoothing** now defaults to 0.4 (was 0). Unlike CPU Holistic it
  has no landmark filter of its own, so it was noticeably jittery; 0.4 is about as steady as CPU
  Holistic for two thirds of a frame of lag.
- MediaPipe Holistic: **Face Mesh** on the Tracking page. Off, the tables carry the body and hands
  only (75 joints instead of 553) and the tracker costs about a quarter as much (4.7 to 1.2 ms per
  person). The Mixamo Puppet doesn't use the face. On Holistic GPU the Model page's **Face Mesh**
  does both: it skips the face model and leaves the face out of the tables.

## v0.3.0 (2026-11-12)

New models:
- **MediaPipe Holistic GPU** (`MediaPipeHolisticGPU.tox`): Holistic's own models converted to ONNX and
  run on the GPU, with Holistic's cropping logic in the plugin. Body, hands and face for several
  people at once, in the same table, so it drops in for the CPU wrapper (Mixamo Puppet included).
  Runs in the background by default: about 1.5 ms of TouchDesigner's main thread, results one frame
  late (5.7 ms with Wait for Result on, exact frame). Agrees with CPU Holistic to a few pixels. Finds
  people with a nested YOLO26 detector (AGPL-3.0, `fetch_models.cmd`) that only runs while someone is
  missing.
- **DAViD** (Microsoft, MIT): `DAViDNormals.tox` (a person's surface normals as a normal map, for
  relighting), `DAViDMatte.tox` (a soft matte with hair detail), about 10 ms each, and
  `DAViDHuman.tox` (depth, normals and matte from one model in one pass, about 21 ms). 2.3 GB, so
  `fetch_models.cmd` downloads them (it asks first).
- **MODNet** (`MODNet.tox`): portrait matting, about 2 ms.
- **Robust Video Matting** (`RVM.tox`): a matte that stays steady under fast motion (it carries the
  previous frames). GPL-3.0, so `fetch_models.cmd` downloads it (it asks first).

New toxes:
- **Mixamo Puppet** (`MixamoPuppet.tox`): poses a Mixamo character from Holistic or Pose tracking,
  over the video or on its own. You supply the character (FBX).
- **Hands Geometry** (`HandsGeometry.tox`): tracked hands as 3D spheres and cylinders, with a render
  rig. Needs TouchDesigner 2025 or later.

Plugins:
- OnnxTOP Output Modes **MediaPipe Holistic**, **Pose Crops** (keypoint models on person boxes) and
  **Landmark Crops** (any model on rotated crops, raw outputs to the Info DAT).
- OnnxTOP Raw: **All Outputs** writes a multi-output model's other outputs to extra color buffers
  (Render Select TOP), so one inference serves several images.
- OnnxTOP: **Letterbox Padding** can repeat the image's edge pixels instead of gray bars.
- OnnxTOP: recurrent models (state carried between frames on the GPU), **Scalar Input** and
  **Reset State** (for RVM).
- OnnxTOP and MediaPipeTOP no longer cook twice in frames where a result is pending; OnnxTOP has a
  `run_ms` info channel (CPU time of the inference launch).

Docs and scripts:
- A user manual (`docs/MANUAL.md`), and a guide in `docs/MODELS.md` for choosing between similar
  models (bodies, hands, faces, objects, mattes, depth) with times and licenses.
- `fetch_models.cmd` downloads RVM and DAViD (both ask first) as well as YOLO26 and Kokoro.

## v0.2.5 (2026-11-10)

- ONNX models (YOLO, MoveNet, YuNet, HSEmotion): **Wait for Result** on the Model page. On (the
  default, as before): detections match the image exactly. Off: TouchDesigner's cook no longer
  waits for the inference; results arrive a frame later (`result_delay` info channel). Depth
  Anything no longer waits for its inference at all, with the same, same-frame result. `infer_ms`
  now reports the GPU inference time.
- ONNX models build their Info DAT faster (header cached, cheaper number formatting).
- Boxes and faces (YOLO, YuNet, HSEmotion) no longer glide in from where a track was held when it's
  found again after 2 or more lost frames; they start at the new detection, like the skeletons.

- MediaPipe Sound Classifier (and Speech to Text): their output channels run at the timeline rate
  instead of the audio's 16 kHz. The wrapper's score smoothing (a Lag CHOP over all 521 classes)
  had been producing ~270 samples per class every frame, about 20 ms of TouchDesigner's frame.

## v0.2.4 (2026-11-10)

- MediaPipe Holistic (and any tracker whose missing parts come in at 0, 0): with **Output
  Smoothing** up, a hand or the face mesh no longer flies in from the corner when it's found again.
  An unseen part holds its last position, hidden; when it's back it starts at its new reading and
  stays hidden for 3 frames while smoothing settles. A whole track picked back up after 2 or more
  lost frames also starts at its new detection instead of gliding from where it was held.
- MediaPipe Hands: **Output World** on its Model page (world landmark columns, as on Pose and
  Holistic).
- MediaPipe models no longer stall TouchDesigner's cook waiting for the GPU: the frame's copy to
  the CPU is asynchronous and reaches MediaPipe a cook later. Results are about two frames behind
  the input; the new `result_delay` info channel says exactly how many (a Cache TOP on the input
  with Output Index `-result_delay` shows the matching frame).
- MediaPipe models: **Wait for Result** and **Max Wait (ms)** on the Model page wait in the cook
  for each frame's result, so fast models (Face Detector, Hands) give landmarks that match the
  image exactly. Off by default.
- MediaPipe models build their Info DAT faster (the header is cached), and the trackers skip
  re-applying a result they've already applied while a newer one is on its way.

## v0.2.3 (2026-11-09)

- MediaPipe models (Hands, Gestures, Face, Pose, Holistic, Face Detector, Segmentation) recover
  from a MediaPipe graph error within about a second. Before, the node stayed dead (0 detections)
  until TouchDesigner restarted; seen with the gesture recognizer's "Packet isn't the sole owner of
  the holder" warning.

## v0.2.2 (2026-11-09)

- YOLO26 Segmentation: masks fade out past their box (**Mask Box Margin**) instead of being cut at
  it, which clipped hair and raised fingertips in blocky steps when a box was tight.
- YOLO26 Segmentation: **Matte Encoding = Instances** keeps which detection owns each pixel (G) and
  its class (B), with overlaps going to the most confident mask. **Matte NMS IoU** keeps a duplicate
  detection of one person out of the matte (it split that person's pixels in two). Tracks get a `det_index` column,
  and `Yolo26Seg.tox` has a new TOP 4 that paints each tracked object in its track color.
- MediaPipe Hands, Pose and Holistic output landmark depth by default (`<joint>_z` columns and
  channels, `z` in `joints`); **Landmark Z** is on their Model page, and on Face's (off there).
- Every tox shows a card with its model name, engine and version in its network node, rendered
  when the tox is built (no renderer runs inside the tox).

## v0.2.1 (2026-11-08)

- `fetch_models.cmd`: double-click to download YOLO26 and the Kokoro voice. Right-clicking
  `fetch_models.ps1` > Run with PowerShell failed on machines whose execution policy refuses
  unsigned scripts from a downloaded zip (`RemoteSigned`, common); the window closed with the error.
- `fetch_models.ps1` checks downloads with .NET's SHA-256 instead of `Get-FileHash`, which can be
  missing when Windows PowerShell is started from a PowerShell 7 terminal.

## v0.2.0 (2026-11-07)

Audio, through the new `MediaPipeCHOP.dll` (MediaPipe audio tasks, CPU):

- `MediaPipeSpeechToText.tox`: streaming speech to text from any audio CHOP (Moonshine or
  Whisper), with transcript / latest-sentence DATs, speaking and new-sentence channels, and a
  closed-caption TOP. Only speech reaches the model (Skip Silence, against a tracked noise floor),
  so room noise doesn't turn into text.
- `MediaPipeTextToSpeech.tox`: Kokoro text to speech (54 voices, 9 languages) from a parameter or
  any DAT, played on the default audio device and output as 24 kHz audio, with a caption of the
  line being spoken. Its voice model (280 MB) downloads with `fetch_models.ps1`, which now also
  continues without YOLO26 if you decline its license.
- `MediaPipeSoundClassifier.tox`: YAMNet's 521 sound classes as smoothed channels, a top-results
  DAT and display, and score / active / onset channels for chosen classes.
- `AudioDemo.tox`: the three together, with an option to speak what it hears.
- The plugins report the package version in their info popup.

## v0.1.0 (2026-11-07)

First release.

- **13 drop-in model toxes**: MediaPipe Hands/Gesture, Pose, Face Landmarks, Face Detector,
  Holistic and Segmentation; YOLO26 Detection, Segmentation and Pose; MoveNet MultiPose; YuNet;
  Depth Anything V2; HSEmotion. Each outputs a transparent debug overlay, the input image, a
  tracking table with stable ids (plus joints/bones), and CHOP channels. Notes: docs/MODELS.md.
- **VisionDemo.tox**: every video model in one COMP, switched on individually; disabled models don't cook.
- Scale-relative skeleton matching (one setting for small and large people), MediaPipe-style joints
  with a white outline.
- **Two native plugins** (Custom OPs): OnnxTOP (GPU, TouchDesigner's ONNX Runtime + TensorRT) and
  MediaPipeTOP (Google MediaPipe Tasks). Install by copying one folder into
  Documents/Derivative/Plugins; relative model paths resolve against that folder.
- Bundles every redistributable model. YOLO26 (AGPL-3.0) is fetched by fetch_models.ps1: downloads checksum-verified ONNX exports from Hugging Face, no Python needed.
