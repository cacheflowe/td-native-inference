# Changelog

## Unreleased

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
