# Changelog

## Unreleased

- `MediaPipeSpeech.tox`: streaming speech to text from any audio CHOP (Moonshine or Whisper),
  with transcript / latest-sentence DATs, speaking and new-sentence channels, and a
  closed-caption TOP. Uses the new `MediaPipeCHOP.dll` (MediaPipe audio tasks, CPU).
- MediaPipeCHOP: Skip Silence (only speech reaches the model), a speech threshold that follows
  the background noise, segment numbers that keep counting past the 200-sentence transcript.

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
