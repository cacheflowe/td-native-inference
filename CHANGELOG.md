# Changelog

## Unreleased

## v0.1.0 (2026-11-07)

First release.

- **13 drop-in model toxes**: MediaPipe Hands/Gesture, Pose, Face Landmarks, Face Detector,
  Holistic and Segmentation; YOLO26 Detection, Segmentation and Pose; MoveNet MultiPose; YuNet;
  Depth Anything V2; HSEmotion. Each outputs a transparent debug overlay, the input image, a
  tracking table with stable ids (plus joints/bones), and CHOP channels. Notes: docs/MODELS.md.
- **ModelDemo.tox**: every model in one COMP, switched on individually; disabled models don't cook.
- **Two native plugins** (Custom OPs): OnnxTOP (GPU, TouchDesigner's ONNX Runtime + TensorRT) and
  MediaPipeTOP (Google MediaPipe Tasks). Install by copying one folder into
  Documents/Derivative/Plugins; relative model paths resolve against that folder.
- Bundles every redistributable model. YOLO26 (AGPL-3.0) is exported locally by fetch_models.ps1.
