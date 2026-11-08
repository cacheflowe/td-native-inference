# Models

Model files aren't committed (they're large, and some licenses don't allow redistribution).
Put them here with this layout. The wrappers refer to them as `models/...`; the plugins look for
that path under the project folder first, then under the folder holding the plugin DLL (an
installed `Documents/Derivative/Plugins/td-native-inference/models/`). The release zip bundles
every model below except YOLO26.

```
models/
  mediapipe/                 MediaPipeTOP (Google's .task / .tflite bundles)
  onnx/<model>/              OnnxTOP; TensorRT engines are cached in onnx/<model>/trt_cache/
```

## MediaPipe (MediaPipeTOP)

`powershell -File cpp\MediaPipeTOP\fetch_mediapipe.ps1` downloads all of these (add `-Heavy` for
`pose_landmarker_heavy.task`) from `storage.googleapis.com/mediapipe-models`. License: Apache-2.0.

| File | Task |
|---|---|
| `face_landmarker.task` | Face (478 landmarks, blendshapes) |
| `hand_landmarker.task` | Hands |
| `gesture_recognizer.task` | Gesture |
| `pose_landmarker_full.task` | Pose |
| `holistic_landmarker.task` | Holistic |
| `blaze_face_short_range.tflite` | Face Detector |
| `selfie_segmenter.tflite` | Segment |

## ONNX (OnnxTOP)

| File | Output Mode | Source | License |
|---|---|---|---|
| `onnx/yolo26/yolo26s.onnx`, `yolo26s-seg.onnx`, `yolo26n-pose.onnx` | YOLO End2End | Ultralytics YOLO26, exported to ONNX ([docs](https://docs.ultralytics.com/integrations/onnx/)) | **AGPL-3.0**: not redistributed. `scripts/fetch_models.ps1` downloads them, pinned to revisions and SHA-256 checked: `yolo26s` and `yolo26n-pose` from [onnx-community/yolo26s-ONNX](https://huggingface.co/onnx-community/yolo26s-ONNX) and [onnx-community/yolo26n-pose-ONNX](https://huggingface.co/onnx-community/yolo26n-pose-ONNX) (`onnx/model.onnx`, input `pixel_values`), `yolo26s-seg` from [zwh20081/yolo26-onnx](https://huggingface.co/zwh20081/yolo26-onnx) (input `images`) |
| `onnx/yunet/face_detection_yunet_2023mar.onnx` | YuNet Faces | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| `onnx/movenet/movenet-multipose-lightning.onnx` | MoveNet Multipose | Google MoveNet multipose lightning, tf2onnx 1.16.1 conversion (original converter unrecorded). Not the `_fp16` variant: its scores are always 0 | Apache-2.0 |
| `onnx/hsemotion/emotion.onnx` | Classify Crops | HSEmotion `enet_b2_7.onnx` (byte-identical), [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) (formerly HSE-asavchenko/face-emotion-recognition), `models/affectnet_emotions/onnx/` | Apache-2.0 |
| `onnx/depth-anything/depth_anything_v2_vits_dynamic.onnx` | Raw | [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small) | Apache-2.0 (Small only; Base/Large are CC-BY-NC) |

Per-model settings (normalization, channel order, input size, thresholds) are in
[cpp/OnnxTOP/README.md](../cpp/OnnxTOP/README.md).
