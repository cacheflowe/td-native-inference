# Models

Model files aren't committed (they're large, and some licenses don't allow redistribution).
Put them here with this layout; the plugins' parameters and `td-native-inference.toe` use these
paths, relative to the project folder.

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
| `onnx/yolo26/yolo26s.onnx`, `yolo26s-seg.onnx`, `yolo26n-pose.onnx` | YOLO End2End | Ultralytics YOLO26, exported to ONNX ([docs](https://docs.ultralytics.com/integrations/onnx/)) | **AGPL-3.0**: don't redistribute; each user exports or downloads their own |
| `onnx/yunet/face_detection_yunet_2023mar.onnx` | YuNet Faces | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| `onnx/movenet/movenet-multipose-lightning.onnx` | MoveNet Multipose | Google MoveNet multipose lightning, ONNX export (exact source to confirm). Not the `_fp16` variant: its scores are always 0 | Apache-2.0 |
| `onnx/hsemotion/emotion.onnx` | Classify Crops | HSEmotion via [Emotion_onnx](https://github.com/Shohruh72/Emotion_onnx/releases) | to check |
| `onnx/depth-anything/depth_anything_v2_vits_dynamic.onnx` | Raw | [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small) | Apache-2.0 (Small only; Base/Large are CC-BY-NC) |

Per-model settings (normalization, channel order, input size, thresholds) are in
[cpp/OnnxTOP/README.md](../cpp/OnnxTOP/README.md).
