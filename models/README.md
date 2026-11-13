# Models

Model files aren't committed (they're large, and some licenses don't allow redistribution).
Put them here with this layout. The wrappers refer to them as `models/...`; the plugins look for
that path under the project folder first, then under the folder holding the plugin DLL (an
installed `Documents/Derivative/Plugins/td-native-inference/models/`). The release zip bundles
every model below except YOLO26, RVM, Kokoro and DAViD, which `scripts/fetch_models.ps1` downloads.

```
models/
  mediapipe/                 MediaPipeTOP (Google's .task / .tflite bundles), yamnet.tflite
  asr/                       MediaPipeCHOP speech to text (.litertlm)
  tts/                       MediaPipeCHOP text to speech (.litertlm)
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

## Audio (MediaPipeCHOP)

| File | Task | Source | License |
|---|---|---|---|
| `mediapipe/yamnet.tflite` | Classify Sounds | [MediaPipe audio classifier](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier) (`storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite`) | Apache-2.0 |
| `asr/moonshine_tiny_5s_i8.litertlm` | Transcribe Speech (default) | [litert-community/moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny) | MIT |
| `asr/whisper_tiny_30s_i8.litertlm` | Transcribe Speech | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | Apache-2.0 (Whisper: MIT) |
| `tts/kokoro_int8_fp16.litertlm` | Speak Text | [litert-community/Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M) at revision `33ad69f` (the CPU build; `kokoro_int8_gpu_fp16` needs a GPU delegate). Not in the zip (size): `scripts/fetch_models.ps1` downloads it, SHA-256 checked | Apache-2.0 |

Larger speech-to-text bundles (Parakeet TDT, Qwen3-ASR) are in litert-community too; point
**Transcriber Model** at one to try it.

## ONNX (OnnxTOP)

| File | Output Mode | Source | License |
|---|---|---|---|
| `onnx/yolo26/yolo26s.onnx`, `yolo26s-seg.onnx`, `yolo26n-pose.onnx` | YOLO End2End | Ultralytics YOLO26, exported to ONNX ([docs](https://docs.ultralytics.com/integrations/onnx/)) | **AGPL-3.0**: not redistributed. `scripts/fetch_models.ps1` downloads them, pinned to revisions and SHA-256 checked: `yolo26s` and `yolo26n-pose` from [onnx-community/yolo26s-ONNX](https://huggingface.co/onnx-community/yolo26s-ONNX) and [onnx-community/yolo26n-pose-ONNX](https://huggingface.co/onnx-community/yolo26n-pose-ONNX) (`onnx/model.onnx`, input `pixel_values`), `yolo26s-seg` from [zwh20081/yolo26-onnx](https://huggingface.co/zwh20081/yolo26-onnx) (input `images`) |
| `onnx/yunet/face_detection_yunet_2023mar.onnx` | YuNet Faces | [opencv_zoo](https://github.com/opencv/opencv_zoo) | MIT |
| `onnx/movenet/movenet-multipose-lightning.onnx` | MoveNet Multipose | Google MoveNet multipose lightning, tf2onnx 1.16.1 conversion (original converter unrecorded). Not the `_fp16` variant: its scores are always 0 | Apache-2.0 |
| `onnx/hsemotion/emotion.onnx` | Classify Crops | HSEmotion `enet_b2_7.onnx` (byte-identical), [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) (formerly HSE-asavchenko/face-emotion-recognition), `models/affectnet_emotions/onnx/` | Apache-2.0 |
| `onnx/depth-anything/depth_anything_v2_vits_dynamic.onnx` | Raw | [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small) | Apache-2.0 (Small only; Base/Large are CC-BY-NC) |
| `onnx/neuflow/neuflow_mixed.onnx` | Raw (frame pair) | [NeuFlow v2](https://github.com/neufieldrobotics/NeuFlow_v2) optical flow, the "mixed" weights; ONNX export from [ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow](https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow) release 0.1.0 (MIT scripts; fp32, 42 MB, sha256 7a07929423133f9267e33aec322873ef1dfcb17224020f30c794e569d353624d). Inputs `input1` (previous frame) and `input2` (current), [1, 3, 432, 768] RGB 0-1; output `output` [1, 2, 432, 768], the flow in pixels (x right, y down) | Apache-2.0 |
| `onnx/modnet/modnet.onnx` | Raw | [MODNet](https://github.com/ZHKKKe/MODNet) portrait matting, ONNX export from [Xenova/modnet](https://huggingface.co/Xenova/modnet) on Hugging Face (fp32, 25 MB). Input normalized to -1..1 (Input Normalize = -1 to 1), dynamic H/W in multiples of 32, output alpha 0-1 | Apache-2.0 |
| `onnx/mediapipe_holistic/pose_landmarks_detector.onnx`, `hand_roi_refinement.onnx`, `hand_landmarks_detector.onnx`, `face_landmarks_detector.onnx` | MediaPipe Holistic | The landmark models inside `mediapipe/holistic_landmarker.task`, converted by `scripts/convert_holistic_onnx.py` (tf2onnx 1.16.1, opset 17; NCHW float input `image`, the pose model's segmentation output dropped). Outputs match the TFLite originals to float32 rounding | Apache-2.0 |
| `onnx/david/normal-model-vitb16_384.onnx`, `foreground-segmentation-model-vitb16_384.onnx`, `multi-task-model-vitl16_384.onnx` | Raw | [DAViD](https://github.com/microsoft/DAViD) (Microsoft, ICCV 2025): human surface normals and soft matte (ViT-B each), and depth + normals + matte in one (ViT-L), from the repo's release links, SHA-256 checked. Not in the zip (2.3 GB): `fetch_models.ps1` downloads them. The ViT-B depth model isn't used: its output range is too narrow for TensorRT FP16 (banded) and it takes 43 ms in FP32 | MIT |
| `onnx/rvm/rvm_mobilenetv3_fp32.onnx` | Raw (recurrent) | [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) MobileNetV3, fp32 ONNX from the v1.0.0 release, SHA-256 `88d45312...`. Inputs `src`, `r1i`..`r4i` (state), `downsample_ratio`; outputs `fgr`, `pha`, `r1o`..`r4o`. Not bundled: `fetch_models.ps1` | GPL-3.0 |

Per-model settings (normalization, channel order, input size, thresholds) are in
[cpp/OnnxTOP/README.md](../cpp/OnnxTOP/README.md).
