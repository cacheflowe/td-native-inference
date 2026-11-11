# Learnings

Design notes and debugging write-ups. The older ones were carried over from haxlib when this work
moved here (Nov 6, 2026); file paths inside them are haxlib's layout at the time: `data/ml/...` is now
`models/...`, `python/util/` is `python/`, `tox/haxlib/ml/onnx/` is `tox/shared/`, and
`python/scripts/onnx_*.py` stayed in haxlib. Newer ones use this repo's paths.

| File | About |
|---|---|
| [onnx-cpp-top.md](onnx-cpp-top.md) | OnnxTOP: TD's bundled ONNX Runtime + TensorRT on the GPU, design and findings |
| [mediapipe-cpp-top.md](mediapipe-cpp-top.md) | MediaPipeTOP: the official MediaPipe Tasks C API in a C++ TOP |
| [mediapipe-landmarks.md](mediapipe-landmarks.md) | Landmark models and tracker tuning (persistence, ghost tracks, freezes) |
| [debug-comp-camera-aspect.md](debug-comp-camera-aspect.md) | Why landmark overlays distorted on non-square canvases, and the fixes |
| [trt-multi-output-bug.md](trt-multi-output-bug.md) | TensorRT multi-output list-wrapping bug in a worker-thread handoff |
| [td-cook-performance.md](td-cook-performance.md) | Keeping TD's cook light: GPU readback stalls, Wait for Result, CHOP sample rates, cooking flags, Python-in-TD limits |
