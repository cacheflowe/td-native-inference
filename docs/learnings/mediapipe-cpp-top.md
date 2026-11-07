# MediaPipeTOP: official MediaPipe Tasks in a C++ TOP

The plugin lives in [cpp/MediaPipeTOP/](../../cpp/MediaPipeTOP/README.md). Here is what we learned
building it.

## The prebuilt C API is the realistic route on Windows

- **The engine is already a C API.** The `mediapipe` pip wheel (1.x) ships
  `mediapipe/tasks/c/libmediapipe.dll` (68 MB, Windows system DLLs only), and the Python package
  calls it through ctypes. Its 24 `Mp*Create` entry points cover vision, text, audio, retrieval and
  decision tasks.
- **There's no import library.** Bind each function with `GetProcAddress`, and take the pointer
  types from the vendored headers with `decltype(&MpFooCreate)`, so signatures can't drift.
- **Vendor headers from the same git tag as the wheel** (google-ai-edge/mediapipe `v1.2.2` for
  1.2.2). The options and result structs are passed by value and pointer across the DLL boundary.
- **It's CPU-only on Windows.** `delegate = GPU` fails at graph init with "ImageCloneCalculator:
  GPU processing is disabled in build flags". MediaPipe's own docs say GPU is Ubuntu-only, and its
  GPU path needs GL ES/EGL, so building from source doesn't help either.

## API gotchas

- **The LIVE_STREAM `result_callback` has no user-data pointer.** It can't route results to one
  of several node instances. Instead, run VIDEO mode (`DetectForVideo`) on a private thread per
  node: you get the same non-blocking behavior, and VIDEO mode still runs MediaPipe's tracking.
- **VIDEO-mode timestamps must strictly increase.** Use wall-clock ms, bumped by 1 on ties.
- **`MpMatrix.data` is column-major.** The Python binding transposes it.
- **`MpClassifierOptions` has no C defaults.** Zero-init means `max_results = 0`. Python passes
  `-1` and `0.0` when unset; do the same for the gesture recognizer's canned and custom classifier
  options.
- **The pose `segmentation_masks` are `MpImagePtr` (VEC32F1).** Copy rows using
  `MpImageGetWidthStep`, because rows may be padded.
- **`MpImageSegmenterOptions.display_names_locale` must not be NULL.** A null pointer crashes
  inside `MpImageSegmenterCreate`, apparently because the C side builds a `std::string` from it.
  The Python binding passes `b''`; pass `""`.
- **Segmenter category masks are GRAY8.** Single-class selfie models write 0 = person,
  255 = background; multi-class models write the class index. Confidence masks are VEC32F1, one
  per class, and class 0 is the background in the multi-class models.
- **A graph error is permanent, and a change in input resolution causes one.** After switching
  the source from a 1024×543 clip to a 640×360 webcam, Holistic failed every frame with
  `SegmentationSmoothingCalculator ... current_mat->rows == previous_mat->rows (360 vs. 543)` and
  never recovered (0 detections, `infer_ms` ≈ 0.1). Pose with segmentation masks smooths the
  same way. MpWorker now recreates the task when the frame size changes, and after any detect
  error (at most once a second).
- **The holistic result holds its parts by value**, not as arrays. A part that wasn't found has
  `landmarks_count` 0. It's one person only and has no per-person score.
- **Face results have no per-face score**, and face landmarks have no visibility or presence. Hand
  and pose landmarks do.

## Verified equal to Python

On identical RGB frames, `mp_test.exe` and the official Python API agree with a max difference of 0
for face (478 landmarks, blendshapes, matrix), pose (landmarks, world landmarks, mask), hands
(landmarks, world landmarks, handedness), gesture, holistic (all 553 points, blendshapes, mask),
face detector (box, keypoints) and the segmenters (per-class coverage, foreground and category
masks). That rules out struct-layout mistakes.

## Performance notes (i9-13900H, 640-wide frames)

- **Face** with blendshapes and matrix: about 5.6 ms. **Pose Full**: about 12 ms. **Pose Heavy**:
  about 40 ms, too slow for real time. **Face + Hands + Pose Full** on three threads: about 19 ms
  wall.
- **`num_hands` = 2 with only one hand visible costs about 3×** (28 ms vs 9.5 ms). The palm
  detector reruns every frame looking for the missing hand. The same applies to `num_faces` and
  `num_poses`.
- **Run-to-run variance is up to 2×** on the hybrid P-core/E-core CPU (the laptop was on AC, in
  Turbo, with TD closed). It didn't come back in four later back-to-back runs (Pose Full held at
  about 12 ms), so it was probably other load on the machine. Input size makes no difference
  (256–1280 px wide all ran about 11.7 ms): the time is all model inference, and the C API exposes
  no thread-count or delegate setting to tune.
- **Holistic** takes about 22 ms. The **face detector** takes about 1.3 ms. The **selfie
  segmenter** takes about 4.5 ms (2.3 ms for landscape), the **hair segmenter** about 32 ms, and the
  6-class **selfie multiclass** about 156 ms, which is not usable in real time on the CPU.
- **Several tasks at once slow each other down.** With six MediaPipeTOPs plus the rest of the
  ONNX_Native network live in TD, inference ran 2–3× slower than in the standalone benchmark (Pose
  30 ms vs 12, selfie segmenter 21 ms vs 4.5). Each task has its own XNNPACK thread pool, and they
  compete with each other and with TD. TD frame rate only dropped a few fps per active task, and the
  bigger cost was the 10 extra node viewers (about 9 fps). Run only the tasks you need.
- **The Face Landmarker uses the short-range face detector**, which only finds faces within about
  2 m. It found nothing in full-body dance and choir footage but locked on immediately to a
  close-up.

## GPU alternatives considered

- **LiteRT** (the successor to TensorFlow Lite; MediaPipe tasks run LiteRT models internally).
  LiteRT 2.3 ships a Windows GPU accelerator, `libLiteRtWebGpuAccelerator.dll`: WebGPU on
  Direct3D, which needs `dxil.dll` and `dxcompiler.dll` next to the exe. You could run the
  `.tflite` files inside the `.task` zips on it, but then you rebuild MediaPipe's pipeline
  (anchors, ROI loop, tracking) by hand. There's also no documented D3D/Vulkan/CUDA interop with
  TD textures, so frames still go through the CPU. OnnxTOP + TensorRT is already the better GPU
  path for that work.
- **MediaPipe Web** (WebGL/WebGPU delegate) in an embedded browser: GPU-accelerated, but the
  frames and results cross the browser boundary.
