# td-native-inference

Native (C++) machine-learning operators for TouchDesigner. Run ONNX models and Google's MediaPipe
on live video inside TD, then track what they find and draw it, with no Python ML stack to install.

For an engineering review, start with [docs/REVIEW_HANDOFF.md](docs/REVIEW_HANDOFF.md): current
architecture, completed work, evidence/limitations, portability plan, and specific review questions.

| Plugin | What it runs | Runs on |
|---|---|---|
| **OnnxTOP** | Image ONNX models through TouchDesigner's own bundled ONNX Runtime, with TensorRT. Built-in decoding for YOLO (detect, segment, pose), MoveNet multipose, YuNet faces, per-face classifiers (e.g. emotion) and raw outputs (e.g. Depth Anything). | NVIDIA GPU. Frames never leave the GPU: `yolo26s-seg` takes ~2.4 ms per frame on an RTX 4090 Laptop. |
| **MediaPipeTOP** | Google's official MediaPipe Tasks: face landmarks (478 points + 52 blendshapes), hands, gesture recognition, pose, holistic, face detection, segmentation. | CPU (Google's Windows build has no GPU support), on a worker thread. |

Both are CPlusPlus TOPs. Each outputs:
- an **Info DAT** with one row per detection (box, score, label, keypoints in TD's UV space),
- an **Info CHOP** with status and timing,
- an **image**: the segmentation mask, depth map or the frame the model saw.

On top of that, the **tracking tables** (Python, in `python/`) give each person, face or hand a stable
id, smoothing, persistence through dropouts and debounced gestures. The **debug overlays**
(`tox/shared/DebugBB.tox`, `DebugKeypoints.tox`) draw boxes and skeletons.

## Requirements

- Windows 10/11.
- TouchDesigner **2025.32302 or later**, installed with the Web Installer's **"Public ONNX"**
  option (OnnxTOP uses TD's bundled ONNX Runtime). Tested on 2025.33230.
- OnnxTOP: an NVIDIA GPU. MediaPipeTOP: any machine.
- macOS: not supported yet.

## Quick start

1. **Get the plugins** into `plugins/`. Prebuilt plugins will come with GitHub releases; for now,
   build them (see [Building](#building)). MediaPipeTOP also needs Google's `libmediapipe.dll`
   next to it: `powershell -File cpp\MediaPipeTOP\fetch_mediapipe.ps1` downloads it, along with
   the MediaPipe models.
2. **Get the models** into `models/`. See [models/README.md](models/README.md) for each model's
   source and license.
3. **Try them:**
   - In the current development project, use `/project1/ModelDemo` and enable models on its Models
     page. `/project1/ONNX_Native` retains the older reference chains. Media is not included;
     connect your own clip or Video Device In. The canonical toe may lag the user's live/numbered saves.
   - Or, in your own project, add a **CPlusPlus TOP**, set **Plugin Path** to
     `plugins/OnnxTOP.dll` (or `plugins/MediaPipeTOP.dll`) and wire a video TOP in. For OnnxTOP,
     set **Model** (e.g. `models/onnx/yolo26/yolo26s-seg.onnx`) and **Output Mode**. For
     MediaPipeTOP, pick a **Task**.

The first time OnnxTOP loads a model it runs on CUDA right away, builds a TensorRT engine in the
background (a few minutes), then switches to it. Later loads use the cached engine.

Plugin parameters, outputs and per-model settings:
[cpp/OnnxTOP/README.md](cpp/OnnxTOP/README.md), [cpp/MediaPipeTOP/README.md](cpp/MediaPipeTOP/README.md).

## Model Wrappers And Demo

The live development workbench is `/project1/ModelDemo`. Its input feeds
`input_frame` (Null TOP), then every model wrapper in a left-to-right layout. The Models page
has one enable toggle per wrapper; all are off by default. Enabling one permits that COMP to
cook; disabling it stops its cooks. The controller lives outside the disabled COMPs so it can
wake them again. The existing reference chains are retained separately in `ONNX_Native`.

TOP 1 (`out_debug`) combines all enabled models' debug outputs using **Over**. Disabled models
contribute transparent black, not their last frozen image. Masks/depth sit behind landmark/box
layers. TOP 2 (`out_passthrough`, or `out_input` in generated builds) retains the input image.

[tox/ModelDemo.tox](tox/ModelDemo.tox) is the embedded, default-off workbench. Set its Package
Root and connect your own video TOP. It needs no external Python or host-project startup setup.
The live workbench accepts the TOP wired to its input. Builder scripts do not save the toe.

Standalone wrappers under `tox/`:

| Family | Wrappers |
|---|---|
| MediaPipe | MediaPipeHands (Hands/Gesture), MediaPipePose, MediaPipeFace, MediaPipeFaceDetector, MediaPipeHolistic, MediaPipeSegment |
| ONNX | Yolo26Pose, Yolo26Detect, Yolo26Seg, MoveNet, YuNet, DepthAnything, HSEmotion |

Set Package Root, connect video, then turn **Active** on. Landmark models offer the same
Both/Bounding Boxes/Joints overlay selector described below. Box-only models expose boxes;
MediaPipeSegment and DepthAnything expose their mask/depth result on TOP 1 and input image on
TOP 2. Yolo26Seg exposes its native inference matte on TOP 1, input image on TOP 2, and boxes
on TOP 3 (`out_boxes`), alongside its tracking data. HSEmotion embeds its own
YuNet face detector and joins emotion scores onto the face tracks. Image-only models expose
native result data rather than synthetic joints/bones. Debug overlays and MediaPipe/depth results
match the incoming video dimensions; YOLO segmentation retains its native inference matte.

All 13 passed sequential live-TD loading, table/CHOP/image output, transparent debug-off and
enable/stop checks. Detector/classifier fixtures produced positive detections; depth produced an
image. Exported wrappers have no source links, and the exported workbench loads inactive. These
checks do not establish simultaneous-model performance or fresh-process portability of every model.

### MediaPipeHands

`tox/MediaPipeHands.tox` embeds the tracker, Python modules and debug visualizers. It uses a generic
CPlusPlus TOP, so it does not need registered custom OPs or the host project's Python startup DAT.

Drag it into a network, set **Package Root** to the folder containing this package's `plugins/`,
`models/` and `tox/`, then connect a video TOP and turn **Active** on. Keep the two MediaPipe DLLs together
under `plugins/`. **Gesture** uses `models/mediapipe/gesture_recognizer.task`; **Hands** uses
`models/mediapipe/hand_landmarker.task`. Model paths can be overridden on the Model page.

Outputs: TOP 1 transparent debug overlay only (transparent black with Debug Overlay off), TOP 2 the model's input image;
DATs 1-3 tracks, joints and bones; CHOP per-track numeric fields, including gesture onset, plus native
status/timing. Tracking and confidence controls are exposed on the wrapper.

On the Debug page, **Overlay Layers** selects **Both**, **Bounding Boxes**, or **Joints**. Both is
the default. **Debug Overlay** turns the entire overlay off without changing the layer selection.

Tested in TD 2025.33230 with a relocated package outside the checkout and no external Python:
one `Thumb_Up`, 21 joints, 21 bones, CHOP data and visible overlays. `loadTox()` does not preserve the
import source in `COMP.filePath`, so programmatic imports need the Package Root override. Automatic
drag/drop path discovery and project save/reopen still need validation.

## Layout

| Path | What |
|---|---|
| `td-native-inference.toe` | Dev project; main workbench `/project1/ModelDemo`, legacy references `/project1/ONNX_Native` |
| `cpp/OnnxTOP/`, `cpp/MediaPipeTOP/` | Plugin sources, build scripts, plugin docs |
| `plugins/` | Built plugins (not committed) |
| `models/` | Model files (not committed; see [models/README.md](models/README.md)) |
| `python/` | Tracking tables: `onnxtop_tables.py` and the trackers it uses (numpy only; OpenCV optional, for head pose) |
| `tox/*.tox` | 13 model wrappers and the default-off ModelDemo (support code embedded) |
| `tox/shared/` | `DebugBB.tox`, `DebugKeypoints.tox` overlays |
| `docs/learnings/` | Design notes and debugging write-ups |
| `TODO.md` | Plan: one drop-in tox per model, releases, model downloads |

## Building

Needs the CUDA Toolkit 12.x, CMake 3.24 or later, Ninja, and an MSVC toolset your CUDA supports
(**CUDA before 12.4 needs MSVC v142**, the VS 2019 Build Tools).

```bat
cpp\OnnxTOP\build.cmd
cpp\MediaPipeTOP\build.cmd
```

Both copy their output into `plugins/`. Set `VCVARS`, `CUDA_ROOT` or `CMAKE_NINJA_ROOT` first if
your installs live elsewhere. After rebuilding with TD open, toggle the CPlusPlus TOP's **Unload
Plugin** to load the new DLL.

## License

Not chosen yet (see `TODO.md`). Third-party pieces keep their own licenses: the ONNX Runtime
headers (MIT) in `cpp/OnnxTOP/third_party`, the MediaPipe headers and `libmediapipe.dll`
(Apache-2.0), and each model's license in [models/README.md](models/README.md).

## Releases And Portable Packaging

Release tooling: [docs/RELEASING.md](docs/RELEASING.md). Run
`python scripts/release.py check` for missing prerequisites. The workflow is prepare the version,
commit, package a manifest-based ZIP, test in a clean TD project, then explicitly publish.
Release artifacts live in the gitignored `build/` folder. Models are excluded by default.

The target install is one folder next to an existing project, with native binaries and models
resolved from each tox's Package Root. All 13 model wrappers and ModelDemo are exported; the older
reference chains still use a startup DAT and registered custom OPs. See the release guide for build
instructions and remaining portability work. Licensing and a reviewed model-fetch workflow still
block a public release.
