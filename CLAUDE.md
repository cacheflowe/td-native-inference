# td-native-inference: dev guide for agents

Native ML operators for TouchDesigner: **OnnxTOP** (ONNX Runtime + TensorRT, GPU) and
**MediaPipeTOP** (Google's MediaPipe Tasks, CPU), plus the Python tracking tables and debug
overlays that consume their Info DATs. Moved out of haxlib on Nov 6, 2026; all C++ and model work
happens here now. `README.md` is the user-facing overview; `TODO.md` is the plan.

## Layout

| Path | What |
|---|---|
| `td-native-inference.toe` | Dev project. Everything lives in `/project1/ONNX_Native`; `/project1/startup` (Execute DAT) puts `python/` on `sys.path` |
| `cpp/OnnxTOP/`, `cpp/MediaPipeTOP/` | Plugin sources. MediaPipeTOP also compiles `../OnnxTOP/src/Kernels.cu`, so keep them side by side |
| `plugins/` | Build output, loaded by the CPlusPlus TOPs (`plugins/OnnxTOP.dll`, `onnxtop_tool.exe` must sit next to it; `libmediapipe.dll` next to `MediaPipeTOP.dll`). Gitignored |
| `models/mediapipe/`, `models/onnx/<model>/` | Model files, gitignored. Sources and licenses: `models/README.md` |
| `python/` | `onnxtop_tables.py` (Info DAT → trackers → tables) and `object_tracker`, `pose_`/`box_`/`face_tracker_pipeline`, `mediapipe_skeletons`. numpy only; `cv2` lazily for solvePnP head pose |
| `tox/shared/` | `DebugBB.tox`, `DebugKeypoints.tox`. Instances in the `.toe` reference these files: edit the shared tox, never an instance |
| `media/` | Test clips used by the `.toe`'s Movie File In TOPs. Gitignored |
| `docs/learnings/` | Write-ups carried over from haxlib (paths in them are haxlib's) |

## Working on it

- **C++:** edit `cpp/<Plugin>/src`, run `cpp\<Plugin>\build.cmd` (MSVC v142 + CUDA 12.3 on this
  machine; override `VCVARS`, `CUDA_ROOT`, `CMAKE_NINJA_ROOT`). The build copies into `plugins/`
  and moves a locked DLL aside as `.old-<time>`. In TD, toggle the node's **Unload Plugin** to pick
  up the new DLL. Update the plugin's README when parameters or outputs change.
- **Python:** the tracker COMPs `import onnxtop_tables`. After editing a module, reload it (the
  td-http-api bridge's `GET /reload` reloads every imported module under the project folder).
- **Models:** reference them by project-relative path (`models/...`). TensorRT engines cache in
  `trt_cache/` next to each model (YOLO detect uses `cache/trt/`); delete a cache folder to force a
  rebuild.
- **Agent access to the running project:** drag `../td-http-api/build/TdHttpApi.tox` into
  `/project1` once and save. Check `GET /server-info`'s `label` before testing, since other TD apps
  may run bridges on 3031–3040. TouchDesigner docs: the bridge's `/help`.
- **Never do heavy Python work on a background thread inside TD.** One deadlocked TD on
  Nov 6, 2026. Slice work across frames with `run()`, or use a separate process.
- **Don't commit** models, media, built plugins or TD's numbered `.toe` saves (`.gitignore`
  covers them).

## Style

Python: tabs, PascalCase classes, snake_case functions, f-strings, `[Prefix]` logging. C++:
follow the existing files. Comments only where the why isn't obvious.
