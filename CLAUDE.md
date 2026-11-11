# td-native-inference: dev guide for agents

Native ML operators for TouchDesigner: **OnnxTOP** (ONNX Runtime + TensorRT, GPU) and
**MediaPipeTOP** (Google's MediaPipe Tasks, CPU), plus the Python tracking tables and debug
overlays that consume their Info DATs. Moved out of haxlib on Nov 6, 2026; all C++ and model work
happens here now. `README.md` is the user-facing overview; `TODO.md` is the plan.

## Layout

| Path | What |
|---|---|
| `td-native-inference.toe` | Dev project. `/project1/WrapperLab`: the generated standalone wrappers on a test image (audio wrappers on `audio_source`, a speech clip in `build/`; text to speech on `text_source`). `/project1/VisionDemo` and `/project1/AudioDemo`: the generated workbenches (one toggle per model). `/project1/AudioLab`: MediaPipe Audio CHOP tests and the caption chain (`scripts/lab/build_captions.py`). `/project1/startup` puts `python/` on `sys.path` |
| `wrappers/<Model>.json` | One spec per model tox: custom OP type, presets, output kind, tracker mode, and the curated parameters the wrapper exposes |
| `cpp/OnnxTOP/`, `cpp/MediaPipeTOP/`, `cpp/MediaPipeCHOP/` | Plugin sources. MediaPipeTOP also compiles `../OnnxTOP/src/Kernels.cu`, so keep them side by side. MediaPipeCHOP (audio: classify, transcribe) loads `libmediapipe.dll` from next to itself |
| `plugins/` | Build output. On Windows this *is* the project's `Plugins/` folder, so TD registers the custom OPs (`OnnxTOP`, `MediapipeTOP`, `MediapipeaudioCHOP`) from it at startup. `onnxtop_tool.exe` must sit next to `OnnxTOP.dll`, `libmediapipe.dll` next to `MediaPipeTOP.dll` / `MediaPipeCHOP.dll`. Gitignored |
| `models/mediapipe/`, `models/onnx/<model>/` | Model files, gitignored. Sources and licenses: `models/README.md` |
| `python/` | `onnxtop_tables.py` (Info DAT → trackers → tables) and `object_tracker`, `pose_`/`box_`/`face_tracker_pipeline`, `mediapipe_skeletons`. numpy only; `cv2` lazily for solvePnP head pose. `wrapper_builder.py` builds/exports wrappers; `wrapper_runtime.py` is embedded in each. `puppet_solver`, `mixamo_retarget`, `hand_retarget` solve the puppets; `puppet_builder.py` + `puppet_runtime.py` build / run the `MixamoPuppet` tox (`scripts/build_puppet_tox.py`; the exported tox has no character: Mixamo's license). |
| `tox/shared/` | `DebugBB.tox`, `DebugKeypoints.tox`. Instances in the `.toe` reference these files: edit the shared tox, never an instance |
| `media/` | Test clips used by the `.toe`'s Movie File In TOPs. Gitignored |
| `docs/learnings/` | Write-ups carried over from haxlib (paths in them are haxlib's) |

## Working on it

- **C++:** edit `cpp/<Plugin>/src`, run `cpp\<Plugin>\build.cmd` (MSVC v142 + CUDA 12.3 on this
  machine; override `VCVARS`, `CUDA_ROOT`, `CMAKE_NINJA_ROOT`). The build copies into `plugins/`
  and moves a locked DLL aside as `.old-<time>`. Custom OP nodes (the wrappers) only pick up a new
  DLL after a TD restart; a CPlusPlus TOP pointed at the DLL can toggle **Unload Plugin** instead.
  Update the plugin's README when parameters or outputs change. With a release installed in
  `Documents/Derivative/Plugins/`, TD loads two copies of each OP: `scripts\plugins.ps1 dev` moves
  the installed binaries aside, `install` copies the local build there, `status` shows both (TD
  closed; the install's `models/` stays put).
- **Model wrappers (`tox/<Model>.tox`) are generated; never hand-edit one.** Change its spec in
  `wrappers/` (or `python/wrapper_builder.py` for all of them), then POST
  `scripts/build_wrappers.py` to the bridge's `/run`. It rebuilds into `/project1/WrapperLab`
  (keeping position and wires) and exports a self-contained tox: Python is embedded, file syncs and
  shared-tox links are stripped from the saved copy only. Wrappers use the registered custom OPs,
  so a release user installs the plugins in `Documents/Derivative/Plugins/<folder>/`. Set
  `DEMO = True` in the script to regenerate both demos and `tox/VisionDemo.tox` / `tox/AudioDemo.tox`. **Whenever a model is added
  (or a spec changes), regenerate the demos too** and commit the demo toxes with the wrapper.
- **Node icons:** `python scripts/build_icons.py` (outside TD; headless Chrome/Edge) renders each
  tox's card to `docs/images/icons/<Name>.png` (td-http-api's app-icon layout; logo in
  `docs/images/icons/logo.svg`). The builder loads it into a locked `icon` TOP as the COMP's
  viewer, so no renderer runs in the toxes. Re-render before rebuilding whenever `VERSION`, a
  spec's label or the plugin changes.
- **Testing wrappers:** POST `tests/td_wrapper_lab.py` to `/run` (checks every lab wrapper on the
  photos in `docs/images/sources/`, report in `build/wrapper-lab.json`; set `THUMBNAILS = True` when
  preparing a release to refresh the README thumbnails in `docs/images/models/`);
  `tests/run_clean_wrapper.ps1` tests an exported tox in a fresh TD process.
- **Python:** the tracker COMPs `import onnxtop_tables`. After editing a module, reload it (the
  td-http-api bridge's `GET /reload` reloads every imported module under the project folder).
- **Models:** reference them by relative path (`models/...`). Both plugins look under the project
  folder first, then under the folder holding the plugin DLL, so a preset path works in dev and in an
  installed `Plugins/<folder>/models/`. TensorRT engines cache in `trt_cache/` next to each model;
  delete a cache folder to force a rebuild.
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
