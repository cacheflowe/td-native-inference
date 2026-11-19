# td-native-inference: dev guide for agents

Native ML operators for TouchDesigner: **OnnxTOP** (ONNX Runtime + TensorRT, GPU) and
**MediaPipeTOP** (Google's MediaPipe Tasks, CPU), plus the Python tracking tables and debug
overlays that consume their Info DATs. Moved out of haxlib on Nov 6, 2026; all C++ and model work
happens here now. `README.md` is the user-facing overview; `TODO.md` is the plan.

## Layout

| Path | What |
|---|---|
| `td-native-inference.toe` | Dev project. `/project1/WrapperLab`: the generated standalone wrappers on a test image (audio wrappers on `audio_source`, a speech clip in `build/`; text to speech on `text_source`). `/project1/VisionDemo` and `/project1/AudioDemo`: the generated workbenches (one toggle per model); `/project1/Components`: the hand-built components (Relight, Projector Map, Capsule Matte, Latency Loop, Lead Check); `/project1/Demos`: the hand-built demos (private, see below), one camera in. `/project1/AudioLab`: MediaPipe Audio CHOP tests and the caption chain (`scripts/lab/build_captions.py`). `/project1/startup` puts `python/` on `sys.path` |
| `wrappers/<Model>.json` | One spec per model tox: custom OP type, presets, output kind, tracker mode, and the curated parameters the wrapper exposes |
| `cpp/OnnxTOP/`, `cpp/MediaPipeTOP/`, `cpp/MediaPipeCHOP/` | Plugin sources. MediaPipeTOP also compiles `../OnnxTOP/src/Kernels.cu`, so keep them side by side. MediaPipeCHOP (audio: classify, transcribe) loads `libmediapipe.dll` from next to itself |
| `cpp/Tracking/` | The tracker in C++ (`docs/TODO-tracking-cpp.md`): the core, the Tracker DAT (`TrackerDAT.dll`, every wrapper's tracker) and `tracker_replay.dll` for the replay tests. Its own output is the reference: `tests/test_tracking_replay.py` / `test_tracking_extras.py` replay the fixtures in `tests/data/tracking/`, `tests/test_joint_gate.py` checks the joint gate; after a deliberate change regenerate with `--golden` and review the diff |
| `cpp/Puppet/` | The puppet solver in C++ (`docs/PUPPETS.md`): the core, the Puppet CHOP (`PuppetCHOP.dll`, MixamoPuppet's solver; it reads the rig from a `rest_pose` table that `puppet_runtime.capture_rest` fills) and `puppet_replay.dll`. Tested against its own stored output: `tests/test_puppet_cpp.py` (`--golden` to regenerate, then review) |
| `plugins/` | Build output. On Windows this *is* the project's `Plugins/` folder, so TD registers the custom OPs (`OnnxTOP`, `MediapipeTOP`, `MediapipeaudioCHOP`, `TrackerDAT`, `PuppetCHOP`) from it at startup. `onnxtop_tool.exe` must sit next to `OnnxTOP.dll`, `libmediapipe.dll` next to `MediaPipeTOP.dll` / `MediaPipeCHOP.dll`. Gitignored |
| `models/mediapipe/`, `models/onnx/<model>/` | Model files, gitignored. Sources and licenses: `models/README.md` |
| `python/` | Build-time and runtime Python, no per-frame tracking (that's C++; the Python tracker and puppet solver were deleted Nov 13, 2026). `wrapper_builder.py` builds/exports wrappers; `tracking_pars.py` is the tracker COMP's Setup / Tracking pages as data (keep it in step with `cpp/Tracking`'s parameters); `mediapipe_skeletons.py` is the skeletons as data (`cpp/Tracking/tools/gen_skeletons.py` generates the C++ header from it); `wrapper_runtime.py` is embedded in each wrapper. `puppet_builder.py` + `puppet_runtime.py` build / run the `MixamoPuppet` tox (`scripts/build_puppet_tox.py`; the exported tox has no character: Mixamo's license). `tox/components/HandsGeometry.tox`: the geometry network is authored in the lab's `/project1/PuppetLab/HandsGeometry` (edit it there), `scripts/build_hands_tox.py` adds the render rig and exports. |
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
  `DEMO = True` in the script to regenerate both demos and `tox/demos/VisionDemo.tox` / `tox/demos/AudioDemo.tox`. **Whenever a model is added
  (or a spec changes), regenerate the demos too** and commit the demo toxes with the wrapper.
- **Three kinds of tox:** models (`tox/`, generated from specs), components (`tox/components/`:
  built on models' outputs, no model inside; open source, tracked) and demos (`tox/demos/`: models
  and components wired together). Components and demos other than MixamoPuppet / HandsGeometry
  (own build scripts) and VisionDemo / AudioDemo (generated) are built by hand: in
  `/project1/Components/<Name>` and `/project1/Demos/<Name>`. Edit them there like any network,
  never in a generator. **Hand-built demos are private** (the user's IP): `Demos/save_demos` saves
  each to `demos/<Name>.tox` (untracked) on project save and the `.toe` keeps only the path (Save
  Backup of External off), so never commit `demos/` or `tox/demos/` beyond the two generated ones and the public
  `PersonSplitDemo` (an example the user chose to publish; `.gitignore` lists it).
  A demo links its components as external toxes (`tox/components/<Name>.tox`). POST
  `scripts/export_hand_built.py` to export both kinds (cuts shared-tox links and file syncs in the
  saved copy, loads the icon); `REFRESH_WRAPPERS = True` first rebuilds the wrappers inside each from
  their specs, keeping position, wires and changed parameters (then save the `.toe`). A new
  component needs its tox in `release-manifest.json`, a card in `scripts/build_icons.py`'s `DEMOS`
  and a section in `docs/MODELS.md` (Components).
- **Node icons:** `python scripts/build_icons.py` (outside TD; headless Chrome/Edge) renders each
  tox's card to `docs/images/icons/<Name>.png` (td-http-api's app-icon layout; logo in
  `docs/images/icons/logo.svg`). The builder loads it into a locked `icon` TOP as the COMP's
  viewer, so no renderer runs in the toxes. Re-render before rebuilding whenever `VERSION`, a
  spec's label or the plugin changes.
- **Testing wrappers:** POST `tests/td_wrapper_lab.py` to `/run` (checks every lab wrapper on the
  photos in `docs/images/sources/`, report in `build/wrapper-lab.json`; set `THUMBNAILS = True` when
  preparing a release to refresh the README thumbnails in `docs/images/models/`);
  `tests/run_clean_wrapper.ps1` tests an exported tox in a fresh TD process.
- **Python:** after editing a module, reload it (the
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
- **Never delete a running ONNX wrapper that feeds a live network: it crashes TD** (reproduced Nov 18,
  2026 with a running RVM wired to Projector Map, on the build before unloading existed too). Use
  `wrapper_builder.remove(comp)` (cuts the outputs, Active off, unloads, cooking off, destroys 60
  frames later), or wire the replacement in first and leave the old one switched off. Write-up,
  other crashes, and how to catch the next one (`plugins/onnxtop-lifecycle.log`,
  `scripts/enable_td_dumps.ps1`): [docs/learnings/td-crashes.md](docs/learnings/td-crashes.md). A
  crash loses everything unsaved in the running project: save demos (`Demos/save_demos`) and the
  `.toe` before risky steps, and do risky steps one at a time with a liveness check between.
- **Models unload when off:** OnnxTOP frees a model's GPU memory `Unload After` (30 s) after Active
  goes off, and the wrappers / VisionDemo pulse `Unload` for models that don't cook. Check it with
  `nvidia-smi`'s total memory: the `loaded` info channel goes stale while a wrapper isn't cooking.
- **Projector tests** (Projector Map, Capsule Matte, Latency Loop, Lead Check; the private
  `Demos/ProjectionRehearsal`): the components are hand-built in `/project1/Components`, rebuilt from
  scratch by `scripts/lab/build_projector_map.py` / `build_capsule_matte.py`, exported with
  `scripts/export_hand_built.py`. A demo links them as external toxes, so reloading one resets its
  parameters and calibration (the demo's `mode_logic` re-applies its settings every frame; save
  `h_auto` before a reload and write it back after). Output to the projector, per
  [docs/MODELS.md](docs/MODELS.md#projector-map-componentsprojectormaptox):
  - **Reopen the Window COMP after every TD restart, and place it with
    `scripts/place_projector_window.ps1`** (TD's Window offsets skew with mixed display scaling and
    its display index renumbers). A Custom-size window shows its TOP at native pixels: feed it the
    projector's resolution.
  - **Check the Test Grid is on the projector before pressing Calibrate.** Calibrating with the
    window closed or on the wrong screen wrote junk solves over good ones; Projector Map now
    rejects an error above Max Error (px). **Verify** (held-out dots, no feedback) is how to check a
    calibration; Output = Camera feeds back on itself.
  - After the camera or projector moves, or the camera feed is flipped / rotated, just recalibrate:
    the mapping is measured. The Flip must be upstream of the demo's input so the pose model, the
    matte and the calibration all see the same upright image.
- **Test scripts through the bridge:** put multi-line Python in a file and POST it
  (`curl.exe --data-binary "@file"`), since PowerShell has no heredoc. Nodes in different networks
  can't be wired (a test source must be a sibling of the COMP it feeds). `numpyArray()` rows start at
  the bottom, like uv. Calls that cook for a while (a model loading, a calibration) need a separate
  call to read the result, not a wait inside the same script.
- **Don't commit** models, media, built plugins or TD's numbered `.toe` saves (`.gitignore`
  covers them).

## Token budget (the user runs out)

- Bridge scripts print one compact line or small dict: counts, a few values, the error's last
  lines. Never print tables, headers, parameter lists or whole JSON reports; put big results in
  `build/*.json` and read only the keys needed. Pipe `/run` output through `head -c`.
- Delayed work (`run(..., delayFrames=)`) wraps its body in try/except and always writes its
  result file (the error included) and restores anything it changed, so a failure isn't a silent
  wait followed by a second probe.
- Find node paths once (`findChildren(..., depth=2)`), then reuse them; don't re-explore networks.
- Read files by line range or grep, not whole (TODO.md, the large .cpp files); `tail` build logs.
- Don't re-read a file just edited or re-verify what a tool already confirmed.

## Style

Python: tabs, PascalCase classes, snake_case functions, f-strings, `[Prefix]` logging. C++:
follow the existing files. Comments only where the why isn't obvious.
