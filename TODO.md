# TODO: td-native-inference

This repo is the move planned below (working name `td-vision`, now **td-native-inference**,
`D:/workspace/td-native-inference`). The plan was written in haxlib; paths below that mention
`data/ml/...` are now `models/...`, and `python/util/` is `python/`.

## Status (Nov 6, 2026)

Done:
- [x] Repo created; copied from haxlib: `cpp/OnnxTOP`, `cpp/MediaPipeTOP` (no build output),
      the six tracking modules into `python/`, `DebugBB`/`DebugKeypoints` into `tox/shared/`,
      the 14 models the native chains use into `models/mediapipe/` and `models/onnx/<model>/`,
      the test clips into `media/`, five learnings into `docs/learnings/`.
- [x] `td-native-inference.toe`: haxlib.toe slimmed (toeexpand → keep `/project1/ONNX_Native`,
      minus the unrelated `PythonPhysics2d` → toecollapse). Every path rewritten to the new layout.
      haxlib's App extension removed from `/project1`. A `startup` Execute DAT puts `python/` on
      `sys.path`.
- [x] MediaPipeTOP's default model paths now `models/mediapipe/...`; `fetch_mediapipe.ps1` writes
      there. Both plugins rebuilt from this repo (`build.cmd` → `plugins/`); author set to
      td-native-inference.
- [x] Build scripts already take `VCVARS`, `CUDA_ROOT`, `CMAKE_NINJA_ROOT`, `CONFIG` overrides.
- [x] `README.md` (user overview + quick start), `models/README.md` (sources, licenses),
      `CLAUDE.md` (dev workflow), `.gitignore` (models, media, plugins, builds, caches).
- [x] History: fresh start (`cpp/` was untracked in haxlib).

Next:
- [ ] **Open `td-native-inference.toe` and check it**: the plugins load from `plugins/`, every chain
      cooks, the tracker COMPs import `onnxtop_tables`, DebugBB/DebugKeypoints load from
                  `tox/shared/`. Then save the validated project.
                  - Live audit (Nov 6): connected to `td-native-inference.1.toe` on port 3031.
                        All inference chains receive `videodevin1` through `null1`; the webcam reports
                        `Failed to start video` and supplies transparent black 128x128 pixels.
                        Temporarily tested with `people_walking` (valid 1920x1080 pixels), then restored
                        the webcam connection. All 13 model nodes and nine tracker outputs force-cooked
                        without errors, but several models were still loading/building engines and all
                        track tables were header-only. Sustained inference/detection validation is pending.
                  - Model and media references use `models/` and `media/`; the startup DAT adds `python/`
                        to `sys.path`. All nine tracker outputs produced their schemas without import errors.
                  - `DebugKeypoints` and `DebugKeypoints1` have no external tox reference; the other
                        inspected DebugBB/DebugKeypoints instances reference `tox/shared/`.
                  - Inference nodes report registered `OnnxTOP` / `MediapipeTOP` custom OP types;
                        conversion to generic CPlusPlus TOPs remains part of the packaging work below.
- [x] Add the td-http-api bridge; verify app identity, `/network`, and built-in `/help`.
- [x] Build all 13 model wrappers plus `tox/ModelDemo.tox`. Live workbench:
      `/project1/ModelDemo`, shared input -> Null -> every wrapper, Models-page
      toggles off by default, parent-owned controller gates each child COMP's allowCooking.
      Existing reference chains untouched; live toe not saved by the builder.
- [x] Moved demo: all 13 native source models accounted for, no operator references back to
      `ONNX_Native`. TOP 1 combines Active-gated debug layers in Over mode; disabled layers
      clear instead of showing frozen images. TOP 2 preserves input. Pixel/order/idle-count
      checks and default-off embedded demo re-import passed; live controls/input restored.
- [x] YOLO segmentation result corrected: TOP 1 native inference matte, TOP 2 input image,
      TOP 3 boxes, existing tracking data preserved. Demo groups it with segmentation/depth
      image layers. Routing/pixel checks passed; standalone and demo exports updated.
- [x] Sequential live inference checks: all models loaded, detector/classifier fixtures positive,
      depth image output, numeric/status CHOPs, debug-off transparency, enable/stop cycle and
      no further native cooks after stopping. All exported wrappers have no source links;
      exported ModelDemo loads with all children disabled. `tests/td_model_wrappers.py` writes
      `build/model-wrappers-live.json`. Fresh-process checks for the new variants and sustained
      video/coexistence tests remain pending.
- [ ] **First commit** (license first, below).
- [ ] Choose the license (Apache-2.0 suggested, next to MediaPipe and TensorRT); add `LICENSE` +
      `NOTICE`.
- [ ] Confirm the MoveNet ONNX source and HSEmotion's license (`models/README.md`).
- [ ] Model fetch script for the ONNX models (`scripts/fetch_models.ps1`), then the release zip.
- [x] Release tooling: `scripts/release.py` supports prerequisite checks, version/changelog
      preparation, manifest-based ZIP packaging, checksums, source commit verification, a manual
      clean-project test acknowledgment, and explicit GitHub publishing. See `docs/RELEASING.md`.
- [ ] First release: choose/add LICENSE + NOTICE, validate the full exported model set in a clean project,
      finish package-root/Python loading and model fetching, then test an extracted ZIP in a
      plain TD project. The release manifest intentionally blocks until required inputs exist.
- [ ] The rest of the plan below: Package Root, one tox per model, shared visualizers, tests.

---


Goal: a **public** repo (working name `td-vision`) holding both C++ TOPs, their shared Python
tracking layer, one tox per model, shared visualizers and a demo `.toe`. A user drops one folder
into a **plain TouchDesigner project**, with no haxlib, no conda/venv and no `sys.path` setup, and
it works.

Decided:

- The repo is public, so model and source licenses have to be clean (see Licenses).
- The haxlib Python ONNX scripts (`python/scripts/onnx_*.py`) **stay in haxlib**.
- The shared Python (tracking plus future shared code) **moves with the tools** and must have no
  haxlib dependencies.

---

## 0. Before migrating (in haxlib)

- [ ] _(your items here)_
- [ ] Restart TD to load the rebuilt MediaPipeTOP (it rebuilds the task on a resolution change),
      then check that Holistic recovers after switching sources.
- [ ] Try the gesture persistence live with real hands: Coast While Lost, Gesture Hold Frames,
      `gesture_onset`.
- [ ] Save `haxlib.toe` with the MediaPipe chains in `/project1/ONNX_Native`. They're the
      reference for the demo `.toe`.
- [ ] Commit the current state in haxlib first (`cpp/`, the new Python modules, docs), so the
      move has a clean "before" to diff against.

## 1. Repo skeleton

```
td-vision/
  plugins/          OnnxTOP.dll, onnxtop_tool.exe, MediaPipeTOP.dll, libmediapipe.dll (fetched, not committed)
  tox/
    MediaPipeHands.tox, MediaPipePose.tox, MediaPipeFace.tox, MediaPipeFaceDetector.tox,
    MediaPipeHolistic.tox, MediaPipeSegment.tox, Yolo26Pose.tox, Yolo26Detect.tox, Yolo26Seg.tox,
    YuNet.tox, MoveNet.tox, ...
    shared/         VisionCore.tox (Python), Tracker.tox, DebugBB.tox, DebugKeypoints.tox
  python/           source of truth for the shared Python (see section 3)
  models/
    mediapipe/      *.task, *.tflite
    onnx/<model>/   *.onnx
  cpp/
    OnnxTOP/        src, cmake, build script, README
    MediaPipeTOP/   src, third_party/mediapipe (headers + LICENSE), tools/mp_test, README
  scripts/          fetch_models.ps1 (+ libmediapipe.dll), build_all.ps1, check_toxes.py
  demo/             demo.toe (webcam + video sources, every tox wired)
  docs/             learnings carried over (onnx-cpp-top, mediapipe-cpp-top, trt-multi-output-bug, ...)
  LICENSE, NOTICE, README.md
```

- [ ] Choose the name and license. MIT or Apache-2.0 for our code; Apache-2.0 sits most easily
      next to MediaPipe and TensorRT.
- [ ] `.gitignore`: `cpp/**/build/`, `plugins/*.dll`, `plugins/*.exe`, `plugins/*.old-*`,
      `plugins/Plugins.json` (TD's per-machine allowlist: it holds the computer name and hashes),
      `models/**` (except READMEs), the TensorRT engine cache.
- [ ] Decide on history: a fresh start, or `git filter-repo` to carry over `cpp/` history.
      (`cpp/` is untracked in haxlib today, so there's no history yet unless we commit first.)

## 2. C++ (both plugins)

- [ ] Move `cpp/OnnxTOP` and `cpp/MediaPipeTOP`. MediaPipeTOP compiles `../OnnxTOP/src/Kernels.cu`,
      so keep them side by side or move the shared kernels to `cpp/common/`.
- [ ] Build scripts: make the VS version, CUDA path and `TD_SDK_DIR` configurable (environment
      variables or CMake presets) instead of the hard-coded VS2019 v142, CUDA 12.3 and Program
      Files Samples path.
- [ ] Have post-build copy into the repo's `plugins/`. Keep the trick of renaming a locked DLL
      aside.
- [ ] Resolve model paths relative to a **Package Root** parameter (or an absolute path), not
      `project.folder`. The defaults currently point at `data/ml/...`, which only exists in haxlib.
- [ ] TensorRT engine cache: write it to a per-machine folder (`<root>/cache/trt/` or
      `%LOCALAPPDATA%`), gitignored.
- [ ] Optional, for a Mac port later: put MediaPipeTOP's CUDA resize behind an interface with a
      CPU fallback, and keep Windows-only code in `#ifdef _WIN32` (`LoadLibrary`/`GetProcAddress`).
- [ ] CI (later): GitHub Actions on Windows with the CUDA toolkit, building the release zip.

## 3. Shared Python with no haxlib dependency

Modules that move: `object_tracker.py`, `box_tracker_pipeline.py`, `face_tracker_pipeline.py`,
`pose_tracker_pipeline.py`, `mediapipe_skeletons.py`, `onnxtop_tables.py` (rename to something like
`vision_tables.py`).

Dependencies today: standard library, `numpy`, and `cv2` (imported lazily, only for solvePnP head
pose). TD ships with numpy. **Check that cv2 ships with current TD builds.** If it doesn't, make
solvePnP optional and default to the geometric head pose, which is the default already.

Problem: the tracker COMPs' scripts do `import onnxtop_tables`, which only works because haxlib's
Python setup puts `python/util` on `sys.path`. In a plain project that import fails.

- [x] Choose a loading strategy: Python embedded as Text DATs inside each release model tox;
      file-synced to `python/*.py` in development. No external Python folder, shared core tox,
      global sys.path changes or startup DAT in the user's project. Duplication between release
      toxes is intentional; development keeps one source of truth.
- [x] Prove all six modules' cross-imports in embedded DATs using `local/modules` (Nov 6,
      TD 2025.33230). Export/reload and two distinguishable copies passed module/dependency
      identity, synthetic stable tracking and state-isolation checks. A separate TD process
      loaded the exported tox with no source directory on sys.path and no cached source modules;
      all six imports resolved to embedded DATs. Dev sync settings/expressions and source bytes
      were unchanged after export. Results are in ignored `build/embedded-imports-live.json` and
      `build/embedded-imports-clean/result-*.json`; these are probes, not a model release test.
- [x] Integrate the proven `local/modules` layout into `tox/MediaPipeHands.tox`, with component-local
      support code and a generic CPlusPlus TOP. Reference tracker networks remain unchanged.
- [ ] Verify development reloads in the wrapper.
- [x] Implement the first wrapper's recursive embedded-tox export: sync/load off,
      clear dev file/external-tox references, save release tox, restore dev settings in finally.
      Preserve parameter modes/expressions and instance settings. Verify the exported tox without
      the source checkout available. The release manifest already excludes external Python files.
      `python/mediapipe_hands_component.py` exports/restores; the TD integration test verifies source
      bytes and settings. A relocated copy outside the checkout passed fresh-process inference,
      tables, CHOP and pixel-level overlay checks with Package Root set explicitly.
- [ ] Strip haxlib references from the docstrings and comments (`onnx_yolo26_pose.py`, Round 9 in
      `td-threaded-inference-optimization.md`, and so on). Repoint them at the new repo's docs.
- [ ] Turn the offline tests in the session scratchpad into `tests/`: the synthetic hand
      persistence test, the holistic and face-detector table test, and the COCO regression. Plain
      `python -m pytest` with a fake DAT/table shim, no TD needed.
- [ ] haxlib side: `python/scripts/onnx_*.py` still import these modules. Either keep a copy in
      haxlib (simple, but it drifts) or add the new repo as a submodule and put its `python/` on
      haxlib's `sys.path`. **Pick one before deleting anything from `python/util/`.**

## 4. Toxes: one per model

Shared contract for every model tox:

| Output | Content |
|---|---|
| TOP 1 | Transparent debug overlay only (boxes + skeleton, no input image), or the mask for segmenters. Transparent black when Debug Overlay is off; visualizers cook only when on. |
| TOP 2 | Passthrough (what the model saw) or the raw mask |
| DAT 1 | `tracks`: id, box, presence, label/gesture, keypoints, extras |
| DAT 2 / 3 | `joints` / `bones` |
| CHOP | Per-track channels (`t{i}_cx`, `t{i}_presence`, `t{i}_gesture_onset`, …) + info stats (fps, `infer_ms`, count) |

Parameter pages: **Model** (the plugin's own parameters, passed through), **Tracking**
(persistence), **Debug**, **Setup** (Package Root, model file).

Wrappers with both boxes and joints share the Debug controls: **Debug Overlay** on/off and
**Overlay Layers** = Both (default), Bounding Boxes, or Joints. The on/off toggle preserves the
selected layers. Implemented and pixel-tested in MediaPipeHands; reuse for subsequent wrappers.

- [ ] Use a **CPlusPlus TOP** inside the tox (Plugin Path = `<root>/plugins/X.dll`), not a Custom
      OP. A Custom OP only loads from the project's `Plugins/` folder and needs a TD restart to
      update.
- [ ] Package Root defaults to an expression based on the tox's own external-tox path, so a
      dropped-in folder resolves itself.
      Target: one versioned folder beside the user's project, no host Plugins/models/python
      folders or startup DAT required. Verify drag/drop source-path discovery, save/reopen,
      relocation and side-by-side versions; keep an override for embedded copies. Proposed layout
      and Python-loader/cache decisions are in `docs/RELEASING.md` (not implemented yet).
- [x] Build the first `MediaPipeHands.tox` template with gesture columns, gesture-onset CHOP, native
      controls, embedded tracking/visualizers and restoring export. Tested Gesture with the official
      thumbs-up image: one `Thumb_Up`, 21 joints, 21 bones and visible overlay, including a relocated
      fresh-process test without external Python. Live development wrapper is unsaved in the toe.
- [ ] Finish automatic import-root discovery (loadTox currently needs Package Root), save/reopen,
      development reload, Hands/Gesture switching and sustained-video tests for the first wrapper.
- [ ] Build the other MediaPipe tasks, then the OnnxTOP models (YOLO26 pose/detect/seg, YuNet,
      MoveNet, Depth Anything, RF-DETR, RVM, HSEmotion as a second-stage classifier).
- [ ] MediaPipe: one tox per task (decided), sharing one internal base.

## 5. Shared visualizers

Today every chain in `/project1/ONNX_Native` has copies of DebugBB, DebugKeypoints and
OnnxTopTables. A copy edited in place silently diverges, and the tracker COMP isn't external at
all.

- [ ] Only one real copy of each: `tox/shared/{Tracker,DebugBB,DebugKeypoints}.tox`. Every instance
      inside a development model tox references the shared source. Edit the shared file, never
      an instance. Release export embeds these networks and clears their external references;
      the user does not need `tox/shared/`.
- [ ] Keep the visualizers purely data-driven: `tracks`/`joints`/`bones` + passthrough in, overlay
      out, with no knowledge of which model fed them.
- [ ] `scripts/check_toxes.py`, run inside TD before each release. It lists any shared-visualizer
      development instance whose external tox isn't under `tox/shared/`, or that has local edits.
      Separately validate release toxes have no dev file-sync/shared-tox dependencies or absolute
      development paths.

## 6. Models

Models stay external: too big for a tox, and the plugins load from a file path anyway.

- [ ] `scripts/fetch_models.ps1`: one script for everything (it supersedes `fetch_mediapipe.ps1`),
      downloading from the original sources into `models/`, with a `-Heavy` switch and checksums.
- [ ] Release zip = `plugins/` + `tox/` + `models/` (only the models whose licenses allow
      redistribution) + `demo/`.
- [ ] `models/README.md`: each model's source URL, license and the tox that uses it.

### Licenses (public repo)

| Model | License | Bundle in the release zip? |
|---|---|---|
| MediaPipe `.task` / `.tflite` | Apache-2.0 | yes |
| `libmediapipe.dll` (from the pip wheel) | Apache-2.0 | yes, with NOTICE |
| YuNet | MIT | yes |
| Depth Anything V2 Small | Apache-2.0 (Base/Large are CC-BY-NC) | Small only |
| RF-DETR | Apache-2.0 | yes |
| MoveNet | Apache-2.0 | yes |
| HSEmotion | check | check |
| **YOLO26** (Ultralytics) | **AGPL-3.0** | **no**: fetch script only, with a README note |
| **RVM** | **GPL-3.0** | **no**: fetch script only |

- [ ] Check each row against the model card before the first release.

## 7. Demo + docs

- [ ] `demo/demo.toe`: a source switcher (webcam / clip), every tox wired, a perf readout. Viewers
      off by default (each one cost about 1 fps here).
- [ ] README: a 3-step quick start (download the release, drop the folder in, drag a tox), the
      output contract, the performance tables, platform notes (Windows + NVIDIA for OnnxTOP;
      MediaPipeTOP CPU-only on Windows), the Mac status.
- [ ] Carry over `docs/learnings/{onnx-cpp-top,mediapipe-cpp-top,trt-multi-output-bug}.md` and the
      MediaPipeTOP / OnnxTOP READMEs.

## 8. Cleanup in haxlib afterwards

- [ ] Remove `cpp/`, `plugins/{OnnxTOP,MediaPipeTOP,libmediapipe}*`, `data/ml/mediapipe_tasks/`,
      and the moved Python (depending on the section 3 decision).
- [ ] Point `/project1/ONNX_Native` at the new repo's toxes, or keep it as a legacy comparison
      network.
- [ ] Update `.ai/skills/td-ml.md`, `docs/README.md` and `.gitignore`, then
      `node .ai/scripts/sync.js`.
