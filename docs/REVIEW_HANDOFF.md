# Native Inference Review Handoff

Snapshot: November 6, 2026. This is an implementation handoff, not a claim that the package is
ready for public release. Start here, then inspect the linked source and test drivers.

## Executive Summary

The project provides native ONNX and MediaPipe inference in TouchDesigner, Python tracking tables,
and transparent visualization layers. We moved the relevant code/models/visualizers out of haxlib,
built 13 model-specific wrappers, and created an individually enabled workbench.

The main live workbench is now `/project1/ModelDemo`. The user moved it up from
`/project1/ONNX_Native/ModelDemo`. All 13 original native model nodes are represented; the demo's
operator parameters/expressions were audited for references back to `ONNX_Native` and none were
found. Keep `/project1/ONNX_Native` as the legacy reference/template network: the builder still
uses its model settings when constructing new wrappers.

The intended install is one extracted package folder, with a self-contained model tox dropped into
an unrelated TD project. Python, tracking networks, and approved visualizers are embedded in the
release tox. Native binaries and model files remain external under a configurable Package Root.
No host startup DAT, global OP shortcut, registered custom OP, or Python search-path setup should
be required by a release wrapper.

## Review Boundaries

- Review source, export behavior, tests, and live wiring before modifying anything.
- Do not rebuild/reset models, overwrite user-edited networks, save the toe, or publish/commit
  without authorization. The original chains and original standalone Hands development COMP were
  deliberately retained.
- The current bridge is `http://127.0.0.1:3031`. Check `/server-info` before mutation. At this
  documentation snapshot it identified `td-native-inference.9.toe`, label `td-native-inference`,
  TD 2025.33230, project folder `D:/workspace/td-native-inference`. Saves/ports can change.
- POST actual project-local script files to `/run`; inspect the returned JSON `error`, not just
  HTTP success. `/network?path=...` inspects live wiring; `/opinfo?opType=...` describes an OP type.
- Automated full-model tests temporarily replace the demo input and change model toggles. They
  finish with all models off. Do not run them while the user is interacting with the workbench.
- Agents have exported toxes, but have not saved the live project. The user has made numbered
  saves; do not assume the canonical checked-in toe contains every current live change.

## What We Changed

1. Audited the migrated network and its relative model/media paths. The initial webcam supplied
   blank pixels, so header-only tables were not treated as proof of positive inference.
2. Added release preparation, explicit manifest packaging, hashes/provenance, verification, and
   explicitly gated GitHub publishing in [scripts/release.py](../scripts/release.py).
3. Proved local Text DAT imports for the six tracking modules, including cross-import identities,
   independent copies, restoring exports, and imports in a separate TD process.
4. Built the first Hands/Gesture wrapper with a generic CPlusPlus TOP, embedded tracker/modules,
   numeric/status CHOP, and the approved visualizers.
5. Replaced creation-order node placement with explicit left-to-right data lanes, right-side
   outputs, and support code below. Shared visualizer sources, not arbitrary instances, remain
   the development source of truth.
6. Changed debug output to transparent overlay only, never the original video composited into
   the overlay. Added Both / Bounding Boxes / Joints selection where appropriate.
7. Built all remaining native model variants and the default-off ModelDemo.
8. Followed the user's relocation of ModelDemo to `/project1`, preserving the camera connection
   and model selections during targeted updates.
9. Added an Over composite of Active-gated model results. Disabled frozen images are replaced
   with transparent black, without reactivating the disabled COMP.
10. Corrected YOLO segmentation: the native inference matte is its primary result, with the source
    image and box overlay on separate outputs. It is grouped with segmentation/depth image layers.

## Runtime Architecture

### Native Plugins

- [OnnxTOP](../cpp/OnnxTOP/README.md): GPU ONNX Runtime using TD's bundled runtime; CUDA inference
  can run while a TensorRT engine builds. Modes decode YOLO, MoveNet, YuNet, classify crops, or
  render raw outputs such as depth.
- [MediaPipeTOP](../cpp/MediaPipeTOP/README.md): Google's MediaPipe Tasks C API on a CPU worker;
  GPU image handling/resize is implemented in the native plugin. CPU inference is not proof that
  the complete plugin runs without an NVIDIA/CUDA-capable environment; verify hardware claims.
- Both publish an Info DAT, Info CHOP, and TOP result. Wrappers use generic `cplusplusTOP` nodes,
  unlike the legacy registered custom OP instances.
- Keep `OnnxTOP.dll` with `onnxtop_tool.exe`, and `MediaPipeTOP.dll` with `libmediapipe.dll`, under
  the package's `plugins/`. TD-provided runtime DLLs are not redistributed by the release helper.
- The plugins' build scripts use MSVC/CUDA/CMake/Ninja and copy results into ignored `plugins/`.
  MediaPipe also compiles shared ONNX CUDA kernels; keep the C++ source directories side by side.

### Model COMP

Typical flow: `input -> inference`, then `Info DAT -> tracker -> tracks/joints/bones`, with
numeric track values and native statistics exposed as a CHOP. Boxes/keypoints consume those
tables, not a separate Python ML runtime. HSEmotion is a two-stage exception: its own nested YuNet
detects faces, pre-score tracks provide crop boxes, and emotion scores are joined back onto tracks.
This avoids requiring another top-level model to be enabled and avoids feeding scored output back
into its own crop input.

Each wrapper owns `local/modules` Text DATs for:

- [object_tracker.py](../python/object_tracker.py)
- [mediapipe_skeletons.py](../python/mediapipe_skeletons.py)
- [box_tracker_pipeline.py](../python/box_tracker_pipeline.py)
- [face_tracker_pipeline.py](../python/face_tracker_pipeline.py)
- [pose_tracker_pipeline.py](../python/pose_tracker_pipeline.py)
- [onnxtop_tables.py](../python/onnxtop_tables.py)

Ordinary imports resolve to the wrapper-local DAT copies. NumPy is the main dependency; OpenCV is
lazy/optional for solvePnP head pose. The component helper is
[mediapipe_hands_component.py](../python/mediapipe_hands_component.py); its historical name now
covers shared model/demo support too.

### Current Model Inventory And Outputs

| Wrapper | Native task/model | Primary TOP result |
|---|---|---|
| MediaPipeHands | Hands or Gesture | Transparent boxes/joints |
| MediaPipePose | Pose | Transparent boxes/joints |
| MediaPipeFace | Face landmarks | Transparent boxes/joints |
| MediaPipeFaceDetector | Face detector | Transparent boxes/keypoints |
| MediaPipeHolistic | Holistic | Transparent boxes/joints |
| MediaPipeSegment | Selfie segmentation | Segmentation image |
| Yolo26Pose | YOLO26 pose | Transparent boxes/joints |
| Yolo26Detect | YOLO26 detection | Transparent boxes |
| Yolo26Seg | YOLO26 segmentation | Native inference matte |
| MoveNet | Multipose | Transparent boxes/joints |
| YuNet | Face detection | Transparent boxes/keypoints |
| DepthAnything | Raw depth model | Depth image |
| HSEmotion | Nested YuNet + crop classification | Face debug overlay; scores in data outputs |

TOP 2 is now the original input image. YOLO26Seg additionally exposes boxes on TOP 3 (`out_boxes`);
its TOP 1 is directly the native inference result, not the box visualization. This supersedes earlier
TOP 1 boxes / TOP 2 raw-mask documentation. Result-off is transparent black.

Tracking models expose tracks, joints, and bones DAT outputs as applicable; non-pose modes can
have header-only joint/bone tables. Image-only models expose native result data/statistics rather
than pretending to have tracked landmarks. Numeric CHOP output includes native loading/timing and
per-track numeric fields where available. Inspect actual schemas before assuming identical fields
across models.

### Demo Cooking And Composition

The demo input feeds `input_frame` (Null TOP), then every wrapper. Its Models page has one toggle
per model. Each child Active parameter binds to its corresponding parent control; a parent-owned
Parameter Execute DAT changes the child's `allowCooking`. All models are off in release exports.

The parent location is essential: an internal callback cannot wake its own disabled COMP. Changes
are applied at frame end, so enable/disable tests must allow the next frame. Native ONNX Active
alone is insufficient to avoid model loading: its execute path calls `manageLoading` before deciding
whether to infer. Stopped cook counts do not prove all already-started background engine work has
finished or that all GPU resources were released.

Each `layer_<Model>` Switch TOP selects transparent black or that wrapper's debug/result TOP using
the parent model toggle. This clears cached output when a COMP stops cooking. `debug_composite`
uses Over; box/joint layers precede image-producing layers, which sit behind them. Over is ordered,
not an image-analysis fusion operation: opaque masks/depth can cover lower image layers when
multiple image models are enabled.

Demo TOP 1 is `out_debug`; TOP 2 is `out_passthrough` (or `out_input` in generated builds). The
demo viewer points at the combined result. Both result gating and independent original-image output
are intentional. Further opacity/blending/individual image views are not implemented.

## Development Versus Portable Export

Development Text DATs sync to repository Python. Visualizer instances refer to
[DebugBB.tox](../tox/shared/DebugBB.tox) and [DebugKeypoints.tox](../tox/shared/DebugKeypoints.tox).
Edit the shared source when changing those tools. Do not add host startup/import setup to solve a
wrapper-local import problem; the legacy startup exists for legacy networks only.

Export traverses the complete COMP tree, disables DAT file sync and startup loading before clearing
file paths, clears external-tox references, and saves embedded content. It restores values,
expressions, bindings, modes, and the root hint in `finally`. `export_demo` also temporarily saves
the demo with model controls off and restores live controls/cooking states afterward.

Package Root resolves native plugins/models. A demo root feeds its children; standalone wrappers
have their own root. Automatic root discovery is incomplete: after `loadTox`, `COMP.filePath`
reports the host toe, not the imported tox. An external-tox reference can provide a source location;
otherwise use the explicit root override. Fresh-process tests used that override.

Only DLLs/tools, models, docs/licensing, and the exported tox should remain external. The release
manifest excludes development Python, shared toxes, private media, toes, engine caches, and machine
plugin allowlists. Models are deliberately not included in the initial ZIP; a download workflow is
still needed for an out-of-box install.

## Entry Points For The Reviewer

| File | Role / side effects |
|---|---|
| [build_model_wrappers.py](../scripts/build_model_wrappers.py) | Two TD `/run` stages: native registration, then configuration/export. Completed builds support refresh with all models off. Uses legacy source templates; can overwrite generated exports and rebuild selected internal branches. |
| [build_mediapipe_hands.py](../scripts/build_mediapipe_hands.py) | Original first-wrapper builder; refuses an existing original wrapper/tox. Historical, not the normal full-workbench updater. |
| [update_model_demo.py](../scripts/update_model_demo.py) | Audit model coverage/references, setup composite, synthetic pixel tests, inactive export/re-import; preserves live model controls/input. |
| [update_yolo_seg_output.py](../scripts/update_yolo_seg_output.py) | Correct matte/source/box outputs, synthetic routing tests, export standalone YOLO segmentation plus demo; preserves live controls/input. |
| [td_model_wrappers.py](../tests/td_model_wrappers.py) | Full sequential native tests plus synthetic composite/segmentation helpers. Full run requires all models off and leaves them off. |
| [td_mediapipe_hands.py](../tests/td_mediapipe_hands.py) | Detailed Hands/Gesture integration, export restoration, pixel/layer checks, and clean-process harness. |
| [test_release.py](../tests/test_release.py) | Offline release regressions using temporary Git repositories. |
| [release.py](../scripts/release.py) | `check`, `prepare`, `package`, `verify`, `publish`; no TD export/build/toe save. Read [RELEASING.md](RELEASING.md) before mutating release state. |

Do not equate running an updater with a read-only review: updaters alter live wiring and exported
files even when they preserve the user's active selections.

## Verification Evidence And Limits

Existing ignored reports were checked while writing this handoff:

| Evidence | Result | Limit |
|---|---|---|
| `build/model-wrappers-live.json` | Passed; all 13 loaded, detector/classifier fixtures positive, depth image, outputs/transparent-off/cooking cycle, exported demo default-off | Full run predates the latest native-matte contract/composite updates. It is not a current full-suite certification. |
| `build/mediapipe-hands-live.json` | Passed; Thumb_Up, 21 joints, 21 bones; distinct Both/Boxes/Joints; transparent overlay and unchanged image | One sample/Gesture task; not sustained-video or Hands-task switching coverage. |
| `build/mediapipe-hands-clean.json` | Passed in a relocated package and separate TD process without source Python on sys.path | Older Hands/Gesture artifact; not a fresh-process test of all current exports. |
| Embedded-import probes | Cross-import identity, separate-copy state, source/settings restoration, clean-process DAT imports passed | Import/state tests, not complete model-release tests. |
| Latest demo updater runs | Over pixel ordering, disabled frozen-layer removal, all-off transparency, stopped cook counts; default-off embedded demo re-import passed | Synthetic images and live structure checks; not simultaneous native-model performance. |
| Latest segmentation updater runs | Synthetic primary=inference, TOP 2=input, TOP 3=boxes, result-off transparency passed; both exports updated | Did not reactivate/retest real native segmentation in the latest run. |
| Release regression tests | 10 passed | Tool correctness, not legal review, binary provenance, or archive usability. |

The latest updater results were returned through `/run`; they are not all persisted as report files.
Ignored reports/fixtures might be absent on another checkout. Tests can be reproduced from the
drivers; do not rely on a report's existence as proof it matches the current artifact bytes.

Tests use official example images under ignored `build/`: `thumbs_up.jpg`, `portrait.jpg`, `pose.jpg`.
See [RELEASING.md](RELEASING.md) for URLs and invocation details. No actual extracted release ZIP
has yet been certified through the release helper's clean-project acknowledgment.

## Portability And Release Plan

1. Review/fix wrapper contracts and stale development/export behavior. Preserve useful custom
   model settings; identify where regeneration currently depends on the old reference network.
2. Validate current exports in fresh TD processes and unrelated plain projects with no repo Python
   on sys.path or registered plugin setup masking missing dependencies. Include nested placement,
   duplicates, different Package Roots, package relocation, save/reopen, and module reloads.
3. Validate sustained video, Gesture/Hands switching, resolutions/aspects, dropout handling,
   native recovery, simultaneous models, activation/deactivation, and resource/engine lifecycle.
4. Finish root discovery UX or explicitly ship a clear manual Package Root workflow. Move caches
   from package-relative locations to a per-user, content/runtime/GPU-keyed cache; test read-only
   package locations and cache invalidation.
5. Choose project license; add LICENSE/NOTICE, verify native redistribution terms, and resolve
   MoveNet source and HSEmotion license questions in [models/README.md](../models/README.md).
6. Add a reviewed model-fetch workflow with sources, licensing, integrity checks, and failure UX.
   Audit native runtime prerequisites, especially the CPU-inference versus CUDA-handling distinction.
7. Build binaries from the intended source, commit non-binary inputs, package the explicit
   [release manifest](../release-manifest.json), and test an extracted ZIP outside the checkout.
8. Only then acknowledge that exact archive/commit with `verify --tested-in-clean-project` and
   explicitly publish. No release, tag, branch, or commit was created by this work.

The last `release.py check` reported missing LICENSE and NOTICE only. Passing that existence check
would still not complete the validation/licensing/model-fetch plan. Runtime hashes identify shipped
bytes; they do not prove the binaries were built from the selected source.

## Specific Review Questions

- Is the current active/cooking/unload policy safe across re-enable, engine construction, resource
  release, and save/reopen? Which work continues after cooks stop?
- Do helper DAT caches and file-sync refreshes behave correctly inside non-cooking COMPs? A live
  sleeping wrapper retained an old helper; targeted updates used the active demo helper instead.
- Are exported model/root values, bindings, storage hints, and caches free from unintended dev or
  host references? Verify every nested COMP, especially HSEmotion's YuNet and standalone copies.
- Are exports/restoration exception-safe, and can updates fail without leaving half-wired graphs,
  changed control states, or disposable test COMPs? Are updater operations sufficiently idempotent?
- Does full regeneration preserve user edits and refresh helper code before using cached modules?
  The old reference templates remain a build-time dependency, not a runtime dependency.
- Is the output schema/UI consistent for overlays versus real images, including Debug Overlay as
  the image-result toggle and the new YOLO box output? Should raw native output and opacity/image
  selection become more explicit public controls?
- Do test assertions fail on missing detections rather than just loading? The sequential driver
  can report a loaded model without positive detections after its wait budget; inspect each result.
- Do current tests cover the exact exported artifacts, not earlier versions or source imports
  already cached in the development project? Should provenance and results be persisted per export?
- Are Windows/TD/CUDA/ONNX Runtime requirements and optional OpenCV behavior accurately stated?

Relevant TD pitfalls already encountered: mixed-family COMP output connectors require selecting
the connector by its internal output OP, not index 0; shortened menus must forward names via
`.eval()`; In/Out wires cannot cross COMP boundaries; variadic disconnections should run in reverse;
Null TOP does not perform the required mask resize (use Resolution TOP); debug sizing must reference
the original image rather than a resized inference image or HSEmotion's crop montage.

## Related Documents

- [README.md](../README.md): user-facing overview and current output contract.
- [RELEASING.md](RELEASING.md): packaging, export, test, and publish procedures.
- [TODO.md](../TODO.md): broader plan; it retains historical migration notes and uncompleted tasks.
- [CLAUDE.md](../CLAUDE.md): local build/TD working rules.
- [Learning notes](learnings/README.md): prior reasoning carried over from haxlib; old paths there
  are historical, not current runtime dependencies.