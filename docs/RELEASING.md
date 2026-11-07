# Releases

The release helper is standard-library Python and Git. Publishing also requires the GitHub CLI
(`gh auth login`). It never builds plugins, exports live TD components, saves a project, commits,
or pushes a branch for you. No live bridge is required, so it cannot export from the wrong TD app.

## Release Contents

`release-manifest.json` is the explicit list of required inputs. Missing files block preparation
and packaging. The current manifest includes all 13 model toxes and `tox/ModelDemo.tox`, alongside
the four native runtime files and documentation. Inclusion is not release certification: live tests
do not replace testing an extracted ZIP or the license/model-fetch gates. For implementation status
and review questions, see [REVIEW_HANDOFF.md](REVIEW_HANDOFF.md).

Every ZIP has one `td-native-inference-vX.Y.Z/` directory containing the package. Only listed files
are included. Models, private test media, demo projects, caches, unknown binaries and plugin
allowlists are excluded. This first release is intentionally model-free. A reviewed model download
workflow is still required; an extracted archive is not yet a complete out-of-box install.

`LICENSE` and `NOTICE` must exist and contain the chosen license and required third-party notices.
Their presence is a prerequisite, not an automated legal review. Check binary redistribution terms,
including libmediapipe and its dependencies, before publishing. TD's runtime DLLs are not bundled.

`VERSION` is the package version source of truth; it does not yet update C++ plugin metadata.
`release.json` inside the ZIP records the source commit and SHA-256 of every included file.
Local plugin hashes identify the shipped binaries but do not prove which source built them.
Build both plugins from the intended release source before packaging.

## Workflow

1. Finish portable tox packaging and licensing. Run the read-only prerequisite check:
   `python scripts/release.py check`.
2. Update `CHANGELOG.md` under `## Unreleased`. Commit the work, including the release helper,
   manifest and all non-binary package inputs. Run
   `python scripts/release.py prepare patch` (or `minor`, `major`, `X.Y.Z`). For the first release,
   the current `0.1.0` is allowed. Preparation refuses dirty tracked files and missing prerequisites
   before changing VERSION or the changelog.
3. Review and commit VERSION and CHANGELOG.md. Build both plugins from this source, then run
   `python scripts/release.py package`. Packaging requires a clean committed tree and checks that
   all non-binary inputs are tracked and match HEAD. Git CRLF/LF checkout conversion is permitted.
   Ignored runtime binaries are included only by their four explicit manifest paths.
4. Extract `build/td-native-inference-vX.Y.Z.zip` somewhere outside this checkout. Test in a fresh
   TD project, without this repository on sys.path or registered plugins masking missing setup.
   Verify plugin loading, model resolution, tracker outputs and overlays. Move the package folder
   and repeat. Check errors and actual detections on suitable video, not just empty table schemas.
5. Run `python scripts/release.py verify --tested-in-clean-project` only after that manual test.
   It checks archive contents, hashes, version, current source commit, local inputs and notes, then
   records an acknowledgment tied to this exact archive and commit. Plain `verify` checks integrity
   without acknowledging a test. Neither command performs the TD test automatically.
6. Push the release commit to your branch. Run `python scripts/release.py publish`. It rechecks
   the artifact and test acknowledgment, refuses existing local/remote tags, prints commands and
   asks before creating the annotated tag, pushing it to origin and creating the GitHub release.
   `--yes` explicitly skips confirmation. The ZIP and its checksum are attached; changelog notes
   supply the release text. Nothing is published by prepare, package or verify.

Published versions are immutable. Prefer a new patch release to replacing an artifact. If publishing
fails after tagging or pushing, inspect Git/GitHub state and complete or clean up that attempted
release manually; the helper deliberately does not delete tags or overwrite releases on retry.

Run offline regression tests with `python -m unittest discover -s tests -p test_release.py`.

## Planned Single-Folder Install

This is the target installation layout. Embedded wrappers are exported, but complete out-of-box
installation, automatic root discovery, and full clean-project validation remain unfinished:

```text
existing-project/
   existing-project.toe
   ...the user's existing files...
   td-native-inference-v0.1.0/
      tox/MediaPipeHands.tox
      plugins/OnnxTOP.dll
      plugins/onnxtop_tool.exe
      plugins/MediaPipeTOP.dll
      plugins/libmediapipe.dll
      models/mediapipe/...
      models/onnx/...
      VERSION
      LICENSE
      NOTICE
```

The intended workflow is extract one folder, fetch the desired models into that folder, then drag
the desired model tox into any network. Nothing needs to be copied into the host project's own
Plugins, models or python directories. No host startup DAT, global OP shortcut or configured Python
search path should be required. Dependencies supplied by TD and Windows remain prerequisites.

Each public model COMP should own one Package Root. Resolve external plugin and model references
from that root, not from project.folder. Given a tox loaded from `<root>/tox/`, the
default root should be derived from its external tox location; retain a user override for embedded
copies or missing source locations. Verify what TD preserves on drag/drop, duplicate, save/reopen
and nested placement before choosing the final expression. TD's externaltox docs confirm reload on
project startup and embedded-content fallback when the external file is absent; that fallback does
not make external DLLs and models portable on its own.

Use a generic CPlusPlus TOP inside each wrapper, with its Plugin Path under Package Root. That avoids
relying on a custom OP registered from the host project's Plugins folder. Models get package-relative
defaults plus a per-model override. Missing files should identify the expected path clearly.

Decided: embed the Python, tracker and debug visualizers inside every release model tox. In the dev
project, the Python Text DATs remain synced to the single source of truth in `python/`, and shared
visualizers remain linked to `tox/shared/`. Release users need neither of those external folders.
Small duplication between exported model toxes is intentional: each tox should be self-contained
apart from its native binaries and models.

Reuse td-http-api's export pattern: disable DAT sync and startup loading before clearing file paths,
clear development external-tox references, save the embedded release tox, then restore development
settings in a finally block. Extend the traversal to nested trackers and visualizers, preserving
parameter expressions/modes and instance configuration. The first wrapper implements this in
`python/mediapipe_hands_component.py`; release.py only packages already-exported toxes.

### Model Workbench Development

Run `scripts/build_model_wrappers.py` through `/run` twice: native registration, then model
configuration/export. It builds 13 model wrappers in `/project1/ModelDemo`, exports
each to `tox/`, and exports the embedded default-off `tox/ModelDemo.tox`. An existing completed
workbench can be refreshed/exported by rerunning the script with every Models-page toggle off.
The first replaced MediaPipeHands artifact is backed up under ignored
`build/before-model-wrappers/`. It does not replace the original live MediaPipeHands COMP or save
the toe. Do not edit the Models toggles while automated validation is running.

For just the demo's combined debug output, run `scripts/update_model_demo.py` through `/run`.
It preserves live model controls/input, validates Over ordering and disabled-layer transparency,
and exports a default-off embedded ModelDemo. TOP 1 is the active debug composite; TOP 2 is the
original image. The moved demo has no operator references back to `ONNX_Native`, but keep that
network as the builder's source templates for now. Older nested demo locations are also detected.

YOLO segmentation's TOP 1 is its native inference matte, TOP 2 the source image, TOP 3 boxes;
tracking DAT/CHOP outputs remain available. `scripts/update_yolo_seg_output.py` applies this
contract without rebuilding models, runs focused pixel checks, and exports Yolo26Seg/ModelDemo
while preserving live controls. The demo treats YOLO/MediaPipe segmentation and depth as image
layers behind box/joint overlays.

The workbench's Package Root feeds every child. Model enables are bound to parent controls;
a parent-owned Parameter Execute DAT controls child `allowCooking`. This placement is required:
an internal callback cannot wake its own disabled COMP. Native Active alone is insufficient to
prevent ONNX model loading. Create/start callbacks restore the external controller after import.
Standalone model toxes are exported inactive, without the workbench binding or cooking gate.

Run `tests/td_model_wrappers.py` through `/run` for sequential live tests. It requires the official
MediaPipe images under ignored `build/`: `thumbs_up.jpg` (URL below), and `portrait.jpg` / `pose.jpg`
from `https://storage.googleapis.com/mediapipe-assets/`. It replaces/restores only the workbench's
input, checks embedded exports and default-off imported demo state, then enables one model at a
time, checks loading/data/images, and verifies native cook counts stop when disabled. It leaves
every model off and writes `build/model-wrappers-live.json`. The temporary source is a sibling of
the demo because TD wires cannot cross COMP boundaries. Check the report's per-model detections;
raw depth has no detections. This does not replace clean-project testing of the release archive.

### Original Hands Builder

Run `scripts/build_mediapipe_hands.py` through the development bridge's `/run` route twice: the first
call constructs an isolated network, the second configures the loaded native DLL and exports
`tox/MediaPipeHands.tox`. It refuses to overwrite an existing wrapper or tox. The resulting
`/project1/ONNX_Native/MediaPipeHands` keeps Python linked to the source files and visualizers linked
to the shared tox files. The script does not save the development project.

To re-export an edited development wrapper inside TD:

```python
comp = op('/project1/ONNX_Native/MediaPipeHands')
comp.op('component').module.export(comp, project.folder + '/tox/MediaPipeHands.tox')
```

Run `tests/td_mediapipe_hands.py` through `/run` for the integration test. First download the official
MediaPipe example image to ignored `build/thumbs_up.jpg`:
`https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg`.
The test checks export restoration and unchanged source bytes, then uses a disposable imported tox
to check native readiness, actual `Thumb_Up` recognition, embedded imports, tracks/joints/bones,
numeric CHOP data, and visible overlay pixels on a transparent background. Both/Bounding Boxes/Joints
layer selections must match their sources and produce distinct images. Debug-off must produce
transparent black, and toggling debug must not change the separate image output. The report is
`build/mediapipe-hands-live.json`.
The test network is removed afterward; the helper also writes an ignored clean-process harness.

Nov 6 validation: a minimal package was copied outside this checkout, moved, and loaded in a new
TD 2025.33230 process. `build/mediapipe-hands-clean.json` records one hand, `Thumb_Up`, 21 joints,
21 bones, 1403 changed overlay pixels and no source modules on sys.path. Only that new test process
was closed. This tested the Gesture task; Hands task switching and sustained video still need tests.

Important: `COMP.filePath` reports the host toe after `loadTox()`, not the original import path.
The helper uses an external-tox reference when available, but programmatic imports currently need
an explicit Package Root. Both integration runs used that override. Automatic UI drag/drop discovery,
development reloads, project save/reopen and side-by-side full native wrappers remain unverified.

The six-module `local/modules` layout passed a disposable probe on Nov 6, 2026 in TD 2025.33230:
all module/dependency identities resolved to DATs, synthetic tracks retained their IDs, and two
exported copies with distinct module markers kept independent module/pipeline state. Export removed
file sync, startup loading and source paths; development expressions/settings and source bytes were
unchanged afterward. A separate TD process loaded the exported tox successfully with no source
directory on sys.path and no cached source modules. Reports are in ignored
`build/embedded-imports-live.json` and `build/embedded-imports-clean/result-*.json`.

This establishes the import approach, not a complete model release. The first wrapper now also
passes real tracker-output and visualizer checks; development reloads remain untested. TD DAT modules do
not support ordinary filesystem package organization. No global sys.path changes or host startup
DAT should be needed. The original reference tracker networks remain unchanged.

Edit shared visualizers at their development source, never arbitrary instances. Embed those approved
sources into release toxes rather than shipping separate shared tox files.
TensorRT caches should ultimately use a per-user cache keyed by model content, runtime and GPU, not
write into the package or host project by default. This also supports read-only installations.

The release manifest omits development Python and shared tox files and requires the full exported
model/demo set. Embedded export is implemented; validate the current artifacts, add the reviewed
model-fetch entry point, then test the extracted folder in a plain TD project. No relocation or
Python-loader changes are made by release.py.