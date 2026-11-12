# Releases

`scripts/release.py` is standard-library Python and Git; publishing also needs the GitHub CLI
(`gh auth login`). It never builds plugins, exports toxes, saves the project, commits or pushes a
branch for you.

## What ships

One zip, `td-native-inference-vX.Y.Z.zip`, with a single top folder:

```
td-native-inference-vX.Y.Z/
  td-native-inference/      <- users copy this into Documents/Derivative/Plugins/
    OnnxTOP.dll  onnxtop_tool.exe  MediaPipeTOP.dll  MediaPipeCHOP.dll  TrackerDAT.dll  libmediapipe.dll
    fetch_models.cmd        (double-click: runs fetch_models.ps1, which downloads YOLO26, RVM, Kokoro and DAViD)
    models/mediapipe/...  models/asr/...  models/tts/...  models/onnx/...
  tox/                      the model toxes + VisionDemo.tox, AudioDemo.tox
  README.md  docs/MODELS.md  docs/MANUAL.md  LICENSE  NOTICE  CHANGELOG.md  VERSION  release.json
```

`release-manifest.json` lists it exactly: `files` keep their path, `map` sends a source path to a
different archive path. `release.py` refuses anything else: media, caches, TensorRT engines,
unknown binaries, model files not listed one by one, and any YOLO model. Built binaries and
models are gitignored, so they're recorded by SHA-256 in `release.json` instead of being checked
against HEAD; build the plugins from the release commit. `VERSION` is the version source of
truth: CMake compiles it into each plugin (shown in its info popup), and `package` refuses a
plugin built for another version.

## Workflow

1. **Regenerate the toxes** if anything in `wrappers/`, `python/` or `tox/shared/` changed: POST
   `scripts/build_wrappers.py` (with `DEMO = True`) to the bridge's `/run`, then
   `tests/td_wrapper_lab.py` and check `build/wrapper-lab.json`.
2. Add notes under `## Unreleased` in `CHANGELOG.md`, commit, then
   `python scripts/release.py prepare patch` (or `minor`, `major`, `X.Y.Z`). Review and commit
   `VERSION` + `CHANGELOG.md`.
3. **Build the plugins** after that commit, so they carry the new version
   (`cpp\OnnxTOP\build.cmd`, `cpp\MediaPipeTOP\build.cmd`, `cpp\MediaPipeCHOP\build.cmd`). Then
   **re-render the icons** (`python scripts/build_icons.py`; each card shows the version) and
   **rebuild the toxes** (step 1) so each tox's About page and node icon carry it too; commit.
4. `python scripts/release.py package` builds `build/td-native-inference-vX.Y.Z.zip` from the clean
   committed tree.
5. **Test the extracted zip** in a fresh TD process with the plugins installed the way users
   install them (`Documents/Derivative/Plugins/td-native-inference/`), not this checkout's
   `plugins/`. `tests/run_clean_wrapper.ps1` loads one exported tox, feeds it an image and reports
   detections. Check at least one MediaPipe and one ONNX model.
6. `python scripts/release.py verify --tested-in-clean-project` records that this exact archive
   and commit were tested.
7. Push the release commit, then `python scripts/release.py publish`: it rechecks everything,
   shows the tag/push/`gh release create` commands and asks before running them.

Published versions are immutable: fix problems with a new patch release. If publishing fails
partway, check Git/GitHub state and finish or clean up by hand; the helper never deletes tags.

Regression tests: `python -m unittest discover -s tests -p test_release.py`.
