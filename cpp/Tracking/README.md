# Tracking: the tracker in C++

The port of the Python tracker (`python/onnxtop_tables.py` and the pipelines) to C++, and the
**Tracker DAT** custom operator that runs it in TouchDesigner. Plan and status:
`docs/TODO-tracking-cpp.md`.

- `src/` (no TouchDesigner dependency except `TrackerDAT.*`): `KeypointTracker` (greedy matching,
  coasting, duplicate suppression), `PoseTracker` (the pose pipeline: smoothing, hide-missing,
  presence, gesture labels, the tracks / joints / bones tables), `Tracker` (Info DAT reading, the
  per-mode defaults, result skipping), `Skeletons.gen.h` (from `tools/gen_skeletons.py`, which reads
  `python/mediapipe_skeletons.py`), `Common.h` (number parsing and formatting that match Python's
  text exactly).
- `TrackerDAT.dll`: the custom DAT, copied into `plugins/`.
- `tracker_replay.dll`: a C API for `tests/tracking_replay.py`, which replays recorded Info DATs and
  diffs the tables against the Python tracker's.

Build: `build.cmd` (MSVC v142, Ninja; no CUDA). TouchDesigner loads `TrackerDAT.dll` at startup
to register the operator and keeps it locked, so a rebuild takes effect after a restart (the build
moves the locked file aside).

## Tracker DAT

Three nodes per tracker, replacing the tracker COMP's Script DATs: **Table** = Tracks reads the
Info DAT wired into it, advances the tracker and writes the tracks table; Joints and Bones are
wired after it and write the same result's joints and bones (they find its tracker through the
wire, in the same DLL).

Modes so far: the skeleton modes (`pose`, `movenet`, `mp_pose`, `mp_hands`, `mp_face`,
`mp_holistic`, `mp_facedet`, `rtmw`). Box and YuNet face modes are still Python.

| Parameter | |
|---|---|
| Table | Tracks / Joints / Bones |
| Tracker Mode | The model's skeleton; a mismatch with the Info DAT writes an `error` row |
| Results CHOP | The plugin's Info CHOP: with MediaPipeTOP's `results` / `result_delay`, the tracker advances once per new result. Empty: every cook |
| Tracking page | Confidence Threshold, Output Smoothing, Track Loss Frames, Scale Match To Skeleton, Relative / Max Match Distance, Track Confirm Frames, Min Box Width / Height, Coast While Lost, Presence Rise / Fall Frames, Gesture Hold Frames, Gesture Min Score, Face Mesh (the Python tracker's names) |

The parameters carry the Pose mode's defaults. A wrapper binds each to its tracker COMP, which
builds the per-mode Tracking page (`wrapper_builder`, engine `cpp`).

Info CHOP: `cook_ms` (the whole cook), `track_ms` (without writing the table), `input_ms`
(getting the input DAT), `num_tracks`, `next_track_id` (the process-wide id counter).

Track ids come from one counter per process, shared by every Tracker DAT, as in Python. Only cells
that changed since the last cook are written (the output table persists between cooks).
