# Tracking: the tracker in C++

The port of the Python tracker (`python/onnxtop_tables.py` and the pipelines) to C++, and the
**Tracker DAT** custom operator that runs it in TouchDesigner. Plan and status:
`docs/TODO-tracking-cpp.md`.

- `src/` (no TouchDesigner dependency except `TrackerDAT.*`): `KeypointTracker` (greedy matching,
  coasting, duplicate suppression), `PoseTracker` (the pose pipeline: smoothing, hide-missing,
  presence, gesture labels, the tracks / joints / bones tables), `BoxTracker` (ByteTrack: Kalman
  filter, two-stage IoU matching with the Hungarian algorithm, float32 NMS), `BoxPipeline` (YOLO
  detect / segment and YuNet faces: smoothing, presence, head pose; `HeadPose.cpp` is the solvePnP
  solve), `Tracker` (Info DAT reading, the per-mode defaults, result skipping), `Skeletons.gen.h` (from `tools/gen_skeletons.py`, which reads
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
wire, in the same DLL). Two more tables replace wrapper Script DATs:

- **Class Scores** (`onnxtop_tables.write_scores`, HSEmotion): input 1 the tracks, input 2 a Classify
  Crops Info DAT; adds `<prefix>_label` and `<prefix>_<class>` (smoothed in float32, like numpy)
  before `r, g, b`, and names the class in the label. Class Column Prefix, Class Score Smoothing.
- **Convert RTMW3D** (`rtmw_convert`): RTMW3D's Pose Crops Info DAT reshaped into the Holistic-like
  table the `rtmw` mode reads, with metric world landmarks. Frame TOP: the source image.

The Tracks and Class Scores nodes also publish `t{id}_{column}` for every numeric cell on their Info
CHOP (`wrapper_runtime.write_channels`), built only when an Info CHOP asks. An Info CHOP doesn't cook
the node it reads, so the wrappers copy them with a Script CHOP that reads the node first
(`wrapper_builder.track_channels`).

Modes: every Tracker Mode of the Python tracker (`pose`, `movenet`, `face`, `detect`, `segment`,
`mp_pose`, `mp_hands`, `mp_face`, `mp_holistic`, `mp_facedet`, `rtmw`). The tables are identical to
the Python tracker's, except YuNet's yaw / pitch / roll with Head Pose Method = SolvePnP: the solve
follows OpenCV's steps but not its rounding, and on badly conditioned frames (5 near-planar points)
either one can wander to a different rotation.

| Parameter | |
|---|---|
| Table | Tracks / Joints / Bones |
| Tracker Mode | The model's skeleton; a mismatch with the Info DAT writes an `error` row |
| Frame TOP | The source image (Face with SolvePnP only: the camera model's frame size; else 1000 x 1000) |
| Results CHOP | The plugin's Info CHOP: with MediaPipeTOP's `results` / `result_delay`, the tracker advances once per new result. Empty: every cook |
| Tracking page | Every mode's parameters, with the Python tracker's names: Confidence Threshold, Output Smoothing, Track Loss Frames, Scale Match To Skeleton, Relative / Max Match Distance, Track Confirm Frames, Min Box Width / Height, Coast While Lost, Presence Rise / Fall Frames, Gesture Hold Frames, Gesture Min Score, Face Mesh; Low Confidence Threshold, NMS IoU Threshold, Min Mask Fill Ratio, Track IoU Threshold; Head Pose Method, Yaw / Pitch / Focal Scale, Head Pose Smoothing; and the C++-only joint gate (body modes): Joint Confidence Threshold, Hide Off-Image Joints (`PoseTracker::gateJoints`: under the threshold, outside or on the image's edge, or a gated parent -> conf 0, not in joints / bones; matching unaffected; off in the Python replays, tested by `tests/test_joint_gate.py`) |

The parameters carry the Pose mode's defaults. A wrapper binds each to its tracker COMP, which
builds the per-mode Tracking page (`wrapper_builder`, engine `cpp`).

Info CHOP: `cook_ms` (the whole cook), `track_ms` (before writing the table), `input_ms`
(getting the input DATs), `write_ms` (setting the output cells), `num_tracks`, `next_track_id` /
`next_box_track_id` (the process-wide id counters), `cook_ms_avg` and `tracker_ms_avg` (Tracks
nodes: their own average plus their Joints / Bones nodes', for the Performance page), then the track
values.

Track ids come from two counters per process (skeleton modes, box and face modes), shared by every
Tracker DAT, as in Python. Only cells
that changed since the last cook are written (the output table persists between cooks).
