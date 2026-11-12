# Plan: the tracker in C++

**Goal:** replace the Python tracking layer (`python/onnxtop_tables.py`, `object_tracker.py`,
`pose_tracker_pipeline.py`, `box_tracker_pipeline.py`, `face_tracker_pipeline.py`, ~2,100 lines)
with C++, for raw speed. The Python tracker costs ~4.5 ms per frame for one Holistic person (553
points), more than the GPU inference it follows; C++ should bring that to ~0.3-0.6 ms, most of it
TouchDesigner's table I/O rather than tracking.

**Status:** planned (Nov 12, 2026), not started. Behavioral inventory of the current Python tracker
was taken the same day; its findings are folded in below.

## Design

**A new custom operator, `TrackerDAT` (CPlusPlus DAT, its own `TrackerDAT.dll` in `plugins/`),
replacing the tracker COMP's three Script DATs one-for-one.**

- Inside each wrapper's `tracker` COMP today: `in_info` -> `script_tracks` -> `script_joints`,
  `script_bones`. Those three become three `TrackerDAT` nodes with **Table** = Tracks / Joints /
  Bones. The Tracks node runs the tracker and keeps the state; Joints and Bones read its snapshot
  from an in-process registry (same DLL), keyed by the Tracks node, so the work happens once.
  Wiring, cook order, Out DATs and every consumer stay as they are.
- **Input:** the plugin's Info DAT, read in C++ through `OP_DATInput` (cell strings -> floats, about
  0.05 ms for 2,450 cells; no Python parse). Works the same for OnnxTOP and MediaPipeTOP, which
  write the same layout, and for RTMW3D's converted table.
- **Why not inside the plugins:** a TOP has one Info DAT, the tracker produces three tables; and a
  tracker op serves both plugins without touching their code.
- **Why a DAT, not a CHOP:** consumers (DebugBB, DebugKeypoints, puppet_solver, HandsGeometry,
  HSEmotion's crops, the user's own networks) read tables by column name. CHOP output can come later.
- **Core library** `cpp/Tracking/` (no TouchDesigner dependency): Kalman + ByteTracker + Hungarian,
  KeypointTracker, the pose / box / face pipelines, skeleton tables, table formatting. The DAT plugin
  and a command-line replay tool both link it.
- **Parameters:** the same names as the tracker COMP's (Trackermode, Confthreshold,
  Outputsmoothing, Tracklossframes, ...), with the same per-mode defaults, so the wrappers' exposed
  Tracking pages (`Track<name>` expressions) keep working. Per-mode defaults and the Tracking page
  rebuild currently live in `setup_parameters`; the C++ op declares the union and enables per mode.

## Behavior to reproduce exactly (from the inventory)

The port is judged by identical output tables on recorded input, so these matter:

- **Result skipping:** update only on a new result when `result_delay > 0` and a `results` channel
  exists (MediaPipeTOP); otherwise every cook, including re-applying a still image's result (needed:
  confirmation and smoothing converge over cooks). OnnxTOP has no `results` channel, so its trackers
  never skip; keep that, or add the channel to OnnxTOP and skip there too (a separate, deliberate change).
- **Cadence:** Kalman dt, `lost_frames`, presence, coasting all step per applied result, not per frame.
- **ByteTracker** (detect / segment / YuNet faces): the Kalman constants, two-stage IoU association,
  the Hungarian variant (padded square matrix, threshold caps), `_cap_detections`, sticky
  confirmation, no score decay, output order = track list order (ties depend on it).
- **KeypointTracker** (all skeleton modes): no Kalman; greedy-global matching sorted by distance
  (stable), absolute or box-relative distance on stable joints, coasting prediction, score decay 0.9
  per lost frame, duplicate suppression at half the threshold, 50-track cap.
- **Pose pipeline:** display gate, per-joint EMA, `hide_missing` (absent joints hold, re-appear
  snapped, hidden for 3 frames while settling), track re-acquire snap after 2+ lost frames, world
  smoothing rules, coasting of keypoints and box, gesture debounce (mp_hands), presence ramp.
- **Box pipeline:** NMS (greedy, float32), min mask area, presence ramp, `det_index` (-1 when lost).
- **Face pipeline (YuNet):** geometric head pose (default); solvePnP mode needs a small 6-DOF
  Levenberg-Marquardt solver in C++ (or keep that one mode on Python; decide in phase 3).
- **Read conventions:** TD UV, world axis flip `[x, -y, -z]` at read time, blendshape extras,
  `handedness` from the class column (mp_hands), error rows `['error'], [msg]` on a mode mismatch.
- **Output formats:** exact column order and number formats (`.3f` / `.4f` / `.5f`), `-0.0000`
  possible, label cells `{#color(R, G, B);}` with Python's round-half-to-even in both the percentage
  and the color, `track_color` golden-ratio hue. `puppet_solver` skips a solve when the row text is
  unchanged, so formatting is behavior, not cosmetics.
- **Ids:** today one process-wide counter for box/face tracks and one for pose tracks (two wrappers
  never reuse an id). Keep it (a static in the DLL).

Known dead code not to port: `track_fade`, `td_to_px`, `freeze_velocity_on_loss`, the mask hold in
the box pipeline, face `size_aspect`, `keypoints_visible` (always true), legacy constants.

## Proving it's the same

1. **Recorded fixtures:** a small TD script records each wrapper's Info DAT text per applied result
   (plus `result_delay` / `results`), on the clips in `media/` and the photos: dancer (Holistic GPU,
   Holistic, Pose, RTMW), people-walking (YOLO detect / seg / pose, MoveNet), hands (Hands, Gesture),
   a face clip (YuNet, Face, Face Detector). A few hundred frames each, gitignored except a small
   set committed under `tests/data/tracking/`.
2. **Replay tool** `tracker_replay.exe` (C++, links the core) and a Python replay harness (the
   existing pipelines, TD objects faked as `tests/test_tracking.py` already does) run the same
   fixture and write the tables per frame. A test diffs them cell by cell: exact text match for
   ids, labels and formatted numbers; any difference is a bug in the port (or a documented,
   deliberate change).
3. The existing `tests/test_tracking.py` cases become fixtures too. Its `unittest.main()` sits
   mid-file, so running it directly misses half the tests; fix that first (discover mode works).
4. **In TD:** the wrapper lab test, the clean-process test, and a side-by-side in the lab (Python
   tracker and TrackerDAT on the same Info DAT, tables diffed live).

## Phases

| Phase | What | Size |
|---|---|---|
| 0 | Fixture recorder, Python replay harness, fix the test file's mid-file `unittest.main()` | 1 session |
| 1 | Core: KeypointTracker + pose pipeline + skeletons + formatting; replay tool; exact diff on every skeleton mode (mp_holistic, mp_pose, mp_hands, mp_face, mp_facedet, pose, movenet, rtmw) | 2 sessions |
| 2 | `TrackerDAT` plugin (build, Plugins.json, release manifest), wrapper builder option to use it, Holistic GPU first; measure; Python stays the default until proven | 1 session |
| 3 | ByteTracker + Kalman + Hungarian, box pipeline (detect / segment), face pipeline (YuNet, geometric; solvePnP decision) | 2 sessions |
| 4 | `write_scores` (HSEmotion class scores) and `write_channels` (the per-track CHOP, thousands of channels for Holistic) in C++ | 1 session |
| 5 | Switch every wrapper to TrackerDAT, regenerate toxes and demos, release; keep the Python modules as the reference implementation for prototyping and for the replay tests | 1 session |

Later, separately: hand the decoded detections from the TOP to the tracker in binary (in-process
registry) to skip the Info DAT strings entirely; CHOP outputs for tracks.

## Decisions for the user

- **A new DLL:** `TrackerDAT.dll` means a fourth Allow prompt for users on first load.
- **Exact first, improvements after:** the port reproduces today's behavior, quirks included (e.g.
  OnnxTOP trackers never skip repeated results; ids shared across wrappers). Improvements land as
  separate, visible changes once the port is proven.
- **Python stays as the reference:** prototype tracking ideas in Python, then port. The replay
  tests keep the two honest.
- **Quick win meanwhile (optional):** a "Face Mesh" toggle on the Holistic tracker to skip the 478
  face points when only body and hands are used (the puppet): a large cut in Python cost now,
  and the same option in C++ later.
