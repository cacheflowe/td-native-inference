# Plan: the tracker in C++

**Goal:** replace the Python tracking layer (`python/onnxtop_tables.py`, `object_tracker.py`,
`pose_tracker_pipeline.py`, `box_tracker_pipeline.py`, `face_tracker_pipeline.py`, ~2,100 lines)
with C++, for raw speed. The Python tracker costs ~4.5 ms per frame for one Holistic person (553
points), more than the GPU inference it follows; C++ should bring that to ~0.3-0.6 ms, most of it
TouchDesigner's table I/O rather than tracking.

**Status (Nov 12, 2026):** phases 0 and 1 done. 16 sessions recorded in TD
(`tests/td_record_tracking.py`, in `/project1/TrackingLab`); the Python replay reproduces every
recorded table exactly, and the C++ core (`cpp/Tracking`) matches it cell for cell on every
skeleton mode (mp_holistic with and without the face mesh, mp_pose, mp_hands, mp_face,
mp_facedet, pose, movenet, rtmw). Trimmed fixtures in `tests/data/tracking/`;
`tests/test_tracking_replay.py` replays them through both engines. Next: phase 2 (the TD plugin).

Run: `cpp\Tracking\build.cmd`, then `python tests/tracking_replay.py build/tracking/*.jsonl.gz
--engine cpp` (full recordings) or `python -m unittest tests.test_tracking_replay` (committed ones).
Things that had to match Python exactly: `math.hypot` (CPython's exact-sum algorithm, not the C
library's), round-half-even, stable sort of match pairs, correctly rounded parse/format
(`from_chars` / `to_chars`), and no FMA (`/fp:precise`).

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

1. **Recorded fixtures:** `tests/td_tracking_recorder.py` records, per frame, each wrapper's Info
   DAT text, `results` / `result_delay`, how many times each tracker script cooked, and the tables
   it wrote, on the clips in `media/` and the photos (faces: `face.jpg` moved by a Transform TOP,
   there's no face clip). Full recordings in `build/tracking/` (gitignored, ~20 MB), trimmed copies
   in `tests/data/tracking/`.
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
| 5 | Switch every wrapper to TrackerDAT, regenerate toxes and demos, release; then retire the Python tracker modules (fixtures switch to recorded C++ output) | 1 session |

Later, separately: hand the decoded detections from the TOP to the tracker in binary (in-process
registry) to skip the Info DAT strings entirely; CHOP outputs for tracks.

## Decisions (Nov 12, 2026)

- **A new DLL:** yes, `TrackerDAT.dll` (a fourth Allow prompt on first load).
- **Port today's behavior first;** fixing things along the way is fine. Each fix is a deliberate,
  recorded difference in the replay test (an expected-diff note), not a silent one.
- **Python is the test oracle only until the port is proven,** not a long-term reference: once
  every wrapper runs TrackerDAT (phase 5) and the replay tests pass, the recorded C++ tables become
  the fixtures and the Python tracker modules can be removed.
- **Face Mesh toggle:** done in Python (Nov 12): `Facemesh` on the `mp_holistic` tracker
  (`MP_HOLISTIC_BODY`, 75 points); Holistic GPU ties it to the plugin's Face Mesh. 4.7 -> 1.2 ms in
  TD for one person. The C++ op needs the same parameter.
