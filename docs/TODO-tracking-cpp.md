# Plan: the tracker in C++

**Goal:** replace the Python tracking layer (`python/onnxtop_tables.py`, `object_tracker.py`,
`pose_tracker_pipeline.py`, `box_tracker_pipeline.py`, `face_tracker_pipeline.py`, ~2,100 lines)
with C++, for raw speed. The Python tracker costs ~4.5 ms per frame for one Holistic person (553
points), more than the GPU inference it follows; C++ should bring that to ~0.3-0.6 ms, most of it
TouchDesigner's table I/O rather than tracking.

**Status (Nov 12, 2026):** phases 0-5 done: every wrapper runs the Tracker DAT (the builder's default
engine) and no longer embeds the Python tracker; released in v0.4.0. Left: retiring the Python tracker
once the release has been in use for a while (see phase 5). 16 sessions recorded in TD
(`tests/td_record_tracking.py`, in `/project1/TrackingLab`); the Python replay reproduces every
recorded table exactly, and the C++ core (`cpp/Tracking`) matches it cell for cell on every
skeleton mode (mp_holistic with and without the face mesh, mp_pose, mp_hands, mp_face,
mp_facedet, pose, movenet, rtmw). Trimmed fixtures in `tests/data/tracking/`;
`tests/test_tracking_replay.py` replays them through both engines.

Phase 2 done (Nov 12): the Tracker DAT (`TrackerDAT.dll`, `cpp/Tracking/README.md`) runs MediaPipe
Holistic GPU (`"tracker_engine": "cpp"` in its spec; `wrapper_builder.build(..., tracker_engine=)`
overrides). The wrapper's tracker COMP keeps its Tracking page; its three Tracker DATs are bound to
it. Before TD has registered the custom OP (a restart after the first build) the builder falls back
to CPlusPlus DATs loading the DLL by path, which `export` refuses. `tests/td_tracker_dat_lab.py`
runs both trackers on the same Info DAT in TD (the Python ids start at the DLL's `next_track_id`):
Holistic GPU on dancer.mp4, 300 frames, every table identical. Per frame, all three tables pulled,
one person: Face Mesh on (2,456-column tracks table) Python 4.5 ms, Tracker DAT 1.3 ms (tracks 0.7,
joints 0.4, bones 0.2); Face Mesh off Python 2.7 ms, Tracker DAT 0.4 ms. Inside the plugin roughly a
third is reading the input DAT and writing cells (only changed cells are written; the output table
persists between cooks), and TD adds 0.05-0.2 ms per node. Lesson: measure with the other demos
paused; a second Holistic GPU running alongside doubled every number.

Phase 3 (Nov 12): ByteTracker (`BoxTracker.cpp`), the box and face pipelines and the solvePnP head
pose in C++; every mode now runs on the Tracker DAT. Replays exactly on detect, segment and YuNet
(geometric). Two things found on the way:
- **NMS tie order** (a deliberate change, Python too): numpy's default `argsort` orders equal
  scores by array size and CPU (SIMD sorts), and the Info DAT's 3-decimal `conf` ties in about half
  the detect / segment frames, which decided which overlapping box survived and which new track got
  the lower id. Both trackers now sort stably (ties in detection order); segment_walking was
  re-recorded.
- **solvePnP can't match exactly:** OpenCV's Levenberg-Marquardt stops on a step that is small
  relative to the whole parameter vector (dominated by the ~800 distance), and on most frames of a
  moving face its rotation vector wanders past pi. The C++ solver follows the same steps and agrees
  within 1e-5 degrees on 234 of 302 solves, within 1e-3 on 279; the rest are frames where the solve
  is chaotic in either. The replay allows yaw / pitch / roll differences on that fixture only.
- The recorder reset the tracker before the new session's image was in place, so one cook ran on
  the previous image unrecorded (yunet_photo); the reset now happens in the first step.

Phase 4 (Nov 12): the Tracker DAT's Class Scores and Convert RTMW3D tables and its track channels
replace `write_scores`, `rtmw_convert` and `write_channels` (C++ engine). In TD
(`tests/td_extras_lab.py`): Class Scores and Convert RTMW3D identical to the Python every frame;
channels match `write_channels` on every recorded table offline (`tests/test_tracking_extras.py`)
and stay current in a C++ wrapper where only `channels` is pulled. Two TD lessons: an Info CHOP
doesn't cook the node it reads (a Script CHOP that reads the node first does), and an expression on
a cook time is cached until a dependency changes (`absTime.frame` keeps it live). Comparing the
Python wrapper's own channels live doesn't work: `write_channels` force-cooks `inference_stats`,
which recooks the plugin mid-frame. Holistic GPU's channels: Python 6-11 ms, C++ 0.3 ms.
`write_instance_colors` stays Python (a multi-sample lookup CHOP, one value per object).
The Performance page: plugin timings (`decode_ms` and running averages added to OnnxTOP and
MediaPipeTOP), the Tracker DAT's, and TD's cook times for the Info DAT, overlay and the tox total.

Kalman exactness: the filter's matrices are 0s and 1s and the dimensions independent, so products
round the same in any order; numpy's solve multiplies by the reciprocal of the (diagonal) Cholesky
factor, checked bit for bit in TD's numpy 2.1 and 2.5. NMS is float32 like numpy's.

In TD (`tests/td_tracker_dat_lab.py`, 300 frames each, every table identical, ids included), per
frame, all tables pulled: Holistic GPU 4.27 -> 1.36 ms, without the face mesh 1.48 -> 0.50, YOLO26
Detect 3.71 -> 0.29, YOLO26 Seg 5.22 -> 0.42, YuNet 1.51 -> 0.22.

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
| 0 | Fixture recorder, Python replay harness, fix the test file's mid-file `unittest.main()` | done |
| 1 | Core: KeypointTracker + pose pipeline + skeletons + formatting; replay tool; exact diff on every skeleton mode | done |
| 2 | `TrackerDAT` plugin, wrapper builder option, Holistic GPU on it, measured | done |
| 3 | ByteTracker + Kalman + Hungarian, box pipeline (detect / segment), face pipeline (YuNet, geometric and solvePnP) | done |
| 4 | The rest of the wrapper's per-frame Python: `write_channels`, `write_scores`, `rtmw_convert` (`write_instance_colors` stays Python); the Performance page | done |
| 5 | Switch every wrapper to TrackerDAT, regenerate toxes and demos, release; then retire the Python tracker modules (fixtures switch to recorded C++ output) | 1 session |
| 6 | The puppet solvers (`puppet_solver`, `mixamo_retarget`, `hand_retarget`) in C++: design in [PUPPETS.md](PUPPETS.md#the-c-solver-phase-6-of-todo-tracking-cppmd); `cpp/Puppet`, matched offline and in TD (Nov 13) | tox switch after a TD restart |

Later, separately: hand the decoded detections from the TOP to the tracker in binary (in-process
registry) to skip the Info DAT strings entirely.

## Everything else in Python, per frame (inventory, Nov 12)

Measured in TD, one person, on this machine. Per-frame work that grows with keypoints or tracks
moves to C++; event-driven or once-per-build code stays Python.

| Code | Where | Cost | Plan |
|---|---|---|---|
| Box tracker (`box_tracker_pipeline`) | Detect / Seg wrappers, and the nested detector in Holistic GPU, RTMW3D, HSEmotion | 0.16-0.22 ms every frame even for Holistic GPU's mostly idle detector | Phase 3: ported; wrappers switch in phase 5 |
| Face tracker (`face_tracker_pipeline`) | YuNet, HSEmotion's detector | not measured (one row per face, plus head pose) | Phase 3: ported; wrappers switch in phase 5 |
| `write_channels` (`wrapper_runtime`) | Every tracking wrapper's `out_channels` CHOP: `t{id}_{column}` per numeric cell | **7.3 ms** for Holistic with the face mesh (2,481 channels), 1.8 ms without (569); only when something reads `out_channels` | Phase 4: a C++ CHOP reading the tracks table |
| `write_scores` (`onnxtop_tables`) | HSEmotion: class scores joined onto the face tracks | small (a row per face) | Phase 4, in the Tracker DAT (a Scores table mode) |
| `rtmw_convert` | RTMW3D: reshapes the Pose Crops Info DAT for the tracker | not measured (75 points a person, a full table pass in Python) | Phase 4: the Tracker DAT reads RTMW3D's layout directly, the Script DAT goes |
| `write_instance_colors` (`wrapper_runtime`) | Seg wrappers, Matte Encoding = Instances: one color per object | small | Phase 4, with the box tracker's outputs |
| Puppet solver (`puppet_solver` + `mixamo_retarget`) | `MixamoPuppet.tox` | **2.7 ms** per body solve | Phase 6: a C++ CHOP; capturing the rig from the FBX COMP stays Python (once per character) and hands the rest pose over as a table |
| Hand puppet (`hand_retarget`) | the PuppetLab's rigged hand (no tox yet) | small per solve | Phase 6 |
| Stays Python | `wrapper_builder`, `puppet_builder` (build time), `wrapper_runtime` model controls and pulses, `puppet_runtime` (character loading), the demos' `model_controls`, Holistic GPU's `detector_gate` (one comparison per frame), the audio wrappers' caption / event / top-results scripts (per new line of text, a few rows) | - | - |

Not Python, but now the largest per-frame costs after inference in the Holistic GPU wrapper: the
debug overlay (DebugBB's Render TOP ~1.3 ms, a Select DAT 0.26 ms), drawn only with Debug
Overlay on. Worth a look separately.

## Decisions (Nov 12, 2026)

- **A new DLL:** yes, `TrackerDAT.dll` (a fourth Allow prompt on first load).
- **Port today's behavior first;** fixing things along the way is fine. Each fix is a deliberate,
  recorded difference in the replay test (an expected-diff note), not a silent one.
- **Python is the test oracle only until the port is proven,** not a long-term reference: once
  every wrapper runs TrackerDAT (phase 5) and the replay tests pass, the recorded C++ tables become
  the fixtures and the Python tracker modules can be removed.
- **More plugins:** each custom operator is its own DLL (one more Allow prompt each). Phase 4's
  channels CHOP and phase 6's puppet CHOP add two, unless they can share one CHOP DLL (to check:
  a CHOP DLL with a mode menu, like MediaPipeCHOP's Task).
- **Face Mesh toggle:** done in Python (Nov 12): `Facemesh` on the `mp_holistic` tracker
  (`MP_HOLISTIC_BODY`, 75 points); Holistic GPU ties it to the plugin's Face Mesh. 4.7 -> 1.2 ms in
  TD for one person. The C++ op needs the same parameter.
