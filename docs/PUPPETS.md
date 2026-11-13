# Puppets: how the solver drives a rigged character

Developer notes on `MixamoPuppet.tox` and the rigged-hand lab: how tracked landmarks become bone
rotations on a skinned FBX character, the conventions that make it work, where the time goes, and
the plan for the C++ solver. User-facing setup is in [MANUAL.md](MANUAL.md#puppets).

Code: `python/mixamo_retarget.py` (the solver: rig, targets, solve, place), `python/hand_retarget.py`
(the hand rig's bone map), `python/puppet_solver.py` (the Script CHOP: tracks table in, channels
out), `python/puppet_runtime.py` (inside the tox: import the FBX, bind the bones),
`python/puppet_builder.py` (builds the tox). Tests: `tests/test_retarget.py`,
`tests/test_hand_retarget.py`, `tests/test_puppet_solver.py`.

## The pipeline

```
video -> Holistic / Pose tox (World Landmark Columns on)
           out_tracks: one row per person, <joint>_wx/_wy/_wz (metres) and <joint>_x/_y/_conf (image)
              |
              v
MixamoPuppet.tox
  in_tracks -> solver (Script CHOP, puppet_solver.cook)
                 1. read the person's row         -> {landmark: [x, y, z]}, {landmark: (u, v, conf)}
                 2. solve (mixamo_retarget.solve) -> {bone: (rx, ry, rz)} local Euler, every bone
                 3. place (mixamo_retarget.place) -> root_tx/ty/tz (Over Image)
              -> rotations (Null CHOP): <bone>_rx/ry/rz, root_tx/ty/tz, mirror
  fbx (FBX COMP): every bone's rx/ry/rz is an expression on `rotations`; the COMP's tx/ty/tz too
  render: camera_image (at the origin, horizontal FOV = Image FOV) or a fixed Center camera
```

The solver only outputs rotations. TouchDesigner's FBX COMP does the skinning: the bones are Null
COMPs, the meshes deform with them on the GPU. Bone *lengths* never change, so the character keeps
its own proportions whatever the person's are; only the root moves (placement).

## The rig

**What TD gives us.** An FBX COMP imports a skeleton as one Null COMP per bone. A bone's parent is
the Null wired into it (or, in Blender exports, the Null COMP containing it). Each bone's own
parameters are its *rest* local transform: `t` (offset from the parent, in the parent's frame),
`r` (Euler degrees, rotate order xyz), and `s` (scale, which matters for Blender's armature at 100).
TD composes a bone's local transform as `T * Rz * Ry * Rx`, with column vectors.

**Capture** (`puppet_solver.capture`, once per character) reads those constant values before they
are bound, into `{bone: {'parent', 't', 'r', 's'}}`, stored on the solver. Binding the bones to the
CHOP replaces their values with expressions, so the capture must come first; `puppet_runtime.bind`
does it in that order.

**`Rig`** runs forward kinematics once on the rest pose. Per bone it keeps:
- `rest_local`, the rest local rotation matrix;
- `world_rot` and `world_pos`, the rest orientation and joint position in the FBX COMP's space
  (centimetres for Mixamo);
- `order`, parents before children.

Mixamo names are prefixed (`mixamorig_Hips`; TD turns `mixamorig:` into that), and the prefix is
found from the root bone ending in `Hips`.

**Mixamo conventions it relies on:**
- the character faces +Z, with its left side at +X and y up;
- each bone's local +Y points to its child.

The solver doesn't actually need the second one. It measures every rest direction from joint
positions, never from the bone axes, so any rest roll works. The synthetic test rig gives each bone
a different roll to prove it.

## Landmarks

**Columns read.** `puppet_solver.read` takes one tracks row and returns world landmarks by name,
with the column prefix dropped:
- body: `left_shoulder`, ... from the `pose_` columns;
- Holistic's hands keep their names (`left_hand_index_finger_mcp`), since the fingers use them;
- a hand rig takes only its side's `<side>_hand_` columns, or a MediaPipe Hands row of that
  handedness, prefix dropped.

A landmark whose `_conf` is 0, such as an unseen hand, is left out. With several tracks, the
solver takes the one matched this frame, longest-lived first.

**Axes.** The tracker has already turned MediaPipe's world landmarks into TD axes: metres, hip
origin, y up, +z toward the camera. A person facing the camera has their left side at +x, so a
character facing the viewer copies them like a mirror image. **Mirror** swaps left and right and
negates x, for a selfie camera.

**Units.** Units don't matter for rotations: every target is a direction. Metres in, centimetres
on the rig; nothing is scaled.

## Solving one bone

Every driven bone is one row of `TARGETS`:

```
(bone, target primary, target hint, rest primary, rest hint)
('LeftArm', (left_shoulder, left_elbow), (left_elbow, left_wrist), (LeftArm, LeftForeArm), +Z)
```

- The **primary** is the bone's own direction: shoulder to elbow, measured on the landmarks, and
  the same segment measured on the rig's rest joints (bone LeftArm to bone LeftForeArm).
- A direction alone leaves the bone free to spin about itself (roll). The **hint** is a second
  vector that fixes the roll. For the upper arm it's the forearm: the elbow's bend shows which way
  the arm is rotated. On the rest skeleton (a T-pose) the arm is straight, so its rest hint is a
  fixed direction instead, +Z (forward, the way an elbow bends).

`turn()` builds an orthonormal frame from each pair, y along the primary and x toward the hint's
component perpendicular to it:

```
F_target = frame(primary, hint)          F_rest = frame(rest primary, rest hint)
R = F_target * F_restᵀ                    (world rotation taking the rest frame onto the target)
```

The bone's posed world orientation is `R * world_rot[bone]` (the rotation is applied in world space
on top of the bone's rest orientation), and its local rotation is `parent_posed_worldᵀ * that`, which
is converted to Euler xyz for the channels.

**Weak hints.** A straight arm has no bend, so the hint is parallel to the primary and says nothing
about roll. The hint's usefulness is the sine of its angle to the primary, the smaller of target
and rest. Below `HINT_FADE` (0.05 to 0.3) the solver blends (slerp) from the full frame turn toward a
**plain turn**: the parent's turn (`base`) plus the shortest swing onto the primary. A straight arm
then keeps the chest's twist instead of spinning at random.

**Undriven bones** (shoulders, toes, the end bones, any bone whose landmarks are missing this frame)
keep their rest *local* rotation, so they ride along with their posed parent. A driven bone without
a driven parent still uses the nearest driven ancestor's turn as its `base`.

### The bone map (Mixamo body)

| Bones | Primary | Roll from |
|---|---|---|
| Hips | hip midpoint → shoulder midpoint | the hip line |
| Spine2 (chest) | hip midpoint → shoulder midpoint | the shoulder line |
| Spine, Spine1, Neck | blended: slerp of Hips→Spine2 at 1/3, 2/3; Spine2→Head at 1/2 | |
| Head | up = nose direction × ear line | the ear line (rest: +Y up, +X ear line) |
| Arm | shoulder → elbow | the forearm (rest: +Z, an elbow's bend) |
| ForeArm | elbow → wrist | the hand's width, pinky → index knuckle (pronation) |
| Hand | wrist → middle knuckle (Holistic hand), or → between index and pinky (Pose) | the hand's width |
| UpLeg, Leg | hip → knee, knee → ankle | the foot's forward direction, heel → toes |
| Foot | ankle → toes | the hip line |
| Fingers 1-3 | each joint → the next (Holistic hand landmarks) | the palm's width; the thumb by the palm's length |

Rest vectors are measured on the rig's joints with the same meaning. MediaPipe's shoulders sit at
the Mixamo `Arm` bones and its hips at `UpLeg`, not at `Shoulder` / `Hips`.

The torso shares a primary (hips to shoulders) but rolls from two different lines. Twisting at the
waist turns the chest against the hips, and the spine bones split that twist between them.

### Mirror and missing hands

- **Mirror** swaps every `left_` / `right_` name and negates x before solving. The left bones then
  follow the person's right side, as a mirror image would.
- **Unseen hands.** When Holistic doesn't see a hand, its finger bones keep their *last* solved
  local rotation (`_hold_fingers`). Finger rotations are local to the hand bone, so they still
  follow the wrist, which the pose landmarks keep driving.
- **Nobody tracked.** The whole last pose holds; before the first solve, the rest pose.

## Placement (Over Image)

**Center** leaves the FBX COMP at the origin. **Over Image** moves it so the posed character lies
over the person in the video, as seen by `camera_image`: a camera at the origin looking down -z,
with the horizontal **Image FOV** and the input's aspect.

For a posed joint at `q` (from `solve(positions=...)`, rig units) and the matching keypoint at
normalized device coordinate `ndc` (`2u - 1`), the root translation `t` must satisfy, per axis:

```
ndc * tan(fov/2) * -(q_z + t_z) = q_axis + t_axis          (linear in t)
```

Twelve joints give up to 24 equations; `place()` solves them by least squares, each row weighted by
the keypoint's sqrt(confidence). The joints are shoulders, elbows, wrists, hips, knees and ankles,
where the Mixamo bones sit on the keypoints.

Perspective sets the depth: a person who looks smaller is put further away. A solution behind the
camera is rejected. The result is smoothed exponentially (**Placement Smoothing**), since depth is
the noisiest axis.

The FOV is a guess per video. A wrong FOV makes the character too big or small, and it drifts in
depth as the person moves.

## The rigged hand

`hand_retarget.spec(rig)` recognizes a Blender-style hand rig (`handRig_02.fbx`: `pulse_R`, `hand_R`,
`<finger>_base_R`, `<finger>_01.._03_R`). It returns its own target list for the same `solve()`, plus
the joints `place()` uses and the palm point that Center keeps at the origin.

- **Root.** The rig's root (`pulse`) carries the whole hand: wrist → middle knuckle, rolled by the
  knuckle line. The metacarpals ride along rigidly; driving each from its own noisy landmark line
  creased the skin.
- **Phalanges are hinges** (`_hinge`). The target direction, seen from the parent's posed frame,
  is split into a bend about the knuckle line and a sideways spread. Each is clamped to
  `FINGER_LIMITS` (the first joint bends -30..95° and spreads ±25°; the others bend only). There is
  no roll of their own. This is the only joint-limit code so far.
- **One model, either hand.** The model is one hand. The other hand is solved mirrored and the
  render flipped back (the `mirror` channel), because mirroring the model itself (sx = -1) turns its
  shading inside out.

## Cost and caching (Python)

The Script CHOP cooks every frame, but a tracker row only changes on a new model result. So
`cook()` keys on the row's text plus the settings and returns early when nothing changed. That is
why the tracker's number formatting is behavior: the C++ Tracker DAT writes byte-identical text.

A body solve with fingers (65 bones, ~50 driven) costs about **2.7 ms** in TD. Three things grow
with the bone count:
- small numpy operations, which carry microseconds of overhead each;
- `slerp`'s matrix-quaternion round trips;
- writing ~200 channels, done in place once they exist.

Each bone parameter's expression (`op('../rotations')['bone_rx']`, about 200 of them) is evaluated
by TD's Python on every change. That cost is outside the solver and hasn't been measured.

## Known limits

- **No rotation smoothing.** Only the tracker's position smoothing applies, and noise in depth
  shows up as jitter in roll.
- **No joint limits on the body:** elbows and knees can bend backward on bad depth.
- **Side-on people:** MediaPipe can swap left and right, and the character turns 180° for a frame.
- **Head yaw** looks under-turned.
- **The FOV for Over Image** is a guess per video.
- **Mixamo naming only** for bodies. Another humanoid rig needs a bone-name map, which would be a
  table, not code.
- **One person** (the first matched track).

## The C++ solver (phase 6 of [TODO-tracking-cpp.md](TODO-tracking-cpp.md))

**Goal:** the same rotations and placement from a C++ custom CHOP, then the features that are
awkward in per-frame Python (rotation smoothing, joint limits).

**Status (Nov 13):** steps 1-3 are done; `MixamoPuppet.tox` runs on the Puppet CHOP. Results:
- **Offline** (`tests/test_puppet_cpp.py`, 24 combinations of rig, fixture and settings):
  rotations within 4e-14, roots within 5e-12.
- **In TD** (`tests/td_puppet_lab.py`, 300 frames each): every channel the same.
- **Per frame:** body 2.19 → 0.34 ms, rigged hand 1.10 → 0.26 ms.

Plugin reference: [cpp/Puppet/README.md](../cpp/Puppet/README.md).

One thing learned in the lab test: the Script CHOP solves whenever anything pulls it, at whatever
moment in the frame that is. A smoothed root therefore depends on *when* the solver cooks, not
only on which rows it saw. To compare two solvers, cook both from the same place, or compare with
Placement Smoothing at 0.

**Design:**
- **`cpp/Puppet/`**, laid out like `cpp/Tracking`:
  - **`puppet_core`:** the rig, targets, solve, place and the hand spec. No TD dependency, doubles
    throughout.
  - **`PuppetCHOP.dll`:** the custom CHOP, type `PuppetCHOP`. A fifth Allow prompt; one custom OP
    per DLL.
  - **`puppet_replay.dll`:** a C API for the tests, through ctypes.
- **Inputs:**
  - **Tracks DAT** parameter: the same table, read with `OP_DATInput`.
  - **Rest Pose DAT** parameter: the captured rig as a table (`bone, parent, tx, ty, tz, rx, ry,
    rz, s`). Capture stays in Python (it walks COMPs, once per character) and writes this table
    instead of `store()`. The CHOP rebuilds its `Rig` when the table's text changes.
  - **Image TOP** parameter: for the aspect only, from `textureDesc`.
- **Parameters:** Hand, Mirror, Placement, Fov, Placesmoothing, as the Script CHOP has them, so the
  tox's bindings stay the same. Rotation smoothing and joint limits come after the port is proven.
- **Output:** one sample. The same channel names in the same order (`<bone>_rx/ry/rz` in rig order,
  then `root_tx/ty/tz`, `mirror`), so the bone expressions and users' networks keep working.
- **Info CHOP:** `cook_ms`, `solve_ms`, `cook_ms_avg`, `bones`, `driven`, so the tox can get a
  Performance page like the wrappers'.
- **Same-row skip:** compare the row's text and the settings, as the Python does.

**Proving it's the same.**
- Rotations in doubles won't match numpy bit for bit: numpy's matrix products sum in a different
  order. The test is agreement within 1e-6 degrees, and the same root translation within 1e-6 rig
  units.
- `puppet_replay` and the Python solver run every frame of the recorded Holistic tables (CPU, GPU,
  GPU without the face mesh) on the synthetic rig. With them, `build/ybot_rest.json` (Y Bot) and
  `build/hand_rest.json` (the hand rig), which aren't committed, run in both Mirror states and both
  placements.
- Near gimbal lock (|ry| ≈ 90°) the Euler split can differ while the matrix is the same, so the
  test compares the matrices there.
- In TD: a side-by-side of the Script CHOP and the Puppet CHOP on the same tracks in
  `/project1/PuppetLab`, every channel diffed, then timed.

**Steps:**
1. The core, the replay library, and the offline equivalence test. No TD restart needed.
2. The CHOP plugin, built and registered (a TD restart). The builder makes the Puppet CHOP, and the
   capture writes the Rest Pose table.
3. The lab side-by-side and timing. Switch the tox, regenerate it, update the manual.
4. Then the new behavior, each piece a deliberate change with its own tests:
   - rotation smoothing (slerp toward the target, per bone, with a speed-dependent factor like the
     tracker's);
   - body joint limits (elbow and knee hinges, as the fingers have);
   - side-on flip rejection.
5. Retire the Python solver together with the Python tracker, once the release has been in use.
