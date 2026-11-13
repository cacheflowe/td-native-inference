# Puppets: how the solver drives a rigged character

Developer notes on `MixamoPuppet.tox` and the rigged-hand lab: how tracked landmarks become bone
rotations on a skinned FBX character, the conventions that make it work, where the time goes, and
how the solver moved from Python to C++. User-facing setup is in [MANUAL.md](MANUAL.md#puppets).

Code (`cpp/Puppet`, [README](../cpp/Puppet/README.md)):
- `src/Retarget.cpp`: the solver (rig, targets, solve, place, the hand rig's bone map, twist
  smoothing);
- `src/Puppet.cpp`: the cook (reading the tracks row, landmark smoothing, relaxing unseen limbs,
  placement smoothing, the channels);
- `src/PuppetCHOP.cpp`: the CHOP.

In Python: `python/puppet_runtime.py` (inside the tox: import the FBX, capture the rest pose, bind
the bones) and `python/puppet_builder.py` (builds the tox). Tests: `tests/test_puppet_cpp.py`.

Until Nov 13, 2026 the solver was Python (`puppet_solver.py`, `mixamo_retarget.py`,
`hand_retarget.py`, in git history); the names below are the C++ ones, which follow it.

## The pipeline

```
video -> Holistic / Pose tox (World Landmark Columns on)
           out_tracks: one row per person, <joint>_wx/_wy/_wz (metres) and <joint>_x/_y/_conf (image)
              |
              v
MixamoPuppet.tox
  in_tracks -> solver (the Puppet CHOP, Solver::cook)
                 1. read the person's row -> {landmark: [x, y, z]}, {landmark: (u, v, conf)}
                 2. solve                  -> {bone: (rx, ry, rz)} local Euler, every bone
                 3. place                  -> root_tx/ty/tz (Over Image)
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

**Capture** (`puppet_runtime.capture_rest`, once per character) reads those constant values before
they are bound, into the `rest_pose` table (`name, parent, tx, ty, tz, rx, ry, rz, s`) the CHOP
reads. Binding the bones to the CHOP replaces their values with expressions, so the capture must
come first; `puppet_runtime.bind` does it in that order.

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

**Columns read.** `Solver::read` takes one tracks row and returns world landmarks by name,
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

Every driven bone is one row of the target table (`mixamoTargets()`):

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
and rest. Below `kHintFade` (0.05 to 0.3) the solver blends (slerp) from the full frame turn toward a
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
  local rotation (`Solver::holdFingers`). Finger rotations are local to the hand bone, so they still
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

`handSpec(rig)` recognizes a Blender-style hand rig (`handRig_02.fbx`: `pulse_R`, `hand_R`,
`<finger>_base_R`, `<finger>_01.._03_R`). It returns its own target list for the same `solve()`, plus
the joints `place()` uses and the palm point that Center keeps at the origin.

- **Root.** The rig's root (`pulse`) carries the whole hand: wrist → middle knuckle, rolled by the
  knuckle line. The metacarpals ride along rigidly; driving each from its own noisy landmark line
  creased the skin.
- **Phalanges are hinges** (`hinge()`). The target direction, seen from the parent's posed frame,
  is split into a bend about the knuckle line and a sideways spread. Each is clamped to
  the finger limits (the first joint bends -30..95° and spreads ±25°; the others bend only). There is
  no roll of their own. This is the only joint-limit code so far.
- **One model, either hand.** The model is one hand. The other hand is solved mirrored and the
  render flipped back (the `mirror` channel), because mirroring the model itself (sx = -1) turns its
  shading inside out.

## Smoothing and limits

The Puppet CHOP adds these over the original solver, on its Smoothing page; with all of them off it
is the plain solve. The defaults were chosen by measuring the recorded Holistic GPU dancer clip.

**Why it was jittery.** CPU MediaPipe Holistic smooths its world landmarks inside MediaPipe (a One
Euro filter); the GPU port of Holistic doesn't, so the puppet got raw per-frame landmarks. The
hands were by far the worst: measured on the dancer clip, almost every large step of a forearm
was *roll* (about the bone), not direction. Two causes:
- **Flattened hands.** The roll comes from the knuckle line (pinky → index knuckle, about 5 cm).
  When the palm is edge-on, Holistic's 3D hand often comes out flattened in depth, with a knuckle
  line of 1-2 cm instead of 4-5, and its direction is then mostly noise.
- **Source switches.** When Holistic loses a hand, the hint fell back to the pose model's coarse
  index / pinky points, which disagree with the hand model's.

The result was roll jumps of 60-120° in one frame.

**1. Landmark smoothing.** A One Euro filter on every world landmark before solving, with the
Filter CHOP's parameters:
- **Cutoff** (Hz): the cutoff while still. Default 1.5; 0 turns it off.
- **Speed Coefficient:** raises the cutoff with speed (m/s), so quick moves aren't delayed.
  Default 4.
- **Slope Cutoff:** smooths that speed. Default 1.

Hand landmarks are filtered *relative to their wrist*. When the arm swings, every hand point
moves fast, which would open the filter and let the fingers' noise through; relative to the wrist
only the hand's own shape moves. The filter steps per solve with the real time between solves,
and starts over for a new track, after a gap of more than half a second, or for a landmark that
reappears.

**2. Hand roll smoothing (swing / twist).** For bones whose roll comes from a knuckle line (the
forearms, hands and fingers, and the hand rig), the turn is split into:
- **swing:** the shortest turn onto the bone's direction, from long, reliable segments;
- **twist:** the roll about that direction.

The twist gets its own One Euro filter (**Hand Roll Cutoff**, default 2 Hz). It is made
continuous by always taking the roll nearest the current one, so ±180° doesn't wrap, and it is
followed only as far as the knuckle line can be trusted:
- the line's length against the hand's (wrist → middle knuckle): Holistic's median is 0.55;
  confidence ramps from 0 at 0.2 to 1 at 0.45;
- a hand the table has but Holistic didn't see this frame gets confidence 0, so its roll holds.

Body bones roll from long lines (hips, shoulders, ears), so their twist isn't filtered;
filtering it only delayed real turns.

**3. Knee limits.** The knees become hinges, like the fingers: a bend of -5° to 150° about the
character's left-right axis and ±10° of spread, with their twist following the thigh. They keep
depth noise from bending a knee backward or sideways; on the dancer clip they change nothing.

**4. Chest-up framing (Relax Unseen Limbs).** At a desk webcam the models still report legs, and
those guesses are noisy: on a recorded chest-up clip the thighs and shins swung about 14° per solve,
with jumps of up to 150°. The tracker's joint gate
([MODELS.md](MODELS.md#tracking)) now reports them as unseen (`conf` 0), so they drop out of the
landmarks.

On its own that would leave undriven bones at the rig's rest pose, so the solver fills in instead:
- hips missing but shoulders seen: hips go 0.5 m under the shoulders (a little narrower), so the
  torso still turns with the shoulder line;
- an elbow missing with its wrist tracked: the elbow goes between the shoulder and the wrist, bent
  a little down and back, so the hand stays where it was tracked (the tracker already keeps an
  elbow whose hand is in view; this covers what's left);
- elbow and wrist missing: they hang, the forearm slightly forward so the upper arm's roll is
  defined;
- legs: left undriven, straight down from the hips.

On that clip the shins now don't move, and the thighs move only while the knees really are in view.
A limb still jumps once when it switches between tracked and relaxed.

**Measured** on the dancer clip (Y Bot, 264 solves). Jitter is the RMS frame-to-frame angular
acceleration, in degrees:

| | Body | Forearms | Hands | Fingers |
|---|---|---|---|---|
| Raw jitter | 10.4 | 56.3 | 42.0 | 31.4 |
| Raw: frames jumping over 30° | 1.5% | 21.5% | 19.8% | 11.2% |
| Default jitter | 4.7 | 10.6 | 18.6 | 13.4 |
| Default: frames jumping over 30° | 0.8% | 0.4% | 5.7% | 2.9% |

Body lag, measured against a centred (zero-lag) smoothing of the raw motion, goes from 4.5° to
8.4°. The hands remain the weakest part: their direction comes from a 9 cm wrist-to-knuckle line
whose depth is noisy.

## Cost and caching

The CHOP cooks whenever the bones pull it, but a tracker row only changes on a new model result. So
`Solver::cook` keys on the row's text plus the settings and keeps its channels when nothing changed.

A body solve with fingers (65 bones, ~50 driven) costs about 0.3 ms in TD (the Python solver took
2.7 ms: small numpy operations carry microseconds of overhead each).

Each bone parameter's expression (`op('../rotations')['bone_rx']`, about 200 of them) is evaluated
by TD's Python on every change. That cost is outside the solver and hasn't been measured.

## Known limits

- **No elbow or arm-twist limits.** An elbow can't bend backward, because the upper arm's roll
  comes from its bend, but bad depth can twist the upper arm instead.
- **Side-on people:** MediaPipe can swap left and right, and the character turns 180° for a frame.
- **Head yaw** looks under-turned.
- **The FOV for Over Image** is a guess per video.
- **Mixamo naming only** for bodies. Another humanoid rig needs a bone-name map, which would be a
  table, not code.
- **One person** (the first matched track).

## The C++ solver (phase 6 of [TODO-tracking-cpp.md](TODO-tracking-cpp.md))

**Goal:** the same rotations and placement from a C++ custom CHOP, then the features that are
awkward in per-frame Python (rotation smoothing, joint limits).

**Status (Nov 13):** steps 1-4 are done; `MixamoPuppet.tox` runs on the Puppet CHOP.
- **The port was proven exact** before anything new went in: rotations within 4e-14 and roots
  within 5e-12 offline (24 combinations of rig, clip and settings), and every channel the same in
  TD over 300 frames, body and hand.
- **Speed, per frame:** body 2.19 → 0.34 ms, rigged hand 1.10 → 0.26 ms.
- **The C++ output is its own reference** (`tests/data/puppet/reference.json`, see
  [cpp/Puppet/README.md](../cpp/Puppet/README.md#tests)); the Python solver was deleted Nov 13.

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

**Proving it's the same** (how the port was checked; done):
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
5. Retire the Python solver together with the Python tracker (done Nov 13).
