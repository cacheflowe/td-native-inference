# Puppet CHOP

`PuppetCHOP.dll` (custom CHOP type `PuppetCHOP`, label "Puppet"): the puppet solver in C++, a port of
`python/puppet_solver.py`, `mixamo_retarget.py` and `hand_retarget.py`. How the solver works:
[docs/PUPPETS.md](../../docs/PUPPETS.md). `MixamoPuppet.tox` uses it as its `solver`.

Build: `cpp\Puppet\build.cmd`. It needs MSVC v142, like the other plugins, and no CUDA. The build
copies the DLL into `plugins/` and also builds `puppet_replay.dll` for the tests.

## Parameters (Puppet page)

| Parameter | What |
|---|---|
| Tracks DAT | A tracker's tracks table with World Landmark Columns on (`<joint>_wx/_wy/_wz`); the person matched this frame, longest-lived first |
| Rest Pose DAT | The rig's rest pose: one row per bone, columns `name, parent, tx, ty, tz, rx, ry, rz, s`. Captured from the FBX COMP's bones by `puppet_runtime.capture_rest`. The rig rebuilds when the table's text changes |
| Tracked Hand | Hand rigs: Any / Right / Left |
| Mirror | Body rigs, Center placement: swap left and right |
| Placement | Center (at the origin) or Over Image (root_tx/ty/tz laid over the person) |
| Image FOV | Horizontal FOV assumed for the source camera (Over Image) |
| Image TOP | The tracked image, for its aspect |
| Placement Smoothing | Exponential smoothing of the root; 0 = raw |

## Parameters (Smoothing page)

None of these are in the Python solver; with all of them off the CHOP matches it exactly.
[docs/PUPPETS.md](../../docs/PUPPETS.md#smoothing-and-limits-c-solver-only) explains them and
has the measurements.

| Parameter | Default | What |
|---|---|---|
| Landmark Cutoff (Hz) | 1.5 | One Euro filter on the world landmarks (hand points relative to their wrist); 0 = off |
| Speed Coefficient | 4 | How fast the cutoff rises with speed (m/s) |
| Slope Cutoff (Hz) | 1 | Smooths that speed |
| Hand Roll Cutoff (Hz) | 2 | The forearms', hands' and fingers' roll, filtered on its own and held while the knuckle line is unreliable; 0 = off |
| Knee Limits | on | Knees as hinges: -5° to 150° bend, ±10° spread |
| Relax Unseen Limbs | on | Body rigs: joints with `conf` 0 (the tracker's joint gate) relax. Hips go under the shoulders, arms hang, legs stay at rest |

The filters step with TD's frame time between solves.

The rig type is chosen from the bone names. A root bone named `<prefix>Hips` makes a Mixamo body;
`hand_R` / `pulse_R` (or `_L`) make a Blender-style hand rig.

## Output

One sample:
- `<bone>_rx`, `_ry`, `_rz` for every bone in rig order (local Euler degrees, rotate order xyz);
- `root_tx`, `root_ty`, `root_tz`;
- `mirror`: 1 when a hand rig shows the other hand mirrored; flip the render.

The channel names are the Script CHOP solver's, so bone bindings don't change.

A cook whose tracks row and settings are the same as last time keeps the channels as they are.

**Info CHOP:**
- `cook_ms`: the whole cook;
- `solve_ms`: reading the row, solving and placing;
- `cook_ms_avg`;
- `bones`;
- `driven`: bones turned toward landmarks or blended;
- `solved`: 1 when this cook solved, 0 when it kept the channels.

## Tests

- **`tests/test_puppet_cpp.py`:** `puppet_replay.dll` against `puppet_solver.cook` on recorded
  Holistic tables, every frame. Rotations are compared as matrices, within 1e-9; roots within 1e-7
  relative. Rigs:
  - the synthetic rig from `test_retarget`;
  - `build/ybot_rest.json` and `build/hand_rest.json` when present.
- **`tests/td_puppet_lab.py`, in TD:** a Puppet CHOP next to each Script CHOP solver in
  `/project1/PuppetLab`, diffed every frame and timed.

**Nov 13 results.** Both tests agree to 1e-14 on rotations and exactly on roots. Per frame in TD:

| Solver | Python | C++ |
|---|---|---|
| Body (Y Bot) | 2.19 ms | 0.34 ms |
| Rigged hand | 1.10 ms | 0.26 ms |
