# User manual

For TouchDesigner users who have installed a release (see the [README](../README.md#install)) and
want to know what each `.tox` does, how to wire it and how the pieces behave. Per-model details and
output columns are in [MODELS.md](MODELS.md); this page covers what they share.

## 1. Quick start

1. Drag a `.tox` from the `tox` folder into a network. The node shows which model it is. If it
   says the plugin was not found, see [Troubleshooting](#8-troubleshooting).
2. Wire the input, by what the model reads:
   - **Video models** (hands, pose, faces, YOLO26, depth, matting, ...): any TOP into the input
     (Video Device In, Movie File In, ...).
   - **Audio models** (Speech to Text, Sound Classifier): an audio CHOP (Audio Device In, Audio
     File In, ...). Any sample rate works; channels are mixed to mono.
   - **Text to Speech**: type into **Text** and pulse **Speak**, or wire a DAT into the input.
3. It runs right away (**Active** is on). View the node's first output: for a video model, a
   debug overlay of boxes and skeletons on a transparent background (or the result image for
   matting and depth); for Speech to Text, a closed-caption strip; for the Sound Classifier, a
   bar list of the top sounds.

To compare models, `VisionDemo.tox` and `AudioDemo.tox` hold all of them with one toggle each on
their **Models** page; a model that is off does not cook.

The first time an ONNX model runs it is slow to start; see [section 4](#4-speed-and-latency).

## 2. What every video tox has

Every video model has the same outputs, so swapping one model for another does not change your
wiring.

| Output | What it is |
|---|---|
| TOP 1 | The debug overlay, sized like the input, transparent where nothing is drawn. Matting and depth models put their result image here instead. |
| TOP 2 | The input image, unchanged. |
| DAT 1 `tracks` | One row per tracked hand, person, face or object. Columns are listed in [section 3](#3-tracking). |
| DAT 2 / 3 `joints`, `bones` | Keypoints and skeleton lines in the format the debug overlays read. Landmark models only. |
| CHOP | `t<id>_<column>` channels for every numeric column of every track (`t1_cx`, `t1_gesture_onset`, ...), plus the model's own status channels: `loaded`, `loading`, `infer_ms`, `num_detections`, `result_delay` and others (OnnxTOP adds `tensorrt` and `trt_building`; MediaPipeTOP adds `results` and `dropped_frames`). |

Positions are 0 to 1 across the image with the origin at the **bottom left**, like TouchDesigner's
UV space. A channel `t1_cx` is the horizontal centre of track 1; multiply by your canvas width.

Matting and depth models (MODNet, RVM, Depth Anything, MediaPipe Segmentation) have no tracks;
TOP 1 is the image result and TOP 2 the input. YOLO26 Segmentation has extra TOPs (see
[MODELS.md](MODELS.md#yolo26-segmentation-yolo26segtox)).

The audio toxes have their own outputs, listed in [MODELS.md](MODELS.md#audio-models).

### Parameters

Parameters sit on pages of the tox:

- **Model**: **Active**, the few settings that matter for that model (how many hands, which
  class, input size, **Wait for Result**, ...) and **Reload**.
- **Tracking**: how detections become stable tracks (video models with tracks only).
- **Debug**: **Debug Overlay** on or off (**Show Result** on image models), and on landmark
  models **Overlay Layers**, **Joint Size** and **Joint Outline**.
- **Performance**: read-only timings in milliseconds, see [section 4](#4-speed-and-latency).
- **About**: the package version. A tox expects the plugins of its own version.

**Active** off stops the model and clears its output. **Reload** (a pulse) retries after a
failure, for example after you put a missing model file in place, or reloads the model in place.
**Debug Overlay** off makes TOP 1 transparent and skips the drawing work; do this in a show if you
only use the tables or channels.

## 3. Tracking

The models report what they see on each frame. A tracker inside the tox turns those reports into
**tracks**: the same hand, person, face or object keeps the same `track_id` from frame to frame.
Without it you would have a list of detections that reshuffles every frame.

What the tracker does between model results:

- **Ids.** A detection is matched to the track it most likely continues. Boxes (YOLO26, YuNet)
  are matched by overlap with the track's predicted box. Skeletons (pose, hands, face) are
  matched by how far the stable joints (torso, wrist and palm, eyes and nose) moved, relative to
  the skeleton's size, because limbs make boxes jump.
- **Smoothing.** **Output Smoothing** steadies box and joint positions (0 = raw). Higher is
  steadier and lags more. MediaPipe models already smooth, so their default adds none.
- **Losing and holding.** A track that is not detected is kept for **Track Loss Frames**, then
  removed. Its `lost_frames` column counts how long it has been missing. A longer hold bridges
  occlusions but leaves a departed person's ghost on screen longer. After you switch the video
  source, old tracks linger for this many frames.
- **Coasting.** On the hands and pose toxes with **Coast While Lost**, a lost track keeps drifting
  along its last motion, so a fast hand that MediaPipe drops for a frame keeps its id.
- **Snap-back after a gap.** A track found again after 2 or more lost frames starts at its new
  position instead of easing in from the old one, so overlays do not fly in. A Holistic hand or
  face that goes missing and returns is hidden for 3 frames while its smoothing settles.
- **Presence.** Some modes add a 0 to 1 `presence` column that fades in and out, handy for fading
  a graphic.
- **Unseen joints.** Body models report every joint even when part of the body is outside the
  picture. A person at a desk webcam, framed from the chest up, still gets knees and ankles, guessed
  below the image or squashed onto its bottom edge. On the body toxes (Pose, Holistic, Holistic GPU,
  YOLO26 Pose, MoveNet, RTMW3D) the tracker outputs such joints as unseen: `conf` 0 in the tracks
  table, left out of the `joints` and `bones` tables (so the debug overlay doesn't draw them). It
  uses two settings on the Tracking page:
  - **Joint Confidence Threshold:** joints the model scores below this (0 = off).
  - **Hide Off-Image Joints:** joints outside the image or on its edge.

  On the legs, a joint whose parent is unseen is unseen too (no knee without a hip). On the arms, a
  hand in view keeps its elbow, even one below the image or scored low (unless YOLO or MoveNet
  squashed it onto the image's edge, where its position means nothing). Tracking itself still uses
  every joint, so ids don't change.

The `tracks` table always begins with `track_id, label, score, cx, cy, w, h, x_left, x_right,
y_top, y_bottom, vx, vy, lost_frames, total_frames`. After those come model-specific columns
(`gesture`, `gesture_onset`, `class_name`, `yaw`, `emotion_label`, ...) and one `<joint>_x`,
`<joint>_y`, `<joint>_conf` triple per keypoint. `cx`, `cy` is the centre, `vx`, `vy` the
velocity per frame. To trigger on something, use a channel such as `t1_gesture_onset` (1 for one
frame when a gesture starts). To follow a body part, read its joint columns (for example
`<joint>_x`) from the table; the joint names for each model are in the `joints` table.

If ids swap between neighbours or fast movement keeps making new ids, tune **Relative Match
Distance** on the Tracking page (lower for swaps, higher for fast movement). The full tuning
notes, depth (`z`) and 3D world columns are in [MODELS.md](MODELS.md#tracking).

## 4. Speed and latency

**Where the work runs.** ONNX models (YOLO26, MoveNet, YuNet, Depth Anything, MODNet, RVM,
HSEmotion) run on the GPU through TouchDesigner's ONNX Runtime and TensorRT. MediaPipe models
run on the CPU on a worker thread.

**Wait for Result** (Model page on most detection and landmark models). The model takes a few
milliseconds, and the question is whether TouchDesigner waits for it.

- **On:** the cook waits for this frame's result, so boxes and joints match the image exactly
  (`result_delay` is 0). Costs the inference time in that cook.
- **Off:** the cook starts the work and moves on; the result is picked up on a later cook, so it
  describes an older frame. TouchDesigner's frame rate is spared, but an overlay can trail the
  picture on fast motion.
- **Defaults:** On for ONNX models (GPU inference is a few milliseconds). Off for MediaPipe
  models (CPU inference is slower, so waiting would cost frame rate).
- **Max Wait (ms)** (MediaPipe models, default 20): with Wait for Result on, the longest a cook
  waits. Past it the result arrives on a later cook. Turning it on is worth it for fast models
  (Face Detector, Hands); a slow one (Holistic) costs frame rate.
- A still image is processed once, so it is exact either way.

**`result_delay`** is a CHOP channel: how many input frames old the current result is. If an
overlay trails, check it. To draw a MediaPipe result on the frame it belongs to, put a Cache TOP
on the input (Cache Size above the largest delay) and set its Output Index to `-result_delay`.

**Raw-mode models** (Depth Anything, MODNet, RVM) never wait and have no delay: their image
stays on the GPU and is always the current frame, which is why they have no Wait for Result
parameter.

**First run and `trt_cache`.** The first time each ONNX model runs, it starts on CUDA within
seconds and builds a TensorRT engine in the background. This takes a few minutes, and the frame
rate is lower meanwhile; the CHOP channel `trt_building` is 1 while it works. The engine is
cached in a `trt_cache` folder next to the model, so later loads take seconds. Delete that folder
to force a rebuild. Each input size of a dynamic-size model (**Dynamic Size**) is its own engine,
so the first use of a new size builds again.

**Staying on CUDA.** The node falls back to CUDA (slower than TensorRT, still on the GPU) if
TensorRT cannot be used. In these cases check the `tensorrt` channel (1 = TensorRT running):

- The engine is still building.
- An NVIDIA Background TOP loaded its own TensorRT earlier in the same TouchDesigner session. It
  disables TensorRT until you restart; cook the ONNX model first or avoid that TOP.
- RVM runs on CUDA by design: its state changes shape after the first frame, which would make
  TensorRT rebuild.

**The Performance page** shows where a tox's time goes, live. The values are only computed while
the page is on screen, so it costs nothing otherwise.

| Value | What it is |
|---|---|
| Inference | The model itself: GPU time for ONNX models, the worker thread's time for MediaPipe. Off TouchDesigner's main thread unless Wait for Result is on. |
| Plugin Cook | The plugin's cook on the main thread: uploading the frame, starting the model, reading results back. |
| Model Launch, Decode | ONNX models: parts of Plugin Cook (starting the run; reading back and decoding boxes, keypoints or images). |
| Info DAT | Writing the detections into the plugin's Info DAT. |
| Tracker | Tracking and writing the tracks, joints and bones tables. |
| Detector | Models with a nested person or face detector: that detector's main-thread total. |
| Debug Overlay | Drawing TOP 1. Turn Debug Overlay off and this drops to near 0. |
| Main Thread Total | Everything inside the tox on the main thread last frame, as TouchDesigner measures it. |
| Result Delay, Dropped Frames | See `result_delay` above; frames a MediaPipe worker skipped while busy. |

Plugin and tracker times are running averages (about ten frames); the others are last frame's, so
the rows don't add up exactly.

**Where to spend less.** Lower **Dynamic Size** on depth and matting models; set **Max
Hands / Faces / Poses** to what you expect (MediaPipe runs its detector every frame while fewer
are tracked than the maximum); set **Classify Interval** higher on HSEmotion; turn **Debug
Overlay** off when you do not need the picture; turn **Active** off on models you are not using.

## 5. Debug overlays

TOP 1 of a video tox is drawn by two shared components, `tox/shared/DebugBB.tox` (boxes and
labels from `tracks`) and `tox/shared/DebugKeypoints.tox` (joints and bones from `joints` and
`bones`). Each track gets its own colour; labels read like `161 person 94%`.

To see it in place, composite TOP 1 over TOP 2 with an Over TOP (TOP 1 into the first input, TOP 2
into the second). The overlay is sized like the input and its background is transparent, so it
also works over any other picture of the same aspect ratio.

On landmark models, the Debug page has:

- **Overlay Layers**: Both, Bounding Boxes or Joints.
- **Joint Size** and **Joint Outline** (a white ring around each joint).

The shared toxes are loaded from `tox/shared/` inside this repository's own project. In a release
tox they are embedded, so there is nothing to wire. In your own project, do not edit them; adjust
the parameters on the model tox instead.

## 6. The toxes

| Tox | Input | What it does | Main outputs |
|---|---|---|---|
| `MediaPipeHands` | video | Hands with 21 joints each; **Task** Gesture also names the hand shape | tracks with `gesture`, `gesture_onset`; joints, bones |
| `MediaPipePose` | video | 33-point body skeleton, a few people | tracks, joints, bones; optional world positions |
| `MediaPipeFace` | video | 478-point face mesh, optional blendshapes and transform matrix | tracks, joints |
| `MediaPipeFaceDetector` | video | Fast face boxes with 6 points | tracks |
| `MediaPipeHolistic` | video | Body, both hands and face mesh of one person | tracks, joints, bones |
| `DAViDNormals` / `DAViDMatte` | video | A person's surface normals / soft matte (one model each) | TOP 1 the image |
| `DAViDHuman` | video | A person's depth, normals and matte from one model | TOP 1 depth, TOP 3 normals, TOP 4 matte |
| `MediaPipeHolisticGPU` | video | The same on the GPU (ONNX), one or more people; needs YOLO26 | tracks, joints, bones |
| `MediaPipeSegment` | video | Person/background mask | TOP 1 mask |
| `Yolo26Detect` | video | Boxes for 80 object classes. Class Filter keeps one class | tracks with `class_name` |
| `Yolo26Seg` | video | Detection plus a mask per object; Matte Encoding Instances keeps who owns each pixel | TOP 1 matte, TOP 3 boxes, TOP 4 `instances` |
| `Yolo26Pose` | video | 17-point skeletons for many people | tracks, joints, bones |
| `MoveNet` | video | 17-point skeletons for up to 6 people | tracks, joints, bones |
| `YuNet` | video | Face boxes, 5 points, head `yaw`, `pitch`, `roll` | tracks |
| `HSEmotion` | video | Finds faces and scores 7 emotions each | tracks with `emotion_label`, `emotion_<name>` |
| `DepthAnything` | video | Relative depth of the whole frame (near is bright) | TOP 1 depth |
| `MODNet` | video | Person alpha matte, hair detail included | TOP 1 matte |
| `NeuFlowV2` | video | Optical flow: how every pixel moved since the previous frame | TOP 1 flow as color, TOP 3 flow in UV (R across, G up) |
| `HandsGeometry` | joints DAT | Draws tracked hands as 3D spheres and cylinders (needs a POP-capable TouchDesigner, 2025+) | TOP render, POP geometry |
| `MixamoPuppet` | video + tracks | Poses a Mixamo character from Holistic / Pose tracking (see Puppets) | TOP 1 character, TOP 2 over the input, CHOP rotations |
| `RVM` | video | Person alpha matte that stays steady across frames (recurrent); **Reset State** after a cut | TOP 1 matte |
| `MediaPipeSpeechToText` | audio CHOP | Speech to text with closed captions | TOP 1 captions; CHOP `speaking`, `new_segment`; DATs `transcript`, `latest` |
| `MediaPipeTextToSpeech` | text or DAT | Speaks text in 54 voices | CHOP 1 audio; TOP 1 caption; DAT `lines` |
| `MediaPipeSoundClassifier` | audio CHOP | Recognizes 521 sound classes | CHOP 1 scores, CHOP 2 `<name>_onset`; DAT `top` |
| `VisionDemo` | video | All video models on toggles | TOP 1 composited overlays |
| `AudioDemo` | audio, optional DAT | The three audio models on toggles | TOP 1, CHOP 1, DATs |

Licenses and fetched models:

- **YOLO26** (Detect, Seg, Pose) is AGPL-3.0 and **Robust Video Matting** is GPL-3.0. Neither is
  in the zip. Run `fetch_models.cmd` in the installed folder; it asks before downloading each.
- **Text to Speech** needs the Kokoro voice (about 280 MB), also from `fetch_models.cmd`.
- Everything else is bundled. Model licenses are in [NOTICE](../NOTICE).

The Models page of each tox in [MODELS.md](MODELS.md#the-models) explains its own parameters and
columns. Choosing between similar ones:

- **Pose:** YOLO26 Pose or MoveNet for crowds on the GPU; MediaPipe Pose for a few people, with
  3D world positions.
- **Faces:** Face Detector or YuNet for where faces are (YuNet also gives head direction and
  finds small, distant faces); Face for the mesh and expressions.
- **Cutout:** MODNet per frame; RVM when the subject moves fast and a per-frame matte flickers;
  YOLO26 Seg when you need a separate mask for each person.

## 7. Recipes

**Mask a person and composite (MODNet or RVM).**

1. Drop `MODNet.tox` (or `RVM.tox`, after running `fetch_models.cmd`) and wire your camera in.
2. Wait for the first-run engine build (MODNet; RVM runs on CUDA and starts straight away).
3. TOP 1 is the matte at the input's size, white where the person is. Use it as the mask of a
   Composite or Over TOP, or multiply it with the input: matte and TOP 2 into a Multiply
   Composite, then Over a background.
4. Lower **Dynamic Size** if frame rate drops. On RVM, pulse **Reset State** after a cut and set
   **Scalar Input** to 0.25 for HD input or 0.5 for SD.

**Drive a graphic from hand landmarks.**

1. Drop `MediaPipeHands.tox`, set **Max Instances** to the number of hands you expect.
2. Select the CHOP channels you need with a Select CHOP: for example `t1_cx` and `t1_cy` for hand
   1's centre. Track ids keep counting up as hands appear, so select by pattern (`t*_cx`) or
   look the id up in the `tracks` table when you need a specific hand.
3. Feed `t1_cx`, `t1_cy` into a Math CHOP (scale to your canvas) and then your Geometry or
   Transform TOP. Fingertip and joint positions are in the `tracks` table columns
   (`<joint>_x`, `<joint>_y`).
4. For triggers, use `t1_gesture_onset` (1 on the frame a gesture starts) and read `gesture` for
   the name. If hands feel laggy, raise **Output Smoothing** or check `result_delay`.
5. MediaPipe reports handedness as for a mirrored (selfie) image, so unmirrored camera input
   reports the opposite hand.

**Read speech captions.**

1. Drop `MediaPipeSpeechToText.tox` and wire an Audio Device In CHOP into it.
2. Speak. TOP 1 shows the newest words on a transparent image with a dark strip; Over it on your
   video. The `latest` DAT holds the newest sentence and the current guess; the CHOP channel
   `new_segment` is 1 for the cook a sentence is finalized.
3. If room noise becomes text, raise **Noise Ratio** or leave **Skip Silence** on; if quiet speech
   is missed, lower **Noise Ratio**. The **Captions** page sets font size, hold time and caption
   length.
4. To have it repeat what it hears, wire `latest` into `MediaPipeTextToSpeech`'s DAT input
   (use headphones, or the microphone hears the voice).

**Overlay a debug view over the input.**

1. Any video tox: TOP 1 into the first input of an Over TOP, TOP 2 into the second.
2. Switch **Overlay Layers** on the Debug page to show only joints or only boxes.
3. In a show, turn **Debug Overlay** off and use the tables and channels only.

## 8. Troubleshooting

- **"The plugin required for this Custom OP was not found"**, or the node is blank: the plugins
  are not loaded. Check that `Documents\Derivative\Plugins\td-native-inference\` holds the `.dll`
  and `.exe` files and restart TouchDesigner. On the first launch TouchDesigner asks permission
  to load `OnnxTOP.dll`, `MediaPipeTOP.dll`, `MediaPipeCHOP.dll`, `TrackerDAT.dll` and `PuppetCHOP.dll`; click **Allow**. If you
  declined, delete nothing, just restart and allow.
- **A warning that plugins were found twice:** two installs are visible, for example one in
  `Documents\Derivative\Plugins\` and one in a `Plugins` folder next to your `.toe`. Keep one and
  restart.
- **"Model not found ... Looked for models/..."**: the model file is not in the installed
  `models\` folder. For YOLO26, RVM or text to speech, run `fetch_models.cmd`. Then pulse
  **Reload** on the tox.
- **The first cook is slow, or the frame rate is low at first, and then it improves:** the
  TensorRT engine is building ([section 4](#4-speed-and-latency)). Wait a few minutes. If it
  seems stuck, check `trt_building` in the CHOP.
- **The overlay trails the picture:** `result_delay` is above 0. Turn **Wait for Result** on, or
  accept the lag, or use a Cache TOP as described in section 4.
- **Tracking ids swap or flicker:** see [section 3](#3-tracking).
- **No sound from Text to Speech:** **Play Audio** (Output page) uses the default audio device.
  To choose a device, turn it off and wire CHOP 1 into your own Audio Device Out.
- **A tox from an older release misbehaves:** replace the tox files in your project along with
  the plugin folder; each tox expects the plugins of its own version (see its **About** page).

## Hands Geometry

`HandsGeometry.tox` turns a MediaPipe Hands tox's `out_joints` DAT into 3D hands: a sphere per
joint and a cylinder per finger bone, colored from the joint table. Wire `out_joints` into its input.
Outputs: a transparent render (TOP) and the geometry itself as a POP (`out_hands`) for your own
scene. **Aspect Ratio** should be the source image's width / height; **Depth Scale** exaggerates the
(relative, not metric) depth; **Joint Radius** and **Bone Radius** set the thickness; **Camera
Distance**, **Field of View** and **Resolution** control the built-in render. It needs a
TouchDesigner with POPs (2025 and later). Separate hands never connect to each other.

## Puppets

`MixamoPuppet.tox` poses a Mixamo-rigged character from one person's tracking and draws it, either
over the input image or on its own stage. It is single-person: it follows the first track.

**Setup**

1. Put a **MediaPipe Holistic** (or **Pose**) tox on your video and turn on **World Landmark Columns**
   on its Model page. Without them the puppet stays in its rest pose and warns. **MediaPipe Holistic
   GPU** works too (it always outputs world landmarks) and is much cheaper per frame. The puppet
   doesn't use the face: turn off **Face Mesh** (Tracking page on Holistic, Model page on Holistic
   GPU) and the tracker costs about a quarter as much.
2. Drop `MixamoPuppet.tox`. Wire the same video into its first input and the tracker's
   `out_tracks` DAT into its second.
3. Set **FBX File** to a Mixamo character (FBX with skin). The tox ships without one: Mixamo's license
   allows using a character in your project, not redistributing it. The character imports and binds
   itself; **Status** says what happened (for example "65 bones bound"). **Load Character** repeats it.

**Outputs:** TOP 1 is the character on a transparent background; TOP 2 is the character over the
input image (Over Image) or the render alone (Center); the CHOP carries every bone's rotation
(`<bone>_rx/ry/rz`) plus `root_tx/ty/tz`, for your own rig or render.

**Parameters**

| Parameter | What it does |
|---|---|
| FBX File / Load Character / Status | The character, and what the last load did |
| Placement | **Over Image**: drawn over the person in the picture, by a camera that matches Image FOV. **Center**: stays at the origin, seen by a fixed camera, at Center Resolution |
| Image FOV | Horizontal field of view assumed for the input camera. It is a guess per video: if the character looks too big or small, or drifts as the person moves in depth, adjust it |
| Mirror | Center only: swap left and right, for a mirrored (selfie) camera image |
| Placement Smoothing | Smooths the character's position over the image (depth is the noisiest part); 0 = raw |
| Capture Rest Pose | Re-reads the character's rest pose; only needed if you edit the rig inside `fbx` |

**Smoothing page**

| Parameter | What it does |
|---|---|
| Landmark Cutoff | Smooths the tracking before the character is posed (a One Euro filter, like the Filter CHOP's). Lower is calmer but slower to follow; 0 turns it off |
| Speed Coefficient | How quickly the smoothing lets go when the person moves fast. Raise it if quick moves lag, lower it if a moving person looks jittery |
| Slope Cutoff | Smooths the speed the filter reacts to; rarely needs changing |
| Hand Roll Cutoff | Extra smoothing for the twist of the forearms, hands and fingers, the noisiest part of the tracking. The twist also holds while a hand is seen edge-on or not at all. 0 turns it off |
| Knee Limits | Knees bend only like knees, so bad depth can't bend them backward or sideways |
| Relax Unseen Limbs | For a person framed from the chest up (a desk webcam): joints the tracker reports as unseen relax instead of jumping to the character's rest pose. The hips go under the shoulders, so the torso still turns, the arms hang and the legs stand straight |

**Notes**

- Fingers follow Holistic's hands; when a hand isn't seen they hold their last pose.
- The solver needs a Mixamo skeleton (`mixamorig_` bones). Other humanoid rigs aren't supported yet.
- Elbows have no limits yet; the rigged-hand variant and a webcam Mirror toggle are open items in
  `TODO.md`.
- How the solver turns landmarks into bone rotations, and its known limits:
  [PUPPETS.md](PUPPETS.md).

