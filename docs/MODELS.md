# Models

Every model is a `.tox` in `tox/`. Drag one into a network, wire a video TOP into its input, and
it runs (**Active** is on by default). All of them share the same outputs and parameter pages, so
switching models doesn't change how you wire them up.

## Shared outputs

| Output | Content |
|---|---|
| TOP 1 | The debug overlay: boxes and skeletons on a transparent background, sized like the input. Image models (segmentation, depth) output their result image here instead. |
| TOP 2 | The input image, unchanged. Composite TOP 1 over it to see the overlay in place. |
| DAT 1 `tracks` | One row per tracked person, face, hand or object (see below). |
| DAT 2 / 3 `joints` / `bones` | Keypoints and skeleton connections, in the format the debug visualizers use. Landmark models only. |
| CHOP | Per-track numbers as `t<id>_<column>` (e.g. `t1_cx`, `t1_gesture_onset`) plus the model's status and timing (`loaded`, `infer_ms`, `num_detections`, ...). |

The `tracks` table always starts with `track_id, label, score, cx, cy, w, h, x_left, x_right,
y_top, y_bottom, vx, vy, lost_frames, total_frames`, followed by model-specific columns and one
`<joint>_x, <joint>_y, <joint>_conf` triple per keypoint. Positions are 0–1 across the image with
the origin at the **bottom left** (TouchDesigner's UV convention). A track id stays with the same
person or object while it's visible and through short dropouts.

**Depth (MediaPipe Hands, Pose, Holistic, Face):** with **Landmark Z** on (Model page), each
keypoint also gets a `<joint>_z` column, a `t<id>_<joint>_z` channel and a filled `z` in `joints`.
- It's on by default for Hands, Pose and Holistic. It's off for Face, where it would add 478 columns per face.
- MediaPipe's z is *relative* depth: 0 at the wrist (hands) or the hips (pose), negative toward the camera, in roughly the same units as x.
- It's good for "which finger is in front" and for gesture depth. It isn't a distance from the camera.

**3D world positions (MediaPipe Pose, Holistic):** with **World Landmark Columns** on (Model page,
off by default), each body and hand joint also gets `<joint>_wx, <joint>_wy, <joint>_wz` columns
and `t<id>_<joint>_wx/_wy/_wz` channels.
- These are real 3D positions in **metres**, with the origin between the hips. Axes follow
  TouchDesigner: x to the image's right, y up, +z toward the camera.
- They're smoothed with **Output Smoothing**, like the keypoints. A Holistic hand that drops out
  holds its last position.
- In Holistic the hands share the body's space: each hand's wrist sits on the body's wrist. The
  face mesh has no world positions.
- Use them to drive a 3D skeleton or a rigged character. They're estimated from one camera, so
  depth is the noisiest axis.

## Shared parameters

- **Model** page: **Active**, the handful of settings that matter for that model, and **Reload**.
- **Tracking** page: how detections become stable tracks. **Confidence Threshold** (ignore
  weaker detections), **Track Loss Frames** (how long a lost track is held), **Output Smoothing**
  (0 = raw), plus model-specific settings.
- **Debug** page: **Debug Overlay** on/off (or **Show Result** on image models; off outputs
  transparent black and skips the drawing work), and on landmark models **Overlay Layers** (Both,
  Bounding Boxes or Joints), **Joint Size** and **Joint Outline** (a white ring around each joint).
- **About** page: the package version.

## Tracking

The plugins report raw detections each frame; the Tracker DAT inside each tox (`TrackerDAT.dll`,
C++) turns them into tracks with stable ids, a fraction of a millisecond per frame for most models.
It's ByteTrack-style, with changes that have worked well for live installations:

- **Boxes** (YOLO Detection/Segmentation, YuNet): ByteTrack. A Kalman filter predicts each
  track's box, and detections are matched by box overlap in two passes: confident detections
  first, then weaker ones that can only extend existing tracks. Segmentation adds a presence
  ramp; YuNet adds a head-pose estimate.
- **Skeletons** (every pose, hand and face-landmark model): matched by how far the stable
  keypoints (torso, wrist and palm, eyes and nose; not fast-moving extremities) moved, not by box
  overlap, because limbs make boxes jump. By default the distance is measured relative to the
  skeleton's size, so small distant and large close-up people use the same setting. Other
  additions: duplicate suppression, **Coast While Lost** (a lost track keeps drifting along its last
  motion, so a fast hand that MediaPipe drops for a frame keeps its id), presence fade in/out,
  and debounced gestures with a one-frame onset.

Tuning, if ids swap or tracks break:
- **Relative Match Distance** (default 0.15 for bodies, 0.35 for hands): how far the stable
  keypoints may move between frames, as a fraction of the skeleton's size. Lower it if
  neighbours swap ids; raise it if fast movement keeps creating new ids. Turning **Scale Match
  To Skeleton** off (inside the tox's `tracker`) restores the original absolute **Max Match
  Distance**.
- **Track Loss Frames**: longer bridges occlusions but keeps a departed person's ghost longer.
  After switching the video source, old tracks linger for this many frames.
- **Output Smoothing**: steadier keypoints at the cost of lag.

**Joints outside the picture.** Body models report every joint even when the person is framed from
the chest up, and each model fails differently:

| Model | Out-of-frame knees and ankles | Their confidence |
|---|---|---|
| MediaPipe Pose / Holistic | Placed below the image | Often high (0.9 at the 90th percentile) |
| YOLO26 Pose | Squashed onto the bottom edge | Mostly near 0 |
| MoveNet | Squashed onto the bottom edge | Mostly near 0, sometimes 0.65 |

The tracker outputs them as unseen (`conf` 0, left out of the joints and bones tables) when they
fall under **Joint Confidence Threshold** or lie outside or on the image's edge (**Hide Off-Image
Joints**). On the legs, a joint whose parent is unseen is unseen too. On the arms, a seen wrist keeps
its elbow, unless the elbow sits on the image's edge.

Default thresholds, from the visible joints' scores:

| Model | Visible joints score | Default |
|---|---|---|
| YOLO26 Pose | 0.6+ | 0.3 |
| MoveNet | 0.26+ | 0.2 |
| MediaPipe | about 1 | 0.5 |
| RTMW3D | uncalibrated | edge check only |

On a recorded chest-up clip the out-of-frame knees and ankles are drawn on 0-7% of frames (all of
them before), while legs on full-body clips stay visible. Matching still uses every joint, so ids
are unaffected.

## Hardware

Both plugins need Windows and an NVIDIA GPU. **ONNX** models run on the GPU through
TouchDesigner's ONNX Runtime with TensorRT. **MediaPipe** models run Google's MediaPipe on the CPU,
on a worker thread, so their results arrive one frame late.

The first time an ONNX model runs, it starts on CUDA right away and builds a TensorRT engine in
the background (a few minutes, lower frame rate meanwhile). The engine is cached in `trt_cache/`
next to the model, so later loads take seconds.

## Choosing between similar models

Times are per frame for one person (or one frame of detections): GPU models on an RTX 4090 Laptop
GPU with TensorRT, MediaPipe (CPU) models on an i9-13900H laptop. MediaPipe models run on a worker
thread, so with **Wait for Result** off (their default) they cost TouchDesigner's main thread little
but deliver results a frame or more late; the CPU times below are what a cook pays if it waits.

**Licenses decide some of these.** The YOLO26 models are **AGPL-3.0** and Robust Video Matting is
**GPL-3.0**: both are fetched, never bundled, and carry obligations if you distribute a project
using them. **MediaPipe Holistic GPU** uses YOLO26 to find people, so the same applies to it.
Everything else is Apache-2.0 or MIT.

### Bodies and skeletons

| You need | Use | Why |
|---|---|---|
| A rigged character or 3D joints (puppets, metric positions) | **MediaPipe Holistic GPU** (~1.5 ms in the background, one frame late; 5.7 ms waiting) | Metric 3D world landmarks for body and hands, several people, the same table as the CPU version. Needs YOLO26 |
| The same without YOLO26 / AGPL, or with blendshapes | **MediaPipe Holistic** (CPU, ~22 ms if waiting) | One person. The CPU wrapper defaults to not waiting, so it runs a frame or more behind |
| Body only, with 3D | **MediaPipe Pose** (CPU, ~12 ms) | 33 points including feet and hand ends, world landmarks, a few people. No AGPL |
| Many people in 2D (crowds, audiences) | **YOLO26 Pose** (GPU, ~2 ms) | 17 COCO keypoints for everyone in frame, cost barely grows with people. AGPL |
| A few people in 2D, Apache-licensed | **MoveNet MultiPose** (GPU, ~1.5 ms) | 17 keypoints, up to 6 people, no AGPL |

Only MediaPipe Pose and Holistic (CPU or GPU) give 3D world landmarks; YOLO26 Pose and MoveNet
are image coordinates only, with no depth. MediaPipe Pose has 33 points, the 17-point models skip
the hand and foot ends.

### Hands

| You need | Use | Why |
|---|---|---|
| Fingers, gestures, hands on their own (a table, a close-up) | **MediaPipe Hands** (CPU, ~7-10 ms for one hand) | 21 joints with depth, Google's 7 gestures with onsets, finds hands without a body |
| Hands that belong to a body (which person, left vs right) | **MediaPipe Holistic GPU** / **Holistic** | Each hand is found from its arm, so sides and ownership are known; hands share the body's 3D space |
| Drawing tracked hands in 3D | **Hands Geometry** on either of the above | Spheres and cylinders from a joints table |

### Faces

| You need | Use | Why |
|---|---|---|
| Where the faces are, fast, plus head direction | **YuNet** (GPU, ~1.2 ms) | Boxes, 5 points, `yaw` / `pitch` / `roll`, many faces |
| Face boxes on the CPU | **MediaPipe Face Detector** (CPU, ~1.5 ms) | 6 points, up to 4 faces; the GPU stays free |
| Expressions, face shape, face tracking for effects | **MediaPipe Face Landmarks** (CPU, ~5.6 ms) | 478-point mesh, 52 blendshapes (smile, blink, jaw open), a 4×4 head transform |
| Emotion per face | **HSEmotion** | 7 emotion scores per tracked face (uses YuNet inside); raise **Classify Interval** to save time |
| A face mesh together with the body | **MediaPipe Holistic GPU** / **Holistic** | The 478-point mesh per person; blendshapes only from the CPU version |

### Objects

| You need | Use | Why |
|---|---|---|
| What's in the frame and where (80 everyday classes) | **YOLO26 Object Detection** (GPU, ~2.5 ms) | Boxes and labels with tracking; **Class Filter** keeps one class |
| The same plus each object's exact shape | **YOLO26 Segmentation** (GPU, ~2.4 ms) | A mask per object; **Instances** keeps overlapping people apart, with a color per track |

Both are AGPL-3.0; there's no Apache-licensed object detector in the set yet.

### Cutting a person out (mattes)

| You need | Use | Why |
|---|---|---|
| A good matte for most live work | **MODNet** (GPU, ~2 ms) | Hair detail; can shimmer under fast motion |
| A steady matte on fast movement | **Robust Video Matting** (GPU, CUDA) | Carries the previous frames, so it holds up under motion blur. GPL-3.0 |
| The cleanest edges | **DAViD Matte** (GPU, ~10 ms) | Sharpest hair and limbs; costs 5× MODNet |
| Each person separately (overlaps, one person's mask) | **YOLO26 Segmentation**, Instances | Per-person masks with track ids. AGPL |
| A matte with the GPU busy | **MediaPipe Segmentation** (CPU, ~4.5 ms) | Selfie segmenter; other MediaPipe segmenters load too (hair, multi-class) |

### Depth and shape

| You need | Use | Why |
|---|---|---|
| Depth of a whole scene | **Depth Anything V2** (GPU, ~5 ms at 518²) | Everything in frame, near = bright |
| A person's depth, normals and matte | **DAViD Depth + Normals + Matte** (GPU, ~21 ms) | Finer body shape; the background is meaningless |
| A person's surface normals (relighting) | **DAViD Normals** (GPU, ~10 ms) | Real surface direction, see below |
| 3D joints rather than surfaces | **MediaPipe Holistic** / **Pose** | Metric world landmarks |

**DAViD, one model or three:** one map is cheapest from its own model (Normals or Matte, about 10 ms
each). Two or three maps are cheaper from **DAViD Depth + Normals + Matte** (21 ms for all three) than
from separate models, and depth only comes from that one.

**Normals from DAViD vs a shader:** a shader (TD's Normal Map TOP, or a normal-map generator tox) turns
an image's brightness into a height field and takes its slope. It's nearly free, but it measures
lighting and texture, not shape: a dark shirt pattern becomes bumps, a shadow becomes a dent, and a
flat-lit face comes out flat. DAViD estimates the actual surface direction of the body (the round of a
cheek, the turn of an arm, folds in clothing), the same whatever the lighting or color. Use the shader
for texture detail and stylized embossing; use DAViD when you need the body's real shape, e.g. to
relight a performer. They also combine: DAViD for the form, a shader layer for fine detail.

## The models

### MediaPipe Hands (`MediaPipeHands.tox`)
Up to N hands (default 2) with 21 landmarks each. **Task** switches between Hands and Gesture
(default): Gesture adds `gesture`, `gesture_score` and a one-frame `gesture_onset` column for
Google's 7 gestures (`Thumb_Up`, `Victory`, `Open_Palm`, ...). Gestures are debounced by
**Gesture Hold Frames** / **Gesture Min Score**. MediaPipe, Apache-2.0.

### MediaPipe Pose (`MediaPipePose.tox`)
Up to N people (default 2), 33 body landmarks each (`pose_landmarker_full`). MediaPipe,
Apache-2.0.

### MediaPipe Face Landmarks (`MediaPipeFace.tox`)
Up to N faces (default 2) with the 478-point face mesh. **Blendshapes** adds 52 expression scores
(`jawOpen`, `eyeBlinkLeft`, ...) as columns. **Eye Gaze** adds `gaze_x` / `gaze_y`, where the eyes
look within the head, from the irises against the eye corners: in half eye-widths, +x toward the
image's right, +y up. They're smoothed with the tracks. **Matrix** adds the face's 4×4 transform.
MediaPipe, Apache-2.0.

### MediaPipe Face Detector (`MediaPipeFaceDetector.tox`)
Fast face boxes with 6 keypoints (eyes, nose, mouth, ears), up to 4 faces. Use it when you need
where faces are, not their shape. MediaPipe (BlazeFace short range), Apache-2.0.

### MediaPipe Holistic (`MediaPipeHolistic.tox`)
One person: body, both hands and the face mesh in one model (75 tracked joints plus face).
**Blendshapes** adds the face expression scores, **Eye Gaze** the face's `gaze_x` / `gaze_y`. Heavier than Pose. **Face Mesh** (Tracking page)
off leaves the face out of the tables: MediaPipe still runs it, but the tracker drops from about
1.4 ms to 0.5 ms per frame. MediaPipe, Apache-2.0.

### MediaPipe Holistic GPU (`MediaPipeHolisticGPU.tox`)
The same models as MediaPipe Holistic, converted to ONNX and run on the GPU by OnnxTOP, with
Holistic's cropping logic reimplemented in the plugin. Same table, so it's a drop-in for the
CPU wrapper, including for Mixamo Puppet. It runs in the background by default (**Wait for Result**
off): about 1.5 ms of TouchDesigner's main thread per frame, plus the tracker's share (the C++
Tracker DAT: about 1.3 ms with the face mesh, 0.4 ms without), with results one frame late, where the CPU wrapper runs two
or more frames behind and skips frames. With Wait for Result on, results match the frame exactly
for 5.7 ms (RTX 4090 Laptop GPU, one person). It can track several people (**Max Detections**): a
nested YOLO26 detector finds them, then each is tracked from its own landmarks, and the detector only
runs while someone is missing. **Hands** and **Face Mesh** switch those parts off to save time;
Face Mesh off also leaves the face out of the tables. No blendshapes. It
agrees with CPU Holistic to a few pixels (face mesh under 1 px on a still photo). It doesn't filter
the landmarks itself the way MediaPipe does, so the tracker's **Output Smoothing** defaults to 0.4
here (0 on the CPU wrapper): about as steady as CPU Holistic on the body and steadier on the hands,
for about two thirds of a frame of lag. Lower it for faster response. Needs YOLO26
(**AGPL-3.0**: `fetch_models.cmd`) for the detector; the converted models are Apache-2.0.

### MediaPipe Segmentation (`MediaPipeSegment.tox`)
A person/background matte. TOP 1 is the mask at the input's resolution. **Segmenter Model**,
**Segment Output** and **Segment Class** let you load another MediaPipe segmenter (e.g. hair or multi-class).
MediaPipe (selfie segmenter), Apache-2.0.

### YOLO26 Object Detection (`Yolo26Detect.tox`)
Boxes for the 80 COCO classes (`class_name` column: person, car, dog, ...). **Class Filter** keeps
one class (-1 = all). GPU. Ultralytics YOLO26s, **AGPL-3.0**: not bundled, run
`fetch_models.cmd` once (see the README).

### YOLO26 Segmentation (`Yolo26Seg.tox`)
Like Object Detection, plus a per-object mask. TOP 1 is the native segmentation matte (at the
model's resolution), TOP 2 the input, TOP 3 the box overlay. `mask_area_ratio`, `presence` and
`det_index` columns per track. GPU. Ultralytics YOLO26s-seg, **AGPL-3.0**: run `fetch_models.cmd`.

- **Mask Box Margin** (Model page) lets each mask fade out past its box instead of being cut at it. Tight boxes otherwise clip hair and raised fingertips in blocky steps. 0 restores the hard crop.
- **Matte Encoding = Instances** keeps the identity of every masked pixel. TOP 1's channels become:
  - **R and A:** the matte, as before.
  - **G:** the detection that owns the pixel. That's the Info DAT `id` + 1, and 0 means nobody. Where people overlap, the pixel goes to the most confident mask.
  - **B:** that detection's class + 1.
- **Matte NMS IoU** (default 0.5) leaves a mask out of the matte when its box nearly duplicates a more confident detection's. The model sometimes boxes one person twice; the tracker already drops the copy, and this keeps the copy from claiming half that person's pixels in Instances mode. Keep it equal to the Tracking page's NMS setting.
- **TOP 4, `instances`,** paints each tracked object's pixels in its track color, the same colors as the box overlay. It's transparent while the encoding is Grayscale.

Detection ids change order from frame to frame; the stable id is the track's. A track's `det_index` is the detection it matched this frame, or -1 while it's coasting. To isolate one track, look up its `det_index` in the tracks table and keep the pixels whose G equals `det_index + 1`. For example, in a GLSL TOP on TOP 1, with a uniform `uOwner` set to that number:

```glsl
out vec4 fragColor;
uniform float uOwner;	// the track's det_index + 1 (0 = nothing)
void main()
{
	vec4 m = texture(sTD2DInputs[0], vUV.st);
	float keep = (uOwner > 0.0 && abs(m.g - uOwner) < 0.5) ? m.a : 0.0;
	fragColor = TDOutputSwizzle(vec4(keep));
}
```

Set the GLSL TOP's Input Smoothness to Nearest, so ids aren't blended between neighboring pixels.

### YOLO26 Pose (`Yolo26Pose.tox`)
Many people at once, 17 COCO body keypoints each. GPU, and much cheaper per person than MediaPipe
Pose in crowds. Ultralytics YOLO26n-pose, **AGPL-3.0**: run `fetch_models.cmd`.

### MoveNet MultiPose (`MoveNet.tox`)
Up to 6 people, 17 keypoints each. GPU. **Dynamic Size** sets the inference resolution (multiple
of 32; larger is more accurate and slower). Google MoveNet MultiPose Lightning, Apache-2.0.

### YuNet Face Detection (`YuNet.tox`)
Face boxes with 5 keypoints, plus a head pose estimate (`yaw`, `pitch`, `roll`; **Head Pose
Method** picks the estimator). GPU, very fast. OpenCV Zoo YuNet, MIT.

### Depth Anything V2 (`DepthAnything.tox`)
Relative depth for the whole frame (near = bright). TOP 1 is the depth image at the input's
resolution. **Dynamic Size** sets the inference resolution (multiple of 14); **Normalize** and
**Range Smoothing** keep the brightness range steady between frames. GPU. Depth Anything V2 Small, Apache-2.0.

### MODNet Portrait Matting (`MODNet.tox`)
Alpha matte of the person in frame (white = person), hair detail included. TOP 1 is the matte at
the input's resolution; use it as a mask over the input. **Dynamic Size** sets the inference
resolution (multiple of 32; 512 x 288 for 16:9 is about 2 ms with TensorRT). GPU. MODNet, Apache-2.0.

### NeuFlow v2 Optical Flow (`NeuFlowV2.tox`)
How every pixel moved since the previous frame. TOP 1 is the flow in the same format as
TouchDesigner's **Optical Flow TOP**, so it drops into networks built for that (a Displace TOP, the
particlesGpu component's forces):
- R is the horizontal motion and G the vertical, 32-bit float.
- Both are in screen widths per second; G is in widths too, so on a 16:9 frame 1.0 is 1.78 frame
  heights per second.
- Right and up are positive.
- **Gain X / Gain Y** (Model page) multiply them, like the Optical Flow TOP's Gain.

Per second uses the measured time between the two frames compared (the plugin's `pair_interval`,
averaged), so a 30 fps video or webcam in a 60 fps project gives true speeds. A repeated frame keeps
showing the last motion; a still image shows none.

- **Cut Threshold** (Model page, default 0.08): a scene change, or a video jumping back to its start,
  restarts the comparison instead of flashing huge motion. It's the mean difference between two
  frames (0-1): moving footage changes up to about 0.04 a frame, a cut 0.1-0.4. 0 turns it off.
- **Flow Smoothing** (Model page, default 0.3): blends each new flow with the last; 0 is raw.
- **Noise Floor** / **Floor Ease** (Model page, defaults 0.02 / 0.03 screen widths per second): the
  model reports a little motion everywhere (about 0.01-0.05 on a still background). Speeds below
  the floor become zero, and above it the flow fades back in over the ease, so slow real motion
  doesn't pop on. 0 / 0 is the raw flow.

TOP 3 is the flow scaled for viewing: R = x, G = y, signed (32-bit float: left and down are
negative), 0 where still. **Color Full Speed** (Debug page) is the speed, in screen widths per
second, that reads as 1.

Unlike the Optical Flow TOP, it isn't limited to newer NVIDIA cards' hardware flow. The model works
on a 768 x 432 version of each frame: 5 ms with TensorRT, about 1.5 ms of TouchDesigner's main
thread. **Reset State** forgets the previous frame, e.g. after a cut. GPU. NeuFlow v2, Apache-2.0.

### Robust Video Matting (`RVM.tox`)
Alpha matte of the person in frame, built to stay steady from frame to frame: the model carries
state from one frame to the next, so it holds up under motion blur and movement better than a
per-frame matte such as MODNet. TOP 1 is the matte at the input's resolution. **Dynamic Size** sets
the inference resolution (multiple of 32); **Scalar Input** is RVM's downsample ratio (0.25 for HD,
0.5 for SD); **Reset State** forgets the previous frames, e.g. after a cut. GPU (CUDA). RVM is
**GPL-3.0**, so it isn't in the zip: run `fetch_models.cmd`.

### DAViD Normals, Matte (`DAViDNormals.tox`, `DAViDMatte.tox`)
Microsoft's human-specific models, trained on synthetic people. TOP 1 at the input's resolution:
- **Normals:** the person's surface normals as a normal map (0..1, i.e. the -1..1 vectors × 0.5 +
  0.5): the shape of the face, clothing folds and limbs, for relighting a performer or driving shaders.
- **Matte:** a soft alpha matte with hair detail.

About 10 ms each (RTX 4090 Laptop GPU, TensorRT). The background is meaningless in the normals: mask
them with a matte. MIT, but 2.3 GB for the DAViD set, so it isn't in the zip: run `fetch_models.cmd`.

### DAViD Depth + Normals + Matte (`DAViDHuman.tox`)
All three from one larger model (ViT-L) in one inference, about 21 ms. TOP 1 is the person's relative
depth (near = bright; **Range Smoothing** steadies its range), TOP 3 the normals (0..1 normal map),
TOP 4 the matte; TOP 2 is the input as usual. This is also the DAViD depth to use: the separate
ViT-B depth model squeezes its output into a range too narrow for TensorRT's half precision (it comes
out banded) and costs 43 ms at full precision. MIT, `fetch_models.cmd`.

### HSEmotion Face Emotion (`HSEmotion.tox`)
Finds faces with a built-in YuNet and adds emotion scores to each face track: `emotion_label`
plus one `emotion_<name>` score for anger, disgust, fear, happiness, neutral, sadness and
surprise. **Classify Interval** runs the classifier every Nth frame to save GPU time;
**Class Score Smoothing** steadies the scores. GPU. HSEmotion enet_b2_7, Apache-2.0.

## Audio models

The audio toxes run on the CPU through `MediaPipeCHOP.dll`:
- **Speech to text and the sound classifier** take an audio CHOP: Audio Device In, Audio File In, and so on. Any sample rate works, and channels are mixed to mono.
- **Text to speech** takes text.

They keep cooking every frame by themselves, so nothing is missed when nothing downstream reads their outputs.

### MediaPipe Speech to Text (`MediaPipeSpeechToText.tox`)
Streaming speech to text with closed captions.

| Output | Content |
|---|---|
| TOP 1 | Closed captions: the newest words in a dark strip along the bottom of a transparent image (**Captions** page sets its size). Composite it over your video. |
| CHOP | `speaking` (1 while speech is detected), `level` / `threshold` (input RMS and the level that counts as speech), `segments` (sentences so far), `new_segment` (1 on the cook a sentence is finalized; trigger on it), `backlog` (seconds of audio waiting; above ~1 the model is too slow), plus `loaded`, `loading`, `infer_ms`. |
| DAT 1 `transcript` | `segment, final, start_ms, end_ms, text`: the last 200 sentences, oldest first, then the current interim guess (`final` 0). `segment` keeps counting; times are milliseconds since the model started. |
| DAT 2 `latest` | One row: the newest sentence (`segment, start_ms, end_ms, text`) and the current `interim` guess. |

**Model** page:
- **Transcriber Model** picks the speech model:
  - `models/asr/moonshine_tiny_5s_i8.litertlm` (default, fast, shows words while you speak).
  - `whisper_tiny_30s_i8` (more accurate, sentence at a time).
  - Larger models such as Parakeet or Qwen3-ASR can be downloaded from huggingface.co/litert-community.
- **Skip Silence** sends only speech to the model; without it, models tend to invent text from room noise.
- Speech is detected when the input is **Noise Ratio** times louder than the tracked background noise, and at least **Silence Level**:
  - If noise still turns into text, raise Noise Ratio.
  - If quiet speech is missed, lower it.
- A **Pause** of that many seconds ends a sentence; **Max Segment** caps continuous speech.
- **Prompt** gives generative models words to expect.
- **Flush** finalizes what's been heard; **Clear** starts over.

**Captions** page:
- **Max Characters**: the longest caption.
- **New Caption After**: a quiet gap that starts a fresh caption instead of adding to the old one.
- **Hold**: how long a caption stays after the last words.
- **Font Size** and **Resolution**.

**Debug > Show Captions** off skips the caption drawing.

Models: Moonshine Tiny (MIT), Whisper Tiny (Apache-2.0), as LiteRT `.litertlm` bundles.

### MediaPipe Text to Speech (`MediaPipeTextToSpeech.tox`)
Speaks text in a natural voice. Lines are queued and spoken one after another. The first audio
arrives a second or two after a line is queued, since the model synthesizes a sentence at a time.

Give it text in either of two ways:
- Type into **Text** and pulse **Speak** (or turn on **Speak On Change**).
- Wire a DAT into the input. Each time its newest row changes, that row is spoken: its `text` column, else its last cell. Wiring in MediaPipe Speech to Text's `latest` DAT makes it repeat what it hears.

| Output | Content |
|---|---|
| TOP 1 | A caption of the line being spoken, in the same style as speech to text's. |
| CHOP 1 | The voice: `audio` at 24 kHz, plus `speaking` (1 while a line is being synthesized or played). |
| CHOP 2 | `loaded`, `loading`, `infer_ms` (time for the last audio chunk), `results` (lines spoken), `queued`. |
| DAT `lines` | `line, state, voice, text`: the line being spoken, then the queue. |

**Model** page:
- **Voice** picks one of 54 voices. The first letters give the language and voice type: `af_` American female, `am_` American male, `bf_` / `bm_` British, `ef_` Spanish, `ff_` French, `hf_` Hindi, `if_` Italian, `jf_` Japanese, `pf_` Portuguese, `zf_` Chinese, and so on. Middle-click the inner `inference` node to list them all. The default is `af_heart`.
- **Speaking Rate** sets the speed. Changing it reloads the model, which takes about half a second.
- **Stop** stops speaking and clears the queue.

**Output** page:
- **Speak Input DAT** turns speaking from the wired DAT on and off.
- **Play Audio** and **Volume** play the voice on the default audio device. Turn Play Audio off to route CHOP 1 yourself.

Model: Kokoro-82M (Apache-2.0), as a LiteRT `.litertlm` bundle with its voices and text front ends.
It isn't in the zip (280 MB): run `fetch_models.cmd` once (see the README).

### MediaPipe Sound Classifier (`MediaPipeSoundClassifier.tox`)
Recognizes 521 kinds of sound, from speech and music to dogs, sirens, clapping and engines. Each
classification looks at the last second of audio, and a new one runs every quarter second.

| Output | Content |
|---|---|
| TOP 1 | The top results with score bars. |
| CHOP 1 | Every class's score (0–1), smoothed, one channel per class: `Speech`, `Music`, `Dog`, `Finger_snapping`, ... |
| CHOP 2 | For each class in **Watch Classes**: `<name>` (its score), `<name>_active` (above **Active Threshold**) and `<name>_onset` (1 on the frame it becomes active; trigger on it). Plus `loaded`, `infer_ms`, `results`. |
| DAT `top` | `rank, class, label, score`: the **Top Results** highest scores. |

**Classes** page:
- **Score Smoothing** lags the scores, which steadies them but makes them slower to react.
- **Watch Classes** lists channel names from CHOP 1, separated by spaces.

Model page: **Window** (how much audio each classification hears) and **Hop** (how often one runs).

Model: YAMNet (Apache-2.0).

## Components

Components take models' outputs (tracks, joints, images) and build something from them; there's no
model inside, so you can feed them from any matching model, a recording or your own network. They're
in `tox/components/`.

### Mixamo Puppet (`components/MixamoPuppet.tox`)
Poses a Mixamo character from Holistic or Pose tracking (video + tracks in). See the
[manual](MANUAL.md#puppets) and [PUPPETS.md](PUPPETS.md).

### Hands Geometry (`components/HandsGeometry.tox`)
Draws tracked hands as 3D spheres and cylinders from a MediaPipe Hands joints DAT. See the
[manual](MANUAL.md#hands-geometry).

### Relight (`components/Relight.tox`)
Lights a person with two point lights or spotlights from DAViD's outputs, each light following a
hand. Inputs:
1. the color image (DAViD Depth + Normals + Matte's TOP 2, the input);
2. the normals (its TOP 3, or DAViD Normals);
3. the matte (its TOP 4, or DAViD Matte);
4. the depth (its TOP 1, with **Show Result** on);
5. optional: a hands `tracks` DAT (MediaPipe Hands' DAT 1). Without it the lights stay at their
   positions.

The shading is physically based (GGX highlights, Fresnel, roughness, light falling off with
distance) in linear light, with the color image as the surface color.

- **Lights page:** **Follow Hands** (Light 1 takes the hand MediaPipe calls Right, Light 2 the Left
  one; MediaPipe names hands as in a mirrored selfie image, so swap the colors if yours read
  reversed. Two hands always get a light each), **Follow Lag** (seconds to ease to a hand or back),
  then per light: on/off, **Color**, **Intensity**, **Position** (where it sits with no hand, 0 to
  1, origin bottom left) and **Distance** (how far in front of the person, in image heights: small
  = side light, large = flat front light). **Light Radius** is the falloff distance.
- **Follow** puts a light on the palm's centre or the index fingertip. **Light Type** Spot turns
  both lights into cones aimed along the index finger (knuckle to tip; with no hand, at the image
  centre): **Spot Angle** is the cone's width, **Spot Softness** its edge, and **Aim Into Scene**
  tips the beam from the image plane toward the person (a finger pointing at the camera can't be
  seen, so 0 is exactly along the finger).
- The light colors are shader uniforms (`uLight1Color`, `uLight2Color`: color × intensity) driven
  by the Light Color / Intensity parameters, so animating them, or exporting a CHOP to them, changes
  the light live.
- **Material page:** **Shading** (PBR, or Simple: just the light, its color × how directly the
  surface faces it × falloff, no highlights or Fresnel), **Ambient** (unlit share of the image),
  **Roughness**, **Metallic**, **Specular** (PBR only), **Depth Scale** (how deep the person is),
  **Exposure**, **Background** (brightness of everything else), **Transparent Background** (alpha =
  the matte) and **Show Lights** (a ring at each light).

TOP 1 is the relit image; CHOP 1 the lights' positions and aims (`light1_x`, `light1_y`,
`light1_dx`, `light1_dy`, `light1_hand`: 1 while following a hand). The shading itself is two
small GLSL passes; the cost is in the models feeding it (DAViD Depth + Normals + Matte about 21 ms
of GPU, MediaPipe Hands on the CPU).

## Demos

Demos wire models and components together to show what they can do. They're in `tox/demos/`.

### VisionDemo (`demos/VisionDemo.tox`)

Every video model in one COMP for trying them side by side. Wire a video TOP in and switch models on
its **Models** page. A model that's off doesn't cook at all. TOP 1 is every enabled model's
overlay composited together (image results behind the overlays); TOP 2 is the input; TOP 3 is DAViD
Depth + Normals + Matte's depth on its own (transparent while that model is off).

### AudioDemo (`demos/AudioDemo.tox`)

The three audio models together. Wire an audio CHOP into input 1 and, optionally, a DAT of lines to speak into input 2. Switch models on its **Models** page; a model that's off doesn't cook at all.

**Speak What It Hears** sends each sentence speech to text hears to text to speech. Use headphones for it: with speakers the mic hears the voice, and it repeats itself.

Outputs:
- **TOP 1:** the classifier's top results over the captions.
- **CHOP 1:** the voice.
- **DATs:** the transcript and the top sounds.
