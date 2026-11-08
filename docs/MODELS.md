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

The plugins report raw detections each frame; Python inside each tox (`python/object_tracker.py`
and the `*_tracker_pipeline.py` modules) turns them into tracks with stable ids. It's
ByteTrack-style, with changes that have worked well for live installations:

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

## Hardware

Both plugins need Windows and an NVIDIA GPU. **ONNX** models run on the GPU through
TouchDesigner's ONNX Runtime with TensorRT. **MediaPipe** models run Google's MediaPipe on the CPU,
on a worker thread, so their results arrive one frame late.

The first time an ONNX model runs, it starts on CUDA right away and builds a TensorRT engine in
the background (a few minutes, lower frame rate meanwhile). The engine is cached in `trt_cache/`
next to the model, so later loads take seconds.

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
(`jawOpen`, `eyeBlinkLeft`, ...) as columns; **Matrix** adds the face's 4×4 transform. MediaPipe,
Apache-2.0.

### MediaPipe Face Detector (`MediaPipeFaceDetector.tox`)
Fast face boxes with 6 keypoints (eyes, nose, mouth, ears), up to 4 faces. Use it when you need
where faces are, not their shape. MediaPipe (BlazeFace short range), Apache-2.0.

### MediaPipe Holistic (`MediaPipeHolistic.tox`)
One person: body, both hands and the face mesh in one model (75 tracked joints plus face).
**Blendshapes** adds the face expression scores. Heavier than Pose. MediaPipe, Apache-2.0.

### MediaPipe Segmentation (`MediaPipeSegment.tox`)
A person/background matte. TOP 1 is the mask at the input's resolution. **Segmenter Model**,
**Segment Output** and **Segment Class** let you load another MediaPipe segmenter (e.g. hair or multi-class).
MediaPipe (selfie segmenter), Apache-2.0.

### YOLO26 Object Detection (`Yolo26Detect.tox`)
Boxes for the 80 COCO classes (`class_name` column: person, car, dog, ...). **Class Filter** keeps
one class (-1 = all). GPU. Ultralytics YOLO26s, **AGPL-3.0**: not bundled, run
`fetch_models.ps1` once (see the README).

### YOLO26 Segmentation (`Yolo26Seg.tox`)
Like Object Detection, plus a per-object mask. TOP 1 is the native segmentation matte (at the
model's resolution), TOP 2 the input, TOP 3 the box overlay. `mask_area_ratio` and `presence`
columns per track. GPU. Ultralytics YOLO26s-seg, **AGPL-3.0**: run `fetch_models.ps1`.

### YOLO26 Pose (`Yolo26Pose.tox`)
Many people at once, 17 COCO body keypoints each. GPU, and much cheaper per person than MediaPipe
Pose in crowds. Ultralytics YOLO26n-pose, **AGPL-3.0**: run `fetch_models.ps1`.

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

### HSEmotion Face Emotion (`HSEmotion.tox`)
Finds faces with a built-in YuNet and adds emotion scores to each face track: `emotion_label`
plus one `emotion_<name>` score for anger, disgust, fear, happiness, neutral, sadness and
surprise. **Classify Interval** runs the classifier every Nth frame to save GPU time;
**Class Score Smoothing** steadies the scores. GPU. HSEmotion enet_b2_7, Apache-2.0.

## VisionDemo (`VisionDemo.tox`)

All 13 models in one COMP for trying them side by side. Wire a video TOP in and switch models on
its **Models** page. A model that's off doesn't cook at all. TOP 1 is every enabled model's
overlay composited together (image results behind the overlays); TOP 2 is the input.

## Audio models

Coming in v0.2. The audio toxes run on the CPU through `MediaPipeCHOP.dll`:
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

## AudioDemo (`AudioDemo.tox`)

The three audio models together. Wire an audio CHOP into input 1 and, optionally, a DAT of lines to speak into input 2. Switch models on its **Models** page; a model that's off doesn't cook at all.

**Speak What It Hears** sends each sentence speech to text hears to text to speech. Use headphones for it: with speakers the mic hears the voice, and it repeats itself.

Outputs:
- **TOP 1:** the classifier's top results over the captions.
- **CHOP 1:** the voice.
- **DATs:** the transcript and the top sounds.
