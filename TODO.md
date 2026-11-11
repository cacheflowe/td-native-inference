# TODO: td-native-inference

## Status (Nov 11, 2026)

v0.2.5 published Nov 10: 13 video toxes + VisionDemo, three audio toxes + AudioDemo, the puppet
lab (Mixamo body + rigged hand) in the dev project. Everything is generated from
`wrappers/*.json`; clean-process tests; Apache-2.0 LICENSE and NOTICE; release tooling that
bundles the redistributable models. The previous plan (Package Root, CPlusPlus TOP wrappers) is in
git history before commit 7ddfb48. Performance lessons from v0.2.4/0.2.5:
[docs/learnings/td-cook-performance.md](docs/learnings/td-cook-performance.md).

## Now (Nov 11 plan, in order) -- handoff state

Work since v0.2.5 is **uncommitted** (both plugins' sources and READMEs, the bench tool, the lab
test, `CLAUDE.md`, `TODO.md`, `docs/learnings/`, the `.toe`, the MODNet wrapper). TD was restarted
Nov 11 on the new DLLs and the double-cook fix verified (below).

- [x] Learnings write-up: [docs/learnings/td-cook-performance.md](docs/learnings/td-cook-performance.md)
      (GPU readback stall, Wait for Result / result_delay, cookEveryFrame double cooks, CHOP sample
      rate, cooking-flag leaks, Python-in-TD limits, CUDA graph result).
- [x] Demo model toggles: verified fine as they are (on -> off, and a freshly rebuilt demo, leave
      0 cooks in switched-off wrappers over 60 frames). No code change.
- [x] **Holistic double cooks: fixed and verified Nov 11** (0.6 cooks/frame on the 30 fps dancer clip, was 1.4; full wrapper lab: all 16 detect). Cause: both plugins set
      `cookEveryFrame` while a result was pending (almost always with async readback), which adds a
      cook of its own besides the input-triggered one: 168 cooks in 120 frames at a steady 60 fps.
      Fix in `MediaPipeTOP.cpp` / `OnnxTOP.cpp` `getGeneralInfo()`: `cookEveryFrameIfAsked` only
      (MediaPipe keeps `cookEveryFrame` while loading). To verify after the restart: sample
      `holistic/inference.totalCooks` per frame in `/project1/PuppetLab` (expect 1 per frame on the
      dancer clip), then run the full wrapper lab test (still images: results must still land, so
      every wrapper should still detect). Optionally re-run the Wait for Result off check on an ONNX
      model with a still input.
- [x] Wrapper lab thumbnails: `THUMBNAILS = False` by default in `tests/td_wrapper_lab.py`; set it
      True when preparing a release (noted in CLAUDE.md).
- [x] ONNX Runtime launch cost: measured with `onnxtop_tool --graph` (new flag; the bench now
      prints launch and inference separately). TensorRT CUDA graphs make `Run()` wait for the
      replay, so the launch costs the whole inference: not offered in the plugin. The ~2.5 ms YOLO
      launch is ORT's own CPU cost. New info channel `run_ms` (launch CPU time) on OnnxTOP.
- [ ] **Matting (MODNet done, BiRefNet / RVM open).** MODNet: `wrappers/MODNet.json`, `tox/MODNet.tox`, icon, lab test + thumbnail, README / MODELS.md / manifest entries (Nov 11); still to do: clean-process test and bundling check at release. Downloaded (gitignored) and benchmarked:
  - MODNet, `models/onnx/modnet/modnet.onnx`, from Xenova/modnet on Hugging Face (Apache-2.0,
    25 MB fp32; also `model_fp16.onnx` 13 MB). Input `input` [1, 3, H, W] float32 (dynamic H/W,
    multiples of 32), output `output` [1, 1, H, W] alpha 0-1. MODNet normalizes the input to
    -1..1 (OnnxTOP Input Normalization = Signed; check against the reference). TensorRT fp16 p50:
    1.8 ms at 512x288, 4.6 ms at 1024x576; launch 1.1-1.4 ms. First TensorRT build ~2 min per size.
    Next: a `wrappers/MODNet.json` like DepthAnything's (Raw mode, Dynamic Size, Raw Map To Input
    on, no Raw Normalize since the output is already 0-1), `models/README.md` + `fetch_models.ps1`
    entry (redistributable: can be bundled), then the lab test and a thumbnail.
    Live check (Nov 11): the matte is very good, but MODNet is sensitive to motion blur and moving
    subjects (per-frame, no temporal state). RVM's recurrent state is the fix; a cheap stopgap is
    temporal smoothing of the matte.
  - BiRefNet lite, `models/onnx/birefnet/birefnet_lite.onnx`, from onnx-community/BiRefNet_lite-ONNX
    (MIT, 224 MB fp32 / 115 MB fp16; the full BiRefNet is 973 MB). Higher quality, much heavier
    (1024x1024). The benchmark was interrupted during its first TensorRT build (several minutes):
    re-run `plugins\onnxtop_tool.exe "C:\Program Files\Derivative\TouchDesigner\bin"
    models\onnx\birefnet\birefnet_lite.onnx tensorrt 60` and check its input/output names and
    whether TensorRT accepts the graph. Likely a stills / quality option rather than real time.
- [x] **RVM (Nov 11).** `OnnxEngine` pairs `<x>i`/`<x>o` as recurrent state (zero start, output rebound
      as the next input, no readback), a lone `[1]` float input is the new **Scalar Input**
      (`downsample_ratio`), plus a **Reset State** pulse. `wrappers/RVM.json` (CUDA provider: the state
      changes shape after frame 1, which TensorRT would rebuild for; Raw index 1 = `pha`),
      `fetch_models.ps1` entry behind a GPL prompt (never bundled). Bench: 3.5 ms p50 at 1024x576. On
      the fast les-twins clip it flickers about 2.4x less than MODNet (pixels flipping and flipping
      back: 1.2% vs 2.7%; second difference 0.104 vs 0.171); equal on the calm dancer clip. Edges are
      softer than MODNet's (downsample 0.25). Open: try fp16 / TensorRT with a fixed-size state
      export, `rvm_resnet50`, and use `fgr` (foreground colour) as a second output.
- [x] Puppets: `MixamoPuppet` tox (Nov 11): `python/puppet_builder.py` + `puppet_runtime.py`, `scripts/build_puppet_tox.py`.
      In TOP + tracks DAT in, FBX File par imports and binds the character at runtime (the tox carries none:
      Mixamo license), outputs character render / over image / bone CHOP. Checked: exported tox dropped in
      fresh, FBX set -> 65 bones bound, same placement as the lab. Open: hand-rig variant, joint limits.
- [ ] A short user manual: what each tox does, how to wire it, its main parameters and outputs,
      and how the pieces work (trackers, Wait for Result / result_delay, debug overlays).
- [ ] RTMPose for puppets: plain RTMPose / RTMW is 2D only, which the Mixamo solver can't use. The
      candidate is **RTMW3D** (MMPose projects/rtmpose3d, 2024, Apache-2.0): 133 whole-body
      keypoints in 2D and 3D, multi-person, top-down (person detector, then one run per person,
      384x288, ~65M params for -L). Likely lower CPU load than Holistic (GPU via TensorRT) and
      multi-person, but coarser fingers (hands from the full-body crop) and a 68-point face. No
      official ONNX export: step one is exporting it and timing it with `onnxtop_tool`, before
      any plugin work (SimCC decoder in the crops path, a 133-keypoint name map for the solver).

## Next: more model types

Wrappers come in three inputs: `video` (In TOP; trackers + overlays), `audio` (In CHOP; kinds
`transcript`, `scores`) and `text` (In DAT; kind `speech`), all from `wrappers/*.json`.

- [x] **Audio (v0.2)**: `cpp/MediaPipeCHOP` (Custom CHOP `MediapipeaudioCHOP`) wraps MediaPipe
      1.2.2's audio tasks from `libmediapipe.dll`: Classify Sounds (YAMNet), Transcribe Speech
      (AudioTranscriber, `.litertlm` from litert-community) and Speak Text (TextToSpeech, Kokoro).
      Toxes `MediaPipeSoundClassifier`, `MediaPipeSpeechToText`, `MediaPipeTextToSpeech` and
      `AudioDemo`, lab-tested; offline tools `audio_test`, `asr_test`, `tts_test`.
- [x] v0.2 release (v0.2.0-v0.2.5 published Nov 8-10).
- [ ] Speech to text: try parakeet-tdt (615 MB, CC-BY-4.0) and Qwen3-ASR (960 MB, Apache-2.0);
      a fetch for the large ones. Moonshine sometimes mishears ("pay" for "fade").
- [ ] Text to speech: word timestamps (`MpTextToSpeechComputeWordTimestamps`) for read-along
      captions; a voice menu; Qwen3-TTS or other `.litertlm` voices.
- [ ] Audio embedder (`MpAudioEmbedder*`): same plugin, another task. Merge MpAudio's loader with
      MediaPipeTOP's MpApi into `cpp/common/`.
- [ ] Also exported and cheap to add: MediaPipe text classifier/embedder, language detector,
      object detector, image classifier/embedder, interactive segmenter.
- [ ] Other ONNX candidates: RF-DETR (Apache-2.0); RTMPose / RTMW whole-body pose (133 keypoints;
      top-down: needs person crops, like Classify Crops); MODNet / BiRefNet / RVM matting (above);
      Real-ESRGAN upscaling and fast style transfer (Raw mode); Segment Anything (MobileSAM /
      EfficientSAM: needs prompt inputs) and RAFT optical flow (needs two frames). Check each
      license before bundling.

## Dev vs installed plugins

- [x] Switching: `scripts\plugins.ps1 dev|install|status` (Nov 9) moves the installed binaries
      aside or copies the local build in; models stay put. README says "one install per machine".
- [ ] A tox/plugin version check that warns on a mismatch (an installed dev build still reports
      the last `VERSION`).

## Puppeteering a rigged character (MediaPipe Holistic, single person)

Target: a Mixamo-rigged FBX COMP in TouchDesigner, driven by one tracked person.

- [x] **World landmarks out of the tox** (Nov 9). **World Landmark Columns** on the Pose and
      Holistic Model pages; `<joint>_wx/_wy/_wz` columns and channels in metres, TD axes (y up,
      +z toward the camera), hip origin, smoothed, unseen hands hold. Holistic's hands turned out
      to be in body space already (hand wrist = body wrist), so no attaching step.
- [x] **Bone rotations for Mixamo** (Nov 9, first pass). `python/mixamo_retarget.py` (numpy,
      `tests/test_retarget.py`): per bone a primary direction plus a roll hint, measured the same
      way on the rest skeleton; a weak hint (straight arm) falls back to the parent's turn plus
      the shortest swing; spine and neck blend between hips, chest and head. ~1 ms per frame.
- [x] **Drive the FBX COMP** (lab). `scripts/lab/build_puppet.py` builds `/project1/PuppetLab`:
      clip or webcam -> Holistic -> solver Script CHOP -> every bone's rx/ry/rz bound to it ->
      render. Checked on `build/pose.jpg` (warrior pose matches). Since Nov 10 the solver lives in
      `python/puppet_solver.py`, and the lab has controls on its Puppet page and two paths (body,
      hand) in labelled network boxes.
- [x] **Over Image placement** (Nov 10). `mixamo_retarget.place()`: the root translation that
      lays the posed model's joints over the 2D keypoints, by least squares, for a camera at the
      origin with an assumed horizontal FOV (Image FOV). Smoothed (Placement Smoothing).
- [x] **Rigged hand** (Nov 10). `python/hand_retarget.py` maps MediaPipe hand landmarks onto a
      Blender-style hand rig (handRig_02.fbx: `pulse`, `hand`, `<finger>_base/_01.._03`), with
      its own close-up clip and MediaPipe Hands tracker in the lab. Either hand drives the one-hand
      model (the other side mirrored, sx = -1). Tests: `tests/test_hand_retarget.py` (needs
      `build/hand_rest.json`, dumped from the lab's hand FBX COMP).
- [ ] Hand: on a mirrored (selfie) webcam image the left/right labels name the mirrored hand; the
      model then mirrors the right way for the image, but Tracked Hand Right/Left picks the other
      hand. A Mirror toggle could swap them.
- [ ] Over Image: the FOV is a guess per video; could be estimated from the person's known
      proportions over a few frames.
- [ ] Tune live with a webcam: head yaw/pitch (looks under-turned), whether the forearm should
      take only part of the hand's twist, foot roll.
- [ ] Rotation smoothing (slerp) and joint limits (no backward elbows/knees); for now only the
      tracker's position smoothing. The hand rig already has finger hinge limits
      (`hand_retarget.FINGER_LIMITS`).
- [ ] Full pose model in Holistic: repack `holistic_landmarker.task` with
      `pose_landmarker_full`'s landmark model (Holistic ships the lite one); steadier, ~2x slower.
- [x] Hips translation: Over Image placement (above) moves the character with the person.
- [ ] Side-on people: MediaPipe swaps left and right, turning the character 180°. Maybe reject
      a frame whose hip line flips faster than a person can turn.
- [x] Package (done, see Now): a `MixamoPuppet` tox generated like the wrappers. Easiest use: drop it in, set
      its FBX file (any Mixamo character) and point it at a Holistic / Pose tox; it imports the
      FBX inside, captures the rest pose, binds the bones and renders (or exposes the FBX COMP for
      your own render). Pars: Mirror, Placement (Center / Over Image), Image FOV, smoothing,
      joint limits. A hand variant for Blender-style hand rigs.
- [ ] Face: Holistic's blendshapes could drive a Mixamo face only if the character has matching
      shape keys (ARKit names); out of scope until a character needs it.

## Tracking (area of interest)

- [ ] The Python tracker (`python/object_tracker.py`, `*_tracker_pipeline.py`) is ByteTrack-based
      with our own changes (keypoint-distance matching, scale-relative matching, coast, presence,
      gesture debounce). Revisit after real-world testing of the scale-relative default; open
      questions to be collected from that testing.

## Housekeeping

- [x] Nov 10: async GPU readback in both plugins, Wait for Result / Max Wait, `result_delay`,
      cheaper Info DATs, trackers apply a result once and snap back after a gap, Sound Classifier
      output at the timeline rate, the wrapper lab restores its Cooking flag.

- [x] Retired `/project1/ONNX_Native` and `python/mediapipe_hands_component.py` (Nov 7).
- [x] Deleted the stale `cache/trt/` engine folder (Nov 7); `.gitattributes` added.
- [ ] TensorRT cache: a per-user cache (`%LOCALAPPDATA%`) keyed by model content, for read-only
      installs. Today it's `trt_cache/` next to each model, which is writable in Documents.
- [x] Plugins compile in the version from `VERSION`; `package` checks it.
- [ ] CI build (GitHub Actions, CUDA toolkit).
- [ ] haxlib side: point its scripts at this repo's Python or keep a copy; remove its `cpp/`.
