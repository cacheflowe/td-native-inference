# TODO: td-native-inference

## Status (Nov 7, 2026)

v0.1.0 (13 video toxes + VisionDemo) published Nov 7. In main for v0.2: three audio toxes +
AudioDemo. Everything is generated from `wrappers/*.json`; clean-process tests; Apache-2.0
LICENSE and NOTICE; release tooling that bundles the redistributable models. The previous plan (Package Root,
CPlusPlus TOP wrappers) is in git history before commit 7ddfb48.

## v0.1.0 release (published Nov 7)

- [x] `fetch_models.ps1` downloads the three YOLO26 ONNX files from Hugging Face (byte-identical
      to the files the wrappers were tested with; tested Nov 7). The seg file is from a personal
      repo (zwh20081/yolo26-onnx): if it disappears, host it as a separate AGPL release download or
      export with Ultralytics (the old Python export path is in git history).
- [x] Packaged, clean-tested, verified and published.

## Next: more model types

Wrappers come in three inputs: `video` (In TOP; trackers + overlays), `audio` (In CHOP; kinds
`transcript`, `scores`) and `text` (In DAT; kind `speech`), all from `wrappers/*.json`.

- [x] **Audio (v0.2)**: `cpp/MediaPipeCHOP` (Custom CHOP `MediapipeaudioCHOP`) wraps MediaPipe
      1.2.2's audio tasks from `libmediapipe.dll`: Classify Sounds (YAMNet), Transcribe Speech
      (AudioTranscriber, `.litertlm` from litert-community) and Speak Text (TextToSpeech, Kokoro).
      Toxes `MediaPipeSoundClassifier`, `MediaPipeSpeechToText`, `MediaPipeTextToSpeech` and
      `AudioDemo`, lab-tested; offline tools `audio_test`, `asr_test`, `tts_test`.
- [ ] v0.2 release: `prepare minor`, rebuild the plugins, package (about 540 MB with Kokoro),
      clean-test a video, an audio and the text-to-speech tox, publish.
- [ ] Speech to text: try parakeet-tdt (615 MB, CC-BY-4.0) and Qwen3-ASR (960 MB, Apache-2.0);
      a fetch for the large ones. Moonshine sometimes mishears ("pay" for "fade").
- [ ] Text to speech: word timestamps (`MpTextToSpeechComputeWordTimestamps`) for read-along
      captions; a voice menu; Qwen3-TTS or other `.litertlm` voices.
- [ ] Audio embedder (`MpAudioEmbedder*`): same plugin, another task. Merge MpAudio's loader with
      MediaPipeTOP's MpApi into `cpp/common/`.
- [ ] Also exported and cheap to add: MediaPipe text classifier/embedder, language detector,
      object detector, image classifier/embedder, interactive segmenter.
- [ ] Other ONNX candidates: RF-DETR (Apache-2.0), RVM (GPL-3.0: fetch only).

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
      tracker's position smoothing.
- [x] Hips translation: Over Image placement (above) moves the character with the person.
- [ ] Side-on people: MediaPipe swaps left and right, turning the character 180°. Maybe reject
      a frame whose hip line flips faster than a person can turn.
- [ ] Package: a `MixamoPuppet` tox (Tracks DAT + FBX COMP in, rotations out, binds the bones)
      generated like the wrappers.
- [ ] Face: Holistic's blendshapes could drive a Mixamo face only if the character has matching
      shape keys (ARKit names); out of scope until a character needs it.

## Tracking (area of interest)

- [ ] The Python tracker (`python/object_tracker.py`, `*_tracker_pipeline.py`) is ByteTrack-based
      with our own changes (keypoint-distance matching, scale-relative matching, coast, presence,
      gesture debounce). Revisit after real-world testing of the scale-relative default; open
      questions to be collected from that testing.

## Housekeeping

- [x] Retired `/project1/ONNX_Native` and `python/mediapipe_hands_component.py` (Nov 7).
- [x] Deleted the stale `cache/trt/` engine folder (Nov 7); `.gitattributes` added.
- [ ] TensorRT cache: a per-user cache (`%LOCALAPPDATA%`) keyed by model content, for read-only
      installs. Today it's `trt_cache/` next to each model, which is writable in Documents.
- [x] Plugins compile in the version from `VERSION`; `package` checks it.
- [ ] CI build (GitHub Actions, CUDA toolkit).
- [ ] haxlib side: point its scripts at this repo's Python or keep a copy; remove its `cpp/`.
