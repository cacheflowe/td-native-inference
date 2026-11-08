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

## Dev vs installed plugins (pinned: decide together)

- [ ] With a release installed in `Documents/Derivative/Plugins/`, the dev project loads two copies
      of each Custom OP (the install plus its own `plugins/` folder, which TD scans as the
      project's Plugins folder); which one serves a node is undefined. Options discussed Nov 8:
      move the build out of `plugins/` and switch the one system-wide folder between the dev build
      and a release with a junction (`use_plugins dev|release|off`); tell users "one install per
      machine" (or a show's own `Plugins/`, never both); a tox/plugin version check that warns on a
      mismatch.

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
