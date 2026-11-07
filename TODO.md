# TODO: td-native-inference

## Status (Nov 7, 2026)

Done: the move out of haxlib; Custom OP plugins with models resolved next to the DLL; 13 model
toxes and ModelDemo generated from `wrappers/*.json`; clean-process tests; Apache-2.0 LICENSE and
NOTICE; release tooling that bundles the redistributable models. The previous plan (Package Root,
CPlusPlus TOP wrappers) is in git history before commit 7ddfb48.

## v0.1.0 release

- [ ] Test `fetch_models.ps1` end to end (Ultralytics export of yolo26s, yolo26s-seg,
      yolo26n-pose) and check the exported files with the YOLO wrappers. The current
      `yolo26s.onnx` / `yolo26n-pose.onnx` have a `pixel_values` input from an unrecorded export;
      the official export may differ (output layout), so confirm OnnxTOP decodes it.
- [ ] Package, test the extracted zip in a clean TD process (`tests/run_clean_wrapper.ps1`), then
      `release.py verify --tested-in-clean-project` and publish (docs/RELEASING.md).

## Next: more model types

The builder assumes video in, a TOP plugin, an optional tracker and an image overlay out. Grow it
per slot when the first non-video model lands, not before:

| Slot | Today | Add |
|---|---|---|
| `input` | video (In TOP) | `audio` (In CHOP), `text` (In DAT) |
| `plugin` | OnnxTOP, MediapipeTOP | a MediaPipe CHOP plugin; a Whisper plugin |
| `kind` (outputs + visualizer) | landmarks, boxes, image, classifier | `scores` (class table + CHOP, bar visualizer), `embedding` (vector CHOP), `transcript` (text DAT, subtitle visualizer) |
| post-processing | tracker (ids, smoothing, gesture onset) | score smoothing and onset events for sound classes |

- [x] **MediaPipe audio classifier**: `cpp/MediaPipeCHOP` (Custom CHOP `MediapipeaudioCHOP`):
      audio CHOP in (any rate, mixed to mono), YAMNet's 521 class scores out as named channels,
      top results in the Info DAT. Worker thread, rolling Window/Hop. `build/audio_test.exe`
      checks it offline (440 Hz sine -> Sine wave 0.996, silence -> Silence 0.80).
- [ ] Live-test MediapipeaudioCHOP in TD (restart to register it), then a `MediaPipeAudio`
      wrapper: an `audio` input slot and a `scores` kind (score smoothing, onset events, a
      top-results debug TOP). Add yamnet.tflite (Apache-2.0) and the DLL to the release.
- [ ] Audio embedder (`MpAudioEmbedder*`): same plugin, second task. Merge MpAudio's loader with
      MediaPipeTOP's MpApi into `cpp/common/`.
- [x] **Speech to text**: MediapipeaudioCHOP's Transcribe Speech task wraps MediaPipe 1.2.2's
      AudioTranscriber (bundled in libmediapipe.dll; no whisper.cpp needed). Models are
      `.litertlm` bundles from huggingface.co/litert-community: moonshine-tiny (30 MB, MIT,
      streams interim text), whisper-tiny (42 MB, Apache-2.0, 30 s windows), parakeet-tdt-0.6b-v3
      (615 MB, CC-BY-4.0, stateful streaming), Qwen3-ASR-0.6B (960 MB, Apache-2.0). Offline
      (`build/asr_test.exe`, synthesized speech): whisper-tiny word-perfect at 0.06x real time,
      moonshine-tiny one word off at 0.10x; 48 kHz input is resampled internally. Our own pause
      detection flushes the stream so sentences finalize (whisper otherwise waits for 30 s).
- [ ] Live-test Transcribe Speech in TD (restart for the new DLL); a `MediaPipeSpeech` wrapper
      (`transcript` kind: text DAT, latest-sentence DAT, new-sentence pulse, subtitle debug TOP);
      try parakeet-tdt and Qwen3-ASR; models fetch for the large ones.
- [ ] The DLL also has text-to-speech (`text_to_speech.h`; Kokoro-82M, Qwen3-TTS .litertlm).
- [ ] Also exported and cheap to add: MediaPipe text classifier/embedder, language detector,
      object detector, image classifier/embedder, interactive segmenter.
- [ ] Other ONNX candidates: RF-DETR (Apache-2.0), RVM (GPL-3.0: fetch only).

## Tracking (area of interest)

- [ ] The Python tracker (`python/object_tracker.py`, `*_tracker_pipeline.py`) is ByteTrack-based
      with our own changes (keypoint-distance matching, scale-relative matching, coast, presence,
      gesture debounce). Revisit after real-world testing of the scale-relative default; open
      questions to be collected from that testing.

## Housekeeping

- [ ] Retire `/project1/ONNX_Native` (legacy chains) and `python/mediapipe_hands_component.py`
      once nothing needs them; delete the stale `cache/trt/` engine folder.
- [ ] TensorRT cache: a per-user cache (`%LOCALAPPDATA%`) keyed by model content, for read-only
      installs. Today it's `trt_cache/` next to each model, which is writable in Documents.
- [ ] Plugin metadata version from `VERSION`; CI build (GitHub Actions, CUDA toolkit).
- [ ] `.gitattributes` (LF/CRLF warnings on every commit).
- [ ] haxlib side: point its scripts at this repo's Python or keep a copy; remove its `cpp/`.
