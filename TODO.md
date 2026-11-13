# TODO: td-native-inference

## Status (Nov 13, 2026)

v0.4.0 published Nov 12. It ships:
- 14 video toxes plus VisionDemo, and three audio toxes plus AudioDemo;
- MixamoPuppet and HandsGeometry;
- four plugins: OnnxTOP, MediaPipeTOP, MediaPipeCHOP, and the new TrackerDAT.

Every tox's tracking runs in C++ on the Tracker DAT ([docs/TODO-tracking-cpp.md](docs/TODO-tracking-cpp.md)),
and video toxes have a Performance page. Everything is generated from `wrappers/*.json`. Releases
follow [docs/RELEASING.md](docs/RELEASING.md), and the history is in [CHANGELOG.md](CHANGELOG.md) and git.
Performance lessons: [docs/learnings/td-cook-performance.md](docs/learnings/td-cook-performance.md).

## Now

- [x] **The puppet solver in C++** (phase 6 of the tracker plan). Design and steps:
      [docs/PUPPETS.md](docs/PUPPETS.md#the-c-solver-phase-6-of-todo-tracking-cppmd).
      - Done (Nov 13): `cpp/Puppet` and `PuppetCHOP.dll`, identical to the Python solver offline and
        in the PuppetLab. `tox/MixamoPuppet.tox` is regenerated on it, with no Python solver inside
        (35 → 22 KB).
      - Left: a fresh-process check of the tox (at the next release's package test).
      - Done (Nov 13, C++ only): landmark smoothing, hand-roll smoothing and knee limits
        ([docs/PUPPETS.md](docs/PUPPETS.md#smoothing-and-limits)).
      - Next: side-on flip rejection (needs a recorded side-on clip first) and arm-twist limits.
- [x] Python is no longer the reference (Nov 13): the fixtures hold the C++ tracker's and puppet's
      own output (`--golden`), the Python comparison labs are gone.
- [x] Deleted the Python tracker, the Python puppet solver and `holistic_onnx.py` (Nov 13): the
      Tracking pages come from `python/tracking_pars.py`, the recorder runs the Tracker DAT, the
      PuppetLab has only the C++ solver.
- [ ] **DAViD: depth alone in the VisionDemo** (asked Nov 13): an option to show just DAViDHuman's
      depth.
- [x] **NeuFlow v2** first steps (Nov 13, see Models): licensed, timed, plan for a frame-pair mode.

## Speed

- [ ] **Debug overlay:** now the largest per-frame main-thread cost in Holistic GPU after the
      plugin: 0.7-1.3 ms, mostly DebugBB's Render TOP and a Select DAT. It's network work in
      `tox/shared/DebugBB.tox`, not C++.
- [ ] Hand decoded detections from the TOP to the tracker in binary (an in-process registry) to
      skip the Info DAT strings.
- [ ] `write_instance_colors` (Seg wrappers, Instances encoding) is still Python; small.

## Puppets

See [docs/PUPPETS.md](docs/PUPPETS.md) for how the solver works.

- [x] Smoothing and knee limits (Nov 13, the Puppet CHOP's Smoothing page).
- [x] Chest-up framing (Nov 13): the tracker's joint gate and Relax Unseen Limbs; the YOLO26 Pose
      visibility fix. Left: a limb jumps once when it switches between tracked and relaxed (blend
      over a few frames); one forearm spike remains on the chest-up clip.
- [ ] Arm-twist limits: elbows can't bend backward (the upper arm's roll comes from the bend), but
      bad depth can twist the upper arm instead.
- [ ] The hands are still the noisiest part (their direction comes from a 9 cm line): try a
      separate hand cutoff, or MediaPipe Hands for the fingers on a hand crop.
- [ ] Side-on people: MediaPipe swaps left and right, turning the character 180°. Maybe reject a
      frame whose hip line flips faster than a person can turn.
- [ ] Tune live with a webcam:
      - head yaw and pitch (they look under-turned);
      - whether the forearm should take only part of the hand's twist;
      - foot roll.
- [ ] Hand on a mirrored (selfie) webcam: the left/right labels name the mirrored hand, so Tracked
      Hand Right/Left picks the other hand. A Mirror toggle could swap them. Also a hand-rig variant
      of the tox.
- [ ] Over Image: the FOV is a guess per video; it could be estimated from the person's known
      proportions over a few frames.
- [ ] Other humanoid rigs: a bone-name map table instead of Mixamo names.
- [ ] Full pose model in Holistic: repack `holistic_landmarker.task` with `pose_landmarker_full`'s
      landmark model (Holistic ships the lite one). Steadier, about 2x slower.
- [ ] Face: Holistic's blendshapes could drive a face only with matching (ARKit-named) shape keys;
      out of scope until a character needs it.

## Models

- [ ] **RTMW3D** (`wrappers/RTMW3D.json`, experimental, `"demo": false`, not shipped):
      - **What it's good for:** multi-person 2D whole-body. Its 3D is clearly worse than Holistic's,
        so puppets stay on Holistic: out-of-frame feet land inverted at the crop edge (gate by
        score), arms go behind the torso, and head yaw comes from reversed face depth.
      - **Fixes:** derive head yaw from the 2D ears versus the nose, and smooth the world scale.
      - **Before shipping:** a fetch_models entry (RTMW3D-X, Apache-2.0, 369 MB), demos, docs, and
        a clean-process test.
- [ ] RTMPose-m (17 points, 2.4 ms) and RTMW-l (133 points, 3.6 ms) wrappers: Pose Crops already
      runs both. The RTMW weights' dataset terms are unchecked.
- [ ] Pinned: dedicated head-pose models for face crops, if landmark head pose falls short:
      - **6DRepNet:** about 4° MAE.
      - **WHENet:** full 360° yaw, including the back of the head.

      Both need a "crop then regress" mode. Licenses unchecked.
- [ ] RVM: fp16 / TensorRT with a fixed-size state export, `rvm_resnet50`, and `fgr` (foreground
      color) as a second output.
- [ ] DAViD on a person crop (YOLO box), so small people get the full 512 px. Also: mask normals and
      depth with the matte inside the wrapper, and a relighting example.
- [ ] Other ONNX candidates (check each license before bundling):
      - RF-DETR (Apache-2.0);
      - Real-ESRGAN upscaling and fast style transfer (Raw mode);
      - MobileSAM / EfficientSAM (needs prompt inputs);
      - RAFT optical flow (needs two frames).
- [ ] **NeuFlow v2** optical flow (2024): built for edge devices, much lighter than RAFT for similar
      accuracy. **First steps done Nov 13:**
      - **License:** the model is Apache-2.0 ([neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)),
        so it can be bundled. ONNX exports (MIT scripts) from
        [ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow](https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow),
        release 0.1.0: `neuflow_mixed.onnx` (in `models/onnx/neuflow/`, sha256 7a079294...),
        `neuflow_sintel.onnx`, `neuflow_things.onnx`, 42 MB each.
      - **Graph:** opset 16, inputs `input1` (previous frame) and `input2` (current), each
        [1, 3, 432, 768] float RGB 0-1 (fixed size). Output `output` [1, 2, 432, 768]: flow in
        pixels of the 768x432 input (dx right, dy down). 9 GridSample layers.
      - **Speed:** TensorRT fp16 takes it: **5.7 ms p50** (p95 8.2) on the RTX 4090 Laptop, launch
        1.6 ms, engine build 2.6 min. TensorRT keeps the layer norms in fp32 (overflow warning).
      - **Sanity (CPU):** the same frame twice gives 0.03 px mean flow; two dancer frames 15 apart
        give coherent motion (the head 156 px to the right).

      **Next:** OnnxTOP has no two-frame input yet. A "frame pair" mode: when a model has two image
      inputs of one shape, feed `input1` the previous frame's tensor (kept on the GPU, like RVM's
      recurrent state) and `input2` the current one. The flow comes out through Raw mode as RG,
      converted to UV (dx / 768, -dy / 432). Then a wrapper spec, a flow-to-color visualization
      (GLSL), the lab test and a fetch / bundle entry. Also compare fp16 with fp32 output on real
      frames.
- [ ] Also exported by MediaPipe and cheap to add: text classifier / embedder, language detector,
      object detector, image classifier / embedder, interactive segmenter.

## Audio

- [ ] Speech to text: try parakeet-tdt (615 MB, CC-BY-4.0) and Qwen3-ASR (960 MB, Apache-2.0), with
      a fetch entry for the large ones. Moonshine sometimes mishears ("pay" for "fade").
- [ ] Text to speech: word timestamps (`MpTextToSpeechComputeWordTimestamps`) for read-along
      captions; a voice menu; other `.litertlm` voices.
- [ ] Audio embedder (`MpAudioEmbedder*`): another task in the same plugin. At the same time, merge
      MpAudio's loader with MediaPipeTOP's MpApi into `cpp/common/`.

## Tracking

- [ ] The tracker is ByteTrack-based with our own changes:
      - keypoint-distance and scale-relative matching;
      - coasting, presence, gesture debounce.

      Revisit after real-world testing of the scale-relative default.

## Housekeeping

- [ ] A tox/plugin version check that warns on a mismatch (an installed dev build still reports the
      last `VERSION`).
- [ ] TensorRT cache: a per-user cache (`%LOCALAPPDATA%`) keyed by model content, for read-only
      installs. Today it's `trt_cache/` next to each model.
- [ ] CI build (GitHub Actions, CUDA toolkit).
- [ ] haxlib side: point its scripts at this repo's Python or keep a copy; remove its `cpp/`.
- [ ] TD exits silently now and then (twice on Nov 12, no dump; Windows logged `RADAR_PRE_LEAK_64`).
      Not reproduced. This dev machine has two unstable P-cores, so random crashes here can be
      hardware.

## Reference: measurements worth keeping

RTMPose 2D on TensorRT fp16, 256x192 crops (OpenMMLab `onnx_sdk` exports):

| Model | Keypoints | batch 1 p50 | batch 4 p50 | engine build |
|---|---|---|---|---|
| RTMPose-m body7 | 17 | 2.4 ms | 1.8 ms | 3.2 / 3.4 min |
| RTMW-l (cocktail14) | 133 | 3.6 ms | 10.0 ms | 5.4 / 5.8 min |

**Dropped:**
- **BiRefNet lite** (Nov 11): 92 ms per frame, a 304 MB engine and a 22-minute build.
- **DAViDDepth (ViT-B):** banded in fp16, 43 ms in fp32. DAViDHuman's depth is used instead.
- **OpenMMLab's YOLOX-m detector:** doesn't run on TensorRT (`TopK K exceeds 3840`); YOLO26 is the
  person detector.
