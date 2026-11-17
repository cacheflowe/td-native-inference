# TODO: td-native-inference

## Status (Nov 14, 2026)

v0.5.0 published Nov 13. It ships:
- 14 video toxes plus VisionDemo, and three audio toxes plus AudioDemo;
- MixamoPuppet and HandsGeometry;
- five plugins: OnnxTOP, MediaPipeTOP, MediaPipeCHOP, TrackerDAT and PuppetCHOP.

Unreleased (CHANGELOG): NeuFlow v2 optical flow, the Relight component, models / components / demos
in three folders (hand-built components and demos: `scripts/export_hand_built.py`; demos are private),
VisionDemo's Show menu and DAViD depth output.

Every tox's tracking runs in C++ on the Tracker DAT ([docs/TODO-tracking-cpp.md](docs/TODO-tracking-cpp.md)),
and video toxes have a Performance page. Everything is generated from `wrappers/*.json`. Releases
follow [docs/RELEASING.md](docs/RELEASING.md), and the history is in [CHANGELOG.md](CHANGELOG.md) and git.
Performance lessons: [docs/learnings/td-cook-performance.md](docs/learnings/td-cook-performance.md).

## Now

- [x] **The puppet solver in C++** (phase 6 of the tracker plan). Design and steps:
      [docs/PUPPETS.md](docs/PUPPETS.md#the-c-solver-phase-6-of-todo-tracking-cppmd).
      - Done (Nov 13): `cpp/Puppet` and `PuppetCHOP.dll`, identical to the Python solver offline and
        in the PuppetLab. `tox/components/MixamoPuppet.tox` is regenerated on it, with no Python solver inside
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
- [x] **DAViD: depth alone in the VisionDemo** (Nov 13): the Show menu, and its depth as TOP 3.
- [x] **NeuFlow v2** (Nov 13): a frame-pair mode in OnnxTOP, `NeuFlowV2.tox` in the Optical Flow
      TOP's format, cut detection, smoothing and a noise floor. Unreleased.
- [x] **Relight** (Nov 13-14): the Relight component (two point / spot lights following the hands on
      DAViD's outputs) and a private RelightDemo; models / components / demos in three folders.
- [ ] **Next release (v0.6.0):** NeuFlow, the Relight component and the folder move are unreleased;
      follow docs/RELEASING.md (the package test also covers MixamoPuppet's fresh-process check).

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
      depth with the matte inside the wrapper. (Relighting is the Relight component.)
- [ ] **Other normal models** (compared Nov 13, `build/normals_compare/`, not shipped):
      - **Sapiens Normal 0.3B** (Meta, CC BY-NC 4.0, onnx-community fp16 ONNX, 1024x768 in,
        512x384 out, not unit length): a bit crisper on people, but 185 ms on CUDA and its TensorRT
        build stalled (15 of 16 GB in use with TD open). Worth one retry with the fp32 ONNX and TD
        closed; DAViD (MIT, 6 ms) stays the default.
      - **DSINE** (CVPR 2024; Imperial College licence: personal, non-commercial, no
        redistribution, so a local experiment only; the user downloads `dsine.pt` from the GitHub
        README). Read through the code: EfficientNet-B5 + a ConvGRU refinement (5 passes, 5x5
        neighbours at 1/8 size), ImageNet input, padding to a multiple of 32, intrinsics from an
        assumed 60° FOV (bake in). Export fixes: in-place masked assignments to `torch.where`,
        maybe `F.unfold` to slices, the 1e-8 epsilons for fp16. Output axes likely OpenCV (flip
        y, z; needs a per-channel remap). Estimate 10-15 ms fp16 / 20-25 ms fp32 at 640x480.
        About half a day; quality on people is the open question.
- [ ] **DepthART** relative depth ([repo](https://github.com/xuefeng-cvr/DepthART), ONNX on
      [Hugging Face](https://huggingface.co/Fengxue93/DepthART/tree/main/onnx); looked at Nov 17):
      - **Only the MobileNetV4 graphs run for us.** The 12 TinyViM S/B/L graphs (relative and
        metric) contain a custom `com.depthart:SelectiveScan` op (ONNX Runtime refuses to load
        them); it needs the authors' CUDA kernel and a TensorRT plugin. The six MobileNetV4 graphs
        (S / M / M-slim-SPF, 224 or 448) are standard opset 17.
      - **MNv4-M-slim-SPF 448** (`relative_mnv4m_slim_spf_448_default.onnx`, 24 MB, in
        `models/onnx/depthart/`): input `image` [1, 3, 448, 448], output `depth` [1, 1, 448, 448]
        (relative, affine-invariant; near = bright). TensorRT fp16 launches in about 1 ms; its
        p50 inference time measured 4-17 ms with TD open (too noisy to compare with Depth Anything
        V2-S, which the authors put at 3.8 against 41 GMAC); engine build 6 min the first time.
        **Re-measure with TD closed.**
      - **Accuracy:** below Depth Anything V2-S (NYUD delta1 0.952 vs 0.974, KITTI 0.928 vs 0.944).
        MobileNetV4 is sensitive to wide aspect ratios (the authors tile KITTI), so 16:9 input
        needs a square-ish or windowed run.
      - **License:** the code and the released parts are CC BY 4.0, but the checkpoints were
        trained on Depth Anything V2-L's output (CC BY-NC 4.0), so treat the weights as
        non-commercial; don't bundle.
      - **Shipped Nov 17:** `DepthART.tox` (in `release-manifest.json`, VisionDemo layer, icon,
        thumbnail, `docs/MODELS.md`, MANUAL, README, `models/README.md`), with a
        `fetch_models.ps1` entry (`-AcceptDepthArt` / `-SkipDepthArt`; the pinned HF revision's
        sha256 `825114de...fb71` matches the authors' CHECKSUMS.sha256; tested from a clean
        folder under Windows PowerShell 5.1). Raw mode, ImageNet normalize; keep **Letterbox
        on**: stretching a 3:2 photo to the square input left big blobs in the background. On a
        portrait it's close to Depth Anything V2-S (a little noisier); on a flat studio wall it's
        worse.
      - **Still open:** the clean-process test (`tests/run_clean_wrapper.ps1`) of `DepthART.tox`;
        the timing above. The metric models need the TinyViM plugin (and intrinsics), so they're
        out for now.
- [ ] **Depth Anything 3 Base** (Nov 17; `wrappers/DepthAnything3.json`, in `release-manifest.json`):
      - **Model:** `onnx-community/depth-anything-v3-base` (Apache-2.0; Small and Base only, Large /
        Giant are CC BY-NC), 413 MB as `model.onnx` + `model.onnx_data`, opset 18, standard ops.
        Input `pixel_values` [batch, views, 3, H, W], four outputs (depth, confidence, extrinsics,
        intrinsics).
      - **Done:** OnnxTOP loads a `[batch, 1, 3, H, W]` input as `[1, 3, H, W]` (`hasViewsAxis` /
        `imageInputShape` in `OnnxEngine.h`), and Raw mode has an **Invert** toggle (DA3 predicts
        depth, far = large). So the original files work unmodified; `fetch_models.ps1` downloads
        them (pinned revision, sha256, tested from a clean folder). Raw Output Index 0 is the depth;
        the other outputs are computed too (a graph with only the depth would be a little cheaper).
      - **Quality:** clearly the cleanest of the three on the lab photos: crisp edges, real
        structure on the portrait's arms and a believable floor / wall gradient, where Depth
        Anything V2-S is blobbier and DepthART is noisy.
      - **Speed:** TensorRT fp16 launches in 2-3 ms; inference measured p50 18 ms (best 7-9 ms) with
        TD sharing the GPU, so **re-measure with TD closed**. CUDA is 40-200 ms. The first
        TensorRT engine build took over 20 minutes (a 2,600-node graph); later loads take a minute.
        Ideas: a fixed input size, or the Small model if one appears (none found).
      - **Shipped Nov 17:** also a VisionDemo layer, icon, thumbnail, docs (MODELS, MANUAL, README,
        `models/README.md`, OnnxTOP README), CHANGELOG; full wrapper lab passes (26).
      - **Still open:** the clean-process test of `DepthAnything3.tox` and `DepthART.tox`. TD
        crashed twice around the wrapper / demo rebuilds that day (no dump found); rebuilding the
        wrappers and the demos in separate bridge calls was stable.
- [ ] **Video Depth Anything Small** (Apache-2.0; looked at Nov 17, not tried): the ONNX exporter
      ([MapMindAI/DepthAnythingOnnx](https://github.com/MapMindAI/DepthAnythingOnnx), MIT) takes a
      window of frames `[B, T, 3, H, W]` (trained on 32). It isn't recurrent, so live use means
      running the last T frames every cook (T times the cost; ViT-S at 518 is about 5 ms a frame),
      and a new OnnxTOP "frame window" mode (like the frame-pair one). The authors' streaming mode
      (cached temporal hidden states) is experimental and drops accuracy (ScanNet delta1 0.926 to
      0.836). It also needs the `.pth` checkpoint and PyTorch to export. Probably 2-3 days for a
      modest gain; try a cheaper temporal filter on the depth output first.
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
- [ ] **Eye gaze** (looked at Nov 14):
      - **First, what we already have:** MediaPipe Face's blendshapes include `eyeLookIn/Out/Up/Down`
        per eye (a gaze proxy, 0-1), and its 478 points include the irises (468-477): iris centre
        against the eye corners gives a 2D gaze direction. A `gaze_x` / `gaze_y` pair per face in
        the tracks, from either, might be all most projects need. Cheapest; try this first.
        **Done (iris route, Nov 14):** MediaPipeTOP's Eye Gaze toggle. Still open: a gaze direction in
        camera space (combine with the head transform) and calibrating the range on real footage.
      - **Qualcomm AI Hub EyeGaze** ([model card](https://aihub.qualcomm.com/models/eyegaze)): EyeNet
        from [david-wb/gaze-estimation](https://github.com/david-wb/gaze-estimation) (stacked
        hourglass, 2.6M parameters). Input one eye crop, grayscale 160x96, histogram-equalized,
        0-1 (crop 1.5x the eye-corner distance); outputs eye heatmaps, 34 eye landmarks and gaze
        pitch / yaw (radians; flip yaw for the right eye). Float ONNX on
        [Hugging Face](https://huggingface.co/qualcomm/EyeGaze) (`release_assets.json`); about 1-2 ms on
        phones, so trivial here. Needs a crop-then-regress path from face landmarks (two crops per
        face, like HSEmotion's). **Licence caveat:** Qualcomm labels it BSD-3, but the upstream
        repo has no licence and was trained only on UnityEyes renders (UnityEyes' terms are
        research-only, unchecked); its README reports ~14 (degrees?) error on MPIIGaze, weak by
        today's standards. Check provenance before shipping; compare with the blendshape route.
- [ ] **VRMBG 3.0** matting ([briaai/VRMBG-3.0](https://huggingface.co/briaai/VRMBG-3.0)): another
      matting option next to RVM. The license is non-commercial, so a local experiment only (like
      DSINE; don't bundle). Check the ONNX export, input size and speed against RVM.
- [ ] **Camera Depth Model (CDM) for RGB+D noise reduction**
      ([depth-anything/camera-depth-model-d435](https://huggingface.co/depth-anything/camera-depth-model-d435)):
      cleans a RealSense D435's raw depth using the RGB frame. Look at the inputs (RGB + raw depth),
      the export / ONNX route, the license, and how a two-image-input model fits OnnxTOP (same
      question as NeuFlow's frame-pair mode).
- [ ] **3D pose / mesh from one image: [ROMP](https://github.com/Arthur151/ROMP) and
      [Multi-HMR](https://github.com/naver/multi-hmr)** (the `naver/multi-hmr2` link as given
      doesn't match the repo name; confirm it): real-time pose estimation whose output includes a
      3D body model (SMPL parameters / mesh), for projection mapping and general pose. Check
      licenses (SMPL is non-commercial, and Naver's terms are unchecked), ONNX export, speed, and
      the output format. If it pans out, it feeds the projection documentation.
      - **Spike first (Nov 16):** Multi-HMR ViT-B (672 px; NAVER's V100 times: S 29 ms, B 43 ms,
        L 74 ms) with the SMPL-X layer baked into the ONNX (joints and vertices out). Measure
        TensorRT speed and joint quality before touching OnnxTOP. ROMP has an `--onnx` option but
        is the 2021 model: jittery. Neither can be bundled (SMPL / SMPL-X non-commercial).
      - **Puppet fit:** the solver reads world landmarks by name from the tracks, so SMPL-X joints
        mapped to those names should drive it unchanged (needs: a new OnnxTOP output mode, like
        MoveNet's). It could fix the puppet's known weak spots: side-on left/right swaps, bad
        depth, hands.
      - **Projecting onto a person (prototype plan):** the matte (RVM / DAViD) plus pose joints
        already give a 2D silhouette and body-part anchors, and DAViD depth / normals the surface:
        calibrate camera to projector with a depth camera (D435 / L515) first. Monocular mesh
        position is only cm-to-dm accurate and 30-70 ms late, so use the mesh for UV texturing and
        multi-person, not for position (depth does that; see the CDM item).
- [ ] Also exported by MediaPipe and cheap to add: text classifier / embedder, language detector,
      object detector, image classifier / embedder, interactive segmenter.

## Ideas from the macOS libraries (Nov 14)

[Apple Frameworks for TouchDesigner](https://github.com/sygnalinc/Apple-Frameworks-for-TouchDesigner)
(60 Apple-framework custom OPs) and [AML](https://forum.derivative.ca/t/aml-a-new-touchdesigner-operator-family-for-ai-ml-on-macos/1014942)
(Apple Vision + a Core ML collection: 3D, audio, Core ML, LLM and Vision node groups; node list from the
user, Nov 14). Worth borrowing:
- [ ] **Stock-OP-compatible outputs:** they match the Body Track CHOP's and Face Track CHOP's channel
      layouts, so existing networks swap in. We did this for NeuFlow (Optical Flow TOP format); a
      Body Track-compatible CHOP option on the pose toxes and a Face Track one on the face toxes
      would make us a drop-in for people moving off NVIDIA's Maxine OPs (or adding multi-person).
- [ ] **In-TD model manager** (AML): see which models are installed, download missing ones from a
      button, with a state per model, instead of `fetch_models.cmd` outside TD. Each tox could
      also offer "download this model" when its file is missing.
- [ ] **An agent skill for the toxes** (they ship `.claude/skills` describing how to wire their OPs):
      a skill telling a coding agent which tox does what, their inputs / outputs and recipes.
- [ ] **A minimal example per tox** (their `demo.toe` has one container per OP), besides VisionDemo.
- Models / tasks we don't have yet, roughly by interest (licences unchecked):
  - [ ] **Saliency + auto-framing:** a saliency map and a crop rectangle of the interesting region,
        for automatic camera moves.
  - [ ] **OCR** (text with boxes): PaddleOCR's ONNX models (Apache-2.0) would fit.
  - [ ] **Point-prompted segmentation** (SAM 2): already listed under Models (MobileSAM /
        EfficientSAM); they show the use: cut out whatever the audience touches.
  - [ ] **Animal pose** (dogs and cats): e.g. an AP-10K-trained RTMPose (we run RTMPose crops).
  - [ ] **QR / barcode reading** and **rectangle detection** (corners into a Corner Pin): classic
        CV, small C++ (OpenCV is not a dependency yet).
  - [ ] Single-person 3D pose: Holistic's world landmarks already cover it.
  - [ ] **Metric depth** (AML's Depth Metric): depth in metres instead of relative, for placing 3D
        content or measuring distances. Depth Anything V2's metric variants (Small is Apache-2.0,
        Base / Large CC-BY-NC) would drop into the existing DepthAnything spec; also Depth Pro.
  - [ ] **Track any object** (AML's Object Tracker, Apple's VNTrackObjectRequest): seed a box (or a
        click) and follow that one thing, whatever it is. A small Siamese tracker (NanoTrack,
        OSTrack-lite; ONNX) per tracked box, through the crop path; the box out as a track row.
  - [ ] **Image similarity / zero-shot labels** (AML's Image Similarity, YOLO Classify): CLIP-style
        embeddings (MobileCLIP) as CHOP channels; with text embeddings, "how much does this look
        like <prompt>" scores for installations. MediaPipe's image classifier / embedder is the
        cheap version (listed under Models).
  - Image or photos to 3D mesh (AML's Geo Gen, Object Capture): heavy offline work (TripoSR,
    photogrammetry), not a live-inference fit.
  - Out of scope here: LLMs, translation, image generation, maps, MIDI and OS integrations.

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
- [ ] The full wrapper lab sometimes misses a MediaPipe wrapper that passes alone (Sound
      Classifier earlier, Hands on Nov 13: no hand in 1200 frames, no errors). Lab ordering or a
      MediaPipe graph not starting after the wrapper before it; retry once before failing?

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
