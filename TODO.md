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
- [ ] **Fix the TD crashes around our nodes** (Nov 18). Write-up and evidence:
      [docs/learnings/td-crashes.md](docs/learnings/td-crashes.md). The one that reproduces: deleting a
      running OnnxTOP wrapper that feeds a live network (RVM wired to Projector Map) takes TD down,
      on the unload build and on the Nov 17 build, so it predates the unload work. A scratch wrapper
      with no consumer, or the same one disconnected / off / unloaded, deletes fine. Workaround:
      `wrapper_builder.remove(comp)`.
      - [x] Documented: reproduction table, other crashes on record, how to capture the next one.
      - [x] Added `plugins/onnxtop-lifecycle.log` (create, each destroy stage, unload) and
            `scripts/enable_td_dumps.ps1` (WER full dumps, needs an elevated shell).
      - [ ] Restart TD on the new `OnnxTOP.dll`, enable dumps (run `enable_td_dumps.ps1` as admin), and
            reproduce: delete a running RVM wired to a GLSL TOP. Read the last lifecycle log line:
            "destroy: ..." = our teardown, "destroy: done" = TD after it.
      - [ ] Fix from what the log shows (see "What to fix" in the write-up): teardown order,
            `cudaDeviceSynchronize()` before freeing, the plain CUDA provider's release with RVM's state.
      - [ ] Minimal repro in the wrapper lab so the fix has a test, and check `remove()` end to end.
      - [ ] Unexplained exits (Nov 12 twice, Nov 17 three times in TensorRT builds, Nov 18 11:08
            `ucrtbase` `0xc0000409`): the dumps above should catch the next one. This machine has two
            unstable P-cores, so some may be hardware.
- [ ] **Next release (v0.6.0):** NeuFlow, the Relight component and the folder move are unreleased;
      follow docs/RELEASING.md (the package test also covers MixamoPuppet's fresh-process check).
- [ ] **Unload models when they aren't in use** (Nov 18): OnnxTOP kept a model's GPU memory after
      Active went off. OnnxTOP has `Unload After (s)` (default 30, 0 = never) and an `Unload` pulse;
      wrappers expose both and pulse Unload (inference and a nested `detector`) from a watcher
      (`unload_watch`) and VisionDemo's `model_controls`, since an off model doesn't cook. Not for the
      MediaPipe nodes (CPU). Works in VisionDemo (Nov 18, 6 s delay; check with `nvidia-smi`, since the
      `loaded` info channel goes stale while a wrapper isn't cooking):
      - [x] Rebuilt VisionDemo / AudioDemo (`DEMO = True`, `WRAPPERS = False`) on the new `OnnxTOP.dll`.
      - [x] On then off frees the memory after the delay: DAViD multi-task ~1000 MB, Holistic GPU ~350
            (its nested detector too), NeuFlow ~140, Yolo26Pose ~300. The first run missed models that
            didn't cook and nested detectors; fixed.
      - [x] Off then on within the delay keeps the session (no drop, no reload).
      - [x] Off while loading: cancelled and unloaded cleanly, no stall.
      - [x] Reload after unload works; DAViD multi-task takes ~17 s from the cached TensorRT engine.
      - [x] The `Unload` pulse reaches a node whose COMP isn't cooking.
      - [x] Rebuilt the 18 OnnxTOP wrappers (`NAMES` in batches of 3-4) so `tox/*.tox` have Unload After /
            Unload and `unload_watch`; the lab test passes on all.
      - [x] Standalone wrapper (cooking on): Active off unloads after the delay (DAViD Matte ~400 MB).
      - [ ] RVM: its recurrent state after a reload; Auto provider off during the CUDA -> TensorRT upgrade.
      - [ ] Decide the default delay (30 s is a guess; reloads are 5-20 s); a MediaPipe equivalent if RAM matters.

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
      - fast style transfer (Raw mode);
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
- [ ] **Fast 2x upscaling (Real-ESRGAN family)**: wanted: a quick 2x ONNX upscaler for live video
      (Raw mode, an output at twice the input's size; a model with a dynamic or fixed small input
      tile). Candidates to compare (none tried, check each license and ONNX source):
      - **`realesr-general-x4v3`** (SRVGGNetCompact, 1.2 M parameters, BSD-3-Clause, general
        scenes, has a denoise-strength blend): 4x only, so run it and downscale to 2x, or run it on
        a half-size input. An ONNX export exists (e.g. `vlad-grig/realesr-general-x4v3-onnx` on
        Hugging Face, the official weights unchanged). Cheapest to try.
      - **`realesr-animevideov3`** (compact, x2 / x3 / x4 variants, for animation video).
      - **`RealESRGAN_x2plus`** (RRDBNet, much heavier; quality reference, probably not real-time).
      - **OpenModelDB 2x compact models** (SPAN, compact RealESRGAN-style; many are permissively
        licensed, each its own terms).
      - **To work out:** OnnxTOP's Raw mode writes the tensor at the input's size or its own size
        (`Map to Input Frame` off gives the tensor's size, so a 2x output works); tiling for big
        frames; an fp16 TensorRT build and the cost at 720p / 1080p; edges between tiles.
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
- [ ] **Noisy RGB+D depth: clean-up and hole filling for the D415 / D435 / Kinect** (summary and
      plan, Nov 17; the per-model notes are in the next item). Goal: fill holes and IR shadows and
      remove noise from a depth camera's map using its RGB, ideally in real time inside TD.
      - **Test data:** `media/cdm/d435/` (gitignored): 254 frames of you at a desk, a raised hand
        close to the lens and the room behind; `rgb/` PNG 1920 x 1080, `depth/` EXR (half float, one
        channel) 1920 x 1080 **aligned to color**, recorded from the RealSense TOP. **1.0 in the EXR
        is 5 m.** An earlier recording without alignment was discarded: unaligned depth made every
        model look worse at edges. No Kinect or D415 frames yet.
      - **Compared on those frames** (panels in `build/cdm_compare/`; judged by eye, no ground
        truth; speeds are `onnxtop_tool` TensorRT fp16 p50 with TD open unless noted):

        | Model | License / size | Cameras | Speed | Quality |
        |---|---|---|---|---|
        | **CDM** (ViT-L x2) | code Apache-2.0, weights unstated; 5.5 GB per camera | D435, Azure Kinect, L515, ... (not D415) | 31 ms at 252 x 448, 45 ms at 322 x 574, 69 ms at 518 x 686 | Best on the hard frame (a hand over the view): crisp fingers; fills holes, shadows and glare |
        | **LingBot-Depth v0.5** (ViT-L) | Apache-2.0; 1.28 GB | RealSense, Orbbec, Azure Kinect (one model; needs intrinsics) | not timed (eager bf16: 80-90 ms at 1,200 tokens) | Cleanest silhouettes on ordinary frames; ragged fingers on the hard frame |
        | **Any2Full-S** (ViT-S) | Apache-2.0; 195 MB (B 764 MB, L 2 GB) | any (scale-prompted) | not timed; expect about 14 ms | Close to CDM on ordinary frames, no halos, fingers separate with ragged tips; valid pixels barely move |
        | **PromptDA-S** | Apache-2.0 tag; 96 MB | any | 14 ms at 504 x 896 | Cleaner than raw; edge halos, blobby fingers on the hard frame |
        | MSG-CHN | Apache-2.0 code; 364 K parameters | any | not run | KITTI outdoor LiDAR only: not suited to dense indoor depth |
        | Plain GPU fill / DA3 + fit / distilled CDM / RealSense SDK filters | n/a | n/a | not tried | see the next item |

      - **Findings so far:** alignment matters; the hand over the lens is the hard case (most raw
        depth is missing); CDM fp16 matches fp32 (median relative diff 0.0003); CDM's output is
        inverse depth; Any2Full and PromptDA output follows the input's units (scale fitted to the
        valid pixels), so no depth-scale setting is needed.
      - **Decision (Nov 17): build a Depth Clean node on Any2Full-S.** CDM stays the quality
        option (and the Kinect's, once its weights are tried); revisit LingBot if a static-shape
        export of its masked path works.
      - **Plan for the node** (about 1.5 days, no C++): (1) export Any2Full-S at a fixed 16:9 size
        (multiple of 14, e.g. 518 x 924) as one graph with a stacked input (RGB on top, depth
        below; OnnxTOP's preprocess passes 32-bit floats through unchanged at the same size), the
        global fit as closed-form 2 x 2 normal equations over the valid pixels and the prompt
        statistics as masked sums; (2) TensorRT build, speed, and an fp16 check against fp32;
        (3) a wrapper with two inputs (RGB and depth; `wrapper_builder.py` has a single `input`
        today), a packing GLSL network and the output (Raw, Normalize None, Map to Input Frame
        off) resized back to the camera's resolution; (4) host the exported ONNX (derived from
        Apache-2.0 weights; a release asset) with a pinned fetch entry, docs, icon, manifest and
        NOTICE; (5) lab test on `media/cdm/d435/` and a clean-process test. Not in VisionDemo
        (one camera input).
      - **Progress (Nov 17): Any2Full-S exported and verified; steps 1-2 of the plan are done.**
        - **Export** (`tmp/cdm/export_a2f.py vits 518 924 out.onnx --check <frame>`): one fixed-size
          graph, input `stacked` `[1, 3, 1036, 924]` (RGB top half, depth in the red channel of the
          bottom half, any units), output `depth` `[1, 1, 518, 924]` in the input's units; 136 MB,
          opset 17 (legacy exporter). The wrapper reproduces the repo's own `forward` exactly
          (relative difference: median 0, p99 1e-5). Three things had to change for a static graph:
          the patch-embedding's `adaptive_avg_pool2d` to 3 x 3 (not a divisor of 14; replaced by
          the same bins as a pooling matrix), its `while not valid_mask.all()` nearest-fill loop
          (unrolled to grid - 1 steps), and the prompt attention's mask, which the repo expands to
          `[B, heads, N, N]` and the exporter baked in as a 143 MB constant per block (broadcast
          `[B, 1, 1, N]` instead; same result). The global scale / shift fit is the closed-form
          least squares and the prompt statistics are masked sums.
        - **TensorRT fp16:** build 5.4 minutes, **p50 11.6 ms (min 8.2, p95 15.4)** with TD open.
        - **End to end in TD** (a scratch OnnxTOP on a stacked 32-bit EXR: Raw mode, Input
          Normalize 0 to 1, Letterbox off, Normalize None, Map to Input Frame off): no C++ change
          needed. The output TOP is 924 x 518 in the input's units. Against the PyTorch fp32
          reference: CUDA fp32 median relative difference 4e-5 (p99 2.6e-4); TensorRT fp16 median
          4e-4, **p99 0.85%, max 24% at a few pixels**, no NaN. The fit sums are computed in the
          graph, so fp16 accumulation is the thing to watch on other scenes.
        - **Next:** the two-input wrapper (RGB + depth) with the stacking GLSL network, a fetch /
          hosting entry for the ONNX, docs and tests.
        - **Wrapper built (Nov 17, `wrappers/DepthClean.json`, `"demo": false`):** a new spec key
          `depth_input` (with `stack_size`) makes `wrapper_builder.py` add a second In TOP,
          `input2`, and a `stack` GLSL TOP (RGB bilinear on top, depth nearest below, 32-bit float
          RGBA, 924 x 1036) feeding the OnnxTOP; the model is `models/onnx/depthclean/
          any2full_vits_518x924.onnx`. TOP 1 is the cleaned depth resized to the RGB input's
          resolution (the input's depth units; Normalize None), TOP 2 the RGB. The built COMP has
          two input connectors, `input` then `input2` (checked). Lab wiring note: a source for
          `input2` must be a sibling of the wrapper (inside another COMP the connect silently
          fails and the In TOP stays 128 x 128).
        - **Speed with TD closed (quiet GPU): TensorRT fp16 p50 7.75 ms (min 7.5).**
        - **Tested (Nov 17):** the rebuilt wrapper with the recorded frames wired in (RGB
          1920 x 1080, aligned depth): output 924 x 518 internally, TOP 1 at 1920 x 1080, every pixel
          filled (min 0.11, max 1.15, no NaN); the lab test passes (a float ramp stands in for the
          depth input: `depth_source` in `scripts/build_wrappers.py`). Panel:
          `build/cdm_compare/depthclean_wrapper_120.jpg`.
        - **Shipping status:** in `release-manifest.json` (tox and the 136 MB ONNX, bundled in the
          zip; Apache-2.0, NOTICE entry), docs in MODELS / MANUAL / `models/README.md`, CHANGELOG.
          **Still open:** a tracked export script (the export is `tmp/cdm/export_a2f.py` plus the
          attention-mask patch, neither in git), a README row and thumbnail (the lab's thumbnails
          use the stock photos, and a depth pair needs a camera: decide whether to use your
          footage), the clean-process test, your live `realsense_rgb` / `realsense_depth` (not in
          the current project), and an optional depth input on VisionDemo.
        - **TD crashed three times on Nov 17** (no dump found), each time while a TensorRT engine
          was building in the background (the wrapper rebuild + engine builds in the lab; this time
          the new model's first build). Pre-building the engine with `onnxtop_tool` (before TD
          opens the model) avoids the in-TD build: the cache is keyed on the model's path.
      - **Open:** Kinect and D415 frames (record the same kind of clips; for the Kinect, CDM's
        `kinect` weights are 5.5 GB); Any2Full fp16 on other scenes (the in-graph fit sums); whether
        ViT-B is worth it (it looked the same as ViT-S); distilling CDM or LingBot into a small net
        if the hands matter.
      - **Reproducing** (`tmp/cdm/`, gitignored, not in git): a venv with torch 2.14 (cu126, Python
        3.14), onnx, OpenEXR, opencv-headless, torchvision; clones of the CDM, PromptDA, Any2Full
        and LingBot-Depth repos; checkpoints in `weights/`; `export_cdm.py`, `export_pda.py`,
        `run_cdm.py`, `run_promptda.py`, `run_a2f.py`, `run_lingbot.py` (panels, fp16 checks),
        `export_a2f.py`, `make_stack_test.py`, `td_a2f_check.py` (the TD check, run through the
        bridge). The Any2Full clone has the attention-mask patch above in `prompt_attention.py`.
        The LingBot clone has a one-sample patch to `vision_transformer.py` (no xFormers). Copy
        the scripts somewhere tracked if the spike is revisited.
- [ ] **Camera Depth Models (CDM) for RGB+D clean-up and hole filling**
      ([project](https://manipulation-as-in-simulation.github.io/),
      [code](https://github.com/ByteDance-Seed/manip-as-in-sim-suite/tree/main/cdm) Apache-2.0,
      [weights](https://huggingface.co/collections/depth-anything/camera-depth-models); per-model
      notes, Nov 17). Cameras here: RealSense D415 and D435, a Kinect.
      - **Models:** one checkpoint per camera (D405, D435, L515, Azure Kinect, ZED 2i quality /
        neural, plus a `base`); 5.47 GB each as an fp32 PyTorch `.ckpt`. **D435 and the Azure Kinect
        are covered; the D415 isn't** (try the D435 model on it, or `base`; a D415 has a different
        baseline and field of view, so accuracy is unknown). Which Kinect (Azure, or v2)? Only the
        Azure Kinect has a model. The weights' license isn't stated on the cards (check before
        anything but a local experiment).
      - **Architecture** (`rgbddepth/dpt.py`): two full DINOv2 **ViT-L** encoders (RGB and depth),
        a 4-head cross-attention over each token's RGB / depth pair, and a DPT head on the
        concatenated features; output is metric depth in metres. Input `[1, 4, H, W]`: RGB (ImageNet
        normalized) plus the raw depth as **inverse depth** (`1/d`, 0 where there's no reading, in
        metres), resized so the short side is 518 (multiple of 14), e.g. 518 x 686 for 640 x 480.
      - **Real-time? Yes at a reduced size (measured Nov 17).** The network exports to ONNX
        (legacy exporter, opset 17, fixed size, 659 M parameters; random weights, since speed
        doesn't depend on them) and builds in TensorRT fp16 in 3-17 minutes (longer when the GPU
        is busy); the engine is 1.27 GB. Inference p50 with TD open and sharing the GPU
        (`onnxtop_tool`, 200 runs):

        | Input (H x W) | p50 | fps |
        |---|---|---|
        | 252 x 448 (16:9) | 32.8 ms | 30 |
        | 322 x 434 | 29.4 ms (p95 33) | 34 |
        | 322 x 574 (16:9) | 45.2 ms (min 35; the GPU was busy) | 22 |
        | 406 x 546 | 38.8 ms (p95 41) | 26 |
        | 518 x 686 (the authors' size for 640 x 480) | 69.2 ms (p95 73) | 14 |

        So about 26-34 fps at 320-400 px and 14 fps at full size, better than the 60-150 ms I first
        estimated for full size. Not yet known: accuracy at the smaller sizes (the authors use 518),
        and fp16 numerics with the real weights (DepthART-L overflowed in FP16; DINOv2-L usually
        doesn't, but the cross-attention is unproven). Other levers: run on every second frame and
        merge with the fresh raw depth; INT8. VRAM is modest: 1.27 GB engine, so it can sit
        alongside other models.
      - **Quality check on recorded D435 frames (Nov 17, `media/cdm/d435/`, 254 frames, 1080p
        color with aligned depth; panels in `tmp/cdm/out/`):** strong. The cleaned depth fills the
        holes and the IR shadows around a person and a raised hand, follows the RGB edges (the
        fingers come out crisp), removes noise, and corrects a bright false blob where a dark
        monitor glared; the result is smooth with some blockiness on busy shelves. 322 px and 518
        px short side look nearly the same (518 has a little more shelf detail). Valid pixels move
        a median 2% from the raw values (p90 about 6-8%); with the first, unaligned recording the
        median was 4-8%, so alignment matters. **fp16 is safe with the real weights**: against
        fp32 the median relative difference is 0.0003, p99 under 0.7%, max 6.6%, no NaN / inf.
        Unknown: what 1.0 in the recorded EXR equals in metres (the model's output scales with its
        input, so absolute values depend on it; I used 5 m per 1.0 for the panels). The network's
        output is **inverse depth** (the authors take `1 / pred`), so the export needs a
        reciprocal at the end. The authors' `infer.py` passes RGB into `infer_image`, which swaps
        channels again; the spike copies that.
      - **Compared with MSG-CHN depth completion** ([ONNX port](https://github.com/ibaiGorordo/ONNX-msg_chn_wacv20-depth-completion);
        364 K parameters, trained on KITTI outdoor LiDAR only, sparse input): a tiny fallback at
        best; not for dense, noisy indoor depth. Not run.
      - **More candidates (Nov 17, same frames; panels `build/cdm_compare/a2f_*.jpg`,
        `lingbot_*.jpg`):**
        - **Any2Full** ([repo](https://github.com/zhiyuandaily/Any2Full), Apache-2.0 code and
          weights, ECCV 2026; ViT-S 195 MB, ViT-B 764 MB, ViT-L 2 GB): a Depth Anything encoder
          prompted with the depth treated as disparity (zeros = holes), then a global scale / shift
          fit to the valid pixels. Better than PromptDA-S here: no edge halos, clean silhouettes,
          fingers separate (with some ragged fingertips and a blobby face on the hardest frame), and
          valid pixels barely move (median 0.5%). ViT-B looks the same as ViT-S. Eager PyTorch fp32
          took 125-130 ms (S) and 220-260 ms (B), mostly the `pinverse` fit over about a million
          points; an export would replace that with the closed-form 2x2 normal equations and use
          masked sums for the prompt statistics. Not exported or timed; expect ViT-S in the
          PromptDA range (about 14 ms). **The most promising fast option.**
        - **LingBot-Depth** ([page](https://technology.robbyant.com/lingbot-depth),
          [code](https://github.com/Robbyant/lingbot-depth), Apache-2.0; v0.5 recommended, 1.28 GB
          fp32, 321 M parameters; trained on 3 M RGB-D pairs from RealSense, Orbbec and Azure Kinect,
          so one model for all your cameras; needs camera intrinsics, normalized by width / height;
          a token budget of 1,200-3,600 sets the resolution): one ViT-L over the RGB tokens plus the
          depth tokens, with depth patches that have no valid pixels dropped. Excellent on ordinary
          frames, with the cleanest silhouettes of anything tried (cleaner than CDM at the
          person's edge); weaker than CDM on the hardest frame (a hand covering the view, mostly no
          raw depth): fingers come out ragged and truncated. Eager bf16: 80-90 ms at 1,200 tokens,
          300-320 ms at 3,600; not timed in TensorRT. **Export is the catch:** the masked path
          drops a variable number of tokens (it needs xFormers' nested tensors, which isn't
          installable here; I patched the one-sample case in `tmp/cdm/lingbot`). Without masking
          (`enable_depth_mask=False`) the holes are fed as zero depth, which it wasn't trained for
          and which loses the fingers. A static-shape export would need an attention mask in place
          of the token drop. The intrinsics I used are approximate (fx = fy = 1387 at 1080p).
        - **Prompt Depth Anything, ViT-S** ([PromptDA](https://github.com/DepthAnything/PromptDA);
          25 M parameters, 96 MB checkpoint, Apache-2.0 tag on the HF repo; camera-agnostic, so it
          also covers the D415 and the Kinect). Tested on the D435 frames with a 4x pooled depth
          prompt, holes pre-filled by a few pooling passes (the model normalizes the prompt by its
          min / max, so zeros in it break it). Exported as one `[1, 4, 504, 896]` input
          (`tmp/cdm/export_pda.py`): TensorRT fp16 **p50 13.9 ms (min 5), about 70 fps**, build
          6.5 minutes. Quality: close to CDM on the static room (frame 180), but blobbier on a
          close hand (frame 60: the fingers merge into the background), and a prompt left with
          holes gives dark artifacts. fp16 numerics not checked.
        - **Not tried:** a plain GPU hole fill (push-pull plus an RGB-guided filter, about 1 ms;
          fills small holes and shadows flatly, no noise correction); Depth Anything 3 / V2
          relative depth fitted to the valid pixels (10-18 ms, existing wrappers); distilling CDM
          into a small U-Net trained on our own camera footage (2-6 ms guess; camera-specific, and
          CDM's weight license limits redistribution); the RealSense SDK's own hole-filling and
          temporal filters.
      - **Spike setup** (`tmp/cdm/`, gitignored): a venv with torch 2.14 (cu126, Python 3.14),
        onnx, OpenEXR, opencv-headless and torchvision; the authors' code sparse-cloned into
        `repo/`; `export_cdm.py <H> <W> <out.onnx>` (random weights, speed only), `run_cdm.py`
        (panels and fp16 checks), the D435 checkpoint in `weights/`.
      - **What it takes** (about 3-4 days):
        1. Export: PyTorch (not installed) to ONNX at a fixed size, fp16 weights; the DINOv2
           position-embedding interpolation and the 2-token cross-attention should export, but it's
           unproven. A long first TensorRT build is likely (Depth Anything 3 took 20+ minutes).
        2. OnnxTOP: a 4-channel image input. The model's input is RGB + depth in one tensor, so no
           second TOP input is needed: the wrapper packs the depth into the alpha of one 32-bit
           float RGBA TOP (Math TOP: `1/d`, a Shuffle / Reorder TOP), and OnnxTOP's preprocess
           emits four planes with the fourth un-normalized. Small kernel change.
        3. Wrapper: the packing network, Raw mode with Normalize None (output stays in metres,
           the clean depth map at the input's size), and a **Camera** menu that swaps the Model
           path per camera (one export and one engine each; the weights are too big for the zip,
           so a fetch entry downloads and the user exports, or we host exported fp16 ONNX files if
           the license allows).
        4. Depth must be registered to the color image (RealSense TOP's align option, the Kinect's
           depth-to-color), with depth in metres.
      - **First step (done Nov 17):** a Python spike on recorded D435 frames showed the clean-up
        is worth it; see the summary item above for what we compared and decided.
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
- [ ] The full wrapper lab sometimes misses a MediaPipe wrapper that passes alone (Sound
      Classifier earlier, Hands on Nov 13: no hand in 1200 frames, no errors). Lab ordering or a
      MediaPipe graph not starting after the wrapper before it; retry once before failing?

## Reference: measurements worth keeping

RTMPose 2D on TensorRT fp16, 256x192 crops (OpenMMLab `onnx_sdk` exports):

| Model | Keypoints | batch 1 p50 | batch 4 p50 | engine build |
|---|---|---|---|---|
| RTMPose-m body7 | 17 | 2.4 ms | 1.8 ms | 3.2 / 3.4 min |
| RTMW-l (cocktail14) | 133 | 3.6 ms | 10.0 ms | 5.4 / 5.8 min |
- [ ] **Where to host our own converted ONNX models.** Depth Clean's `any2full_vits_518x924.onnx`
      (136 MB, converted by us from Apache-2.0 weights, sha256 `cb9d9064...`) is **bundled in the
      zip for now** (decided Nov 17). It's a special-use model (it needs a depth camera), so it
      would be better as a download: pick a host (a GitHub release asset of a separate "models"
      release, or a Hugging Face repo), then add a pinned `fetch_models.ps1` entry (no accept
      prompt, like Depth Anything 3), drop it from `release-manifest.json` and update the docs.
      The same host would also take any future converted model.

**Dropped:**
- **BiRefNet lite** (Nov 11): 92 ms per frame, a 304 MB engine and a 22-minute build.
- **DAViDDepth (ViT-B):** banded in fp16, 43 ms in fp32. DAViDHuman's depth is used instead.
- **OpenMMLab's YOLOX-m detector:** doesn't run on TensorRT (`TopK K exceeds 3840`); YOLO26 is the
  person detector.
