# C++ ONNX TOP (OnnxTOP): TD's bundled ONNX Runtime + TensorRT, fully on the GPU

Code: [cpp/OnnxTOP/](../../cpp/OnnxTOP/) (start with its [README](../../cpp/OnnxTOP/README.md)).
Started 2026-11-05 as a port of the Python `ONNXInferenceManager` design to a CUDA-mode CPlusPlus TOP.

## TouchDesigner already ships ONNX Runtime with the TensorRT EP

**Finding:** TD 2025.33230's `bin/` contains `onnxruntime.dll` **1.23.2** and
`onnxruntime_providers_{cuda,tensorrt,shared}.dll`. It also has TensorRT 10 (`nvinfer_10.dll`,
`nvonnxparser_10.dll`, the plugin DLLs), cuDNN 9.8, cuBLAS 12, cudart 12.9 and DirectML. Check with:

```powershell
Get-ChildItem "C:\Program Files\Derivative\TouchDesigner\bin" -Filter *.dll |
  ? Name -match 'onnx|nvinfer|cudnn|cudart|directml' | % { "$($_.Name) $($_.VersionInfo.FileVersion)" }
```

**Why it matters:**
- The Python path's TensorRT attempts crashed TD inside `ort.get_available_providers()`. The note in
  [python/util/onnx_inference_manager.py](../../python/util/onnx_inference_manager.py) `providers()`
  describes this. The likely cause is pip's `onnxruntime-gpu` 1.22 provider DLLs colliding with the
  1.23 copies TD has already loaded.
- A C++ plugin avoids the clash entirely by using **TD's own** ORT. `OrtLoader.cpp` calls
  `LoadLibraryExW(<exe dir>\onnxruntime.dll)`, then `GetProcAddress("OrtGetApiBase")`, then
  `Ort::InitApi(api)`, with `ORT_API_MANUAL_INIT` defined everywhere. The plugin has no
  `onnxruntime.lib` import. Check with `dumpbin /dependents`: it should list only `cudart64_12.dll`
  plus the CRT.
- This is the **officially intended** route. Derivative's own
  [ONNXCandyStyleTOP](https://github.com/TouchDesigner/CustomOperatorSamples/tree/main/TOP/ONNXCandyStyleTOP)
  sample says "the .dlls that come with TouchDesigner will be what is loaded". That sample uses
  ORT 1.23.2 for CUDA 12, the CUDA EP only, and the same getCUDAArray → planar kernel → device
  OrtValue path.
- Requirements, per that sample: **TD 2025.32302+ installed via the Web Installer with the "Public
  ONNX" option**, because of a conflict with the ORT version Kinect Azure needs. Machines installed
  without that option may lack the matching DLLs. OrtLoader then reports a load error or an
  API-version error rather than crashing.
- Derivative may bump ORT in future builds. The C API is backward compatible (old headers + newer DLL
  is fine), and `GetApi(23)` returning null is reported as a clear error.
- What OnnxTOP adds over that sample: the TensorRT EP, out-of-process engine builds, letterboxing
  from any input size or format, and YOLO det/seg/pose decoding.

## Toolchain trap: CUDA 12.3 + MSVC 14.4x → `STL1002`

**Symptom:** CMake's CUDA compiler probe fails. First with `unsupported Microsoft Visual Studio
version` (fixable with `-allow-unsupported-compiler`, passed through `NVCC_APPEND_FLAGS` so the probe
sees it). Then with `static_assert ... STL1002: Unexpected compiler version, expected CUDA 12.4 or
newer`.

**Root cause:** MSVC 14.4x's STL refuses to compile under nvcc older than 12.4, whatever nvcc's own
flags say.

**Fix:** build with the **v142 toolset (VS 2019 Build Tools, MSVC 14.29)**, the same toolset TD's own
`Samples/CPlusPlus/CudaTOP` project uses. Use Ninja from VS 2022's bundled CMake, because the VS
generator also needs the CUDA MSBuild integration, which isn't installed. All of this is in
[cpp/OnnxTOP/build.cmd](../../cpp/OnnxTOP/build.cmd). The alternative is installing CUDA ≥ 12.4.

## TensorRT engine builds crash intermittently, so they run out of process

**Symptom:** with no engine cache, the TensorRT build (4.5–5 min for yolo26s/n-seg on an RTX 4090
Laptop GPU) **crashed the process in 2 of 5 fresh builds**. There was no output, the exit codes were
116 or 139, there was no Application Error event, and there was no TDR or display-driver event.
Re-running the identical build succeeded. Cached loads (about 4.6 s) and inference never failed, over
thousands of runs.

**Not the cause:**
- Host memory: 96 GB RAM, 80 GB free commit.
- GPU resets: the System log was clean.
- ORT DLL conflicts: the same process succeeded on retry.

Windows did log a `RADAR_PRE_LEAK_64` report against the bench during one build, which shows how
memory-hungry the builder is. The root cause is **still unknown**. Next step if it matters: catch it
under `cdb` (see [td-crash-dumps.md](td-crash-dumps.md)) or with the bench's crash reporter, which
prints the faulting module and a backtrace.

**Mitigation (the design):** the TOP never builds an engine in TD's process.
`TrtBuilder.cpp` runs `onnxtop_tool.exe build ...` as a child process:
- Its PATH is set so TD's `bin` comes first.
- Its output is logged to `trt_cache/<model>.build.log`.
- It is retried up to 3 times.
- It is killed if the node is deleted, so TD never blocks on a minutes-long build.
- On success, the child writes `trt_cache/<model>.<hash>.built`.

The hash covers the model path, size and mtime, fp16, the SM version and the ORT version. Once that
marker exists, the in-process load is a pure cache hit.

Verified with `onnxtop_tool selftest-build`, which runs the exact launcher the TOP uses. A fresh
yolo26n-pose build succeeded on attempt 1 in 272 s, wrote its engine, timing cache, marker and
build log, and the next load came straight from the cache.

## First run inside TD: two failures the standalone bench couldn't show

### TD crashed: the Nvidia Background TOP loads a different TensorRT

**Symptom:** the out-of-process engine build succeeded. One second later, the in-process load
crashed TD. The dump (`TouchDesignerCrash.2025.33230_18.dmp`, read with cdb per
[td-crash-dumps.md](td-crash-dumps.md)) shows a null-pointer read in
`onnxruntime_providers_tensorrt!nvinfer1::IBuilder::createNetworkV2`, inlined in
`TensorrtExecutionProvider::GetSupportedList`. That step runs **even when a cached engine exists**,
because partitioning always parses the graph with a fresh TensorRT builder.

**Root cause:** `lmf` showed ORT and its providers loaded from TD's `bin`, but `nvinfer_10`,
`nvinfer_plugin_10` and `nvonnxparser_10` loaded from
**`C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects\`**. That's the NVIDIA Maxine VFX SDK
(a TensorRT 10 build from Nov 2024), which TD's **Nvidia Background TOP** loads. haxlib.toe has four
of them. Windows reuses an already-loaded DLL by module name, so TD's ORT TensorRT provider bound to
the VFX SDK's TensorRT, `createInferBuilder()` returned null, and the next call crashed.

It depends on load order: in the next session OnnxTOP loaded first, and TD's own copies were the
ones in memory. The standalone bench never loads the VFX SDK, so it can't reproduce this.

**Fix:** `ortload::claimTensorRT()` runs before any TensorRT session is created:
- If nothing has loaded TensorRT yet, it pins TD's three DLLs by full path, so later by-name loads
  resolve to them.
- If a foreign copy is already loaded, the engine skips TensorRT (and the out-of-process build) and
  runs on the CUDA EP. The node warns `CUDA (TensorRT disabled: nvinfer_10.dll is already loaded
  from ...)` instead of crashing.

**Open question:** when OnnxTOP pins TD's TensorRT first, the Nvidia Background TOPs bind to TD's
newer TensorRT build instead of their own. Whether they still work that way is untested.

**General lesson:** for in-process native ML code in TD, check `lmf` (or `EnumProcessModules`) for
*which copy* of each shared DLL is loaded. TD itself ships some of these runtimes (ORT, TensorRT,
cuDNN), and so do NVIDIA's own SDKs.

### `Input 'pixel_values' has unsupported type 1267736864`: a dangling ORT TypeInfo

`session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()` returns a **non-owning** view into the
temporary `TypeInfo`, which is destroyed at the end of the statement. The bench happened to read
still-intact freed memory; in TD the memory had been reused. Fix: keep `Ort::TypeInfo` in a local.

The name `pixel_values` was real, not garbage: the yolo26n-pose export is HF-style
(`pixel_values` → `logits`).

### `Preprocess kernel failed: invalid resource handle` after switching input videos

**Root cause:** surface objects were cached across frames and rebuilt only when TD returned a
different `cudaArray` pointer. When an input changes resolution, TD frees its array and allocates a
new one, which can come back with the *same handle value*. The cache then kept a surface for a
destroyed array.

**Fix:** create the surface objects every frame and destroy them before `endCUDAOperations()`,
the same as TD's CudaTOP sample. The cost is negligible. Never key CUDA interop resources on TD's
array pointers.

### Crashes that turned out not to be OnnxTOP: Python item assignment into a TD table (2026-11-05)

These happened while running OnnxTOP + `OnnxTopTables` (the pose tracker) + both visualizers.
- `_19.dmp` and `_21.dmp`: an access violation inside libTD's cook code.
- `_20.dmp`: "Failed inside operator .../OnnxTopTables/script_tracks". The interpreter called a
  null function pointer while the object in hand was an ordinary `str`.

**What settled it:** compare the top frames of *every* dump, not just the new ones. `_16` (Oct 16),
`_17` (Oct 28), `_19` and `_21` have an identical stack: `python311!PyObject_SetItem` → libTD → a
cook chain → access violation. `_16` and `_17` predate OnnxTOP entirely. Every dump except one
in September is from 2026-10-09 or later, the day the `PythonParticles` demo went in.
`_20` (a different site, in the tracker script) fits heap damage left by the same thing.

```powershell
# top frames of every dump, side by side (cdb from the WinDbg Store app)
& $cdb -z <dump> -c ".ecxr; kc 9; q"   # take the lines after "Call Site"
```

**Suspect:** `/project1/TESTING/PythonParticles/PythonParticles.writeParties()`, called from an
Execute DAT's `onFrameEnd`, writes ~1,050 table cells one at a time every frame
(`self.partieDAT[row, c] = v`). It's the only per-frame Python item assignment into a TD object
in the project (scan every text DAT for `x[...] = ...` and drop dicts/numpy). Each assignment
dirties the table and its dependents from inside the frame-end callback.

**Status (parked):** `logs/faulthandler.log` (the Python stack at each crash) later confirmed
the 2026-10-16 crash was inside `writeParties()` itself. Its `execute1` is Active = off. Two
more crashes happened later that day with it off, in sessions running the pymunk physics demo;
pymunk's teardown order is the next suspect. Full notes are in a text DAT,
`/project1/ONNX_Native/PythonPhysics2d/debug_notes`; the out-of-TD repro scripts are
`tmp/stress_pymunk.py` and `tmp/stress_pymunk_teardown.py`. If crashes stop,
rewrite it to keep the data in numpy and *pull* it in a Script CHOP's `onCook`
(`scriptOp.appendChan(name).copyNumpyArray(row)`, see [.ai/skills/td-warp.md](../../.ai/skills/td-warp.md)),
instead of pushing cells from a callback. `getInfoDATEntries` also bounds-checks its row index now, defensively.

### TensorRT `.built` markers were keyed on the path as typed

The marker that tells the node "this engine is already built" hashed `fs::absolute(model)`. TD's
file par hands over `D:/workspace/...` while a command-line build of `data/ml/...` gives
`D:\workspace\haxlib\data/ml/...`: same file, different hash. So an engine pre-built with
`onnxtop_tool build` wasn't recognized in TD, and Auto rebuilt it (a fast ORT engine-cache hit,
but a surprise). The key now uses the absolute, lexically normal, backslash, lower-case path.
That changed every marker name once: after updating, each node's first load does a quick
background "build" that is an ORT engine-cache hit.

## Updating the DLL while TD is open

A node created from the TAB menu is a **Custom OP**, auto-loaded from the project's `Plugins/`
folder. It has no Unload Plugin toggle, and TD keeps the DLL locked. Windows does allow renaming a
loaded DLL, so `cmake/copy_plugin.cmake` moves it aside as `OnnxTOP.dll.old-<time>`, copies the new
build in, and cleans up stale copies later. **Restart TD** to pick it up.

## Results (RTX 4090 Laptop GPU, TD-bundled ORT 1.23.2, 640×640)

| Model / provider | Load | Inference mean / p95 | Full TOP pipeline* |
|---|---|---|---|
| yolo26s-seg, TensorRT FP16 (cached) | 4.6 s (first build ~4.5 min) | 2.0 / 2.6 ms | 2.4 ms mean / 2.3 ms p95 |
| yolo26n-seg, TensorRT FP16 | first build ~5 min | 2.0 / 2.4 ms | — |
| yolo26n-pose, TensorRT FP16 (cached) | 12 s cold disk (out-of-process build 272 s) | 1.65 / 2.1 ms | 1.4 ms mean / 1.7 ms p95 |
| yolo26s-seg, CUDA EP | 0.7 s | 5.4 / 7.1 ms | — |
| Depth Anything V2 ViT-S, 518² (dynamic input), TensorRT FP16 (cached) | 8.7 s | 4.9 / 8.4 ms | 5.1 ms (Raw: range + map to 810×1080) |
| Depth Anything V2 ViT-S, 518², CUDA EP | 1.7 s | ~25 ms | ~26 ms |

Dynamic-H/W models work with the TensorRT EP without explicit `trt_profile_*` options: the
out-of-process build's warm-up run at Dynamic Size creates the engine and its `.profile`, and the
in-TD load at the same size is a cache hit. Fixed-size models leave the size out of the `.built`
marker key, so adding Dynamic Size didn't invalidate existing YOLO engines.

\* The pipeline covers BGRA8 texture → letterbox → inference → readback of 45 KB of detections →
mask decode → matte composite, with an image upload that emulates TD's bottom-up BGRA layout.

Nano and small run in the same ~2 ms at this size, so the floor here is launch/sync overhead, not FLOPs.

The first pipeline frame costs about 140 ms, all of it one-time `cudaMallocHost`/`cudaMalloc` of the
decode buffers. Visual check: the masks line up exactly on the source image
(`onnxtop_tool ... --image --out` writes an overlay PPM).

## What changed versus the Python design

These rounds in
[.ai/skills/td-threaded-inference-optimization.md](../../.ai/skills/td-threaded-inference-optimization.md)
no longer apply:
- The worker thread and queue (Rounds 2, 3, 7, 8).
- copyNumpyArray and readback (Round 9).
- The `Framedelay=2` cache workaround (Round 10).
- GC freezes (Round 11).

Inference runs **inline in `execute()`, so a result shows up on the same frame as its input.**

What still applies:
- Fixed input shapes (Rounds 4/5). TensorRT is stricter still: dynamic H/W inputs are rejected with
  a "re-export with a fixed input size" error.
- The `cudnn_conv_algo_search=HEURISTIC` setting for the CUDA EP fallback.

**Gotcha:** every `inputs->get*()` and `createCUDAArray()` call must happen *before*
`beginCUDAOperations()`, the same rule as in TD's CudaTOP sample.
