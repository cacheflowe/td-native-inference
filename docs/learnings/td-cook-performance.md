# Keeping TouchDesigner's cook light: readback stalls, sample rates, cooking flags

What we found in Nov 2026 (v0.2.4 and v0.2.5) while chasing frame rate in the puppet lab. Each
section is a pitfall, how it showed up, and what the code does about it now.

## A GPU readback in the cook stalls TD's main thread

**Symptom.** MediaPipe Holistic's node cost TD 2.5 ms CPU per cook (spikes to 9 ms) although the
plugin's own work was ~0.3 ms; the ONNX models' cooks cost about as much as their whole inference
(YOLO26 ~3.2 ms, YuNet ~1.9 ms).

**Cause.** Reading results back to the CPU ended with `cudaStreamSynchronize` inside `execute()`,
so TD's main thread waited for everything queued on the stream: MediaPipeTOP's resize before the
copy, OnnxTOP's whole inference (ONNX Runtime also syncs at the end of `Run()` by default).

**Fix (both plugins).** Queue the copy (or the run) plus a `cudaEvent`, return, and pick the result
up on a later cook once `cudaEventQuery` says it's done:

- MediaPipeTOP: the resized frame's copy lands one cook later and only then goes to the worker
  thread. Holistic in the puppet lab: TD CPU 2.5 → 0.9 ms per cook, worst case 9.1 → 1.9 ms.
- OnnxTOP: `Run()` with the run option `disable_synchronize_execution_providers = 1`, the next cook
  decodes the outputs (nothing overwrites them until the next run is queued after the decode on the
  same stream). YOLO26: 3.8 → 2.2 ms per cook. The remaining ~1.5–2 ms is ONNX Runtime's CPU cost
  of launching the run, which no sync setting removes.
- `infer_ms` is now GPU time from the two events (`cudaEventElapsedTime`), not a wall-clock span
  that included the stall. `run_ms` is the CPU time of the launch alone.

**What the launch costs, and CUDA graphs don't fix it.** Measured with `onnxtop_tool --graph`
(TensorRT fp16, 300 runs, p50): plain launch YOLO26s 2.5 ms / seg 2.3 ms / YuNet 0.6 ms / MoveNet
0.7 ms. With TensorRT CUDA graphs (`trt_cuda_graph_enable`), ONNX Runtime waits for the replay
before `Run()` returns, so the launch costs the whole inference (YOLO26s 3.1 ms, seg 4.4 ms), loads
take 2-3x longer, and only YuNet broke even. The plugin doesn't offer it; the bench keeps the flag
for future ORT versions.

**Gotcha found on the way.** Querying a CUDA event that was never recorded returns an error, and
the error stays pending: the next kernel launch's `cudaGetLastError()` check reported it ("invalid
resource handle") and every ONNX model failed its first cook for good. Only touch the events once
a run has recorded them, and clear the not-ready status after a query.

## Exact frame vs not waiting: Wait for Result and result_delay

Not waiting makes results a frame or more older than the image they're drawn on. Both plugins now
have **Wait for Result** and report **`result_delay`** (input frames between the image and the
result) so the choice is explicit:

| | Default | Why |
|---|---|---|
| OnnxTOP | On (exact frame) | GPU inference is a few ms; overlays expect exact sync |
| MediaPipeTOP | Off | CPU inference is 20+ ms; waiting would cost frame rate. With **Max Wait (ms)**, fast models (Face Detector, Hands) still sync exactly |

A Cache TOP on the input with Output Index `-result_delay` shows the frame that matches the
landmarks. Images the plugins produce on the GPU (Depth Anything, the seg matte) are always the
current frame.

Downstream, the trackers apply a MediaPipe result only once while a newer one is on its way
(`result_delay > 0`): re-applying it every cook in between double-advanced the smoothing. A result
that's already for the newest frame (a still image, Wait for Result) still runs every cook: a
still image has only one result, and its tracks need several cooks to be confirmed.

## A node cooked twice per frame: cookEveryFrame

**Symptom.** MediaPipe Holistic's node cooked ~1.4 times per drawn frame (168 cooks in 120 frames at
a steady 60 fps), and its Info DAT and tracker script with it. The movie input and the render
cooked exactly once per frame.

**Cause.** Both plugins set `TOP_GeneralInfo::cookEveryFrame` while a result was on its way (with
async readback: nearly always). That makes TD cook the node on its own each frame, separately from
the cook its changing input causes later in the same frame.

**Fix.** `cookEveryFrameIfAsked` alone: the node cooks every frame while anything uses its output,
which also lands a pending result for a still input. `cookEveryFrame` only while a model loads.

## A CHOP's sample rate defaults to its input's

**Symptom.** The Sound Classifier wrapper's `scores` Lag CHOP cost ~20 ms per cook.

**Cause.** `MediaPipeCHOP` outputs one sample of 521 class scores per cook but never set
`CHOP_OutputInfo::sampleRate`. TD gave the output the audio input's 16 kHz, so the time-sliced Lag
CHOP after it produced ~270 samples per class every frame.

**Fix.** Report the timeline rate (`inputs->getTimeInfo()->rate`) for one-sample-per-cook outputs.
Check the rate (middle-click the CHOP) of any custom CHOP that feeds a time-sliced one.

## Nodes can keep cooking inside a COMP with cooking off

**Symptom.** With `/project1/WrapperLab`'s Cooking flag off, its wrappers still cost ~350 ms of
CPU per second; the puppet lab ran at 31 fps instead of 55–59.

**Cause.** Not pinned down. The nodes had been rebuilt and force-cooked by scripts while the COMP's
cooking was off (wrapper rebuilds, the lab test). A single forced cook didn't reproduce it.

**Fix.** Switching the flag on and off again stopped them immediately. Scripts that build or test
inside a COMP with cooking off now restore the flag with a real switch afterwards
(`wrapper_builder.refresh_cooking()`, and `tests/td_wrapper_lab.py` turns cooking on for the run
and back off at the end; forced output cooks don't reach a plugin in a non-cooking COMP anyway).
To check a COMP, compare `totalCooks` of its nodes over 60 frames.

## Python in TD: plain modules and Script ops

- A plain module (`python/*.py`) doesn't see TD's globals: no `op`, `run`, `nullCOMP`, `ParMode`.
  Use `import td` (`td.op`, `td.run`, `td.project`), or match `o.OPType == 'nullCOMP'` instead of
  `type=nullCOMP`.
- A Script CHOP's data can only be written inside its own `onCook`; calling the cook function from
  elsewhere raises "Expected locked operator, or execution within onCook". Reading `numChans`
  inside `onCook` raises too ("unavailable for this CHOP while it is cooking"): keep your own
  record of what you created.
- Writing ~200 channels: keep the `Channel` objects from `appendChan` and assign `chan[0]` on later
  cooks instead of `clear()` + `appendChan` every frame.
- Reading a wide table (Holistic's ~2500 columns): split `dat.text` once instead of `dat.row(i)`,
  which builds a Cell object per column (read_detections 2.2 → 0.7 ms).
- Writing many rows: `appendRows(rows)` in one call beats a loop of `appendRow`.

## Tracking: start fresh after a gap

Smoothing that eases from a held position looked like overlays "flying in" whenever something
was found again. Two rules, shared by every tracker pipeline (`object_tracker.reacquired()`,
`box_smooth(snap=)`):

- A track matched again after 2 or more lost frames starts at its new detection.
- A skeleton part that went missing (Holistic's hands and face, which come in at 0, 0) holds its
  last position hidden, restarts at its new reading, and stays hidden for 3 frames while smoothing
  settles.
