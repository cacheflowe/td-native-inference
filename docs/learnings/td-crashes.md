# TD crashes, and deleting a model wrapper (Nov 2026)

TouchDesigner has exited on this machine several times while the native plugins were loaded. Most
were never explained. This note keeps what is known, what was ruled out, how to catch the next one,
and the workaround for the one that reproduces on demand: **deleting a running ONNX wrapper that
feeds a live network.**

## The reproducible one: deleting a running wrapper

**Symptom (Nov 18, 2026):** in `/project1/Demos/ProjectionRehearsal`, deleting the `RVM` wrapper
(an OnnxTOP, loaded, running at the camera's rate, its matte wired into Projector Map) took TD down
within seconds. No dialog the user could use, no Windows Application Error event, no
`TouchDesignerCrash*.dmp` next to the project.

**Reproduced twice, on both plugin builds:**
- On the build with the unload feature (`OnnxTOP.dll` of 11:24): crashed.
- On the Nov 17 build, from before any unloading existed (`OnnxTOP.dll.old-112446`), with the RVM
  loaded and running: crashed. (An earlier old-build try deleted an RVM that hadn't finished loading
  and survived, so a delete only counts once `loaded` is 1 and `infer_ms` is above 0.)

So the unload work is not the cause, and the bug is older.

**What does not crash:**

| Case | Result |
|---|---|
| Running scratch Yolo26Detect in the lab, nothing downstream, deleted | survived |
| Running scratch RVM in the lab, nothing downstream, deleted (also after `Unload`) | survived |
| Scratch RVM deleted while still loading | survived |
| Demo RVM after: output disconnected, Active off, `Unload` pulsed, cooking off | survived |
| Demo RVM, loaded and running, output wired to Projector Map | **crashed**, twice |

**Likely shape:** the delete happens while a downstream node still cooks the wrapper's output in the
same TD cook, so the TOP's output (a CUDA array TD allocated for it) is freed or the node destroyed
under something still using it. Not proven: there is no dump. Things the destructor does that could
matter: it destroys the node's CUDA streams and events, frees its own device buffers, and releases the
ONNX session (RVM is the only wrapper on the plain CUDA provider, and it carries recurrent state, so
a TensorRT wrapper may behave differently). The scratch cases above differ from the crashing one only
by having no consumer, which is why a consumer is the suspect.

**Workaround:** never delete a wrapper that is running.
1. To change a model, wire the new one in first, then retire the old one.
2. Disconnect the old one's outputs, set **Active** off, pulse **Unload**, switch its cooking off.
3. Delete it a few frames later, or leave it switched off in the network.

`wrapper_builder.remove(comp)` does exactly this (outputs cut, Active off, models unloaded, cooking
off, `destroy()` 60 frames later). Rebuilding wrappers and demos with the build scripts is not
affected: they replace a node's contents in a separate bridge call and have run many times.

## Other crashes on record

| When | What | Evidence | State |
|---|---|---|---|
| Nov 12 | TD exited silently, twice | no dump; Windows logged `RADAR_PRE_LEAK_64`; this machine has two unstable P-cores | unexplained |
| Nov 13 | Aborts when an ORT session was destroyed (its intra-op threads exiting ran the CUDA provider's per-thread cleanup) | crash dumps showed only onnxruntime / CUDA frames on the dying thread | fixed: sessions use one intra-op thread (`OnnxEngine::makeOptions`) |
| Nov 17 | Three crashes while a TensorRT engine was building in the background (wrapper rebuilds, a new model's first build) | no dump | mitigated: build engines with `onnxtop_tool` before TD opens the model (see `onnx-cpp-top.md`) |
| Nov 18 11:08 | `TouchDesigner.exe` faulted in `ucrtbase.dll`, exception `0xc0000409` (a fast-fail, i.e. `abort()` / an uncaught exception in a noexcept path) | Windows Application Error event only; before the unload code existed | trigger unknown |
| Nov 18 afternoon | The `RVM` deletion crashes above, plus one while wiring a Yolo26Seg matte and switching RVM off in the same step | no dump, no event | the delete is reproducible |

## Catching the next one

- **Lifecycle log (added Nov 18):** OnnxTOP appends to `plugins/onnxtop-lifecycle.log` (next to
  `OnnxTOP.dll`, gitignored with the build output) at `create`, each `destroy:` stage, and `unload:`
  start and done, with the node pointer and thread. After a crash, the last line says how far the
  destructor got: "destroy: start" or "releasing the ONNX session" last means the crash is in our
  teardown; "destroy: done" last means it is TD using the node afterwards.
- **Windows crash dumps:** the usual `TouchDesignerCrash.<build>_<n>.dmp` next to the project did not
  appear for these crashes (it did for the haxlib ones). To get a full dump regardless, run once as
  administrator: `scripts\enable_td_dumps.ps1` (registers Windows Error Reporting LocalDumps for
  `TouchDesigner.exe`, full dumps into `build\dumps\`). Read a dump with `cdb` (Windows SDK
  Debugging Tools; not installed here): `cdb -z <dump> -c "!analyze -v; k; q"`.
- **Reproduce with a log, not a guess:** delete a running RVM that feeds Projector Map, reopen TD, and
  read the last lines of the lifecycle log.

## What to fix

1. Read the lifecycle log from a reproduction to split "our destructor" from "TD after it".
2. If it is the destructor: sequence teardown so nothing the GPU still uses is freed first
   (a `cudaDeviceSynchronize()` before freeing, and the session released before the streams and
   events it was given are destroyed), and look at what the plain CUDA provider does at session
   release with the recurrent state tensors (`OnnxEngine::release`).
3. If it is TD using the output after the delete: the output is TD-owned; the fix would be to stop
   touching the output during teardown, or to give TD a final empty cook first.
4. Add a minimal repro to the wrapper lab (a running wrapper with a downstream GLSL TOP, deleted)
   so the fix has a test that does not need the projection demo.
