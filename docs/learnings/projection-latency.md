# Projection latency: what we measured (Nov 18, 2026)

End to end, with Latency Loop (flash the projector, count until the camera sees it, 40-50 samples,
median): TouchDesigner at 60 fps, RealSense D435 color, a 4K projector at 12 ft. The loop's spread
between runs is about ±15 ms, so a change under that is noise.

| Setup | Median (min – max) |
|---|---|
| RealSense TOP, 1280×720 (30 fps), defaults, earlier room setup | 208 ms (168 – 282) |
| RealSense TOP, 1280×720 (30 fps), defaults | 184 ms (156 – 232) |
| + `frames_queue_size` 2 (Options CHOP) | 201 ms (168 – 280) |
| + manual exposure 10 ms, gain 90, `auto_exposure_priority` 0, queue 2 | 212 ms (164 – 294) |
| RealSense TOP at 848×480 | the same 30 fps (measured 30.0) |
| **Video Device In TOP on the camera's color stream (UVC), 1280×720, 30 fps** | **162 ms (136 – 201)** |

## What it says

- **The RealSense TOP can't give a faster color stream.** It has no frame-rate parameter and stays at
  30 fps at every color resolution, 848×480 included. The D435's color sensor itself tops out at
  **60 Hz** (960×540, 848×480, 640×480, 640×360, 424×240 and below; from `rs-enumerate-devices`);
  **90 Hz exists only for depth and infrared.** Advice to run the RGB at 90 fps (848×480@90, project
  FPS 90) doesn't apply to this camera.
- **The Options CHOP works** (channels are the SDK option names lowercased: `enable_auto_exposure`,
  `exposure`, `gain`, `frames_queue_size`, ...), but none of it moved the latency at 30 fps, and a short
  manual exposure just darkens a dim room. The RGB sensor's defaults: exposure 156, gain 64, auto
  exposure priority already 0, frames queue size 16.
- **The SDK path costs about 25-40 ms.** Reading the same sensor as a plain camera through TD's
  Video Device In TOP (DirectShow, the `... RGB Module RGB` device, with the RealSense TOP's image
  set to Depth so it releases the color stream) measured 162 ms against 184-212. The trade: no
  depth aligned to color from the same TOP.
- **Not tried yet:** a 60 fps UVC mode (the Video Device In TOP's `Device Options` dialog picks the
  format; the menu is empty for DirectShow cameras), which should save another ~16 ms of frame
  period; another camera that runs 60+ fps; the projector's own lag (a game / low-latency mode, a 1080p
  signal instead of 4K upscaled); the model's delay (`result_delay`, Wait for Result off on Holistic).

## Where the rest of the 160-210 ms goes (estimates, not measured)

Camera exposure, readout and USB, then the driver / SDK buffering (the ~30 ms above), TD's cook at 60 fps
(a frame to cook the input, a frame for the model's result, a frame to render and present), then the
projector's processing and its 60 Hz scan-out (often the biggest single piece). Split them by changing
one thing at a time and re-running the loop, as above.
