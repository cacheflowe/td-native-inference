#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Packed RGB8 (top-down) -> RGBA8 surface (TD's bottom-up rows), alpha = 255.
cudaError_t launchRgbToSurface(const uint8_t* rgb, int w, int h, cudaSurfaceObject_t out, cudaStream_t stream);

// Float mask (top-down) -> Mono32Float surface (bottom-up).
cudaError_t launchMaskToSurface(const float* mask, int w, int h, cudaSurfaceObject_t out, cudaStream_t stream);
