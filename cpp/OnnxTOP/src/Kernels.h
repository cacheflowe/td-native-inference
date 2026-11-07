// CUDA kernels for OnnxTOP. All image I/O goes through surface objects bound to the
// cudaArrays TouchDesigner hands out (TOP textures are stored bottom row first; every
// kernel here converts to/from top-down "image" order internally).
#pragma once

#include <cuda_runtime.h>

enum class TexFormat : int
{
	BGRA8,
	RGBA8,
	RGBA16Fixed,
	RGBA16Float,
	RGBA32Float,
	Mono8,
	Mono16Float,
	Mono32Float,
	Unsupported,
};

// Maps model-input space <-> source-image space. Model pixel = src pixel * scale + pad.
struct Letterbox
{
	int		srcW = 0, srcH = 0;
	int		dstW = 0, dstH = 0;
	float	scaleX = 1.f, scaleY = 1.f;
	float	padX = 0.f, padY = 0.f;
};

Letterbox computeLetterbox(int srcW, int srcH, int dstW, int dstH, bool keepAspect);

// Per-channel input normalization applied after sampling: tensor = rgb01 * scale + offset
// (per channel in MODEL channel order, i.e. after an RGB->BGR swap).
enum class InputNormKind : int { Unit = 0, ImageNet = 1, Signed = 2, Byte = 3, Custom = 4 };
struct InputNorm
{
	float	scale[3] = { 1.f, 1.f, 1.f };
	float	offset[3] = { 0.f, 0.f, 0.f };
};
InputNorm makeInputNorm(InputNormKind kind);	// Custom returns identity; fill it yourself

// Model input tensor element type.
enum class TensorElem : int { F32, F16, I32, U8 };

struct InputLayout
{
	TensorElem	elem = TensorElem::F32;
	bool		nhwc = false;	// [1, H, W, 3] (e.g. MoveNet) instead of [1, 3, H, W]
	bool		bgr = false;	// channel order fed to the model
};

// Resize/letterbox the source texture into a 3-channel tensor (layout.nhwc ? NHWC : NCHW),
// normalized with `norm` (padding is the 0..1 padValue, normalized the same way). Integer
// element types truncate after clamping to their range, like numpy's astype.
cudaError_t launchPreprocess(cudaSurfaceObject_t src, TexFormat fmt, const Letterbox& lb,
	const InputNorm& norm, const InputLayout& layout, void* dst, float padValue, cudaStream_t stream);

// A crop of the source image, in source pixels, top-down (row 0 = top of the image).
struct CropBox
{
	float	x1, y1, x2, y2;
};

// Crop-and-classify input: batch item k (< numBoxes) is boxes[k] resized to dstW x dstH
// (bilinear), normalized like launchPreprocess; items numBoxes..batch-1 are zeroed.
// boxes is device memory.
cudaError_t launchCropPreprocess(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
	const CropBox* boxes, int numBoxes, int batch, const InputNorm& norm, const InputLayout& layout,
	void* dst, int dstW, int dstH, cudaStream_t stream);

// Debug view of the crops: a grid of `cols` columns of cell x cell tiles (crop k at row-major
// position k, top-left first) written to an RGBA16Float surface; empty tiles are black.
cudaError_t launchCropSheet(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
	const CropBox* boxes, int numBoxes, int cols, int cell, cudaSurfaceObject_t out, int outW, int outH,
	cudaStream_t stream);

// One kept detection, for mask decode. Box in model-input pixels.
struct SegDet
{
	float	x1, y1, x2, y2;
	int		row;	// row index into the [N, stride] detection tensor
};

// masksOut[k][py][px] = sigmoid(coeffs_k . protos[:, py, px]), zero outside det k's box.
cudaError_t launchProtoMasks(const float* det, int detStride, int coefOffset,
	const SegDet* dets, int numDets,
	const float* protos, int numCoef, int protoH, int protoW,
	int modelW, int modelH, float* masksOut, cudaStream_t stream);

// countsOut[k] = number of pixels of mask k above threshold (device int[numDets]).
cudaError_t launchMaskCounts(const float* masks, int numDets, int protoH, int protoW,
	float threshold, int* countsOut, cudaStream_t stream);

// Writes the max over all decoded masks, mapped back into source-image space, to an
// RGBA16Float output surface (outW x outH). numDets == 0 writes zeros.
cudaError_t launchComposeMatte(const float* masks, int numDets, int protoH, int protoW,
	const Letterbox& lb, cudaSurfaceObject_t out, int outW, int outH, cudaStream_t stream);

// Device-resident running value range of a tensor, for normalizing raw outputs without a
// CPU round trip. Allocate with cudaMalloc; launchTensorRange() maintains it.
struct TensorRange
{
	unsigned int	minOrd, maxOrd;		// this frame's min/max as order-preserving uints
	float			lo, hi;				// temporally smoothed range used for normalizing
	int				valid;				// lo/hi hold a real value (0 = take this frame as-is)
};

// Updates range->lo/hi from the first C planes of an [.., C, H, W] tensor: lo/hi =
// lerp(frame range, previous, smoothing). Only pixels inside the letterboxed image count (tensor
// W x H spans lb.dstW x lb.dstH model pixels). reset = forget the previous range. Non-finite
// values are ignored.
cudaError_t launchTensorRange(const void* data, bool isHalf, int C, int H, int W, const Letterbox& lb,
	float smoothing, bool reset, TensorRange* range, cudaStream_t stream);

// Writes an [.., C, H, W] tensor (C <= 4 used) into an output surface, tensor row 0 at the
// top of the image. C == 1 is replicated to RGB, alpha = 1.
//   range:   if non-null, values are mapped (v - lo) / (hi - lo).
//   mapToSource: output is outW x outH in source-image space and samples the tensor through
//            the letterbox (tensor W x H spans lb.dstW x lb.dstH model pixels); otherwise the
//            output must be W x H and is a 1:1 copy.
//   mono:    output is Mono32Float (channel 0 only) instead of RGBA32Float.
cudaError_t launchTensorToTexture(const void* data, bool isHalf, int C, int H, int W,
	const TensorRange* range, bool mapToSource, const Letterbox& lb, bool mono,
	cudaSurfaceObject_t out, int outW, int outH, cudaStream_t stream);

// Zeroes a surface. bytesPerPixel: 4 = Mono32Float, 8 = RGBA16Float, 16 = RGBA32Float.
cudaError_t launchClear(cudaSurfaceObject_t out, int outW, int outH, int bytesPerPixel, cudaStream_t stream);
