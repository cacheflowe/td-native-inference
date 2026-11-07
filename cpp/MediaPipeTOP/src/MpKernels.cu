#include "MpKernels.h"

namespace
{
	__global__ void kRgbToSurface(const uint8_t* rgb, int w, int h, cudaSurfaceObject_t out)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= w || y >= h)
			return;
		const uint8_t* p = rgb + ((size_t)(h - 1 - y) * w + x) * 3;
		surf2Dwrite(make_uchar4(p[0], p[1], p[2], 255), out, x * 4, y);
	}

	__global__ void kMaskToSurface(const float* mask, int w, int h, cudaSurfaceObject_t out)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= w || y >= h)
			return;
		surf2Dwrite(mask[(size_t)(h - 1 - y) * w + x], out, x * 4, y);
	}

	dim3 gridFor(int w, int h, dim3 block)
	{
		return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	}
}

cudaError_t launchRgbToSurface(const uint8_t* rgb, int w, int h, cudaSurfaceObject_t out, cudaStream_t stream)
{
	dim3 block(16, 16);
	kRgbToSurface<<<gridFor(w, h, block), block, 0, stream>>>(rgb, w, h, out);
	return cudaGetLastError();
}

cudaError_t launchMaskToSurface(const float* mask, int w, int h, cudaSurfaceObject_t out, cudaStream_t stream)
{
	dim3 block(16, 16);
	kMaskToSurface<<<gridFor(w, h, block), block, 0, stream>>>(mask, w, h, out);
	return cudaGetLastError();
}
