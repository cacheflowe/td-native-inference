#include "Kernels.h"

#include <cuda_fp16.h>

#include <algorithm>

namespace
{
	__device__ __forceinline__ float u8(unsigned char v) { return v * (1.f / 255.f); }
	__device__ __forceinline__ float u16(unsigned short v) { return v * (1.f / 65535.f); }
	__device__ __forceinline__ float h16(unsigned short v) { return __half2float(__ushort_as_half(v)); }

	// Reads one texel as RGBA float (x/y in texel indices, already clamped).
	__device__ float4 fetch(cudaSurfaceObject_t s, TexFormat fmt, int x, int y)
	{
		switch (fmt)
		{
			case TexFormat::BGRA8:
			{
				uchar4 c = surf2Dread<uchar4>(s, x * 4, y);
				return make_float4(u8(c.z), u8(c.y), u8(c.x), u8(c.w));
			}
			case TexFormat::RGBA8:
			{
				uchar4 c = surf2Dread<uchar4>(s, x * 4, y);
				return make_float4(u8(c.x), u8(c.y), u8(c.z), u8(c.w));
			}
			case TexFormat::RGBA16Fixed:
			{
				ushort4 c = surf2Dread<ushort4>(s, x * 8, y);
				return make_float4(u16(c.x), u16(c.y), u16(c.z), u16(c.w));
			}
			case TexFormat::RGBA16Float:
			{
				ushort4 c = surf2Dread<ushort4>(s, x * 8, y);
				return make_float4(h16(c.x), h16(c.y), h16(c.z), h16(c.w));
			}
			case TexFormat::RGBA32Float:
				return surf2Dread<float4>(s, x * 16, y);
			case TexFormat::Mono8:
			{
				float v = u8(surf2Dread<unsigned char>(s, x, y));
				return make_float4(v, v, v, 1.f);
			}
			case TexFormat::Mono16Float:
			{
				float v = h16(surf2Dread<unsigned short>(s, x * 2, y));
				return make_float4(v, v, v, 1.f);
			}
			case TexFormat::Mono32Float:
			{
				float v = surf2Dread<float>(s, x * 4, y);
				return make_float4(v, v, v, 1.f);
			}
			default:
				return make_float4(0.f, 0.f, 0.f, 1.f);
		}
	}

	__device__ __forceinline__ float4 lerp4(float4 a, float4 b, float t)
	{
		return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
	}

	// Bilinear sample at continuous pixel coords (texel centers at i + 0.5), clamp-to-edge.
	__device__ float4 sampleBilinear(cudaSurfaceObject_t s, TexFormat fmt, int w, int h, float fx, float fy)
	{
		fx -= 0.5f;
		fy -= 0.5f;
		int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
		float tx = fx - x0, ty = fy - y0;
		int x1 = min(max(x0 + 1, 0), w - 1), y1 = min(max(y0 + 1, 0), h - 1);
		x0 = min(max(x0, 0), w - 1);
		y0 = min(max(y0, 0), h - 1);
		float4 top = lerp4(fetch(s, fmt, x0, y0), fetch(s, fmt, x1, y0), tx);
		float4 bot = lerp4(fetch(s, fmt, x0, y1), fetch(s, fmt, x1, y1), tx);
		return lerp4(top, bot, ty);
	}

	__device__ __forceinline__ float bilerpPlane(const float* p, int w, int h, float fx, float fy)
	{
		fx -= 0.5f;
		fy -= 0.5f;
		int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
		float tx = fx - x0, ty = fy - y0;
		int x1 = min(max(x0 + 1, 0), w - 1), y1 = min(max(y0 + 1, 0), h - 1);
		x0 = min(max(x0, 0), w - 1);
		y0 = min(max(y0, 0), h - 1);
		float a = p[y0 * w + x0] + (p[y0 * w + x1] - p[y0 * w + x0]) * tx;
		float b = p[y1 * w + x0] + (p[y1 * w + x1] - p[y1 * w + x0]) * tx;
		return a + (b - a) * ty;
	}

	template <typename T>
	__device__ __forceinline__ T toElem(float v);
	template <>
	__device__ __forceinline__ float toElem<float>(float v) { return v; }
	template <>
	__device__ __forceinline__ __half toElem<__half>(float v) { return __float2half(v); }
	template <>
	__device__ __forceinline__ int toElem<int>(float v) { return (int)fminf(fmaxf(v, -2147483520.f), 2147483520.f); }
	template <>
	__device__ __forceinline__ unsigned char toElem<unsigned char>(float v) { return (unsigned char)fminf(fmaxf(v, 0.f), 255.f); }

	template <typename T>
	__global__ void kPreprocess(cudaSurfaceObject_t src, TexFormat fmt, Letterbox lb, InputNorm norm,
		bool nhwc, bool bgr, T* dst, float pad)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= lb.dstW || y >= lb.dstH)
			return;

		// Model pixel center -> continuous source coords (top-down).
		float sx = (x + 0.5f - lb.padX) / lb.scaleX;
		float sy = (y + 0.5f - lb.padY) / lb.scaleY;

		float4 c;
		if (sx < 0.f || sy < 0.f || sx >= lb.srcW || sy >= lb.srcH)
			c = make_float4(pad, pad, pad, 1.f);
		else
			c = sampleBilinear(src, fmt, lb.srcW, lb.srcH, sx, lb.srcH - sy);	// flip: TD rows are bottom-up

		float ch[3] = { bgr ? c.z : c.x, c.y, bgr ? c.x : c.z };
		size_t plane = (size_t)lb.dstW * lb.dstH;
		size_t i = (size_t)y * lb.dstW + x;
		for (int k = 0; k < 3; k++)
		{
			T v = toElem<T>(ch[k] * norm.scale[k] + norm.offset[k]);
			if (nhwc)
				dst[i * 3 + k] = v;
			else
				dst[k * plane + i] = v;
		}
	}

	template <typename T>
	__device__ __forceinline__ void writePixel(T* dst, size_t i, size_t plane, bool nhwc, bool bgr,
		const InputNorm& norm, float4 c)
	{
		float ch[3] = { bgr ? c.z : c.x, c.y, bgr ? c.x : c.z };
		for (int k = 0; k < 3; k++)
		{
			T v = toElem<T>(ch[k] * norm.scale[k] + norm.offset[k]);
			if (nhwc)
				dst[i * 3 + k] = v;
			else
				dst[k * plane + i] = v;
		}
	}

	template <typename T>
	__global__ void kCropPreprocess(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
		const CropBox* boxes, int numBoxes, InputNorm norm, bool nhwc, bool bgr, T* dst, int w, int h)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;	// top-down
		int k = blockIdx.z;
		if (x >= w || y >= h)
			return;
		size_t plane = (size_t)w * h;
		T* item = dst + (size_t)k * plane * 3;
		size_t i = (size_t)y * w + x;
		if (k >= numBoxes)
		{
			for (int c = 0; c < 3; c++)
				item[nhwc ? i * 3 + c : c * plane + i] = toElem<T>(0.f);
			return;
		}
		CropBox b = boxes[k];
		float sx = b.x1 + (x + 0.5f) * (b.x2 - b.x1) / w;
		float sy = b.y1 + (y + 0.5f) * (b.y2 - b.y1) / h;
		// Pose crops (a person box grown to the model's aspect) can reach past the image: black there,
		// like the affine warp the models were trained with. Classify crops are clamped inside.
		float4 c = (sx < 0.f || sy < 0.f || sx >= srcW || sy >= srcH) ? make_float4(0.f, 0.f, 0.f, 1.f)
			: sampleBilinear(src, fmt, srcW, srcH, sx, srcH - sy);	// flip: TD rows are bottom-up
		writePixel<T>(item, i, plane, nhwc, bgr, norm, c);
	}

	// SimCC decode: one block per (crop, keypoint). out[4] = argmax bin of x, y, z and the score
	// 0.5 * (peak x + peak y), as mmpose does. A zero-length axis (no z head) gives bin 0.
	__global__ void kSimccArgmax(const float* sx, const float* sy, const float* sz, int nx, int ny, int nz, float* out)
	{
		const size_t ck = blockIdx.x;
		const float* arr[3] = { sx + ck * nx, sy + ck * ny, sz ? sz + ck * nz : nullptr };
		const int len[3] = { nx, ny, sz ? nz : 0 };
		__shared__ float sv[128];
		__shared__ int si[128];
		float best[3];
		int bi[3];
		for (int a = 0; a < 3; a++)
		{
			float m = -3.4e38f;
			int mi = 0;
			for (int t = threadIdx.x; t < len[a]; t += blockDim.x)
				if (arr[a][t] > m)
				{
					m = arr[a][t];
					mi = t;
				}
			sv[threadIdx.x] = m;
			si[threadIdx.x] = mi;
			__syncthreads();
			for (int s = blockDim.x / 2; s > 0; s >>= 1)
			{
				if (threadIdx.x < s && (sv[threadIdx.x + s] > sv[threadIdx.x] ||
					(sv[threadIdx.x + s] == sv[threadIdx.x] && si[threadIdx.x + s] < si[threadIdx.x])))
				{
					sv[threadIdx.x] = sv[threadIdx.x + s];
					si[threadIdx.x] = si[threadIdx.x + s];
				}
				__syncthreads();
			}
			best[a] = sv[0];
			bi[a] = si[0];
			__syncthreads();
		}
		if (threadIdx.x == 0)
		{
			out[ck * 4 + 0] = (float)bi[0];
			out[ck * 4 + 1] = (float)bi[1];
			out[ck * 4 + 2] = len[2] ? (float)bi[2] : 0.f;
			out[ck * 4 + 3] = 0.5f * (best[0] + best[1]);
		}
	}

	__global__ void kProtoMasks(const float* det, int stride, int coefOffset, const SegDet* dets, int numDets,
		const float* protos, int numCoef, int ph, int pw, int modelW, int modelH, float* out)
	{
		int p = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.y;
		if (p >= ph * pw || k >= numDets)
			return;

		int px = p % pw, py = p / pw;
		SegDet d = dets[k];
		float mx = (px + 0.5f) * modelW / pw;
		float my = (py + 0.5f) * modelH / ph;
		// 1 inside the box, easing to 0 across the margin outside it.
		auto fade = [](float outside, float margin) {
			if (outside <= 0.f)
				return 1.f;
			if (margin <= 0.f)
				return 0.f;
			float t = fminf(outside / margin, 1.f);
			return 1.f - t * t * (3.f - 2.f * t);
		};
		float w = fade(fmaxf(d.x1 - mx, mx - d.x2), d.marginX) * fade(fmaxf(d.y1 - my, my - d.y2), d.marginY);
		float v = 0.f;
		if (w > 0.f)
		{
			const float* coef = det + (size_t)d.row * stride + coefOffset;
			float acc = 0.f;
			size_t plane = (size_t)ph * pw;
			for (int c = 0; c < numCoef; c++)
				acc += coef[c] * protos[c * plane + p];
			v = w / (1.f + __expf(-acc));
		}
		out[(size_t)k * ph * pw + p] = v;
	}

	// counts[k] += pixels of mask k above threshold (masks are already zero outside the box).
	__global__ void kMaskCounts(const float* masks, int numDets, int planeSize, float threshold, int* counts)
	{
		int p = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.y;
		bool on = p < planeSize && k < numDets && masks[(size_t)k * planeSize + p] > threshold;
		// One atomic per warp instead of per pixel.
		unsigned ballot = __ballot_sync(0xffffffffu, on);
		if ((threadIdx.x & 31) == 0 && ballot && k < numDets)
			atomicAdd(&counts[k], __popc(ballot));
	}

	__device__ __forceinline__ void writeHalf4(cudaSurfaceObject_t s, int x, int y, float4 c)
	{
		ushort4 v = make_ushort4(__half_as_ushort(__float2half(c.x)), __half_as_ushort(__float2half(c.y)),
			__half_as_ushort(__float2half(c.z)), __half_as_ushort(__float2half(c.w)));
		surf2Dwrite(v, s, x * 8, y);
	}

	__global__ void kCropSheet(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
		const CropBox* boxes, int numBoxes, int cols, int cell, cudaSurfaceObject_t out, int outW, int outH)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;	// texture row (bottom-up)
		if (x >= outW || y >= outH)
			return;
		int yt = outH - 1 - y;
		int k = (yt / cell) * cols + x / cell;
		float4 c = make_float4(0.f, 0.f, 0.f, 1.f);
		if (x / cell < cols && k < numBoxes)
		{
			CropBox b = boxes[k];
			float sx = b.x1 + ((x % cell) + 0.5f) * (b.x2 - b.x1) / cell;
			float sy = b.y1 + ((yt % cell) + 0.5f) * (b.y2 - b.y1) / cell;
			c = sampleBilinear(src, fmt, srcW, srcH, sx, srcH - sy);
			c.w = 1.f;
		}
		writeHalf4(out, x, y, c);
	}

	__global__ void kComposeMatte(const float* masks, const SegDet* dets, int numDets, int ph, int pw, Letterbox lb,
		bool instances, float idThreshold, cudaSurfaceObject_t out, int outW, int outH)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;	// texture row (bottom-up)
		if (x >= outW || y >= outH)
			return;

		float yTop = outH - 1 - y + 0.5f;
		float sx = (x + 0.5f) * lb.srcW / outW;
		float sy = yTop * lb.srcH / outH;
		float mx = sx * lb.scaleX + lb.padX;
		float my = sy * lb.scaleY + lb.padY;
		float fx = mx * pw / lb.dstW;
		float fy = my * ph / lb.dstH;

		float m = 0.f;
		int best = -1;
		for (int k = 0; k < numDets; k++)
		{
			float v = bilerpPlane(masks + (size_t)k * ph * pw, pw, ph, fx, fy);
			if (v > m)
			{
				m = v;
				best = k;
			}
		}
		if (!instances)
		{
			writeHalf4(out, x, y, make_float4(m, m, m, m));
			return;
		}
		// Where people overlap, the pixel belongs to the mask that's most confident there.
		bool owned = best >= 0 && m > idThreshold;
		writeHalf4(out, x, y, make_float4(m, owned ? (float)(best + 1) : 0.f, owned ? (float)(dets[best].cls + 1) : 0.f, m));
	}

	template <typename T>
	__device__ __forceinline__ float fromElem(const T* p, size_t i);
	template <>
	__device__ __forceinline__ float fromElem<float>(const float* p, size_t i) { return p[i]; }
	template <>
	__device__ __forceinline__ float fromElem<__half>(const __half* p, size_t i) { return __half2float(p[i]); }

	template <typename T>
	__device__ float bilerpT(const T* p, int w, int h, float fx, float fy)
	{
		fx -= 0.5f;
		fy -= 0.5f;
		int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
		float tx = fx - x0, ty = fy - y0;
		int x1 = min(max(x0 + 1, 0), w - 1), y1 = min(max(y0 + 1, 0), h - 1);
		x0 = min(max(x0, 0), w - 1);
		y0 = min(max(y0, 0), h - 1);
		float v00 = fromElem<T>(p, (size_t)y0 * w + x0), v10 = fromElem<T>(p, (size_t)y0 * w + x1);
		float v01 = fromElem<T>(p, (size_t)y1 * w + x0), v11 = fromElem<T>(p, (size_t)y1 * w + x1);
		float a = v00 + (v10 - v00) * tx;
		float b = v01 + (v11 - v01) * tx;
		return a + (b - a) * ty;
	}

	template <typename T>
	__global__ void kTensorToTexture(const T* data, int C, int H, int W, const TensorRange* range,
		bool mapToSource, Letterbox lb, bool mono, cudaSurfaceObject_t out, int outW, int outH)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;	// texture row (bottom-up)
		if (x >= outW || y >= outH)
			return;
		size_t plane = (size_t)H * W;
		int nc = mono ? 1 : min(C, 4);
		float v[4] = { 0.f, 0.f, 0.f, 1.f };
		if (mapToSource)
		{
			// Output pixel -> source (top-down) -> model pixel -> tensor pixel.
			float sx = (x + 0.5f) * lb.srcW / outW;
			float sy = (outH - 1 - y + 0.5f) * lb.srcH / outH;
			float fx = (sx * lb.scaleX + lb.padX) * W / lb.dstW;
			float fy = (sy * lb.scaleY + lb.padY) * H / lb.dstH;
			for (int c = 0; c < nc; c++)
				v[c] = bilerpT<T>(data + c * plane, W, H, fx, fy);
		}
		else
		{
			size_t i = (size_t)(H - 1 - y) * W + x;	// tensor row 0 -> top of image
			for (int c = 0; c < nc; c++)
				v[c] = fromElem<T>(data, c * plane + i);
		}
		if (range)
		{
			float lo = range->lo, span = range->hi - range->lo;
			float inv = span > 1e-12f ? 1.f / span : 0.f;
			for (int c = 0; c < nc; c++)
				v[c] = (v[c] - lo) * inv;
		}
		if (mono)
		{
			surf2Dwrite(v[0], out, x * 4, y);
			return;
		}
		if (C == 1)
			v[1] = v[2] = v[0];
		surf2Dwrite(make_float4(v[0], v[1], v[2], v[3]), out, x * 16, y);
	}

	// Order-preserving float <-> uint, so atomicMin/atomicMax work on floats.
	__device__ __forceinline__ unsigned int floatToOrd(float f)
	{
		unsigned int u = __float_as_uint(f);
		return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
	}
	__device__ __forceinline__ float ordToFloat(unsigned int u)
	{
		return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
	}

	__global__ void kRangeReset(TensorRange* r)
	{
		r->minOrd = 0xffffffffu;
		r->maxOrd = 0u;
	}

	template <typename T>
	__global__ void kRange(const T* data, int C, int H, int W, Letterbox lb, TensorRange* r)
	{
		// Only tensor pixels inside the letterboxed image count -- the model also predicts
		// values for the padding bars, which would skew the range.
		float x0 = lb.padX * W / lb.dstW, x1 = (lb.dstW - lb.padX) * W / lb.dstW;
		float y0 = lb.padY * H / lb.dstH, y1 = (lb.dstH - lb.padY) * H / lb.dstH;
		size_t plane = (size_t)H * W, count = plane * C;
		float lo = INFINITY, hi = -INFINITY;
		for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
		{
			size_t p = i % plane;
			float px = (float)(p % W) + 0.5f, py = (float)(p / W) + 0.5f;
			if (px < x0 || px > x1 || py < y0 || py > y1)
				continue;
			float v = fromElem<T>(data, i);
			if (isfinite(v))
			{
				lo = fminf(lo, v);
				hi = fmaxf(hi, v);
			}
		}
		for (int o = 16; o > 0; o >>= 1)
		{
			lo = fminf(lo, __shfl_down_sync(0xffffffffu, lo, o));
			hi = fmaxf(hi, __shfl_down_sync(0xffffffffu, hi, o));
		}
		if ((threadIdx.x & 31) == 0 && lo <= hi)
		{
			atomicMin(&r->minOrd, floatToOrd(lo));
			atomicMax(&r->maxOrd, floatToOrd(hi));
		}
	}

	__global__ void kRangeFinish(TensorRange* r, float smoothing, bool reset)
	{
		if (r->minOrd == 0xffffffffu)
			return;	// no finite values this frame: keep the previous range
		float lo = ordToFloat(r->minOrd), hi = ordToFloat(r->maxOrd);
		if (reset || !r->valid)
		{
			r->lo = lo;
			r->hi = hi;
			r->valid = 1;
		}
		else
		{
			r->lo = lo + (r->lo - lo) * smoothing;
			r->hi = hi + (r->hi - hi) * smoothing;
		}
	}

	__global__ void kClear(cudaSurfaceObject_t out, int w, int h, int bpp)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= w || y >= h)
			return;
		if (bpp == 4)
			surf2Dwrite(0.f, out, x * 4, y);
		else if (bpp == 8)
			surf2Dwrite(make_ushort4(0, 0, 0, 0), out, x * 8, y);
		else
			surf2Dwrite(make_float4(0.f, 0.f, 0.f, 0.f), out, x * 16, y);
	}

	dim3 grid2D(int w, int h, dim3 block) { return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y); }
}

Letterbox computeLetterbox(int srcW, int srcH, int dstW, int dstH, bool keepAspect)
{
	Letterbox lb;
	lb.srcW = srcW;
	lb.srcH = srcH;
	lb.dstW = dstW;
	lb.dstH = dstH;
	if (keepAspect)
	{
		float s = std::min((float)dstW / srcW, (float)dstH / srcH);
		lb.scaleX = lb.scaleY = s;
		lb.padX = (dstW - srcW * s) * 0.5f;
		lb.padY = (dstH - srcH * s) * 0.5f;
	}
	else
	{
		lb.scaleX = (float)dstW / srcW;
		lb.scaleY = (float)dstH / srcH;
	}
	return lb;
}

InputNorm makeInputNorm(InputNormKind kind)
{
	InputNorm n;
	if (kind == InputNormKind::ImageNet)
	{
		const float mean[3] = { 0.485f, 0.456f, 0.406f }, stdev[3] = { 0.229f, 0.224f, 0.225f };
		for (int c = 0; c < 3; c++)
		{
			n.scale[c] = 1.f / stdev[c];
			n.offset[c] = -mean[c] / stdev[c];
		}
	}
	else if (kind == InputNormKind::Byte)
	{
		for (int c = 0; c < 3; c++)
			n.scale[c] = 255.f;
	}
	else if (kind == InputNormKind::Signed)
	{
		for (int c = 0; c < 3; c++)
		{
			n.scale[c] = 2.f;
			n.offset[c] = -1.f;
		}
	}
	return n;
}

cudaError_t launchPreprocess(cudaSurfaceObject_t src, TexFormat fmt, const Letterbox& lb,
	const InputNorm& norm, const InputLayout& layout, void* dst, float padValue, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid = grid2D(lb.dstW, lb.dstH, block);
	bool nhwc = layout.nhwc, bgr = layout.bgr;
	switch (layout.elem)
	{
		case TensorElem::F16:
			kPreprocess<__half><<<grid, block, 0, stream>>>(src, fmt, lb, norm, nhwc, bgr, (__half*)dst, padValue);
			break;
		case TensorElem::I32:
			kPreprocess<int><<<grid, block, 0, stream>>>(src, fmt, lb, norm, nhwc, bgr, (int*)dst, padValue);
			break;
		case TensorElem::U8:
			kPreprocess<unsigned char><<<grid, block, 0, stream>>>(src, fmt, lb, norm, nhwc, bgr, (unsigned char*)dst, padValue);
			break;
		default:
			kPreprocess<float><<<grid, block, 0, stream>>>(src, fmt, lb, norm, nhwc, bgr, (float*)dst, padValue);
			break;
	}
	return cudaGetLastError();
}

cudaError_t launchProtoMasks(const float* det, int detStride, int coefOffset,
	const SegDet* dets, int numDets,
	const float* protos, int numCoef, int protoH, int protoW,
	int modelW, int modelH, float* masksOut, cudaStream_t stream)
{
	if (numDets <= 0)
		return cudaSuccess;
	dim3 block(256);
	dim3 grid((protoH * protoW + block.x - 1) / block.x, numDets);
	kProtoMasks<<<grid, block, 0, stream>>>(det, detStride, coefOffset, dets, numDets,
		protos, numCoef, protoH, protoW, modelW, modelH, masksOut);
	return cudaGetLastError();
}

cudaError_t launchMaskCounts(const float* masks, int numDets, int protoH, int protoW,
	float threshold, int* countsOut, cudaStream_t stream)
{
	if (numDets <= 0)
		return cudaSuccess;
	cudaMemsetAsync(countsOut, 0, numDets * sizeof(int), stream);
	dim3 block(256);
	dim3 grid((protoH * protoW + block.x - 1) / block.x, numDets);
	kMaskCounts<<<grid, block, 0, stream>>>(masks, numDets, protoH * protoW, threshold, countsOut);
	return cudaGetLastError();
}

cudaError_t launchComposeMatte(const float* masks, const SegDet* dets, int numDets, int protoH, int protoW,
	const Letterbox& lb, bool instances, float idThreshold, cudaSurfaceObject_t out, int outW, int outH,
	cudaStream_t stream)
{
	dim3 block(16, 16);
	kComposeMatte<<<grid2D(outW, outH, block), block, 0, stream>>>(masks, dets, numDets, protoH, protoW, lb,
		instances, idThreshold, out, outW, outH);
	return cudaGetLastError();
}

cudaError_t launchTensorRange(const void* data, bool isHalf, int C, int H, int W, const Letterbox& lb,
	float smoothing, bool reset, TensorRange* range, cudaStream_t stream)
{
	size_t count = (size_t)C * H * W;
	if (reset)
		cudaMemsetAsync(range, 0, sizeof(TensorRange), stream);
	kRangeReset<<<1, 1, 0, stream>>>(range);
	dim3 block(256);
	dim3 grid((unsigned)std::min<size_t>((count + block.x - 1) / block.x, 1024));
	if (grid.x == 0)
		grid.x = 1;
	if (isHalf)
		kRange<__half><<<grid, block, 0, stream>>>((const __half*)data, C, H, W, lb, range);
	else
		kRange<float><<<grid, block, 0, stream>>>((const float*)data, C, H, W, lb, range);
	kRangeFinish<<<1, 1, 0, stream>>>(range, smoothing, reset);
	return cudaGetLastError();
}

cudaError_t launchTensorToTexture(const void* data, bool isHalf, int C, int H, int W,
	const TensorRange* range, bool mapToSource, const Letterbox& lb, bool mono,
	cudaSurfaceObject_t out, int outW, int outH, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid = grid2D(outW, outH, block);
	if (isHalf)
		kTensorToTexture<__half><<<grid, block, 0, stream>>>((const __half*)data, C, H, W, range, mapToSource, lb, mono, out, outW, outH);
	else
		kTensorToTexture<float><<<grid, block, 0, stream>>>((const float*)data, C, H, W, range, mapToSource, lb, mono, out, outW, outH);
	return cudaGetLastError();
}

cudaError_t launchClear(cudaSurfaceObject_t out, int outW, int outH, int bytesPerPixel, cudaStream_t stream)
{
	dim3 block(16, 16);
	kClear<<<grid2D(outW, outH, block), block, 0, stream>>>(out, outW, outH, bytesPerPixel);
	return cudaGetLastError();
}

cudaError_t launchSimccArgmax(const float* sx, const float* sy, const float* sz, int numCrops, int K,
	int nx, int ny, int nz, float* out, cudaStream_t stream)
{
	if (numCrops <= 0 || K <= 0)
		return cudaSuccess;
	kSimccArgmax<<<numCrops * K, 128, 0, stream>>>(sx, sy, sz, nx, ny, nz, out);
	return cudaGetLastError();
}

cudaError_t launchCropPreprocess(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
	const CropBox* boxes, int numBoxes, int batch, const InputNorm& norm, const InputLayout& layout,
	void* dst, int dstW, int dstH, cudaStream_t stream)
{
	dim3 block(16, 16);
	dim3 grid = grid2D(dstW, dstH, block);
	grid.z = batch;
	bool nhwc = layout.nhwc, bgr = layout.bgr;
	switch (layout.elem)
	{
		case TensorElem::F16:
			kCropPreprocess<__half><<<grid, block, 0, stream>>>(src, fmt, srcW, srcH, boxes, numBoxes, norm, nhwc, bgr, (__half*)dst, dstW, dstH);
			break;
		case TensorElem::I32:
			kCropPreprocess<int><<<grid, block, 0, stream>>>(src, fmt, srcW, srcH, boxes, numBoxes, norm, nhwc, bgr, (int*)dst, dstW, dstH);
			break;
		case TensorElem::U8:
			kCropPreprocess<unsigned char><<<grid, block, 0, stream>>>(src, fmt, srcW, srcH, boxes, numBoxes, norm, nhwc, bgr, (unsigned char*)dst, dstW, dstH);
			break;
		default:
			kCropPreprocess<float><<<grid, block, 0, stream>>>(src, fmt, srcW, srcH, boxes, numBoxes, norm, nhwc, bgr, (float*)dst, dstW, dstH);
			break;
	}
	return cudaGetLastError();
}

cudaError_t launchCropSheet(cudaSurfaceObject_t src, TexFormat fmt, int srcW, int srcH,
	const CropBox* boxes, int numBoxes, int cols, int cell, cudaSurfaceObject_t out, int outW, int outH,
	cudaStream_t stream)
{
	dim3 block(16, 16);
	kCropSheet<<<grid2D(outW, outH, block), block, 0, stream>>>(src, fmt, srcW, srcH, boxes, numBoxes, cols, cell, out, outW, outH);
	return cudaGetLastError();
}
