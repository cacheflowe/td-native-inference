// onnxtop_tool: companion executable for OnnxTOP, run against TouchDesigner's bundled ONNX
// Runtime/TensorRT DLLs outside of TD.
//
// Build mode (used by the TOP itself -- see TrtBuilder.h):
//   onnxtop_tool.exe build <TD bin dir> <model.onnx> <cache dir> <fp16 0|1> <device id> [<dyn W> <dyn H>]
//
// Bench mode:
//   onnxtop_tool.exe <TD bin dir> <model.onnx> [tensorrt|cuda] [iterations] [fp32]
//                    [--size <w> <h>] [--norm unit|imagenet|signed|byte] [--bgr] [--raw | --movenet]
//                    [--image <raw RGBA8 file> <w> <h> --out <prefix>]
//
// --size sets the input size for models with dynamic H/W (e.g. Depth Anything: multiples of 14).
// --raw replaces the YOLO decode with the TOP's Raw mode (min/max normalized output 0, mapped
// back to the image) and writes <prefix>_raw.pgm.
//
// Without --image: times bare inference on a zeroed input.
// With --image: emulates exactly what the TOP does with a TD texture -- uploads the image as a
// BGRA8 cudaArray stored bottom row first (TD's layout), then letterbox -> inference -> YOLO
// decode -> matte composite into an RGBA16Float cudaArray. Writes <prefix>_matte.pgm and
// <prefix>_overlay.ppm (top-down) and prints the detections.
#include "OnnxEngine.h"
#include "TrtBuilder.h"
#include "YoloDecoder.h"

#include <cuda_fp16.h>
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// Crash reporter: prints exception code, faulting module+offset and a short backtrace, so a
// crash inside TensorRT/ORT/CUDA DLLs can be attributed without a debugger installed.
static void describeAddress(void* addr)
{
	HMODULE mod = nullptr;
	char name[MAX_PATH] = "?";
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		(LPCSTR)addr, &mod))
		GetModuleFileNameA(mod, name, MAX_PATH);
	const char* base = strrchr(name, '\\');
	printf("    %p  %s+0x%llx\n", addr, base ? base + 1 : name, (unsigned long long)((char*)addr - (char*)mod));
}

static LONG WINAPI crashFilter(EXCEPTION_POINTERS* ep)
{
	printf("\n*** CRASH: exception 0x%08lX on thread %lu at:\n", ep->ExceptionRecord->ExceptionCode, GetCurrentThreadId());
	describeAddress(ep->ExceptionRecord->ExceptionAddress);
	if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
		printf("    (%s address %p)\n", ep->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
			(void*)ep->ExceptionRecord->ExceptionInformation[1]);
	printf("  backtrace (handler frames first):\n");
	void* frames[48];
	USHORT n = CaptureStackBackTrace(0, 48, frames, nullptr);
	for (USHORT i = 0; i < n; i++)
		describeAddress(frames[i]);
	printf("  recent ORT log:\n%s", ortload::recentLog(30).c_str());
	fflush(stdout);
	return EXCEPTION_EXECUTE_HANDLER;
}

static double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static void printStats(const char* label, std::vector<double> t)
{
	std::sort(t.begin(), t.end());
	double sum = 0;
	for (double v : t)
		sum += v;
	size_t n = t.size();
	printf("%-34s mean %.2f ms | p50 %.2f | p95 %.2f | min %.2f | max %.2f  (%zu runs)\n",
		label, sum / n, t[n / 2], t[n * 95 / 100], t.front(), t.back(), n);
}

static cudaSurfaceObject_t makeSurface(cudaArray_t a)
{
	cudaResourceDesc d = {};
	d.resType = cudaResourceTypeArray;
	d.res.array.array = a;
	cudaSurfaceObject_t s = 0;
	cudaCreateSurfaceObject(&s, &d);
	return s;
}

int main(int argc, char** argv)
{
	setvbuf(stdout, nullptr, _IONBF, 0);
	SetUnhandledExceptionFilter(crashFilter);

	// build <TD bin> <model> <cache dir> <fp16 0|1> <device> [<dynamic W> <dynamic H>]
	if ((argc == 7 || argc == 9 || argc == 10) && std::string(argv[1]) == "build")
	{
		std::string bin = argv[2];
		EngineConfig cfg;
		cfg.modelPath = argv[3];
		cfg.cacheDir = argv[4];
		cfg.fp16 = std::string(argv[5]) != "0";
		cfg.deviceId = atoi(argv[6]);
		if (argc >= 9)
		{
			cfg.inputW = atoi(argv[7]);
			cfg.inputH = atoi(argv[8]);
		}
		if (argc == 10)
			cfg.batch = atoi(argv[9]);
		return runBuildCommand(std::wstring(bin.begin(), bin.end()), cfg);
	}

	// Exercises the exact launcher the TOP uses (child process, PATH, log, retries, marker):
	//   onnxtop_tool.exe selftest-build <TD bin dir> <model.onnx>
	if (argc == 4 && std::string(argv[1]) == "selftest-build")
	{
		wchar_t self[MAX_PATH * 2];
		GetModuleFileNameW(nullptr, self, MAX_PATH * 2);
		std::string bin = argv[2];
		std::wstring wbin(bin.begin(), bin.end());
		SetDllDirectoryW(wbin.c_str());
		std::string err;
		if (!ortload::init(err, wbin))
		{
			printf("ORT init failed: %s\n", err.c_str());
			return 1;
		}
		EngineConfig cfg;
		cfg.modelPath = argv[3];
		printf("marker: %s (exists=%d)\n", trtBuildMarkerPath(cfg).c_str(), (int)GetFileAttributesA(trtBuildMarkerPath(cfg).c_str()) != -1);
		std::atomic<bool> cancel{ false };
		auto t = Clock::now();
		TrtBuildResult r = buildEngineOutOfProcess(self, wbin, cfg, 3, cancel,
			[](void*, int n) { printf("attempt %d...\n", n); });
		printf("ok=%d attempts=%d %.1f s %s\n", r.ok, r.attempts, msSince(t) / 1000.0, r.message.c_str());
		return r.ok ? 0 : 1;
	}

	std::vector<std::string> pos;
	std::string imagePath, outPrefix;
	int imgW = 0, imgH = 0, dynW = 0, dynH = 0;
	bool rawMode = false, moveNet = false, yuNet = false, bgr = false, graph = false;
	std::vector<CropBox> crops;	// --crop x1 y1 x2 y2 (source px, top-down), repeatable
	int batch = 0;
	InputNormKind normKind = InputNormKind::Unit;
	for (int i = 1; i < argc; i++)
	{
		std::string a = argv[i];
		if (a == "--image" && i + 3 < argc)
		{
			imagePath = argv[++i];
			imgW = atoi(argv[++i]);
			imgH = atoi(argv[++i]);
		}
		else if (a == "--out" && i + 1 < argc)
			outPrefix = argv[++i];
		else if (a == "--size" && i + 2 < argc)
		{
			dynW = atoi(argv[++i]);
			dynH = atoi(argv[++i]);
		}
		else if (a == "--norm" && i + 1 < argc)
		{
			std::string n = argv[++i];
			normKind = n == "imagenet" ? InputNormKind::ImageNet : n == "signed" ? InputNormKind::Signed :
				n == "byte" ? InputNormKind::Byte : InputNormKind::Unit;
		}
		else if (a == "--raw")
			rawMode = true;
		else if (a == "--movenet")
			moveNet = true;
		else if (a == "--yunet")
			yuNet = true;
		else if (a == "--crop" && i + 4 < argc)
		{
			CropBox b;
			b.x1 = (float)atof(argv[++i]);
			b.y1 = (float)atof(argv[++i]);
			b.x2 = (float)atof(argv[++i]);
			b.y2 = (float)atof(argv[++i]);
			crops.push_back(b);
		}
		else if (a == "--batch" && i + 1 < argc)
			batch = atoi(argv[++i]);
		else if (a == "--bgr")
			bgr = true;
		else if (a == "--graph")
			graph = true;
		else
			pos.push_back(a);
	}
	if (pos.size() < 2)
	{
		printf("usage: onnxtop_tool <TD bin dir> <model.onnx> [tensorrt|cuda] [iterations] [fp32] "
			"[--size <w> <h>] [--norm unit|imagenet|signed|byte] [--bgr] [--graph] [--raw | --movenet | --yunet] [--image <raw rgba> <w> <h> --out <prefix>]\n");
		return 1;
	}
	std::wstring wbin(pos[0].begin(), pos[0].end());
	EngineConfig cfg;
	cfg.modelPath = pos[1];
	cfg.provider = (pos.size() > 2 && pos[2] == "cuda") ? OrtProvider::CUDA : OrtProvider::TensorRT;
	int iters = pos.size() > 3 ? atoi(pos[3].c_str()) : 200;
	cfg.fp16 = !(pos.size() > 4 && pos[4] == "fp32");
	cfg.inputW = dynW;
	cfg.inputH = dynH;
	cfg.batch = batch;
	cfg.cudaGraph = graph;

	// TD's bin dir must win DLL resolution (nvinfer, cudnn, cublas) over anything on PATH.
	SetDllDirectoryW(wbin.c_str());

	std::string err;
	if (!ortload::init(err, wbin))
	{
		printf("ORT init failed: %s\n", err.c_str());
		return 1;
	}
	printf("ONNX Runtime %s  (%s)\n", ortload::version().c_str(), ortload::loadedPath().c_str());

	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cfg.stream = stream;

	auto t0 = Clock::now();
	OnnxEngine engine;
	if (!engine.load(cfg, err))
	{
		printf("Load failed: %s\n%s", err.c_str(), ortload::recentLog(20).c_str());
		return 1;
	}
	printf("Loaded in %.0f ms -- provider: %s\n", msSince(t0), engine.providerDescription().c_str());
	for (const auto& t : engine.inputs())
		printf("  in  %-12s %s %s\n", t.name.c_str(), shapeString(t.shape).c_str(), ortTypeName(t.type));
	for (const auto& t : engine.outputs())
		printf("  out %-12s %s %s\n", t.name.c_str(), shapeString(t.shape).c_str(), ortTypeName(t.type));

	// Launch: the CPU time until run() returns without syncing (what TD's cook pays with Wait for
	// Result off); inference: until the GPU is done.
	std::vector<double> times, launches;
	for (int i = 0; i < iters; i++)
	{
		auto a = Clock::now();
		if (!engine.run(err, false))
		{
			printf("Run failed: %s\n", err.c_str());
			return 1;
		}
		launches.push_back(msSince(a));
		cudaStreamSynchronize(stream);
		times.push_back(msSince(a));
	}
	printStats("Launch (CPU, no sync):", launches);
	printStats("Inference only:", times);

	if (!imagePath.empty())
	{
		std::vector<unsigned char> rgba((size_t)imgW * imgH * 4);
		std::ifstream f(imagePath, std::ios::binary);
		if (!f.read((char*)rgba.data(), rgba.size()))
		{
			printf("Could not read %zu bytes from %s\n", rgba.size(), imagePath.c_str());
			return 1;
		}

		// Emulate a TD BGRA8 texture: swap R/B and store rows bottom-first.
		std::vector<unsigned char> tex(rgba.size());
		for (int y = 0; y < imgH; y++)
			for (int x = 0; x < imgW; x++)
			{
				const unsigned char* s = &rgba[((size_t)y * imgW + x) * 4];
				unsigned char* d = &tex[((size_t)(imgH - 1 - y) * imgW + x) * 4];
				d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; d[3] = s[3];
			}

		cudaChannelFormatDesc inDesc = cudaCreateChannelDesc<uchar4>();
		cudaArray_t inArr, outArr;
		cudaMallocArray(&inArr, &inDesc, imgW, imgH, cudaArraySurfaceLoadStore);
		cudaMemcpy2DToArray(inArr, 0, 0, tex.data(), imgW * 4, imgW * 4, imgH, cudaMemcpyHostToDevice);
		cudaChannelFormatDesc outDesc = rawMode ? cudaCreateChannelDesc<float>() : cudaCreateChannelDescHalf4();
		cudaMallocArray(&outArr, &outDesc, imgW, imgH, cudaArraySurfaceLoadStore);
		cudaSurfaceObject_t inSurf = makeSurface(inArr), outSurf = makeSurface(outArr);

		const TensorSlot& in = engine.inputs()[0];
		Letterbox lb = computeLetterbox(imgW, imgH, (int)in.shape[widthDim(in.shape)], (int)in.shape[heightDim(in.shape)], true);
		InputNorm norm = makeInputNorm(normKind);
		InputLayout layout;
		layout.elem = in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? TensorElem::F16 :
			in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? TensorElem::I32 :
			in.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? TensorElem::U8 : TensorElem::F32;
		layout.nhwc = isNHWC(in.shape);
		layout.bgr = bgr;

		if (!crops.empty())
		{
			// Same path as the TOP's Classify Crops mode: crop -> batch -> run -> softmax.
			CropBox* devBoxes = nullptr;
			cudaMalloc(&devBoxes, crops.size() * sizeof(CropBox));
			cudaMemcpy(devBoxes, crops.data(), crops.size() * sizeof(CropBox), cudaMemcpyHostToDevice);
			const int B = (int)in.shape[0];
			const int n = (std::min)((int)crops.size(), B);
			std::vector<double> pipe;
			for (int i = 0; i < iters; i++)
			{
				auto a = Clock::now();
				launchCropPreprocess(inSurf, TexFormat::BGRA8, imgW, imgH, devBoxes, n, B, norm, layout, in.device,
					(int)in.shape[widthDim(in.shape)], (int)in.shape[heightDim(in.shape)], stream);
				if (!engine.run(err))
				{
					printf("Run failed: %s\n", err.c_str());
					return 1;
				}
				cudaStreamSynchronize(stream);
				pipe.push_back(msSince(a));
			}
			printStats("Crop pipeline (crop+infer):", pipe);
			const TensorSlot& o = engine.outputs()[0];
			int C = (int)o.shape[1];
			std::vector<float> logits((size_t)n * C);
			cudaMemcpy(logits.data(), o.device, logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
			for (int k = 0; k < n; k++)
			{
				float* l = logits.data() + (size_t)k * C;
				float mx = *std::max_element(l, l + C), sum = 0.f;
				std::vector<float> pr(C);
				for (int c = 0; c < C; c++)
					sum += (pr[c] = std::exp(l[c] - mx));
				printf("  crop %d:", k);
				for (int c = 0; c < C; c++)
					printf(" %.3f", pr[c] / sum);
				printf("\n");
			}
			cudaFree(devBoxes);
			return 0;
		}
		if (rawMode)
		{
			// Same path as the TOP's Raw mode with Normalize = Min/Max and Map to Input Frame:
			// output 0, channel 0 -> Mono32Float at the image's size.
			TensorRange* range = nullptr;
			cudaMalloc(&range, sizeof(TensorRange));
			std::vector<double> pipe;
			for (int i = 0; i < iters; i++)
			{
				auto a = Clock::now();
				launchPreprocess(inSurf, TexFormat::BGRA8, lb, norm, layout, in.device, 114.f / 255.f, stream);
				if (!engine.run(err))
				{
					printf("Pipeline failed: %s\n", err.c_str());
					return 1;
				}
				const TensorSlot& o = engine.outputs()[0];
				const auto& s = o.shape;
				int H = (int)s[s.size() - 2], W = (int)s[s.size() - 1];
				int C = s.size() >= 4 ? (int)s[s.size() - 3] : 1;
				bool oHalf = o.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
				launchTensorRange(o.device, oHalf, (std::min)(C, 4), H, W, lb, 0.f, i == 0, range, stream);
				launchTensorToTexture(o.device, oHalf, C, H, W, range, true, lb, true, outSurf, imgW, imgH, stream);
				cudaStreamSynchronize(stream);
				pipe.push_back(msSince(a));
			}
			cudaError_t ce = cudaGetLastError();
			printStats("Full raw pipeline (pre+infer+range+texture):", pipe);
			if (ce != cudaSuccess)
				printf("CUDA error: %s\n", cudaGetErrorString(ce));
			TensorRange hr = {};
			cudaMemcpy(&hr, range, sizeof(hr), cudaMemcpyDeviceToHost);
			printf("output 0 %s range lo %.4f hi %.4f\n", shapeString(engine.outputs()[0].shape).c_str(), hr.lo, hr.hi);
			cudaFree(range);
			if (!outPrefix.empty())
			{
				std::vector<float> out((size_t)imgW * imgH);
				cudaMemcpy2DFromArray(out.data(), imgW * 4, outArr, 0, 0, imgW * 4, imgH, cudaMemcpyDeviceToHost);
				std::ofstream pgm(outPrefix + "_raw.pgm", std::ios::binary);
				pgm << "P5\n" << imgW << " " << imgH << "\n255\n";
				for (int y = 0; y < imgH; y++)
					for (int x = 0; x < imgW; x++)
						pgm.put((char)(unsigned char)std::clamp(out[(size_t)(imgH - 1 - y) * imgW + x] * 255.f, 0.f, 255.f));
				printf("Wrote %s_raw.pgm\n", outPrefix.c_str());
			}
			cudaDestroySurfaceObject(inSurf);
			cudaDestroySurfaceObject(outSurf);
			cudaFreeArray(inArr);
			cudaFreeArray(outArr);
			return 0;
		}

		YoloDecoder yolo;
		std::vector<double> pipe;
		for (int i = 0; i < iters; i++)
		{
			auto a = Clock::now();
			launchPreprocess(inSurf, TexFormat::BGRA8, lb, norm, layout, in.device, 114.f / 255.f, stream);
			if (!engine.run(err) ||
				!(yuNet ? yolo.decodeYuNet(engine, lb, 0.5f, 0.3f, 16, outSurf, imgW, imgH, stream, err)
					: moveNet ? yolo.decodeMoveNet(engine, lb, 0.1f, 16, outSurf, imgW, imgH, stream, err)
					: yolo.decode(engine, lb, 0.4f, -1, 16, SegOptions{}, outSurf, imgW, imgH, stream, err)))
			{
				printf("Pipeline failed: %s\n", err.c_str());
				return 1;
			}
			cudaStreamSynchronize(stream);
			pipe.push_back(msSince(a));
		}
		cudaError_t ce = cudaGetLastError();
		printStats("Full pipeline (pre+infer+post):", pipe);
		if (ce != cudaSuccess)
			printf("CUDA error: %s\n", cudaGetErrorString(ce));

		printf("%zu detections (conf >= 0.4, all classes):\n", yolo.detections().size());
		for (const auto& d : yolo.detections())
		{
			printf("  class %2d  conf %.3f  uv (%.3f, %.3f)-(%.3f, %.3f)  fill %.3f\n", d.cls, d.conf, d.u1, d.v1, d.u2, d.v2, d.maskFill);
			if (d.keypoints.size() >= 3)
				printf("      kp0 uv (%.3f, %.3f) c %.2f  (%zu keypoints)\n", d.keypoints[0], d.keypoints[1], d.keypoints[2], d.keypoints.size() / 3);
		}

		if (!outPrefix.empty())
		{
			std::vector<__half> out((size_t)imgW * imgH * 4);
			cudaMemcpy2DFromArray(out.data(), imgW * 8, outArr, 0, 0, imgW * 8, imgH, cudaMemcpyDeviceToHost);
			std::ofstream pgm(outPrefix + "_matte.pgm", std::ios::binary);
			std::ofstream ppm(outPrefix + "_overlay.ppm", std::ios::binary);
			pgm << "P5\n" << imgW << " " << imgH << "\n255\n";
			ppm << "P6\n" << imgW << " " << imgH << "\n255\n";
			for (int y = 0; y < imgH; y++)
				for (int x = 0; x < imgW; x++)
				{
					float m = __half2float(out[((size_t)(imgH - 1 - y) * imgW + x) * 4]);	// back to top-down
					unsigned char mv = (unsigned char)std::clamp(m * 255.f, 0.f, 255.f);
					pgm.put((char)mv);
					const unsigned char* s = &rgba[((size_t)y * imgW + x) * 4];
					for (int c = 0; c < 3; c++)
					{
						float tint = c == 1 ? 255.f : 0.f;	// green over the matte
						float v = s[c] * (1.f - 0.6f * m) + tint * 0.6f * m;
						ppm.put((char)(unsigned char)std::clamp(v, 0.f, 255.f));
					}
				}
			printf("Wrote %s_matte.pgm and %s_overlay.ppm\n", outPrefix.c_str(), outPrefix.c_str());
		}
		cudaDestroySurfaceObject(inSurf);
		cudaDestroySurfaceObject(outSurf);
		cudaFreeArray(inArr);
		cudaFreeArray(outArr);
	}

	std::string log = ortload::recentLog(10);
	if (!log.empty())
		printf("Recent ORT log:\n%s", log.c_str());
	return 0;
}
