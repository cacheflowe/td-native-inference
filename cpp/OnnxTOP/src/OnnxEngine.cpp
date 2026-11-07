#include "OnnxEngine.h"

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>

namespace fs = std::filesystem;

size_t ortTypeSize(ONNXTensorElementDataType t)
{
	switch (t)
	{
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:	return 4;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:	return 2;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:	return 8;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:	return 4;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:	return 1;
		default:									return 0;
	}
}

const char* ortTypeName(ONNXTensorElementDataType t)
{
	switch (t)
	{
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:	return "float32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:	return "float16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:	return "int64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:	return "int32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:	return "uint8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:	return "int8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:	return "bool";
		default:									return "unsupported";
	}
}

std::string shapeString(const std::vector<int64_t>& shape)
{
	std::string s = "[";
	for (size_t i = 0; i < shape.size(); i++)
		s += (i ? ", " : "") + std::to_string(shape[i]);
	return s + "]";
}

std::string resolveCacheDir(const EngineConfig& cfg)
{
	if (!cfg.cacheDir.empty())
		return cfg.cacheDir;
	return (fs::path(cfg.modelPath).parent_path() / "trt_cache").string();
}

std::string trtBuildMarkerPath(const EngineConfig& cfg)
{
	// Key = everything that invalidates a TensorRT engine: model bytes (size + mtime),
	// precision, GPU architecture, and the ORT/TensorRT build shipped with TD.
	std::error_code ec;
	fs::path model(cfg.modelPath);
	auto size = fs::file_size(model, ec);
	auto mtime = fs::last_write_time(model, ec).time_since_epoch().count();
	cudaDeviceProp prop = {};
	cudaGetDeviceProperties(&prop, cfg.deviceId);
	// Normalized path, so "data/x.onnx", "D:/.../x.onnx" and "D:\...\x.onnx" (TD par vs command
	// line) share one marker: absolute, lexically normal, backslashes, lower case (Windows).
	std::string where = fs::absolute(model, ec).lexically_normal().make_preferred().string();
	std::transform(where.begin(), where.end(), where.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
	std::string key = where + "|" + std::to_string(size) + "|" + std::to_string(mtime) +
		"|" + (cfg.fp16 ? "fp16" : "fp32") + "|sm" + std::to_string(prop.major) + std::to_string(prop.minor) +
		"|" + ortload::version();
	// Only dynamic-size models carry a size, so fixed-size models keep their existing markers.
	if (cfg.inputW > 0 && cfg.inputH > 0)
		key += "|" + std::to_string(cfg.inputW) + "x" + std::to_string(cfg.inputH);
	if (cfg.batch > 1)
		key += "|b" + std::to_string(cfg.batch);
	char hash[32];
	snprintf(hash, sizeof(hash), "%016llx", (unsigned long long)std::hash<std::string>{}(key));
	return (fs::path(resolveCacheDir(cfg)) / (model.stem().string() + "." + hash + ".built")).string();
}

bool probeDynamicInput(const std::string& modelPath, bool& dynamic, std::string& err)
{
	static std::mutex cacheMutex;
	static std::map<std::string, bool> cache;

	std::error_code ec;
	fs::path model(modelPath);
	std::string key = fs::absolute(model, ec).string() + "|" + std::to_string(fs::file_size(model, ec)) + "|" +
		std::to_string(fs::last_write_time(model, ec).time_since_epoch().count());
	{
		std::lock_guard<std::mutex> lock(cacheMutex);
		auto it = cache.find(key);
		if (it != cache.end())
		{
			dynamic = it->second;
			return true;
		}
	}
	if (!fs::exists(model, ec))
	{
		err = "Model not found: " + modelPath;
		return false;
	}
	try
	{
		Ort::SessionOptions so;
		so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
		Ort::Session session(ortload::env(), model.wstring().c_str(), so);
		dynamic = false;
		if (session.GetInputCount() > 0)
		{
			Ort::TypeInfo typeInfo = session.GetInputTypeInfo(0);
			auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
			dynamic = shape.size() == 4 && (shape[heightDim(shape)] <= 0 || shape[widthDim(shape)] <= 0);
		}
	}
	catch (const Ort::Exception& e)
	{
		err = std::string("Could not read model inputs: ") + e.what();
		return false;
	}
	std::lock_guard<std::mutex> lock(cacheMutex);
	cache[key] = dynamic;
	return true;
}

size_t TensorSlot::elementCount() const
{
	return std::accumulate(shape.begin(), shape.end(), (size_t)1,
		[](size_t a, int64_t d) { return a * (size_t)(d > 0 ? d : 0); });
}

size_t TensorSlot::bytes() const
{
	return elementCount() * ortTypeSize(type);
}

OnnxEngine::~OnnxEngine()
{
	release();
}

void OnnxEngine::release()
{
	// Order matters: binding/values reference the session and our device buffers.
	myBinding.reset();
	myInputValues.clear();
	myOutputValues.clear();
	mySession.reset();
	for (auto* list : { &myInputs, &myOutputs })
	{
		for (auto& t : *list)
			if (t.owned && t.device)
				cudaFree(t.device);
		list->clear();
	}
}

Ort::SessionOptions OnnxEngine::makeOptions(OrtProvider provider, std::string& desc)
{
	const OrtApi& api = Ort::GetApi();
	Ort::SessionOptions so;
	so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	std::string dev = std::to_string(myConfig.deviceId);

	if (provider == OrtProvider::TensorRT)
	{
		std::string cache = resolveCacheDir(myConfig);
		fs::create_directories(cache);

		OrtTensorRTProviderOptionsV2* trt = nullptr;
		Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
		std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
			trtGuard(trt, api.ReleaseTensorRTProviderOptions);

		const char* keys[] = {
			"device_id",
			"trt_fp16_enable",
			"trt_engine_cache_enable",
			"trt_engine_cache_path",
			"trt_timing_cache_enable",
			"trt_timing_cache_path",
		};
		const char* vals[] = {
			dev.c_str(),
			myConfig.fp16 ? "1" : "0",
			"1",
			cache.c_str(),
			"1",
			cache.c_str(),
		};
		Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys, vals, std::size(keys)));
		if (myConfig.stream)
			Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(trt, "user_compute_stream", myConfig.stream));
		so.AppendExecutionProvider_TensorRT_V2(*trt);
		desc = myConfig.fp16 ? "TensorRT (fp16)" : "TensorRT (fp32)";
	}

	// CUDA EP is always appended: as the main provider, or as the fallback for any node
	// TensorRT declines. HEURISTIC conv algo search avoids the per-shape EXHAUSTIVE stalls
	// documented in docs/learnings/onnx-runtime.md.
	OrtCUDAProviderOptionsV2* cuda = nullptr;
	Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
	std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
		cudaGuard(cuda, api.ReleaseCUDAProviderOptions);
	const char* ckeys[] = { "device_id", "cudnn_conv_algo_search" };
	const char* cvals[] = { dev.c_str(), "HEURISTIC" };
	Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, ckeys, cvals, std::size(ckeys)));
	if (myConfig.stream)
		Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(cuda, "user_compute_stream", myConfig.stream));
	so.AppendExecutionProvider_CUDA_V2(*cuda);
	if (provider == OrtProvider::CUDA)
		desc = "CUDA";

	return so;
}

bool OnnxEngine::load(const EngineConfig& cfg, std::string& err)
{
	release();
	myConfig = cfg;

	if (cudaSetDevice(cfg.deviceId) != cudaSuccess)
	{
		err = "cudaSetDevice(" + std::to_string(cfg.deviceId) + ") failed";
		return false;
	}

	const std::wstring wpath = fs::path(cfg.modelPath).wstring();
	if (!fs::exists(wpath))
	{
		err = "Model not found: " + cfg.modelPath;
		return false;
	}

	// TensorRT first; on any failure (missing DLL, unsupported op, build error) retry with
	// the CUDA EP alone rather than leaving the node dead.
	// Never create a TensorRT session against a foreign TensorRT build -- it crashes the
	// process (see ortload::claimTensorRT). Run on the CUDA EP instead and say why.
	OrtProvider provider = cfg.provider == OrtProvider::Auto ? OrtProvider::TensorRT : cfg.provider;
	std::string trtBlocked;
	if (provider == OrtProvider::TensorRT)
	{
		trtBlocked = ortload::claimTensorRT();
		if (!trtBlocked.empty())
			provider = OrtProvider::CUDA;
	}

	std::string desc;
	try
	{
		Ort::SessionOptions so = makeOptions(provider, desc);
		mySession = std::make_unique<Ort::Session>(ortload::env(), wpath.c_str(), so);
	}
	catch (const Ort::Exception& e)
	{
		if (provider != OrtProvider::TensorRT)
		{
			err = std::string("Session creation failed: ") + e.what();
			return false;
		}
		try
		{
			Ort::SessionOptions so = makeOptions(OrtProvider::CUDA, desc);
			mySession = std::make_unique<Ort::Session>(ortload::env(), wpath.c_str(), so);
			desc = std::string("CUDA (TensorRT failed: ") + e.what() + ")";
		}
		catch (const Ort::Exception& e2)
		{
			err = std::string("TensorRT failed: ") + e.what() + " | CUDA fallback failed: " + e2.what();
			return false;
		}
	}
	catch (const std::exception& e)
	{
		err = std::string("Session creation failed: ") + e.what();
		return false;
	}
	if (!trtBlocked.empty())
		desc = "CUDA (TensorRT disabled: " + trtBlocked + ")";
	myProviderDesc = desc;

	try
	{
		myCudaMem = std::make_unique<Ort::MemoryInfo>("Cuda", OrtArenaAllocator, cfg.deviceId, OrtMemTypeDefault);
		myBinding = std::make_unique<Ort::IoBinding>(*mySession);
		Ort::AllocatorWithDefaultOptions cpuAlloc;

		for (size_t i = 0; i < mySession->GetInputCount(); i++)
		{
			TensorSlot slot;
			slot.name = mySession->GetInputNameAllocated(i, cpuAlloc).get();
			// Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() returns a non-owning view into it,
			// so chaining it off the temporary reads freed memory (garbage names/types in TD).
			Ort::TypeInfo typeInfo = mySession->GetInputTypeInfo(i);
			auto info = typeInfo.GetTensorTypeAndShapeInfo();
			slot.type = info.GetElementType();
			slot.shape = info.GetShape();
			for (size_t d = 0; d < slot.shape.size(); d++)
			{
				if (slot.shape[d] > 0)
					continue;
				if (d == 0)
					slot.shape[d] = cfg.batch > 0 ? cfg.batch : 1;	// dynamic batch -> Batch Size (crop mode) or 1
				else if (slot.shape.size() == 4 && cfg.inputW > 0 && cfg.inputH > 0 &&
					((int)d == heightDim(info.GetShape()) || (int)d == widthDim(info.GetShape())))
					slot.shape[d] = (int)d == heightDim(info.GetShape()) ? cfg.inputH : cfg.inputW;	// dynamic H/W -> Dynamic Size
				else
				{
					err = "Input '" + slot.name + "' has dynamic dim " + std::to_string(d) +
						" " + shapeString(info.GetShape()) + " -- set Dynamic Size, or re-export with a fixed input size";
					return false;
				}
			}
			if (ortTypeSize(slot.type) == 0)
			{
				err = "Input '" + slot.name + "' has unsupported type " + std::to_string(slot.type);
				return false;
			}
			if (cudaMalloc(&slot.device, slot.bytes()) != cudaSuccess)
			{
				err = "cudaMalloc failed for input '" + slot.name + "'";
				return false;
			}
			cudaMemset(slot.device, 0, slot.bytes());
			slot.owned = true;
			myInputValues.emplace_back(Ort::Value::CreateTensor(*myCudaMem, slot.device, slot.bytes(),
				slot.shape.data(), slot.shape.size(), slot.type));
			myBinding->BindInput(slot.name.c_str(), myInputValues.back());
			myInputs.push_back(std::move(slot));
		}

		for (size_t i = 0; i < mySession->GetOutputCount(); i++)
		{
			TensorSlot slot;
			slot.name = mySession->GetOutputNameAllocated(i, cpuAlloc).get();
			// Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() returns a non-owning view into it,
			// so chaining it off the temporary reads freed memory (garbage names/types in TD).
			Ort::TypeInfo typeInfo = mySession->GetOutputTypeInfo(i);
			auto info = typeInfo.GetTensorTypeAndShapeInfo();
			slot.type = info.GetElementType();
			slot.shape = info.GetShape();
			if (!slot.shape.empty() && slot.shape[0] < 0)	// dynamic batch: same as the input
				slot.shape[0] = cfg.batch > 0 ? cfg.batch : 1;
			bool isStatic = std::all_of(slot.shape.begin(), slot.shape.end(), [](int64_t d) { return d > 0; });

			if (isStatic && ortTypeSize(slot.type) > 0)
			{
				// Preallocate: output pointers stay fixed for the session's lifetime.
				if (cudaMalloc(&slot.device, slot.bytes()) != cudaSuccess)
				{
					err = "cudaMalloc failed for output '" + slot.name + "'";
					return false;
				}
				slot.owned = true;
				myOutputValues.emplace_back(Ort::Value::CreateTensor(*myCudaMem, slot.device, slot.bytes(),
					slot.shape.data(), slot.shape.size(), slot.type));
				myBinding->BindOutput(slot.name.c_str(), myOutputValues.back());
			}
			else
			{
				// Dynamic: ORT allocates on device each run; run() refreshes shape + pointer.
				myBinding->BindOutput(slot.name.c_str(), *myCudaMem);
			}
			myOutputs.push_back(std::move(slot));
		}
	}
	catch (const Ort::Exception& e)
	{
		err = std::string("Binding setup failed: ") + e.what();
		release();
		return false;
	}

	// Warm-up run: TensorRT/cuDNN do their remaining lazy setup on the first Run, so pay it
	// here on the loader thread instead of as a hitch on the first live frame.
	if (!run(err))
	{
		err = "Warm-up run failed: " + err;
		release();
		return false;
	}
	return true;
}

bool OnnxEngine::run(std::string& err)
{
	if (!mySession)
	{
		err = "No session";
		return false;
	}
	try
	{
		mySession->Run(myRunOptions, *myBinding);

		// Refresh output pointers/shapes (no-op for preallocated static outputs, required
		// for dynamic ones since ORT may hand back a different buffer each run).
		std::vector<Ort::Value> values = myBinding->GetOutputValues();
		for (size_t i = 0; i < values.size() && i < myOutputs.size(); i++)
		{
			TensorSlot& slot = myOutputs[i];
			if (slot.owned)
				continue;
			auto info = values[i].GetTensorTypeAndShapeInfo();
			slot.shape = info.GetShape();
			slot.type = info.GetElementType();
			slot.device = values[i].GetTensorMutableRawData();
		}
		bool anyDynamic = std::any_of(myOutputs.begin(), myOutputs.end(), [](const TensorSlot& s) { return !s.owned; });
		if (anyDynamic)
		{
			// Hold this run's Values so dynamic outputs' device memory outlives this call.
			// (Static outputs' Values just re-wrap our own cudaMalloc'd buffers.)
			myOutputValues = std::move(values);
		}
	}
	catch (const Ort::Exception& e)
	{
		err = std::string("Run failed: ") + e.what();
		return false;
	}
	return true;
}
