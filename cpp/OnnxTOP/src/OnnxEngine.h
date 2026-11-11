// GPU-resident ONNX Runtime session: TensorRT EP (falls back to CUDA EP), every input and
// output bound to device memory through IoBinding, all work on a caller-supplied CUDA stream.
// No TouchDesigner dependencies -- shared by the TOP plugin and the standalone bench tool.
#pragma once

#include "Ort.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

enum class OrtProvider
{
	TensorRT,
	CUDA,
	Auto,	// TOP-level policy (CUDA now, TensorRT once built); OnnxEngine::load treats it as TensorRT
};

struct EngineConfig
{
	std::string		modelPath;
	OrtProvider		provider = OrtProvider::TensorRT;
	bool			fp16 = true;			// TensorRT only
	std::string		cacheDir;				// TensorRT engine/timing cache; empty = <model dir>/trt_cache
	int				deviceId = 0;
	cudaStream_t	stream = nullptr;		// all inference runs on this stream
	// Size for models whose input H/W are dynamic (e.g. Depth Anything). 0 = model has a
	// fixed size; keep it 0 for those so it doesn't change the TensorRT cache key.
	int				inputW = 0, inputH = 0;
	// Batch size for a dynamic batch dim (crop-and-classify); 0 = 1. Only a size > 1 changes
	// the TensorRT cache key.
	int				batch = 0;
	// Capture the run as a CUDA graph and replay it (TensorRT / CUDA EP options). Bench only
	// (onnxtop_tool --graph): ORT then waits for the replay, so the launch costs the whole
	// inference (docs/learnings/td-cook-performance.md).
	bool			cudaGraph = false;

	bool operator==(const EngineConfig& o) const
	{
		return modelPath == o.modelPath && provider == o.provider && fp16 == o.fp16 &&
			cacheDir == o.cacheDir && deviceId == o.deviceId && stream == o.stream &&
			inputW == o.inputW && inputH == o.inputH && batch == o.batch && cudaGraph == o.cudaGraph;
	}
	bool operator!=(const EngineConfig& o) const { return !(*this == o); }
};

struct TensorSlot
{
	std::string					name;
	std::vector<int64_t>		shape;		// concrete shape (inputs: resolved at load; outputs: refreshed every run)
	ONNXTensorElementDataType	type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	void*						device = nullptr;	// current device pointer
	bool						owned = false;		// true = we cudaMalloc'd it (static output shape)

	size_t elementCount() const;
	size_t bytes() const;
};

class OnnxEngine
{
public:
	OnnxEngine() = default;
	~OnnxEngine();
	OnnxEngine(const OnnxEngine&) = delete;
	OnnxEngine& operator=(const OnnxEngine&) = delete;

	// Blocking. With TensorRT and no cached engine this can take 30s+ (engine build), so
	// call it from a worker thread. Safe off the main thread: touches no TD objects.
	bool load(const EngineConfig& cfg, std::string& err);

	// Runs inference on cfg.stream. Inputs must already be written to inputs()[i].device.
	// sync: returns once results are complete (ORT synchronizes the stream). Otherwise it returns
	// once the work is queued: anything later on cfg.stream sees the outputs, and the caller syncs
	// (e.g. with an event) before reading them on the CPU.
	bool run(std::string& err, bool sync = true);

	const std::vector<TensorSlot>&	inputs() const { return myInputs; }
	const std::vector<TensorSlot>&	outputs() const { return myOutputs; }
	const EngineConfig&				config() const { return myConfig; }
	// What actually got created, e.g. "TensorRT (fp16)" or "CUDA (TensorRT failed: ...)".
	const std::string&				providerDescription() const { return myProviderDesc; }

private:
	Ort::SessionOptions	makeOptions(OrtProvider provider, std::string& desc);
	void				release();

	EngineConfig					myConfig;
	std::string						myProviderDesc;
	std::unique_ptr<Ort::Session>	mySession;
	std::unique_ptr<Ort::IoBinding>	myBinding;
	std::unique_ptr<Ort::MemoryInfo> myCudaMem;
	std::vector<Ort::Value>			myInputValues;
	std::vector<Ort::Value>			myOutputValues;	// keeps static-shape output tensors alive
	std::vector<TensorSlot>			myInputs;
	std::vector<TensorSlot>			myOutputs;
	Ort::RunOptions					myRunOptions;
	Ort::RunOptions					myQueueRunOptions;	// disable_synchronize_execution_providers
	bool							myQueueOptionsSet = false;
};

// TensorRT cache folder for cfg (cfg.cacheDir, or <model dir>/trt_cache).
std::string resolveCacheDir(const EngineConfig& cfg);

// Whether the model's first input has dynamic H/W (an [N, C, H, W] input with H or W <= 0).
// Opens a CPU-only, unoptimized session the first time per model file (cached by path, size
// and mtime afterwards). Needs ortload::init(). Thread-safe.
bool probeDynamicInput(const std::string& modelPath, bool& dynamic, std::string& err);

// Marker file written after a TensorRT engine for cfg was built successfully (by the
// out-of-process builder). Its name hashes everything that invalidates the engine, so if it
// exists, an in-process load is a cache hit and never runs the (crash-prone) TRT builder.
std::string trtBuildMarkerPath(const EngineConfig& cfg);

// Image input layout of a 4D shape: true = NHWC [N, H, W, 3] (e.g. MoveNet), false = NCHW.
inline bool isNHWC(const std::vector<int64_t>& s) { return s.size() == 4 && s[1] != 3 && s[3] == 3; }
inline int heightDim(const std::vector<int64_t>& s) { return isNHWC(s) ? 1 : 2; }
inline int widthDim(const std::vector<int64_t>& s) { return isNHWC(s) ? 2 : 3; }

const char* ortTypeName(ONNXTensorElementDataType t);
size_t ortTypeSize(ONNXTensorElementDataType t);
std::string shapeString(const std::vector<int64_t>& shape);
