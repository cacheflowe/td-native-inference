#include "OnnxTOP.h"

#include <algorithm>
#include <charconv>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>

#include <windows.h>

#include "TrtBuilder.h"

namespace fs = std::filesystem;

namespace
{
	using Clock = std::chrono::high_resolution_clock;

	double msSince(Clock::time_point t0)
	{
		return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
	}

	// Info DAT cells: to_chars is several times cheaper than snprintf, and this runs for every cell.
	void formatValue(char* buf, size_t size, float v, bool integer)
	{
		auto res = integer ? std::to_chars(buf, buf + size - 1, (int)v)
			: std::to_chars(buf, buf + size - 1, v, std::chars_format::fixed, 4);
		*res.ptr = 0;
	}

	TexFormat mapFormat(OP_PixelFormat f)
	{
		switch (f)
		{
			case OP_PixelFormat::BGRA8Fixed:	return TexFormat::BGRA8;
			case OP_PixelFormat::RGBA8Fixed:	return TexFormat::RGBA8;
			case OP_PixelFormat::RGBA16Fixed:	return TexFormat::RGBA16Fixed;
			case OP_PixelFormat::RGBA16Float:	return TexFormat::RGBA16Float;
			case OP_PixelFormat::RGBA32Float:	return TexFormat::RGBA32Float;
			case OP_PixelFormat::Mono8Fixed:	return TexFormat::Mono8;
			case OP_PixelFormat::Mono16Float:	return TexFormat::Mono16Float;
			case OP_PixelFormat::Mono32Float:	return TexFormat::Mono32Float;
			default:							return TexFormat::Unsupported;
		}
	}

	// Per-frame surface object. Never cache these across frames: when an input changes size TD
	// frees its cudaArray and the replacement can reuse the same handle value, so a pointer
	// compare can't tell them apart and a stale surface fails with "invalid resource handle".
	struct FrameSurface
	{
		cudaSurfaceObject_t surf = 0;
		explicit FrameSurface(cudaArray_t array)
		{
			if (!array)
				return;
			cudaResourceDesc desc = {};
			desc.resType = cudaResourceTypeArray;
			desc.res.array.array = array;
			if (cudaCreateSurfaceObject(&surf, &desc) != cudaSuccess)
				surf = 0;
		}
		~FrameSurface()
		{
			if (surf)
				cudaDestroySurfaceObject(surf);
		}
		FrameSurface(const FrameSurface&) = delete;
		FrameSurface& operator=(const FrameSurface&) = delete;
	};

	// Menu values are saved by name, so existing nodes keep "tensorrt" / "cuda".
	const char* kProviderNames[] = { "auto", "tensorrt", "cuda" };
	const char* kProviderLabels[] = { "Auto (CUDA now, TensorRT once built)", "TensorRT", "CUDA" };
	const OrtProvider kProviders[] = { OrtProvider::Auto, OrtProvider::TensorRT, OrtProvider::CUDA };
	// Menu values are saved by name, so existing nodes keep "yolo" / "raw".
	const char* kModeNames[] = { "yolo", "movenet", "yunet", "crops", "raw" };
	const char* kModeLabels[] = { "YOLO End2End (det / seg / pose)", "MoveNet Multipose", "YuNet Faces",
		"Classify Crops (boxes from a DAT)", "Raw Output Tensor" };
	enum Mode { ModeYolo = 0, ModeMoveNet = 1, ModeYuNet = 2, ModeCrops = 3, ModeRaw = 4 };
	constexpr int kCropSheetCell = 128;	// px per tile of the Classify Crops debug image

	// Crop DAT rows -> source-pixel boxes (top-down) + ids. Accepts a tracks table
	// (track_id, x_left, y_bottom, x_right, y_top) or an OnnxTOP Info DAT (id, u1, v1, u2, v2),
	// both in TD UV of the source image (origin bottom-left).
	bool parseCropDat(const OP_DATInput* dat, int srcW, int srcH, float scale, bool square, int maxN,
		std::vector<CropBox>& boxes, std::vector<std::string>& ids, std::string& err)
	{
		boxes.clear();
		ids.clear();
		if (!dat)
		{
			err = "Set Crop DAT to a table of boxes (e.g. OnnxTopTables' tracks, or an OnnxTOP Info DAT)";
			return false;
		}
		if (!dat->isTable || dat->numRows < 1 || dat->numCols < 1)
			return true;
		auto col = [&](const char* name) {
			for (int c = 0; c < dat->numCols; c++)
			{
				const char* h = dat->getCell(0, c);
				if (h && !strcmp(h, name))
					return c;
			}
			return -1;
		};
		int idc = col("track_id");
		if (idc < 0)
			idc = col("id");
		int c[4] = { col("x_left"), col("y_bottom"), col("x_right"), col("y_top") };
		if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[3] < 0)
		{
			c[0] = col("u1");
			c[1] = col("v1");
			c[2] = col("u2");
			c[3] = col("v2");
		}
		if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[3] < 0)
		{
			err = "Crop DAT needs box columns x_left, y_bottom, x_right, y_top (a tracks table) or u1, v1, u2, v2 (an Info DAT)";
			return false;
		}
		auto num = [&](int r, int cc) { const char* v = dat->getCell(r, cc); return v ? (float)atof(v) : 0.f; };
		for (int r = 1; r < dat->numRows && (int)boxes.size() < maxN; r++)
		{
			float u1 = num(r, c[0]), v1 = num(r, c[1]), u2 = num(r, c[2]), v2 = num(r, c[3]);
			float x1 = u1 * srcW, x2 = u2 * srcW, y1 = (1.f - v2) * srcH, y2 = (1.f - v1) * srcH;
			float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
			float w = std::fabs(x2 - x1) * scale, h = std::fabs(y2 - y1) * scale;
			if (square)
				w = h = std::max(w, h);
			CropBox b{ std::max(0.f, cx - w * 0.5f), std::max(0.f, cy - h * 0.5f),
				std::min((float)srcW, cx + w * 0.5f), std::min((float)srcH, cy + h * 0.5f) };
			if (b.x2 - b.x1 < 2.f || b.y2 - b.y1 < 2.f)
				continue;
			boxes.push_back(b);
			const char* id = idc >= 0 ? dat->getCell(r, idc) : nullptr;
			ids.push_back(id ? id : std::to_string(r - 1));
		}
		return true;
	}

	std::vector<std::string> splitNames(const char* names)
	{
		std::vector<std::string> out;
		std::string cur;
		for (const char* q = names ? names : ""; ; q++)
		{
			if (*q == 0 || *q == ' ' || *q == ',' || *q == '\t')
			{
				if (!cur.empty())
					out.push_back(cur);
				cur.clear();
				if (*q == 0)
					break;
			}
			else
				cur += *q;
		}
		return out;
	}
	const char* kInputNormNames[] = { "unit", "imagenet", "signed", "byte", "custom" };
	const char* kInputNormLabels[] = { "0 to 1 (YOLO)", "ImageNet Mean/Std (Depth Anything, ViTs)", "-1 to 1",
		"0 to 255 (MoveNet, YuNet)", "Custom Scale / Offset" };
	const char* kChannelNames[] = { "rgb", "bgr" };
	const char* kChannelLabels[] = { "RGB", "BGR (OpenCV-trained, e.g. YuNet)" };

	bool tensorElem(ONNXTensorElementDataType t, TensorElem& e)
	{
		switch (t)
		{
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:	e = TensorElem::F32; return true;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:	e = TensorElem::F16; return true;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:	e = TensorElem::I32; return true;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:	e = TensorElem::U8; return true;
			default:									return false;
		}
	}
	const char* kRawNormNames[] = { "none", "minmax" };
	const char* kRawNormLabels[] = { "None (Raw Values)", "Min / Max to 0-1" };

	// Raw output tensor -> C, H, W (C = 1 for [H, W] / [N, H, W]).
	bool rawDims(const TensorSlot& s, int& C, int& H, int& W)
	{
		const auto& sh = s.shape;
		C = H = W = 0;
		if (sh.size() >= 4)
			C = (int)sh[sh.size() - 3];
		else if (sh.size() >= 2)
			C = 1;
		if (sh.size() >= 2)
		{
			H = (int)sh[sh.size() - 2];
			W = (int)sh[sh.size() - 1];
		}
		bool okType = s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
		return okType && C > 0 && H > 0 && W > 0 && s.device;
	}

	// TouchDesigner's bin folder (holds onnxruntime.dll, TensorRT, cuDNN, cudart).
	fs::path exeDir()
	{
		wchar_t buf[MAX_PATH * 2] = {};
		GetModuleFileNameW(nullptr, buf, (DWORD)std::size(buf));
		return fs::path(buf).parent_path();
	}

	// Folder this plugin DLL was loaded from (onnxtop_tool.exe is expected next to it).
	fs::path pluginDir()
	{
		HMODULE mod = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&exeDir), &mod);
		wchar_t buf[MAX_PATH * 2] = {};
		GetModuleFileNameW(mod, buf, (DWORD)std::size(buf));
		return fs::path(buf).parent_path();
	}

	// TD resolves a relative file parameter against the project folder. When that file doesn't
	// exist, fall back to the same relative path under the plugin's folder, so a tox can preset
	// "models/..." and work from an installed Plugins/<package>/models/ folder.
	// `missing` gets both places looked in when neither has the file.
	std::string resolveModelPath(const char* resolved, const char* raw, std::string& missing)
	{
		std::string path = resolved ? resolved : "";
		missing.clear();
		std::error_code ec;
		if (path.empty() || fs::exists(path, ec) || !raw || !*raw)
			return path;
		fs::path rel(raw);
		if (rel.is_absolute())
			return path;
		fs::path candidate = (pluginDir() / rel).lexically_normal();
		if (fs::exists(candidate, ec))
			return candidate.generic_string();
		missing = "Looked for " + rel.generic_string() + " in the project folder and in " +
			pluginDir().generic_string() + " (run fetch_models.ps1 there for models not in the release).";
		return path;
	}

	constexpr float kLetterboxPad = 114.f / 255.f;	// Ultralytics letterbox gray
	constexpr int kMaxDetsCap = YoloDecoder::kMaxDets;
}

extern "C"
{
DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
	if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
		return;
	info->executeMode = TOP_ExecuteMode::CUDA;
	info->customOPInfo.opType->setString("Onnx");
	info->customOPInfo.opLabel->setString("ONNX");
	info->customOPInfo.opIcon->setString("ONX");
	info->customOPInfo.authorName->setString("td-native-inference");
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	return new OnnxTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
	delete static_cast<OnnxTOP*>(instance);
}
}

OnnxTOP::OnnxTOP(const OP_NodeInfo*, TOP_Context* context) :
	myContext(context)
{
	cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
	cudaEventCreate(&myRunStart);
	cudaEventCreate(&myRunDone);
}

OnnxTOP::~OnnxTOP()
{
	// Kills an in-progress out-of-process TensorRT build. An in-process ORT session load
	// can't be interrupted, but with a cached engine that's only a few seconds.
	myCancelLoad = true;
	if (myLoader.joinable())
		myLoader.join();
	if (myStream)
		cudaStreamSynchronize(myStream);
	if (myRunStart)
		cudaEventDestroy(myRunStart);
	if (myRunDone)
		cudaEventDestroy(myRunDone);
	myEngine.reset();
	myNextEngine.reset();
	if (myRawRange)
		cudaFree(myRawRange);
	if (myDevCropBoxes)
		cudaFree(myDevCropBoxes);
	if (myHostLogits)
		cudaFreeHost(myHostLogits);
	if (myStream)
		cudaStreamDestroy(myStream);
}

void OnnxTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// Cooks every frame while its output is used, which also delivers a queued run (Wait for
	// Result off). No cookEveryFrame: it adds a cook of its own before the input's, so the node
	// would cook twice in frames where the input changes (see MediaPipeTOP).
	ginfo->cookEveryFrameIfAsked = true;
}

void OnnxTOP::manageLoading(const EngineConfig& desired)
{
	// Hand over a finished load (main thread only touches myEngine).
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		if (myNextEngine)
		{
			// Last frame's kernels may still be reading the outgoing engine's buffers.
			if (myStream)
				cudaStreamSynchronize(myStream);
			myEngine = std::move(myNextEngine);
			myRawRangeReset = true;
		}
	}

	if (myLoadState == LoadState::Loading)
		return;

	bool changed = !myHasRequested || !sameModelConfig(desired, myRequestedConfig);
	if (!changed && !myForceReload)
		return;	// Ready (possibly still upgrading to TensorRT), or Failed with the same config
	myForceReload = false;

	// Stops a background TensorRT build (Auto); a finished loader just joins.
	myCancelLoad = true;
	if (myLoader.joinable())
		myLoader.join();
	myUpgrading = false;

	// Drop the old engine BEFORE building the new one: both would share myStream, and two
	// TensorRT engines for the same model can double GPU memory during the swap.
	if (myStream)
		cudaStreamSynchronize(myStream);
	myEngine.reset();
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		myNextEngine.reset();
		myLoadError.clear();
		myLoadWarning.clear();
	}

	myRequestedConfig = desired;
	myHasRequested = true;

	if (desired.modelPath.empty())
	{
		myLoadState = LoadState::Idle;
		return;
	}

	myLoadState = LoadState::Loading;
	myCancelLoad = false;
	setLoadStatus("Loading model...");
	myLoader = std::thread([this, desired]() { loadEngines(desired); });
}

bool OnnxTOP::sameModelConfig(const EngineConfig& a, const EngineConfig& b)
{
	// Dynamic Size doesn't apply to a fixed-size model, so changing it mustn't reload one.
	std::string staticPath;
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		staticPath = myStaticModelPath;
	}
	if (a.modelPath == b.modelPath && a.modelPath == staticPath)
	{
		EngineConfig a2 = a, b2 = b;
		a2.inputW = a2.inputH = b2.inputW = b2.inputH = 0;
		return a2 == b2;
	}
	return a == b;
}

void OnnxTOP::loadEngines(EngineConfig cfg)
{
	std::string err;
	bool dynamic = false;
	if (!probeDynamicInput(cfg.modelPath, dynamic, err))
	{
		failLoad(err);
		return;
	}
	if (!dynamic)
	{
		cfg.inputW = cfg.inputH = 0;	// keeps fixed-size models' TensorRT cache key unchanged
		std::lock_guard<std::mutex> lock(myLoadMutex);
		myStaticModelPath = cfg.modelPath;
	}
	else if (cfg.inputW <= 0 || cfg.inputH <= 0)
	{
		failLoad("This model's input size is dynamic -- set Dynamic Size (e.g. 518 x 518 for Depth Anything)");
		return;
	}

	// TensorRT engine builds run in onnxtop_tool.exe, never in TD's process (see
	// TrtBuilder.h). Once the marker exists, loading is a pure cache hit.
	// claimTensorRT() also pins TD's TensorRT DLLs before anything else can load another
	// build; if a foreign one is already loaded, OnnxEngine::load() falls back to CUDA.
	bool trtUsable = cfg.provider != OrtProvider::CUDA && ortload::claimTensorRT().empty();
	bool trtCached = trtUsable && fs::exists(trtBuildMarkerPath(cfg));

	if (cfg.provider == OrtProvider::Auto && trtUsable && !trtCached)
	{
		// Auto, first run for this engine: run on CUDA now, build TensorRT in the background,
		// then swap. Both engines are briefly resident, and the build competes for the GPU.
		EngineConfig cudaCfg = cfg;
		cudaCfg.provider = OrtProvider::CUDA;
		setLoadStatus("Loading model on CUDA...");
		auto engine = std::make_unique<OnnxEngine>();
		if (!engine->load(cudaCfg, err))
		{
			failLoad(err);
			return;
		}
		myUpgrading = true;
		if (!publishEngine(std::move(engine), "Running on CUDA while the TensorRT engine builds in the background "
			"(first run for this model / size / precision: a few minutes, lower frame rate meanwhile)"))
		{
			myUpgrading = false;
			return;
		}

		std::string message;
		int built = buildTrtEngine(cfg, true, message);
		if (built < 0)
		{
			myUpgrading = false;
			return;	// cancelled
		}
		if (built == 0)
		{
			setLoadWarning("Running on CUDA -- TensorRT engine build failed: " + message);
			myUpgrading = false;
			return;
		}
		cfg.provider = OrtProvider::TensorRT;
		auto trt = std::make_unique<OnnxEngine>();
		if (!trt->load(cfg, err))
			setLoadWarning("Running on CUDA -- TensorRT engine load failed: " + err);
		else
			publishEngine(std::move(trt), message);
		myUpgrading = false;
		return;
	}

	std::string warning;
	if (cfg.provider == OrtProvider::Auto)
		cfg.provider = OrtProvider::TensorRT;	// cached engine (or TensorRT unusable: load() picks CUDA)
	if (cfg.provider == OrtProvider::TensorRT && trtUsable && !trtCached)
	{
		int built = buildTrtEngine(cfg, false, warning);
		if (built < 0)
			return;
		if (built == 0)
		{
			failLoad("TensorRT engine build failed: " + warning +
				". Set Execution Provider to Auto or CUDA to run without it.");
			return;
		}
		setLoadStatus("Loading cached TensorRT engine...");
	}

	auto engine = std::make_unique<OnnxEngine>();
	if (!engine->load(cfg, err))
	{
		failLoad(err);
		return;
	}
	publishEngine(std::move(engine), warning);
}

int OnnxTOP::buildTrtEngine(const EngineConfig& cfg, bool background, std::string& message)
{
	fs::path tool = pluginDir() / "onnxtop_tool.exe";
	if (!fs::exists(tool))
	{
		message = "onnxtop_tool.exe not found next to OnnxTOP.dll -- built TensorRT engine inside TD (crash risk)";
		return 1;	// the in-process load will build it
	}
	struct Ctx { OnnxTOP* self; bool background; } ctx{ this, background };
	auto onAttempt = [](void* user, int n) {
		auto* c = static_cast<Ctx*>(user);
		std::string attempt = " (attempt " + std::to_string(n) + ")";
		if (c->background)
			c->self->setLoadWarning("Running on CUDA while the TensorRT engine builds in the background" + attempt +
				" -- first run for this model / size / precision takes a few minutes, lower frame rate meanwhile");
		else
			c->self->setLoadStatus("Building TensorRT engine in onnxtop_tool.exe" + attempt +
				" -- first build for a model takes several minutes; TD stays responsive. "
				"Execution Provider Auto runs on CUDA meanwhile.");
	};
	TrtBuildResult r = buildEngineOutOfProcess(tool.wstring(), exeDir().wstring(), cfg, 3, myCancelLoad, onAttempt, &ctx);
	if (r.cancelled)
		return -1;
	if (!r.ok)
	{
		message = r.message;
		return 0;
	}
	message = r.attempts > 1 ? "TensorRT build needed " + std::to_string(r.attempts) + " attempts (see trt_cache/*.build.log)" : "";
	return 1;
}

bool OnnxTOP::publishEngine(std::unique_ptr<OnnxEngine> engine, const std::string& warning)
{
	std::lock_guard<std::mutex> lock(myLoadMutex);
	if (myCancelLoad)
		return false;
	myNextEngine = std::move(engine);
	myLoadWarning = warning;
	myLoadState = LoadState::Ready;
	return true;
}

void OnnxTOP::failLoad(const std::string& err)
{
	std::lock_guard<std::mutex> lock(myLoadMutex);
	myLoadError = err;
	myLoadState = LoadState::Failed;
}

void OnnxTOP::setLoadWarning(const std::string& s)
{
	std::lock_guard<std::mutex> lock(myLoadMutex);
	myLoadWarning = s;
}

void OnnxTOP::setLoadStatus(const std::string& s)
{
	std::lock_guard<std::mutex> lock(myLoadMutex);
	myLoadStatus = s;
}

void OnnxTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
	auto tExec = Clock::now();
	myExecuteCount++;
	myError.clear();
	myWarning.clear();

	std::string err;
	if (!ortload::init(err))
	{
		myError = err;
		return;
	}

	// --- read every parameter/input BEFORE beginCUDAOperations() (TD requirement) ---
	const char* modelPath = inputs->getParFilePath("Model");
	int providerIdx = std::clamp(inputs->getParInt("Provider"), 0, 2);
	OrtProvider provider = kProviders[providerIdx];
	bool fp16 = inputs->getParInt("Fp16") != 0;
	const char* cacheDir = inputs->getParFilePath("Cachefolder");
	bool active = inputs->getParInt("Active") != 0;
	int mode = inputs->getParInt("Mode");
	bool letterbox = inputs->getParInt("Letterbox") != 0;
	InputNormKind normKind = (InputNormKind)std::clamp(inputs->getParInt("Inputnorm"), 0, 4);
	InputNorm inputNorm = makeInputNorm(normKind);
	if (normKind == InputNormKind::Custom)
	{
		for (int c = 0; c < 3; c++)
		{
			inputNorm.scale[c] = (float)inputs->getParDouble("Normscale", c);
			inputNorm.offset[c] = (float)inputs->getParDouble("Normoffset", c);
		}
	}
	bool bgr = inputs->getParInt("Channelorder") == 1;
	int dynW = inputs->getParInt("Dynamicsize", 0);
	int dynH = inputs->getParInt("Dynamicsize", 1);
	float confThresh = (float)inputs->getParDouble("Confthreshold");
	int classFilter = inputs->getParInt("Classfilter");
	int maxDets = std::clamp(inputs->getParInt("Maxdetections"), 1, kMaxDetsCap);
	float nmsThresh = (float)inputs->getParDouble("Nmsthreshold");
	int batchSize = std::clamp(inputs->getParInt("Batchsize"), 1, (int)kMaxBatch);
	float cropScale = (float)inputs->getParDouble("Cropscale");
	bool cropSquare = inputs->getParInt("Cropsquare") != 0;
	bool cropSoftmax = inputs->getParInt("Softmax") != 0;
	int cropInterval = std::max(1, inputs->getParInt("Classifyinterval"));
	const OP_DATInput* cropDat = mode == ModeCrops ? inputs->getParDAT("Cropdat") : nullptr;
	myClassNames = splitNames(inputs->getParString("Classnames"));
	RawOptions rawOpt;
	rawOpt.index = inputs->getParInt("Rawoutput");
	rawOpt.minMax = inputs->getParInt("Rawnormalize") == 1;
	rawOpt.smoothing = (float)std::clamp(inputs->getParDouble("Rawsmoothing"), 0.0, 0.999);
	rawOpt.mapToSource = inputs->getParInt("Rawmaptoinput") != 0;
	SegOptions segOpt;
	segOpt.maskThreshold = (float)inputs->getParDouble("Maskthreshold");
	segOpt.matteConf = (float)inputs->getParDouble("Matteconf");
	segOpt.matteMinFill = (float)inputs->getParDouble("Matteminfill");
	segOpt.maskMargin = (float)inputs->getParDouble("Maskmargin");
	segOpt.matteNms = (float)inputs->getParDouble("Mattenms");
	segOpt.instances = inputs->getParInt("Matteencoding") == 1;
	myWaitResult = inputs->getParInt("Waitforresult") != 0;

	inputs->enablePar("Fp16", provider != OrtProvider::CUDA);
	inputs->enablePar("Cachefolder", provider != OrtProvider::CUDA);
	for (const char* p : { "Confthreshold", "Maxdetections" })
		inputs->enablePar(p, mode != ModeRaw && mode != ModeCrops);
	for (const char* p : { "Cropdat", "Batchsize", "Cropscale", "Cropsquare", "Softmax", "Classnames", "Classifyinterval" })
		inputs->enablePar(p, mode == ModeCrops);
	inputs->enablePar("Letterbox", mode != ModeCrops);
	inputs->enablePar("Classfilter", mode == ModeYolo);
	inputs->enablePar("Nmsthreshold", mode == ModeYuNet);
	inputs->enablePar("Normscale", normKind == InputNormKind::Custom);
	inputs->enablePar("Normoffset", normKind == InputNormKind::Custom);
	for (const char* p : { "Maskthreshold", "Matteconf", "Matteminfill", "Maskmargin", "Mattenms", "Matteencoding" })
		inputs->enablePar(p, mode == ModeYolo && myYolo.isSegmentation());
	for (const char* p : { "Rawoutput", "Rawnormalize", "Rawmaptoinput" })
		inputs->enablePar(p, mode == ModeRaw);
	inputs->enablePar("Rawsmoothing", mode == ModeRaw && rawOpt.minMax);
	inputs->enablePar("Waitforresult", mode != ModeRaw);

	EngineConfig desired;
	desired.modelPath = resolveModelPath(modelPath, inputs->getParString("Model"), myModelHint);
	desired.provider = provider;
	desired.fp16 = fp16;
	desired.cacheDir = cacheDir ? cacheDir : "";
	desired.deviceId = std::max(0, myContext->getCUDADeviceIndex(nullptr));
	desired.stream = myStream;
	desired.inputW = std::max(0, dynW);
	desired.inputH = std::max(0, dynH);
	desired.batch = mode == ModeCrops ? batchSize : 0;
	manageLoading(desired);
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		std::string staticPath = myStaticModelPath;
		inputs->enablePar("Dynamicsize", desired.modelPath != staticPath);
	}

	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		switch (myLoadState.load())
		{
			case LoadState::Idle:		myWarning = "Set the Model parameter to an .onnx file"; break;
			case LoadState::Loading:	myWarning = myLoadStatus; break;
			case LoadState::Failed:		myError = myLoadError; break;
			case LoadState::Ready:		myWarning = myLoadWarning; break;
		}
	}
	if (myEngine && myEngine->providerDescription().rfind("CUDA (", 0) == 0)
		myWarning = "Running on " + myEngine->providerDescription();

	const OP_TOPInput* top = inputs->getNumInputs() > 0 ? inputs->getInputTOP(0) : nullptr;
	if (!top)
	{
		myError = "Connect an input TOP";
		return;
	}
	TexFormat fmt = mapFormat(top->textureDesc.pixelFormat);
	if (fmt == TexFormat::Unsupported)
	{
		myError = "Unsupported input pixel format (use 8-bit, 16-bit or 32-bit RGBA, or Mono)";
		return;
	}

	const int srcW = (int)top->textureDesc.width, srcH = (int)top->textureDesc.height;
	myInputFrame = top->totalCooks;
	if (mode == ModeCrops && !parseCropDat(cropDat, srcW, srcH, cropScale, cropSquare, kMaxBatch, myCropBoxes, myCropIds, err))
		myWarning = err;

	OnnxEngine* engine = active ? myEngine.get() : nullptr;
	const TensorSlot* in = nullptr;
	InputLayout inLayout;
	if (engine)
	{
		in = &engine->inputs()[0];
		const auto& sh = in->shape;
		bool okInput = engine->inputs().size() == 1 && sh.size() == 4 && (sh[1] == 3 || sh[3] == 3) &&
			tensorElem(in->type, inLayout.elem);
		if (!okInput)
		{
			myError = "Model must have one image input shaped [1, 3, H, W] or [1, H, W, 3] "
				"(float32 / float16 / int32 / uint8), got " + shapeString(sh) + " " + ortTypeName(in->type);
			engine = nullptr;
		}
		else
		{
			inLayout.nhwc = isNHWC(sh);
			inLayout.bgr = bgr;
			myModelW = (int)sh[widthDim(sh)];
			myModelH = (int)sh[heightDim(sh)];
		}
	}

	// --- output texture ---
	TOP_CUDAOutputInfo outInfo;
	outInfo.stream = myStream;
	output->getSuggestedOutputDesc(&outInfo.textureDesc, nullptr);
	outInfo.textureDesc.texDim = OP_TexDim::e2D;
	outInfo.textureDesc.depth = 1;
	outInfo.textureDesc.pixelFormat = OP_PixelFormat::RGBA16Float;

	if (engine && mode == ModeCrops)
	{
		// Debug image: a contact sheet of every batch slot's crop.
		int B = (int)in->shape[0];
		int cols = (int)std::ceil(std::sqrt((double)B));
		outInfo.textureDesc.width = cols * kCropSheetCell;
		outInfo.textureDesc.height = ((B + cols - 1) / cols) * kCropSheetCell;
	}
	if (engine && mode == ModeRaw)
	{
		int C, H, W;
		if (rawOpt.index < 0 || rawOpt.index >= (int)engine->outputs().size())
		{
			myError = "Raw Output index out of range (model has " + std::to_string(engine->outputs().size()) + " outputs)";
			engine = nullptr;
		}
		else if (!rawDims(engine->outputs()[rawOpt.index], C, H, W))
		{
			const TensorSlot& s = engine->outputs()[rawOpt.index];
			myError = "Raw Output " + std::to_string(rawOpt.index) + " " + shapeString(s.shape) + " " +
				ortTypeName(s.type) + " can't be shown as an image";
			engine = nullptr;
		}
		else
		{
			// Mapped back to the input frame: input's size. Otherwise the tensor's own size.
			outInfo.textureDesc.width = rawOpt.mapToSource ? top->textureDesc.width : W;
			outInfo.textureDesc.height = rawOpt.mapToSource ? top->textureDesc.height : H;
			rawOpt.mono = C == 1;
			outInfo.textureDesc.pixelFormat = rawOpt.mono ? OP_PixelFormat::Mono32Float : OP_PixelFormat::RGBA32Float;
		}
	}

	const OP_CUDAArrayInfo* outArray = output->createCUDAArray(outInfo, nullptr);
	if (!outArray)
		return;
	OP_CUDAAcquireInfo acquire;
	acquire.stream = myStream;
	const OP_CUDAArrayInfo* inArray = top->getCUDAArray(acquire, nullptr);

	if (!myContext->beginCUDAOperations(nullptr))
		return;

	{
	FrameSurface outSurface(outArray->cudaArray);
	FrameSurface inSurface(inArray ? inArray->cudaArray : nullptr);
	cudaSurfaceObject_t outSurf = outSurface.surf;
	cudaSurfaceObject_t inSurf = inSurface.surf;
	int outW = (int)outInfo.textureDesc.width;
	int outH = (int)outInfo.textureDesc.height;

	bool wrote = false;
	// Wait for Result off: decode the run queued last cook, then queue this frame's without waiting.
	// Only for a new input frame: a still input (or a parameter change on one) runs synchronously,
	// so the node never keeps itself cooking. Raw output never reads back, so it always just queues.
	const bool async = !myWaitResult && mode != ModeRaw;
	const bool newInput = top->totalCooks != myLaunchedFrame;
	if (myPending.active && (!async || myPending.engine != engine || myPending.mode != mode))
		myPending.active = false;	// switched to waiting, or engine / mode changed: not wanted
	readInferMs();
	auto queueRun = [&](bool sync) -> bool
	{
		cudaEventRecord(myRunStart, myStream);
		auto tRun = Clock::now();
		const bool ran = engine->run(err, sync);
		myRunMs = msSince(tRun);
		if (!ran)
		{
			myError = err;
			return false;
		}
		cudaEventRecord(myRunDone, myStream);
		myRunRecorded = true;
		myLaunchedFrame = top->totalCooks;
		return true;
	};
	auto finishPending = [&]()
	{
		cudaEventSynchronize(myRunDone);	// normally long done: it ran on the GPU between cooks
		readInferMs();
		myResultFrame = myPending.frame;
		myPending.active = false;
	};

	if (engine && inSurf && mode == ModeCrops)
	{
		if (!myDevCropBoxes)
			cudaMalloc(&myDevCropBoxes, kMaxBatch * sizeof(CropBox));
		const int B = (int)in->shape[0];
		const int n = std::min((int)myCropBoxes.size(), B);
		if (n > 0)
			cudaMemcpyAsync(myDevCropBoxes, myCropBoxes.data(), n * sizeof(CropBox), cudaMemcpyHostToDevice, myStream);
		// Classify Interval: only every Nth frame crops + runs the model; the frames in between
		// keep the last results (like onnx_hsemotion.py's Emotioninterval).
		const bool due = !myCropMode || (myCropCounter++ % cropInterval) == 0;
		bool decoded = false;
		if (async && myPending.active)
		{
			finishPending();
			myCrops.clear();
			std::swap(myCropIds, myPending.cropIds);	// the ids the queued crops were made from
			if (!runCrops(*engine, myPending.crops, cropSoftmax, err))
				myError = err;
			std::swap(myCropIds, myPending.cropIds);
			decoded = true;
		}
		if (due && !(async && decoded && !newInput))
		{
			const bool queue = async && newInput;
			if (!queue)
				myCrops.clear();
			if (n > 0)
			{
				cudaError_t ce = launchCropPreprocess(inSurf, fmt, srcW, srcH, myDevCropBoxes, n, B, inputNorm, inLayout,
					in->device, myModelW, myModelH, myStream);
				if (ce != cudaSuccess)
					myError = std::string("Crop kernel failed: ") + cudaGetErrorString(ce);
				else if (queueRun(!queue))
				{
					if (queue)
					{
						myPending.active = true;
						myPending.engine = engine;
						myPending.mode = mode;
						myPending.crops = n;
						myPending.cropIds.assign(myCropIds.begin(), myCropIds.begin() + n);
						myPending.frame = top->totalCooks;
					}
					else
					{
						if (!runCrops(*engine, n, cropSoftmax, err))
							myError = err;
						readInferMs();
						myResultFrame = top->totalCooks;
					}
				}
			}
			else
			{
				myCrops.clear();
				myResultFrame = top->totalCooks;
			}
		}
		if (myError.empty())
		{
			int cols = (int)std::ceil(std::sqrt((double)B));
			launchCropSheet(inSurf, fmt, srcW, srcH, myDevCropBoxes, n, cols, kCropSheetCell, outSurf, outW, outH, myStream);
			wrote = true;
		}
		if ((int)myCropBoxes.size() > B && myWarning.empty())
			myWarning = std::to_string(myCropBoxes.size()) + " boxes but Batch Size is " + std::to_string(B) +
				" -- only the first " + std::to_string(B) + " are classified";
	}
	else if (engine && inSurf)
	{
		Letterbox lb = computeLetterbox((int)top->textureDesc.width, (int)top->textureDesc.height,
			myModelW, myModelH, letterbox);
		// Detections (and the seg matte) from the engine's current outputs, for the frame lb maps.
		auto decode = [&](const Letterbox& from) -> bool
		{
			bool ok = true;
			if (mode == ModeYolo)
				ok = myYolo.decode(*engine, from, confThresh, classFilter, maxDets, segOpt, outSurf, outW, outH, myStream, err);
			else if (mode == ModeMoveNet)
				ok = myYolo.decodeMoveNet(*engine, from, confThresh, maxDets, outSurf, outW, outH, myStream, err);
			else if (mode == ModeYuNet)
				ok = myYolo.decodeYuNet(*engine, from, confThresh, nmsThresh, maxDets, outSurf, outW, outH, myStream, err);
			else
				return runRaw(*engine, rawOpt, from, outSurf, outW, outH);
			if (!ok)
				myError = err;
			myYoloMode = true;
			return true;	// the decoders always write (or clear) the output
		};

		bool decoded = false;
		if (async && myPending.active)
		{
			finishPending();
			wrote = decode(myPending.lb);
			decoded = true;
		}
		if (!(async && decoded && !newInput))
		{
			const bool queue = async && newInput;
			cudaError_t ce = launchPreprocess(inSurf, fmt, lb, inputNorm, inLayout, in->device, kLetterboxPad, myStream);
			if (ce != cudaSuccess)
				myError = std::string("Preprocess kernel failed: ") + cudaGetErrorString(ce);
			else if (queueRun(!queue && mode != ModeRaw))
			{
				if (queue)
				{
					myPending.active = true;
					myPending.engine = engine;
					myPending.mode = mode;
					myPending.lb = lb;
					myPending.frame = top->totalCooks;
				}
				else
				{
					wrote = decode(lb);
					if (mode != ModeRaw)
						readInferMs();
					myResultFrame = top->totalCooks;
				}
			}
		}
	}
	if (mode == ModeRaw || mode == ModeCrops || !wrote)
	{
		myYolo.clear();
		myYoloMode = false;
	}
	myCropMode = mode == ModeCrops && wrote;
	if (!myCropMode)
		myCrops.clear();
	if (!wrote && outSurf)
	{
		auto pf = outInfo.textureDesc.pixelFormat;
		int bpp = pf == OP_PixelFormat::Mono32Float ? 4 : pf == OP_PixelFormat::RGBA16Float ? 8 : 16;
		launchClear(outSurf, outW, outH, bpp, myStream);
	}
	}	// surfaces destroyed here, before TD unmaps the arrays

	myContext->endCUDAOperations(nullptr);
	myExecuteMs = msSince(tExec);
}

void OnnxTOP::readInferMs()
{
	// Before the first recorded run the events are empty, and querying them reports an error that
	// cudaGetLastError() would hand to the next kernel launch's check.
	if (!myRunRecorded)
		return;
	float ms = 0.f;
	if (cudaEventQuery(myRunDone) == cudaSuccess && cudaEventElapsedTime(&ms, myRunStart, myRunDone) == cudaSuccess)
		myInferMs = ms;
	cudaGetLastError();	// not-ready / timing errors here aren't the frame's errors
}

bool OnnxTOP::runRaw(OnnxEngine& engine, const RawOptions& opt, const Letterbox& lb,
	cudaSurfaceObject_t outSurf, int outW, int outH)
{
	const TensorSlot& s = engine.outputs()[opt.index];
	int C, H, W;
	if (!rawDims(s, C, H, W))
	{
		myError = "Raw Output " + std::to_string(opt.index) + " " + shapeString(s.shape) + " changed shape";
		return false;
	}
	// Output sized for an earlier frame's tensor (dynamic output changed shape): skip one frame.
	if (!opt.mapToSource && (outW != W || outH != H))
		return false;
	bool isHalf = s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

	const TensorRange* range = nullptr;
	cudaError_t ce = cudaSuccess;
	if (opt.minMax)
	{
		if (!myRawRange && cudaMalloc(&myRawRange, sizeof(TensorRange)) != cudaSuccess)
		{
			myRawRange = nullptr;
			myError = "cudaMalloc failed for the raw output range";
			return false;
		}
		bool reset = myRawRangeReset || opt.index != myRawRangeIndex;
		ce = launchTensorRange(s.device, isHalf, opt.mono ? 1 : std::min(C, 4), H, W, lb, opt.smoothing, reset,
			myRawRange, myStream);
		myRawRangeReset = false;
		myRawRangeIndex = opt.index;
		range = myRawRange;
	}
	if (ce == cudaSuccess)
		ce = launchTensorToTexture(s.device, isHalf, C, H, W, range, opt.mapToSource, lb, opt.mono,
			outSurf, outW, outH, myStream);
	if (ce != cudaSuccess)
	{
		myError = std::string("Raw output kernel failed: ") + cudaGetErrorString(ce);
		return false;
	}
	return true;
}

bool OnnxTOP::runCrops(OnnxEngine& engine, int n, bool softmax, std::string& err)
{
	const TensorSlot& o = engine.outputs()[0];
	if (o.shape.size() != 2 || o.shape[1] <= 0 || o.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		err = "Classify Crops expects output0 float32 [batch, classes], got " + shapeString(o.shape) + " " + ortTypeName(o.type);
		return false;
	}
	const int C = (int)o.shape[1];
	size_t bytes = (size_t)n * C * sizeof(float);
	if (bytes > myHostLogitsBytes)
	{
		if (myHostLogits)
			cudaFreeHost(myHostLogits);
		cudaMallocHost(&myHostLogits, bytes);
		myHostLogitsBytes = bytes;
	}
	cudaMemcpyAsync(myHostLogits, o.device, bytes, cudaMemcpyDeviceToHost, myStream);
	cudaStreamSynchronize(myStream);
	myCropClasses = C;
	for (int k = 0; k < n; k++)
	{
		CropResult r;
		r.id = myCropIds[k];
		r.probs.assign(myHostLogits + (size_t)k * C, myHostLogits + (size_t)(k + 1) * C);
		if (softmax)
		{
			float mx = *std::max_element(r.probs.begin(), r.probs.end()), sum = 0.f;
			for (float& v : r.probs)
				sum += (v = std::exp(v - mx));
			for (float& v : r.probs)
				v /= sum;
		}
		r.cls = (int)(std::max_element(r.probs.begin(), r.probs.end()) - r.probs.begin());
		r.conf = r.probs[r.cls];
		myCrops.push_back(std::move(r));
	}
	return true;
}

// ---------------------------------------------------------------- info outputs

int32_t OnnxTOP::getNumInfoCHOPChans(void*)
{
	return 11;
}

void OnnxTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
	LoadState st = myLoadState.load();
	switch (index)
	{
		case 0: chan->name->setString("loaded");			chan->value = myEngine ? 1.f : 0.f; break;
		case 1: chan->name->setString("loading");			chan->value = st == LoadState::Loading ? 1.f : 0.f; break;
		case 2: chan->name->setString("tensorrt");			chan->value = myEngine && myEngine->providerDescription().rfind("TensorRT", 0) == 0 ? 1.f : 0.f; break;
		case 3: chan->name->setString("num_detections");	chan->value = (float)(myCropMode ? myCrops.size() : myYolo.detections().size()); break;
		case 4: chan->name->setString("infer_ms");			chan->value = (float)myInferMs; break;
		case 5: chan->name->setString("execute_ms");		chan->value = (float)myExecuteMs; break;
		case 6: chan->name->setString("model_w");			chan->value = (float)myModelW; break;
		case 7: chan->name->setString("model_h");			chan->value = (float)myModelH; break;
		case 8: chan->name->setString("trt_building");		chan->value = myUpgrading ? 1.f : 0.f; break;
		// Input frames since the one the current results are from (0 with Wait for Result on).
		case 10: chan->name->setString("run_ms");			chan->value = (float)myRunMs; break;
		case 9: chan->name->setString("result_delay");		chan->value = myResultFrame >= 0 && myInputFrame >= myResultFrame ?
			(float)(myInputFrame - myResultFrame) : 0.f; break;
	}
}

bool OnnxTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void*)
{
	if (myCropMode)
	{
		infoSize->rows = 1 + (int32_t)myCrops.size();
		infoSize->cols = 3 + myCropClasses;
		infoSize->byColumn = false;
		return true;
	}
	infoSize->rows = 1 + (int32_t)myYolo.detections().size();
	infoSize->cols = infoBaseCols() + myYolo.keypointCount() * 3;
	infoSize->byColumn = false;
	return true;
}

void OnnxTOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	// Columns: id, class, conf, u1, v1, u2, v2, [mask_fill (seg)], [kp{i}_u/v/c (pose)]
	char buf[64];
	if (myCropMode)
	{
		// Columns: id (the Crop DAT's track_id / id), class (argmax), conf, then one per class.
		const bool valid = index >= 1 && index - 1 < (int32_t)myCrops.size();
		for (int c = 0; c < nEntries; c++)
		{
			if (index == 0)
			{
				const char* fixed[] = { "id", "class", "conf" };
				if (c < 3)
					entries->values[c]->setString(fixed[c]);
				else if (c - 3 < (int)myClassNames.size())
					entries->values[c]->setString(myClassNames[c - 3].c_str());
				else
				{
					snprintf(buf, sizeof(buf), "c%d", c - 3);
					entries->values[c]->setString(buf);
				}
				continue;
			}
			if (!valid)
			{
				entries->values[c]->setString("0");
				continue;
			}
			const CropResult& r = myCrops[index - 1];
			if (c == 0)
				entries->values[c]->setString(r.id.c_str());
			else if (c == 1)
			{
				snprintf(buf, sizeof(buf), "%d", r.cls);
				entries->values[c]->setString(buf);
			}
			else
			{
				float v = c == 2 ? r.conf : (c - 3 < (int)r.probs.size() ? r.probs[c - 3] : 0.f);
				formatValue(buf, sizeof(buf), v, false);
				entries->values[c]->setString(buf);
			}
		}
		return;
	}
	const int base = infoBaseCols();
	if (index == 0)
	{
		headerRow(nEntries);
		for (int c = 0; c < nEntries; c++)
			entries->values[c]->setString(myHeader[(size_t)c].c_str());
		return;
	}
	// TD asks for rows based on an earlier getInfoDATSize(); never trust the index blindly.
	if (index < 1 || index - 1 >= (int32_t)myYolo.detections().size())
	{
		for (int c = 0; c < nEntries; c++)
			entries->values[c]->setString("0");
		return;
	}
	const Detection& d = myYolo.detections()[index - 1];
	float vals[] = { (float)(index - 1), (float)d.cls, d.conf, d.u1, d.v1, d.u2, d.v2, d.maskFill };
	for (int c = 0; c < nEntries; c++)
	{
		float v = c < base ? vals[c] : (c - base < (int)d.keypoints.size() ? d.keypoints[c - base] : 0.f);
		formatValue(buf, sizeof(buf), v, c < 2);
		entries->values[c]->setString(buf);
	}
}

void OnnxTOP::headerRow(int cols)
{
	const int base = infoBaseCols();
	const std::string key = std::to_string(base) + "/" + std::to_string(cols);
	if (key == myHeaderKey)
		return;
	myHeaderKey = key;
	myHeader.assign((size_t)std::max(cols, 0), std::string());
	const char* header[] = { "id", "class", "conf", "u1", "v1", "u2", "v2", "mask_fill" };
	const char* suffix[] = { "u", "v", "c" };
	char buf[64];
	for (int c = 0; c < cols; c++)
	{
		if (c < base)
			myHeader[(size_t)c] = header[c];
		else
		{
			snprintf(buf, sizeof(buf), "kp%d_%s", (c - base) / 3, suffix[(c - base) % 3]);
			myHeader[(size_t)c] = buf;
		}
	}
}

int OnnxTOP::infoBaseCols() const
{
	return myYolo.isSegmentation() ? 8 : 7;
}

void OnnxTOP::getWarningString(OP_String* warning, void*)
{
	if (myError.empty() && !myWarning.empty())
		warning->setString(myWarning.c_str());
}

void OnnxTOP::getErrorString(OP_String* error, void*)
{
	if (!myError.empty())
		error->setString((myModelHint.empty() ? myError : myError + "\n" + myModelHint).c_str());
}

void OnnxTOP::getInfoPopupString(OP_String* info, void*)
{
	std::string s = "td-native-inference " TDNI_VERSION "\nONNX Runtime " + ortload::version() + "\n" +
		ortload::loadedPath() + "\n";
	if (myEngine)
	{
		s += "Provider: " + myEngine->providerDescription() + "\n";
		for (const auto& t : myEngine->inputs())
			s += "in  " + t.name + " " + shapeString(t.shape) + " " + ortTypeName(t.type) + "\n";
		for (const auto& t : myEngine->outputs())
			s += "out " + t.name + " " + shapeString(t.shape) + " " + ortTypeName(t.type) + "\n";
	}
	std::string log = ortload::recentLog();
	if (!log.empty())
		s += "\nRecent ORT log:\n" + log;
	info->setString(s.c_str());
}

// ---------------------------------------------------------------- parameters

void OnnxTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	auto check = [](OP_ParAppendResult r) { assert(r == OP_ParAppendResult::Success); (void)r; };

	{
		OP_StringParameter sp("Model");
		sp.label = "Model (.onnx)";
		sp.page = "ONNX";
		check(manager->appendFile(sp));
	}
	{
		OP_StringParameter sp("Provider");
		sp.label = "Execution Provider";
		sp.page = "ONNX";
		sp.defaultValue = kProviderNames[0];
		sp.help = "Auto runs on CUDA right away and switches to TensorRT (faster) once its engine is built "
			"in the background; later loads use the cached engine directly. TensorRT waits for the build.";
		check(manager->appendMenu(sp, 3, kProviderNames, kProviderLabels));
	}
	{
		OP_NumericParameter np("Fp16");
		np.label = "TensorRT FP16";
		np.page = "ONNX";
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Reload");
		np.label = "Reload Model";
		np.page = "ONNX";
		check(manager->appendPulse(np));
	}
	{
		OP_NumericParameter np("Active");
		np.label = "Active";
		np.page = "ONNX";
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Waitforresult");
		np.label = "Wait for Result";
		np.page = "ONNX";
		np.defaultValues[0] = 1.0;
		np.help = "On: the cook waits for this frame's inference, so detections match the image exactly. Off: "
			"the cook queues the inference and returns; results arrive a cook later (info channel result_delay), "
			"freeing TouchDesigner's main thread for the inference time. Raw output never waits either way.";
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Mode");
		sp.label = "Output Mode";
		sp.page = "ONNX";
		sp.defaultValue = kModeNames[0];
		sp.help = "YOLO: Ultralytics end2end det / seg / pose. MoveNet: multipose [1, 6, 56] (set Input Normalize "
			"0 to 255 and Dynamic Size, e.g. 256 x 256). YuNet: OpenCV Zoo face detector (set Input Normalize "
			"0 to 255 and Channel Order BGR). Raw: show an output tensor as an image (e.g. Depth Anything).";
		check(manager->appendMenu(sp, 5, kModeNames, kModeLabels));
	}
	{
		OP_NumericParameter np("Letterbox");
		np.label = "Letterbox (Keep Aspect)";
		np.page = "ONNX";
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Inputnorm");
		sp.label = "Input Normalize";
		sp.page = "ONNX";
		sp.defaultValue = kInputNormNames[0];
		sp.help = "How RGB is scaled into the model input. YOLO: 0 to 1. Depth Anything and most ViT models: ImageNet.";
		check(manager->appendMenu(sp, 5, kInputNormNames, kInputNormLabels));
	}
	{
		OP_NumericParameter np("Normscale");
		np.label = "Custom Scale";
		np.page = "ONNX";
		np.help = "Custom normalize: tensor = rgb01 * scale + offset, per channel in model channel order.";
		for (int i = 0; i < 3; i++)
		{
			np.defaultValues[i] = 1.0;
			np.minSliders[i] = 0.0;
			np.maxSliders[i] = 255.0;
		}
		check(manager->appendFloat(np, 3));
	}
	{
		OP_NumericParameter np("Normoffset");
		np.label = "Custom Offset";
		np.page = "ONNX";
		for (int i = 0; i < 3; i++)
		{
			np.defaultValues[i] = 0.0;
			np.minSliders[i] = -2.0;
			np.maxSliders[i] = 2.0;
		}
		check(manager->appendFloat(np, 3));
	}
	{
		OP_StringParameter sp("Channelorder");
		sp.label = "Channel Order";
		sp.page = "ONNX";
		sp.defaultValue = kChannelNames[0];
		check(manager->appendMenu(sp, 2, kChannelNames, kChannelLabels));
	}
	{
		OP_NumericParameter np("Dynamicsize");
		np.label = "Dynamic Size (W, H)";
		np.page = "ONNX";
		np.help = "Input size for models whose input H/W are dynamic (ignored otherwise). Depth Anything: "
			"multiples of 14, e.g. 518 x 518, or 924 x 518 for 16:9 without letterbox bars. "
			"Each size is its own TensorRT engine.";
		for (int i = 0; i < 2; i++)
		{
			np.defaultValues[i] = 518;
			np.minValues[i] = np.minSliders[i] = 14;
			np.maxSliders[i] = 1036;
			np.clampMins[i] = true;
		}
		check(manager->appendInt(np, 2));
	}
	{
		OP_NumericParameter np("Rawoutput");
		np.label = "Raw Output Index";
		np.page = "Raw";
		np.minValues[0] = np.minSliders[0] = 0;
		np.maxSliders[0] = 4;
		np.clampMins[0] = true;
		check(manager->appendInt(np));
	}
	{
		OP_StringParameter sp("Rawnormalize");
		sp.label = "Normalize";
		sp.page = "Raw";
		sp.defaultValue = kRawNormNames[1];
		sp.help = "Min / Max maps each frame's value range to 0-1 on the GPU (e.g. Depth Anything's relative depth).";
		check(manager->appendMenu(sp, 2, kRawNormNames, kRawNormLabels));
	}
	{
		OP_NumericParameter np("Rawsmoothing");
		np.label = "Range Smoothing";
		np.page = "Raw";
		np.defaultValues[0] = 0.9;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = 0.999;
		np.maxSliders[0] = 0.99;
		np.clampMins[0] = np.clampMaxes[0] = true;
		np.help = "Temporal smoothing of the min/max range, so brightness doesn't pump when the scene changes. 0 = off.";
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Rawmaptoinput");
		np.label = "Map to Input Frame";
		np.page = "Raw";
		np.defaultValues[0] = 1.0;
		np.help = "On: output matches the input's size and framing (letterbox undone). Off: the tensor's own size.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Confthreshold");
		np.label = "Confidence Threshold";
		np.page = "Detection";
		np.defaultValues[0] = 0.4;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = np.maxSliders[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Classfilter");
		np.label = "Class Filter (-1 = All)";
		np.page = "Detection";
		np.defaultValues[0] = 0;	// COCO 'person'
		np.minValues[0] = np.minSliders[0] = -1;
		np.maxSliders[0] = 79;
		np.clampMins[0] = true;
		check(manager->appendInt(np));
	}
	{
		OP_NumericParameter np("Maxdetections");
		np.label = "Max Detections";
		np.page = "Detection";
		np.defaultValues[0] = 16;
		np.minValues[0] = np.minSliders[0] = 1;
		np.maxValues[0] = np.maxSliders[0] = kMaxDetsCap;
		np.clampMins[0] = np.clampMaxes[0] = true;
		check(manager->appendInt(np));
	}
	{
		OP_StringParameter sp("Cropdat");
		sp.label = "Crop DAT";
		sp.page = "Crops";
		sp.help = "Boxes to classify: a tracks table (track_id, x_left, y_bottom, x_right, y_top -- e.g. "
			"OnnxTopTables' script_tracks) or an OnnxTOP Info DAT (id, u1, v1, u2, v2). The node's input must be "
			"the same image those boxes came from.";
		check(manager->appendDAT(sp));
	}
	{
		OP_NumericParameter np("Batchsize");
		np.label = "Batch Size (Max Crops)";
		np.page = "Crops";
		np.defaultValues[0] = 16;
		np.minValues[0] = np.minSliders[0] = 1;
		np.maxValues[0] = np.maxSliders[0] = kMaxBatch;
		np.clampMins[0] = np.clampMaxes[0] = true;
		np.help = "Fixed batch for models with a dynamic batch dim (every frame runs this many; unused slots are "
			"zero), so the shape never changes. Part of the TensorRT cache key.";
		check(manager->appendInt(np));
	}
	{
		OP_NumericParameter np("Cropscale");
		np.label = "Crop Scale";
		np.page = "Crops";
		np.defaultValues[0] = 1.0;
		np.minValues[0] = np.minSliders[0] = 0.25;
		np.maxSliders[0] = 2.0;
		np.clampMins[0] = true;
		np.help = "Grows (> 1) or shrinks each box around its center before cropping.";
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Cropsquare");
		np.label = "Square Crops";
		np.page = "Crops";
		np.defaultValues[0] = 0.0;
		np.help = "Off: the box is stretched to the model input, like onnx_hsemotion.py. On: square around the box center.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Classifyinterval");
		np.label = "Classify Interval (Frames)";
		np.page = "Crops";
		np.defaultValues[0] = 1;
		np.minValues[0] = np.minSliders[0] = 1;
		np.maxSliders[0] = 10;
		np.clampMins[0] = true;
		np.help = "Run the classifier every Nth frame and hold the results in between (3 = onnx_hsemotion.py's "
			"default). Expressions change slowly, so 2-4 cuts the cost proportionally with little visible difference.";
		check(manager->appendInt(np));
	}
	{
		OP_NumericParameter np("Softmax");
		np.label = "Softmax";
		np.page = "Crops";
		np.defaultValues[0] = 1.0;
		np.help = "Turn logits into probabilities (off if the model already outputs them).";
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Classnames");
		sp.label = "Class Names";
		sp.page = "Crops";
		sp.help = "Space- or comma-separated column names for the Info DAT, in output order (empty = c0, c1, ...). "
			"HSEmotion: anger disgust fear happiness neutral sadness surprise";
		check(manager->appendString(sp));
	}
	{
		OP_NumericParameter np("Nmsthreshold");
		np.label = "NMS IoU Threshold";
		np.page = "Detection";
		np.defaultValues[0] = 0.3;	// cv2.FaceDetectorYN's, as used by onnx_yunet.py
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = np.maxSliders[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		np.help = "YuNet: overlapping faces with IoU above this are merged (YOLO end2end and MoveNet need no NMS).";
		check(manager->appendFloat(np));
	}
	// Seg-only. Defaults match onnx_yolo26_seg.py (MASK_THRESHOLD, CONF_THRESHOLD,
	// MIN_MASK_AREA_RATIO). Every detection above Confidence Threshold is still reported.
	auto segFloat = [&](const char* name, const char* label, double def) {
		OP_NumericParameter np(name);
		np.label = label;
		np.page = "Detection";
		np.defaultValues[0] = def;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = np.maxSliders[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		check(manager->appendFloat(np));
	};
	// Mask Box Margin: masks fade out past their box over this fraction of its size; 0 = hard crop.
	segFloat("Maskthreshold", "Mask Threshold (Fill Ratio)", 0.5);
	segFloat("Matteconf", "Matte Min Confidence", 0.35);
	segFloat("Matteminfill", "Matte Min Mask Fill", 0.15);
	segFloat("Maskmargin", "Mask Box Margin", 0.1);
	segFloat("Mattenms", "Matte NMS IoU", 0.5);
	{
		const char* names[] = { "grayscale", "instances" };
		const char* labels[] = { "Grayscale", "Instances (R/A matte, G detection, B class)" };
		OP_StringParameter sp("Matteencoding");
		sp.label = "Matte Encoding";
		sp.page = "Detection";
		sp.defaultValue = names[0];
		sp.help = "Instances: G = the detection that owns each pixel (Info DAT id + 1, 0 = none; overlaps go to "
			"the most confident mask), B = its class + 1. R and A stay the matte.";
		check(manager->appendMenu(sp, 2, names, labels));
	}

	{
		OP_StringParameter sp("Cachefolder");
		sp.label = "TensorRT Cache Folder (optional)";
		sp.page = "ONNX";
		sp.help = "Leave empty to cache TensorRT engines in trt_cache/ next to the model. Set only to share one folder.";
		check(manager->appendFolder(sp));
	}
}

void OnnxTOP::pulsePressed(const char* name, void*)
{
	if (!strcmp(name, "Reload"))
		myForceReload = true;
}
