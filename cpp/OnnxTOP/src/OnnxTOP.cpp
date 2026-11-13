#include "OnnxTOP.h"

#include <algorithm>
#include <array>
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

	// Running average of a timing (a time constant of about ten cooks), for the wrappers' Performance page.
	void average(double& avg, double value)
	{
		avg = avg < 0.0 ? value : avg + (value - avg) * 0.1;
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
	const char* kModeNames[] = { "yolo", "movenet", "yunet", "crops", "raw", "posecrops", "landmarkcrops", "holistic" };
	const char* kModeLabels[] = { "YOLO End2End (det / seg / pose)", "MoveNet Multipose", "YuNet Faces",
		"Classify Crops (boxes from a DAT)", "Raw Output Tensor", "Pose Crops (RTMPose / RTMW SimCC)",
		"Landmark Crops (rotated ROIs, raw outputs)", "MediaPipe Holistic (pose, hands, face)" };
	enum Mode { ModeYolo = 0, ModeMoveNet = 1, ModeYuNet = 2, ModeCrops = 3, ModeRaw = 4, ModePoseCrops = 5,
		ModeLandmarkCrops = 6, ModeHolistic = 7 };
	// Holistic: the models loaded next to the pose model (Model), in OnnxTOP::myCompanions order.
	const char* kHolisticCompanions[] = { "hand_roi_refinement.onnx", "hand_landmarks_detector.onnx",
		"face_landmarks_detector.onnx" };
	constexpr int kCropSheetCell = 128;	// px per tile of the Classify Crops debug image
	// Landmark Crops: outputs with at most this many values per crop go to the Info DAT (landmarks,
	// scores, world points); bigger ones (heatmaps, masks) are left on the GPU.
	constexpr int64_t kLandmarkMaxValues = 4096;
	const char* kRoiCols[] = { "roi_x", "roi_y", "roi_w", "roi_h", "roi_angle" };

	// Landmark Crops' DAT: MediaPipe NormalizedRects, one per row: roi_x, roi_y (center), roi_w,
	// roi_h (fractions of the image width / height, so a square ROI has roi_w * W == roi_h * H),
	// roi_angle (radians), all top-down like MediaPipe; plus an optional id.
	bool parseRoiDat(const OP_DATInput* dat, int srcW, int srcH, int maxN, std::vector<CropBox>& boxes,
		std::vector<std::string>& ids, std::vector<std::array<float, 5>>& rois, std::string& err)
	{
		boxes.clear();
		ids.clear();
		rois.clear();
		if (!dat)
		{
			err = "Set Crop DAT to a table of ROIs (roi_x, roi_y, roi_w, roi_h, roi_angle)";
			return false;
		}
		if (!dat->isTable || dat->numRows < 1 || dat->numCols < 1)
			return true;
		int c[5], idc = -1;
		for (int k = 0; k < 5; k++)
			c[k] = -1;
		for (int col = 0; col < dat->numCols; col++)
		{
			const char* h = dat->getCell(0, col);
			if (!h)
				continue;
			if (!strcmp(h, "id"))
				idc = col;
			for (int k = 0; k < 5; k++)
				if (!strcmp(h, kRoiCols[k]))
					c[k] = col;
		}
		if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[3] < 0)
		{
			err = "Crop DAT needs columns roi_x, roi_y, roi_w, roi_h (and roi_angle) for Landmark Crops";
			return false;
		}
		auto num = [&](int r, int cc) { const char* v = cc >= 0 ? dat->getCell(r, cc) : nullptr; return v ? (float)atof(v) : 0.f; };
		for (int r = 1; r < dat->numRows && (int)boxes.size() < maxN; r++)
		{
			std::array<float, 5> roi = { num(r, c[0]), num(r, c[1]), num(r, c[2]), num(r, c[3]), num(r, c[4]) };
			float cx = roi[0] * srcW, cy = roi[1] * srcH, w = roi[2] * srcW, h = roi[3] * srcH;
			if (w < 2.f || h < 2.f)
				continue;
			CropBox b{ cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f };
			b.cosA = std::cos(roi[4]);
			b.sinA = std::sin(roi[4]);
			b.replicate = 1;
			boxes.push_back(b);
			rois.push_back(roi);
			const char* id = idc >= 0 ? dat->getCell(r, idc) : nullptr;
			ids.push_back(id ? id : std::to_string(r - 1));
		}
		return true;
	}

	// Crop DAT rows -> source-pixel boxes (top-down) + ids. Accepts a tracks table
	// (track_id, x_left, y_bottom, x_right, y_top) or an OnnxTOP Info DAT (id, u1, v1, u2, v2),
	// both in TD UV of the source image (origin bottom-left).
	// aspect > 0 (Pose Crops): each box grows to that width / height ratio and is NOT clamped to the
	// image (the model sees black beyond it), so the crop maps back to the image by a plain scale.
	bool parseCropDat(const OP_DATInput* dat, int srcW, int srcH, float scale, bool square, int maxN,
		std::vector<CropBox>& boxes, std::vector<std::string>& ids, std::string& err, float aspect = 0.f)
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
			if (aspect > 0.f && w > 0.f && h > 0.f)
			{
				if (w / h < aspect)
					w = h * aspect;
				else
					h = w / aspect;
			}
			CropBox b = aspect > 0.f ? CropBox{ cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f }
				: CropBox{ std::max(0.f, cx - w * 0.5f), std::max(0.f, cy - h * 0.5f),
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
	cudaStreamCreateWithFlags(&myIoStream, cudaStreamNonBlocking);
	cudaEventCreate(&myRunStart);
	cudaEventCreate(&myRunDone);
	cudaEventCreateWithFlags(&myHoloCopied, cudaEventDisableTiming);
}

OnnxTOP::~OnnxTOP()
{
	// Kills an in-progress out-of-process TensorRT build. An in-process ORT session load
	// can't be interrupted, but with a cached engine that's only a few seconds.
	myCancelLoad = true;
	if (myLoader.joinable())
		myLoader.join();
	stopHolisticWorker();
	if (myStream)
		cudaStreamSynchronize(myStream);
	if (myIoStream)
		cudaStreamSynchronize(myIoStream);
	if (myHoloFrame)
		cudaFreeArray(myHoloFrame);
	if (myHoloCopied)
		cudaEventDestroy(myHoloCopied);
	if (myIoStream)
		cudaStreamDestroy(myIoStream);
	if (myRunStart)
		cudaEventDestroy(myRunStart);
	if (myRunDone)
		cudaEventDestroy(myRunDone);
	myEngine.reset();
	myNextEngine.reset();
	myCompanions.clear();
	myNextCompanions.clear();
	if (myHostStage)
		cudaFreeHost(myHostStage);
	if (myDevSheetBoxes)
		cudaFree(myDevSheetBoxes);
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

void OnnxTOP::manageLoading(const EngineConfig& desired, const std::vector<std::string>& companions)
{
	// Hand over a finished load (main thread only touches myEngine).
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		if (myNextEngine)
		{
			// The background Holistic run may be using the outgoing engines; last frame's kernels may
			// still be reading their buffers.
			waitHolisticIdle();
			if (myStream)
				cudaStreamSynchronize(myStream);
			myEngine = std::move(myNextEngine);
			myCompanions = std::move(myNextCompanions);
			myRawRangeReset = true;
		}
	}

	if (myLoadState == LoadState::Loading)
		return;

	bool changed = !myHasRequested || !sameModelConfig(desired, myRequestedConfig) || companions != myRequestedCompanions;
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
	waitHolisticIdle();
	if (myStream)
		cudaStreamSynchronize(myStream);
	myEngine.reset();
	myCompanions.clear();
	{
		std::lock_guard<std::mutex> lock(myLoadMutex);
		myNextEngine.reset();
		myNextCompanions.clear();
		myLoadError.clear();
		myLoadWarning.clear();
	}

	myRequestedConfig = desired;
	myRequestedCompanions = companions;
	myHasRequested = true;

	if (desired.modelPath.empty())
	{
		myLoadState = LoadState::Idle;
		return;
	}

	myLoadState = LoadState::Loading;
	myCancelLoad = false;
	setLoadStatus("Loading model...");
	myLoader = std::thread([this, desired, companions]() { loadEngines(desired, companions); });
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

void OnnxTOP::loadEngines(EngineConfig cfg, std::vector<std::string> companions)
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

	// The main model, then its companions (Holistic: the hand and face models next to the pose
	// model), each a fixed-size engine of its own. All load and upgrade together.
	std::vector<EngineConfig> cfgs{ cfg };
	for (const std::string& path : companions)
	{
		std::error_code ec;
		if (!fs::exists(path, ec))
		{
			failLoad("Missing " + path + " (Holistic needs the hand and face models next to the pose model; "
				"run scripts/convert_holistic_onnx.py)");
			return;
		}
		EngineConfig c = cfg;
		c.modelPath = path;
		c.inputW = c.inputH = 0;
		c.batch = 0;
		cfgs.push_back(c);
	}
	auto loadAll = [&](OrtProvider provider, std::vector<std::unique_ptr<OnnxEngine>>& out) -> bool
	{
		out.clear();
		for (EngineConfig c : cfgs)
		{
			c.provider = provider;
			auto engine = std::make_unique<OnnxEngine>();
			if (!engine->load(c, err))
			{
				if (cfgs.size() > 1)
					err = fs::path(c.modelPath).filename().string() + ": " + err;
				return false;
			}
			out.push_back(std::move(engine));
		}
		return true;
	};
	auto publishAll = [&](std::vector<std::unique_ptr<OnnxEngine>>& set, const std::string& warning) -> bool
	{
		auto main = std::move(set[0]);
		set.erase(set.begin());
		return publishEngine(std::move(main), warning, std::move(set));
	};

	// TensorRT engine builds run in onnxtop_tool.exe, never in TD's process (see
	// TrtBuilder.h). Once the marker exists, loading is a pure cache hit.
	// claimTensorRT() also pins TD's TensorRT DLLs before anything else can load another
	// build; if a foreign one is already loaded, OnnxEngine::load() falls back to CUDA.
	bool trtUsable = cfg.provider != OrtProvider::CUDA && ortload::claimTensorRT().empty();
	std::vector<bool> cached;
	for (const EngineConfig& c : cfgs)
		cached.push_back(trtUsable && fs::exists(trtBuildMarkerPath(c)));
	const bool allCached = std::all_of(cached.begin(), cached.end(), [](bool b) { return b; });
	std::vector<std::unique_ptr<OnnxEngine>> set;

	if (cfg.provider == OrtProvider::Auto && trtUsable && !allCached)
	{
		// Auto, first run for this engine: run on CUDA now, build TensorRT in the background,
		// then swap. Both engines are briefly resident, and the build competes for the GPU.
		setLoadStatus("Loading model on CUDA...");
		if (!loadAll(OrtProvider::CUDA, set))
		{
			failLoad(err);
			return;
		}
		myUpgrading = true;
		if (!publishAll(set, "Running on CUDA while the TensorRT engine builds in the background "
			"(first run for this model / size / precision: a few minutes, lower frame rate meanwhile)"))
		{
			myUpgrading = false;
			return;
		}

		std::string message;
		for (size_t i = 0; i < cfgs.size(); i++)
		{
			if (cached[i])
				continue;
			int built = buildTrtEngine(cfgs[i], true, message);
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
		}
		if (!loadAll(OrtProvider::TensorRT, set))
			setLoadWarning("Running on CUDA -- TensorRT engine load failed: " + err);
		else
			publishAll(set, message);
		myUpgrading = false;
		return;
	}

	std::string warning;
	OrtProvider provider = cfg.provider == OrtProvider::Auto ? OrtProvider::TensorRT : cfg.provider;	// cached (or TensorRT unusable: load() picks CUDA)
	if (provider == OrtProvider::TensorRT && trtUsable && !allCached)
	{
		for (size_t i = 0; i < cfgs.size(); i++)
		{
			if (cached[i])
				continue;
			int built = buildTrtEngine(cfgs[i], false, warning);
			if (built < 0)
				return;
			if (built == 0)
			{
				failLoad("TensorRT engine build failed: " + warning +
					". Set Execution Provider to Auto or CUDA to run without it.");
				return;
			}
		}
		setLoadStatus("Loading cached TensorRT engine...");
	}

	if (!loadAll(provider, set))
	{
		failLoad(err);
		return;
	}
	publishAll(set, warning);
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

bool OnnxTOP::publishEngine(std::unique_ptr<OnnxEngine> engine, const std::string& warning,
	std::vector<std::unique_ptr<OnnxEngine>> companions)
{
	std::lock_guard<std::mutex> lock(myLoadMutex);
	if (myCancelLoad)
		return false;
	myNextEngine = std::move(engine);
	myNextCompanions = std::move(companions);
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
	const bool cropDatMode = mode == ModeCrops || mode == ModePoseCrops || mode == ModeLandmarkCrops || mode == ModeHolistic;
	HolisticOptions holisticOpt;
	holisticOpt.hands = inputs->getParInt("Holistichands") != 0;
	holisticOpt.face = inputs->getParInt("Holisticface") != 0;
	const OP_DATInput* cropDat = cropDatMode ? inputs->getParDAT("Cropdat") : nullptr;
	myClassNames = splitNames(inputs->getParString("Classnames"));
	RawOptions rawOpt;
	rawOpt.index = inputs->getParInt("Rawoutput");
	rawOpt.minMax = inputs->getParInt("Rawnormalize") == 1;
	rawOpt.smoothing = (float)std::clamp(inputs->getParDouble("Rawsmoothing"), 0.0, 0.999);
	rawOpt.mapToSource = inputs->getParInt("Rawmaptoinput") != 0;
	const bool rawAll = inputs->getParInt("Rawalloutputs") != 0;
	const float letterboxPad = inputs->getParInt("Letterboxpad") == 1 ? -1.f : kLetterboxPad;
	SegOptions segOpt;
	segOpt.maskThreshold = (float)inputs->getParDouble("Maskthreshold");
	segOpt.matteConf = (float)inputs->getParDouble("Matteconf");
	segOpt.matteMinFill = (float)inputs->getParDouble("Matteminfill");
	segOpt.maskMargin = (float)inputs->getParDouble("Maskmargin");
	segOpt.matteNms = (float)inputs->getParDouble("Mattenms");
	segOpt.instances = inputs->getParInt("Matteencoding") == 1;
	myWaitResult = inputs->getParInt("Waitforresult") != 0;
	// Holistic with Wait for Result off runs in the background; anything else must not race it.
	const bool holoAsync = mode == ModeHolistic && !myWaitResult;
	if (!holoAsync)
		waitHolisticIdle();
	cudaStream_t ioStream = holoAsync ? myIoStream : myStream;
	const float scalarInput = (float)inputs->getParDouble("Scalarinput");

	inputs->enablePar("Fp16", provider != OrtProvider::CUDA);
	inputs->enablePar("Cachefolder", provider != OrtProvider::CUDA);
	for (const char* p : { "Confthreshold", "Maxdetections" })
		inputs->enablePar(p, mode != ModeRaw && mode != ModeCrops && mode != ModeLandmarkCrops);
	for (const char* p : { "Batchsize", "Cropsquare", "Softmax", "Classnames", "Classifyinterval" })
		inputs->enablePar(p, mode == ModeCrops);
	inputs->enablePar("Cropdat", cropDatMode);
	inputs->enablePar("Cropscale", mode == ModeCrops || mode == ModePoseCrops);
	inputs->enablePar("Letterbox", !cropDatMode);
	inputs->enablePar("Holistichands", mode == ModeHolistic);
	inputs->enablePar("Holisticface", mode == ModeHolistic);
	inputs->enablePar("Inputnorm", mode != ModeHolistic);
	inputs->enablePar("Classfilter", mode == ModeYolo);
	inputs->enablePar("Nmsthreshold", mode == ModeYuNet);
	inputs->enablePar("Normscale", normKind == InputNormKind::Custom);
	inputs->enablePar("Normoffset", normKind == InputNormKind::Custom);
	for (const char* p : { "Maskthreshold", "Matteconf", "Matteminfill", "Maskmargin", "Mattenms", "Matteencoding" })
		inputs->enablePar(p, mode == ModeYolo && myYolo.isSegmentation());
	for (const char* p : { "Rawoutput", "Rawnormalize", "Rawmaptoinput", "Rawalloutputs" })
		inputs->enablePar(p, mode == ModeRaw);
	inputs->enablePar("Letterboxpad", !cropDatMode && inputs->getParInt("Letterbox") != 0);
	inputs->enablePar("Rawsmoothing", mode == ModeRaw && rawOpt.minMax);
	inputs->enablePar("Waitforresult", mode != ModeRaw);
	inputs->enablePar("Scalarinput", myEngine && myEngine->hasScalarInput());
	inputs->enablePar("Resetstate", myEngine && myEngine->hasState());

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
	desired.scalarInput = scalarInput;
	std::vector<std::string> companions;
	if (mode == ModeHolistic && !desired.modelPath.empty())
		for (const char* name : kHolisticCompanions)
			companions.push_back((fs::path(desired.modelPath).parent_path() / name).generic_string());
	holisticOpt.maxPeople = maxDets;
	manageLoading(desired, companions);
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
	// Pose Crops: boxes at the model's own aspect (width / height of its input).
	float poseAspect = 0.f;
	if (mode == ModePoseCrops && myEngine && !myEngine->inputs().empty() && myEngine->inputs()[0].shape.size() == 4)
	{
		const auto& ps = myEngine->inputs()[0].shape;
		poseAspect = (float)ps[widthDim(ps)] / (float)std::max<int64_t>(1, ps[heightDim(ps)]);
	}
	if ((mode == ModeCrops || mode == ModePoseCrops) &&
		!parseCropDat(cropDat, srcW, srcH, cropScale, cropSquare && mode == ModeCrops, kMaxBatch, myCropBoxes, myCropIds, err, poseAspect))
		myWarning = err;
	if (mode == ModeLandmarkCrops && !parseRoiDat(cropDat, srcW, srcH, kMaxBatch, myCropBoxes, myCropIds, myRois, err))
		myWarning = err;
	if (mode == ModeHolistic && cropDat && !parseCropDat(cropDat, srcW, srcH, 1.f, false, kMaxBatch, myCropBoxes, myCropIds, err))
		myWarning = err;
	if (mode == ModeHolistic && !cropDat)
	{
		myCropBoxes.clear();
		myCropIds.clear();
		if (myWarning.empty())
			myWarning = "Set Crop DAT to a person detector's tracks (e.g. Yolo26Detect's), so new people are found";
	}

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
	outInfo.stream = ioStream;
	output->getSuggestedOutputDesc(&outInfo.textureDesc, nullptr);
	outInfo.textureDesc.texDim = OP_TexDim::e2D;
	outInfo.textureDesc.depth = 1;
	outInfo.textureDesc.pixelFormat = OP_PixelFormat::RGBA16Float;

	if (engine && cropDatMode)
	{
		// Debug image: a contact sheet of every batch slot's crop (Pose / Landmark Crops: one cell per crop;
		// Holistic: a fixed grid with room for each person's pose, hand and face crops).
		int B = mode != ModeCrops ? std::max(1, (int)std::min<size_t>(myCropBoxes.size(), kMaxBatch)) : (int)in->shape[0];
		if (mode == ModeHolistic)
			B = std::min(kMaxBatch, std::clamp(maxDets, 1, 8) * (1 + (holisticOpt.hands ? 2 : 0) + (holisticOpt.face ? 1 : 0)));
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
	// Raw + All Outputs: every other image-shaped output in its own color buffer (1, 2, ... in
	// output order; a Render Select TOP reads them), always raw values, same framing as the main one.
	struct ExtraOutput
	{
		RawOptions				opt;
		const OP_CUDAArrayInfo*	array = nullptr;
		int						w = 0, h = 0;
	};
	std::vector<ExtraOutput> extras;
	if (engine && mode == ModeRaw && rawAll)
	{
		uint32_t slot = 1;
		for (int k = 0; k < (int)engine->outputs().size(); k++)
		{
			int C, H, W;
			if (k == rawOpt.index || !rawDims(engine->outputs()[k], C, H, W))
				continue;
			ExtraOutput x;
			x.opt = rawOpt;
			x.opt.index = k;
			x.opt.minMax = false;
			x.opt.mono = C == 1;
			TOP_CUDAOutputInfo info = outInfo;
			info.colorBufferIndex = slot++;
			info.textureDesc.width = rawOpt.mapToSource ? top->textureDesc.width : W;
			info.textureDesc.height = rawOpt.mapToSource ? top->textureDesc.height : H;
			info.textureDesc.pixelFormat = x.opt.mono ? OP_PixelFormat::Mono32Float : OP_PixelFormat::RGBA32Float;
			x.w = (int)info.textureDesc.width;
			x.h = (int)info.textureDesc.height;
			x.array = output->createCUDAArray(info, nullptr);
			if (x.array)
				extras.push_back(x);
		}
	}
	OP_CUDAAcquireInfo acquire;
	acquire.stream = ioStream;
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
	std::vector<std::unique_ptr<FrameSurface>> extraSurfaces;
	for (const ExtraOutput& x : extras)
		extraSurfaces.push_back(std::make_unique<FrameSurface>(x.array->cudaArray));
	bool extrasWritten = false;

	bool wrote = false;
	// Wait for Result off: decode the run queued last cook, then queue this frame's without waiting.
	// Only for a new input frame: a still input (or a parameter change on one) runs synchronously,
	// so the node never keeps itself cooking. Raw output never reads back, so it always just queues.
	const bool async = !myWaitResult && mode != ModeRaw;
	const bool newInput = top->totalCooks != myLaunchedFrame;
	if (myPending.active && (!async || myPending.engine != engine || myPending.mode != mode))
		myPending.active = false;	// switched to waiting, or engine / mode changed: not wanted
	readInferMs();
	if (myResetState.exchange(false) && engine)
		engine->resetState();
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
	else if (engine && inSurf && mode == ModePoseCrops)
	{
		// One model run per person when the model's batch is 1 (RTMW3D-X), else per batch of crops.
		// Always synchronous: each run's SimCC heads are decoded before the next reuses the buffers.
		if (!myDevCropBoxes)
			cudaMalloc(&myDevCropBoxes, kMaxBatch * sizeof(CropBox));
		const int B = std::max(1, (int)in->shape[0]);
		const int n = (int)std::min<size_t>(myCropBoxes.size(), kMaxBatch);
		if (n > 0)
			cudaMemcpyAsync(myDevCropBoxes, myCropBoxes.data(), n * sizeof(CropBox), cudaMemcpyHostToDevice, myStream);
		myYolo.clearPose();
		bool ok = true;
		for (int start = 0; start < n && ok; start += B)
		{
			const int m = std::min(B, n - start);
			cudaError_t ce = launchCropPreprocess(inSurf, fmt, srcW, srcH, myDevCropBoxes + start, m, B, inputNorm, inLayout,
				in->device, myModelW, myModelH, myStream);
			if (ce != cudaSuccess)
			{
				myError = std::string("Crop kernel failed: ") + cudaGetErrorString(ce);
				ok = false;
			}
			else if (!queueRun(true))
				ok = false;
			else if (!myYolo.addSimcc3D(*engine, myCropBoxes.data() + start, m, srcW, srcH, confThresh, myStream, err))
			{
				myError = err;
				ok = false;
			}
		}
		readInferMs();
		myResultFrame = top->totalCooks;
		myYoloMode = true;
		if (ok)
		{
			const int cells = std::max(1, n);
			launchCropSheet(inSurf, fmt, srcW, srcH, myDevCropBoxes, n, (int)std::ceil(std::sqrt((double)cells)), kCropSheetCell,
				outSurf, outW, outH, myStream);
			wrote = true;
		}
		if ((int)myCropBoxes.size() > kMaxBatch && myWarning.empty())
			myWarning = std::to_string(myCropBoxes.size()) + " boxes but at most " + std::to_string(kMaxBatch) + " people are tracked";
	}
	else if (engine && inSurf && mode == ModeHolistic && !holoAsync)
	{
		// Wait for Result on: the whole chain in this cook (each stage's crops depend on the previous
		// stage's landmarks, so it waits for the GPU three times).
		std::vector<CropBox> sheet;
		if (runHolistic(inSurf, fmt, srcW, srcH, holisticOpt, myCropBoxes, myCropIds, true, sheet, err))
		{
			drawHolisticSheet(inSurf, fmt, srcW, srcH, sheet, outSurf, outW, outH, myStream);
			wrote = true;
		}
		else
		{
			myError = err;
			myHolistic.reset();
		}
		myHoloShown = myHolistic.people();
		readInferMs();
		myResultFrame = top->totalCooks;
		myLaunchedFrame = top->totalCooks;
	}
	else if (engine && inSurf && inArray && mode == ModeHolistic)
	{
		// Wait for Result off: show the worker's last finished result, and hand it this frame if it's free.
		if (!myHoloThread.joinable())
		{
			myHoloQuit = false;
			myHoloThread = std::thread([this]() { holisticWorker(); });
		}
		{
			std::lock_guard<std::mutex> lock(myHoloMutex);
			if (myHoloHasResult)
			{
				myHoloShown = std::move(myHoloPublished);
				myHoloSheet = std::move(myHoloPublishedSheet);
				if (!myHoloPublishedError.empty())
					myError = myHoloPublishedError;
				myResultFrame = myHoloPublishedFrame;
				myHoloHasResult = false;
			}
		}
		if (!myHoloBusy)
		{
			cudaChannelFormatDesc desc = {};
			cudaExtent ext = {};
			unsigned int flags = 0;
			cudaArrayGetInfo(&desc, &ext, &flags, inArray->cudaArray);
			const size_t h = std::max<size_t>(ext.height, 1);
			if (!myHoloFrame || ext.width != myHoloFrameW || h != myHoloFrameH || memcmp(&desc, &myHoloFrameDesc, sizeof(desc)) != 0)
			{
				if (myHoloFrame)
					cudaFreeArray(myHoloFrame);
				myHoloFrame = nullptr;
				if (cudaMallocArray(&myHoloFrame, &desc, ext.width, h, cudaArraySurfaceLoadStore) != cudaSuccess)
					myHoloFrame = nullptr;
				myHoloFrameW = ext.width;
				myHoloFrameH = h;
				myHoloFrameDesc = desc;
			}
			cudaMemcpy3DParms copy = {};
			copy.srcArray = inArray->cudaArray;
			copy.dstArray = myHoloFrame;
			copy.extent = make_cudaExtent(ext.width, h, 1);
			copy.kind = cudaMemcpyDeviceToDevice;
			if (myHoloFrame && cudaMemcpy3DAsync(&copy, myIoStream) == cudaSuccess)
			{
				cudaEventRecord(myHoloCopied, myIoStream);
				myHoloBusy = true;
				{
					std::lock_guard<std::mutex> lock(myHoloMutex);
					myHoloJob = { myCropBoxes, myCropIds, holisticOpt, fmt, srcW, srcH, top->totalCooks };
					myHoloHasJob = true;
				}
				myHoloCv.notify_one();
				myLaunchedFrame = top->totalCooks;
			}
			else
				myError = "Holistic: could not copy the input frame for the background run";
		}
		else
			myHoloDropped++;
		myInferMs = myHoloMs;
		drawHolisticSheet(inSurf, fmt, srcW, srcH, myHoloSheet, outSurf, outW, outH, myIoStream);
		wrote = true;
	}
	else if (engine && inSurf && mode == ModeLandmarkCrops)
	{
		// One run per batch of ROIs (MediaPipe's landmark models have batch 1); each run's small
		// outputs are read back before the next reuses the buffers. Always synchronous.
		if (!myDevCropBoxes)
			cudaMalloc(&myDevCropBoxes, kMaxBatch * sizeof(CropBox));
		const int B = std::max(1, (int)in->shape[0]);
		const int n = (int)std::min<size_t>(myCropBoxes.size(), kMaxBatch);
		if (n > 0)
			cudaMemcpyAsync(myDevCropBoxes, myCropBoxes.data(), n * sizeof(CropBox), cudaMemcpyHostToDevice, myStream);
		myLandmarks.clear();
		bool ok = true;
		for (int start = 0; start < n && ok; start += B)
		{
			const int m = std::min(B, n - start);
			cudaError_t ce = launchCropPreprocess(inSurf, fmt, srcW, srcH, myDevCropBoxes + start, m, B, inputNorm, inLayout,
				in->device, myModelW, myModelH, myStream);
			if (ce != cudaSuccess)
			{
				myError = std::string("Crop kernel failed: ") + cudaGetErrorString(ce);
				ok = false;
			}
			else if (!queueRun(true))
				ok = false;
			else if (!runLandmarkCrops(*engine, start, m, err))
			{
				myError = err;
				ok = false;
			}
		}
		readInferMs();
		myResultFrame = top->totalCooks;
		if (ok)
		{
			const int cells = std::max(1, n);
			launchCropSheet(inSurf, fmt, srcW, srcH, myDevCropBoxes, n, (int)std::ceil(std::sqrt((double)cells)), kCropSheetCell,
				outSurf, outW, outH, myStream);
			wrote = true;
		}
	}
	else if (engine && inSurf)
	{
		Letterbox lb = computeLetterbox((int)top->textureDesc.width, (int)top->textureDesc.height,
			myModelW, myModelH, letterbox);
		// Detections (and the seg matte) from the engine's current outputs, for the frame lb maps.
		auto decodeOutputs = [&](const Letterbox& from) -> bool
		{
			bool ok = true;
			if (mode == ModeYolo)
				ok = myYolo.decode(*engine, from, confThresh, classFilter, maxDets, segOpt, outSurf, outW, outH, myStream, err);
			else if (mode == ModeMoveNet)
				ok = myYolo.decodeMoveNet(*engine, from, confThresh, maxDets, outSurf, outW, outH, myStream, err);
			else if (mode == ModeYuNet)
				ok = myYolo.decodeYuNet(*engine, from, confThresh, nmsThresh, maxDets, outSurf, outW, outH, myStream, err);
			else
			{
				bool ok = runRaw(*engine, rawOpt, from, outSurf, outW, outH);
				for (size_t i = 0; i < extras.size(); i++)
					extrasWritten |= runRaw(*engine, extras[i].opt, from, extraSurfaces[i]->surf, extras[i].w, extras[i].h);
				return ok;
			}
			if (!ok)
				myError = err;
			myYoloMode = true;
			return true;	// the decoders always write (or clear) the output
		};
		auto decode = [&](const Letterbox& from) -> bool
		{
			const auto tDecode = Clock::now();
			const bool written = decodeOutputs(from);
			myDecodeMs = msSince(tDecode);
			return written;
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
			cudaError_t ce = launchPreprocess(inSurf, fmt, lb, inputNorm, inLayout, in->device, letterboxPad, myStream);
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
	if (mode == ModeRaw || mode == ModeCrops || mode == ModeLandmarkCrops || mode == ModeHolistic || !wrote)
	{
		myYolo.clear();
		myYoloMode = false;
	}
	myHolisticMode = mode == ModeHolistic && wrote;
	if (mode != ModeHolistic)
	{
		myHolistic.reset();	// the worker is idle (waitHolisticIdle above)
		myHoloShown.clear();
	}
	myLandmarkMode = mode == ModeLandmarkCrops && wrote;
	if (!myLandmarkMode)
		myLandmarks.clear();
	myCropMode = mode == ModeCrops && wrote;
	if (!myCropMode)
		myCrops.clear();
	if (!wrote && outSurf)
	{
		auto pf = outInfo.textureDesc.pixelFormat;
		int bpp = pf == OP_PixelFormat::Mono32Float ? 4 : pf == OP_PixelFormat::RGBA16Float ? 8 : 16;
		launchClear(outSurf, outW, outH, bpp, ioStream);
	}
	if (!extrasWritten)
		for (size_t i = 0; i < extras.size(); i++)
			if (extraSurfaces[i]->surf)
				launchClear(extraSurfaces[i]->surf, extras[i].w, extras[i].h, extras[i].opt.mono ? 4 : 16, myStream);
	}	// surfaces destroyed here, before TD unmaps the arrays

	myContext->endCUDAOperations(nullptr);
	myExecuteMs = msSince(tExec);
	average(myAvgMs[0], myInferMs);
	average(myAvgMs[1], myExecuteMs);
	average(myAvgMs[2], myRunMs);
	average(myAvgMs[3], myDecodeMs);
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

bool OnnxTOP::runHolistic(cudaSurfaceObject_t inSurf, TexFormat fmt, int srcW, int srcH, const HolisticOptions& opt,
	const std::vector<CropBox>& personBoxes, const std::vector<std::string>& personIds, bool timed,
	std::vector<CropBox>& sheet, std::string& err)
{
	using namespace holistic;
	sheet.clear();
	if (myCompanions.size() < 3)
	{
		err = "Holistic: the hand and face models are not loaded";
		return false;
	}
	// 0 pose, 1 hand ROI refinement, 2 hand landmarks, 3 face landmarks (Crop::part 0 / 1-2 recrop
	// or hand / 3 face map onto these per stage).
	OnnxEngine* engines[4] = { myEngine.get(), myCompanions[0].get(), myCompanions[1].get(), myCompanions[2].get() };
	const int minOutputs[4] = { 4, 1, 4, 2 };
	InputLayout layouts[4];
	for (int e = 0; e < 4; e++)
	{
		const OnnxEngine& en = *engines[e];
		const auto& sh = en.inputs().empty() ? std::vector<int64_t>() : en.inputs()[0].shape;
		bool ok = en.inputs().size() == 1 && sh.size() == 4 && sh[0] == 1 && (sh[1] == 3 || sh[3] == 3) &&
			tensorElem(en.inputs()[0].type, layouts[e].elem) && (int)en.outputs().size() >= minOutputs[e];
		for (const TensorSlot& o : en.outputs())
			ok &= o.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && o.device;
		if (!ok)
		{
			err = "Holistic: " + fs::path(en.config().modelPath).filename().string() +
				" is not one of the converted MediaPipe Holistic models (scripts/convert_holistic_onnx.py)";
			return false;
		}
		layouts[e].nhwc = isNHWC(sh);
	}
	const InputNorm norm = makeInputNorm(InputNormKind::Unit);	// MediaPipe's 0-1 float input
	if (!myDevCropBoxes)
		cudaMalloc(&myDevCropBoxes, kMaxBatch * sizeof(CropBox));

	auto toBox = [&](const Roi& r)
	{
		CropBox b{ (r.x - r.w * 0.5f) * srcW, (r.y - r.h * 0.5f) * srcH, (r.x + r.w * 0.5f) * srcW, (r.y + r.h * 0.5f) * srcH };
		b.cosA = std::cos(r.angle);
		b.sinA = std::sin(r.angle);
		b.replicate = 1;
		return b;
	};
	// Runs each (engine, crop) back to back on the stream and reads every output back into pinned
	// memory, then waits once. out[j][o] points at job j's output o.
	struct Job
	{
		int		engine;
		Crop	crop;
	};
	std::vector<std::vector<const float*>> out;
	auto runStage = [&](const std::vector<Job>& jobs) -> bool
	{
		out.clear();
		const int n = std::min((int)jobs.size(), kMaxBatch);
		if (n == 0)
			return true;
		size_t floats = 0;
		for (int j = 0; j < n; j++)
			for (const TensorSlot& o : engines[jobs[(size_t)j].engine]->outputs())
				floats += o.elementCount();
		if (floats * sizeof(float) > myHostStageBytes)
		{
			if (myHostStage)
				cudaFreeHost(myHostStage);
			if (cudaMallocHost(&myHostStage, floats * sizeof(float)) != cudaSuccess)
			{
				myHostStage = nullptr;
				myHostStageBytes = 0;
				err = "cudaMallocHost failed for the Holistic outputs";
				return false;
			}
			myHostStageBytes = floats * sizeof(float);
		}
		std::vector<CropBox> boxes;
		for (int j = 0; j < n; j++)
			boxes.push_back(toBox(jobs[(size_t)j].crop.roi));
		cudaMemcpyAsync(myDevCropBoxes, boxes.data(), n * sizeof(CropBox), cudaMemcpyHostToDevice, myStream);
		size_t at = 0;
		for (int j = 0; j < n; j++)
		{
			OnnxEngine& en = *engines[jobs[(size_t)j].engine];
			const TensorSlot& in = en.inputs()[0];
			cudaError_t ce = launchCropPreprocess(inSurf, fmt, srcW, srcH, myDevCropBoxes + j, 1, 1, norm,
				layouts[jobs[(size_t)j].engine], in.device, (int)in.shape[widthDim(in.shape)], (int)in.shape[heightDim(in.shape)], myStream);
			if (ce != cudaSuccess)
			{
				err = std::string("Crop kernel failed: ") + cudaGetErrorString(ce);
				return false;
			}
			if (!en.run(err, false))
				return false;
			std::vector<const float*> ptrs;
			for (const TensorSlot& o : en.outputs())
			{
				cudaMemcpyAsync(myHostStage + at, o.device, o.elementCount() * sizeof(float), cudaMemcpyDeviceToHost, myStream);
				ptrs.push_back(myHostStage + at);
				at += o.elementCount();
			}
			out.push_back(std::move(ptrs));
		}
		sheet.insert(sheet.end(), boxes.begin(), boxes.end());
		if (cudaStreamSynchronize(myStream) != cudaSuccess)
		{
			err = "Holistic: reading the model outputs failed";
			return false;
		}
		return true;
	};

	if (timed)
	{
		cudaEventRecord(myRunStart, myStream);
		myRunRecorded = true;
	}
	auto tRun = Clock::now();

	// Stage 1: pose, per person (tracked ROI, or a new detector box).
	std::vector<std::array<float, 4>> boxes;
	for (const CropBox& b : personBoxes)
		boxes.push_back({ b.x1, b.y1, b.x2, b.y2 });
	std::vector<Job> jobs;
	for (const Crop& c : myHolistic.poseCrops(srcW, srcH, boxes, personIds, std::clamp(opt.maxPeople, 1, 8)))
		jobs.push_back({ 0, c });
	if (!runStage(jobs))
		return false;
	for (size_t j = 0; j < out.size(); j++)
	{
		const auto& o = out[j];
		// Outputs: landmarks [195], presence [1], heatmap [64 x 64 x 39], world [117].
		myHolistic.decodePose(jobs[j].crop, o[0], o[1][0], engines[0]->outputs()[2].elementCount() == (size_t)kHeatmapValues ? o[2] : nullptr, o[3]);
	}
	myHolistic.endPose();

	// Stage 2: hand ROI refinement (from the pose's wrist / index / pinky) and the face mesh.
	jobs.clear();
	if (opt.hands)
		for (const Crop& c : myHolistic.handRecropCrops())
			jobs.push_back({ 1, c });
	if (opt.face)
		for (const Crop& c : myHolistic.faceCrops())
			jobs.push_back({ 3, c });
	if (!runStage(jobs))
		return false;
	std::vector<Job> handJobs;
	for (size_t j = 0; j < out.size(); j++)
	{
		if (jobs[j].engine == 1)
			handJobs.push_back({ 2, myHolistic.handCrop(jobs[j].crop, out[j][0]) });
		else
			myHolistic.decodeFace(jobs[j].crop, out[j][0], out[j][1][0]);	// landmarks [1434], presence logit
	}

	// Stage 3: hand landmarks.
	if (!runStage(handJobs))
		return false;
	for (size_t j = 0; j < out.size(); j++)
		myHolistic.decodeHand(handJobs[j].crop, out[j][0], out[j][1][0], out[j][3]);	// landmarks, presence, (handedness), world
	myHolistic.endFrame();

	if (timed)
	{
		myRunMs = msSince(tRun);
		cudaEventRecord(myRunDone, myStream);
	}
	else
		myHoloMs = msSince(tRun);
	return true;
}

void OnnxTOP::drawHolisticSheet(cudaSurfaceObject_t inSurf, TexFormat fmt, int srcW, int srcH, const std::vector<CropBox>& sheet,
	cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream)
{
	if (!myDevSheetBoxes)
		cudaMalloc(&myDevSheetBoxes, kMaxBatch * sizeof(CropBox));
	myHolisticCrops = std::min((int)sheet.size(), kMaxBatch);
	if (myHolisticCrops > 0)
		cudaMemcpyAsync(myDevSheetBoxes, sheet.data(), myHolisticCrops * sizeof(CropBox), cudaMemcpyHostToDevice, stream);
	const int cells = std::max(1, (outW / kCropSheetCell) * (outH / kCropSheetCell));
	launchCropSheet(inSurf, fmt, srcW, srcH, myDevSheetBoxes, std::min(myHolisticCrops, cells),
		std::max(1, outW / kCropSheetCell), kCropSheetCell, outSurf, outW, outH, stream);
}

// Background Holistic (Wait for Result off): the worker runs the whole chain on a private copy of the
// frame while TouchDesigner cooks on; each cook publishes the last finished result and hands the
// worker the newest frame if it's free (otherwise that frame is dropped, as MediaPipe does).
void OnnxTOP::holisticWorker()
{
	for (;;)
	{
		HoloJob job;
		{
			std::unique_lock<std::mutex> lock(myHoloMutex);
			myHoloCv.wait(lock, [&] { return myHoloQuit || myHoloHasJob; });
			if (myHoloQuit)
				return;
			job = myHoloJob;
			myHoloHasJob = false;
		}
		cudaStreamWaitEvent(myStream, myHoloCopied, 0);	// the cook's copy of the frame (myIoStream)
		cudaResourceDesc desc = {};
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = myHoloFrame;
		cudaSurfaceObject_t surf = 0;
		std::string err;
		std::vector<CropBox> sheet;
		bool ok = cudaCreateSurfaceObject(&surf, &desc) == cudaSuccess;
		if (!ok)
			err = "Holistic: could not read the copied frame";
		else
			ok = runHolistic(surf, job.fmt, job.w, job.h, job.opt, job.boxes, job.ids, false, sheet, err);
		if (surf)
			cudaDestroySurfaceObject(surf);
		if (!ok)
			myHolistic.reset();
		{
			std::lock_guard<std::mutex> lock(myHoloMutex);
			myHoloPublished = myHolistic.people();
			myHoloPublishedSheet = std::move(sheet);
			myHoloPublishedError = ok ? std::string() : err;
			myHoloPublishedFrame = job.frame;
			myHoloHasResult = true;
		}
		myHoloBusy = false;
	}
}

void OnnxTOP::waitHolisticIdle()
{
	while (myHoloBusy)
		std::this_thread::sleep_for(std::chrono::microseconds(200));
}

void OnnxTOP::stopHolisticWorker()
{
	if (!myHoloThread.joinable())
		return;
	{
		std::lock_guard<std::mutex> lock(myHoloMutex);
		myHoloQuit = true;
	}
	myHoloCv.notify_all();
	myHoloThread.join();
	myHoloBusy = false;
}

bool OnnxTOP::runLandmarkCrops(OnnxEngine& engine, int start, int count, std::string& err)
{
	// Which outputs are small enough to report (fixed per engine, so the header stays put).
	const auto& outs = engine.outputs();
	const int B = (int)std::max<int64_t>(1, engine.inputs()[0].shape[0]);
	std::vector<std::pair<int, int>> picked;
	size_t perCrop = 0;
	for (int o = 0; o < (int)outs.size(); o++)
	{
		const TensorSlot& s = outs[o];
		int64_t total = 1;
		for (int64_t d : s.shape)
			total *= std::max<int64_t>(d, 1);
		const bool isFloat = s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
		if (!isFloat || !s.device || total % B != 0 || total / B > kLandmarkMaxValues)
			continue;
		picked.emplace_back(o, (int)(total / B));
		perCrop += (size_t)(total / B);
	}
	myLmOutputs = picked;
	size_t bytes = perCrop * (size_t)B * sizeof(float);
	if (bytes > myHostLogitsBytes)
	{
		if (myHostLogits)
			cudaFreeHost(myHostLogits);
		if (cudaMallocHost(&myHostLogits, bytes) != cudaSuccess)
		{
			myHostLogits = nullptr;
			myHostLogitsBytes = 0;
			err = "cudaMallocHost failed for the landmark outputs";
			return false;
		}
		myHostLogitsBytes = bytes;
	}
	// Copy every picked output (all B items) back, packed one after another.
	size_t at = 0;
	for (const auto& [o, per] : picked)
	{
		const TensorSlot& s = outs[o];
		const size_t vals = (size_t)per * B;
		if (s.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
		{
			// Half: read the raw bits into the tail of the buffer region, widen after the sync.
			cudaMemcpyAsync(reinterpret_cast<char*>(myHostLogits + at) + vals * 2, s.device, vals * 2, cudaMemcpyDeviceToHost, myStream);
		}
		else
			cudaMemcpyAsync(myHostLogits + at, s.device, vals * sizeof(float), cudaMemcpyDeviceToHost, myStream);
		at += vals;
	}
	if (cudaStreamSynchronize(myStream) != cudaSuccess)
	{
		err = "Reading the landmark outputs failed";
		return false;
	}
	at = 0;
	for (const auto& [o, per] : picked)
	{
		const size_t vals = (size_t)per * B;
		if (outs[o].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
		{
			const uint16_t* h = reinterpret_cast<const uint16_t*>(reinterpret_cast<char*>(myHostLogits + at) + vals * 2);
			for (size_t i = 0; i < vals; i++)
			{
				// IEEE half -> float, in place front to back: float i ends at byte 4i + 4, at or before
				// half i + 1 (byte 2 vals + 2i + 2), so no half is overwritten before it is read.
				uint16_t v = h[i];
				uint32_t sign = (uint32_t)(v & 0x8000) << 16, exp = (v >> 10) & 0x1f, man = v & 0x3ff, bits;
				if (exp == 0)
				{
					if (man == 0)
						bits = sign;
					else
					{
						exp = 127 - 15 + 1;
						while (!(man & 0x400))
						{
							man <<= 1;
							exp--;
						}
						bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
					}
				}
				else if (exp == 31)
					bits = sign | 0x7f800000 | (man << 13);
				else
					bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
				memcpy(myHostLogits + at + i, &bits, 4);
			}
		}
		at += vals;
	}
	for (int k = 0; k < count; k++)
	{
		LandmarkRow row;
		row.id = myCropIds[(size_t)(start + k)];
		row.roi = myRois[(size_t)(start + k)];
		row.values.reserve(perCrop);
		size_t base = 0;
		for (const auto& [o, per] : picked)
		{
			const float* p = myHostLogits + base + (size_t)k * per;
			row.values.insert(row.values.end(), p, p + per);
			base += (size_t)per * B;
		}
		myLandmarks.push_back(std::move(row));
	}
	return true;
}

// ---------------------------------------------------------------- info outputs

int32_t OnnxTOP::getNumInfoCHOPChans(void*)
{
	return 16;
}

void OnnxTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
	LoadState st = myLoadState.load();
	switch (index)
	{
		case 0: chan->name->setString("loaded");			chan->value = myEngine ? 1.f : 0.f; break;
		case 1: chan->name->setString("loading");			chan->value = st == LoadState::Loading ? 1.f : 0.f; break;
		case 2: chan->name->setString("tensorrt");			chan->value = myEngine && myEngine->providerDescription().rfind("TensorRT", 0) == 0 ? 1.f : 0.f; break;
		case 3: chan->name->setString("num_detections");	chan->value = (float)(myHolisticMode ? myHoloShown.size() : myLandmarkMode ? myLandmarks.size()
			: myCropMode ? myCrops.size() : myYolo.detections().size()); break;
		case 4: chan->name->setString("infer_ms");			chan->value = (float)myInferMs; break;
		case 5: chan->name->setString("execute_ms");		chan->value = (float)myExecuteMs; break;
		case 6: chan->name->setString("model_w");			chan->value = (float)myModelW; break;
		case 7: chan->name->setString("model_h");			chan->value = (float)myModelH; break;
		case 8: chan->name->setString("trt_building");		chan->value = myUpgrading ? 1.f : 0.f; break;
		// Input frames since the one the current results are from (0 with Wait for Result on).
		case 10: chan->name->setString("run_ms");			chan->value = (float)myRunMs; break;
		case 9: chan->name->setString("result_delay");		chan->value = myResultFrame >= 0 && myInputFrame >= myResultFrame ?
			(float)(myInputFrame - myResultFrame) : 0.f; break;
		// Reading back and decoding the model's outputs (YOLO, MoveNet, YuNet, Raw), and running averages.
		case 11: chan->name->setString("decode_ms");		chan->value = (float)myDecodeMs; break;
		case 12: chan->name->setString("infer_ms_avg");		chan->value = (float)std::max(myAvgMs[0], 0.0); break;
		case 13: chan->name->setString("execute_ms_avg");	chan->value = (float)std::max(myAvgMs[1], 0.0); break;
		case 14: chan->name->setString("run_ms_avg");		chan->value = (float)std::max(myAvgMs[2], 0.0); break;
		case 15: chan->name->setString("decode_ms_avg");	chan->value = (float)std::max(myAvgMs[3], 0.0); break;
	}
}

bool OnnxTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void*)
{
	if (myHolisticMode)
	{
		infoSize->rows = 1 + (int32_t)myHoloShown.size();
		infoSize->cols = 8 + holistic::kTotal * 4 + holistic::kWorld * 3;
		infoSize->byColumn = false;
		return true;
	}
	if (myLandmarkMode)
	{
		int cols = 6;
		for (const auto& p : myLmOutputs)
			cols += p.second;
		infoSize->rows = 1 + (int32_t)myLandmarks.size();
		infoSize->cols = cols;
		infoSize->byColumn = false;
		return true;
	}
	if (myCropMode)
	{
		infoSize->rows = 1 + (int32_t)myCrops.size();
		infoSize->cols = 3 + myCropClasses;
		infoSize->byColumn = false;
		return true;
	}
	infoSize->rows = 1 + (int32_t)myYolo.detections().size();
	infoSize->cols = infoBaseCols() + myYolo.keypointCount() * 3 + myYolo.keypointZCount() + myYolo.poseExtraCols();
	infoSize->byColumn = false;
	return true;
}

void OnnxTOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	// Columns: id, class, conf, u1, v1, u2, v2, [mask_fill (seg)], [kp{i}_u/v/c (pose)]
	char buf[64];
	if (myHolisticMode)
	{
		// MediaPipeTOP's Holistic layout: id, class, conf, u1, v1, u2, v2, label, kp{i}_u/v/c (553:
		// pose 0-32, left hand 33-53, right hand 54-74, face 75-552; TD UV), kp{i}_z, w{i}_x/y/z (75,
		// metres, hip origin, MediaPipe axes). Parts not found are 0.
		using namespace holistic;
		const int kp0 = 8, z0 = kp0 + kTotal * 3, w0 = z0 + kTotal;
		if (index == 0)
		{
			const char* base[] = { "id", "class", "conf", "u1", "v1", "u2", "v2", "label" };
			const char* uvc[] = { "u", "v", "c" };
			const char* xyz[] = { "x", "y", "z" };
			for (int c = 0; c < nEntries; c++)
			{
				if (c < kp0)
					snprintf(buf, sizeof(buf), "%s", base[c]);
				else if (c < z0)
					snprintf(buf, sizeof(buf), "kp%d_%s", (c - kp0) / 3, uvc[(c - kp0) % 3]);
				else if (c < w0)
					snprintf(buf, sizeof(buf), "kp%d_z", c - z0);
				else
					snprintf(buf, sizeof(buf), "w%d_%s", (c - w0) / 3, xyz[(c - w0) % 3]);
				entries->values[c]->setString(buf);
			}
			return;
		}
		const auto& people = myHoloShown;
		if (index < 1 || index - 1 >= (int32_t)people.size())
		{
			for (int c = 0; c < nEntries; c++)
				entries->values[c]->setString("0");
			return;
		}
		const Person& p = people[(size_t)(index - 1)];
		for (int c = 0; c < nEntries; c++)
		{
			float v = 0.f;
			if (c == 0)
			{
				entries->values[c]->setString(p.id.c_str());
				continue;
			}
			if (c == 7)
			{
				entries->values[c]->setString("");
				continue;
			}
			if (c == 1)
				v = 0.f;
			else if (c == 2)
				v = p.score;
			else if (c < kp0)
				v = std::clamp(p.box[c - 3], 0.f, 1.f);
			else if (c < z0)
			{
				const Landmark& l = p.lm[(size_t)((c - kp0) / 3)];
				int f = (c - kp0) % 3;
				v = l.c <= 0.f ? 0.f : f == 0 ? l.x : f == 1 ? 1.f - l.y : l.c;
			}
			else if (c < w0)
				v = p.lm[(size_t)(c - z0)].z;
			else
				v = p.world[(size_t)((c - w0) / 3)][(size_t)((c - w0) % 3)];
			snprintf(buf, sizeof(buf), "%.5g", v);
			entries->values[c]->setString(buf);
		}
		return;
	}
	if (myLandmarkMode)
	{
		// Columns: id, roi_x, roi_y, roi_w, roi_h, roi_angle (the Crop DAT's), then o{k}_{i}: value i
		// of output k for this crop, raw (decoding is the consumer's, e.g. holistic_onnx.py).
		if (index == 0)
		{
			int c = 0;
			auto put = [&](const char* s) { if (c < nEntries) entries->values[c++]->setString(s); };
			put("id");
			for (const char* r : kRoiCols)
				put(r);
			for (const auto& [o, per] : myLmOutputs)
				for (int i = 0; i < per && c < nEntries; i++)
				{
					snprintf(buf, sizeof(buf), "o%d_%d", o, i);
					put(buf);
				}
			while (c < nEntries)
				put("");
			return;
		}
		const bool valid = index >= 1 && index - 1 < (int32_t)myLandmarks.size();
		for (int c = 0; c < nEntries; c++)
		{
			if (!valid)
			{
				entries->values[c]->setString("0");
				continue;
			}
			const LandmarkRow& r = myLandmarks[index - 1];
			if (c == 0)
			{
				entries->values[c]->setString(r.id.c_str());
				continue;
			}
			float v = c < 6 ? r.roi[c - 1] : (c - 6 < (int)r.values.size() ? r.values[c - 6] : 0.f);
			snprintf(buf, sizeof(buf), "%.6g", v);
			entries->values[c]->setString(buf);
		}
		return;
	}
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
		const int kp3 = myYolo.keypointCount() * 3;
		float v = c < base ? vals[c] : c < base + kp3 ? (c - base < (int)d.keypoints.size() ? d.keypoints[c - base] : 0.f)
			: c < base + kp3 + myYolo.keypointZCount() ? (c - base - kp3 < (int)d.keypointsZ.size() ? d.keypointsZ[c - base - kp3] : 0.f)
			: (c - base - kp3 - myYolo.keypointZCount() == 0 ? d.cropW : d.cropH);
		formatValue(buf, sizeof(buf), v, c < 2);
		entries->values[c]->setString(buf);
	}
}

void OnnxTOP::headerRow(int cols)
{
	const int base = infoBaseCols();
	const int kp3 = myYolo.keypointCount() * 3;
	const std::string key = std::to_string(base) + "/" + std::to_string(cols) + "/" + std::to_string(kp3);
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
		else if (c < base + kp3)
		{
			snprintf(buf, sizeof(buf), "kp%d_%s", (c - base) / 3, suffix[(c - base) % 3]);
			myHeader[(size_t)c] = buf;
		}
		else if (c < base + kp3 + myYolo.keypointZCount())
		{
			snprintf(buf, sizeof(buf), "kp%d_z", c - base - kp3);
			myHeader[(size_t)c] = buf;
		}
		else
			myHeader[(size_t)c] = c - base - kp3 - myYolo.keypointZCount() == 0 ? "crop_w" : "crop_h";
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
		OP_NumericParameter np("Scalarinput");
		np.label = "Scalar Input";
		np.page = "ONNX";
		np.defaultValues[0] = 0.25;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = np.maxSliders[0] = 1.0;
		np.clampMins[0] = true;
		np.help = "Value for a model's extra single-number input (RVM's downsample_ratio: the model works on the frame "
			"scaled by this, e.g. 0.25 for HD, 0.5 for SD). Changing it reloads the model.";
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Resetstate");
		np.label = "Reset State";
		np.page = "ONNX";
		np.help = "For a recurrent model (RVM): forget the previous frames, e.g. after a cut.";
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
			"freeing TouchDesigner's main thread for the inference time. Raw output never waits either way. "
			"MediaPipe Holistic: Off runs the whole chain on a background thread on a copy of the frame; each cook "
			"shows the last finished result and frames that arrive while it's busy are skipped.";
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Mode");
		sp.label = "Output Mode";
		sp.page = "ONNX";
		sp.defaultValue = kModeNames[0];
		sp.help = "YOLO: Ultralytics end2end det / seg / pose. MoveNet: multipose [1, 6, 56] (set Input Normalize "
			"0 to 255 and Dynamic Size, e.g. 256 x 256). YuNet: OpenCV Zoo face detector (set Input Normalize "
			"0 to 255 and Channel Order BGR). Raw: show an output tensor as an image (e.g. Depth Anything). "
			"Pose Crops: person boxes from the Crop DAT, cropped to the model's aspect, run through an RTMPose / RTMW / RTMW3D "
			"SimCC model (Input Normalize ImageNet); keypoints (and depth for RTMW3D) go to the Info DAT. "
			"Landmark Crops: rotated ROIs from the Crop DAT (MediaPipe NormalizedRects: roi_x, roi_y, roi_w, roi_h, "
			"roi_angle) run through a landmark model (e.g. MediaPipe's, converted); every small output goes to the "
			"Info DAT raw, one row per ROI, for Python to decode. "
			"MediaPipe Holistic: Model = the converted pose_landmarks_detector.onnx (the hand and face models load from "
			"the same folder); people from the Crop DAT (a detector's tracks), then tracked from their own landmarks; "
			"the Info DAT matches MediaPipeTOP's Holistic table. Max Detections = people.";
		check(manager->appendMenu(sp, 8, kModeNames, kModeLabels));
	}
	{
		OP_NumericParameter np("Holistichands");
		np.label = "Hands";
		np.page = "Holistic";
		np.defaultValues[0] = 1.0;
		np.help = "Holistic: run the hand models (two per person). Off saves about 2 ms per person.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Holisticface");
		np.label = "Face Mesh";
		np.page = "Holistic";
		np.defaultValues[0] = 1.0;
		np.help = "Holistic: run the face mesh (478 points per person).";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Letterbox");
		np.label = "Letterbox (Keep Aspect)";
		np.page = "ONNX";
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		const char* names[] = { "gray", "edge" };
		const char* labels[] = { "Gray (Ultralytics)", "Edge Pixels (Repeat)" };
		OP_StringParameter sp("Letterboxpad");
		sp.label = "Letterbox Padding";
		sp.page = "ONNX";
		sp.defaultValue = names[0];
		sp.help = "What fills the letterbox bars. Gray: YOLO's training padding. Edge Pixels: the image's edge "
			"repeated (cv2.BORDER_REPLICATE), what DAViD's own code feeds; gray bars leave artifacts along its edges.";
		check(manager->appendMenu(sp, 2, names, labels));
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
		OP_NumericParameter np("Rawalloutputs");
		np.label = "All Outputs (Color Buffers)";
		np.page = "Raw";
		np.defaultValues[0] = 0.0;
		np.help = "Also write every other image-shaped output, raw values (no normalize), to color buffers 1, 2, ... "
			"in output order; read them with a Render Select TOP. One inference for a multi-output model "
			"(DAViD's multi-task depth / normals / matte).";
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
			"OnnxTopTables' script_tracks) or an OnnxTOP Info DAT (id, u1, v1, u2, v2). Landmark Crops: ROIs "
			"(id, roi_x, roi_y, roi_w, roi_h, roi_angle; MediaPipe NormalizedRects, top-down). The node's input must be "
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
	else if (!strcmp(name, "Resetstate"))
		myResetState = true;
}
