#include "MediaPipeTOP.h"

#include "Kernels.h"
#include "MpApi.h"
#include "MpKernels.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
	using Clock = std::chrono::high_resolution_clock;

	double msSince(Clock::time_point t0)
	{
		return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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

	// Per-frame surface object (never cache across frames -- see OnnxTOP.cpp's FrameSurface).
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

	// Menu values are saved by name.
	constexpr int kNumTasks = 7;
	const char* kTaskNames[] = { "face", "hands", "gesture", "pose", "holistic", "facedetector", "segment" };
	const char* kTaskLabels[] = { "Face Landmarker (478 points, blendshapes)", "Hand Landmarker",
		"Gesture Recognizer (hands + gestures)", "Pose Landmarker (33 points)",
		"Holistic Landmarker (pose + hands + face)", "Face Detector (boxes + 6 points)", "Image Segmenter (mask)" };
	const char* kModelPars[] = { "Facemodel", "Handmodel", "Gesturemodel", "Posemodel", "Holisticmodel",
		"Facedetmodel", "Segmentmodel" };
	const int kLandmarkCount[] = { 478, 21, 21, 33, kHolisticCount, 6, 0 };
	const char* kSegOutputNames[] = { "foreground", "class", "category" };
	const char* kSegOutputLabels[] = { "Foreground (1 - background)", "One Class (Segment Class)", "Category Index" };

	bool hasWorld(MpTaskKind k)
	{
		return k == MpTaskKind::Hands || k == MpTaskKind::Gesture || k == MpTaskKind::Pose || k == MpTaskKind::Holistic;
	}

	constexpr int kMaxInstances = 8;
	constexpr int kBaseCols = 8;	// id, class, conf, u1, v1, u2, v2, label
}

extern "C"
{
DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
	if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
		return;
	info->executeMode = TOP_ExecuteMode::CUDA;
	info->customOPInfo.opType->setString("Mediapipe");
	info->customOPInfo.opLabel->setString("MediaPipe");
	info->customOPInfo.opIcon->setString("MP");
	info->customOPInfo.authorName->setString("td-native-inference");
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	return new MediaPipeTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
	delete static_cast<MediaPipeTOP*>(instance);
}
}

MediaPipeTOP::MediaPipeTOP(const OP_NodeInfo*, TOP_Context* context) :
	myContext(context)
{
	cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
	cudaEventCreateWithFlags(&myReadEvent, cudaEventDisableTiming);
}

MediaPipeTOP::~MediaPipeTOP()
{
	myWorker.reset();	// joins the worker (finishes the frame in flight, closes the task)
	if (myStream)
		cudaStreamSynchronize(myStream);
	if (myReadEvent)
		cudaEventDestroy(myReadEvent);
	if (myDevRgb)
		cudaFree(myDevRgb);
	if (myHostRgb)
		cudaFreeHost(myHostRgb);
	if (myDevMask)
		cudaFree(myDevMask);
	if (myStream)
		cudaStreamDestroy(myStream);
}

void MediaPipeTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = true;
	// Keep cooking while a frame or a result is still on its way, so it lands even when the input
	// is static.
	ginfo->cookEveryFrame = myWorker && myActive &&
		(myWorker->state() == MpWorker::State::Loading || myWorker->pending() || myReadInFlight);
}

bool MediaPipeTOP::ensureBuffers(int w, int h)
{
	size_t bytes = (size_t)w * h * 3;
	if (bytes <= myRgbBytes)
		return true;
	if (myReadInFlight)	// a readback still targets the old host buffer
	{
		cudaEventSynchronize(myReadEvent);
		myReadInFlight = false;
	}
	if (myDevRgb)
		cudaFree(myDevRgb);
	if (myHostRgb)
		cudaFreeHost(myHostRgb);
	myDevRgb = nullptr;
	myHostRgb = nullptr;
	myRgbBytes = 0;
	if (cudaMalloc(&myDevRgb, bytes) != cudaSuccess || cudaMallocHost(&myHostRgb, bytes) != cudaSuccess)
		return false;
	myRgbBytes = bytes;
	return true;
}

void MediaPipeTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
	auto tExec = Clock::now();
	myError.clear();
	myWarning.clear();

	// --- read every parameter/input BEFORE beginCUDAOperations() (TD requirement) ---
	int task = std::clamp(inputs->getParInt("Task"), 0, kNumTasks - 1);
	const MpTaskKind kind = (MpTaskKind)task;
	const bool isFace = kind == MpTaskKind::Face, isHolistic = kind == MpTaskKind::Holistic;
	const bool isSeg = kind == MpTaskKind::Segment, isFaceDet = kind == MpTaskKind::FaceDetector;
	const bool tracks = kind == MpTaskKind::Face || kind == MpTaskKind::Hands || kind == MpTaskKind::Gesture ||
		kind == MpTaskKind::Pose;	// tasks with a tracking threshold and a max-instances option
	MpTaskConfig cfg;
	cfg.kind = kind;
	const char* model = inputs->getParFilePath(kModelPars[task]);
	cfg.modelPath = mpapi::resolveModelPath(model, inputs->getParString(kModelPars[task]), myModelHint);
	cfg.maxInstances = std::clamp(inputs->getParInt("Maxinstances"), 1, kMaxInstances);
	cfg.minDetection = (float)inputs->getParDouble("Mindetection");
	cfg.minPresence = (float)inputs->getParDouble("Minpresence");
	cfg.minTracking = (float)inputs->getParDouble("Mintracking");
	cfg.blendshapes = (isFace || isHolistic) && inputs->getParInt("Blendshapes") != 0;
	cfg.matrix = isFace && inputs->getParInt("Matrix") != 0;
	cfg.segmentation = (kind == MpTaskKind::Pose || isHolistic) && inputs->getParInt("Segmentation") != 0;
	myActive = inputs->getParInt("Active") != 0;
	int detectWidth = inputs->getParInt("Detectwidth");
	myOutputZ = !isSeg && !isFaceDet && inputs->getParInt("Outputz") != 0;
	myOutputWorld = hasWorld(kind) && inputs->getParInt("Outputworld") != 0;
	const bool waitResult = inputs->getParInt("Waitforresult") != 0;
	const double maxWaitMs = std::max(0.0, inputs->getParDouble("Maxwait"));
	MpSegOutput segOutput = (MpSegOutput)std::clamp(inputs->getParInt("Segoutput"), 0, 2);
	int segClass = std::max(0, inputs->getParInt("Segclass"));

	for (int t = 0; t < kNumTasks; t++)
		inputs->enablePar(kModelPars[t], t == task);
	inputs->enablePar("Maxinstances", tracks || isFaceDet);
	inputs->enablePar("Mindetection", !isSeg);
	inputs->enablePar("Minpresence", tracks || isHolistic);
	inputs->enablePar("Mintracking", tracks);
	inputs->enablePar("Blendshapes", isFace || isHolistic);
	inputs->enablePar("Matrix", isFace);
	inputs->enablePar("Segmentation", kind == MpTaskKind::Pose || isHolistic);
	inputs->enablePar("Outputz", !isSeg && !isFaceDet);
	inputs->enablePar("Outputworld", hasWorld(kind));
	inputs->enablePar("Maxwait", waitResult);
	inputs->enablePar("Segoutput", isSeg);
	inputs->enablePar("Segclass", isSeg && segOutput == MpSegOutput::Class);

	// --- (re)start the worker when the task setup changes; it loads the model on its own thread ---
	if (cfg.kind != myTask)
		myResult = MpFrameResult();
	myTask = cfg.kind;
	myModelPath = cfg.modelPath;
	if (cfg.modelPath.empty())
	{
		myWorker.reset();
		myResult = MpFrameResult();
		myWarning = std::string("Set the ") + kModelPars[task] + " parameter to a .task file";
	}
	else if (!myWorker || myWorker->config() != cfg || myReload)
	{
		myWorker.reset();
		myWorker = std::make_unique<MpWorker>(cfg);
		myResult = MpFrameResult();
		myLastInputCooks = -1;
	}
	myReload = false;

	if (myWorker)
	{
		myWorker->setSegOutput(segOutput, segClass);
		switch (myWorker->state())
		{
			case MpWorker::State::Loading:	myWarning = "Loading " + cfg.modelPath + "..."; break;
			case MpWorker::State::Failed:	myError = myWorker->loadError(); break;
			case MpWorker::State::Ready:
				myWorker->latest(myResult);
				if (!myResult.error.empty())
					myWarning = "MediaPipe: " + myResult.error;
				break;
		}
	}
	if (!myActive)
		myResult.instances.clear();

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
	int dw = detectWidth > 0 ? std::min(detectWidth, srcW) : srcW;
	int dh = std::max(1, (int)std::lround((double)srcH * dw / srcW));
	myFrameW = dw;
	myFrameH = dh;
	if (!ensureBuffers(dw, dh))
	{
		myError = "Out of GPU / pinned memory for the " + std::to_string(dw) + "x" + std::to_string(dh) + " frame";
		return;
	}

	bool ready = myWorker && myActive && myWorker->state() == MpWorker::State::Ready;
	bool newFrame = top->totalCooks != myLastInputCooks;
	bool submit = ready && newFrame;

	// --- output: the frame MediaPipe sees, or a mask (segmenter, or pose / holistic segmentation) ---
	bool maskOut = cfg.segmentation || isSeg;
	TOP_CUDAOutputInfo outInfo;
	outInfo.stream = myStream;
	output->getSuggestedOutputDesc(&outInfo.textureDesc, nullptr);
	outInfo.textureDesc.texDim = OP_TexDim::e2D;
	outInfo.textureDesc.depth = 1;
	outInfo.textureDesc.width = maskOut && myResult.maskW > 0 ? myResult.maskW : dw;
	outInfo.textureDesc.height = maskOut && myResult.maskH > 0 ? myResult.maskH : dh;
	outInfo.textureDesc.pixelFormat = maskOut ? OP_PixelFormat::Mono32Float : OP_PixelFormat::RGBA8Fixed;

	const OP_CUDAArrayInfo* outArray = output->createCUDAArray(outInfo, nullptr);
	if (!outArray)
		return;
	OP_CUDAAcquireInfo acquire;
	acquire.stream = myStream;
	const OP_CUDAArrayInfo* inArray = top->getCUDAArray(acquire, nullptr);

	int64_t waitFrame = -1;	// Wait for Result: the frame submitted this cook
	if (!myContext->beginCUDAOperations(nullptr))
		return;
	{
		FrameSurface outSurface(outArray->cudaArray);
		FrameSurface inSurface(inArray ? inArray->cudaArray : nullptr);
		const int outW = (int)outInfo.textureDesc.width, outH = (int)outInfo.textureDesc.height;

		bool haveFrame = false;
		if (inSurface.surf)
		{
			// Bilinear resize to dw x dh, 0-255 RGB, top-down (offset 0.5 rounds instead of truncating).
			Letterbox lb = computeLetterbox(srcW, srcH, dw, dh, false);
			InputNorm norm;
			for (int c = 0; c < 3; c++)
			{
				norm.scale[c] = 255.f;
				norm.offset[c] = 0.5f;
			}
			InputLayout layout;
			layout.elem = TensorElem::U8;
			layout.nhwc = true;
			layout.bgr = false;
			cudaError_t ce = launchPreprocess(inSurface.surf, fmt, lb, norm, layout, myDevRgb, 0.f, myStream);
			if (ce != cudaSuccess)
				myError = std::string("Resize kernel failed: ") + cudaGetErrorString(ce);
			else
				haveFrame = true;
		}

		if (waitResult)
		{
			// This frame goes to MediaPipe now (the copy back waits for the GPU), and the cook waits
			// for its result below.
			if (myReadInFlight)	// left over from the asynchronous mode: drop it
			{
				cudaEventSynchronize(myReadEvent);
				myReadInFlight = false;
			}
			if (haveFrame && submit)
			{
				size_t bytes = (size_t)dw * dh * 3;
				cudaMemcpyAsync(myHostRgb, myDevRgb, bytes, cudaMemcpyDeviceToHost, myStream);
				if (cudaStreamSynchronize(myStream) == cudaSuccess)
				{
					mySubmitBuf.resize(bytes);
					memcpy(mySubmitBuf.data(), myHostRgb, bytes);
					myWorker->submit(mySubmitBuf, dw, dh, top->totalCooks);
					myLastInputCooks = top->totalCooks;
					waitFrame = top->totalCooks;
				}
			}
		}
		// The frame read back on an earlier cook goes to MediaPipe once its copy has landed.
		else if (myReadInFlight && cudaEventQuery(myReadEvent) == cudaSuccess)
		{
			myReadInFlight = false;
			if (ready)
			{
				size_t bytes = (size_t)myReadW * myReadH * 3;
				mySubmitBuf.resize(bytes);
				memcpy(mySubmitBuf.data(), myHostRgb, bytes);
				myWorker->submit(mySubmitBuf, myReadW, myReadH, myReadFrame);
			}
		}
		// Queue this frame's readback (at most one in flight: MediaPipe takes one frame at a
		// time anyway, so a newer frame while one is pending would be dropped by the worker).
		if (!waitResult && haveFrame && submit && !myReadInFlight)
		{
			size_t bytes = (size_t)dw * dh * 3;
			if (cudaMemcpyAsync(myHostRgb, myDevRgb, bytes, cudaMemcpyDeviceToHost, myStream) == cudaSuccess &&
				cudaEventRecord(myReadEvent, myStream) == cudaSuccess)
			{
				myReadInFlight = true;
				myReadW = dw;
				myReadH = dh;
				myReadFrame = top->totalCooks;
				myLastInputCooks = top->totalCooks;
			}
		}

		if (outSurface.surf)
		{
			if (!maskOut && haveFrame)
				launchRgbToSurface(myDevRgb, outW, outH, outSurface.surf, myStream);
			else if (maskOut && myActive && !myResult.mask.empty() && myResult.maskW == outW && myResult.maskH == outH)
			{
				size_t bytes = myResult.mask.size() * sizeof(float);
				if (bytes > myMaskBytes)
				{
					if (myDevMask)
						cudaFree(myDevMask);
					myDevMask = nullptr;
					myMaskBytes = cudaMalloc(&myDevMask, bytes) == cudaSuccess ? bytes : 0;
				}
				if (myDevMask)
				{
					cudaMemcpyAsync(myDevMask, myResult.mask.data(), bytes, cudaMemcpyHostToDevice, myStream);
					launchMaskToSurface(myDevMask, outW, outH, outSurface.surf, myStream);
				}
			}
			else
				launchClear(outSurface.surf, outW, outH, 4, myStream);	// RGBA8 and Mono32Float are both 4 bytes/px
		}
	}	// surfaces destroyed before TD unmaps the arrays
	myContext->endCUDAOperations(nullptr);
	// Outside the CUDA block, so TD's context isn't held while waiting. A fast model's result lands
	// within Max Wait and matches this frame (result_delay 0); a slow one still arrives later.
	if (waitFrame >= 0 && myWorker->waitFor(waitFrame, maxWaitMs))
		myWorker->latest(myResult);
	myExecuteMs = msSince(tExec);
}

// ---------------------------------------------------------------- info CHOP / DAT

int32_t MediaPipeTOP::getNumInfoCHOPChans(void*)
{
	return 10;
}

void MediaPipeTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
	MpWorker::State st = myWorker ? myWorker->state() : MpWorker::State::Failed;
	switch (index)
	{
		case 0: chan->name->setString("loaded");			chan->value = myWorker && st == MpWorker::State::Ready ? 1.f : 0.f; break;
		case 1: chan->name->setString("loading");			chan->value = myWorker && st == MpWorker::State::Loading ? 1.f : 0.f; break;
		case 2: chan->name->setString("num_detections");	chan->value = (float)myResult.instances.size(); break;
		case 3: chan->name->setString("infer_ms");			chan->value = (float)myResult.inferMs; break;
		case 4: chan->name->setString("execute_ms");		chan->value = (float)myExecuteMs; break;
		case 5: chan->name->setString("frame_w");			chan->value = (float)myFrameW; break;
		case 6: chan->name->setString("frame_h");			chan->value = (float)myFrameH; break;
		case 7: chan->name->setString("results");			chan->value = (float)myResult.seq; break;
		case 8: chan->name->setString("dropped_frames");	chan->value = myWorker ? (float)myWorker->droppedFrames() : 0.f; break;
		// Input frames since the one the current result was computed from: a Cache TOP on the input
		// with Output Index -result_delay shows the frame that matches the landmarks.
		case 9: chan->name->setString("result_delay");		chan->value = myResult.inputFrame >= 0 && myInputFrame >= myResult.inputFrame ?
			(float)(myInputFrame - myResult.inputFrame) : 0.f; break;
	}
}

MediaPipeTOP::Columns MediaPipeTOP::columns() const
{
	Columns c;
	c.landmarks = kLandmarkCount[(int)myTask];
	c.kp = kBaseCols;
	int at = kBaseCols + c.landmarks * 3;
	if (myOutputZ)
	{
		c.z = at;
		at += c.landmarks;
	}
	if (myOutputWorld)
	{
		// Holistic's face mesh has no world landmarks: only the body and hands get columns.
		c.worldCount = myTask == MpTaskKind::Holistic ? kHolisticFace : c.landmarks;
		c.world = at;
		at += c.worldCount * 3;
	}
	if ((myTask == MpTaskKind::Face || myTask == MpTaskKind::Holistic) && !myResult.scoreNames.empty())
	{
		c.scores = at;
		at += (int)myResult.scoreNames.size();
	}
	bool hasMatrix = false;
	for (const MpInstance& inst : myResult.instances)
		hasMatrix |= !inst.matrix.empty();
	if (hasMatrix)
	{
		c.matrix = at;
		at += 16;
	}
	c.total = at;
	return c;
}

bool MediaPipeTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void*)
{
	infoSize->rows = 1 + (int32_t)myResult.instances.size();
	infoSize->cols = columns().total;
	infoSize->byColumn = false;
	return true;
}

void MediaPipeTOP::headerRow(const Columns& c)
{
	char key[160];
	snprintf(key, sizeof(key), "%d/%d/%d/%d/%d/%d/%d/%zu/%s", (int)myTask, c.total, c.landmarks, c.z, c.world,
		c.worldCount, c.matrix, myResult.scoreNames.size(),
		myResult.scoreNames.empty() ? "" : myResult.scoreNames.front().c_str());
	if (myHeaderKey == key && (int)myHeader.size() == c.total)
		return;
	myHeaderKey = key;
	myHeader.assign((size_t)c.total, std::string());
	const char* base[] = { "id", "class", "conf", "u1", "v1", "u2", "v2", "label" };
	const char* uvc[] = { "u", "v", "c" };
	const char* xyz[] = { "x", "y", "z" };
	char buf[64];
	for (int col = 0; col < c.total; col++)
	{
		buf[0] = 0;
		if (col < kBaseCols)
			snprintf(buf, sizeof(buf), "%s", base[col]);
		else if (col < kBaseCols + c.landmarks * 3)
			snprintf(buf, sizeof(buf), "kp%d_%s", (col - c.kp) / 3, uvc[(col - c.kp) % 3]);
		else if (c.z >= 0 && col >= c.z && col < c.z + c.landmarks)
			snprintf(buf, sizeof(buf), "kp%d_z", col - c.z);
		else if (c.world >= 0 && col >= c.world && col < c.world + c.worldCount * 3)
			snprintf(buf, sizeof(buf), "w%d_%s", (col - c.world) / 3, xyz[(col - c.world) % 3]);
		else if (c.scores >= 0 && col >= c.scores && col < c.scores + (int)myResult.scoreNames.size())
		{
			myHeader[(size_t)col] = myResult.scoreNames[(size_t)(col - c.scores)];
			continue;
		}
		else if (c.matrix >= 0 && col >= c.matrix && col < c.matrix + 16)
			snprintf(buf, sizeof(buf), "m%d%d", (col - c.matrix) / 4, (col - c.matrix) % 4);
		myHeader[(size_t)col] = buf;
	}
}

void MediaPipeTOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	// Columns: id, class, conf, u1, v1, u2, v2, label, kp{i}_u/v/c, [kp{i}_z], [w{i}_x/y/z],
	//          [<blendshape names>], [m00..m33]. Same base layout as OnnxTOP's Info DAT, in TD UV
	//          (origin bottom-left) of the input image.
	const Columns c = columns();
	char buf[64];
	if (index == 0)
	{
		headerRow(c);
		for (int col = 0; col < nEntries; col++)
			entries->values[col]->setString(col < (int)myHeader.size() ? myHeader[(size_t)col].c_str() : "");
		return;
	}
	// TD asks for rows based on an earlier getInfoDATSize(); never trust the index blindly.
	if (index < 1 || index - 1 >= (int32_t)myResult.instances.size())
	{
		for (int col = 0; col < nEntries; col++)
			entries->values[col]->setString("0");
		return;
	}
	const MpInstance& inst = myResult.instances[index - 1];
	const int n = std::min(c.landmarks, (int)inst.lm.size() / kLmStride);
	auto lm = [&](int k, int f) { return inst.lm[(size_t)k * kLmStride + f]; };
	// Box: the model's own (face detector), else the extent of the landmarks that are present
	// (holistic leaves missing parts at 0 with visibility = presence = 0).
	float xmin = 1.f, xmax = 0.f, ymin = 1.f, ymax = 0.f;
	if (inst.hasBox)
	{
		xmin = inst.box[0];
		ymin = inst.box[1];
		xmax = inst.box[2];
		ymax = inst.box[3];
	}
	else
	{
		for (int k = 0; k < n; k++)
		{
			if (lm(k, 3) <= 0.f && lm(k, 4) <= 0.f)
				continue;
			xmin = std::min(xmin, lm(k, 0));
			xmax = std::max(xmax, lm(k, 0));
			ymin = std::min(ymin, lm(k, 1));
			ymax = std::max(ymax, lm(k, 1));
		}
	}
	auto clamp01 = [](float v) { return std::clamp(v, 0.f, 1.f); };
	float box[4] = { clamp01(xmin), clamp01(1.f - ymax), clamp01(xmax), clamp01(1.f - ymin) };
	if (xmax < xmin)
		box[0] = box[1] = box[2] = box[3] = 0.f;	// no landmarks (segment rows)
	// kp_c: pose visibility for pose / holistic (holistic's face and hand points are 1 when
	// present, 0 when missing), presence otherwise.
	const int confField = myTask == MpTaskKind::Pose || myTask == MpTaskKind::Holistic ? 3 : 4;

	for (int col = 0; col < nEntries; col++)
	{
		float v = 0.f;
		bool isInt = false;
		if (col == 0)
		{
			v = (float)(index - 1);
			isInt = true;
		}
		else if (col == 1)
		{
			v = (float)inst.cls;
			isInt = true;
		}
		else if (col == 2)
			v = myTask == MpTaskKind::Gesture ? inst.gestureScore : inst.conf;
		else if (col >= 3 && col <= 6)
			v = box[col - 3];
		else if (col == 7)
		{
			entries->values[col]->setString(inst.label.c_str());
			continue;
		}
		else if (col < kBaseCols + c.landmarks * 3)
		{
			int k = (col - c.kp) / 3, f = (col - c.kp) % 3;
			if (k < n)
				v = f == 0 ? lm(k, 0) : f == 1 ? 1.f - lm(k, 1) : lm(k, confField);
		}
		else if (c.z >= 0 && col >= c.z && col < c.z + c.landmarks)
			v = col - c.z < n ? lm(col - c.z, 2) : 0.f;
		else if (c.world >= 0 && col >= c.world && col < c.world + c.worldCount * 3)
		{
			size_t i = (size_t)(col - c.world);
			v = i < inst.world.size() ? inst.world[i] : 0.f;
		}
		else if (c.scores >= 0 && col >= c.scores && col < c.scores + (int)myResult.scoreNames.size())
		{
			size_t i = (size_t)(col - c.scores);
			v = i < inst.scores.size() ? inst.scores[i] : 0.f;
		}
		else if (c.matrix >= 0 && col >= c.matrix && col < c.matrix + 16)
		{
			size_t i = (size_t)(col - c.matrix);
			v = i < inst.matrix.size() ? inst.matrix[i] : 0.f;
		}
		if (isInt)
			snprintf(buf, sizeof(buf), "%d", (int)v);
		else
		{
			// to_chars: several times cheaper than snprintf, and this runs for every cell
			// (~2500 per holistic row) on TD's main thread.
			auto res = std::to_chars(buf, buf + sizeof(buf) - 1, v, std::chars_format::fixed, 4);
			*res.ptr = 0;
		}
		entries->values[col]->setString(buf);
	}
}

void MediaPipeTOP::getWarningString(OP_String* warning, void*)
{
	if (myError.empty() && !myWarning.empty())
		warning->setString(myWarning.c_str());
}

void MediaPipeTOP::getErrorString(OP_String* error, void*)
{
	if (!myError.empty())
		error->setString((myModelHint.empty() ? myError : myError + "\n" + myModelHint).c_str());
}

void MediaPipeTOP::getInfoPopupString(OP_String* info, void*)
{
	std::string lib = mpapi::loadedPath();
	std::string s = "td-native-inference " TDNI_VERSION "\nMediaPipe Tasks C API (CPU / XNNPACK)\n" +
		(lib.empty() ? std::string("libmediapipe.dll not loaded") : lib) + "\n";
	s += std::string("Task: ") + kTaskLabels[(int)myTask] + "\nModel: " + myModelPath + "\n";
	s += "Frame sent to MediaPipe: " + std::to_string(myFrameW) + "x" + std::to_string(myFrameH) + "\n";
	char buf[128];
	snprintf(buf, sizeof(buf), "Last inference: %.2f ms, %d detections, result #%llu\n", myResult.inferMs,
		(int)myResult.instances.size(), (unsigned long long)myResult.seq);
	s += buf;
	info->setString(s.c_str());
}

// ---------------------------------------------------------------- parameters

void MediaPipeTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	auto check = [](OP_ParAppendResult r) { assert(r == OP_ParAppendResult::Success); (void)r; };
	const char* page = "MediaPipe";

	{
		OP_StringParameter sp("Task");
		sp.label = "Task";
		sp.page = page;
		sp.defaultValue = kTaskNames[0];
		sp.help = "Which official MediaPipe task runs. Each loads its own .task bundle (fetch_mediapipe.ps1 "
			"puts them in models/mediapipe). Runs on the CPU on a worker thread; results arrive about two frames late.";
		check(manager->appendMenu(sp, kNumTasks, kTaskNames, kTaskLabels));
	}
	const char* defaults[] = { "models/mediapipe/face_landmarker.task", "models/mediapipe/hand_landmarker.task",
		"models/mediapipe/gesture_recognizer.task", "models/mediapipe/pose_landmarker_full.task",
		"models/mediapipe/holistic_landmarker.task", "models/mediapipe/blaze_face_short_range.tflite",
		"models/mediapipe/selfie_segmenter.tflite" };
	const char* labels[] = { "Face Model", "Hand Model", "Gesture Model", "Pose Model", "Holistic Model",
		"Face Detector Model", "Segmenter Model" };
	const char* helps[] = { nullptr, nullptr, nullptr,
		"pose_landmarker_lite (~8 ms), _full (~12 ms) or _heavy (~40 ms -- too slow for real time on CPU).",
		"One person: pose, both hands and face in one pass (~22 ms).",
		"BlazeFace short range (~1.3 ms): faces within ~2 m.",
		"selfie_segmenter (~4.5 ms, person), selfie_segmenter_landscape (~2.3 ms), hair_segmenter (~32 ms), "
			"selfie_multiclass_256x256 (6 classes, ~156 ms -- fetch_mediapipe.ps1 -Heavy)." };
	for (int t = 0; t < kNumTasks; t++)
	{
		OP_StringParameter sp(kModelPars[t]);
		sp.label = labels[t];
		sp.page = page;
		sp.defaultValue = defaults[t];
		if (helps[t])
			sp.help = helps[t];
		check(manager->appendFile(sp));
	}
	{
		OP_NumericParameter np("Active");
		np.label = "Active";
		np.page = page;
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Reload");
		np.label = "Reload Model";
		np.page = page;
		check(manager->appendPulse(np));
	}
	{
		OP_NumericParameter np("Maxinstances");
		np.label = "Max Faces / Hands / Poses";
		np.page = page;
		np.defaultValues[0] = 1;
		np.minValues[0] = np.minSliders[0] = 1;
		np.maxValues[0] = np.maxSliders[0] = kMaxInstances;
		np.clampMins[0] = np.clampMaxes[0] = true;
		np.help = "Set it to how many you really expect: while fewer are tracked, MediaPipe re-runs its "
			"detector every frame looking for the rest (2 hands with 1 visible costs ~3x).";
		check(manager->appendInt(np));
	}
	{
		OP_NumericParameter np("Detectwidth");
		np.label = "Detect Width";
		np.page = page;
		np.defaultValues[0] = 640;
		np.minValues[0] = np.minSliders[0] = 0;
		np.maxSliders[0] = 1920;
		np.clampMins[0] = true;
		np.help = "The input is resized on the GPU to this width (aspect kept) before the CPU copy. "
			"0 = input size. MediaPipe's models run at 128-256 px, so bigger mostly costs copy time.";
		check(manager->appendInt(np));
	}
	const char* thrNames[] = { "Mindetection", "Minpresence", "Mintracking" };
	const char* thrLabels[] = { "Min Detection Confidence", "Min Presence Confidence", "Min Tracking Confidence" };
	for (int i = 0; i < 3; i++)
	{
		OP_NumericParameter np(thrNames[i]);
		np.label = thrLabels[i];
		np.page = page;
		np.defaultValues[0] = 0.5;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxValues[0] = np.maxSliders[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Blendshapes");
		np.label = "Face Blendshapes";
		np.page = page;
		np.defaultValues[0] = 1.0;
		np.help = "Adds the 52 ARKit-style blendshape scores as Info DAT columns.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Matrix");
		np.label = "Face Transform Matrix";
		np.page = page;
		np.help = "Adds the 4x4 facial transformation matrix (canonical face -> camera space) as m00..m33.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Segmentation");
		np.label = "Pose Segmentation Mask";
		np.page = page;
		np.help = "Pose / Holistic: outputs the person mask (Mono 32-bit float) as this TOP's image instead of the frame.";
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Segoutput");
		sp.label = "Segment Output";
		sp.page = page;
		sp.defaultValue = kSegOutputNames[0];
		sp.help = "Image Segmenter: Foreground = the only mask (selfie) or 1 - background (multi-class models); "
			"One Class = Segment Class's confidence; Category Index = per-pixel class index (0, 1, 2...).";
		check(manager->appendMenu(sp, 3, kSegOutputNames, kSegOutputLabels));
	}
	{
		OP_NumericParameter np("Segclass");
		np.label = "Segment Class";
		np.page = page;
		np.minValues[0] = np.minSliders[0] = 0;
		np.maxSliders[0] = 5;
		np.clampMins[0] = true;
		np.help = "Class index for Segment Output One Class; the Info DAT lists the model's classes in order.";
		check(manager->appendInt(np));
	}
	{
		OP_NumericParameter np("Outputz");
		np.label = "Landmark Z Columns";
		np.page = page;
		np.help = "Adds kp{i}_z: MediaPipe's relative depth, roughly in the same units as u.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Outputworld");
		np.label = "World Landmark Columns";
		np.page = page;
		np.help = "Hands / pose / holistic: adds w{i}_x/y/z, metric 3D landmarks (meters, origin at the hand / hip center).";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Waitforresult");
		np.label = "Wait for Result";
		np.page = page;
		np.help = "Waits in the cook (up to Max Wait) for this frame's result, so the landmarks match the image "
			"(info channel result_delay 0). Worth it for fast models (Face Detector, Hands); a slow one "
			"(Holistic) costs TD frame rate. Off: the cook never waits, and results arrive a couple of frames "
			"late (result_delay says how many).";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Maxwait");
		np.label = "Max Wait (ms)";
		np.page = page;
		np.defaultValues[0] = 20.0;
		np.minValues[0] = np.minSliders[0] = 0.0;
		np.maxSliders[0] = 100.0;
		np.clampMins[0] = true;
		np.help = "Longest a cook waits for its result; past it the result lands on a later cook.";
		check(manager->appendFloat(np));
	}
}

void MediaPipeTOP::pulsePressed(const char* name, void*)
{
	if (!strcmp(name, "Reload"))
		myReload = true;
}
