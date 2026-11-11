// OnnxTOP -- CUDA-mode CPlusPlus TOP that runs ONNX models fully on the GPU.
//
//   input TOP (cudaArray) -> CUDA letterbox/normalize -> ORT (TensorRT EP, CUDA EP fallback,
//   IoBinding on device memory) -> CUDA postprocess -> output TOP (cudaArray)
//
// Detections go out through the Info DAT/CHOP; only the small detection tensor ever touches
// the CPU. See docs/learnings/onnx-cpp-top.md for design notes and measured results.
#pragma once

#include "TOP_CPlusPlusBase.h"
#include "Kernels.h"
#include "OnnxEngine.h"
#include "YoloDecoder.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace TD;

class OnnxTOP : public TOP_CPlusPlusBase
{
public:
	OnnxTOP(const OP_NodeInfo* info, TOP_Context* context);
	~OnnxTOP() override;

	void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void*) override;
	void		execute(TOP_Output*, const OP_Inputs*, void*) override;

	int32_t		getNumInfoCHOPChans(void*) override;
	void		getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
	bool		getInfoDATSize(OP_InfoDATSize* infoSize, void*) override;
	void		getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*) override;
	int			infoBaseCols() const;

	void		getWarningString(OP_String* warning, void*) override;
	void		getErrorString(OP_String* error, void*) override;
	void		getInfoPopupString(OP_String* info, void*) override;

	void		setupParameters(OP_ParameterManager* manager, void*) override;
	void		pulsePressed(const char* name, void*) override;

private:
	enum class LoadState { Idle, Loading, Ready, Failed };

	struct RawOptions
	{
		int		index = 0;
		bool	minMax = false;			// normalize by the (smoothed) per-frame value range
		float	smoothing = 0.f;		// 0 = this frame's range only
		bool	mapToSource = true;		// output at the input's size, letterbox undone
		bool	mono = false;			// single-channel tensor -> Mono32Float output
	};

	void		manageLoading(const EngineConfig& desired);
	bool		sameModelConfig(const EngineConfig& a, const EngineConfig& b);
	void		loadEngines(EngineConfig cfg);		// loader thread
	int			buildTrtEngine(const EngineConfig& cfg, bool background, std::string& message);
	bool		publishEngine(std::unique_ptr<OnnxEngine> engine, const std::string& warning);
	void		failLoad(const std::string& err);
	void		setLoadStatus(const std::string& s);
	void		setLoadWarning(const std::string& s);
	bool		runRaw(OnnxEngine& engine, const RawOptions& opt, const Letterbox& lb,
					cudaSurfaceObject_t outSurf, int outW, int outH);
	bool		runCrops(OnnxEngine& engine, int numCrops, bool softmax, std::string& err);

	TOP_Context*		myContext;
	cudaStream_t		myStream = nullptr;

	// Loader thread hands a finished engine to the main thread through myNextEngine.
	std::thread					myLoader;
	std::mutex					myLoadMutex;
	std::atomic<LoadState>		myLoadState{ LoadState::Idle };
	std::unique_ptr<OnnxEngine>	myNextEngine;
	std::string					myLoadError;
	std::string					myLoadStatus;	// shown as warning while Loading
	std::string					myLoadWarning;	// shown as warning once Ready
	std::atomic<bool>			myCancelLoad{ false };
	std::atomic<bool>			myResetState{ false };	// Reset State pulse: consumed by the next cook
	std::atomic<bool>			myUpgrading{ false };	// Auto: running on CUDA, TensorRT building
	std::string					myStaticModelPath;		// last model probed as fixed-size (Dynamic Size ignored)
	EngineConfig				myRequestedConfig;
	bool						myHasRequested = false;
	bool						myForceReload = false;
	std::unique_ptr<OnnxEngine>	myEngine;		// main thread only

	YoloDecoder					myYolo;
	bool						myYoloMode = false;	// whether myYolo's detections are current

	// Classify Crops mode: boxes parsed from the Crop DAT each frame, results per crop.
	struct CropResult
	{
		std::string			id;		// the Crop DAT's track_id / id cell
		int					cls = 0;
		float				conf = 0.f;
		std::vector<float>	probs;
	};
	static constexpr int		kMaxBatch = 64;
	std::vector<CropBox>		myCropBoxes;
	std::vector<std::string>	myCropIds;
	CropBox*					myDevCropBoxes = nullptr;	// kMaxBatch
	float*						myHostLogits = nullptr;		// pinned
	size_t						myHostLogitsBytes = 0;
	std::vector<CropResult>		myCrops;
	std::vector<std::string>	myClassNames;
	int							myCropClasses = 0;
	bool						myCropMode = false;	// whether myCrops is what the Info DAT shows
	int64_t						myCropCounter = 0;	// Classify Interval phase

	TensorRange*				myRawRange = nullptr;	// device; Raw mode min/max normalization
	bool						myRawRangeReset = true;	// forget the smoothed range (new engine/output)
	int							myRawRangeIndex = -1;

	std::string					myError;
	std::string					myWarning;
	std::string					myModelHint;	// where a missing model was looked for
	int							myModelW = 0, myModelH = 0;
	double						myInferMs = 0.0;
	double						myRunMs = 0.0;		// CPU time spent in OnnxEngine::run (the launch)
	double						myExecuteMs = 0.0;
	int64_t						myExecuteCount = 0;

	// Wait for Result off: this cook queues the inference without waiting for it, and the next
	// cook decodes its outputs (the decoders read the engine's output buffers, untouched until the
	// next run is queued after them on myStream). myRunStart/myRunDone time it on the GPU.
	struct PendingRun
	{
		bool						active = false;
		const OnnxEngine*			engine = nullptr;	// dropped if the engine was swapped meanwhile
		int							mode = 0;
		Letterbox					lb = {};
		int							crops = 0;			// Classify Crops: crops in the batch
		std::vector<std::string>	cropIds;
		int64_t						frame = -1;			// the input's totalCooks
	};
	PendingRun					myPending;
	cudaEvent_t					myRunStart = nullptr, myRunDone = nullptr;
	bool						myRunRecorded = false;	// the events hold a run
	bool						myWaitResult = true;
	int64_t						myLaunchedFrame = -1;	// the input's totalCooks at the last run
	void						readInferMs();			// infer_ms from the events, once they're done
	int64_t						myInputFrame = -1;		// the input's totalCooks this cook
	int64_t						myResultFrame = -1;		// ... and that of the frame the results are from

	// Info DAT header row, rebuilt only when its layout changes.
	std::vector<std::string>	myHeader;
	std::string					myHeaderKey;
	void						headerRow(int cols);
};
