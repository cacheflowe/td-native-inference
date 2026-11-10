// MediaPipeTOP -- CUDA-mode CPlusPlus TOP running Google's official MediaPipe Tasks
// (Face Landmarker, Hand Landmarker, Gesture Recognizer, Pose Landmarker) through the prebuilt
// MediaPipe Tasks C API (libmediapipe.dll, CPU / XNNPACK on Windows).
//
//   input TOP (cudaArray) -> CUDA resize to Detect Width, RGB8 -> async pinned host copy
//     -> (next cook) MpWorker thread (DetectForVideo, MediaPipe's own tracking) -> Info DAT / CHOP
//   output TOP: the frame MediaPipe sees, or the pose segmentation mask
//
// Neither inference nor the GPU readback blocks TD's cook: a frame reaches MediaPipe a cook after
// it was read back, and its result arrives on a later cook. See cpp/MediaPipeTOP/README.md.
#pragma once

#include "TOP_CPlusPlusBase.h"
#include "MpWorker.h"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

using namespace TD;

class MediaPipeTOP : public TOP_CPlusPlusBase
{
public:
	MediaPipeTOP(const OP_NodeInfo* info, TOP_Context* context);
	~MediaPipeTOP() override;

	void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void*) override;
	void		execute(TOP_Output*, const OP_Inputs*, void*) override;

	int32_t		getNumInfoCHOPChans(void*) override;
	void		getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
	bool		getInfoDATSize(OP_InfoDATSize* infoSize, void*) override;
	void		getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*) override;

	void		getWarningString(OP_String* warning, void*) override;
	void		getErrorString(OP_String* error, void*) override;
	void		getInfoPopupString(OP_String* info, void*) override;

	void		setupParameters(OP_ParameterManager* manager, void*) override;
	void		pulsePressed(const char* name, void*) override;

private:
	// Info DAT column layout for the current task / toggles.
	struct Columns
	{
		int		landmarks = 0;		// per instance (fixed by the task)
		int		kp = 0;				// first kp{i}_u column
		int		z = -1;				// first kp{i}_z column
		int		world = -1;			// first w{i}_x column
		int		worldCount = 0;		// landmarks with world columns (holistic: body + hands)
		int		scores = -1;		// first blendshape column
		int		matrix = -1;		// first m00 column
		int		total = 0;
	};
	void		headerRow(const Columns& c);
	Columns		columns() const;
	bool		ensureBuffers(int w, int h);

	TOP_Context*		myContext;
	cudaStream_t		myStream = nullptr;

	std::unique_ptr<MpWorker>	myWorker;
	bool						myReload = false;
	MpFrameResult				myResult;
	int64_t						myLastInputCooks = -1;
	std::vector<uint8_t>		mySubmitBuf;

	// GPU resize target + pinned host copy of the frame handed to MediaPipe. The copy back is
	// asynchronous: queued on one cook, handed to the worker on a later cook once myReadEvent
	// has fired, so TD's cook never waits for the GPU.
	uint8_t*			myDevRgb = nullptr;
	uint8_t*			myHostRgb = nullptr;
	size_t				myRgbBytes = 0;
	cudaEvent_t			myReadEvent = nullptr;
	bool				myReadInFlight = false;
	int					myReadW = 0, myReadH = 0;
	int64_t				myReadFrame = -1;		// the read-back frame's tag (input totalCooks)
	int64_t				myInputFrame = -1;		// the input's totalCooks at this cook
	float*				myDevMask = nullptr;
	size_t				myMaskBytes = 0;

	// Info DAT header row (thousands of names for holistic), rebuilt only when the layout changes.
	std::vector<std::string>	myHeader;
	std::string					myHeaderKey;

	MpTaskKind			myTask = MpTaskKind::Face;
	bool				myOutputZ = false;
	bool				myOutputWorld = false;
	bool				myActive = true;
	int					myFrameW = 0, myFrameH = 0;
	double				myExecuteMs = 0.0;
	std::string			myModelPath;
	std::string			myError;
	std::string			myWarning;
	std::string			myModelHint;	// where a missing model was looked for
};
