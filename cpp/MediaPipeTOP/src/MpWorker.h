// MpWorker -- one MediaPipe task (face / hands / gesture / pose / holistic / face detector /
// image segmenter) running on its own thread.
//
// The TOP hands it RGB frames with submit(); the worker runs the task in VIDEO mode (so
// MediaPipe's own detector-skipping and landmark tracking between frames apply) and publishes
// each result for the TOP to pick up with latest(). Only one frame waits at a time: a newer
// submit() replaces a frame the worker hasn't started yet, so TD never queues up stale work.
//
// LIVE_STREAM mode isn't used because its C callback has no user-data pointer (it can't tell
// two TOPs apart); a private thread calling DetectForVideo gives the same "never blocks TD"
// behavior with per-instance state.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum class MpTaskKind : int { Face = 0, Hands = 1, Gesture = 2, Pose = 3, Holistic = 4, FaceDetector = 5, Segment = 6 };

// Holistic packs one person's parts into a single landmark list in this order.
constexpr int kHolisticPose = 0, kHolisticLeftHand = 33, kHolisticRightHand = 54, kHolisticFace = 75;
constexpr int kHolisticCount = 553;	// 33 + 21 + 21 + 478

// Which image Segment mode outputs as the TOP's mask.
enum class MpSegOutput : int { Foreground = 0, Class = 1, Category = 2 };

struct MpTaskConfig
{
	MpTaskKind	kind = MpTaskKind::Face;
	std::string	modelPath;
	int			maxInstances = 1;
	float		minDetection = 0.5f;
	float		minPresence = 0.5f;
	float		minTracking = 0.5f;
	bool		blendshapes = true;		// face, holistic
	bool		matrix = false;			// face
	bool		segmentation = false;	// pose, holistic

	bool operator==(const MpTaskConfig& o) const
	{
		return kind == o.kind && modelPath == o.modelPath && maxInstances == o.maxInstances &&
			minDetection == o.minDetection && minPresence == o.minPresence && minTracking == o.minTracking &&
			blendshapes == o.blendshapes && matrix == o.matrix && segmentation == o.segmentation;
	}
	bool operator!=(const MpTaskConfig& o) const { return !(*this == o); }
};

// One detected face / hand / pose / person / segmentation class. Landmarks are
// MediaPipe-normalized: x right, y DOWN from the top-left of the submitted frame, z roughly in
// x units (relative depth).
struct MpInstance
{
	int					cls = 0;			// hands/gesture: 0 = Left, 1 = Right; segment: class index
	float				conf = 1.f;			// hands: handedness score; pose: mean visibility;
											// face detector: detection score; segment: mean confidence
	std::string			label;				// "Left"/"Right", top gesture name, or segment class name
	float				gestureScore = 0.f;
	std::vector<float>	lm;					// per landmark (kLmStride floats): x, y, z, visibility, presence
	std::vector<float>	world;				// per landmark: x, y, z in meters (hands / pose / holistic)
	std::vector<float>	scores;				// face blendshape scores (names in MpFrameResult)
	std::vector<float>	matrix;				// face: 16 floats, row-major 4x4
	bool				hasBox = false;		// face detector: box from the model, not the landmarks
	float				box[4] = {};		// x1, y1, x2, y2, normalized, top-down
};
constexpr int kLmStride = 5;

struct MpFrameResult
{
	uint64_t					seq = 0;		// increments per processed frame
	int							frameW = 0, frameH = 0;
	double						inferMs = 0.0;
	std::vector<MpInstance>		instances;
	std::vector<std::string>	scoreNames;		// face blendshape category names
	std::vector<float>			mask;			// pose / holistic / segment mask, top-down
	int							maskW = 0, maskH = 0;
	std::string					error;			// per-frame detect error (task stays loaded)
};

class MpWorker
{
public:
	enum class State { Loading, Ready, Failed };

	explicit MpWorker(const MpTaskConfig& cfg);
	~MpWorker();

	const MpTaskConfig&	config() const { return myConfig; }
	State				state() const { return myState.load(); }
	std::string			loadError();

	// Takes an RGB (3 bytes/px, top-down) frame. Replaces a frame still waiting to start.
	void				submit(std::vector<uint8_t>& rgb, int w, int h);
	// True while a submitted frame hasn't produced a result yet.
	bool				pending() const { return mySubmitted.load() != myCompleted.load(); }
	// Copies the newest result into out if it's newer than out.seq. Returns true if it copied.
	bool				latest(MpFrameResult& out);
	uint64_t			droppedFrames() const { return myDropped.load(); }

	// Segment mode: which mask to produce (applies from the next frame; no task reload).
	void				setSegOutput(MpSegOutput output, int classIndex)
	{
		mySegOutput = (int)output;
		mySegClass = classIndex;
	}

private:
	void		run();
	bool		create(std::string& err);
	void		destroy();
	bool		detect(const uint8_t* rgb, int w, int h, int64_t ts, MpFrameResult& out);

	MpTaskConfig			myConfig;
	void*					myTask = nullptr;
	std::vector<std::string>	mySegLabels;

	std::thread				myThread;
	std::mutex				myMutex;
	std::condition_variable	myWake;
	bool					myStop = false;
	std::atomic<State>		myState{ State::Loading };
	std::string				myLoadError;

	std::vector<uint8_t>	myPendingFrame;		// guarded by myMutex
	int						myPendingW = 0, myPendingH = 0;
	bool					myHasPending = false;
	std::atomic<uint64_t>	mySubmitted{ 0 }, myCompleted{ 0 }, myDropped{ 0 };
	std::atomic<int>		mySegOutput{ 0 }, mySegClass{ 0 };

	MpFrameResult			myResult;			// guarded by myMutex
	int64_t					myLastTs = -1;
	// Worker-thread only: frame size the task last ran on, and graph-error recovery.
	int						myTaskW = -1, myTaskH = -1;
	bool					myTaskError = false;
	std::chrono::steady_clock::time_point	myRebuildTime;
};
