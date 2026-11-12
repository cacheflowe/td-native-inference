// box_tracker_pipeline.BoxTrackerPipeline (YOLO detect / segment) and
// face_tracker_pipeline.FaceTrackerPipeline (YuNet): ByteTracker plus smoothing, the display gate,
// presence (segment), keypoints and head pose (faces), and their tables.
#pragma once

#include <array>
#include <map>
#include <memory>
#include <vector>

#include "BoxTracker.h"
#include "Common.h"

namespace tracking {

struct BoxParams {
	double confThreshold = 0.5, lowConfThreshold = 0.1, trackIouThreshold = 0.3, trackBuffer = 30;
	int minHits = 3;
	double smoothing = 0.5;
	double presenceRise = 5, presenceFall = 15;
};

class BoxPipeline {
public:
	explicit BoxPipeline(bool segmentation) : segmentation_(segmentation) {}

	// Admission gate: score above the low threshold, min box size, NMS, (segment) min mask fill.
	static std::vector<std::shared_ptr<const Detection>> filter(const std::vector<std::shared_ptr<const Detection>>& dets,
		double lowConf, double minW, double minH, double nmsIou, bool segmentation, double minMaskArea);

	void update(const std::vector<std::shared_ptr<const Detection>>& dets, const BoxParams& p);
	void writeTracks(Table& out) const;

private:
	struct Obj {
		long long id;
		int classId;
		std::string className;
		double score, box[4], vx, vy;
		int lost, total;
		double maskArea = 0, presence = 0;
		int detIndex = -1;
	};
	bool segmentation_;
	ByteTracker tracker_;
	std::map<long long, std::array<double, 4>> box_;
	std::map<long long, int> gap_;
	std::map<long long, double> presence_;
	std::vector<Obj> objects_;
};

struct FaceParams {
	double confThreshold = 0.7, lowConfThreshold = 0.3, trackIouThreshold = 0.2, trackBuffer = 45;
	int minHits = 3;
	double smoothing = 0.3;
	bool solvePnP = false;
	double yawScale = 340, pitchScale = 100, focalScale = 1, poseSmoothing = 0.5;
	double frameW = 1, frameH = 1;
};

class FacePipeline {
public:
	static constexpr int kKeypoints = 5;

	static std::vector<std::shared_ptr<const Detection>> filter(const std::vector<std::shared_ptr<const Detection>>& dets,
		double lowConf, double minW, double minH);

	void update(const std::vector<std::shared_ptr<const Detection>>& dets, const FaceParams& p);
	void writeTracks(Table& out) const;
	void writeJoints(Table& out) const;

private:
	struct Pose {
		double kp[kKeypoints][2];
		double yaw = 0, pitch = 0, roll = 0;
	};
	struct Obj {
		long long id;
		double score, box[4], vx, vy;
		int lost, total;
		Pose pose;
	};
	ByteTracker tracker_;
	std::map<long long, std::array<double, 4>> box_;
	std::map<long long, int> gap_;
	std::map<long long, Pose> pose_;
	std::vector<Obj> objects_;
};

// Head direction in degrees (yaw, pitch, roll) from YuNet's 5 keypoints (TD UV).
void headPoseGeometric(const double kp[5][2], double yawScale, double pitchScale, double out[3]);
// cv2.solvePnP (iterative, from a facing-the-camera guess) against a canonical face; false if the
// solve fails.
bool headPoseSolvePnP(const double kp[5][2], double width, double height, double focalScale, double out[3]);

void boxHeader(Row& row);
void jointsHeader(Row& row);
void bonesHeader(Row& row);

}  // namespace tracking
