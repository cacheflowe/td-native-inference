// pose_tracker_pipeline.PoseTrackerPipeline: KeypointTracker plus smoothing, hide-missing parts,
// coasting, presence and gesture debounce, and the tracks / joints / bones tables.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Common.h"
#include "KeypointTracker.h"
#include "Skeleton.h"

namespace tracking {

struct PoseParams {
	double confThreshold = 0.005;
	double smoothing = 0.23;
	double maxMatchDist = 0.5;
	double relativeMatchDist = 0;	// 0 = absolute
	double trackBuffer = 15;
	int minHits = 3;
	double coast = 0.0;
	bool presence = false;
	double presenceRise = 3, presenceFall = 3;
	bool labels = false;	// gesture debounce (MediaPipe Hands)
	int labelHold = 3;
	double labelMinScore = 0.0;
};

class PoseTracker {
public:
	explicit PoseTracker(const Skeleton& skeleton);

	void update(const std::vector<std::shared_ptr<const Detection>>& dets, const PoseParams& p);
	void writeTracks(Table& out);
	void writeJoints(Table& out) const;
	void writeBones(Table& out) const;

	const Skeleton& skeleton() const { return skeleton_; }

	static constexpr int kPartConfirmFrames = 3;
	static constexpr int kReacquireSnapFrames = 2;

private:
	struct OptVec3 {
		bool set = false;
		Vec3 v;
	};
	struct State {
		std::vector<Kp> smoothed;
		std::vector<int> lost;
		double box[4];
		std::vector<std::pair<std::string, double>> extras;
		bool hasWorld = false;
		std::vector<OptVec3> world;
		std::vector<int> seen;
		int gap = 0;
		bool snap = false;
	};
	struct LabelState {
		std::string stable;
		std::optional<std::string> cand;
		int candN = 0;
		double score = 0;
		long long frames = 1;
		int onset = 1;
	};
	struct Object {
		long long trackId;
		double score;
		double box[4];
		std::vector<Kp> keypoints;
		std::optional<std::string> label;
		std::vector<std::pair<std::string, double>> extras;
		bool hasWorld = false;
		std::vector<OptVec3> world;
		std::vector<char> visible;
		double vx, vy;
		int lostFrames, totalFrames;
		double presence = 0;
		LabelState gesture;
	};

	void smoothWorld(State& s, const std::vector<Vec3>& rawWorld, const std::vector<Kp>& rawKpts, double smoothing);
	std::string stableLabel(const KeypointTrack& t, std::optional<std::string> raw, int hold, double minScore, double smoothing);
	std::vector<Kp> confirmedKeypoints(const State& s) const;

	Skeleton skeleton_;
	KeypointTracker tracker_;
	std::map<long long, State> state_;
	std::map<long long, double> presence_;
	std::map<long long, LabelState> labels_;
	bool presenceOn_ = false, labelsOn_ = false;
	std::vector<Object> objects_;
};

}  // namespace tracking
