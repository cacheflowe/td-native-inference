// object_tracker.KeypointTracker: greedy-global matching on stable keypoints (no Kalman), for every
// skeleton mode. Ported to reproduce the Python output exactly (see tests/tracking_replay.py).
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tracking {

// One keypoint: x, y, conf (+ z when n == 4). Python keeps these as 3- or 4-element lists.
struct Kp {
	double v[4] = {0, 0, 0, 0};
	int n = 3;
};

struct Vec3 {
	double x = 0, y = 0, z = 0;
};

// A decoded detection (onnxtop_tables.read_detections' dict). box = [x_left, y_bottom, x_right, y_top].
struct Detection {
	double box[4] = {0, 0, 0, 0};
	double score = 0;
	std::vector<Kp> keypoints;
	std::vector<Vec3> world;	// empty = no world landmarks
	bool hasLabel = false;
	std::string label;
	std::vector<std::pair<std::string, double>> extras;	// in column order (dict insertion order)
	bool hasExtras = false;	// MediaPipe modes always carry an extras dict, possibly empty
	// Box modes (detect / segment).
	int classId = 0;
	std::string className;
	bool hasMaskArea = false;
	double maskArea = 0;	// mask_fill: the mask's share of its own box
	int detIndex = -1;	// the Info DAT id: the detection's value (+ 1) in an Instances matte
};

class KeypointTrack {
public:
	KeypointTrack(const Detection& det, int minHits);
	void updateMatched(const Detection& det);
	void markMissed(double scoreDecay);

	long long trackId;
	double box[4];
	double score;
	std::shared_ptr<const Detection> payload;
	int lostFrames = 0;
	int totalFrames = 1;
	int hits = 1;
	int minHits;
	bool confirmed;
	double vx = 0, vy = 0;	// mean[4], mean[5]: frame-to-frame box-center delta

	static long long nextId;	// process-wide, like the Python class attribute

private:
	double cx_, cy_;
};

class KeypointTracker {
public:
	explicit KeypointTracker(double maxMatchDist, std::vector<int> distance, double dupDistFactor = 0.5,
		double trackBuffer = 15, int minHits = 1, double scoreDecay = 0.9, int maxTracks = 50, double relativeMatchDist = 0);

	// Detections must stay alive until the next update (tracks share them as payload).
	const std::vector<std::shared_ptr<KeypointTrack>>& update(const std::vector<std::shared_ptr<const Detection>>& dets);
	std::pair<double, double> predictedOffset(const KeypointTrack& t, int framesAhead = 1) const;

	double maxMatchDist;
	double relativeMatchDist;	// 0 = absolute maxMatchDist
	std::vector<int> distance;
	double dupDistFactor;
	double trackBuffer;
	int minHits;
	double scoreDecay;
	int maxTracks;
	double coast = 0.0;
	std::vector<std::shared_ptr<KeypointTrack>> tracks;

private:
	std::pair<double, double> match(const std::vector<Kp>& a, const double* boxA, const std::vector<Kp>& b,
		const double* boxB) const;
};

}  // namespace tracking
