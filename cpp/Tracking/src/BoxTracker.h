// object_tracker.ByteTracker in C++: a constant-velocity Kalman filter per box, ByteTrack's two-stage
// (high, then low confidence) IoU association solved with the Hungarian algorithm, and the float32
// NMS the box pipeline runs first. Numbers follow numpy's: see the notes in BoxTracker.cpp.
#pragma once

#include <memory>
#include <vector>

#include "KeypointTracker.h"	// Detection

namespace tracking {

class BoxTrack {
public:
	BoxTrack(const double box[4], double score, std::shared_ptr<const Detection> payload, int minHits);

	void box(double out[4]) const;	// the Kalman estimate as [x1, y1, x2, y2]
	void predict();
	void update(const double box[4], double score, std::shared_ptr<const Detection> payload);
	void markLost();

	long long trackId;
	double mean[8];
	double cov[8][8];
	double score;
	std::shared_ptr<const Detection> payload;
	int lostFrames = 0;
	int totalFrames = 1;
	int hits = 1;
	int minHits;
	bool confirmed;

	static long long nextId;	// process-wide, like object_tracker.Track._next_id
};

class ByteTracker {
public:
	using Dets = std::vector<std::shared_ptr<const Detection>>;

	const std::vector<std::shared_ptr<BoxTrack>>& update(Dets dets);

	double highThresh = 0.5, lowThresh = 0.1, matchThresh = 0.7, trackBuffer = 30;
	int maxDetections = 50, maxTracks = 50, minHits = 1;
	std::vector<std::shared_ptr<BoxTrack>> tracks;

private:
	Dets capDetections(const Dets& dets) const;
	// Matched tracks are updated; returns the unmatched track indices and detection indices.
	std::pair<std::vector<int>, std::vector<int>> associate(const std::vector<int>& trackIdx, const Dets& dets);
};

// Indices of the detections NMS keeps, highest score first (ties in detection order), computed in
// float32 like the numpy version.
std::vector<int> nmsFloat32(const std::vector<std::shared_ptr<const Detection>>& dets, double iouThreshold);

// object_tracker.linear_sum_assignment_thresh: accepted (row, col) pairs, in row order.
std::vector<std::pair<int, int>> assignWithThreshold(const std::vector<std::vector<double>>& cost, double thresh);

}  // namespace tracking
