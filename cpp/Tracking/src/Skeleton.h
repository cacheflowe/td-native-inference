// A keypoint layout: names, bones, the stable subset matching uses, and per-skeleton defaults.
// Mirrors python/mediapipe_skeletons.Skeleton; the instances are generated (Skeletons.gen.h).
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace tracking {

struct Skeleton {
	std::string name;
	std::vector<std::string> names;
	std::vector<std::pair<int, int>> edges;
	std::vector<int> distance;	// stable keypoints KeypointTracker matches on
	double maxMatchDist = 0.5;
	bool hideMissing = false;	// conf 0 = keypoint absent (Holistic's unseen hands / face)
	double relMatchDist = 0.15;
	int worldCount = 0;

	int count() const { return int(names.size()); }
};

inline Skeleton makeSkeleton(const char* name, const char* const* names, int count, const int* edges, int edgeCount,
		const int* distance, int distanceCount, double maxMatchDist, bool hideMissing, double relMatchDist, int worldCount) {
	Skeleton s;
	s.name = name;
	s.names.assign(names, names + count);
	for (int i = 0; i < edgeCount; ++i)
		s.edges.emplace_back(edges[2 * i], edges[2 * i + 1]);
	s.distance.assign(distance, distance + distanceCount);
	s.maxMatchDist = maxMatchDist;
	s.hideMissing = hideMissing;
	s.relMatchDist = relMatchDist;
	s.worldCount = worldCount;
	return s;
}

}  // namespace tracking
