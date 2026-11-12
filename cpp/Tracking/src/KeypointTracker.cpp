#include "KeypointTracker.h"

#include <algorithm>
#include <cmath>
#include <tuple>

#include "Common.h"

namespace tracking {

long long KeypointTrack::nextId = 1;

KeypointTrack::KeypointTrack(const Detection& det, int minHits_) : minHits(minHits_) {
	trackId = nextId++;
	std::copy(det.box, det.box + 4, box);
	score = det.score;
	confirmed = hits >= minHits;
	cx_ = (box[0] + box[2]) / 2.0;
	cy_ = (box[1] + box[3]) / 2.0;
}

void KeypointTrack::updateMatched(const Detection& det) {
	double cx = (det.box[0] + det.box[2]) / 2.0, cy = (det.box[1] + det.box[3]) / 2.0;
	vx = cx - cx_;
	vy = cy - cy_;
	cx_ = cx;
	cy_ = cy;
	std::copy(det.box, det.box + 4, box);
	score = det.score;
	lostFrames = 0;
	totalFrames += 1;
	hits += 1;
	if (hits >= minHits) confirmed = true;
}

void KeypointTrack::markMissed(double scoreDecay) {
	lostFrames += 1;
	totalFrames += 1;
	score *= scoreDecay;
}

KeypointTracker::KeypointTracker(double maxMatchDist_, std::vector<int> distance_, double dupDistFactor_,
		double trackBuffer_, int minHits_, double scoreDecay_, int maxTracks_, double relativeMatchDist_)
	: maxMatchDist(maxMatchDist_), relativeMatchDist(relativeMatchDist_), distance(std::move(distance_)),
	  dupDistFactor(dupDistFactor_), trackBuffer(trackBuffer_), minHits(minHits_), scoreDecay(scoreDecay_),
	  maxTracks(maxTracks_) {}

std::pair<double, double> KeypointTracker::predictedOffset(const KeypointTrack& t, int framesAhead) const {
	double c = coast;
	if (c <= 0.0) return {0.0, 0.0};
	int n = t.lostFrames + framesAhead;
	double gain = c < 1.0 ? c * (1.0 - std::pow(c, n)) / (1.0 - c) : double(n);
	return {t.vx * gain, t.vy * gain};
}

std::pair<double, double> KeypointTracker::match(const std::vector<Kp>& a, const double* boxA, const std::vector<Kp>& b,
		const double* boxB) const {
	double total = 0.0;
	for (int i : distance) total += pyHypot(a[i].v[0] - b[i].v[0], a[i].v[1] - b[i].v[1]);
	if (!relativeMatchDist) return {total, maxMatchDist};
	int count = int(distance.size());
	double size = std::max({pyHypot(boxA[2] - boxA[0], boxA[3] - boxA[1]), pyHypot(boxB[2] - boxB[0], boxB[3] - boxB[1]), 1e-6});
	return {total / std::max(count, 1) / size, relativeMatchDist};
}

const std::vector<std::shared_ptr<KeypointTrack>>& KeypointTracker::update(
		const std::vector<std::shared_ptr<const Detection>>& dets) {
	std::vector<std::tuple<double, int, int>> pairs;
	std::vector<Kp> shifted;
	for (int ti = 0; ti < int(tracks.size()); ++ti) {
		auto& track = *tracks[ti];
		const std::vector<Kp>* trackKpts = &track.payload->keypoints;
		if (track.lostFrames > 0 && coast > 0.0) {
			auto [dx, dy] = predictedOffset(track);
			shifted = *trackKpts;
			for (auto& kp : shifted) {
				kp.v[0] = kp.v[0] + dx;
				kp.v[1] = kp.v[1] + dy;
			}
			trackKpts = &shifted;
		}
		for (int di = 0; di < int(dets.size()); ++di) {
			auto [dist, threshold] = match(*trackKpts, track.box, dets[di]->keypoints, dets[di]->box);
			if (dist < threshold) pairs.emplace_back(dist, ti, di);
		}
	}
	std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

	std::vector<char> trackMatched(tracks.size(), 0), detMatched(dets.size(), 0);
	std::vector<std::shared_ptr<KeypointTrack>> matchedTracks;
	for (auto& [dist, ti, di] : pairs) {
		if (trackMatched[ti] || detMatched[di]) continue;
		tracks[ti]->updateMatched(*dets[di]);
		tracks[ti]->payload = dets[di];
		trackMatched[ti] = detMatched[di] = 1;
		matchedTracks.push_back(tracks[ti]);
	}

	std::vector<std::shared_ptr<KeypointTrack>> surviving;
	for (size_t ti = 0; ti < tracks.size(); ++ti) {
		if (!trackMatched[ti]) {
			tracks[ti]->markMissed(scoreDecay);
			if (tracks[ti]->lostFrames > trackBuffer) continue;
		}
		surviving.push_back(tracks[ti]);
	}
	tracks.swap(surviving);

	for (size_t di = 0; di < dets.size(); ++di) {
		if (detMatched[di]) continue;
		bool duplicate = false;
		for (auto& mt : matchedTracks) {
			auto [dist, threshold] = match(mt->payload->keypoints, mt->box, dets[di]->keypoints, dets[di]->box);
			if (dist < threshold * dupDistFactor) {
				duplicate = true;
				break;
			}
		}
		if (duplicate) continue;
		if (int(tracks.size()) >= maxTracks) break;
		auto t = std::make_shared<KeypointTrack>(*dets[di], minHits);
		t->payload = dets[di];
		tracks.push_back(t);
	}
	return tracks;
}

}  // namespace tracking
