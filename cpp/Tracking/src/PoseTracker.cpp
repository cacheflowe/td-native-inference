#include "PoseTracker.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace tracking {

namespace {

// PARAMETERS' defaults for the tracker created before the first update.
constexpr double kDefaultTrackBuffer = 15;
constexpr int kDefaultMinHits = 3;
constexpr double kDupDistFactor = 0.5;

void setExtra(std::vector<std::pair<std::string, double>>& extras, const std::string& name, double value) {
	for (auto& e : extras)
		if (e.first == name) {
			e.second = value;
			return;
		}
	extras.emplace_back(name, value);
}

const double* findExtra(const std::vector<std::pair<std::string, double>>& extras, const std::string& name) {
	for (auto& e : extras)
		if (e.first == name) return &e.second;
	return nullptr;
}

}  // namespace

PoseTracker::PoseTracker(const Skeleton& skeleton)
	: skeleton_(skeleton),
	  tracker_(skeleton.maxMatchDist, skeleton.distance, kDupDistFactor, kDefaultTrackBuffer, kDefaultMinHits, 0.9, 50,
		  skeleton.relMatchDist) {}

void PoseTracker::smoothWorld(State& s, const std::vector<Vec3>& rawWorld, const std::vector<Kp>& rawKpts, double smoothing) {
	bool snap = s.snap;
	s.snap = false;
	if (!s.hasWorld || s.world.size() != rawWorld.size()) {
		s.hasWorld = true;
		s.world.assign(rawWorld.size(), OptVec3());
	}
	const bool hide = skeleton_.hideMissing;
	for (size_t k = 0; k < rawWorld.size(); ++k) {
		if (hide && !rawKpts.empty() && rawKpts[k].v[2] <= 0.0) continue;
		auto& prev = s.world[k];
		bool back = snap || (hide && s.seen[k] == 1);
		const Vec3& p = rawWorld[k];
		if (!prev.set || back) {
			prev.set = true;
			prev.v = p;
		} else {
			prev.v.x = prev.v.x * smoothing + p.x * (1.0 - smoothing);
			prev.v.y = prev.v.y * smoothing + p.y * (1.0 - smoothing);
			prev.v.z = prev.v.z * smoothing + p.z * (1.0 - smoothing);
		}
	}
}

std::string PoseTracker::stableLabel(const KeypointTrack& t, std::optional<std::string> raw, int hold, double minScore,
		double smoothing) {
	double score = 0;
	if (t.lostFrames == 0) {
		score = t.score;
		if (!raw || raw->empty() || score < minScore) raw = std::string("None");
	}
	auto it = labels_.find(t.trackId);
	if (it == labels_.end()) {
		LabelState st;
		st.stable = t.lostFrames == 0 ? *raw : "None";
		st.score = t.score;
		st.frames = 1;
		st.onset = 1;
		labels_[t.trackId] = st;
		return st.stable;
	}
	LabelState& st = it->second;
	st.onset = 0;
	st.frames += 1;
	if (t.lostFrames == 0) {
		if (*raw == st.stable) {
			st.cand.reset();
			st.candN = 0;
			st.score = st.score * smoothing + score * (1.0 - smoothing);
		} else {
			st.candN = (st.cand && *st.cand == *raw) ? st.candN + 1 : 1;
			st.cand = *raw;
			if (st.candN >= hold) {
				st.stable = *raw;
				st.cand.reset();
				st.candN = 0;
				st.score = score;
				st.frames = 1;
				st.onset = 1;
			}
		}
	}
	return st.stable;
}

std::vector<Kp> PoseTracker::confirmedKeypoints(const State& s) const {
	std::vector<Kp> out = s.smoothed;
	if (!skeleton_.hideMissing) return out;
	for (size_t k = 0; k < out.size(); ++k)
		if (s.seen[k] < kPartConfirmFrames) out[k].v[2] = 0.0;
	return out;
}

void PoseTracker::update(const std::vector<std::shared_ptr<const Detection>>& dets, const PoseParams& p) {
	presenceOn_ = p.presence;
	labelsOn_ = p.labels;
	tracker_.maxMatchDist = p.maxMatchDist;
	tracker_.relativeMatchDist = p.relativeMatchDist;
	tracker_.trackBuffer = p.trackBuffer;
	tracker_.minHits = p.minHits;
	tracker_.coast = p.coast;

	const auto& active = tracker_.update(dets);
	std::set<long long> activeIds;
	for (auto& t : active) activeIds.insert(t->trackId);
	const int numKp = skeleton_.count();
	const bool hide = skeleton_.hideMissing;
	const double smoothing = p.smoothing;
	objects_.clear();

	for (auto& tp : active) {
		const KeypointTrack& t = *tp;
		if (t.score < p.confThreshold || !t.confirmed) continue;
		const double* boxRaw = t.box;
		const Detection& payload = *t.payload;
		const std::vector<Kp>& rawKpts = payload.keypoints;
		auto it = state_.find(t.trackId);
		if (it == state_.end()) {
			State s;
			if (!rawKpts.empty())
				s.smoothed = rawKpts;
			else
				s.smoothed.assign(numKp, Kp());
			s.lost.assign(numKp, 0);
			std::copy(boxRaw, boxRaw + 4, s.box);
			s.extras = payload.extras;
			if (rawKpts.empty())
				s.seen.assign(numKp, kPartConfirmFrames);
			else
				for (auto& kp : rawKpts) s.seen.push_back(kp.v[2] > 0.0 ? kPartConfirmFrames : 0);
			it = state_.emplace(t.trackId, std::move(s)).first;
			if (!payload.world.empty()) smoothWorld(it->second, payload.world, rawKpts, smoothing);
		} else if (t.lostFrames == 0) {
			State& s = it->second;
			if (s.gap >= kReacquireSnapFrames) {
				std::copy(boxRaw, boxRaw + 4, s.box);
				s.extras = payload.extras;
				s.snap = true;
				for (size_t k = 0; k < rawKpts.size(); ++k)
					if (!(hide && rawKpts[k].v[2] <= 0.0)) s.smoothed[k] = rawKpts[k];
			}
			for (size_t k = 0; k < rawKpts.size(); ++k) {
				const Kp& kp = rawKpts[k];
				Kp& prev = s.smoothed[k];
				if (hide && kp.v[2] <= 0.0) {
					prev.v[2] = kp.v[2];
					s.seen[k] = 0;
					continue;
				}
				s.seen[k] += 1;
				if (hide && s.seen[k] == 1) {
					s.smoothed[k] = kp;
				} else {
					Kp next;
					next.n = kp.n;
					for (int i = 0; i < kp.n; ++i)
						next.v[i] = (i != 2 && i < prev.n) ? prev.v[i] * smoothing + kp.v[i] * (1.0 - smoothing) : kp.v[i];
					s.smoothed[k] = next;
				}
				s.lost[k] = 0;
			}
			for (auto& [name, value] : payload.extras) {
				const double* prevValue = findExtra(s.extras, name);
				double prevV = prevValue ? *prevValue : value;
				setExtra(s.extras, name, prevV * smoothing + value * (1.0 - smoothing));
			}
			if (!payload.world.empty()) smoothWorld(s, payload.world, rawKpts, smoothing);
			for (int i = 0; i < 4; ++i) s.box[i] = s.box[i] * smoothing + boxRaw[i] * (1.0 - smoothing);
		}
		State& s = it->second;
		s.gap = t.lostFrames;
		if (t.lostFrames > 0) {
			for (int k = 0; k < numKp; ++k) s.lost[k] += 1;
			if (p.coast > 0.0) {
				double f = std::pow(p.coast, t.lostFrames);
				double dx = t.vx * f, dy = t.vy * f;
				for (auto& kp : s.smoothed) {
					kp.v[0] += dx;
					kp.v[1] += dy;
				}
				s.box[0] += dx;
				s.box[1] += dy;
				s.box[2] += dx;
				s.box[3] += dy;
			}
		}

		Object obj;
		if (p.presence) {
			double current = presence_.count(t.trackId) ? presence_[t.trackId] : 0.0;
			double step = 1.0 / std::max(t.lostFrames == 0 ? p.presenceRise : p.presenceFall, 1.0);
			current = t.lostFrames == 0 ? current + step : current - step;
			current = std::max(0.0, std::min(1.0, current));
			presence_[t.trackId] = current;
			obj.presence = current;
		}
		std::optional<std::string> label;
		if (payload.hasLabel) label = payload.label;
		if (p.labels) {
			label = stableLabel(t, label, p.labelHold, p.labelMinScore, smoothing);
			obj.gesture = labels_[t.trackId];
		}
		obj.trackId = t.trackId;
		obj.score = t.score;
		std::copy(s.box, s.box + 4, obj.box);
		obj.keypoints = confirmedKeypoints(s);
		obj.label = label;
		obj.extras = s.extras;
		obj.hasWorld = s.hasWorld;
		obj.world = s.world;
		obj.visible.resize(numKp);
		for (int k = 0; k < numKp; ++k) obj.visible[k] = s.lost[k] <= tracker_.trackBuffer;
		obj.vx = t.vx;
		obj.vy = t.vy;
		obj.lostFrames = t.lostFrames;
		obj.totalFrames = t.totalFrames;
		objects_.push_back(std::move(obj));
	}

	for (auto it = state_.begin(); it != state_.end();) it = activeIds.count(it->first) ? std::next(it) : state_.erase(it);
	for (auto it = presence_.begin(); it != presence_.end();) it = activeIds.count(it->first) ? std::next(it) : presence_.erase(it);
	for (auto it = labels_.begin(); it != labels_.end();) it = activeIds.count(it->first) ? std::next(it) : labels_.erase(it);
}

void PoseTracker::writeTracks(Table& out) {
	bool hasZ = false;
	for (auto& o : objects_)
		if (!o.keypoints.empty() && o.keypoints[0].n > 3) hasZ = true;
	const int n = hasZ ? 4 : 3;
	static const char* kSuffixes[] = {"x", "y", "conf", "z"};
	std::vector<std::string> extraNames;
	if (!objects_.empty())
		for (auto& e : objects_[0].extras) extraNames.push_back(e.first);
	size_t worldCount = 0;
	for (auto& o : objects_)
		if (o.hasWorld && !o.world.empty()) worldCount = std::max(worldCount, o.world.size());

	out.clear();
	Row header = {"track_id", "label", "score", "cx", "cy", "w", "h", "x_left", "x_right", "y_top", "y_bottom", "vx", "vy",
		"lost_frames", "total_frames"};
	if (presenceOn_) header.push_back("presence");
	if (labelsOn_)
		for (const char* h : {"gesture", "gesture_score", "gesture_frames", "gesture_onset"}) header.push_back(h);
	for (auto& name : skeleton_.names)
		for (int i = 0; i < n; ++i) header.push_back(name + "_" + kSuffixes[i]);
	for (size_t k = 0; k < worldCount; ++k)
		for (const char* s : {"x", "y", "z"}) header.push_back(skeleton_.names[k] + "_w" + s);
	for (auto& e : extraNames) header.push_back(e);
	for (const char* h : {"r", "g", "b"}) header.push_back(h);
	out.push_back(std::move(header));

	for (auto& o : objects_) {
		Row row;
		row.reserve(out[0].size());
		row.push_back(integer(o.trackId));
		if (o.label && !o.label->empty()) {
			std::string text = integer(o.trackId) + " " + *o.label;
			row.push_back(buildLabel(o.trackId, o.score, &text));
		} else {
			row.push_back(buildLabel(o.trackId, o.score));
		}
		double cx = (o.box[0] + o.box[2]) / 2, cy = (o.box[1] + o.box[3]) / 2;
		row.push_back(fixed(o.score, 3));
		for (double v : {cx, cy, o.box[2] - o.box[0], o.box[3] - o.box[1], o.box[0], o.box[2], o.box[3], o.box[1], o.vx, o.vy})
			row.push_back(fixed(v, 4));
		row.push_back(integer(o.lostFrames));
		row.push_back(integer(o.totalFrames));
		if (presenceOn_) row.push_back(fixed(o.presence, 4));
		if (labelsOn_) {
			row.push_back(o.label ? *o.label : std::string());
			row.push_back(fixed(o.gesture.score, 4));
			row.push_back(integer(o.gesture.frames));
			row.push_back(integer(o.gesture.onset));
		}
		for (auto& kp : o.keypoints)
			for (int i = 0; i < n; ++i) row.push_back(fixed(i < kp.n ? kp.v[i] : 0.0, 4));
		for (size_t k = 0; k < worldCount; ++k) {
			Vec3 p;
			if (o.hasWorld && k < o.world.size() && o.world[k].set) p = o.world[k].v;
			row.push_back(fixed(p.x, 4));
			row.push_back(fixed(p.y, 4));
			row.push_back(fixed(p.z, 4));
		}
		for (auto& name : extraNames) {
			const double* v = findExtra(o.extras, name);
			row.push_back(fixed(v ? *v : 0.0, 4));
		}
		appendColor(row, o.trackId);
		out.push_back(std::move(row));
	}
}

void PoseTracker::writeJoints(Table& out) const {
	out.clear();
	out.push_back({"track_id", "name", "x", "y", "z", "conf", "r", "g", "b"});
	const bool hide = skeleton_.hideMissing;
	for (auto& o : objects_) {
		std::string id = integer(o.trackId);
		auto c = trackColor(o.trackId);
		std::string r = fixed(c[0], 4), g = fixed(c[1], 4), b = fixed(c[2], 4);
		size_t count = std::min(skeleton_.names.size(), o.keypoints.size());
		for (size_t k = 0; k < count; ++k) {
			const Kp& kp = o.keypoints[k];
			if (!o.visible[k] || (hide && kp.v[2] <= 0.0)) continue;
			out.push_back({id, skeleton_.names[k], fixed(kp.v[0], 5), fixed(kp.v[1], 5), fixed(kp.n > 3 ? kp.v[3] : 0.0, 5),
				fixed(kp.v[2], 4), r, g, b});
		}
	}
}

void PoseTracker::writeBones(Table& out) const {
	out.clear();
	out.push_back({"track_id", "bx", "by", "bangle", "blen", "conf", "r", "g", "b"});
	const bool hide = skeleton_.hideMissing;
	for (auto& o : objects_) {
		std::string id = integer(o.trackId);
		auto c = trackColor(o.trackId);
		std::string r = fixed(c[0], 4), g = fixed(c[1], 4), b = fixed(c[2], 4);
		const auto& k = o.keypoints;
		for (auto [a, bi] : skeleton_.edges) {
			if (!o.visible[a] || !o.visible[bi] || (hide && (k[a].v[2] <= 0.0 || k[bi].v[2] <= 0.0))) continue;
			double ax = k[a].v[0], ay = k[a].v[1], aconf = k[a].v[2];
			double bx = k[bi].v[0], by = k[bi].v[1], bconf = k[bi].v[2];
			double dx = bx - ax, dy = by - ay;
			out.push_back({id, fixed((ax + bx) / 2.0, 5), fixed((ay + by) / 2.0, 5), fixed(pyDegrees(std::atan2(dy, dx)), 4),
				fixed(pyHypot(dx, dy), 5), fixed(std::min(aconf, bconf), 4), r, g, b});
		}
	}
}

}  // namespace tracking
