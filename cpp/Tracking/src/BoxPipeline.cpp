#include "BoxPipeline.h"

#include <algorithm>
#include <cmath>

namespace tracking {

namespace {

const char* const kFaceKeypoints[5] = {"eye_r", "eye_l", "nose", "mouth_r", "mouth_l"};

// object_tracker.reacquired: matched again after REACQUIRE_SNAP_FRAMES or more lost frames.
bool reacquired(std::map<long long, int>& gap, long long id, int lostFrames) {
	auto it = gap.find(id);
	const int previous = it == gap.end() ? 0 : it->second;
	gap[id] = lostFrames;
	return lostFrames == 0 && previous >= 2;
}

// object_tracker.box_smooth
std::array<double, 4>& boxSmooth(std::map<long long, std::array<double, 4>>& state, long long id, const double box[4],
		double smoothing, bool snap) {
	auto it = state.find(id);
	if (it == state.end() || snap) {
		auto& s = state[id];
		for (int k = 0; k < 4; ++k) s[k] = box[k];
		return s;
	}
	auto& s = it->second;
	for (int k = 0; k < 4; ++k) s[k] = s[k] * smoothing + box[k] * (1.0 - smoothing);
	return s;
}

// object_tracker.presence_ramp
double presenceRamp(std::map<long long, double>& state, long long id, bool present, double rise, double fall) {
	auto it = state.find(id);
	double current = it == state.end() ? 0.0 : it->second;
	const double step = 1.0 / std::max(present ? rise : fall, 1.0);
	current = present ? current + step : current - step;
	current = std::max(0.0, std::min(1.0, current));
	state[id] = current;
	return current;
}

template <class Map>
void pruneStale(Map& m, const std::vector<std::shared_ptr<BoxTrack>>& tracks) {
	for (auto it = m.begin(); it != m.end();) {
		bool active = false;
		for (auto& t : tracks) active |= t->trackId == it->first;
		it = active ? std::next(it) : m.erase(it);
	}
}

void boxRow(Row& row, double score, const double s[4], double vx, double vy, int lost, int total, double w, double h) {
	row.push_back(fixed(score, 3));
	row.push_back(fixed((s[0] + s[2]) / 2, 4));
	row.push_back(fixed((s[1] + s[3]) / 2, 4));
	row.push_back(fixed(w, 4));
	row.push_back(fixed(h, 4));
	row.push_back(fixed(s[0], 4));
	row.push_back(fixed(s[2], 4));
	row.push_back(fixed(s[3], 4));
	row.push_back(fixed(s[1], 4));
	row.push_back(fixed(vx, 4));
	row.push_back(fixed(vy, 4));
	row.push_back(integer(lost));
	row.push_back(integer(total));
}

double norm2(double x, double y) { return std::sqrt(x * x + y * y); }	// np.linalg.norm of a 2-vector

}  // namespace

void boxHeader(Row& row) {
	for (const char* c : {"score", "cx", "cy", "w", "h", "x_left", "x_right", "y_top", "y_bottom", "vx", "vy", "lost_frames",
			 "total_frames"})
		row.push_back(c);
}

void jointsHeader(Row& row) { row = {"track_id", "name", "x", "y", "z", "conf", "r", "g", "b"}; }

void bonesHeader(Row& row) { row = {"track_id", "bx", "by", "bangle", "blen", "conf", "r", "g", "b"}; }

// ---------------------------------------------------------------- BoxPipeline

std::vector<std::shared_ptr<const Detection>> BoxPipeline::filter(const std::vector<std::shared_ptr<const Detection>>& dets,
		double lowConf, double minW, double minH, double nmsIou, bool segmentation, double minMaskArea) {
	std::vector<std::shared_ptr<const Detection>> kept, out;
	for (auto& d : dets)
		if (d->score > lowConf && d->box[2] - d->box[0] >= minW && d->box[3] - d->box[1] >= minH) kept.push_back(d);
	if (!kept.empty())
		for (int i : nmsFloat32(kept, nmsIou)) out.push_back(kept[i]);
	if (segmentation) {
		std::vector<std::shared_ptr<const Detection>> masked;
		for (auto& d : out)
			if ((d->hasMaskArea ? d->maskArea : 1.0) > minMaskArea) masked.push_back(d);
		out = std::move(masked);
	}
	return out;
}

void BoxPipeline::update(const std::vector<std::shared_ptr<const Detection>>& dets, const BoxParams& p) {
	tracker_.highThresh = p.confThreshold;
	tracker_.lowThresh = p.lowConfThreshold;
	tracker_.matchThresh = p.trackIouThreshold;
	tracker_.trackBuffer = p.trackBuffer;
	tracker_.minHits = p.minHits;
	const auto& tracks = tracker_.update(dets);
	objects_.clear();
	for (auto& t : tracks) {
		const bool snap = reacquired(gap_, t->trackId, t->lostFrames);
		if (!t->confirmed) continue;
		if (!segmentation_ && t->score < p.confThreshold) continue;
		double kalman[4];
		t->box(kalman);
		const auto& s = boxSmooth(box_, t->trackId, kalman, p.smoothing, snap);
		Obj o;
		o.id = t->trackId;
		o.classId = t->payload->classId;
		o.className = t->payload->className;
		o.score = t->score;
		for (int k = 0; k < 4; ++k) o.box[k] = s[k];
		o.vx = t->mean[4];
		o.vy = t->mean[5];
		o.lost = t->lostFrames;
		o.total = t->totalFrames;
		if (segmentation_) {
			o.presence = presenceRamp(presence_, t->trackId, t->lostFrames == 0, p.presenceRise, p.presenceFall);
			o.maskArea = t->payload->hasMaskArea ? t->payload->maskArea : 0.0;
			o.detIndex = t->lostFrames == 0 ? t->payload->detIndex : -1;
		}
		objects_.push_back(std::move(o));
	}
	pruneStale(box_, tracks);
	pruneStale(presence_, tracks);
	pruneStale(gap_, tracks);
}

void BoxPipeline::writeTracks(Table& out) const {
	out.clear();
	Row header = {"track_id", "label", "class_id", "class_name"};
	boxHeader(header);
	if (segmentation_)
		for (const char* c : {"mask_area_ratio", "presence", "det_index"}) header.push_back(c);
	for (const char* c : {"r", "g", "b"}) header.push_back(c);
	out.push_back(std::move(header));
	for (const Obj& o : objects_) {
		Row row;
		const std::string text = integer(o.id) + " " + o.className + " " + integer(roundHalfEven(o.score * 100)) + "%";
		row.push_back(integer(o.id));
		row.push_back(buildLabel(o.id, o.score, &text));
		row.push_back(integer(o.classId));
		row.push_back(o.className);
		boxRow(row, o.score, o.box, o.vx, o.vy, o.lost, o.total, o.box[2] - o.box[0], o.box[3] - o.box[1]);
		if (segmentation_) {
			row.push_back(fixed(o.maskArea, 4));
			row.push_back(fixed(o.presence, 4));
			row.push_back(integer(o.detIndex));
		}
		appendColor(row, o.id);
		out.push_back(std::move(row));
	}
}

// ---------------------------------------------------------------- FacePipeline

std::vector<std::shared_ptr<const Detection>> FacePipeline::filter(const std::vector<std::shared_ptr<const Detection>>& dets,
		double lowConf, double minW, double minH) {
	std::vector<std::shared_ptr<const Detection>> out;
	for (auto& d : dets)
		if (d->score > lowConf && d->box[2] - d->box[0] >= minW && d->box[3] - d->box[1] >= minH) out.push_back(d);
	return out;
}

void headPoseGeometric(const double kp[5][2], double yawScale, double pitchScale, double out[3]) {
	const double* eyeR = kp[0];
	const double* eyeL = kp[1];
	const double* nose = kp[2];
	const double* mouthR = kp[3];
	const double* mouthL = kp[4];
	const double distR = norm2(nose[0] - eyeR[0], nose[1] - eyeR[1]) + norm2(nose[0] - mouthR[0], nose[1] - mouthR[1]);
	const double distL = norm2(nose[0] - eyeL[0], nose[1] - eyeL[1]) + norm2(nose[0] - mouthL[0], nose[1] - mouthL[1]);
	double yawRatio = (distR - distL) / std::max(distL + distR, 1e-6);
	const double eyeMid[2] = {(eyeR[0] + eyeL[0]) * 0.5, (eyeR[1] + eyeL[1]) * 0.5};
	const double mouthMid[2] = {(mouthR[0] + mouthL[0]) * 0.5, (mouthR[1] + mouthL[1]) * 0.5};
	const double distEyes = norm2(nose[0] - eyeMid[0], nose[1] - eyeMid[1]);
	const double distMouth = norm2(nose[0] - mouthMid[0], nose[1] - mouthMid[1]);
	double pitchRatio = (distEyes - distMouth) / std::max(distMouth + distEyes, 1e-6);
	const double roll = pyDegrees(std::atan2(eyeL[1] - eyeR[1], eyeL[0] - eyeR[0]));
	yawRatio = std::max(-1.0, std::min(1.0, yawRatio));
	pitchRatio = std::max(-1.0, std::min(1.0, pitchRatio));
	out[0] = yawRatio * yawScale;
	out[1] = pitchRatio * pitchScale;
	out[2] = roll;
}

void FacePipeline::update(const std::vector<std::shared_ptr<const Detection>>& dets, const FaceParams& p) {
	tracker_.highThresh = p.confThreshold;
	tracker_.lowThresh = p.lowConfThreshold;
	tracker_.matchThresh = p.trackIouThreshold;
	tracker_.trackBuffer = p.trackBuffer;
	tracker_.minHits = p.minHits;
	const auto& tracks = tracker_.update(dets);
	objects_.clear();
	for (auto& t : tracks) {
		const bool snap = reacquired(gap_, t->trackId, t->lostFrames);
		if (t->score < p.confThreshold || !t->confirmed) continue;
		double kalman[4];
		t->box(kalman);
		const auto& s = boxSmooth(box_, t->trackId, kalman, p.smoothing, snap);
		const auto& raw = t->payload->keypoints;
		if (t->lostFrames == 0 && int(raw.size()) >= kKeypoints) {
			const Pose* prev = nullptr;
			if (!snap) {
				auto it = pose_.find(t->trackId);
				if (it != pose_.end()) prev = &it->second;
			}
			Pose next;
			for (int k = 0; k < kKeypoints; ++k)
				for (int a = 0; a < 2; ++a)
					next.kp[k][a] = prev ? prev->kp[k][a] * p.smoothing + raw[k].v[a] * (1.0 - p.smoothing) : raw[k].v[a];
			double angles[3] = {0, 0, 0};
			if (p.solvePnP) {
				if (!headPoseSolvePnP(next.kp, p.frameW, p.frameH, p.focalScale, angles)) angles[0] = angles[1] = angles[2] = 0;
			} else {
				headPoseGeometric(next.kp, p.yawScale, p.pitchScale, angles);
			}
			next.yaw = angles[0];
			next.pitch = angles[1];
			next.roll = angles[2];
			if (prev) {
				next.yaw = prev->yaw * p.poseSmoothing + next.yaw * (1.0 - p.poseSmoothing);
				next.pitch = prev->pitch * p.poseSmoothing + next.pitch * (1.0 - p.poseSmoothing);
				next.roll = prev->roll * p.poseSmoothing + next.roll * (1.0 - p.poseSmoothing);
			}
			pose_[t->trackId] = next;
		}
		Obj o;
		o.id = t->trackId;
		o.score = t->score;
		for (int k = 0; k < 4; ++k) o.box[k] = s[k];
		o.vx = t->mean[4];
		o.vy = t->mean[5];
		o.lost = t->lostFrames;
		o.total = t->totalFrames;
		auto it = pose_.find(t->trackId);
		if (it != pose_.end())
			o.pose = it->second;
		else
			o.pose = Pose{};
		objects_.push_back(o);
	}
	pruneStale(box_, tracks);
	pruneStale(pose_, tracks);
	pruneStale(gap_, tracks);
}

void FacePipeline::writeTracks(Table& out) const {
	out.clear();
	Row header = {"track_id", "label"};
	boxHeader(header);
	for (const char* c : {"yaw", "pitch", "roll"}) header.push_back(c);
	for (const char* name : kFaceKeypoints) {
		header.push_back(std::string(name) + "_x");
		header.push_back(std::string(name) + "_y");
	}
	for (const char* c : {"r", "g", "b"}) header.push_back(c);
	out.push_back(std::move(header));
	for (const Obj& o : objects_) {
		Row row;
		row.push_back(integer(o.id));
		row.push_back(buildLabel(o.id, o.score));
		// size_aspect is 1 (OnnxTOP letterboxes): w / sqrt(1), h * sqrt(1).
		boxRow(row, o.score, o.box, o.vx, o.vy, o.lost, o.total, (o.box[2] - o.box[0]) / 1.0, (o.box[3] - o.box[1]) * 1.0);
		row.push_back(fixed(o.pose.yaw, 4));
		row.push_back(fixed(o.pose.pitch, 4));
		row.push_back(fixed(o.pose.roll, 4));
		for (int k = 0; k < kKeypoints; ++k)
			for (int a = 0; a < 2; ++a) row.push_back(fixed(o.pose.kp[k][a], 4));
		appendColor(row, o.id);
		out.push_back(std::move(row));
	}
}

void FacePipeline::writeJoints(Table& out) const {
	out.clear();
	Row header;
	jointsHeader(header);
	out.push_back(std::move(header));
	for (const Obj& o : objects_) {
		for (int k = 0; k < kKeypoints; ++k) {
			Row row = {integer(o.id), kFaceKeypoints[k], fixed(o.pose.kp[k][0], 5), fixed(o.pose.kp[k][1], 5), fixed(0.0, 5),
				fixed(1.0, 4)};
			appendColor(row, o.id);
			out.push_back(std::move(row));
		}
	}
}

}  // namespace tracking
