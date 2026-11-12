#include "Tracker.h"

#include <cstring>

#include "Skeletons.gen.h"

namespace tracking {

// ---------------------------------------------------------------- TextInfo

TextInfo::TextInfo(const std::string& text) {
	size_t start = 0;
	while (start <= text.size()) {
		size_t end = text.find_first_of("\r\n", start);
		if (end == std::string::npos) end = text.size();
		std::string line = text.substr(start, end - start);
		if (!(line.empty() && end == text.size())) {
			std::vector<std::string> cells;
			size_t c = 0;
			while (true) {
				size_t tab = line.find('\t', c);
				cells.push_back(line.substr(c, tab == std::string::npos ? std::string::npos : tab - c));
				if (tab == std::string::npos) break;
				c = tab + 1;
			}
			rows_.push_back(std::move(cells));
		}
		if (end >= text.size()) break;
		start = end + ((text[end] == '\r' && end + 1 < text.size() && text[end + 1] == '\n') ? 2 : 1);
	}
}

const char* TextInfo::cell(int row, int col) const {
	if (row < 0 || row >= int(rows_.size()) || col < 0 || col >= int(rows_[row].size())) return "";
	return rows_[row][col].c_str();
}

// ---------------------------------------------------------------- modes and defaults

namespace {

struct ModeInfo {
	const char* mode;
	const Skeleton* skeleton;
	const char* task;	// MediaPipeTOP Task named in the mismatch error (MediaPipe modes)
};

const ModeInfo kModes[] = {
	{"pose", &COCO_17, nullptr},
	{"movenet", &COCO_17, nullptr},
	{"mp_pose", &MP_POSE, "Pose"},
	{"mp_hands", &MP_HANDS, "Hands or Gesture"},
	{"mp_face", &MP_FACE, "Face"},
	{"mp_holistic", &MP_HOLISTIC, "Holistic"},
	{"mp_facedet", &MP_FACE_DET, "Face Detector"},
	{"rtmw", &RTMW_BODY, "Pose Crops (the RTMW3D wrapper)"},
};

const ModeInfo* findMode(const std::string& mode) {
	for (auto& m : kModes)
		if (mode == m.mode) return &m;
	return nullptr;
}

bool isMediaPipe(const std::string& mode) { return mode != "pose" && mode != "movenet"; }

bool matchesNotExtra(const std::string& h) {
	// ^(kp\d+_[uvcz]|w\d+_[xyz]|m\d\d)$
	auto digits = [&](size_t from, size_t& to) {
		to = from;
		while (to < h.size() && h[to] >= '0' && h[to] <= '9') ++to;
		return to > from;
	};
	size_t e;
	if (h.size() >= 4 && h.compare(0, 2, "kp") == 0 && digits(2, e) && e + 2 == h.size() && h[e] == '_' &&
		std::strchr("uvcz", h[e + 1]) && h[e + 1])
		return true;
	if (h.size() >= 4 && h[0] == 'w' && digits(1, e) && e + 2 == h.size() && h[e] == '_' && std::strchr("xyz", h[e + 1]) &&
		h[e + 1])
		return true;
	if (h.size() == 3 && h[0] == 'm' && h[1] >= '0' && h[1] <= '9' && h[2] >= '0' && h[2] <= '9') return true;
	return false;
}

const char* const kKnownCols[] = {"id", "class", "conf", "u1", "v1", "u2", "v2", "label", "mask_fill"};

double cellDouble(const InfoView& info, int r, int c) {
	const char* s = info.cell(r, c);
	double v;
	parseDouble(s, std::strlen(s), v);
	return v;
}

}  // namespace

bool isSupportedMode(const std::string& mode) { return findMode(mode) != nullptr; }

Tracker::Tracker(std::string mode) : mode_(std::move(mode)) {}

void Tracker::setMode(const std::string& mode) { mode_ = mode; }

void Tracker::resetIds() { KeypointTrack::nextId = 1; }

void Tracker::setResultCounter(std::optional<double> results, std::optional<double> resultDelay) {
	results_ = results;
	resultDelay_ = resultDelay;
}

double Tracker::par(const char* name) const {
	auto it = pars_.find(name);
	if (it != pars_.end()) return it->second;
	const std::string n = name;
	// Per-mode defaults (onnxtop_tables.pose_defaults), then pose_tracker_pipeline's PARAMETERS.
	if (mode_ == "movenet") {
		if (n == "Confthreshold") return 0.1;
		if (n == "Outputsmoothing") return 0.4;
		if (n == "Minboxwidth" || n == "Minboxheight") return 0.05;
	} else if (const ModeInfo* m = findMode(mode_); m && isMediaPipe(mode_)) {
		if (mode_ == "mp_hands") {
			if (n == "Tracklossframes") return 6;
			if (n == "Coastwhilelost") return 0.8;
			if (n == "Presencefall") return 6;
		}
		if (n == "Confthreshold") return 0.1;
		if (n == "Outputsmoothing") return 0.0;
		if (n == "Trackconfirmframes") return 1;
		if (n == "Tracklossframes") return 3;
		if (n == "Minboxwidth" || n == "Minboxheight") return 0.01;
		if (n == "Maxmatchdist") return m->skeleton->maxMatchDist;
		if (n == "Relmatchdist") return m->skeleton->relMatchDist;
	}
	static const std::map<std::string, double> base = {
		{"Confthreshold", 0.005}, {"Outputsmoothing", 0.23}, {"Tracklossframes", 15}, {"Scalematch", 1},
		{"Relmatchdist", 0.15}, {"Maxmatchdist", 0.5}, {"Trackconfirmframes", 3}, {"Minboxwidth", 0.02},
		{"Minboxheight", 0.02}, {"Coastwhilelost", 0.0}, {"Presencerise", 3}, {"Presencefall", 3},
		{"Gestureholdframes", 3}, {"Gestureminscore", 0.5}, {"Facemesh", 1},
	};
	auto b = base.find(n);
	return b != base.end() ? b->second : 0.0;
}

const Skeleton& Tracker::skeleton() const {
	if (mode_ == "mp_holistic" && !par("Facemesh")) return MP_HOLISTIC_BODY;
	const ModeInfo* m = findMode(mode_);
	return m ? *m->skeleton : COCO_17;
}

PoseTracker& Tracker::pipeline() {
	const Skeleton& sk = skeleton();
	std::string key = mode_ + "|" + sk.name;
	if (!pose_ || pipelineKey_ != key) {
		pose_ = std::make_unique<PoseTracker>(sk);
		pipelineKey_ = key;
	}
	return *pose_;
}

const Tracker::Layout& Tracker::layout(const InfoView& info) {
	const int n = info.numRows() ? info.numCols() : 0;
	bool same = layoutValid_ && int(layout_.header.size()) == n;
	for (int c = 0; same && c < n; ++c) same = layout_.header[c] == info.cell(0, c);
	if (same) return layout_;
	Layout l;
	for (int c = 0; c < n; ++c) l.header.emplace_back(info.cell(0, c));
	for (int c = 0; c < n; ++c) l.col[l.header[c]] = c;
	for (auto& h : l.header)
		if (h.size() >= 4 && h.compare(0, 2, "kp") == 0 && h.compare(h.size() - 2, 2, "_u") == 0) ++l.numKp;
	auto find = [&](const std::string& name) {
		auto it = l.col.find(name);
		return it == l.col.end() ? -1 : it->second;
	};
	for (int k = 0; k < l.numKp; ++k) l.kp.push_back(find("kp" + std::to_string(k) + "_u"));
	if (find("kp0_z") >= 0)
		for (int k = 0; k < l.numKp; ++k) l.kpZ.push_back(find("kp" + std::to_string(k) + "_z"));
	for (int k = 0; k < l.numKp; ++k) {
		int c = find("w" + std::to_string(k) + "_x");
		if (c >= 0) l.world.push_back(c);
	}
	for (int c = 0; c < n; ++c) {
		const std::string& h = l.header[c];
		bool known = false;
		for (const char* k : kKnownCols) known |= h == k;
		if (!known && !matchesNotExtra(h)) l.extra.emplace_back(h, c);
	}
	layout_ = std::move(l);
	layoutValid_ = true;
	return layout_;
}

std::string Tracker::modeError(const InfoView* info) {
	if (!info || info->numRows() < 1) return {};
	const Layout& l = layout(*info);
	const ModeInfo* m = findMode(mode_);
	if (!m) return "Tracker Mode " + mode_ + " isn't supported by TrackerDAT yet";
	const int expected = m->skeleton->count();
	if (isMediaPipe(mode_) && l.numKp != expected)
		return "Tracker Mode " + mode_ + " needs a MediaPipeTOP Info DAT with " + std::to_string(expected) + " keypoints, got " +
			std::to_string(l.numKp) + " -- set the MediaPipeTOP's Task to " + m->task;
	if (l.numKp != expected && info->numRows() > 1)
		return "Tracker Mode is Pose but the model reports " + std::to_string(l.numKp) +
			" keypoints (needs 17) -- set Tracker Mode to Face (5 keypoints), Detect or Segment";
	return {};
}

std::vector<std::shared_ptr<const Detection>> Tracker::readDetections(const InfoView* info, int maxKeypoints) {
	std::vector<std::shared_ptr<const Detection>> dets;
	if (!info || info->numRows() < 2) return dets;
	const Layout& l = layout(*info);
	const bool mp = isMediaPipe(mode_);
	const ModeInfo* m = findMode(mode_);
	const int kpCount = std::min(int(l.kp.size()), maxKeypoints);
	const bool z = mp && !l.kpZ.empty();
	const int worldCount = mp ? std::min(int(l.world.size()), m->skeleton->worldCount) : 0;
	auto colOf = [&](const char* name) {
		auto it = l.col.find(name);
		return it == l.col.end() ? -1 : it->second;
	};
	const int box[4] = {colOf("u1"), colOf("v1"), colOf("u2"), colOf("v2")};
	const int conf = colOf("conf"), labelCol = colOf("label"), classCol = colOf("class");
	for (int r = 1; r < info->numRows(); ++r) {
		auto det = std::make_shared<Detection>();
		for (int i = 0; i < 4; ++i) det->box[i] = cellDouble(*info, r, box[i]);
		det->score = cellDouble(*info, r, conf);
		det->keypoints.resize(kpCount);
		for (int k = 0; k < kpCount; ++k) {
			Kp& kp = det->keypoints[k];
			int base = l.kp[k];
			kp.v[0] = cellDouble(*info, r, base);
			kp.v[1] = cellDouble(*info, r, base + 1);
			kp.v[2] = cellDouble(*info, r, base + 2);
			kp.n = 3;
			if (z) {
				kp.v[3] = cellDouble(*info, r, l.kpZ[k]);
				kp.n = 4;
			}
		}
		if (worldCount) {
			det->world.resize(worldCount);
			for (int k = 0; k < worldCount; ++k) {
				int b = l.world[k];
				det->world[k] = {cellDouble(*info, r, b), -cellDouble(*info, r, b + 1), -cellDouble(*info, r, b + 2)};
			}
		}
		if (mp) {
			if (labelCol >= 0 && *info->cell(r, labelCol)) {
				det->hasLabel = true;
				det->label = info->cell(r, labelCol);
			}
			det->hasExtras = true;
			for (auto& [name, c] : l.extra) {
				double v = cellDouble(*info, r, c);
				bool found = false;
				for (auto& e : det->extras)
					if (e.first == name) {
						e.second = v;
						found = true;
					}
				if (!found) det->extras.emplace_back(name, v);
			}
			if (mode_ == "mp_hands") {
				double v = cellDouble(*info, r, classCol);	// 0 = Left, 1 = Right
				bool found = false;
				for (auto& e : det->extras)
					if (e.first == "handedness") {
						e.second = v;
						found = true;
					}
				if (!found) det->extras.emplace_back("handedness", v);
			}
		}
		dets.push_back(std::move(det));
	}
	return dets;
}

bool Tracker::fresh(TableKind kind, const Table& out) {
	// onnxtop_tables._result_number: a counter only while a newer result is on its way.
	if (!results_ || !resultDelay_ || *resultDelay_ <= 0 || out.empty()) return true;
	std::pair<std::string, double> key(mode_, *results_);
	auto it = written_.find(int(kind));
	if (it != written_.end() && it->second == key) return false;
	written_[int(kind)] = key;
	return true;
}

bool Tracker::cook(TableKind kind, const InfoView* info, Table& out) {
	if (kind == TableKind::Tracks) {
		std::string err = modeError(info);
		if (!err.empty()) {
			out = {{"error"}, {err}};
			return true;
		}
	}
	if (!fresh(kind, out)) return false;
	PoseTracker& pose = pipeline();
	if (kind == TableKind::Joints) {
		pose.writeJoints(out);
		return true;
	}
	if (kind == TableKind::Bones) {
		pose.writeBones(out);
		return true;
	}
	auto dets = readDetections(info, pose.skeleton().count());
	PoseParams p;
	p.confThreshold = par("Confthreshold");
	p.smoothing = par("Outputsmoothing");
	p.maxMatchDist = par("Maxmatchdist");
	p.relativeMatchDist = par("Scalematch") ? par("Relmatchdist") : 0.0;
	p.trackBuffer = par("Tracklossframes");
	p.minHits = int(par("Trackconfirmframes"));
	const double minW = par("Minboxwidth"), minH = par("Minboxheight");
	std::vector<std::shared_ptr<const Detection>> kept;
	for (auto& d : dets)
		if (d->score > p.confThreshold && d->box[2] - d->box[0] >= minW && d->box[3] - d->box[1] >= minH) kept.push_back(d);
	if (isMediaPipe(mode_)) {
		p.coast = par("Coastwhilelost");
		p.presence = true;
		p.presenceRise = par("Presencerise");
		p.presenceFall = par("Presencefall");
	}
	if (mode_ == "mp_hands") {
		p.labels = true;
		p.labelHold = int(par("Gestureholdframes"));
		p.labelMinScore = par("Gestureminscore");
	}
	pose.update(kept, p);
	pose.writeTracks(out);
	return true;
}

}  // namespace tracking
