#include "Puppet.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace puppet {

namespace {

// Python's float(str) for a cell (correctly rounded); unparsable -> 0.
double num(const char* s) {
	if (!s) return 0.0;
	size_t n = std::strlen(s);
	while (n && (*s == ' ' || *s == '\t')) {
		++s;
		--n;
	}
	while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r' || s[n - 1] == '\n')) --n;
	if (n && *s == '+') {
		++s;
		--n;
	}
	double v = 0.0;
	auto r = std::from_chars(s, s + n, v);
	return r.ec == std::errc() ? v : 0.0;
}

bool endsWith(const std::string& s, const char* suffix) {
	const size_t n = std::strlen(suffix);
	return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool startsWith(const std::string& s, const std::string& prefix) { return s.compare(0, prefix.size(), prefix) == 0; }

std::string g17(double v) {
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%.17g", v);
	return buf;
}

const char* const kFingers[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};

}  // namespace

std::optional<std::vector<BoneDef>> readRestTable(const TableView& t) {
	if (t.numRows() < 1) return std::nullopt;
	const char* cols[] = {"name", "parent", "tx", "ty", "tz", "rx", "ry", "rz", "s"};
	int idx[9];
	for (int k = 0; k < 9; ++k) {
		idx[k] = -1;
		for (int c = 0; c < t.numCols(); ++c)
			if (std::strcmp(t.cell(0, c), cols[k]) == 0) idx[k] = c;
		if (idx[k] < 0 && k < 8) return std::nullopt;
	}
	std::vector<BoneDef> bones;
	for (int r = 1; r < t.numRows(); ++r) {
		BoneDef b;
		b.name = t.cell(r, idx[0]);
		if (b.name.empty()) continue;
		b.parent = t.cell(r, idx[1]);
		for (int k = 0; k < 3; ++k) {
			b.t[k] = num(t.cell(r, idx[2 + k]));
			b.r[k] = num(t.cell(r, idx[5 + k]));
		}
		b.s = idx[8] >= 0 ? num(t.cell(r, idx[8])) : 1.0;
		bones.push_back(std::move(b));
	}
	return bones;
}

bool Solver::setRest(const TableView* table) {
	if (!table) {
		clearRest();
		return false;
	}
	std::string text;
	for (int r = 0; r < table->numRows(); ++r) {
		for (int c = 0; c < table->numCols(); ++c) {
			text += table->cell(r, c);
			text += '\t';
		}
		text += '\n';
	}
	if (rig_ && text == restText_) return true;
	clearRest();
	restText_ = text;
	auto bones = readRestTable(*table);
	if (!bones || bones->empty()) return false;
	rig_ = std::make_unique<Rig>(std::move(*bones));
	hand_ = handSpec(*rig_);
	driven_ = 0;
	for (const Target& t : hand_ ? hand_->targets : mixamoTargets())
		if (rig_->find(t.bone) >= 0) ++driven_;
	for (const Blend& b : hand_ ? std::vector<Blend>{} : mixamoBlends())
		if (rig_->find(b.bone) >= 0) ++driven_;
	return true;
}

void Solver::clearRest() {
	rig_.reset();
	hand_.reset();
	restText_.clear();
	last_.reset();
	key_.reset();
	root_.reset();
	center_.reset();
	sideHeld_.clear();
	names_.clear();
	values_.clear();
	driven_ = 0;
}

// ---------------------------------------------------------------- tracks table

const Solver::Layout& Solver::layout(const TableView& t) {
	const int nc = t.numCols();
	const char* first = t.cell(0, 0);
	const char* last = t.cell(0, nc - 1);
	if (layout_.numCols == nc && layout_.first == first && layout_.last == last) return layout_;
	Layout l;
	l.numCols = nc;
	l.first = first;
	l.last = last;
	std::unordered_map<std::string, int> index;
	std::vector<std::string> header(nc);
	for (int c = 0; c < nc; ++c) {
		header[c] = t.cell(0, c);
		index[header[c]] = c;
	}
	auto col = [&](const std::string& name) {
		auto it = index.find(name);
		return it == index.end() ? -1 : it->second;
	};
	for (int c = 0; c < nc; ++c) {
		const std::string& name = header[c];
		if (endsWith(name, "_wx")) {
			const std::string key = name.substr(0, name.size() - 3);
			auto it = l.worldIndex.find(key);
			if (it == l.worldIndex.end()) {
				l.worldIndex[key] = (int)l.world.size();
				l.world.emplace_back(key, c);
			} else {
				l.world[it->second].second = c;
			}
		} else if (endsWith(name, "_x")) {
			const std::string key = name.substr(0, name.size() - 2);
			const int y = col(key + "_y");
			if (y >= 0) l.image.push_back({key, c, y, col(key + "_conf")});
		}
	}
	for (const auto& w : l.world) l.conf.push_back(col(w.first + "_conf"));
	l.lost = col("lost_frames");
	l.total = col("total_frames");
	l.handedness = col("handedness");
	l.trackId = col("track_id");
	layout_ = std::move(l);
	return layout_;
}

int Solver::pickRow(const TableView& t, const Layout& l, const std::vector<int>& rows) const {
	if (l.lost < 0 || l.total < 0 || rows.size() < 2) return rows[0];
	int best = rows[0];
	double bestLost = num(t.cell(best, l.lost)), bestTotal = -num(t.cell(best, l.total));
	for (size_t i = 1; i < rows.size(); ++i) {
		double lost = num(t.cell(rows[i], l.lost)), total = -num(t.cell(rows[i], l.total));
		if (lost < bestLost || (lost == bestLost && total < bestTotal)) {
			best = rows[i];
			bestLost = lost;
			bestTotal = total;
		}
	}
	return best;
}

Solver::ReadResult Solver::read(const TableView* t, const std::string* hand) {
	ReadResult out;
	if (!t || t->numRows() < 2) return out;
	const Layout& l = layout(*t);
	if (l.world.empty()) return out;
	std::vector<int> rows;
	for (int r = 1; r < t->numRows(); ++r) rows.push_back(r);
	std::string prefix = "pose_", side;
	if (hand && l.worldIndex.count("right_hand_wrist")) {	// Holistic: a column prefix per side
		const int row = pickRow(*t, l, rows);
		side = *hand;
		if (*hand == "any") {	// the hand seen, the right one first
			auto seen = [&](const std::string& s) {
				auto it = l.worldIndex.find(s + "_hand_wrist");
				if (it == l.worldIndex.end()) return false;
				const int conf = l.conf[it->second];
				return conf < 0 || num(t->cell(row, conf)) > 0.0;
			};
			side = !seen("right") && seen("left") ? "left" : "right";
		}
		prefix = side + "_hand_";
		rows = {row};
	} else if (hand) {	// a MediaPipe Hands table: one row per hand, by handedness
		prefix.clear();
		side = *hand == "any" ? "right" : *hand;
		if (l.handedness >= 0) {
			auto sideOf = [&](int r) { return std::nearbyint(num(t->cell(r, l.handedness))) == 0.0 ? "left" : "right"; };
			if (*hand != "any") {
				std::vector<int> kept;
				for (int r : rows)
					if (sideOf(r) == *hand) kept.push_back(r);
				rows = kept;
			}
			if (rows.empty()) return out;
			side = sideOf(pickRow(*t, l, rows));
		}
	}
	const int row = pickRow(*t, l, rows);
	for (int c = 0; c < t->numCols(); ++c) {
		if (c) out.line += '\t';
		out.line += t->cell(row, c);
	}
	const bool body = hand == nullptr;
	auto shortName = [&](const std::string& name, std::string& key) {
		if (startsWith(name, prefix)) {
			key = name.substr(prefix.size());
			return true;
		}
		key = name;
		return body;
	};
	std::string key;
	for (size_t i = 0; i < l.world.size(); ++i) {
		const auto& [name, c] = l.world[i];
		if (!shortName(name, key)) continue;
		const int conf = l.conf[i];
		if (conf < 0 || num(t->cell(row, conf)) > 0.0)
			out.world[key] = {num(t->cell(row, c)), num(t->cell(row, c + 1)), num(t->cell(row, c + 2))};
	}
	for (const auto& im : l.image) {
		if (!shortName(im.name, key) || !l.worldIndex.count(im.name)) continue;
		const double c = im.conf >= 0 ? num(t->cell(row, im.conf)) : 1.0;
		out.image[key] = {num(t->cell(row, im.x)), num(t->cell(row, im.y)), c};
	}
	out.side = side;
	if (l.trackId >= 0) out.trackId = t->cell(row, l.trackId);
	out.ok = true;
	return out;
}

// ---------------------------------------------------------------- unseen limbs

void Solver::relaxUnseen(Landmarks& world) {
	// World landmarks: metres, hip origin, y up, +z toward the camera.
	auto get = [&](const char* name) -> const Vec3* {
		auto it = world.find(name);
		return it == world.end() ? nullptr : &it->second;
	};
	// Values, not pointers into `world`: inserting into it can rehash and move every entry.
	const Vec3* lsp = get("left_shoulder");
	const Vec3* rsp = get("right_shoulder");
	if (!lsp || !rsp) return;
	const Vec3 ls = *lsp, rs = *rsp;
	const Vec3 mid = {(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2};
	// Hips: under the shoulders, a little narrower. The torso then turns with the shoulder line
	// (which also rolls the hips) instead of standing at the rig's rest.
	if (!get("left_hip") || !get("right_hip")) {
		for (const auto& [hip, s] : {std::pair<const char*, Vec3>{"left_hip", ls}, {"right_hip", rs}})
			world[hip] = {mid[0] + 0.6 * (s[0] - mid[0]), mid[1] + 0.6 * (s[1] - mid[1]) - 0.5, mid[2] + 0.6 * (s[2] - mid[2])};
	}
	// Arms: an elbow between the shoulder and a tracked wrist (bent a little down and back, the way
	// elbows bend), so the hand stays where it was tracked; with no wrist either, hanging from the
	// shoulder with the forearm a little forward (so the upper arm's roll is set).
	for (const auto& [side, p] : {std::pair<std::string, Vec3>{"left", ls}, {"right", rs}}) {
		if (!get((side + "_elbow").c_str())) {
			Vec3 elbow = {p[0], p[1] - 0.27, p[2]};
			if (const Vec3* w = get((side + "_wrist").c_str()))
				elbow = {(p[0] + (*w)[0]) / 2, (p[1] + (*w)[1]) / 2 - 0.06, (p[2] + (*w)[2]) / 2 - 0.04};
			world[side + "_elbow"] = elbow;
		}
		if (!get((side + "_wrist").c_str())) {
			const Vec3 e = world[side + "_elbow"];
			world[side + "_wrist"] = {e[0], e[1] - 0.24, e[2] + 0.06};
		}
	}
}

// ---------------------------------------------------------------- smoothing

void Solver::filter(ReadResult& r, const Settings& s) {
	if (s.cutoff <= 0.0 && s.twistCutoff <= 0.0) {
		filters_.clear();
		twist_.bones.clear();
		filterTime_ = -1.0;
		return;
	}
	double dt = 1.0 / 30.0;
	if (s.time >= 0.0 && filterTime_ >= 0.0) dt = s.time - filterTime_;
	// A new person, a gap of more than half a second or time going backward: start over.
	if (r.trackId != filterTrack_ || dt <= 0.0 || dt > 0.5) {
		filters_.clear();
		twist_.bones.clear();
		dt = 1.0 / 30.0;
	}
	filterTrack_ = r.trackId;
	filterTime_ = s.time;
	filterDt_ = dt;
	twist_.dt = dt;
	twist_.cutoff = s.twistCutoff;
	twist_.beta = s.twistSpeedCoefficient;
	twist_.slopeCutoff = s.slopeCutoff;
	if (s.cutoff <= 0.0) {
		filters_.clear();
		return;
	}
	// Landmarks not seen this time start fresh when they come back (a hand re-entering snaps).
	for (auto it = filters_.begin(); it != filters_.end();)
		it = r.world.count(it->first) ? std::next(it) : filters_.erase(it);
	// Hand points are filtered relative to their wrist: when the arm moves, every hand point moves
	// fast, which would open the filter (speed) and let the fingers' noise through. Relative to the
	// wrist only the hand's own shape moves; the wrist itself is filtered like the body.
	auto wristOf = [this](const std::string& name) -> std::string {
		if (hand_) return name == "wrist" ? std::string() : std::string("wrist");
		const size_t at = name.find("_hand_");
		if (at == std::string::npos || name.compare(at + 6, std::string::npos, "wrist") == 0) return {};
		return name.substr(0, at) + "_hand_wrist";
	};
	const Landmarks raw = r.world;
	auto step = [&](const std::string& name, Vec3 p) {
		auto& f = filters_[name];
		for (int k = 0; k < 3; ++k) p[k] = f[k].step(p[k], dt, s.cutoff, s.speedCoefficient, s.slopeCutoff);
		return p;
	};
	for (auto& [name, p] : r.world)
		if (wristOf(name).empty()) p = step(name, p);
	for (auto& [name, p] : r.world) {
		const std::string wrist = wristOf(name);
		if (wrist.empty()) continue;
		auto rw = raw.find(wrist);
		auto fw = r.world.find(wrist);
		if (rw == raw.end()) {	// no wrist this time: filter it as it is
			p = step(name, p);
			continue;
		}
		const Vec3 offset = step(name, {p[0] - rw->second[0], p[1] - rw->second[1], p[2] - rw->second[2]});
		p = {fw->second[0] + offset[0], fw->second[1] + offset[1], fw->second[2] + offset[2]};
	}
}

// ---------------------------------------------------------------- cook

void Solver::holdFingers(const Landmarks& world, std::vector<Vec3>& rotations, bool mirror) const {
	if (!last_) return;
	for (const auto& [side, mpSide] : {std::pair<std::string, std::string>{"Left", "left"}, {"Right", "right"}}) {
		std::string mp = mpSide;
		if (mirror) mp = mp == "left" ? "right" : "left";	// solve() swaps sides for a mirrored image
		if (world.count(mp + "_hand_wrist")) continue;
		const std::string prefix = rig_->prefix + side + "Hand";
		for (int i : rig_->order) {
			const std::string& name = rig_->bones[i].name;
			if (!startsWith(name, prefix)) continue;
			const std::string rest = name.substr(prefix.size());
			for (const char* f : kFingers)
				if (startsWith(rest, f)) {
					rotations[i] = (*last_)[i];
					break;
				}
		}
	}
}

bool Solver::cook(const TableView* tracks, const Settings& s) {
	warning_.clear();
	if (!rig_) {
		names_.clear();
		values_.clear();
		return true;
	}
	const auto start = std::chrono::steady_clock::now();
	const std::string* hand = hand_ ? &s.hand : nullptr;
	ReadResult r = read(tracks, hand);
	if (tracks && tracks->numRows() > 1 && layout(*tracks).world.empty())
		warning_ = "The tracks table has no world landmark columns: turn on World Landmark Columns on the tracker";
	// The same tracked row and settings as last cook (the tracker only moves on a new model
	// result): keep the channels as they are.
	std::string key = r.line + '\x1f' + (s.overImage ? "1" : "0") + s.hand + (s.mirror ? "1" : "0") + g17(s.fov) + ' ' +
		g17(s.placeSmoothing) + ' ' + std::to_string(s.imageW) + 'x' + std::to_string(s.imageH) + ' ' + g17(s.cutoff) + ' ' +
		g17(s.speedCoefficient) + ' ' + g17(s.slopeCutoff) + ' ' + g17(s.twistCutoff) + ' ' + g17(s.twistSpeedCoefficient) +
		(s.limits ? "L" : "") + (s.relaxUnseen ? "R" : "");
	if (r.ok && key_ && *key_ == key && !names_.empty()) {
		solveMs_ = 0.0;
		return false;
	}
	key_ = key;

	std::string side = r.side;
	if (hand_ && side.empty()) side = sideHeld_.empty() ? std::string(1, hand_->side) : sideHeld_;
	sideHeld_ = side;
	const bool flip = hand_ && std::toupper((unsigned char)side[0]) != hand_->side;
	const bool mirror = hand_ ? flip : (s.mirror && !s.overImage);
	std::vector<Vec3> positions, rotations;
	const bool solved = r.ok && !r.world.empty();
	if (solved) {
		if (s.relaxUnseen && !hand_) relaxUnseen(r.world);
		filter(r, s);
		TwistFilter* twist = s.twistCutoff > 0.0 ? &twist_ : nullptr;
		twist_.unseenHands.clear();
		if (twist && !hand_)
			for (const char* side : {"left", "right"}) {
				const std::string wrist = std::string(side) + "_hand_wrist";
				if (layout_.worldIndex.count(wrist) && !r.world.count(wrist))
					twist_.unseenHands.push_back(mirror ? (side[0] == 'l' ? "right" : "left") : side);
			}
		rotations = hand_ ? solve(*rig_, r.world, mirror, &positions, hand_->targets, {}, twist)
						  : solve(*rig_, r.world, mirror, &positions, mixamoTargets(s.limits), mixamoBlends(), twist);
		if (!hand_) holdFingers(r.world, rotations, mirror);
		last_ = rotations;
	} else if (last_) {	// nobody tracked: hold the last pose (rest before the first)
		rotations = *last_;
	} else {
		rotations.resize(rig_->bones.size());
		for (size_t i = 0; i < rig_->bones.size(); ++i) rotations[i] = rig_->bones[i].r;
	}

	std::optional<Vec3> root = root_;
	if (!s.overImage) {
		root_.reset();
		root.reset();
		auto ia = hand_ ? rig_->index.find(hand_->center[0]) : rig_->index.end();
		auto ib = hand_ ? rig_->index.find(hand_->center[1]) : rig_->index.end();
		if (hand_ && solved && ia != rig_->index.end() && ib != rig_->index.end()) {
			// A hand turns about its forearm: keep the palm at the origin instead.
			const Vec3& a = positions[ia->second];
			const Vec3& b = positions[ib->second];
			root = Vec3{-(a[0] + b[0]) / 2, -(a[1] + b[1]) / 2, -(a[2] + b[2]) / 2};
			center_ = root;
		} else if (hand_) {
			root = center_;
		}
	} else if (solved && !r.image.empty()) {
		const double aspect = s.imageH ? double(s.imageW) / double(s.imageH) : 16.0 / 9.0;
		if (flip)	// placed in the mirrored image; the render is flipped back
			for (auto& [name, p] : r.image) p.u = 1.0 - p.u;
		auto fit = place(*rig_, positions, r.image, aspect, s.fov, hand_ ? hand_->placeJoints : mixamoPlaceJoints());
		if (fit) {
			const double k = root ? std::min(std::max(s.placeSmoothing, 0.0), 0.99) : 0.0;
			const Vec3 from = root ? *root : *fit;
			Vec3 smoothed;
			for (int i = 0; i < 3; ++i) smoothed[i] = from[i] * k + (*fit)[i] * (1.0 - k);
			root = root_ = smoothed;
		}
	}
	write(rotations, root ? &*root : nullptr, flip);
	solveMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	return true;
}

void Solver::write(const std::vector<Vec3>& rotations, const Vec3* root, bool flip) {
	const size_t n = rig_->order.size() * 3 + 4;
	if (names_.size() != n) {
		names_.clear();
		static const char* axes[] = {"_rx", "_ry", "_rz"};
		for (int i : rig_->order)
			for (const char* a : axes) names_.push_back(rig_->bones[i].name + a);
		for (const char* name : {"root_tx", "root_ty", "root_tz", "mirror"}) names_.push_back(name);
	}
	values_.resize(n);
	size_t k = 0;
	for (int i : rig_->order)
		for (int a = 0; a < 3; ++a) values_[k++] = rotations[i][a];
	for (int a = 0; a < 3; ++a) values_[k++] = root ? (*root)[a] : 0.0;
	values_[k] = flip ? 1.0 : 0.0;
}

}  // namespace puppet
