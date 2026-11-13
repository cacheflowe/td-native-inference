#include "Retarget.h"

#include <algorithm>
#include <cmath>
#include <deque>

namespace puppet {

namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kDegToRad = kPi / 180.0;	// CPython's math.radians / math.degrees constants
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kHintFade[2] = {0.05, 0.3};

Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 mul(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 mid(const Vec3& a, const Vec3& b) { return {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
	return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
std::optional<Vec3> unit(const Vec3& v) {
	double n = norm(v);
	if (n > 1e-9) return Vec3{v[0] / n, v[1] / n, v[2] / n};
	return std::nullopt;
}

Mat3 mul(const Mat3& a, const Mat3& b) {
	Mat3 r;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
	return r;
}
Vec3 mul(const Mat3& a, const Vec3& v) {
	return {a.m[0][0] * v[0] + a.m[0][1] * v[1] + a.m[0][2] * v[2], a.m[1][0] * v[0] + a.m[1][1] * v[1] + a.m[1][2] * v[2],
		a.m[2][0] * v[0] + a.m[2][1] * v[1] + a.m[2][2] * v[2]};
}
Mat3 transpose(const Mat3& a) {
	Mat3 r;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[j][i];
	return r;
}
Mat3 columns(const Vec3& x, const Vec3& y, const Vec3& z) {
	return {{{x[0], y[0], z[0]}, {x[1], y[1], z[1]}, {x[2], y[2], z[2]}}};
}

using Quat = std::array<double, 4>;	// w, x, y, z

Quat normalized(Quat q) {
	double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

Quat quat(const Mat3& a) {
	const auto& m = a.m;
	double t = m[0][0] + m[1][1] + m[2][2];
	Quat q;
	if (t > 0) {
		double s = std::sqrt(t + 1.0) * 2;
		q = {0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s};
	} else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
		double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
		q = {(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s};
	} else if (m[1][1] > m[2][2]) {
		double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
		q = {(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s};
	} else {
		double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
		q = {(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s};
	}
	return normalized(q);
}

Mat3 matrix(const Quat& q) {
	double w = q[0], x = q[1], y = q[2], z = q[3];
	return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
		{2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
		{2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}

// Shortest rotation turning direction a onto direction b.
Mat3 swing(const Vec3& from, const Vec3& to) {
	Vec3 a = unit(from).value_or(Vec3{0, 0, 0}), b = unit(to).value_or(Vec3{0, 0, 0});
	Vec3 axis = cross(a, b);
	double s = norm(axis), c = dot(a, b);
	if (s < 1e-9) {
		if (c > 0) return Mat3::identity();
		// Opposite: half a turn about any axis perpendicular to a.
		Vec3 perp = std::abs(a[0]) < 0.9 ? unit(cross(a, {1.0, 0.0, 0.0})).value_or(Vec3{0, 1, 0})
										: unit(cross(a, {0.0, 1.0, 0.0})).value_or(Vec3{0, 0, 1});
		Mat3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) r.m[i][j] = 2.0 * perp[i] * perp[j] - (i == j ? 1.0 : 0.0);
		return r;
	}
	Vec3 k = {axis[0] / s, axis[1] / s, axis[2] / s};
	Mat3 kx = {{{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}};
	Mat3 kx2 = mul(kx, kx);
	Mat3 r;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) r.m[i][j] = (i == j ? 1.0 : 0.0) + s * kx.m[i][j] + (1 - c) * kx2.m[i][j];
	return r;
}

// Orthonormal frame (columns x, y, z), y along primary and x toward hint, plus the sine of the
// hint's angle to the primary.
std::optional<Mat3> frame(const Vec3& primary, const Vec3& hint, double& s) {
	s = 0.0;
	auto y = unit(primary);
	double hn = norm(hint);
	if (!y || hn < 1e-9) return std::nullopt;
	Vec3 x = sub(hint, mul(*y, dot(hint, *y)));
	s = norm(x) / hn;
	if (s < 1e-6) {
		s = 0.0;
		return std::nullopt;
	}
	double xn = norm(x);
	x = {x[0] / xn, x[1] / xn, x[2] / xn};
	return columns(x, *y, cross(x, *y));
}

Mat3 turn(const Vec3& restPrimary, const std::optional<Vec3>& restHint, const Vec3& primary,
	const std::optional<Vec3>& hint, const Mat3& base) {
	Mat3 plain = mul(swing(mul(base, restPrimary), primary), base);
	if (!hint || !restHint) return plain;
	double restS, s;
	auto restFrame = frame(restPrimary, *restHint, restS);
	auto target = frame(primary, *hint, s);
	if (!restFrame || !target) return plain;
	double weight = std::min(1.0, std::max(0.0, (std::min(s, restS) - kHintFade[0]) / (kHintFade[1] - kHintFade[0])));
	return slerp(plain, mul(*target, transpose(*restFrame)), weight);
}

Mat3 hinge(const Vec3& restPrimary, const Vec3& primary, const Vec3& restAxis, const Target& t, const Mat3& base) {
	Vec3 r = unit(restPrimary).value_or(Vec3{0, 1, 0});
	auto bu = unit(cross(restAxis, r));
	if (!bu) return mul(swing(mul(base, restPrimary), primary), base);
	Vec3 b = *bu;
	Vec3 a = cross(r, b);
	Vec3 d = mul(transpose(base), unit(primary).value_or(Vec3{0, 0, 0}));
	double bend = std::atan2(dot(d, b), dot(d, r));
	double spread = std::asin(std::max(-1.0, std::min(1.0, dot(d, a))));
	bend = std::min(std::max(bend, t.bend[0] * kDegToRad), t.bend[1] * kDegToRad);
	spread = std::min(std::max(spread, t.spread[0] * kDegToRad), t.spread[1] * kDegToRad);
	Vec3 dir = add(mul(add(mul(r, std::cos(bend)), mul(b, std::sin(bend))), std::cos(spread)), mul(a, std::sin(spread)));
	return mul(base, swing(r, dir));
}

// ---------------------------------------------------------------- landmark access

class Points {
public:
	explicit Points(const Landmarks& lm) : lm_(lm) {}

	std::optional<Vec3> get(const std::string& name) const {
		auto it = lm_.find(name);
		if (it == lm_.end()) return std::nullopt;
		return it->second;
	}

	std::optional<Vec3> get(const Point& p) const {
		if (p.b.empty()) return get(p.a);
		auto a = get(p.a), b = get(p.b);
		if (!a || !b) return std::nullopt;
		return mid(*a, *b);
	}

	std::optional<Vec3> vector(const VecSpec& spec) const {
		using K = VecSpec::Kind;
		switch (spec.kind) {
			case K::Pair: {
				auto a = get(spec.from), b = get(spec.to);
				if (!a || !b) return std::nullopt;
				return sub(*b, *a);
			}
			case K::Head: {
				auto r = get("right_ear"), l = get("left_ear"), nose = get("nose");
				if (!r || !l || !nose) return std::nullopt;
				return cross(sub(*nose, mid(*r, *l)), sub(*l, *r));
			}
			case K::HandDir: {
				const std::string hand = spec.side + "_hand_";
				auto wrist = get(hand + "wrist");
				if (!wrist) wrist = get(spec.side + "_wrist");
				auto tip = get(hand + "middle_finger_mcp");
				if (!tip) tip = get(Point{spec.side + "_index", spec.side + "_pinky"});
				if (!wrist || !tip) return std::nullopt;
				return sub(*tip, *wrist);
			}
			case K::HandWidth: {
				const std::string hand = spec.side + "_hand_";
				auto a = get(hand + "pinky_mcp");
				if (!a) a = get(spec.side + "_pinky");
				auto b = get(hand + "index_finger_mcp");
				if (!b) b = get(spec.side + "_index");
				if (!a || !b) return std::nullopt;
				return sub(*b, *a);
			}
			default:
				return std::nullopt;
		}
	}

private:
	const Landmarks& lm_;
};

std::optional<Vec3> restPoint(const Rig& rig, const Point& p) {
	if (p.b.empty()) {
		const Vec3* v = rig.pos(p.a);
		return v ? std::optional<Vec3>(*v) : std::nullopt;
	}
	const Vec3 *a = rig.pos(p.a), *b = rig.pos(p.b);
	if (!a || !b) return std::nullopt;
	return mid(*a, *b);
}

std::optional<Vec3> restVector(const Rig& rig, const VecSpec& spec) {
	if (spec.kind == VecSpec::Kind::Fixed) return spec.dir;
	if (spec.kind != VecSpec::Kind::Pair) return std::nullopt;
	auto a = restPoint(rig, spec.from), b = restPoint(rig, spec.to);
	if (!a || !b) return std::nullopt;
	return sub(*b, *a);
}

Landmarks mirrored(const Landmarks& lm) {
	Landmarks out;
	out.reserve(lm.size());
	for (const auto& [name, p] : lm) {
		std::string n = name;
		if (n.compare(0, 5, "left_") == 0)
			n = "right_" + n.substr(5);
		else if (n.compare(0, 6, "right_") == 0)
			n = "left_" + n.substr(6);
		out[n] = {-p[0], p[1], p[2]};
	}
	return out;
}

const Mat3* parentTurn(const Rig& rig, int bone, const std::unordered_map<std::string, Mat3>& turns) {
	for (int p = rig.parent[bone]; p >= 0; p = rig.parent[p]) {
		auto it = turns.find(rig.shortName(p));
		if (it != turns.end()) return &it->second;
	}
	return nullptr;
}

// Builders for the target tables.
VecSpec pair(const Point& from, const Point& to) {
	VecSpec v;
	v.kind = VecSpec::Kind::Pair;
	v.from = from;
	v.to = to;
	return v;
}
VecSpec pair(const std::string& from, const std::string& to) { return pair(Point{from, ""}, Point{to, ""}); }
VecSpec special(VecSpec::Kind kind, const std::string& side = "") {
	VecSpec v;
	v.kind = kind;
	v.side = side;
	return v;
}
VecSpec fixedDir(double x, double y, double z) {
	VecSpec v;
	v.kind = VecSpec::Kind::Fixed;
	v.dir = {x, y, z};
	return v;
}
Target target(const std::string& bone, VecSpec primary, VecSpec hint, VecSpec restPrimary, VecSpec restHint) {
	Target t;
	t.bone = bone;
	t.primary = std::move(primary);
	t.hint = std::move(hint);
	t.restPrimary = std::move(restPrimary);
	t.restHint = std::move(restHint);
	return t;
}

std::vector<Target> buildMixamoTargets() {
	const Point shoulders{"right_shoulder", "left_shoulder"}, hips{"right_hip", "left_hip"};
	const Point restShoulders{"RightArm", "LeftArm"}, restHips{"RightUpLeg", "LeftUpLeg"};
	std::vector<Target> t = {
		target("Hips", pair(hips, shoulders), pair("right_hip", "left_hip"), pair(restHips, restShoulders),
			pair("RightUpLeg", "LeftUpLeg")),
		target("Spine2", pair(hips, shoulders), pair("right_shoulder", "left_shoulder"), pair(restHips, restShoulders),
			pair("RightArm", "LeftArm")),
		target("Head", special(VecSpec::Kind::Head), pair("right_ear", "left_ear"), fixedDir(0, 1, 0), fixedDir(1, 0, 0)),
	};
	for (const auto& [side, mp] : {std::pair<std::string, std::string>{"Left", "left"}, {"Right", "right"}}) {
		const std::string hand = mp + "_hand_";
		t.push_back(target(side + "Arm", pair(mp + "_shoulder", mp + "_elbow"), pair(mp + "_elbow", mp + "_wrist"),
			pair(side + "Arm", side + "ForeArm"), fixedDir(0, 0, 1)));
		t.push_back(target(side + "ForeArm", pair(mp + "_elbow", mp + "_wrist"), special(VecSpec::Kind::HandWidth, mp),
			pair(side + "ForeArm", side + "Hand"), pair(side + "HandPinky1", side + "HandIndex1")));
		t.push_back(target(side + "Hand", special(VecSpec::Kind::HandDir, mp), special(VecSpec::Kind::HandWidth, mp),
			pair(side + "Hand", side + "HandMiddle1"), pair(side + "HandPinky1", side + "HandIndex1")));
		t.push_back(target(side + "UpLeg", pair(mp + "_hip", mp + "_knee"), pair(mp + "_heel", mp + "_foot_index"),
			pair(side + "UpLeg", side + "Leg"), pair(side + "Foot", side + "ToeBase")));
		t.push_back(target(side + "Leg", pair(mp + "_knee", mp + "_ankle"), pair(mp + "_heel", mp + "_foot_index"),
			pair(side + "Leg", side + "Foot"), pair(side + "Foot", side + "ToeBase")));
		t.push_back(target(side + "Foot", pair(mp + "_ankle", mp + "_foot_index"), pair("right_hip", "left_hip"),
			pair(side + "Foot", side + "ToeBase"), pair("RightUpLeg", "LeftUpLeg")));
		const VecSpec width = pair(hand + "pinky_mcp", hand + "index_finger_mcp");
		const VecSpec restWidth = pair(side + "HandPinky1", side + "HandIndex1");
		const std::pair<const char*, const char*> fingers[] = {
			{"Index", "index_finger"}, {"Middle", "middle_finger"}, {"Ring", "ring_finger"}, {"Pinky", "pinky"}};
		for (const auto& [finger, mpFinger] : fingers) {
			std::string joints[4];
			const char* names[4] = {"mcp", "pip", "dip", "tip"};
			for (int j = 0; j < 4; ++j) joints[j] = hand + mpFinger + "_" + names[j];
			for (int i = 0; i < 3; ++i) {
				const std::string bone = side + "Hand" + finger + std::to_string(i + 1);
				t.push_back(target(bone, pair(joints[i], joints[i + 1]), width,
					pair(bone, side + "Hand" + finger + std::to_string(i + 2)), restWidth));
			}
		}
		std::string thumb[4];
		const char* names[4] = {"cmc", "mcp", "ip", "tip"};
		for (int j = 0; j < 4; ++j) thumb[j] = hand + "thumb_" + names[j];
		for (int i = 0; i < 3; ++i) {
			const std::string bone = side + "HandThumb" + std::to_string(i + 1);
			t.push_back(target(bone, pair(thumb[i], thumb[i + 1]), pair(hand + "wrist", hand + "middle_finger_mcp"),
				pair(bone, side + "HandThumb" + std::to_string(i + 2)), pair(side + "Hand", side + "HandMiddle1")));
		}
	}
	return t;
}

}  // namespace

// ---------------------------------------------------------------- rotations

Mat3 Mat3::identity() { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; }

Mat3 eulerXyz(double rx, double ry, double rz) {
	double x = rx * kDegToRad, y = ry * kDegToRad, z = rz * kDegToRad;
	double cx = std::cos(x), sx = std::sin(x), cy = std::cos(y), sy = std::sin(y), cz = std::cos(z), sz = std::sin(z);
	return {{{cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx}, {sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx},
		{-sy, cy * sx, cy * cx}}};
}

Vec3 toEulerXyz(const Mat3& a) {
	const auto& m = a.m;
	double sy = -m[2][0], rx, ry, rz;
	if (std::abs(sy) < 0.99999) {
		rx = std::atan2(m[2][1], m[2][2]);
		ry = std::asin(sy);
		rz = std::atan2(m[1][0], m[0][0]);
	} else {	// gimbal lock: put it all in x
		rx = std::atan2(-m[1][2], m[1][1]);
		ry = std::copysign(kPi / 2, sy);
		rz = 0.0;
	}
	return {rx * kRadToDeg, ry * kRadToDeg, rz * kRadToDeg};
}

Mat3 slerp(const Mat3& a, const Mat3& b, double t) {
	if (t <= 0.0) return a;
	if (t >= 1.0) return b;
	Quat qa = quat(a), qb = quat(b);
	double d = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
	if (d < 0) {
		qb = {-qb[0], -qb[1], -qb[2], -qb[3]};
		d = -d;
	}
	Quat q;
	if (d > 0.9995) {
		for (int i = 0; i < 4; ++i) q[i] = qa[i] + t * (qb[i] - qa[i]);
	} else {
		double theta = std::acos(d), st = std::sin(theta);
		double wa = std::sin((1 - t) * theta), wb = std::sin(t * theta);
		for (int i = 0; i < 4; ++i) q[i] = (wa * qa[i] + wb * qb[i]) / st;
	}
	return matrix(normalized(q));
}

// ---------------------------------------------------------------- rig

Rig::Rig(std::vector<BoneDef> defs) : bones(std::move(defs)) {
	const int n = (int)bones.size();
	for (int i = 0; i < n; ++i) index.emplace(bones[i].name, i);
	for (const auto& b : bones) {
		const std::string suffix = "Hips";
		if (b.parent.empty() && b.name.size() >= suffix.size() &&
			b.name.compare(b.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
			prefix = b.name.substr(0, b.name.size() - suffix.size());
			break;
		}
	}
	parent.assign(n, -1);
	std::unordered_map<std::string, std::vector<int>> children;
	std::vector<int> roots;
	for (int i = 0; i < n; ++i) {
		if (bones[i].parent.empty())
			roots.push_back(i);
		else
			children[bones[i].parent].push_back(i);
	}
	std::deque<int> pending(roots.begin(), roots.end());
	while (!pending.empty()) {
		int i = pending.front();
		pending.pop_front();
		order.push_back(i);
		auto it = children.find(bones[i].name);
		if (it != children.end())
			for (int c : it->second) {
				parent[c] = i;
				pending.push_back(c);
			}
	}
	restLocal.assign(n, Mat3::identity());
	worldRot.assign(n, Mat3::identity());
	worldPos.assign(n, Vec3{0, 0, 0});
	worldScale.assign(n, 1.0);
	for (int i : order) {
		const auto& b = bones[i];
		restLocal[i] = eulerXyz(b.r[0], b.r[1], b.r[2]);
		const int p = parent[i];
		if (p < 0) {
			worldRot[i] = restLocal[i];
			worldPos[i] = b.t;
			worldScale[i] = b.s;
		} else {
			worldRot[i] = mul(worldRot[p], restLocal[i]);
			worldPos[i] = add(worldPos[p], mul(mul(worldRot[p], b.t), worldScale[p]));
			worldScale[i] = worldScale[p] * b.s;
		}
	}
}

int Rig::find(const std::string& shortName) const {
	auto it = index.find(prefix + shortName);
	return it == index.end() ? -1 : it->second;
}

const Vec3* Rig::pos(const std::string& shortName) const {
	int i = find(shortName);
	return i < 0 ? nullptr : &worldPos[i];
}

std::string Rig::shortName(int bone) const {
	const std::string& n = bones[bone].name;
	return n.size() >= prefix.size() ? n.substr(prefix.size()) : std::string();
}

// ---------------------------------------------------------------- solve

const std::vector<Target>& mixamoTargets() {
	static const std::vector<Target> targets = buildMixamoTargets();
	return targets;
}

const std::vector<Blend>& mixamoBlends() {
	static const std::vector<Blend> blends = {
		{"Spine", "Hips", "Spine2", 1.0 / 3}, {"Spine1", "Hips", "Spine2", 2.0 / 3}, {"Neck", "Spine2", "Head", 0.5}};
	return blends;
}

std::vector<Vec3> solve(const Rig& rig, const Landmarks& landmarks, bool mirror, std::vector<Vec3>* positions,
	const std::vector<Target>& targets, const std::vector<Blend>& blends) {
	Landmarks flipped;
	if (mirror) flipped = mirrored(landmarks);
	const Points points(mirror ? flipped : landmarks);
	std::unordered_map<std::string, Mat3> turns;
	const Mat3 eye = Mat3::identity();
	for (const Target& t : targets) {
		const int bone = rig.find(t.bone);
		if (bone < 0) continue;
		auto p = points.vector(t.primary), h = points.vector(t.hint);
		auto rp = restVector(rig, t.restPrimary), rh = restVector(rig, t.restHint);
		if (!p || !rp || !unit(*p)) continue;
		const Mat3* base = parentTurn(rig, bone, turns);
		if (t.hinge) {
			auto axis = restVector(rig, t.axis);
			turns[t.bone] = hinge(*rp, *p, axis.value_or(Vec3{0, 0, 0}), t, base ? *base : eye);
			continue;
		}
		turns[t.bone] = turn(*rp, rh, *p, h, base ? *base : eye);
	}
	for (const Blend& b : blends) {
		if (rig.find(b.bone) < 0) continue;
		auto a = turns.find(b.a), c = turns.find(b.b);
		if (a != turns.end() && c != turns.end()) turns[b.bone] = slerp(a->second, c->second, b.f);
	}

	const int n = (int)rig.bones.size();
	std::vector<Mat3> world(n, eye);
	std::vector<Vec3> out(n, Vec3{0, 0, 0});
	if (positions) positions->assign(n, Vec3{0, 0, 0});
	for (int i : rig.order) {
		const int p = rig.parent[i];
		const Mat3& parentWorld = p >= 0 ? world[p] : eye;
		auto it = turns.find(rig.shortName(i));
		Mat3 local;
		if (it != turns.end()) {
			world[i] = mul(it->second, rig.worldRot[i]);
			local = mul(transpose(parentWorld), world[i]);
		} else {
			local = rig.restLocal[i];
			world[i] = mul(parentWorld, local);
		}
		out[i] = toEulerXyz(local);
		if (positions) {
			const Vec3& t = rig.bones[i].t;
			(*positions)[i] = p >= 0 ? add((*positions)[p], mul(mul(parentWorld, t), rig.worldScale[p])) : t;
		}
	}
	return out;
}

// ---------------------------------------------------------------- placement

const PlaceJoints& mixamoPlaceJoints() {
	static const PlaceJoints joints = [] {
		PlaceJoints j;
		for (const auto& [side, mp] : {std::pair<std::string, std::string>{"Left", "left"}, {"Right", "right"}})
			for (const auto& [joint, bone] : {std::pair<std::string, std::string>{"shoulder", "Arm"}, {"elbow", "ForeArm"},
					 {"wrist", "Hand"}, {"hip", "UpLeg"}, {"knee", "Leg"}, {"ankle", "Foot"}})
				j.emplace_back(mp + "_" + joint, side + bone);
		return j;
	}();
	return joints;
}

std::optional<Vec3> place(const Rig& rig, const std::vector<Vec3>& positions, const ImagePoints& image, double aspect,
	double fov, const PlaceJoints& joints) {
	const double tanH = std::tan(fov * kDegToRad / 2);
	const double tanV = tanH / aspect;
	// Normal equations of the weighted least squares (3 unknowns).
	double ata[3][3] = {}, atb[3] = {};
	int rows = 0;
	for (const auto& [name, shortName] : joints) {
		auto it = image.find(name);
		const int bone = rig.find(shortName);
		if (it == image.end() || bone < 0 || bone >= (int)positions.size() || it->second.conf <= 0.0) continue;
		const Vec3& q = positions[bone];
		const double w = std::sqrt(it->second.conf);
		const double ndcs[2] = {2 * it->second.u - 1, 2 * it->second.v - 1}, tans[2] = {tanH, tanV};
		for (int axis = 0; axis < 2; ++axis) {
			// x_ndc * tan * -(qz + tz) = qx + tx, linear in t; the same for y.
			double row[3] = {0.0, 0.0, -ndcs[axis] * tans[axis]};
			row[axis] = -1.0;
			for (double& c : row) c *= w;
			const double rhs = (q[axis] + ndcs[axis] * tans[axis] * q[2]) * w;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) ata[i][j] += row[i] * row[j];
				atb[i] += row[i] * rhs;
			}
			++rows;
		}
	}
	if (rows < 6) return std::nullopt;
	// Gaussian elimination with partial pivoting.
	double a[3][4];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) a[i][j] = ata[i][j];
		a[i][3] = atb[i];
	}
	for (int c = 0; c < 3; ++c) {
		int pivot = c;
		for (int r = c + 1; r < 3; ++r)
			if (std::abs(a[r][c]) > std::abs(a[pivot][c])) pivot = r;
		if (std::abs(a[pivot][c]) < 1e-300) return std::nullopt;
		if (pivot != c)
			for (int j = 0; j < 4; ++j) std::swap(a[c][j], a[pivot][j]);
		for (int r = c + 1; r < 3; ++r) {
			double f = a[r][c] / a[c][c];
			for (int j = c; j < 4; ++j) a[r][j] -= f * a[c][j];
		}
	}
	Vec3 t;
	for (int i = 2; i >= 0; --i) {
		double s = a[i][3];
		for (int j = i + 1; j < 3; ++j) s -= a[i][j] * t[j];
		t[i] = s / a[i][i];
	}
	if (!(t[2] < 0)) return std::nullopt;	// behind the camera: no usable fit
	return t;
}

// ---------------------------------------------------------------- hand rigs

std::optional<HandSpec> handSpec(const Rig& rig) {
	for (char side : {'R', 'L'}) {
		const std::string s = std::string("_") + side;
		if (!rig.index.count("hand" + s) || !rig.index.count("pulse" + s)) continue;
		HandSpec spec;
		spec.side = side;
		spec.center[0] = "hand" + s;
		spec.center[1] = "middle_01" + s;
		const VecSpec width = pair("pinky_mcp", "index_finger_mcp");
		const VecSpec restWidth = pair("pinky_01" + s, "index_01" + s);
		// Fingers bend about the knuckle line; pinky -> index on a right hand puts cross(axis,
		// finger) on the palm side, so a positive bend curls the finger in.
		const VecSpec axis = side == 'R' ? restWidth : pair("index_01" + s, "pinky_01" + s);
		spec.targets.push_back(
			target("pulse" + s, pair("wrist", "middle_finger_mcp"), width, pair("hand" + s, "middle_01" + s), restWidth));
		static const double limits[3][4] = {{-30, 95, -25, 25}, {-5, 115, 0, 0}, {-5, 90, 0, 0}};
		const std::pair<const char*, const char*> fingers[] = {
			{"index", "index_finger"}, {"middle", "middle_finger"}, {"ring", "ring_finger"}, {"pinky", "pinky"}};
		for (const auto& [finger, mp] : fingers) {
			const char* names[4] = {"mcp", "pip", "dip", "tip"};
			std::string joints[4], rest[4];
			for (int j = 0; j < 4; ++j) joints[j] = std::string(mp) + "_" + names[j];
			for (int j = 0; j < 3; ++j) rest[j] = std::string(finger) + "_0" + std::to_string(j + 1) + s;
			rest[3] = std::string(finger) + "_03" + s + "_end";
			for (int i = 0; i < 3; ++i) {
				Target t = target(rest[i], pair(joints[i], joints[i + 1]), width, pair(rest[i], rest[i + 1]), restWidth);
				t.hinge = true;
				t.axis = axis;
				t.bend[0] = limits[i][0];
				t.bend[1] = limits[i][1];
				t.spread[0] = limits[i][2];
				t.spread[1] = limits[i][3];
				spec.targets.push_back(t);
			}
		}
		// Thumb: metacarpal, proximal, then the distal phalanx in two bones (both ip -> tip); rolled
		// by the palm's length.
		const std::pair<const char*, const char*> segments[4] = {
			{"thumb_cmc", "thumb_mcp"}, {"thumb_mcp", "thumb_ip"}, {"thumb_ip", "thumb_tip"}, {"thumb_ip", "thumb_tip"}};
		const std::string bones[4] = {"thumb_base" + s, "thumb_01" + s, "thumb_02" + s, "thumb_03" + s};
		const std::pair<std::string, std::string> rest[4] = {{"thumb_base" + s, "thumb_01" + s},
			{"thumb_01" + s, "thumb_02" + s}, {"thumb_02" + s, "thumb_03" + s + "_end"}, {"thumb_02" + s, "thumb_03" + s + "_end"}};
		for (int i = 0; i < 4; ++i)
			spec.targets.push_back(target(bones[i], pair(segments[i].first, segments[i].second),
				pair("wrist", "middle_finger_mcp"), pair(rest[i].first, rest[i].second), pair("hand" + s, "middle_01" + s)));
		spec.placeJoints = {{"wrist", "hand" + s}, {"thumb_cmc", "thumb_base" + s}, {"thumb_mcp", "thumb_01" + s},
			{"thumb_ip", "thumb_02" + s}, {"thumb_tip", "thumb_03" + s + "_end"}};
		for (const auto& [finger, mp] : fingers) {
			const std::string f(finger), m(mp);
			spec.placeJoints.insert(spec.placeJoints.end(), {{m + "_mcp", f + "_01" + s}, {m + "_pip", f + "_02" + s},
				{m + "_dip", f + "_03" + s}, {m + "_tip", f + "_03" + s + "_end"}});
		}
		return spec;
	}
	return std::nullopt;
}

}  // namespace puppet
