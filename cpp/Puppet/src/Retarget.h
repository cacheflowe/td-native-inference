// Retargeting tracked landmarks onto a rigged skeleton: a port of python/mixamo_retarget.py and
// python/hand_retarget.py (how it works: docs/PUPPETS.md). No TouchDesigner dependency.
//
// Conventions as in the Python: TD's rotate order xyz (local = Rz * Ry * Rx, column vectors),
// matrices row-major m[row][col], Euler angles in degrees. Results agree with numpy to rounding
// (tests/test_puppet_cpp.py), not bit for bit: numpy's matrix products sum in another order.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace puppet {

using Vec3 = std::array<double, 3>;

struct Mat3 {
	double m[3][3];
	static Mat3 identity();
};

Mat3 eulerXyz(double rx, double ry, double rz);
Vec3 toEulerXyz(const Mat3& m);
Mat3 slerp(const Mat3& a, const Mat3& b, double t);

// ---------------------------------------------------------------- rig

struct BoneDef {
	std::string name;
	std::string parent;	// empty: a root
	Vec3 t{}, r{};
	double s = 1.0;		// uniform scale (a Blender armature's 100), scales the children
};

// The rest pose, with forward kinematics done once. Bones are indexed in input order; `order`
// lists them parents first (breadth first from the roots, as the Python does).
class Rig {
public:
	explicit Rig(std::vector<BoneDef> bones);

	int find(const std::string& shortName) const;	// prefix + short, or -1
	const Vec3* pos(const std::string& shortName) const;
	std::string shortName(int bone) const;			// the name without the prefix

	std::vector<BoneDef> bones;
	std::unordered_map<std::string, int> index;
	std::string prefix;		// 'mixamorig_' from the root bone ending in Hips, else ''
	std::vector<int> order;
	std::vector<int> parent;	// -1 for roots (and parents that aren't bones)
	std::vector<Mat3> restLocal, worldRot;
	std::vector<Vec3> worldPos;
	std::vector<double> worldScale;
};

// ---------------------------------------------------------------- targets

// A landmark (b empty) or the midpoint of two; on the rig, bone short names.
struct Point {
	std::string a, b;
};

struct VecSpec {
	enum class Kind { None, Pair, Head, HandDir, HandWidth, Fixed };
	Kind kind = Kind::None;
	Point from, to;		// Pair
	std::string side;	// HandDir / HandWidth: 'left' / 'right'
	Vec3 dir{};			// Fixed (rest vectors only)
};

struct Target {
	std::string bone;	// short name
	VecSpec primary, hint, restPrimary, restHint;
	bool hinge = false;	// a finger joint: bend about `axis` and spread, clamped (degrees)
	VecSpec axis;
	double bend[2] = {-180, 180}, spread[2] = {0, 0};
};

struct Blend {
	std::string bone, a, b;
	double f;
};

const std::vector<Target>& mixamoTargets();
const std::vector<Blend>& mixamoBlends();

using Landmarks = std::unordered_map<std::string, Vec3>;

// {bone: local Euler degrees} for every bone, aligned with rig.bones (indexed like it); bones
// outside rig.order are left at {0, 0, 0}. positions (optional): each bone's posed position.
std::vector<Vec3> solve(const Rig& rig, const Landmarks& landmarks, bool mirror, std::vector<Vec3>* positions,
	const std::vector<Target>& targets, const std::vector<Blend>& blends);

// ---------------------------------------------------------------- placement

struct ImagePoint {
	double u, v, conf;
};
using ImagePoints = std::unordered_map<std::string, ImagePoint>;
using PlaceJoints = std::vector<std::pair<std::string, std::string>>;	// (keypoint, bone short name)

const PlaceJoints& mixamoPlaceJoints();

// Root translation laying the posed character over the image (mixamo_retarget.place); nullopt
// with fewer than three joints or a fit behind the camera.
std::optional<Vec3> place(const Rig& rig, const std::vector<Vec3>& positions, const ImagePoints& image, double aspect,
	double fov, const PlaceJoints& joints);

// ---------------------------------------------------------------- hand rigs (hand_retarget)

struct HandSpec {
	char side = 'R';	// which hand the model is
	std::vector<Target> targets;
	PlaceJoints placeJoints;
	std::string center[2];	// bones whose midpoint Center placement keeps at the origin
};

std::optional<HandSpec> handSpec(const Rig& rig);

}  // namespace puppet
