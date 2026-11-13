// The puppet CHOP's cook, without TouchDesigner: a port of python/puppet_solver.py (read + cook).
// A tracks table and the rig's rest-pose table in, channel names and values out.
#pragma once

#include "Retarget.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace puppet {

// A table's cells, row 0 the header ("" outside the table).
class TableView {
public:
	virtual ~TableView() = default;
	virtual int numRows() const = 0;
	virtual int numCols() const = 0;
	virtual const char* cell(int row, int col) const = 0;
};

struct Settings {
	std::string hand = "any";	// hand rigs: 'any', 'right', 'left'
	bool mirror = false;
	bool overImage = true;		// Placement: Over Image, else Center
	double fov = 50.0;
	double placeSmoothing = 0.6;
	int imageW = 0, imageH = 0;	// the Image TOP's size; 0 = none

	// Additions over the Python solver (all off reproduces it exactly):
	// One Euro filter on the world landmarks before solving (TD's Filter CHOP, One Euro type):
	// cutoff in Hz while still, rising by speedCoefficient * speed (m/s); slopeCutoff (Hz)
	// smooths that speed. Off when cutoff <= 0.
	double cutoff = 0.0, speedCoefficient = 0.0, slopeCutoff = 1.0;
	// Twist smoothing (TwistFilter): each bone's roll, One Euro on degrees. Off when <= 0.
	double twistCutoff = 0.0, twistSpeedCoefficient = 0.0;
	bool limits = false;	// knee hinges (mixamoTargets(true))
	// Body rigs: limbs the tracker reports as unseen (its joint gate: a person framed from the
	// chest up) relax instead of falling back to the rig's rest pose: hips under the shoulders,
	// arms hanging. Legs stay at rest (straight down from the hips).
	bool relaxUnseen = false;
	double time = -1.0;		// seconds (the cook's frame / rate); < 0 unknown: 1/30 s per solve
};

// Rest pose table: name, parent, tx, ty, tz, rx, ry, rz, s (header row first; parent "" for roots).
// Returns nullopt for a table without the columns.
std::optional<std::vector<BoneDef>> readRestTable(const TableView& table);

class Solver {
public:
	// The rig: rebuilt only when the rest table's text changes. Returns whether one is set.
	bool setRest(const TableView* table);
	void clearRest();

	// One cook. Returns false when the channels were kept as they were (same row and settings).
	bool cook(const TableView* tracks, const Settings& settings);

	const std::vector<std::string>& names() const { return names_; }
	const std::vector<double>& values() const { return values_; }
	const std::string& warning() const { return warning_; }
	bool hasRig() const { return rig_ != nullptr; }
	bool isHand() const { return hand_.has_value(); }
	int numBones() const { return rig_ ? (int)rig_->order.size() : 0; }
	int numDriven() const { return driven_; }
	double solveMs() const { return solveMs_; }

private:
	struct Layout {
		int numCols = -1;
		std::string first, last;
		std::vector<std::pair<std::string, int>> world;	// landmark (column prefix) -> its _wx column, header order
		std::unordered_map<std::string, int> worldIndex;	// landmark -> index in `world`
		std::vector<int> conf;		// per world landmark: its _conf column or -1
		struct Image {
			std::string name;
			int x, y, conf;
		};
		std::vector<Image> image;	// keypoints with _x and _y columns (and their _conf)
		int lost = -1, total = -1, handedness = -1, trackId = -1;
	};

	struct ReadResult {
		bool ok = false;
		Landmarks world;
		ImagePoints image;
		std::string side;	// hands: 'left' / 'right'; empty for a body
		std::string line;	// the row's text
		std::string trackId;
	};

	void filter(ReadResult& r, const Settings& s);
	static void relaxUnseen(Landmarks& world);

	const Layout& layout(const TableView& t);
	int pickRow(const TableView& t, const Layout& l, const std::vector<int>& rows) const;
	ReadResult read(const TableView* t, const std::string* hand);
	void holdFingers(const Landmarks& world, std::vector<Vec3>& rotations, bool mirror) const;
	void write(const std::vector<Vec3>& rotations, const Vec3* root, bool flip);

	std::unique_ptr<Rig> rig_;
	std::optional<HandSpec> hand_;
	std::string restText_;
	Layout layout_;

	// Per-instance state, as puppet_solver keeps it per Script CHOP path.
	std::optional<std::vector<Vec3>> last_;	// last solved rotations
	std::optional<Vec3> root_;				// smoothed Over Image root
	std::optional<Vec3> center_;			// last Center root of a hand
	std::string sideHeld_;					// the hand side last driven
	std::optional<std::string> key_;		// what the channels were last solved from
	std::unordered_map<std::string, std::array<OneEuro, 3>> filters_;	// per world landmark
	std::string filterTrack_;				// the track the filters follow
	double filterTime_ = -1.0;				// the last filtered solve's time
	double filterDt_ = 1.0 / 30.0;			// this solve's time step
	TwistFilter twist_;

	std::vector<std::string> names_;
	std::vector<double> values_;
	std::string warning_;
	int driven_ = 0;
	double solveMs_ = 0.0;
};

}  // namespace puppet
