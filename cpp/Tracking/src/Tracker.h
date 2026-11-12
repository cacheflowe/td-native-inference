// onnxtop_tables in C++: one tracker per TrackerDAT (or replay handle). Reads a plugin's Info DAT,
// advances the mode's pipeline and writes the tracks / joints / bones tables.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "BoxPipeline.h"
#include "Common.h"
#include "PoseTracker.h"

namespace tracking {

// Read access to an Info DAT (a TouchDesigner DAT input, or parsed text in the replay tool).
class InfoView {
public:
	virtual ~InfoView() = default;
	virtual int numRows() const = 0;
	virtual int numCols() const = 0;
	virtual const char* cell(int row, int col) const = 0;	// never null; "" when missing
};

class TextInfo : public InfoView {
public:
	explicit TextInfo(const std::string& text);
	int numRows() const override { return int(rows_.size()); }
	int numCols() const override { return rows_.empty() ? 0 : int(rows_[0].size()); }
	const char* cell(int row, int col) const override;

private:
	std::vector<std::vector<std::string>> rows_;
};

enum class TableKind { Tracks = 0, Joints = 1, Bones = 2 };

bool isSupportedMode(const std::string& mode);

class Tracker {
public:
	explicit Tracker(std::string mode);

	// Parameters by their TouchDesigner names (Confthreshold, Outputsmoothing, Facemesh, ...).
	void setPar(const std::string& name, double value) { pars_[name] = value; }
	void setMode(const std::string& mode);
	const std::string& mode() const { return mode_; }

	// The MediaPipeTOP's results / result_delay channels, when it has them (none = never skip).
	void setResultCounter(std::optional<double> results, std::optional<double> resultDelay);

	// One cook of the tracks / joints / bones table. Returns false when the table was left as it
	// was (the same result again while a newer one is on its way).
	bool cook(TableKind kind, const InfoView* info, Table& out);

	static void resetIds();

private:
	struct Layout {
		std::vector<std::string> header;
		std::unordered_map<std::string, int> col;
		int numKp = 0;
		std::vector<int> kp, kpZ, world;
		std::vector<std::pair<std::string, int>> extra;
	};

	double par(const char* name) const;
	const Skeleton& skeleton() const;
	const Layout& layout(const InfoView& info);
	std::string modeError(const InfoView* info);
	std::vector<std::shared_ptr<const Detection>> readDetections(const InfoView* info, int maxKeypoints);
	bool fresh(TableKind kind, const Table& out);
	bool isPoseMode() const;
	PoseTracker& pipeline();
	void cookBox(const InfoView* info, Table& out);
	void cookFace(const InfoView* info, Table& out);

	std::string mode_;
	std::map<std::string, double> pars_;
	std::optional<double> results_, resultDelay_;
	std::unique_ptr<PoseTracker> pose_;
	std::unique_ptr<BoxPipeline> box_;
	std::unique_ptr<FacePipeline> face_;
	std::string pipelineKey_;
	Layout layout_;
	bool layoutValid_ = false;
	// Per table: the (mode, result number) last written, like onnxtop_tables._written.
	std::map<int, std::pair<std::string, double>> written_;
};

}  // namespace tracking
