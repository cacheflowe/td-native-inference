#include "TrackerDAT.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cstring>
#include <unordered_map>

using namespace tracking;

namespace {

const char* kTableNames[] = {"tracks", "joints", "bones"};
const char* kTableLabels[] = {"Tracks", "Joints", "Bones"};

const char* kModeNames[] = {"pose", "movenet", "mp_pose", "mp_hands", "mp_face", "mp_holistic", "mp_facedet", "rtmw"};
const char* kModeLabels[] = {"Pose (YOLO26-pose)", "Pose (MoveNet multipose)", "MediaPipe Pose (33)",
	"MediaPipe Hands / Gesture (21)", "MediaPipe Face (478)", "MediaPipe Holistic (553)", "MediaPipe Face Detector (6)",
	"RTMW3D Whole Body (75)"};

enum class Kind { Float, Toggle };

// pose_tracker_pipeline's PARAMETERS + PERSISTENCE / LABEL parameters + Holistic's Face Mesh. Defaults
// here are the Pose mode's; a mode's own defaults apply where the wrapper doesn't set a value (it binds
// every parameter to its tracker COMP, which builds the per-mode Tracking page).
struct Par {
	const char* name;
	const char* label;
	Kind kind;
	double def, min, max;
	bool clampMax;
	const char* help;
};

const Par kPars[] = {
	{"Confthreshold", "Confidence Threshold", Kind::Float, 0.005, 0.0, 0.05, false,
		"Minimum detection confidence for a person to be admitted to the tracker and to stay displayed."},
	{"Outputsmoothing", "Output Smoothing", Kind::Float, 0.23, 0.0, 1.0, true,
		"Per-keypoint smoothing (0 = raw, higher = steadier but laggier)."},
	{"Tracklossframes", "Track Loss Frames", Kind::Float, 15, 0.0, 90.0, false,
		"Model results a track survives without a match before it's dropped."},
	{"Scalematch", "Scale Match To Skeleton", Kind::Toggle, 1, 0, 1, true,
		"Match tracks by how far keypoints moved relative to the skeleton's size (Relative Match Distance). "
		"Off = the absolute Max Match Distance."},
	{"Relmatchdist", "Relative Match Distance", Kind::Float, 0.15, 0.0, 1.0, false,
		"How far the stable keypoints may move between results and still be the same track, as a fraction of "
		"the skeleton's box diagonal."},
	{"Maxmatchdist", "Max Match Distance", Kind::Float, 0.5, 0.0, 2.0, false,
		"Absolute match distance (UV), used with Scale Match To Skeleton off."},
	{"Trackconfirmframes", "Track Confirm Frames", Kind::Float, 3, 1.0, 30.0, false,
		"Matched results before a new track is output."},
	{"Minboxwidth", "Min Box Width", Kind::Float, 0.02, 0.0, 0.2, false, "Detections narrower than this (UV) are ignored."},
	{"Minboxheight", "Min Box Height", Kind::Float, 0.02, 0.0, 0.2, false, "Detections shorter than this (UV) are ignored."},
	{"Coastwhilelost", "Coast While Lost", Kind::Float, 0.0, 0.0, 1.0, true,
		"MediaPipe modes: while lost, keep moving along the last velocity, times this every lost result."},
	{"Presencerise", "Presence Rise Frames", Kind::Float, 3, 1.0, 60.0, false, "MediaPipe modes: presence ramp up."},
	{"Presencefall", "Presence Fall Frames", Kind::Float, 3, 1.0, 90.0, false, "MediaPipe modes: presence ramp down."},
	{"Gestureholdframes", "Gesture Hold Frames", Kind::Float, 3, 1.0, 30.0, false,
		"MediaPipe Hands: a new gesture must lead for this many results before the gesture column switches."},
	{"Gestureminscore", "Gesture Min Score", Kind::Float, 0.5, 0.0, 1.0, true, "MediaPipe Hands: weaker gestures count as None."},
	{"Facemesh", "Face Mesh", Kind::Toggle, 1, 0, 1, true,
		"MediaPipe Holistic: track and output the 478 face mesh points. Off: pose and hands only (75 points)."},
};

// Tracks nodes by op id, so the Joints / Bones nodes wired after one can read its tracker.
std::unordered_map<uint32_t, TrackerDAT*>& registry() {
	static std::unordered_map<uint32_t, TrackerDAT*> nodes;
	return nodes;
}

class DatInfo : public InfoView {
public:
	explicit DatInfo(const OP_DATInput* dat) : dat_(dat) {}
	int numRows() const override { return dat_->numRows; }
	int numCols() const override { return dat_->numCols; }
	const char* cell(int row, int col) const override {
		if (row < 0 || row >= dat_->numRows || col < 0 || col >= dat_->numCols) return "";
		const char* s = dat_->getCell(row, col);
		return s ? s : "";
	}

private:
	const OP_DATInput* dat_;
};

TableKind tableOf(const OP_Inputs* inputs) {
	const int t = inputs->getParInt("Table");
	return t == 1 ? TableKind::Joints : t == 2 ? TableKind::Bones : TableKind::Tracks;
}

}  // namespace

extern "C" {

DLLEXPORT void FillDATPluginInfo(DAT_PluginInfo* info) {
	if (!info->setAPIVersion(DATCPlusPlusAPIVersion)) return;
	info->customOPInfo.opType->setString("Tracker");
	info->customOPInfo.opLabel->setString("Tracker");
	info->customOPInfo.opIcon->setString("TRK");
	info->customOPInfo.authorName->setString("td-native-inference");
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT DAT_CPlusPlusBase* CreateDATInstance(const OP_NodeInfo* info) { return new TrackerDAT(info); }

DLLEXPORT void DestroyDATInstance(DAT_CPlusPlusBase* instance) { delete (TrackerDAT*)instance; }
}

TrackerDAT::TrackerDAT(const OP_NodeInfo* info) : opId_(info->opId) {}

TrackerDAT::~TrackerDAT() {
	auto& nodes = registry();
	auto it = nodes.find(opId_);
	if (it != nodes.end() && it->second == this) nodes.erase(it);
}

void TrackerDAT::getGeneralInfo(DAT_GeneralInfo* ginfo, const OP_Inputs*, void*) {
	ginfo->cookEveryFrame = false;
	ginfo->cookEveryFrameIfAsked = false;
}

void TrackerDAT::setupParameters(OP_ParameterManager* manager, void*) {
	{
		OP_StringParameter sp("Table");
		sp.label = "Table";
		sp.page = "Tracker";
		sp.defaultValue = kTableNames[0];
		sp.help = "Tracks runs the tracker on the Info DAT wired in. Joints / Bones: wire a Tracks Tracker DAT in; "
			"they write its joints and bones for the same result.";
		manager->appendMenu(sp, 3, kTableNames, kTableLabels);
	}
	{
		OP_StringParameter sp("Trackermode");
		sp.label = "Tracker Mode";
		sp.page = "Tracker";
		sp.defaultValue = kModeNames[0];
		sp.help = "The model's skeleton: must match the Info DAT (a mismatch writes an error row).";
		manager->appendStringMenu(sp, int32_t(std::size(kModeNames)), kModeNames, kModeLabels);
	}
	{
		OP_StringParameter sp("Resultschop");
		sp.label = "Results CHOP";
		sp.page = "Tracker";
		sp.help = "The plugin's Info CHOP (results, result_delay): with a MediaPipeTOP, the tracker advances once per "
			"new result instead of every cook. Empty = every cook.";
		manager->appendCHOP(sp);
	}
	for (const Par& p : kPars) {
		OP_NumericParameter np(p.name);
		np.label = p.label;
		np.page = "Tracking";
		np.help = p.help;
		np.defaultValues[0] = p.def;
		if (p.kind == Kind::Toggle) {
			manager->appendToggle(np);
			continue;
		}
		np.minValues[0] = np.minSliders[0] = p.min;
		np.maxValues[0] = np.maxSliders[0] = p.max;
		np.clampMins[0] = true;
		np.clampMaxes[0] = p.clampMax;
		manager->appendFloat(np);
	}
}

void TrackerDAT::setParameterEnableStates(const OP_Inputs* inputs, OP_ParEnableState* state, void*) {
	const bool tracks = tableOf(inputs) == TableKind::Tracks;
	state->setEnableState("Trackermode", tracks);
	state->setEnableState("Resultschop", tracks);
	for (const Par& p : kPars) state->setEnableState(p.name, tracks);
}

void TrackerDAT::execute(DAT_Output* output, const OP_Inputs* inputs, void*) {
	const auto start = std::chrono::steady_clock::now();
	error_.clear();
	warning_.clear();
	const TableKind kind = tableOf(inputs);
	if (kind == TableKind::Tracks)
		cookTracks(inputs);
	else
		cookFollower(kind, inputs);
	const auto written = std::chrono::steady_clock::now();
	trackMs_ = std::chrono::duration<double, std::milli>(written - start).count();
	write(output, table_);
	numTracks_ = kind == TableKind::Tracks && !table_.empty() && table_[0].size() > 1 ? int(table_.size()) - 1 : 0;
	cookMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void TrackerDAT::cookTracks(const OP_Inputs* inputs) {
	auto& nodes = registry();
	nodes[opId_] = this;
	const std::string mode = inputs->getParString("Trackermode");
	if (!isSupportedMode(mode)) {
		table_ = {{"error"}, {"Tracker Mode " + mode + " isn't supported by the Tracker DAT yet"}};
		error_ = table_[1][0];
		return;
	}
	if (!tracker_) tracker_ = std::make_unique<Tracker>(mode);
	tracker_->setMode(mode);
	for (const Par& p : kPars)
		tracker_->setPar(p.name, p.kind == Kind::Toggle ? double(inputs->getParInt(p.name)) : inputs->getParDouble(p.name));

	// onnxtop_tables._result_number: both channels, or neither (never skip).
	std::optional<double> results, delay;
	if (const OP_CHOPInput* chop = inputs->getParCHOP("Resultschop")) {
		for (int i = 0; i < chop->numChannels; ++i) {
			const char* name = chop->getChannelName(i);
			if (chop->numSamples < 1 || !name) continue;
			if (std::strcmp(name, "results") == 0) results = double(chop->getChannelData(i)[0]);
			if (std::strcmp(name, "result_delay") == 0) delay = double(chop->getChannelData(i)[0]);
		}
	}
	if (!results || !delay) results = delay = std::nullopt;
	tracker_->setResultCounter(results, delay);

	const auto t0 = std::chrono::steady_clock::now();
	const OP_DATInput* dat = inputs->getNumInputs() > 0 ? inputs->getInputDAT(0) : nullptr;
	inputMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	if (dat && !dat->isTable) {
		warning_ = "The input isn't a table (wire the plugin's Info DAT in)";
		dat = nullptr;
	}
	std::unique_ptr<DatInfo> info;
	if (dat) info = std::make_unique<DatInfo>(dat);
	tracker_->cook(TableKind::Tracks, info.get(), table_);
}

void TrackerDAT::cookFollower(TableKind kind, const OP_Inputs* inputs) {
	const auto t0 = std::chrono::steady_clock::now();
	const OP_DATInput* dat = inputs->getNumInputs() > 0 ? inputs->getInputDAT(0) : nullptr;
	inputMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	auto& nodes = registry();
	auto it = dat ? nodes.find(dat->opId) : nodes.end();
	if (it == nodes.end() || !it->second->tracker_) {
		table_.clear();
		error_ = "Wire a Tracker DAT with Table = Tracks into this one";
		return;
	}
	it->second->tracker_->cook(kind, nullptr, table_);
}

void TrackerDAT::write(DAT_Output* output, const Table& table) {
	// Most cells repeat from the last cook (header, ids, names, colors; everything on a skipped
	// result), and TouchDesigner keeps the output table between cooks: set only the cells that
	// changed. A different shape is written whole.
	output->setOutputDataType(DAT_OutDataType::Table);
	size_t cols = 0;
	for (const Row& row : table) cols = std::max(cols, row.size());
	int32_t rows = -1, cur = -1;
	output->getTableSize(&rows, &cur);
	const bool same = rows == int32_t(table.size()) && cur == int32_t(cols) && written_.size() == table.size();
	if (!same) {
		output->setTableSize(int32_t(table.size()), int32_t(cols));
		written_.assign(table.size(), Row());
	}
	for (size_t r = 0; r < table.size(); ++r) {
		const Row& row = table[r];
		Row& last = written_[r];
		if (same && last == row) continue;
		const size_t n = row.size();
		for (size_t c = 0; c < n; ++c)
			if (!same || c >= last.size() || last[c] != row[c]) output->setCellString(int32_t(r), int32_t(c), row[c].c_str());
		for (size_t c = n; same && c < last.size(); ++c) output->setCellString(int32_t(r), int32_t(c), "");
		last = row;
	}
}

int32_t TrackerDAT::getNumInfoCHOPChans(void*) { return 5; }

void TrackerDAT::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) {
	// cook_ms: the whole execute; track_ms: without writing the table; input_ms: getting the input DAT.
	// next_track_id: the process-wide id counter (the lab test starts the Python tracker's there).
	const char* names[] = {"cook_ms", "track_ms", "input_ms", "num_tracks", "next_track_id"};
	const double values[] = {cookMs_, trackMs_, inputMs_, double(numTracks_), double(KeypointTrack::nextId)};
	chan->name->setString(names[index]);
	chan->value = float(values[index]);
}

void TrackerDAT::getErrorString(OP_String* error, void*) {
	if (!error_.empty()) error->setString(error_.c_str());
}

void TrackerDAT::getWarningString(OP_String* warning, void*) {
	if (!warning_.empty()) warning->setString(warning_.c_str());
}

void TrackerDAT::getInfoPopupString(OP_String* info, void*) {
	std::string s = std::string("td-native-inference " TDNI_VERSION "\nTracker DAT, last cook: ") + std::to_string(cookMs_) + " ms";
	info->setString(s.c_str());
}
