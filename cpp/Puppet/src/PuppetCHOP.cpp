#include "PuppetCHOP.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace {

class DatTable : public puppet::TableView {
public:
	explicit DatTable(const OP_DATInput* dat) : dat_(dat) {}
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

const char* kHandNames[] = {"any", "right", "left"};
const char* kHandLabels[] = {"Any", "Right", "Left"};
const char* kPlacementNames[] = {"center", "image"};
const char* kPlacementLabels[] = {"Center", "Over Image"};
const char* kInfoNames[] = {"cook_ms", "solve_ms", "cook_ms_avg", "bones", "driven", "solved"};

void check(OP_ParAppendResult r) {
	if (r != OP_ParAppendResult::Success) throw std::runtime_error("parameter");
}

}  // namespace

extern "C" {

DLLEXPORT void FillCHOPPluginInfo(CHOP_PluginInfo* info) {
	if (!info->setAPIVersion(CHOPCPlusPlusAPIVersion)) return;
	info->customOPInfo.opType->setString("Puppet");
	info->customOPInfo.opLabel->setString("Puppet");
	info->customOPInfo.opIcon->setString("PUP");
	info->customOPInfo.authorName->setString("td-native-inference");
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 0;
}

DLLEXPORT CHOP_CPlusPlusBase* CreateCHOPInstance(const OP_NodeInfo* info) { return new PuppetCHOP(info); }

DLLEXPORT void DestroyCHOPInstance(CHOP_CPlusPlusBase* instance) { delete (PuppetCHOP*)instance; }
}

PuppetCHOP::PuppetCHOP(const OP_NodeInfo*) {}

void PuppetCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*) {
	// Cook whenever something reads the bones; an unchanged row costs a string compare.
	ginfo->cookEveryFrameIfAsked = true;
	ginfo->timeslice = false;
}

bool PuppetCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*) {
	const auto start = std::chrono::steady_clock::now();
	warning_.clear();
	const OP_DATInput* rest = inputs->getParDAT("Restpose");
	if (rest && rest->isTable) {
		DatTable t(rest);
		if (!solver_.setRest(&t) && rest->numRows > 1)
			warning_ = "Rest Pose DAT: needs columns name, parent, tx, ty, tz, rx, ry, rz (and s)";
	} else {
		solver_.clearRest();
	}
	puppet::Settings s;
	const int hand = inputs->getParInt("Hand");
	s.hand = kHandNames[hand >= 0 && hand < 3 ? hand : 0];
	s.mirror = inputs->getParInt("Mirror") != 0;
	s.overImage = inputs->getParInt("Placement") == 1;
	s.fov = inputs->getParDouble("Fov");
	s.placeSmoothing = inputs->getParDouble("Placesmoothing");
	s.cutoff = inputs->getParDouble("Smoothcutoff");
	s.speedCoefficient = inputs->getParDouble("Smoothspeed");
	s.slopeCutoff = std::max(0.01, inputs->getParDouble("Smoothslope"));
	s.twistCutoff = inputs->getParDouble("Twistcutoff");
	s.limits = inputs->getParInt("Kneelimits") != 0;
	s.relaxUnseen = inputs->getParInt("Relaxunseen") != 0;
	if (const OP_TimeInfo* time = inputs->getTimeInfo())
		if (time->rate > 0) s.time = double(time->absFrame) / time->rate;
	if (const OP_TOPInput* top = inputs->getParTOP("Image")) {
		s.imageW = top->textureDesc.width;
		s.imageH = top->textureDesc.height;
	}
	const OP_DATInput* tracks = inputs->getParDAT("Tracks");
	if (tracks && tracks->isTable) {
		DatTable t(tracks);
		solved_ = solver_.cook(&t, s);
	} else {
		solved_ = solver_.cook(nullptr, s);
	}
	if (warning_.empty()) warning_ = solver_.warning();
	info->numChannels = (int32_t)solver_.names().size();
	info->numSamples = 1;
	info->startIndex = 0;
	cookMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	cookAvg_ = cookAvg_ < 0 ? cookMs_ : cookAvg_ + (cookMs_ - cookAvg_) * 0.1;
	return true;
}

void PuppetCHOP::getChannelName(int32_t index, OP_String* name, const OP_Inputs*, void*) {
	const auto& names = solver_.names();
	name->setString(index >= 0 && index < (int32_t)names.size() ? names[index].c_str() : "chan");
}

void PuppetCHOP::execute(CHOP_Output* output, const OP_Inputs*, void*) {
	const auto& values = solver_.values();
	for (int32_t c = 0; c < output->numChannels && c < (int32_t)values.size(); ++c)
		for (int32_t i = 0; i < output->numSamples; ++i) output->channels[c][i] = (float)values[c];
}

int32_t PuppetCHOP::getNumInfoCHOPChans(void*) { return (int32_t)std::size(kInfoNames); }

void PuppetCHOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) {
	const double values[] = {cookMs_, solver_.solveMs(), std::max(cookAvg_, 0.0), double(solver_.numBones()),
		double(solver_.numDriven()), solved_ ? 1.0 : 0.0};
	if (index < 0 || index >= (int32_t)std::size(kInfoNames)) return;
	chan->name->setString(kInfoNames[index]);
	chan->value = (float)values[index];
}

void PuppetCHOP::getWarningString(OP_String* warning, void*) {
	if (!warning_.empty()) warning->setString(warning_.c_str());
}

void PuppetCHOP::getInfoPopupString(OP_String* info, void*) {
	std::string s = std::string("td-native-inference " TDNI_VERSION "\nPuppet CHOP, last cook: ") + std::to_string(cookMs_) +
		" ms\n" + std::to_string(solver_.numBones()) + " bones, " + std::to_string(solver_.numDriven()) + " driven" +
		(solver_.isHand() ? " (hand rig)" : "");
	info->setString(s.c_str());
}

void PuppetCHOP::setupParameters(OP_ParameterManager* manager, void*) {
	{
		OP_StringParameter sp("Tracks");
		sp.label = "Tracks DAT";
		sp.page = "Puppet";
		sp.help = "A tracker's tracks table with World Landmark Columns on (<joint>_wx/_wy/_wz).";
		check(manager->appendDAT(sp));
	}
	{
		OP_StringParameter sp("Restpose");
		sp.label = "Rest Pose DAT";
		sp.page = "Puppet";
		sp.help = "The rig's rest pose, captured from the FBX COMP's bones: name, parent, tx, ty, tz, rx, ry, rz, s.";
		check(manager->appendDAT(sp));
	}
	{
		OP_StringParameter sp("Hand");
		sp.label = "Tracked Hand";
		sp.page = "Puppet";
		sp.defaultValue = kHandNames[0];
		sp.help = "Hand rigs: which of the person's hands drives the model (Any: the one in view). The other side than "
			"the model's mirrors onto it.";
		check(manager->appendMenu(sp, 3, kHandNames, kHandLabels));
	}
	{
		OP_NumericParameter np("Mirror");
		np.label = "Mirror";
		np.page = "Puppet";
		np.defaultValues[0] = 0;
		np.help = "Body rigs, Center placement: swap left and right, for a mirrored (selfie) camera image.";
		check(manager->appendToggle(np));
	}
	{
		OP_StringParameter sp("Placement");
		sp.label = "Placement";
		sp.page = "Puppet";
		sp.defaultValue = kPlacementNames[0];
		sp.help = "Center: the model stays at the origin. Over Image: root_tx/ty/tz put it over the tracked person in the image.";
		check(manager->appendMenu(sp, 2, kPlacementNames, kPlacementLabels));
	}
	{
		OP_NumericParameter np("Fov");
		np.label = "Image FOV";
		np.page = "Puppet";
		np.defaultValues[0] = 50.0;
		np.minSliders[0] = 20.0;
		np.maxSliders[0] = 90.0;
		np.help = "Horizontal field of view assumed for the source camera; give the overlay camera the same.";
		check(manager->appendFloat(np));
	}
	{
		OP_StringParameter sp("Image");
		sp.label = "Image TOP";
		sp.page = "Puppet";
		sp.help = "The tracked image, for its aspect ratio.";
		check(manager->appendTOP(sp));
	}
	{
		OP_NumericParameter np("Placesmoothing");
		np.label = "Placement Smoothing";
		np.page = "Puppet";
		np.defaultValues[0] = 0.6;
		np.minSliders[0] = 0.0;
		np.maxSliders[0] = 0.99;
		np.help = "Smooths root_tx/ty/tz (depth from the image is the noisiest). 0 = raw.";
		check(manager->appendFloat(np));
	}
	// Smoothing: a One Euro filter (like the Filter CHOP's) on the world landmarks, the hand's points
	// relative to its wrist; then the hand-driven rolls on their own.
	struct Num {
		const char* name;
		const char* label;
		double value, lo, hi;
		const char* help;
	};
	const Num nums[] = {
		{"Smoothcutoff", "Landmark Cutoff (Hz)", 1.5, 0.0, 5.0,
			"One Euro filter on the tracked landmarks before solving: the cutoff frequency while still. Lower is smoother "
			"and slower. 0 = off."},
		{"Smoothspeed", "Speed Coefficient", 4.0, 0.0, 20.0,
			"How fast the cutoff rises with speed (metres per second): higher follows quick moves more closely."},
		{"Smoothslope", "Slope Cutoff (Hz)", 1.0, 0.1, 5.0, "Smooths the speed the filter reacts to."},
		{"Twistcutoff", "Hand Roll Cutoff (Hz)", 2.0, 0.0, 5.0,
			"Smooths the forearms', hands' and fingers' roll, which comes from the short knuckle line, and holds it "
			"while that line is unreliable (a hand seen edge-on, or not seen). 0 = off."},
	};
	for (const Num& n : nums) {
		OP_NumericParameter np(n.name);
		np.label = n.label;
		np.page = "Smoothing";
		np.defaultValues[0] = n.value;
		np.minSliders[0] = n.lo;
		np.maxSliders[0] = n.hi;
		np.minValues[0] = 0.0;
		np.clampMins[0] = true;
		np.help = n.help;
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Kneelimits");
		np.label = "Knee Limits";
		np.page = "Smoothing";
		np.defaultValues[0] = 1;
		np.help = "Knees bend like hinges, within a natural range: depth noise can't bend them backward or sideways.";
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Relaxunseen");
		np.label = "Relax Unseen Limbs";
		np.page = "Smoothing";
		np.defaultValues[0] = 1;
		np.help = "Joints the tracker reports as unseen (a person framed from the chest up): the hips go under the "
			"shoulders and the arms hang, instead of the character's rest pose. Legs stay straight.";
		check(manager->appendToggle(np));
	}
}
