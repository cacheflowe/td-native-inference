// TrackerDAT: the tracker as a custom DAT. Three nodes per tracker, one per table (Table = Tracks /
// Joints / Bones), replacing the tracker COMP's Script DATs one for one. The Tracks node reads the
// Info DAT on its input, runs the tracker and keeps its state; Joints and Bones are wired after it
// and format the same frame's state, found through the Tracks node's id (an in-process registry).
// Two more tables replace wrapper Script DATs: Class Scores (HSEmotion's scores joined onto the
// tracks) and Convert RTMW3D (RTMW3D's Info DAT reshaped for the tracker). The Tracks and Class
// Scores nodes also publish t{id}_{column} for every numeric cell on their Info CHOP.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CPlusPlus_Common.h"
#include "DAT_CPlusPlusBase.h"
#include "Extras.h"
#include "Tracker.h"

using namespace TD;

class TrackerDAT : public DAT_CPlusPlusBase {
public:
	enum class Output { Tracks, Joints, Bones, Scores, ConvertRtmw };

	explicit TrackerDAT(const OP_NodeInfo* info);
	~TrackerDAT() override;

	void getGeneralInfo(DAT_GeneralInfo*, const OP_Inputs*, void*) override;
	void execute(DAT_Output*, const OP_Inputs*, void*) override;
	void setupParameters(OP_ParameterManager*, void*) override;
	void setParameterEnableStates(const OP_Inputs*, OP_ParEnableState*, void*) override;
	int32_t getNumInfoCHOPChans(void*) override;
	void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
	void getErrorString(OP_String* error, void*) override;
	void getWarningString(OP_String* warning, void*) override;
	void getInfoPopupString(OP_String* info, void*) override;

private:
	void cookTracks(const OP_Inputs* inputs);
	void cookFollower(tracking::TableKind kind, const OP_Inputs* inputs);
	void cookScores(const OP_Inputs* inputs);
	void cookConvertRtmw(const OP_Inputs* inputs);
	void write(DAT_Output* output, const tracking::Table& table);
	void buildChannels();

	uint32_t opId_;
	Output tableKind_ = Output::Tracks;
	std::unique_ptr<tracking::Tracker> tracker_;	// Tracks nodes only
	tracking::ClassScores scores_;	// Class Scores nodes only
	tracking::Table table_;
	tracking::Table written_;	// what the output holds (write() sets only changed cells)
	std::string error_, warning_;
	double cookMs_ = 0.0, trackMs_ = 0.0, inputMs_ = 0.0, writeMs_ = 0.0, cookAvg_ = -1.0;
	int numTracks_ = 0;
	long long cooks_ = 0;
	// Tracks nodes: the Joints / Bones nodes' average cook time, and this node's cook count when they
	// last cooked (a follower nothing pulls any more stops counting).
	double followerAvg_[2] = {-1.0, -1.0};
	long long followerAt_[2] = {-10, -10};
	bool channelsDirty_ = true;
	std::vector<std::pair<std::string, float>> channels_;
};
