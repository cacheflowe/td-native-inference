// TrackerDAT: the tracker as a custom DAT. Three nodes per tracker, one per table (Table = Tracks /
// Joints / Bones), replacing the tracker COMP's Script DATs one for one. The Tracks node reads the
// Info DAT on its input, runs the tracker and keeps its state; Joints and Bones are wired after it
// and format the same frame's state, found through the Tracks node's id (an in-process registry).
#pragma once

#include <memory>
#include <string>

#include "CPlusPlus_Common.h"
#include "DAT_CPlusPlusBase.h"
#include "Tracker.h"

using namespace TD;

class TrackerDAT : public DAT_CPlusPlusBase {
public:
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
	void write(DAT_Output* output, const tracking::Table& table);

	uint32_t opId_;
	std::unique_ptr<tracking::Tracker> tracker_;	// Tracks nodes only
	tracking::Table table_;
	tracking::Table written_;	// what the output holds (write() sets only changed cells)
	std::string error_, warning_;
	double cookMs_ = 0.0, trackMs_ = 0.0, inputMs_ = 0.0;
	int numTracks_ = 0;
};
