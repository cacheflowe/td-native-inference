// PuppetCHOP: poses a rigged character from a tracks table (python/puppet_solver.py in C++, see
// docs/PUPPETS.md). Parameters name a Tracks DAT (World Landmark Columns on), a Rest Pose DAT
// (the rig captured from the FBX COMP: name, parent, tx, ty, tz, rx, ry, rz, s) and an Image TOP
// (its aspect, for Over Image). Output: one sample, <bone>_rx/ry/rz per bone, root_tx/ty/tz, mirror.
#pragma once

#include "CHOP_CPlusPlusBase.h"
#include "Puppet.h"

#include <string>

using namespace TD;

class PuppetCHOP : public CHOP_CPlusPlusBase {
public:
	explicit PuppetCHOP(const OP_NodeInfo* info);
	~PuppetCHOP() override = default;

	void getGeneralInfo(CHOP_GeneralInfo*, const OP_Inputs*, void*) override;
	bool getOutputInfo(CHOP_OutputInfo*, const OP_Inputs*, void*) override;
	void getChannelName(int32_t index, OP_String* name, const OP_Inputs*, void*) override;
	void execute(CHOP_Output*, const OP_Inputs*, void*) override;

	int32_t getNumInfoCHOPChans(void*) override;
	void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
	void getWarningString(OP_String* warning, void*) override;
	void getInfoPopupString(OP_String* info, void*) override;
	void setupParameters(OP_ParameterManager* manager, void*) override;

private:
	puppet::Solver solver_;
	std::string warning_;
	double cookMs_ = 0.0;
	double cookAvg_ = -1.0;
	bool solved_ = false;
};
