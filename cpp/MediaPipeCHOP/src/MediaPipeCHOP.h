// MediaPipeCHOP -- Google's MediaPipe audio classifier (YAMNet: 521 sound classes) as a Custom CHOP.
//
// Input: any audio CHOP (timesliced, any sample rate; channels are mixed to mono). Every Hop seconds
// the latest Window seconds go to a worker thread; the newest scores come out as one channel per
// class, named after the class ("Speech", "Music", "Dog", ...). The Info DAT lists the top results.
#pragma once

#include "CHOP_CPlusPlusBase.h"
#include "MpAudio.h"

#include <memory>
#include <string>
#include <vector>

using namespace TD;

class MediaPipeCHOP : public CHOP_CPlusPlusBase
{
public:
	explicit MediaPipeCHOP(const OP_NodeInfo* info);
	~MediaPipeCHOP() override;

	void		getGeneralInfo(CHOP_GeneralInfo*, const OP_Inputs*, void*) override;
	bool		getOutputInfo(CHOP_OutputInfo*, const OP_Inputs*, void*) override;
	void		getChannelName(int32_t index, OP_String* name, const OP_Inputs*, void*) override;
	void		execute(CHOP_Output*, const OP_Inputs*, void*) override;

	int32_t		getNumInfoCHOPChans(void*) override;
	void		getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
	bool		getInfoDATSize(OP_InfoDATSize* infoSize, void*) override;
	void		getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*) override;
	void		getWarningString(OP_String* warning, void*) override;
	void		getErrorString(OP_String* error, void*) override;
	void		getInfoPopupString(OP_String* info, void*) override;
	void		setupParameters(OP_ParameterManager* manager, void*) override;
	void		pulsePressed(const char* name, void*) override;

private:
	void		appendInput(const OP_CHOPInput* input);
	void		updateChannelNames();

	std::unique_ptr<mpaudio::Classifier>	myClassifier;
	bool						myReload = false;
	mpaudio::Result				myResult;
	std::vector<std::string>	myChannelNames;	// sanitized, unique; fixed once the model reports its classes

	// Rolling mono buffer of the newest input samples, and where the input left off last cook.
	std::vector<float>			myBuffer;
	double						mySampleRate = 0.0;
	double						myLastEnd = -1.0;
	size_t						mySinceSubmit = 0;

	int							myTopCount = 5;
	std::string					myError;
	std::string					myWarning;
	std::string					myModelHint;
};
