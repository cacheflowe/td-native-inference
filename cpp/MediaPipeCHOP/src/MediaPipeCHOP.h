// MediaPipeCHOP -- Google's MediaPipe audio tasks as a Custom CHOP. Input: any audio CHOP
// (timesliced, any sample rate; channels are mixed to mono). Task:
//
//   Classify Sounds    YAMNet (521 sound classes). Every Hop seconds the latest Window seconds are
//                      classified on a worker thread; one output channel per class ("Speech",
//                      "Music", "Dog", ...). The Info DAT lists the top results.
//   Transcribe Speech  streaming speech to text (Whisper, Moonshine, Parakeet, ... .litertlm). Speech
//                      is detected against the tracked noise floor; with Skip Silence only speech
//                      (plus a short lead-in) reaches the transcriber, which keeps models from
//                      inventing text from room noise. After speech and a pause (or Max Segment
//                      seconds) the stream is flushed so the sentence is finalized. The Info DAT
//                      holds the transcript; channels report speaking / new segments / backlog.
//   Speak Text         text to speech (Kokoro .litertlm), no input needed. Speak queues Text; a
//                      worker streams each line's audio and the CHOP plays it out as a 24 kHz
//                      timeslice (`audio`, plus `speaking`). The Info DAT lists the lines.
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
	void		setParameterEnableStates(const OP_Inputs* inputs, OP_ParEnableState* state, void*) override;

private:
	enum class Task { Classify = 0, Transcribe = 1, Speak = 2 };

	static Task	taskOf(const OP_Inputs* inputs);
	std::vector<float>	takeNewSamples(const OP_CHOPInput* input);	// new samples since last cook, mono
	void		classify(const OP_Inputs* inputs, const std::vector<float>& fresh, const std::string& model);
	void		transcribe(const OP_Inputs* inputs, const std::vector<float>& fresh, const std::string& model);
	void		speak(const OP_Inputs* inputs, const std::string& model);
	void		updateChannelNames();
	void		resetStream();

	Task						myTask = Task::Classify;
	bool						myReload = false;
	bool						myFlush = false;
	bool						myClear = false;

	// Classify Sounds
	std::unique_ptr<mpaudio::Classifier>	myClassifier;
	mpaudio::Result				myResult;
	std::vector<std::string>	myChannelNames;	// sanitized, unique
	std::vector<float>			myWindow;		// rolling mono buffer of the newest samples
	size_t						mySinceSubmit = 0;
	int							myTopCount = 5;
	std::vector<int>			myTopOrder;		// class indices of the top results, best first

	// Transcribe Speech
	std::unique_ptr<mpaudio::Transcriber>	myTranscriber;
	mpaudio::Transcript			myTranscript;
	uint64_t					myLastFinals = 0;	// the current transcriber's totalFinals at the last cook
	uint64_t					myTotalFinals = 0;	// keeps counting past Transcriber::kMaxFinals
	bool						myNewSegment = false;
	bool						mySpeaking = false;
	double						mySilentFor = 0.0;	// seconds below Silence Level since speech
	double						mySegmentFor = 0.0;	// seconds since the last flush, while speaking
	float						myLevel = 0.0f;
	float						myNoiseFloor = -1.0f;	// tracked input RMS between words; < 0 = not measured yet
	float						myThreshold = 0.0f;		// RMS that counts as speech this cook
	std::vector<float>			myPreroll;		// Skip Silence: the latest unsent audio, sent when speech starts

	// Speak Text
	std::unique_ptr<mpaudio::Speaker>	mySpeaker;
	mpaudio::SpeakerStatus		mySpeakerStatus;	// taken once per cook for the Info DAT / CHOP
	bool						mySpeakPulse = false;
	bool						myStopPulse = false;
	std::string					myLastText;		// Speak On Change compares against it
	bool						myHasLastText = false;

	// Input tracking
	double						mySampleRate = 0.0;
	double						myLastEnd = -1.0;

	std::string					myError;
	std::string					myWarning;
	std::string					myModelHint;
};
