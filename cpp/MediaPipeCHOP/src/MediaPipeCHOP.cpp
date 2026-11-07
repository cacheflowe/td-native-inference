#include "MediaPipeCHOP.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <numeric>
#include <set>

namespace
{
	// "Child speech, kid speaking" -> "Child_speech_kid_speaking": letters, digits and underscores,
	// no leading digit, as TD channel names expect.
	std::string channelName(const std::string& label)
	{
		std::string out;
		bool gap = false;
		for (unsigned char ch : label)
		{
			if (std::isalnum(ch))
			{
				if (gap && !out.empty())
					out += '_';
				out += (char)ch;
				gap = false;
			}
			else
				gap = true;
		}
		if (out.empty() || std::isdigit((unsigned char)out[0]))
			out = "c" + out;
		return out;
	}
}

extern "C"
{
DLLEXPORT void FillCHOPPluginInfo(CHOP_PluginInfo* info)
{
	if (!info->setAPIVersion(CHOPCPlusPlusAPIVersion))
		return;
	info->customOPInfo.opType->setString("Mediapipeaudio");
	info->customOPInfo.opLabel->setString("MediaPipe Audio");
	info->customOPInfo.opIcon->setString("MPA");
	info->customOPInfo.authorName->setString("td-native-inference");
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT CHOP_CPlusPlusBase* CreateCHOPInstance(const OP_NodeInfo* info)
{
	return new MediaPipeCHOP(info);
}

DLLEXPORT void DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	delete (MediaPipeCHOP*)instance;
}
}

MediaPipeCHOP::MediaPipeCHOP(const OP_NodeInfo*)
{
}

MediaPipeCHOP::~MediaPipeCHOP() = default;

void MediaPipeCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = true;	// audio keeps arriving even when nothing else changes
	ginfo->timeslice = false;
}

bool MediaPipeCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs*, void*)
{
	info->numChannels = (int32_t)myChannelNames.size();
	info->numSamples = 1;
	info->startIndex = 0;
	return true;
}

void MediaPipeCHOP::getChannelName(int32_t index, OP_String* name, const OP_Inputs*, void*)
{
	name->setString(index >= 0 && index < (int32_t)myChannelNames.size() ? myChannelNames[index].c_str() : "chan");
}

void MediaPipeCHOP::appendInput(const OP_CHOPInput* input)
{
	const double rate = input->sampleRate;
	if (rate != mySampleRate)
	{
		myBuffer.clear();
		myLastEnd = -1.0;
		mySinceSubmit = 0;
		mySampleRate = rate;
	}
	// Timesliced audio advances startIndex every cook: only the samples past last cook's end are
	// new. A jump (seek, restart, non-timesliced input) takes the whole input as new.
	const double start = input->startIndex;
	const double end = start + input->numSamples;
	int32_t first = 0;
	if (myLastEnd >= start && myLastEnd <= end)
		first = (int32_t)std::llround(myLastEnd - start);
	else if (myLastEnd > end)
		first = input->numSamples;	// the same (non-timesliced) input again: nothing new
	myLastEnd = std::max(myLastEnd, end);
	const int32_t channels = std::max(1, input->numChannels);
	for (int32_t s = first; s < input->numSamples; s++)
	{
		float sum = 0.0f;
		for (int32_t c = 0; c < input->numChannels; c++)
			sum += input->getChannelData(c)[s];
		myBuffer.push_back(sum / (float)channels);
	}
	mySinceSubmit += (size_t)std::max(0, input->numSamples - first);
}

void MediaPipeCHOP::updateChannelNames()
{
	if (myResult.names.size() == myChannelNames.size())
		return;
	myChannelNames.clear();
	std::set<std::string> used;
	for (size_t i = 0; i < myResult.names.size(); i++)
	{
		std::string name = channelName(myResult.names[i]);
		if (!used.insert(name).second)
			name += "_" + std::to_string(i);
		used.insert(name);
		myChannelNames.push_back(name);
	}
}

void MediaPipeCHOP::execute(CHOP_Output* output, const OP_Inputs* inputs, void*)
{
	myError.clear();
	myWarning.clear();
	const bool active = inputs->getParInt("Active") != 0;
	const double window = std::max(0.1, inputs->getParDouble("Window"));
	const double hop = std::max(0.02, inputs->getParDouble("Hop"));
	myTopCount = std::max(1, inputs->getParInt("Topresults"));
	const char* raw = inputs->getParString("Model");
	const std::string model = mpaudio::resolveModelPath(inputs->getParFilePath("Model"), raw, myModelHint);

	if (!active || model.empty() || myReload || (myClassifier && myClassifier->modelPath() != model))
	{
		myClassifier.reset();
		myReload = false;
	}
	if (!active)
	{
		myResult = mpaudio::Result();
		myBuffer.clear();
		myLastEnd = -1.0;
	}
	else if (model.empty())
		myWarning = "Set Model to an audio classifier .tflite (e.g. models/mediapipe/yamnet.tflite)";
	else if (!myClassifier)
		myClassifier = std::make_unique<mpaudio::Classifier>(model);

	const OP_CHOPInput* input = inputs->getInputCHOP(0);
	if (active && input && input->sampleRate > 0)
	{
		appendInput(input);
		const size_t windowSamples = (size_t)std::llround(window * mySampleRate);
		if (myBuffer.size() > windowSamples)
			myBuffer.erase(myBuffer.begin(), myBuffer.end() - windowSamples);
		// Classify once a full window has arrived, then every hop.
		if (myClassifier && myClassifier->loaded() && myBuffer.size() >= windowSamples &&
			mySinceSubmit >= (size_t)std::llround(hop * mySampleRate))
		{
			myClassifier->submit(myBuffer, mySampleRate);
			mySinceSubmit = 0;
		}
	}
	else if (active)
		myWarning = "Connect an audio CHOP (e.g. Audio Device In or Audio File In)";

	if (myClassifier)
	{
		myClassifier->latest(myResult);
		myError = myClassifier->error();
		if (!myClassifier->loaded() && myError.empty() && myWarning.empty())
			myWarning = "Loading " + model + "...";
	}
	updateChannelNames();
	for (int32_t c = 0; c < output->numChannels; c++)
		output->channels[c][0] = c < (int32_t)myResult.scores.size() ? myResult.scores[c] : 0.0f;
}

int32_t MediaPipeCHOP::getNumInfoCHOPChans(void*)
{
	return 6;
}

void MediaPipeCHOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
	const bool loaded = myClassifier && myClassifier->loaded();
	switch (index)
	{
		case 0: chan->name->setString("loaded");		chan->value = loaded ? 1.f : 0.f; break;
		case 1: chan->name->setString("loading");		chan->value = myClassifier && !loaded && myError.empty() ? 1.f : 0.f; break;
		case 2: chan->name->setString("infer_ms");		chan->value = (float)myResult.inferMs; break;
		case 3: chan->name->setString("results");		chan->value = (float)myResult.serial; break;
		case 4: chan->name->setString("dropped");		chan->value = myClassifier ? (float)myClassifier->dropped() : 0.f; break;
		case 5: chan->name->setString("sample_rate");	chan->value = (float)mySampleRate; break;
	}
}

bool MediaPipeCHOP::getInfoDATSize(OP_InfoDATSize* infoSize, void*)
{
	infoSize->rows = 1 + std::min(myTopCount, (int32_t)myResult.scores.size());
	infoSize->cols = 4;
	infoSize->byColumn = false;
	return true;
}

void MediaPipeCHOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	// Columns: rank, class (model index), label, score -- the top results, best first.
	if (index == 0)
	{
		const char* header[] = { "rank", "class", "label", "score" };
		for (int col = 0; col < nEntries && col < 4; col++)
			entries->values[col]->setString(header[col]);
		return;
	}
	std::vector<int> order(myResult.scores.size());
	std::iota(order.begin(), order.end(), 0);
	const int rank = index - 1;
	if (rank >= (int)order.size())
	{
		for (int col = 0; col < nEntries; col++)
			entries->values[col]->setString("");
		return;
	}
	std::partial_sort(order.begin(), order.begin() + rank + 1, order.end(),
		[this](int a, int b) { return myResult.scores[a] > myResult.scores[b]; });
	const int cls = order[rank];
	char buf[32];
	snprintf(buf, sizeof(buf), "%d", rank + 1);
	entries->values[0]->setString(buf);
	snprintf(buf, sizeof(buf), "%d", cls);
	entries->values[1]->setString(buf);
	entries->values[2]->setString(myResult.names[cls].c_str());
	snprintf(buf, sizeof(buf), "%.4f", myResult.scores[cls]);
	entries->values[3]->setString(buf);
}

void MediaPipeCHOP::getWarningString(OP_String* warning, void*)
{
	if (myError.empty() && !myWarning.empty())
		warning->setString(myWarning.c_str());
}

void MediaPipeCHOP::getErrorString(OP_String* error, void*)
{
	if (!myError.empty())
		error->setString((myModelHint.empty() ? myError : myError + "\n" + myModelHint).c_str());
}

void MediaPipeCHOP::getInfoPopupString(OP_String* info, void*)
{
	std::string s = "MediaPipe audio classifier (CPU, worker thread)\n" + std::to_string(myResult.scores.size()) +
		" classes, " + std::to_string((int)mySampleRate) + " Hz input\n";
	info->setString(s.c_str());
}

void MediaPipeCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	auto check = [](OP_ParAppendResult r) { assert(r == OP_ParAppendResult::Success); (void)r; };
	const char* page = "Audio";
	{
		OP_StringParameter sp("Model");
		sp.label = "Model (.tflite)";
		sp.page = page;
		sp.defaultValue = "models/mediapipe/yamnet.tflite";
		sp.help = "A MediaPipe audio classifier. YAMNet: 521 AudioSet sound classes. A relative path is "
			"looked up under the project folder, then next to the plugin.";
		check(manager->appendFile(sp));
	}
	{
		OP_NumericParameter np("Active");
		np.label = "Active";
		np.page = page;
		np.defaultValues[0] = 1.0;
		check(manager->appendToggle(np));
	}
	{
		OP_NumericParameter np("Reload");
		np.label = "Reload Model";
		np.page = page;
		check(manager->appendPulse(np));
	}
	{
		OP_NumericParameter np("Window");
		np.label = "Window (Seconds)";
		np.page = page;
		np.defaultValues[0] = 0.975;	// YAMNet's own analysis window
		np.minValues[0] = np.minSliders[0] = 0.1;
		np.maxSliders[0] = 5.0;
		np.clampMins[0] = true;
		np.help = "How much of the most recent audio each classification looks at. YAMNet analyses 0.975 s.";
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Hop");
		np.label = "Hop (Seconds)";
		np.page = page;
		np.defaultValues[0] = 0.25;
		np.minValues[0] = np.minSliders[0] = 0.02;
		np.maxSliders[0] = 2.0;
		np.clampMins[0] = true;
		np.help = "How often a new classification starts. Smaller = more responsive, more CPU.";
		check(manager->appendFloat(np));
	}
	{
		OP_NumericParameter np("Topresults");
		np.label = "Top Results (Info DAT)";
		np.page = page;
		np.defaultValues[0] = 5;
		np.minValues[0] = np.minSliders[0] = 1;
		np.maxSliders[0] = 20;
		np.clampMins[0] = true;
		check(manager->appendInt(np));
	}
}

void MediaPipeCHOP::pulsePressed(const char* name, void*)
{
	if (std::string(name) == "Reload")
		myReload = true;
}
