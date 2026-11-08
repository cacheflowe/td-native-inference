#include "MediaPipeCHOP.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <numeric>
#include <set>

namespace
{
	const char* kTaskNames[] = { "classify", "transcribe" };
	const char* kTaskLabels[] = { "Classify Sounds", "Transcribe Speech" };
	const char* kClassifyPars[] = { "Classifiermodel", "Window", "Hop", "Topresults" };
	const char* kTranscribePars[] = { "Transcribermodel", "Interim", "Prompt", "Skipsilence", "Autoflush",
		"Pause", "Maxsegment", "Silencelevel", "Noiseratio", "Flush", "Clear" };
	const char* kTranscribeChannels[] = { "speaking", "level", "segments", "new_segment", "backlog",
		"threshold" };
	const double kPrerollSeconds = 0.3;	// Skip Silence: audio kept from before speech was detected

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

	std::string number(double value, const char* format = "%.0f")
	{
		char buf[32];
		snprintf(buf, sizeof(buf), format, value);
		return buf;
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

bool MediaPipeCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	const bool transcribe = inputs->getParInt("Task") == (int)Task::Transcribe;
	info->numChannels = transcribe ? (int32_t)std::size(kTranscribeChannels) : (int32_t)myChannelNames.size();
	info->numSamples = 1;
	info->startIndex = 0;
	return true;
}

void MediaPipeCHOP::getChannelName(int32_t index, OP_String* name, const OP_Inputs* inputs, void*)
{
	if (inputs->getParInt("Task") == (int)Task::Transcribe)
		name->setString(index < (int32_t)std::size(kTranscribeChannels) ? kTranscribeChannels[index] : "chan");
	else
		name->setString(index >= 0 && index < (int32_t)myChannelNames.size() ? myChannelNames[index].c_str() : "chan");
}

std::vector<float> MediaPipeCHOP::takeNewSamples(const OP_CHOPInput* input)
{
	std::vector<float> fresh;
	const double rate = input->sampleRate;
	if (rate != mySampleRate)
	{
		resetStream();
		mySampleRate = rate;
	}
	// Timesliced audio advances startIndex every cook: only the samples past last cook's end are
	// new (the same non-timesliced input again gives nothing new). Any jump, forward or back (a
	// looping or seeking Audio File In restarts its index), takes the whole input.
	const double start = input->startIndex;
	const double end = start + input->numSamples;
	int32_t first = 0;
	if (myLastEnd >= start && myLastEnd <= end)
		first = (int32_t)std::llround(myLastEnd - start);
	myLastEnd = end;
	const float scale = 1.0f / (float)std::max(1, input->numChannels);
	fresh.reserve((size_t)std::max(0, input->numSamples - first));
	for (int32_t s = first; s < input->numSamples; s++)
	{
		float sum = 0.0f;
		for (int32_t c = 0; c < input->numChannels; c++)
			sum += input->getChannelData(c)[s];
		fresh.push_back(sum * scale);
	}
	return fresh;
}

void MediaPipeCHOP::resetStream()
{
	myWindow.clear();
	mySinceSubmit = 0;
	myLastEnd = -1.0;
	mySpeaking = false;
	mySilentFor = mySegmentFor = 0.0;
	myLevel = 0.0f;
	myNoiseFloor = -1.0f;
	myPreroll.clear();
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

void MediaPipeCHOP::classify(const OP_Inputs* inputs, const std::vector<float>& fresh, const std::string& model)
{
	if (myClassifier && myClassifier->modelPath() != model)
		myClassifier.reset();
	if (!myClassifier)
		myClassifier = std::make_unique<mpaudio::Classifier>(model);
	const double window = std::max(0.1, inputs->getParDouble("Window"));
	const double hop = std::max(0.02, inputs->getParDouble("Hop"));
	myTopCount = std::max(1, inputs->getParInt("Topresults"));
	myWindow.insert(myWindow.end(), fresh.begin(), fresh.end());
	mySinceSubmit += fresh.size();
	const size_t windowSamples = (size_t)std::llround(window * mySampleRate);
	if (myWindow.size() > windowSamples)
		myWindow.erase(myWindow.begin(), myWindow.end() - windowSamples);
	// Classify once a full window has arrived, then every hop.
	if (myClassifier->loaded() && windowSamples > 0 && myWindow.size() >= windowSamples &&
		mySinceSubmit >= (size_t)std::llround(hop * mySampleRate))
	{
		myClassifier->submit(myWindow, mySampleRate);
		mySinceSubmit = 0;
	}
	myClassifier->latest(myResult);
	myError = myClassifier->error();
	if (!myClassifier->loaded() && myError.empty())
		myWarning = "Loading " + model + "...";
}

void MediaPipeCHOP::transcribe(const OP_Inputs* inputs, const std::vector<float>& fresh, const std::string& model)
{
	const int rate = (int)std::lround(mySampleRate);
	const bool interim = inputs->getParInt("Interim") != 0;
	const std::string prompt = inputs->getParString("Prompt") ? inputs->getParString("Prompt") : "";
	if (myTranscriber && (myTranscriber->modelPath() != model || myTranscriber->sampleRate() != rate))
		myTranscriber.reset();
	if (myClear)
	{
		myTranscriber.reset();
		myTranscript = mpaudio::Transcript();
		myLastFinals = 0;
		myTotalFinals = 0;
		myClear = false;
	}
	if (!myTranscriber && rate > 0)
	{
		myTranscriber = std::make_unique<mpaudio::Transcriber>(model, rate, interim, prompt);
		myLastFinals = 0;
	}
	if (!myTranscriber)
		return;
	// Speech detection: input RMS above Noise Ratio x the noise floor (and at least Silence Level).
	// The floor follows quiet stretches quickly and louder ones slowly, so a steady noise source
	// (a fan, a hum) raises it over tens of seconds without speech doing the same.
	double sum = 0.0;
	for (float v : fresh)
		sum += (double)v * v;
	myLevel = fresh.empty() ? myLevel : (float)std::sqrt(sum / fresh.size());
	const double seconds = mySampleRate > 0 ? fresh.size() / mySampleRate : 0.0;
	if (!fresh.empty())
	{
		if (myNoiseFloor < 0.0f)
			myNoiseFloor = myLevel;
		const double tau = myLevel < myNoiseFloor ? 0.25 : (mySpeaking ? 60.0 : 10.0);
		myNoiseFloor += (myLevel - myNoiseFloor) * (float)std::min(1.0, seconds / tau);
	}
	myThreshold = std::max((float)inputs->getParDouble("Silencelevel"),
		std::max(0.0f, myNoiseFloor) * (float)inputs->getParDouble("Noiseratio"));
	const bool voiced = myLevel > myThreshold;
	if (voiced)
	{
		mySpeaking = true;
		mySilentFor = 0.0;
	}
	else if (mySpeaking)
		mySilentFor += seconds;
	if (mySpeaking)
		mySegmentFor += seconds;

	// Skip Silence: between utterances audio waits in a short pre-roll (so the first syllable isn't
	// cut) instead of reaching the model, which would otherwise transcribe noise into text.
	const bool skipSilence = inputs->getParInt("Skipsilence") != 0;
	if (!skipSilence || mySpeaking)
	{
		if (!myPreroll.empty())
		{
			myTranscriber->submit(myPreroll.data(), myPreroll.size());
			myPreroll.clear();
		}
		myTranscriber->submit(fresh.data(), fresh.size());
	}
	else
	{
		myPreroll.insert(myPreroll.end(), fresh.begin(), fresh.end());
		const size_t keep = (size_t)std::llround(kPrerollSeconds * mySampleRate);
		if (myPreroll.size() > keep)
		{
			const size_t drop = myPreroll.size() - keep;
			myTranscriber->skip(drop);
			myPreroll.erase(myPreroll.begin(), myPreroll.begin() + drop);
		}
	}

	// Pause seconds of silence after speech (or Max Segment seconds of continuous speech) flush
	// the stream so the sentence comes out as a final segment.
	const bool autoflush = inputs->getParInt("Autoflush") != 0;
	const bool paused = mySpeaking && mySilentFor >= inputs->getParDouble("Pause");
	const bool tooLong = autoflush && mySegmentFor >= std::max(1.0, inputs->getParDouble("Maxsegment"));
	if (paused)
		mySpeaking = false;
	if (myFlush || tooLong || (paused && (autoflush || skipSilence)))
	{
		myTranscriber->finishStream();
		myFlush = false;
		mySegmentFor = 0.0;
	}

	myNewSegment = false;
	if (myTranscriber->latest(myTranscript))
	{
		// Count from the transcriber's running total: finals.size() stops growing at kMaxFinals.
		const uint64_t finals = myTranscript.totalFinals;
		if (finals > myLastFinals)
		{
			myTotalFinals += finals - myLastFinals;
			myNewSegment = true;
		}
		myLastFinals = finals;
	}
	myError = myTranscriber->error();
	if (!myTranscriber->loaded() && myError.empty())
		myWarning = "Loading " + model + "...";
	else if (myTranscriber->backlogSeconds() > 2.0)
		myWarning = "Transcription is " + number(myTranscriber->backlogSeconds(), "%.1f") +
			" s behind real time: try a smaller model";
}

void MediaPipeCHOP::execute(CHOP_Output* output, const OP_Inputs* inputs, void*)
{
	myError.clear();
	myWarning.clear();
	const Task task = inputs->getParInt("Task") == (int)Task::Transcribe ? Task::Transcribe : Task::Classify;
	const bool active = inputs->getParInt("Active") != 0;
	const char* modelPar = task == Task::Transcribe ? "Transcribermodel" : "Classifiermodel";
	const std::string model = mpaudio::resolveModelPath(inputs->getParFilePath(modelPar),
		inputs->getParString(modelPar), myModelHint);

	if (task != myTask || !active || model.empty() || myReload)
	{
		myClassifier.reset();
		myTranscriber.reset();
		myResult = mpaudio::Result();
		if (task != myTask)
		{
			myTranscript = mpaudio::Transcript();
			myLastFinals = 0;
			myTotalFinals = 0;
		}
		resetStream();
		myTask = task;
		myReload = false;
	}
	const OP_CHOPInput* input = inputs->getInputCHOP(0);
	if (!active)
		;
	else if (model.empty())
		myWarning = std::string("Set ") + (task == Task::Transcribe ? "Transcriber Model" : "Classifier Model");
	else if (!input || input->sampleRate <= 0)
		myWarning = "Connect an audio CHOP (e.g. Audio Device In or Audio File In)";
	else
	{
		std::vector<float> fresh = takeNewSamples(input);
		if (task == Task::Transcribe)
			transcribe(inputs, fresh, model);
		else
			classify(inputs, fresh, model);
	}

	if (task == Task::Transcribe)
	{
		const float values[] = { mySpeaking ? 1.0f : 0.0f, myLevel, (float)myTotalFinals,
			myNewSegment ? 1.0f : 0.0f, myTranscriber ? (float)myTranscriber->backlogSeconds() : 0.0f,
			myThreshold };
		for (int32_t c = 0; c < output->numChannels && c < (int32_t)std::size(values); c++)
			output->channels[c][0] = values[c];
		return;
	}
	updateChannelNames();
	// The Info DAT's top results, sorted once per cook (ties ordered by class index, so rows
	// asked for one at a time can't repeat or skip a class).
	myTopOrder.resize(myResult.scores.size());
	std::iota(myTopOrder.begin(), myTopOrder.end(), 0);
	const size_t top = std::min(myTopOrder.size(), (size_t)myTopCount);
	std::partial_sort(myTopOrder.begin(), myTopOrder.begin() + top, myTopOrder.end(), [this](int a, int b) {
		return myResult.scores[a] != myResult.scores[b] ? myResult.scores[a] > myResult.scores[b] : a < b;
	});
	myTopOrder.resize(top);
	for (int32_t c = 0; c < output->numChannels; c++)
		output->channels[c][0] = c < (int32_t)myResult.scores.size() ? myResult.scores[c] : 0.0f;
}

int32_t MediaPipeCHOP::getNumInfoCHOPChans(void*)
{
	return 6;
}

void MediaPipeCHOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
	const bool transcribe = myTask == Task::Transcribe;
	const bool loaded = transcribe ? myTranscriber && myTranscriber->loaded() : myClassifier && myClassifier->loaded();
	const bool exists = transcribe ? (bool)myTranscriber : (bool)myClassifier;
	switch (index)
	{
		case 0: chan->name->setString("loaded");		chan->value = loaded ? 1.f : 0.f; break;
		case 1: chan->name->setString("loading");		chan->value = exists && !loaded && myError.empty() ? 1.f : 0.f; break;
		case 2: chan->name->setString("infer_ms");		chan->value = (float)(transcribe ? myTranscript.inferMs : myResult.inferMs); break;
		case 3: chan->name->setString("results");		chan->value = (float)(transcribe ? myTranscript.serial : myResult.serial); break;
		case 4: chan->name->setString("dropped");		chan->value = !transcribe && myClassifier ? (float)myClassifier->dropped() : 0.f; break;
		case 5: chan->name->setString("sample_rate");	chan->value = (float)mySampleRate; break;
	}
}

bool MediaPipeCHOP::getInfoDATSize(OP_InfoDATSize* infoSize, void*)
{
	infoSize->byColumn = false;
	if (myTask == Task::Transcribe)
	{
		// One row per final segment (oldest first), then the interim hypothesis if there is one.
		infoSize->rows = 1 + (int32_t)myTranscript.finals.size() + (myTranscript.interim.text.empty() ? 0 : 1);
		infoSize->cols = 5;
		return true;
	}
	infoSize->rows = 1 + std::min(myTopCount, (int32_t)myResult.scores.size());
	infoSize->cols = 4;
	return true;
}

void MediaPipeCHOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	auto set = [&](int col, const std::string& value) {
		if (col < nEntries)
			entries->values[col]->setString(value.c_str());
	};
	if (myTask == Task::Transcribe)
	{
		// Columns: segment (running number), final, start_ms, end_ms, text.
		if (index == 0)
		{
			const char* header[] = { "segment", "final", "start_ms", "end_ms", "text" };
			for (int col = 0; col < 5; col++)
				set(col, header[col]);
			return;
		}
		const size_t row = (size_t)index - 1;
		const bool isFinal = row < myTranscript.finals.size();
		if (!isFinal && myTranscript.interim.text.empty())
		{
			for (int col = 0; col < nEntries; col++)
				set(col, "");
			return;
		}
		const mpaudio::Segment& s = isFinal ? myTranscript.finals[row] : myTranscript.interim;
		const uint64_t firstNumber = myTotalFinals - myTranscript.finals.size() + 1;
		set(0, number((double)(firstNumber + row)));
		set(1, isFinal ? "1" : "0");
		set(2, number((double)s.fromMs));
		set(3, number((double)s.toMs));
		set(4, s.text);
		return;
	}
	// Columns: rank, class (model index), label, score -- the top results, best first.
	if (index == 0)
	{
		const char* header[] = { "rank", "class", "label", "score" };
		for (int col = 0; col < 4; col++)
			set(col, header[col]);
		return;
	}
	const int rank = index - 1;
	if (rank >= (int)myTopOrder.size())
	{
		for (int col = 0; col < nEntries; col++)
			set(col, "");
		return;
	}
	const int cls = myTopOrder[rank];
	set(0, number(rank + 1));
	set(1, number(cls));
	set(2, myResult.names[cls]);
	set(3, number(myResult.scores[cls], "%.4f"));
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
	std::string s = std::string("MediaPipe audio, CPU, worker thread: ") + kTaskLabels[(int)myTask] + "\n" +
		number(mySampleRate) + " Hz input\n";
	info->setString(s.c_str());
}

void MediaPipeCHOP::setParameterEnableStates(const OP_Inputs* inputs, OP_ParEnableState* state, void*)
{
	const bool transcribe = inputs->getParInt("Task") == (int)Task::Transcribe;
	for (const char* name : kClassifyPars)
		state->setEnableState(name, !transcribe);
	for (const char* name : kTranscribePars)
		state->setEnableState(name, transcribe);
}

void MediaPipeCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	auto check = [](OP_ParAppendResult r) { assert(r == OP_ParAppendResult::Success); (void)r; };
	auto toggle = [&](const char* name, const char* label, const char* page, bool value, const char* help) {
		OP_NumericParameter np(name);
		np.label = label;
		np.page = page;
		np.defaultValues[0] = value ? 1.0 : 0.0;
		np.help = help;
		check(manager->appendToggle(np));
	};
	auto pulse = [&](const char* name, const char* label, const char* page, const char* help) {
		OP_NumericParameter np(name);
		np.label = label;
		np.page = page;
		np.help = help;
		check(manager->appendPulse(np));
	};
	auto real = [&](const char* name, const char* label, const char* page, double value, double low, double high,
		const char* help) {
		OP_NumericParameter np(name);
		np.label = label;
		np.page = page;
		np.defaultValues[0] = value;
		np.minValues[0] = np.minSliders[0] = low;
		np.maxSliders[0] = high;
		np.clampMins[0] = true;
		np.help = help;
		check(manager->appendFloat(np));
	};
	auto file = [&](const char* name, const char* label, const char* page, const char* value, const char* help) {
		OP_StringParameter sp(name);
		sp.label = label;
		sp.page = page;
		sp.defaultValue = value;
		sp.help = help;
		check(manager->appendFile(sp));
	};

	const char* page = "Audio";
	{
		OP_StringParameter sp("Task");
		sp.label = "Task";
		sp.page = page;
		sp.defaultValue = kTaskNames[0];
		sp.help = "Classify Sounds: 521 sound classes (YAMNet). Transcribe Speech: streaming speech to text.";
		check(manager->appendMenu(sp, 2, kTaskNames, kTaskLabels));
	}
	toggle("Active", "Active", page, true, "Off = stop and release the model.");
	pulse("Reload", "Reload Model", page, "Reload the model (and restart the transcript).");

	page = "Classify";
	file("Classifiermodel", "Classifier Model", page, "models/mediapipe/yamnet.tflite",
		"A MediaPipe audio classifier .tflite (YAMNet: 521 AudioSet classes). A relative path is looked up "
		"under the project folder, then next to the plugin.");
	real("Window", "Window (Seconds)", page, 0.975, 0.1, 5.0,
		"How much of the most recent audio each classification looks at. YAMNet analyses 0.975 s.");
	real("Hop", "Hop (Seconds)", page, 0.25, 0.02, 2.0,
		"How often a new classification starts. Smaller = more responsive, more CPU.");
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

	page = "Transcribe";
	file("Transcribermodel", "Transcriber Model", page, "models/asr/moonshine_tiny_5s_i8.litertlm",
		"A MediaPipe speech-to-text .litertlm bundle: moonshine_tiny_5s_i8 (streams interim text), "
		"whisper_tiny_30s_i8 (most accurate of the small ones), parakeet_tdt_0.6b_v3, qwen3_asr_0.6b.");
	toggle("Interim", "Interim Results", page, true,
		"Also show the transcriber's current guess before a segment is final (models that support it).");
	{
		OP_StringParameter sp("Prompt");
		sp.label = "Prompt";
		sp.page = page;
		sp.help = "Optional prompt for generative speech models (e.g. names or terms to expect). Empty = "
			"the model's default. Takes effect when the model loads.";
		check(manager->appendString(sp));
	}
	toggle("Skipsilence", "Skip Silence", page, true,
		"Only send speech (plus a short lead-in) to the model. Models tend to invent text from room "
		"noise; this also saves CPU. Off = send everything.");
	toggle("Autoflush", "Finalize On Pause", page, true,
		"After speech, a pause finalizes the sentence (and Max Segment caps continuous speech). Off = "
		"only the model's own segmentation and the Flush pulse (Skip Silence still finalizes on a pause).");
	real("Pause", "Pause (Seconds)", page, 0.5, 0.1, 3.0,
		"Silence after speech that ends a sentence. Shorter = captions sooner, but sentences may split.");
	real("Maxsegment", "Max Segment (Seconds)", page, 12.0, 1.0, 30.0,
		"Finalize continuous speech after this long, even without a pause.");
	real("Silencelevel", "Silence Level", page, 0.003, 0.0, 0.1,
		"The lowest input RMS that counts as speech. The threshold channel shows the level in use; "
		"the level channel shows the current RMS.");
	real("Noiseratio", "Noise Ratio", page, 4.0, 1.0, 20.0,
		"Speech must be this many times louder than the tracked noise floor. Raise it if background "
		"noise gets transcribed, lower it if quiet speech is missed.");
	pulse("Flush", "Flush", page, "Finalize whatever has been heard so far.");
	pulse("Clear", "Clear Transcript", page, "Forget the transcript and start a new stream.");
}

void MediaPipeCHOP::pulsePressed(const char* name, void*)
{
	const std::string pulse = name;
	if (pulse == "Reload")
		myReload = true;
	else if (pulse == "Flush")
		myFlush = true;
	else if (pulse == "Clear")
		myClear = true;
}
