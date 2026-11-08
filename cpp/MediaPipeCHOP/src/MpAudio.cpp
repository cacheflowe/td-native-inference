#include "MpAudio.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <windows.h>

namespace fs = std::filesystem;

namespace
{
	struct Api
	{
		decltype(&MpErrorFree)					errorFree = nullptr;
		decltype(&MpAudioClassifierCreate)		create = nullptr;
		decltype(&MpAudioClassifierClassify)	classify = nullptr;
		decltype(&MpAudioClassifierCloseResult)	closeResult = nullptr;
		decltype(&MpAudioClassifierClose)		close = nullptr;
		decltype(&MpAudioTranscriberCreate)		asrCreate = nullptr;
		decltype(&MpAudioTranscriberTranscribe)	asrTranscribe = nullptr;
		decltype(&MpAudioTranscriberFinishStream)	asrFinish = nullptr;
		decltype(&MpAudioTranscriberCloseResult)	asrCloseResult = nullptr;
		decltype(&MpAudioTranscriberClose)		asrClose = nullptr;
		decltype(&MpTextToSpeechOptionsInitDefault)	ttsOptionsInit = nullptr;
		decltype(&MpTextToSpeechSynthesisOptionsInitDefault)	ttsSynthesisOptionsInit = nullptr;
		decltype(&MpTextToSpeechCreate)			ttsCreate = nullptr;
		decltype(&MpTextToSpeechListVoices)		ttsListVoices = nullptr;
		decltype(&MpTextToSpeechVoiceListFree)	ttsVoiceListFree = nullptr;
		decltype(&MpTextToSpeechStartStream)	ttsStartStream = nullptr;
		decltype(&MpTextToSpeechStreamReadNext)	ttsReadNext = nullptr;
		decltype(&MpTextToSpeechStreamCancel)	ttsStreamCancel = nullptr;
		decltype(&MpTextToSpeechStreamClose)	ttsStreamClose = nullptr;
		decltype(&MpTextToSpeechFreeAudioResult)	ttsFreeAudio = nullptr;
		decltype(&MpTextToSpeechClose)			ttsClose = nullptr;
		decltype(&MpTextToSpeechFreeError)		ttsFreeError = nullptr;
	};

	Api			gApi;
	std::mutex	gApiMutex;
	bool		gApiLoaded = false;
	std::string	gApiError;

	fs::path pluginDir()
	{
		HMODULE mod = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&pluginDir), &mod);
		wchar_t buf[MAX_PATH * 2] = {};
		GetModuleFileNameW(mod, buf, (DWORD)std::size(buf));
		return fs::path(buf).parent_path();
	}

	template <typename T>
	void bind(HMODULE lib, T& fn, const char* name, std::string& missing)
	{
		fn = reinterpret_cast<T>(GetProcAddress(lib, name));
		if (!fn)
			missing += std::string(missing.empty() ? "" : ", ") + name;
	}

	// libmediapipe.dll is shared with MediaPipeTOP; LoadLibrary returns the same module if it's loaded.
	bool loadApi(std::string& err)
	{
		std::lock_guard<std::mutex> lock(gApiMutex);
		if (gApiLoaded)
			return true;
		if (gApiError.empty())
		{
			// MP_AUDIO_DLL_DIR: the console test runs from the build folder.
			const char* dir = std::getenv("MP_AUDIO_DLL_DIR");
			fs::path dll = (dir && *dir ? fs::u8path(dir) : pluginDir()) / "libmediapipe.dll";
			HMODULE lib = fs::exists(dll) ? LoadLibraryExW(dll.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) : nullptr;
			if (!lib)
				gApiError = "Could not load libmediapipe.dll next to the plugin (" + dll.parent_path().u8string() + ")";
			else
			{
				std::string missing;
				bind(lib, gApi.errorFree, "MpErrorFree", missing);
				bind(lib, gApi.create, "MpAudioClassifierCreate", missing);
				bind(lib, gApi.classify, "MpAudioClassifierClassify", missing);
				bind(lib, gApi.closeResult, "MpAudioClassifierCloseResult", missing);
				bind(lib, gApi.close, "MpAudioClassifierClose", missing);
				bind(lib, gApi.asrCreate, "MpAudioTranscriberCreate", missing);
				bind(lib, gApi.asrTranscribe, "MpAudioTranscriberTranscribe", missing);
				bind(lib, gApi.asrFinish, "MpAudioTranscriberFinishStream", missing);
				bind(lib, gApi.asrCloseResult, "MpAudioTranscriberCloseResult", missing);
				bind(lib, gApi.asrClose, "MpAudioTranscriberClose", missing);
				bind(lib, gApi.ttsOptionsInit, "MpTextToSpeechOptionsInitDefault", missing);
				bind(lib, gApi.ttsSynthesisOptionsInit, "MpTextToSpeechSynthesisOptionsInitDefault", missing);
				bind(lib, gApi.ttsCreate, "MpTextToSpeechCreate", missing);
				bind(lib, gApi.ttsListVoices, "MpTextToSpeechListVoices", missing);
				bind(lib, gApi.ttsVoiceListFree, "MpTextToSpeechVoiceListFree", missing);
				bind(lib, gApi.ttsStartStream, "MpTextToSpeechStartStream", missing);
				bind(lib, gApi.ttsReadNext, "MpTextToSpeechStreamReadNext", missing);
				bind(lib, gApi.ttsStreamCancel, "MpTextToSpeechStreamCancel", missing);
				bind(lib, gApi.ttsStreamClose, "MpTextToSpeechStreamClose", missing);
				bind(lib, gApi.ttsFreeAudio, "MpTextToSpeechFreeAudioResult", missing);
				bind(lib, gApi.ttsClose, "MpTextToSpeechClose", missing);
				bind(lib, gApi.ttsFreeError, "MpTextToSpeechFreeError", missing);
				if (!missing.empty())
					gApiError = "libmediapipe.dll is missing " + missing + " (needs the mediapipe 1.2.2 build)";
				else
					gApiLoaded = true;
			}
		}
		err = gApiError;
		return gApiLoaded;
	}

	std::string takeError(char* msg)
	{
		std::string s = msg ? msg : "unknown error";
		if (msg && gApi.errorFree)
			gApi.errorFree(msg);
		return s;
	}

	// Text to speech frees its error strings with its own function.
	std::string takeTtsError(char*& msg)
	{
		std::string s = msg ? msg : "unknown error";
		if (msg && gApi.ttsFreeError)
			gApi.ttsFreeError(msg);
		msg = nullptr;
		return s;
	}
}

namespace mpaudio
{
	std::string resolveModelPath(const char* resolved, const char* raw, std::string& missing)
	{
		std::string path = resolved ? resolved : "";
		missing.clear();
		std::error_code ec;
		if (path.empty() || fs::exists(path, ec) || !raw || !*raw)
			return path;
		fs::path rel(raw);
		if (rel.is_absolute())
			return path;
		fs::path candidate = (pluginDir() / rel).lexically_normal();
		if (fs::exists(candidate, ec))
			return candidate.generic_string();
		missing = "Looked for " + rel.generic_string() + " in the project folder and in " +
			pluginDir().generic_string() + ".";
		return path;
	}

	Classifier::Classifier(const std::string& modelPath) : myModelPath(modelPath)
	{
		myThread = std::thread(&Classifier::run, this);
	}

	Classifier::~Classifier()
	{
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myStop = true;
		}
		myWake.notify_all();
		if (myThread.joinable())
			myThread.join();
	}

	void Classifier::submit(std::vector<float> mono, double sampleRate)
	{
		{
			std::lock_guard<std::mutex> lock(myMutex);
			if (myHasPending)
				myDropped++;
			myPending = std::move(mono);
			myPendingRate = sampleRate;
			myHasPending = true;
		}
		myWake.notify_one();
	}

	bool Classifier::latest(Result& out) const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		if (myResult.serial == 0)
			return false;
		out = myResult;
		return true;
	}

	bool Classifier::loaded() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myLoaded;
	}

	std::string Classifier::error() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myError;
	}

	uint64_t Classifier::dropped() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myDropped;
	}

	void Classifier::run()
	{
		std::string err;
		if (!loadApi(err))
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = err;
			return;
		}
		MpAudioClassifierOptions options = {};
		options.base_options.model_asset_path = myModelPath.c_str();
		options.base_options.delegate = MP_DELEGATE_CPU;	// the Windows build has no GPU support
		options.classifier_options.max_results = -1;		// every class, so channels stay stable
		options.classifier_options.score_threshold = 0.0f;
		options.running_mode = kMpAudioRunningModeAudioClips;
		options.result_callback = nullptr;
		MpAudioClassifierPtr classifier = nullptr;
		char* msg = nullptr;
		if (gApi.create(&options, &classifier, &msg) != kMpOk)
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = "Could not load " + myModelPath + ": " + takeError(msg);
			return;
		}
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myLoaded = true;
		}

		Result result;
		for (;;)
		{
			std::vector<float> clip;
			double rate = 0.0;
			{
				std::unique_lock<std::mutex> lock(myMutex);
				myWake.wait(lock, [this] { return myStop || myHasPending; });
				if (myStop)
					break;
				clip.swap(myPending);
				rate = myPendingRate;
				myHasPending = false;
			}
			MpAudioData data = {};
			data.num_channels = 1;
			data.sample_rate = rate;
			data.audio_data = clip.data();
			data.audio_data_size = clip.size();
			MpAudioClassifierResult classified = {};
			auto start = std::chrono::steady_clock::now();
			MpStatus status = gApi.classify(classifier, &data, &classified, &msg);
			double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			if (status != kMpOk)
			{
				std::lock_guard<std::mutex> lock(myMutex);
				myError = "Classification failed: " + takeError(msg);
				continue;
			}
			// A clip longer than the model's window (YAMNet: 0.975 s) yields one result per window;
			// the last one is usually a partial window padded with silence. Average the full ones.
			const int windows = classified.results_count > 1 ? classified.results_count - 1 : classified.results_count;
			if (windows > 0)
			{
				std::fill(result.scores.begin(), result.scores.end(), 0.0f);
				for (int w = 0; w < windows; w++)
				{
					if (classified.results[w].classifications_count == 0)
						continue;
					const MpClassifications& head = classified.results[w].classifications[0];
					for (uint32_t i = 0; i < head.categories_count; i++)
					{
						const MpCategory& category = head.categories[i];
						if (category.index < 0)
							continue;
						size_t index = (size_t)category.index;
						if (index >= result.names.size())
						{
							result.names.resize(index + 1);
							result.scores.resize(index + 1, 0.0f);
						}
						if (result.names[index].empty())
							result.names[index] = category.category_name ? category.category_name :
								(category.display_name ? category.display_name : "class" + std::to_string(index));
						result.scores[index] += category.score / (float)windows;
					}
				}
				result.inferMs = ms;
				result.serial++;
				std::lock_guard<std::mutex> lock(myMutex);
				myResult = result;
				myError.clear();
			}
			gApi.closeResult(&classified);
		}
		gApi.close(classifier, &msg);
		if (msg)
			takeError(msg);
	}

	Transcriber::Transcriber(const std::string& modelPath, int sampleRate, bool interim, const std::string& prompt)
		: myModelPath(modelPath), mySampleRate(sampleRate), myInterim(interim), myPrompt(prompt)
	{
		myThread = std::thread(&Transcriber::run, this);
	}

	Transcriber::~Transcriber()
	{
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myStop = true;
		}
		myWake.notify_all();
		if (myThread.joinable())
			myThread.join();
	}

	void Transcriber::submit(const float* samples, size_t count)
	{
		if (count == 0)
			return;
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myQueue.insert(myQueue.end(), samples, samples + count);
		}
		myWake.notify_one();
	}

	void Transcriber::finishStream()
	{
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myFinish = true;
		}
		myWake.notify_one();
	}

	void Transcriber::skip(size_t count)
	{
		std::lock_guard<std::mutex> lock(myMutex);
		mySkipSamples += count;
	}

	bool Transcriber::latest(Transcript& out) const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		if (myTranscript.serial == out.serial)
			return false;
		out = myTranscript;
		return true;
	}

	bool Transcriber::loaded() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myLoaded;
	}

	std::string Transcriber::error() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myError;
	}

	double Transcriber::backlogSeconds() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return mySampleRate > 0 ? (double)myQueue.size() / mySampleRate : 0.0;
	}

	bool Transcriber::idle() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myLoaded && !myBusy && myQueue.empty() && !myFinish;
	}

	void Transcriber::collect(const MpAudioTranscriberResult& result)
	{
		std::lock_guard<std::mutex> lock(myMutex);
		bool changed = false;
		const int64_t offsetMs = mySampleRate > 0 ? (int64_t)(myOffsetSamples * 1000 / (uint64_t)mySampleRate) : 0;
		for (uint32_t i = 0; i < result.segments_count; i++)
		{
			const MpTranscriptionSegment& s = result.segments[i];
			Segment segment;
			segment.text = s.transcript ? s.transcript : "";
			segment.final = s.is_final;
			segment.fromMs = s.from_timestamp_ms + offsetMs;
			segment.toMs = s.to_timestamp_ms + offsetMs;
			if (segment.final)
			{
				myTranscript.finals.push_back(std::move(segment));
				myTranscript.totalFinals++;
				if (myTranscript.finals.size() > kMaxFinals)
					myTranscript.finals.erase(myTranscript.finals.begin());
				myTranscript.interim = Segment();
			}
			else
				myTranscript.interim = std::move(segment);
			changed = true;
		}
		if (changed)
			myTranscript.serial++;
	}

	void Transcriber::run()
	{
		std::string err;
		if (!loadApi(err))
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = err;
			return;
		}
		MpAudioTranscriberOptions options = {};
		options.base_options.model_asset_path = myModelPath.c_str();
		options.base_options.delegate = MP_DELEGATE_CPU;
		options.sample_rate_hz = mySampleRate;
		options.enable_interim_results = myInterim;
		options.model_architecture = MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_AUTO;
		options.prompt = myPrompt.empty() ? nullptr : myPrompt.c_str();
		options.cache_dir = nullptr;
		MpAudioTranscriberPtr transcriber = nullptr;
		char* msg = nullptr;
		if (gApi.asrCreate(&options, &transcriber, &msg) != kMpOk)
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = "Could not load " + myModelPath + ": " + takeError(msg);
			return;
		}
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myLoaded = true;
		}
		for (;;)
		{
			std::vector<float> chunk;
			bool finish = false;
			{
				std::unique_lock<std::mutex> lock(myMutex);
				myWake.wait(lock, [this] { return myStop || !myQueue.empty() || myFinish; });
				if (myStop)
					break;
				chunk.swap(myQueue);	// everything queued so far, in order
				finish = myFinish;
				myFinish = false;
				myBusy = true;
			}
			auto start = std::chrono::steady_clock::now();
			MpAudioTranscriberResult result = {};
			MpStatus status = kMpOk;
			if (!chunk.empty())
			{
				{
					std::lock_guard<std::mutex> lock(myMutex);
					if (myStreamSamples == 0)
					{
						// A new stream: count the audio skipped since the last one.
						myOffsetSamples += mySkipSamples;
						mySkipSamples = 0;
					}
					myStreamSamples += chunk.size();
				}
				status = gApi.asrTranscribe(transcriber, chunk.data(), chunk.size(), false, &result, &msg);
				if (status == kMpOk)
					collect(result);
				gApi.asrCloseResult(&result);
			}
			if (status == kMpOk && finish)
			{
				result = {};
				status = gApi.asrFinish(transcriber, &result, &msg);
				if (status == kMpOk)
					collect(result);
				gApi.asrCloseResult(&result);
				std::lock_guard<std::mutex> lock(myMutex);
				myOffsetSamples += myStreamSamples;	// the next stream's timestamps start here
				myStreamSamples = 0;
			}
			double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			std::lock_guard<std::mutex> lock(myMutex);
			myBusy = false;
			myTranscript.inferMs = ms;
			if (status != kMpOk)
				myError = "Transcription failed: " + takeError(msg);
		}
		gApi.asrClose(transcriber, &msg);
		if (msg)
			takeError(msg);
	}

	Speaker::Speaker(const std::string& modelPath, float rate) : myModelPath(modelPath), myRate(rate)
	{
		myThread = std::thread(&Speaker::run, this);
	}

	Speaker::~Speaker()
	{
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myStop = true;
			if (myStream)
				gApi.ttsStreamCancel(myStream);	// lock-free; unblocks ReadNext
		}
		myWake.notify_all();
		if (myThread.joinable())
			myThread.join();
	}

	void Speaker::say(const std::string& text, const std::string& voice)
	{
		if (text.find_first_not_of(" \t\r\n") == std::string::npos)
			return;
		{
			std::lock_guard<std::mutex> lock(myMutex);
			Utterance line;
			line.id = myNextId++;
			line.text = text;
			line.voice = voice;
			myQueue.push_back(std::move(line));
		}
		myWake.notify_one();
	}

	void Speaker::stop()
	{
		std::lock_guard<std::mutex> lock(myMutex);
		myQueue.clear();
		myPlaying.clear();
		myAudio.clear();
		myGeneration++;
		if (myStream)
			gApi.ttsStreamCancel(myStream);
	}

	size_t Speaker::read(float* out, size_t count)
	{
		std::lock_guard<std::mutex> lock(myMutex);
		const size_t n = std::min(count, myAudio.size());
		std::copy(myAudio.begin(), myAudio.begin() + n, out);
		myAudio.erase(myAudio.begin(), myAudio.begin() + n);
		// Account the played samples to their lines, oldest first; a fully played, closed line is done.
		size_t left = n;
		while (!myPlaying.empty())
		{
			Utterance& line = myPlaying.front();
			const size_t used = std::min(left, line.samples);
			line.samples -= used;
			left -= used;
			if (line.samples > 0 || line.open)
				break;
			myPlaying.pop_front();
			mySpoken++;
		}
		return n;
	}

	bool Speaker::loaded() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myLoaded;
	}

	std::string Speaker::error() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myError;
	}

	int Speaker::sampleRate() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return mySampleRate;
	}

	SpeakerStatus Speaker::status() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		SpeakerStatus s;
		s.lines.assign(myPlaying.begin(), myPlaying.end());
		s.lines.insert(s.lines.end(), myQueue.begin(), myQueue.end());
		s.spoken = mySpoken;
		s.synthMs = mySynthMs;
		s.speaking = !myPlaying.empty();
		return s;
	}

	std::vector<std::string> Speaker::voices() const
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return myVoices;
	}

	void Speaker::run()
	{
		std::string err;
		if (!loadApi(err))
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = err;
			return;
		}
		MpTextToSpeechOptions options;
		gApi.ttsOptionsInit(&options);
		options.base_options.model_asset_path = myModelPath.c_str();
		options.base_options.delegate = MP_DELEGATE_CPU;
		options.speaking_rate = myRate;
		MpTextToSpeechPtr tts = nullptr;
		char* msg = nullptr;
		if (gApi.ttsCreate(&options, &tts, &msg) != kMpOk)
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myError = "Could not load " + myModelPath + ": " + takeTtsError(msg);
			return;
		}
		std::vector<std::string> voices;
		MpTextToSpeechVoiceList list = {};
		if (gApi.ttsListVoices(tts, &list, &msg) == kMpOk)
		{
			for (size_t i = 0; i < list.num_voices; i++)
				if (list.voices[i].voice_id)
					voices.push_back(list.voices[i].voice_id);
			gApi.ttsVoiceListFree(&list);
		}
		else
			takeTtsError(msg);
		{
			std::lock_guard<std::mutex> lock(myMutex);
			myLoaded = true;
			myVoices = std::move(voices);
		}
		for (;;)
		{
			Utterance line;
			uint64_t generation = 0;
			{
				std::unique_lock<std::mutex> lock(myMutex);
				myWake.wait(lock, [this] { return myStop || !myQueue.empty(); });
				if (myStop)
					break;
				line = std::move(myQueue.front());
				myQueue.pop_front();
				line.open = true;
				line.samples = 0;
				myPlaying.push_back(line);
				generation = myGeneration;
			}
			MpTextToSpeechSynthesisOptions synthesis;
			gApi.ttsSynthesisOptionsInit(&synthesis);
			synthesis.voice_id = line.voice.empty() ? nullptr : line.voice.c_str();
			MpTextToSpeechStreamPtr stream = nullptr;
			MpStatus status = gApi.ttsStartStream(tts, line.text.c_str(), &synthesis, true, &stream, &msg);
			std::string failure = status == kMpOk ? "" : "Speaking failed: " + takeTtsError(msg);
			if (stream)
			{
				std::lock_guard<std::mutex> lock(myMutex);
				myStream = stream;
				if (myStop || myGeneration != generation)
					gApi.ttsStreamCancel(stream);	// stop() came before the stream existed
			}
			while (stream)
			{
				auto start = std::chrono::steady_clock::now();
				MpTextToSpeechAudioResult chunk = {};
				int32_t index = 0;
				bool final = false;
				status = gApi.ttsReadNext(stream, &chunk, &index, &final, &msg);
				const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				if (status == kMpOk && chunk.samples && chunk.num_samples)
				{
					std::lock_guard<std::mutex> lock(myMutex);
					if (myGeneration == generation && !myPlaying.empty() && myPlaying.back().id == line.id)
					{
						myAudio.insert(myAudio.end(), chunk.samples, chunk.samples + chunk.num_samples);
						myPlaying.back().samples += chunk.num_samples;
						if (chunk.sample_rate_hz > 0)
							mySampleRate = chunk.sample_rate_hz;
						mySynthMs = ms;
					}
				}
				gApi.ttsFreeAudio(&chunk);
				if (status == kMpOk && !final)
					continue;
				if (status != kMpOk && status != kMpOutOfRange && status != kMpCancelled)
					failure = "Speaking failed: " + takeTtsError(msg);
				else if (msg)
					takeTtsError(msg);
				break;
			}
			std::lock_guard<std::mutex> lock(myMutex);
			if (stream)
			{
				myStream = nullptr;
				gApi.ttsStreamClose(stream);
			}
			myError = failure;	// a line that works clears an earlier line's error
			// The line is complete: once its audio has played, read() retires it.
			for (Utterance& playing : myPlaying)
				if (playing.id == line.id)
					playing.open = false;
			while (!myPlaying.empty() && !myPlaying.front().open && myPlaying.front().samples == 0)
			{
				myPlaying.pop_front();
				mySpoken++;
			}
		}
		gApi.ttsClose(tts);
	}
}
