#include "MpAudio.h"

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
}
