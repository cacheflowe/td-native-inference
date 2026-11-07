#include "Ort.h"

#include <windows.h>

#include <deque>
#include <memory>
#include <mutex>

namespace
{
	std::mutex gMutex;
	bool gReady = false;
	std::string gError;
	std::string gVersion;
	std::string gPath;
	std::wstring gDir;
	std::unique_ptr<Ort::Env> gEnv;

	std::mutex gLogMutex;
	std::deque<std::string> gLog;
	constexpr size_t kMaxLogLines = 64;

	std::string narrow(const std::wstring& w)
	{
		if (w.empty())
			return {};
		int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
		std::string s(n, '\0');
		WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
		return s;
	}

	std::wstring exeDir()
	{
		wchar_t buf[MAX_PATH * 2] = {};
		GetModuleFileNameW(nullptr, buf, (DWORD)std::size(buf));
		std::wstring p(buf);
		size_t slash = p.find_last_of(L"\\/");
		return slash == std::wstring::npos ? p : p.substr(0, slash);
	}

	void ORT_API_CALL logCallback(void*, OrtLoggingLevel severity, const char* category,
		const char* logid, const char* codeLocation, const char* message)
	{
		static const char* names[] = { "VERBOSE", "INFO", "WARNING", "ERROR", "FATAL" };
		std::string line = std::string("[ORT ") + names[severity < 5 ? severity : 4] + "] " + (message ? message : "");
		OutputDebugStringA((line + "\n").c_str());
		std::lock_guard<std::mutex> lock(gLogMutex);
		gLog.push_back(std::move(line));
		while (gLog.size() > kMaxLogLines)
			gLog.pop_front();
	}
}

namespace ortload
{
	bool init(std::string& err, const std::wstring& dllDir)
	{
		std::lock_guard<std::mutex> lock(gMutex);
		if (gReady)
			return true;
		if (!gError.empty())
		{
			err = gError;
			return false;
		}

		std::wstring dir = dllDir.empty() ? exeDir() : dllDir;
		std::wstring dll = dir + L"\\onnxruntime.dll";

		// LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: resolve onnxruntime.dll's own dependencies from
		// its folder first, so a different onnxruntime/cuDNN on PATH can never be picked up.
		HMODULE h = LoadLibraryExW(dll.c_str(), nullptr,
			LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
		if (!h)
		{
			gError = "Could not load " + narrow(dll) + " (error " + std::to_string(GetLastError()) + ")";
			err = gError;
			return false;
		}

		using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
		auto getApiBase = reinterpret_cast<GetApiBaseFn>(GetProcAddress(h, "OrtGetApiBase"));
		if (!getApiBase)
		{
			gError = "OrtGetApiBase not exported by " + narrow(dll);
			err = gError;
			return false;
		}

		const OrtApiBase* base = getApiBase();
		const OrtApi* api = base->GetApi(ORT_API_VERSION);
		if (!api)
		{
			gError = "onnxruntime.dll " + std::string(base->GetVersionString()) +
				" does not provide ORT API version " + std::to_string(ORT_API_VERSION) +
				" (plugin was built against newer ORT headers than this TouchDesigner ships)";
			err = gError;
			return false;
		}

		Ort::InitApi(api);
		gVersion = base->GetVersionString();
		gPath = narrow(dll);
		gDir = dir;

		try
		{
			gEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxTOP", logCallback, nullptr);
		}
		catch (const Ort::Exception& e)
		{
			gError = std::string("Ort::Env creation failed: ") + e.what();
			err = gError;
			return false;
		}

		gReady = true;
		return true;
	}

	std::string claimTensorRT()
	{
		std::lock_guard<std::mutex> lock(gMutex);
		if (!gReady)
			return "ONNX Runtime not initialized";
		std::wstring dir = gDir;
		for (const wchar_t* name : { L"nvinfer_10.dll", L"nvinfer_plugin_10.dll", L"nvonnxparser_10.dll" })
		{
			std::wstring ours = dir + L"\\" + name;
			if (HMODULE loaded = GetModuleHandleW(name))
			{
				wchar_t buf[MAX_PATH * 2] = {};
				GetModuleFileNameW(loaded, buf, (DWORD)std::size(buf));
				if (_wcsicmp(buf, ours.c_str()) != 0)
					return narrow(name) + " is already loaded from " + narrow(buf) +
						" (a different TensorRT build, e.g. from an Nvidia Background TOP / NVIDIA Video Effects SDK)";
				continue;
			}
			// Not loaded by anyone yet: load (and keep) TD's copy so later by-name loads --
			// ORT's TensorRT provider included -- resolve to it.
			if (!LoadLibraryExW(ours.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))
				return "Could not load " + narrow(ours) + " (error " + std::to_string(GetLastError()) + ")";
		}
		return {};
	}

	std::string version() { return gVersion; }
	std::string loadedPath() { return gPath; }
	Ort::Env& env() { return *gEnv; }

	std::string recentLog(size_t maxLines)
	{
		std::lock_guard<std::mutex> lock(gLogMutex);
		std::string out;
		size_t start = gLog.size() > maxLines ? gLog.size() - maxLines : 0;
		for (size_t i = start; i < gLog.size(); i++)
			out += gLog[i] + "\n";
		return out;
	}
}
