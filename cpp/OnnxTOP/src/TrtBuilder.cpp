#include "TrtBuilder.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace
{
	std::wstring widen(const std::string& s)
	{
		if (s.empty())
			return {};
		int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
		std::wstring w(n, L'\0');
		MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
		return w;
	}

	std::wstring quote(const std::wstring& s) { return L"\"" + s + L"\""; }
}

TrtBuildResult buildEngineOutOfProcess(const std::wstring& toolExe, const std::wstring& tdBinDir,
	const EngineConfig& cfg, int maxAttempts, const std::atomic<bool>& cancel,
	void (*onAttempt)(void*, int), void* user)
{
	TrtBuildResult result;
	std::string cacheDir = resolveCacheDir(cfg);
	std::error_code ec;
	fs::create_directories(cacheDir, ec);
	std::wstring logPath = (fs::path(cacheDir) / (fs::path(cfg.modelPath).stem().string() + ".build.log")).wstring();

	std::wstring cmd = quote(toolExe) + L" build " + quote(tdBinDir) + L" " + quote(widen(cfg.modelPath)) + L" " +
		quote(widen(cacheDir)) + L" " + (cfg.fp16 ? L"1" : L"0") + L" " + std::to_wstring(cfg.deviceId) +
		L" " + std::to_wstring(cfg.inputW) + L" " + std::to_wstring(cfg.inputH) + L" " + std::to_wstring(cfg.batch);

	// The tool links cudart64_12.dll, which lives in TD's bin folder: put it first on PATH.
	std::wstring path = tdBinDir;
	wchar_t oldPath[32767];
	if (GetEnvironmentVariableW(L"PATH", oldPath, 32767))
		path += L";" + std::wstring(oldPath);
	std::wstring env;
	wchar_t* block = GetEnvironmentStringsW();
	for (wchar_t* p = block; *p; p += wcslen(p) + 1)
		if (_wcsnicmp(p, L"PATH=", 5) != 0)
			env.append(p).push_back(L'\0');
	FreeEnvironmentStringsW(block);
	env.append(L"PATH=" + path).push_back(L'\0');
	env.push_back(L'\0');

	for (int attempt = 1; attempt <= maxAttempts; attempt++)
	{
		result.attempts = attempt;
		if (onAttempt)
			onAttempt(user, attempt);

		SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
		HANDLE log = CreateFileW(logPath.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
			OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

		STARTUPINFOW si = { sizeof(si) };
		si.dwFlags = STARTF_USESTDHANDLES;
		si.hStdOutput = si.hStdError = log;
		si.hStdInput = nullptr;
		PROCESS_INFORMATION pi = {};
		std::wstring cmdMutable = cmd;
		BOOL started = CreateProcessW(nullptr, cmdMutable.data(), nullptr, nullptr, TRUE,
			CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT, (void*)env.c_str(), tdBinDir.c_str(), &si, &pi);
		if (log != INVALID_HANDLE_VALUE)
			CloseHandle(log);
		if (!started)
		{
			result.message = "Could not start builder (error " + std::to_string(GetLastError()) + ")";
			return result;
		}

		DWORD wait;
		while ((wait = WaitForSingleObject(pi.hProcess, 200)) == WAIT_TIMEOUT)
		{
			if (cancel)
			{
				TerminateProcess(pi.hProcess, 1);
				WaitForSingleObject(pi.hProcess, 5000);
				CloseHandle(pi.hThread);
				CloseHandle(pi.hProcess);
				result.cancelled = true;
				result.message = "Cancelled";
				return result;
			}
		}
		DWORD code = 1;
		GetExitCodeProcess(pi.hProcess, &code);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);

		if (code == 0 && fs::exists(trtBuildMarkerPath(cfg), ec))
		{
			result.ok = true;
			result.message.clear();
			return result;
		}
		char buf[160];
		snprintf(buf, sizeof(buf), "builder exited with 0x%08lX", code);
		result.message = buf;	// retry: observed failures have been intermittent
	}
	result.message += " after " + std::to_string(maxAttempts) + " attempts -- see " +
		fs::path(logPath).string();
	return result;
}

int runBuildCommand(const std::wstring& tdBinDir, const EngineConfig& cfgIn)
{
	setvbuf(stdout, nullptr, _IONBF, 0);
	auto t0 = std::chrono::steady_clock::now();
	printf("=== onnxtop_tool build: %s (fp16=%d, device=%d, size=%dx%d)\n", cfgIn.modelPath.c_str(), cfgIn.fp16,
		cfgIn.deviceId, cfgIn.inputW, cfgIn.inputH);

	SetDllDirectoryW(tdBinDir.c_str());
	std::string err;
	if (!ortload::init(err, tdBinDir))
	{
		printf("ORT init failed: %s\n", err.c_str());
		return 2;
	}
	EngineConfig cfg = cfgIn;
	cfg.provider = OrtProvider::TensorRT;
	cudaSetDevice(cfg.deviceId);
	cudaStreamCreateWithFlags(&cfg.stream, cudaStreamNonBlocking);

	OnnxEngine engine;
	if (!engine.load(cfg, err))
	{
		printf("Build failed: %s\n%s", err.c_str(), ortload::recentLog(20).c_str());
		return 3;
	}
	if (engine.providerDescription().rfind("TensorRT", 0) != 0)
	{
		printf("TensorRT unavailable, got: %s\n", engine.providerDescription().c_str());
		return 4;
	}
	std::ofstream(trtBuildMarkerPath(cfg)) << cfg.modelPath << "\n" << engine.providerDescription() << "\n";
	double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	printf("OK in %.1f s -- %s\n", secs, engine.providerDescription().c_str());
	return 0;
}
