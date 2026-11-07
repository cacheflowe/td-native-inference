// Out-of-process TensorRT engine builds.
//
// Building a TensorRT engine (tactic profiling, minutes per model) is the one step in the
// pipeline that has been observed to crash intermittently -- and a crash inside TD's process
// takes the whole project down. So the TOP never builds engines itself: it runs
// onnxtop_tool.exe in "build" mode, which populates the same TensorRT cache folder and writes
// trtBuildMarkerPath(cfg) on success. The TOP then loads in-process from the cache.
#pragma once

#include "OnnxEngine.h"

#include <atomic>
#include <string>

struct TrtBuildResult
{
	bool		ok = false;
	bool		cancelled = false;
	int			attempts = 0;
	std::string	message;
};

// Blocks until the build succeeds, fails maxAttempts times, or cancel becomes true (the child
// process is then terminated). Child stdout/stderr go to <cache>/<model>.build.log.
// onAttempt(n) is called before each attempt (for status display).
TrtBuildResult buildEngineOutOfProcess(const std::wstring& toolExe, const std::wstring& tdBinDir,
	const EngineConfig& cfg, int maxAttempts, const std::atomic<bool>& cancel,
	void (*onAttempt)(void* user, int attempt) = nullptr, void* user = nullptr);

// Entry point for `onnxtop_tool.exe build ...`: builds (loads) the engine and writes the
// marker. Returns the process exit code.
int runBuildCommand(const std::wstring& tdBinDir, const EngineConfig& cfg);
