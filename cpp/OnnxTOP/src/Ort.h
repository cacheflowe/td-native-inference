// Single include point for the ONNX Runtime C++ API.
//
// ORT_API_MANUAL_INIT must be defined identically in every translation unit that includes
// onnxruntime_cxx_api.h (it changes the definition of Ort::Global<T>::api_), so nothing in
// this project includes the ORT header directly -- always include this file instead.
//
// Manual init means this plugin never links against onnxruntime.lib: OrtLoader resolves
// OrtGetApiBase at runtime from TouchDesigner's OWN bundled onnxruntime.dll, so there is
// exactly one ORT in the process and no version collision with TD's copy.
#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <string>

namespace ortload
{
	// Loads onnxruntime.dll from dllDir (empty = the host process' exe directory, i.e.
	// TouchDesigner's bin folder) and initializes the Ort:: C++ API. Idempotent and
	// thread-safe; returns false with err set if the DLL or a compatible API version
	// can't be found.
	bool init(std::string& err, const std::wstring& dllDir = L"");

	// Version string reported by the loaded DLL (e.g. "1.23.2"), empty before init().
	std::string version();

	// Full path of the loaded onnxruntime.dll, empty before init().
	std::string loadedPath();

	// Process-wide Ort::Env (one per process is the ORT recommendation). Valid after init().
	Ort::Env& env();

	// Makes sure TensorRT in this process is the copy shipped next to onnxruntime.dll.
	// Windows reuses an already-loaded DLL by module name, so if something else (e.g. TD's
	// Nvidia Background TOP via the NVIDIA Video Effects SDK) loaded its own nvinfer_10.dll
	// first, ORT's TensorRT EP binds to that build, createInferBuilder() returns null and the
	// process crashes inside session creation. Returns "" if TD's copies are (now) the loaded
	// ones -- pinning them if nothing had loaded TensorRT yet -- or a description of the
	// conflicting module, in which case the TensorRT EP must not be used.
	std::string claimTensorRT();

	// Last few ORT log lines at WARNING or above, newest last (thread-safe snapshot).
	std::string recentLog(size_t maxLines = 8);
}
