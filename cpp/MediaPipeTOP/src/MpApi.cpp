#include "MpApi.h"

#include <filesystem>
#include <mutex>

#include <windows.h>

namespace fs = std::filesystem;

namespace
{
	MpApi		gApi;
	std::mutex	gMutex;
	bool		gLoaded = false;
	std::string	gError;
	std::string	gPath;
	std::string	gDirOverride;

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
	bool bind(HMODULE lib, T& fn, const char* name, std::string& missing)
	{
		fn = reinterpret_cast<T>(GetProcAddress(lib, name));
		if (!fn)
			missing += std::string(missing.empty() ? "" : ", ") + name;
		return fn != nullptr;
	}
}

namespace mpapi
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

	bool load(std::string& err)
	{
		std::lock_guard<std::mutex> lock(gMutex);
		if (gLoaded)
			return true;
		if (!gError.empty())
		{
			err = gError;
			return false;
		}

		fs::path dll = (gDirOverride.empty() ? pluginDir() : fs::u8path(gDirOverride)) / "libmediapipe.dll";
		if (!fs::exists(dll))
		{
			gError = "libmediapipe.dll not found next to the plugin (" + dll.parent_path().u8string() +
				"). Run cpp/MediaPipeTOP/fetch_mediapipe.ps1 to install it.";
			err = gError;
			return false;
		}
		HMODULE lib = LoadLibraryExW(dll.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		if (!lib)
		{
			gError = "LoadLibrary failed for " + dll.u8string() + " (error " + std::to_string(GetLastError()) + ")";
			err = gError;
			return false;
		}

		std::string missing;
		MpApi& a = gApi;
		bind(lib, a.errorFree, "MpErrorFree", missing);
		bind(lib, a.stringListFree, "MpStringListFree", missing);
		bind(lib, a.imageCreateFromUint8Data, "MpImageCreateFromUint8Data", missing);
		bind(lib, a.imageFree, "MpImageFree", missing);
		bind(lib, a.imageGetWidth, "MpImageGetWidth", missing);
		bind(lib, a.imageGetHeight, "MpImageGetHeight", missing);
		bind(lib, a.imageGetFormat, "MpImageGetFormat", missing);
		bind(lib, a.imageDataFloat32, "MpImageDataFloat32", missing);
		bind(lib, a.imageDataUint8, "MpImageDataUint8", missing);
		bind(lib, a.imageGetWidthStep, "MpImageGetWidthStep", missing);
		bind(lib, a.faceCreate, "MpFaceLandmarkerCreate", missing);
		bind(lib, a.faceDetectForVideo, "MpFaceLandmarkerDetectForVideo", missing);
		bind(lib, a.faceCloseResult, "MpFaceLandmarkerCloseResult", missing);
		bind(lib, a.faceClose, "MpFaceLandmarkerClose", missing);
		bind(lib, a.handCreate, "MpHandLandmarkerCreate", missing);
		bind(lib, a.handDetectForVideo, "MpHandLandmarkerDetectForVideo", missing);
		bind(lib, a.handCloseResult, "MpHandLandmarkerCloseResult", missing);
		bind(lib, a.handClose, "MpHandLandmarkerClose", missing);
		bind(lib, a.gestureCreate, "MpGestureRecognizerCreate", missing);
		bind(lib, a.gestureRecognizeForVideo, "MpGestureRecognizerRecognizeForVideo", missing);
		bind(lib, a.gestureCloseResult, "MpGestureRecognizerCloseResult", missing);
		bind(lib, a.gestureClose, "MpGestureRecognizerClose", missing);
		bind(lib, a.poseCreate, "MpPoseLandmarkerCreate", missing);
		bind(lib, a.poseDetectForVideo, "MpPoseLandmarkerDetectForVideo", missing);
		bind(lib, a.poseCloseResult, "MpPoseLandmarkerCloseResult", missing);
		bind(lib, a.poseClose, "MpPoseLandmarkerClose", missing);
		bind(lib, a.holisticCreate, "MpHolisticLandmarkerCreate", missing);
		bind(lib, a.holisticDetectForVideo, "MpHolisticLandmarkerDetectForVideo", missing);
		bind(lib, a.holisticCloseResult, "MpHolisticLandmarkerCloseResult", missing);
		bind(lib, a.holisticClose, "MpHolisticLandmarkerClose", missing);
		bind(lib, a.faceDetCreate, "MpFaceDetectorCreate", missing);
		bind(lib, a.faceDetDetectForVideo, "MpFaceDetectorDetectForVideo", missing);
		bind(lib, a.faceDetCloseResult, "MpFaceDetectorCloseResult", missing);
		bind(lib, a.faceDetClose, "MpFaceDetectorClose", missing);
		bind(lib, a.segCreate, "MpImageSegmenterCreate", missing);
		bind(lib, a.segSegmentForVideo, "MpImageSegmenterSegmentForVideo", missing);
		bind(lib, a.segCloseResult, "MpImageSegmenterCloseResult", missing);
		bind(lib, a.segClose, "MpImageSegmenterClose", missing);
		bind(lib, a.segGetLabels, "MpImageSegmenterGetLabels", missing);
		if (!missing.empty())
		{
			// Leave the DLL loaded: unloading a library that may have started threads is riskier.
			gError = dll.u8string() + " is missing exports (" + missing + ") -- expected MediaPipe 1.2.2; "
				"re-run fetch_mediapipe.ps1";
			err = gError;
			return false;
		}
		gPath = dll.u8string();
		gLoaded = true;
		return true;
	}

	void setDllDir(const std::string& dir)
	{
		std::lock_guard<std::mutex> lock(gMutex);
		gDirOverride = dir;
	}

	const MpApi& get()
	{
		return gApi;
	}

	std::string loadedPath()
	{
		std::lock_guard<std::mutex> lock(gMutex);
		return gPath;
	}

	std::string takeError(char* msg)
	{
		if (!msg)
			return "unknown MediaPipe error";
		std::string s = msg;
		if (gApi.errorFree)
			gApi.errorFree(msg);
		return s;
	}
}
