// MpApi -- runtime binding to Google's prebuilt MediaPipe Tasks C API (libmediapipe.dll).
//
// The DLL ships in the official mediapipe pip wheel (mediapipe/tasks/c/libmediapipe.dll) and has
// no import library, so every entry point is resolved with GetProcAddress. The function pointer
// types come straight from the vendored v1.2.2 headers via decltype, so a signature can't drift
// from the header it was declared in. The Windows build is CPU-only (XNNPACK): it rejects the GPU
// delegate with "GPU processing is disabled in build flags".
#pragma once

#ifndef MP_EXPORT
#define MP_EXPORT	// declarations only; nothing here is linked against
#endif

#include "mediapipe/tasks/c/components/containers/keypoint.h"	// detection_result.h only forward-declares it
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"

#include <string>

struct MpApi
{
	decltype(&MpErrorFree)							errorFree = nullptr;
	decltype(&MpStringListFree)						stringListFree = nullptr;

	decltype(&MpImageCreateFromUint8Data)			imageCreateFromUint8Data = nullptr;
	decltype(&MpImageFree)							imageFree = nullptr;
	decltype(&MpImageGetWidth)						imageGetWidth = nullptr;
	decltype(&MpImageGetHeight)						imageGetHeight = nullptr;
	decltype(&MpImageGetFormat)						imageGetFormat = nullptr;
	decltype(&MpImageDataFloat32)					imageDataFloat32 = nullptr;
	decltype(&MpImageDataUint8)						imageDataUint8 = nullptr;
	decltype(&MpImageGetWidthStep)					imageGetWidthStep = nullptr;

	decltype(&MpFaceLandmarkerCreate)				faceCreate = nullptr;
	decltype(&MpFaceLandmarkerDetectForVideo)		faceDetectForVideo = nullptr;
	decltype(&MpFaceLandmarkerCloseResult)			faceCloseResult = nullptr;
	decltype(&MpFaceLandmarkerClose)				faceClose = nullptr;

	decltype(&MpHandLandmarkerCreate)				handCreate = nullptr;
	decltype(&MpHandLandmarkerDetectForVideo)		handDetectForVideo = nullptr;
	decltype(&MpHandLandmarkerCloseResult)			handCloseResult = nullptr;
	decltype(&MpHandLandmarkerClose)				handClose = nullptr;

	decltype(&MpGestureRecognizerCreate)			gestureCreate = nullptr;
	decltype(&MpGestureRecognizerRecognizeForVideo)	gestureRecognizeForVideo = nullptr;
	decltype(&MpGestureRecognizerCloseResult)		gestureCloseResult = nullptr;
	decltype(&MpGestureRecognizerClose)				gestureClose = nullptr;

	decltype(&MpPoseLandmarkerCreate)				poseCreate = nullptr;
	decltype(&MpPoseLandmarkerDetectForVideo)		poseDetectForVideo = nullptr;
	decltype(&MpPoseLandmarkerCloseResult)			poseCloseResult = nullptr;
	decltype(&MpPoseLandmarkerClose)				poseClose = nullptr;

	decltype(&MpHolisticLandmarkerCreate)			holisticCreate = nullptr;
	decltype(&MpHolisticLandmarkerDetectForVideo)	holisticDetectForVideo = nullptr;
	decltype(&MpHolisticLandmarkerCloseResult)		holisticCloseResult = nullptr;
	decltype(&MpHolisticLandmarkerClose)			holisticClose = nullptr;

	decltype(&MpFaceDetectorCreate)					faceDetCreate = nullptr;
	decltype(&MpFaceDetectorDetectForVideo)			faceDetDetectForVideo = nullptr;
	decltype(&MpFaceDetectorCloseResult)			faceDetCloseResult = nullptr;
	decltype(&MpFaceDetectorClose)					faceDetClose = nullptr;

	decltype(&MpImageSegmenterCreate)				segCreate = nullptr;
	decltype(&MpImageSegmenterSegmentForVideo)		segSegmentForVideo = nullptr;
	decltype(&MpImageSegmenterCloseResult)			segCloseResult = nullptr;
	decltype(&MpImageSegmenterClose)				segClose = nullptr;
	decltype(&MpImageSegmenterGetLabels)			segGetLabels = nullptr;
};

namespace mpapi
{
	// Loads libmediapipe.dll from the folder this plugin DLL lives in (once per process; thread-safe).
	bool			load(std::string& err);
	void			setDllDir(const std::string& dir);	// override the folder (test tool); call before load()
	const MpApi&	get();				// valid after load() returned true
	std::string		loadedPath();

	// TD resolves a relative file parameter against the project folder. When that file doesn't
	// exist, returns the same relative path under this plugin DLL's folder if it exists there
	// (an installed Plugins/<package>/models/ folder); otherwise returns `resolved` unchanged and
	// describes both places looked in via `missing` (empty when the file was found).
	std::string		resolveModelPath(const char* resolved, const char* raw, std::string& missing);

	// Takes ownership of a C API error string (may be null) and returns it as std::string.
	std::string		takeError(char* msg);
}
