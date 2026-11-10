#include "MpWorker.h"

#include "MpApi.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace
{
	using Clock = std::chrono::steady_clock;

	MpBaseOptions baseOptions(const std::string& path)
	{
		MpBaseOptions b = {};
		b.model_asset_path = path.c_str();
		b.delegate = MP_DELEGATE_CPU;	// the Windows libmediapipe.dll is built without GPU support
		return b;
	}

	MpClassifierOptions classifierDefaults()
	{
		MpClassifierOptions c = {};
		c.max_results = -1;		// what the official Python bindings pass when unset
		c.score_threshold = 0.f;
		return c;
	}

	void copyLandmarks(const MpNormalizedLandmarks& src, std::vector<float>& dst)
	{
		dst.resize((size_t)src.landmarks_count * kLmStride);
		for (uint32_t i = 0; i < src.landmarks_count; i++)
		{
			const MpNormalizedLandmark& l = src.landmarks[i];
			float* d = &dst[(size_t)i * kLmStride];
			d[0] = l.x;
			d[1] = l.y;
			d[2] = l.z;
			d[3] = l.has_visibility ? l.visibility : 1.f;
			d[4] = l.has_presence ? l.presence : 1.f;
		}
	}

	void copyWorld(const MpLandmarks& src, std::vector<float>& dst)
	{
		dst.resize((size_t)src.landmarks_count * 3);
		for (uint32_t i = 0; i < src.landmarks_count; i++)
		{
			dst[(size_t)i * 3 + 0] = src.landmarks[i].x;
			dst[(size_t)i * 3 + 1] = src.landmarks[i].y;
			dst[(size_t)i * 3 + 2] = src.landmarks[i].z;
		}
	}

	void copyHandedness(const MpCategories& c, MpInstance& inst)
	{
		if (c.categories_count == 0)
			return;
		const MpCategory& top = c.categories[0];
		inst.label = top.category_name ? top.category_name : "";
		inst.cls = inst.label == "Right" ? 1 : 0;
		inst.conf = top.score;
	}

	void copyBlendshapes(const MpCategories& bs, MpInstance& inst, std::vector<std::string>& names)
	{
		inst.scores.resize(bs.categories_count);
		// Blendshape order is fixed by the model; take the names from the first face.
		bool setNames = names.size() != bs.categories_count;
		if (setNames)
			names.resize(bs.categories_count);
		for (uint32_t k = 0; k < bs.categories_count; k++)
		{
			inst.scores[k] = bs.categories[k].score;
			if (setNames)
				names[k] = bs.categories[k].category_name ? bs.categories[k].category_name : "";
		}
	}

	// Holistic: one part's landmarks into the packed list at `at`. A missing part (count 0) stays
	// zero with visibility = presence = 0, which the TOP and the Python side treat as "not there".
	void packPart(const MpNormalizedLandmarks& src, const MpLandmarks* world, int at, int count, MpInstance& inst)
	{
		uint32_t n = std::min<uint32_t>(src.landmarks_count, (uint32_t)count);
		for (uint32_t i = 0; i < n; i++)
		{
			const MpNormalizedLandmark& l = src.landmarks[i];
			float* d = &inst.lm[(size_t)(at + i) * kLmStride];
			d[0] = l.x;
			d[1] = l.y;
			d[2] = l.z;
			d[3] = l.has_visibility ? l.visibility : 1.f;
			d[4] = l.has_presence ? l.presence : 1.f;
		}
		if (world)
		{
			uint32_t nw = std::min<uint32_t>(world->landmarks_count, (uint32_t)count);
			for (uint32_t i = 0; i < nw; i++)
			{
				inst.world[(size_t)(at + i) * 3 + 0] = world->landmarks[i].x;
				inst.world[(size_t)(at + i) * 3 + 1] = world->landmarks[i].y;
				inst.world[(size_t)(at + i) * 3 + 2] = world->landmarks[i].z;
			}
		}
	}

	// VEC32F1 (confidence) or GRAY8 (category index) mask -> float plane, rows de-padded.
	bool copyMask(const MpApi& api, MpImagePtr m, std::vector<float>& out, int& w, int& h)
	{
		if (!m)
			return false;
		w = api.imageGetWidth(m);
		h = api.imageGetHeight(m);
		int stride = api.imageGetWidthStep(m);	// bytes per row (rows may be padded)
		char* err = nullptr;
		out.resize((size_t)w * h);
		MpImageFormat fmt = api.imageGetFormat(m);
		if (fmt == kMpImageFormatVec32F1)
		{
			const float* data = nullptr;
			if (api.imageDataFloat32(m, &data, &err) != kMpOk || !data)
			{
				mpapi::takeError(err);
				return false;
			}
			for (int y = 0; y < h; y++)
				memcpy(&out[(size_t)y * w], reinterpret_cast<const uint8_t*>(data) + (size_t)y * stride, w * sizeof(float));
			return true;
		}
		if (fmt == kMpImageFormatGray8)
		{
			const uint8_t* data = nullptr;
			if (api.imageDataUint8(m, &data, &err) != kMpOk || !data)
			{
				mpapi::takeError(err);
				return false;
			}
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					out[(size_t)y * w + x] = (float)data[(size_t)y * stride + x];
			return true;
		}
		return false;
	}
}

MpWorker::MpWorker(const MpTaskConfig& cfg) :
	myConfig(cfg)
{
	myThread = std::thread([this] { run(); });
}

MpWorker::~MpWorker()
{
	{
		std::lock_guard<std::mutex> lock(myMutex);
		myStop = true;
	}
	myWake.notify_all();
	myDone.notify_all();
	if (myThread.joinable())
		myThread.join();
}

std::string MpWorker::loadError()
{
	std::lock_guard<std::mutex> lock(myMutex);
	return myLoadError;
}

void MpWorker::submit(std::vector<uint8_t>& rgb, int w, int h, int64_t frameId)
{
	{
		std::lock_guard<std::mutex> lock(myMutex);
		if (myHasPending)
			myDropped++;
		else
			mySubmitted++;
		myPendingFrame.swap(rgb);	// caller gets the old buffer back to reuse
		myPendingW = w;
		myPendingH = h;
		myPendingId = frameId;
		myHasPending = true;
	}
	myWake.notify_one();
}

bool MpWorker::latest(MpFrameResult& out)
{
	std::lock_guard<std::mutex> lock(myMutex);
	if (myResult.seq == out.seq)
		return false;
	out = myResult;
	return true;
}

bool MpWorker::waitFor(int64_t frameId, double timeoutMs)
{
	std::unique_lock<std::mutex> lock(myMutex);
	return myDone.wait_for(lock, std::chrono::duration<double, std::milli>(timeoutMs),
		[&] { return myStop || myResult.inputFrame >= frameId; }) && !myStop;
}

void MpWorker::run()
{
	std::string err;
	if (!mpapi::load(err) || !create(err))
	{
		std::lock_guard<std::mutex> lock(myMutex);
		myLoadError = err;
		myState = State::Failed;
		return;
	}
	myState = State::Ready;

	std::vector<uint8_t> frame;
	MpFrameResult result;
	for (;;)
	{
		int w, h;
		int64_t frameId;
		{
			std::unique_lock<std::mutex> lock(myMutex);
			myWake.wait(lock, [this] { return myStop || myHasPending; });
			if (myStop)
				break;
			frame.swap(myPendingFrame);
			w = myPendingW;
			h = myPendingH;
			frameId = myPendingId;
			myHasPending = false;
		}

		// VIDEO mode needs strictly increasing timestamps; wall-clock ms keeps MediaPipe's
		// tracking/smoothing behaving as if the frames were a real-time stream.
		int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
		if (ts <= myLastTs)
			ts = myLastTs + 1;
		myLastTs = ts;

		// A MediaPipe graph error is permanent: every later frame fails too. The usual trigger
		// is a resolution change (the holistic/pose SegmentationSmoothingCalculator compares
		// against the previous mask), so rebuild the task on a size change, and after an error
		// (at most once a second, in case the error repeats). Throttle from the last rebuild, not
		// the last failure: a broken graph fails every frame, which would postpone the retry forever.
		bool sizeChanged = myTaskW >= 0 && (w != myTaskW || h != myTaskH);
		bool retryError = myTaskError && Clock::now() - myRebuildTime > std::chrono::seconds(1);
		if (sizeChanged || retryError)
		{
			myRebuildTime = Clock::now();
			destroy();
			std::string cerr;
			if (!create(cerr))
			{
				std::lock_guard<std::mutex> lock(myMutex);
				myResult.error = cerr;
				myResult.seq++;
				myTaskError = true;
				continue;
			}
		}
		myTaskW = w;
		myTaskH = h;

		auto t0 = Clock::now();
		result.instances.clear();
		result.mask.clear();
		result.maskW = result.maskH = 0;
		result.error.clear();
		myTaskError = !detect(frame.data(), w, h, ts, result) && !result.error.empty();
		result.inferMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		result.frameW = w;
		result.frameH = h;
		result.inputFrame = frameId;
		{
			std::lock_guard<std::mutex> lock(myMutex);
			result.seq = myResult.seq + 1;
			myResult = result;
		}
		myCompleted++;
		myDone.notify_all();
	}
	destroy();
}

bool MpWorker::create(std::string& err)
{
	const MpApi& api = mpapi::get();
	char* msg = nullptr;
	MpStatus st = kMpOk;
	const MpTaskConfig& c = myConfig;
	switch (c.kind)
	{
		case MpTaskKind::Face:
		{
			MpFaceLandmarkerOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.num_faces = c.maxInstances;
			o.min_face_detection_confidence = c.minDetection;
			o.min_face_presence_confidence = c.minPresence;
			o.min_tracking_confidence = c.minTracking;
			o.output_face_blendshapes = c.blendshapes;
			o.output_facial_transformation_matrixes = c.matrix;
			st = api.faceCreate(&o, reinterpret_cast<MpFaceLandmarkerPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::Hands:
		{
			MpHandLandmarkerOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.num_hands = c.maxInstances;
			o.min_hand_detection_confidence = c.minDetection;
			o.min_hand_presence_confidence = c.minPresence;
			o.min_tracking_confidence = c.minTracking;
			st = api.handCreate(&o, reinterpret_cast<MpHandLandmarkerPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::Gesture:
		{
			MpGestureRecognizerOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.num_hands = c.maxInstances;
			o.min_hand_detection_confidence = c.minDetection;
			o.min_hand_presence_confidence = c.minPresence;
			o.min_tracking_confidence = c.minTracking;
			o.canned_gestures_classifier_options = classifierDefaults();
			o.custom_gestures_classifier_options = classifierDefaults();
			st = api.gestureCreate(&o, reinterpret_cast<MpGestureRecognizerPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::Pose:
		{
			MpPoseLandmarkerOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.num_poses = c.maxInstances;
			o.min_pose_detection_confidence = c.minDetection;
			o.min_pose_presence_confidence = c.minPresence;
			o.min_tracking_confidence = c.minTracking;
			o.output_segmentation_masks = c.segmentation;
			st = api.poseCreate(&o, reinterpret_cast<MpPoseLandmarkerPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::Holistic:
		{
			MpHolisticLandmarkerOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.min_face_detection_confidence = c.minDetection;
			o.min_pose_detection_confidence = c.minDetection;
			o.min_face_presence_confidence = c.minPresence;
			o.min_pose_presence_confidence = c.minPresence;
			o.min_hand_landmarks_confidence = c.minPresence;
			o.min_face_suppression_threshold = 0.3f;	// header defaults
			o.min_pose_suppression_threshold = 0.3f;
			o.output_face_blendshapes = c.blendshapes;
			o.output_pose_segmentation_masks = c.segmentation;
			st = api.holisticCreate(&o, reinterpret_cast<MpHolisticLandmarkerPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::FaceDetector:
		{
			MpFaceDetectorOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.min_detection_confidence = c.minDetection;
			o.min_suppression_threshold = 0.5f;	// header default
			st = api.faceDetCreate(&o, reinterpret_cast<MpFaceDetectorPtr*>(&myTask), &msg);
			break;
		}
		case MpTaskKind::Segment:
		{
			MpImageSegmenterOptions o = {};
			o.base_options = baseOptions(c.modelPath);
			o.running_mode = MP_RUNNING_MODE_VIDEO;
			o.display_names_locale = "";	// must not be null: the C API builds a std::string from it
			o.output_confidence_masks = true;
			o.output_category_mask = true;
			st = api.segCreate(&o, reinterpret_cast<MpImageSegmenterPtr*>(&myTask), &msg);
			if (st == kMpOk && myTask)
			{
				MpStringList labels = {};
				char* lerr = nullptr;
				mySegLabels.clear();	// create() reruns after a size change
				if (api.segGetLabels((MpImageSegmenterPtr)myTask, &labels, &lerr) == kMpOk)
				{
					for (int i = 0; i < labels.num_strings; i++)
						mySegLabels.push_back(labels.strings[i] ? labels.strings[i] : "");
					api.stringListFree(&labels);
				}
				else
					mpapi::takeError(lerr);
			}
			break;
		}
	}
	if (st != kMpOk || !myTask)
	{
		err = "Could not load " + c.modelPath + ": " + mpapi::takeError(msg);
		myTask = nullptr;
		return false;
	}
	return true;
}

void MpWorker::destroy()
{
	if (!myTask)
		return;
	const MpApi& api = mpapi::get();
	char* msg = nullptr;
	switch (myConfig.kind)
	{
		case MpTaskKind::Face:		api.faceClose((MpFaceLandmarkerPtr)myTask, &msg); break;
		case MpTaskKind::Hands:		api.handClose((MpHandLandmarkerPtr)myTask, &msg); break;
		case MpTaskKind::Gesture:	api.gestureClose((MpGestureRecognizerPtr)myTask, &msg); break;
		case MpTaskKind::Pose:		api.poseClose((MpPoseLandmarkerPtr)myTask, &msg); break;
		case MpTaskKind::Holistic:	api.holisticClose((MpHolisticLandmarkerPtr)myTask, &msg); break;
		case MpTaskKind::FaceDetector:	api.faceDetClose((MpFaceDetectorPtr)myTask, &msg); break;
		case MpTaskKind::Segment:	api.segClose((MpImageSegmenterPtr)myTask, &msg); break;
	}
	if (msg)
		mpapi::takeError(msg);
	myTask = nullptr;
}

bool MpWorker::detect(const uint8_t* rgb, int w, int h, int64_t ts, MpFrameResult& out)
{
	const MpApi& api = mpapi::get();
	char* msg = nullptr;
	MpImagePtr image = nullptr;
	if (api.imageCreateFromUint8Data(kMpImageFormatSrgb, w, h, rgb, w * h * 3, &image, &msg) != kMpOk || !image)
	{
		out.error = "MpImage: " + mpapi::takeError(msg);
		return false;
	}

	MpStatus st = kMpOk;
	switch (myConfig.kind)
	{
		case MpTaskKind::Face:
		{
			MpFaceLandmarkerResult r = {};
			st = api.faceDetectForVideo((MpFaceLandmarkerPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			for (uint32_t i = 0; i < r.face_landmarks_count; i++)
			{
				MpInstance inst;
				copyLandmarks(r.face_landmarks[i], inst.lm);
				if (i < r.face_blendshapes_count)
					copyBlendshapes(r.face_blendshapes[i], inst, out.scoreNames);
				if (i < r.facial_transformation_matrixes_count)
				{
					const MpMatrix& m = r.facial_transformation_matrixes[i];
					if (m.rows == 4 && m.cols == 4 && m.data)
					{
						inst.matrix.resize(16);
						for (int row = 0; row < 4; row++)
							for (int col = 0; col < 4; col++)
								inst.matrix[row * 4 + col] = m.data[col * 4 + row];	// C API data is column-major
					}
				}
				out.instances.push_back(std::move(inst));
			}
			api.faceCloseResult(&r);
			break;
		}
		case MpTaskKind::Hands:
		{
			MpHandLandmarkerResult r = {};
			st = api.handDetectForVideo((MpHandLandmarkerPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			for (uint32_t i = 0; i < r.hand_landmarks_count; i++)
			{
				MpInstance inst;
				copyLandmarks(r.hand_landmarks[i], inst.lm);
				if (i < r.hand_world_landmarks_count)
					copyWorld(r.hand_world_landmarks[i], inst.world);
				if (i < r.handedness_count)
					copyHandedness(r.handedness[i], inst);
				out.instances.push_back(std::move(inst));
			}
			api.handCloseResult(&r);
			break;
		}
		case MpTaskKind::Gesture:
		{
			MpGestureRecognizerResult r = {};
			st = api.gestureRecognizeForVideo((MpGestureRecognizerPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			for (uint32_t i = 0; i < r.hand_landmarks_count; i++)
			{
				MpInstance inst;
				copyLandmarks(r.hand_landmarks[i], inst.lm);
				if (i < r.hand_world_landmarks_count)
					copyWorld(r.hand_world_landmarks[i], inst.world);
				if (i < r.handedness_count)
					copyHandedness(r.handedness[i], inst);
				if (i < r.gestures_count && r.gestures[i].categories_count > 0)
				{
					const MpCategory& g = r.gestures[i].categories[0];
					inst.label = g.category_name ? g.category_name : "";
					inst.gestureScore = g.score;
				}
				out.instances.push_back(std::move(inst));
			}
			api.gestureCloseResult(&r);
			break;
		}
		case MpTaskKind::Pose:
		{
			MpPoseLandmarkerResult r = {};
			st = api.poseDetectForVideo((MpPoseLandmarkerPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			for (uint32_t i = 0; i < r.pose_landmarks_count; i++)
			{
				MpInstance inst;
				copyLandmarks(r.pose_landmarks[i], inst.lm);
				if (i < r.pose_world_landmarks_count)
					copyWorld(r.pose_world_landmarks[i], inst.world);
				float sum = 0.f;
				int n = (int)inst.lm.size() / kLmStride;
				for (int k = 0; k < n; k++)
					sum += inst.lm[(size_t)k * kLmStride + 3];
				inst.conf = n ? sum / n : 0.f;
				out.instances.push_back(std::move(inst));
			}
			if (r.segmentation_masks_count > 0 && !copyMask(api, r.segmentation_masks[0], out.mask, out.maskW, out.maskH))
				out.mask.clear();
			api.poseCloseResult(&r);
			break;
		}
		case MpTaskKind::Holistic:
		{
			MpHolisticLandmarkerResult r = {};
			st = api.holisticDetectForVideo((MpHolisticLandmarkerPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			bool any = r.pose_landmarks.landmarks_count || r.face_landmarks.landmarks_count ||
				r.left_hand_landmarks.landmarks_count || r.right_hand_landmarks.landmarks_count;
			if (any)
			{
				MpInstance inst;
				inst.lm.assign((size_t)kHolisticCount * kLmStride, 0.f);
				inst.world.assign((size_t)kHolisticCount * 3, 0.f);
				packPart(r.pose_landmarks, &r.pose_world_landmarks, kHolisticPose, 33, inst);
				packPart(r.left_hand_landmarks, &r.left_hand_world_landmarks, kHolisticLeftHand, 21, inst);
				packPart(r.right_hand_landmarks, &r.right_hand_world_landmarks, kHolisticRightHand, 21, inst);
				packPart(r.face_landmarks, nullptr, kHolisticFace, 478, inst);
				float sum = 0.f;
				for (int k = 0; k < 33; k++)
					sum += inst.lm[(size_t)(kHolisticPose + k) * kLmStride + 3];
				inst.conf = r.pose_landmarks.landmarks_count ? sum / 33.f : 1.f;
				if (r.face_blendshapes.categories_count)
					copyBlendshapes(r.face_blendshapes, inst, out.scoreNames);
				out.instances.push_back(std::move(inst));
			}
			if (r.pose_segmentation_mask && !copyMask(api, r.pose_segmentation_mask, out.mask, out.maskW, out.maskH))
				out.mask.clear();
			api.holisticCloseResult(&r);
			break;
		}
		case MpTaskKind::FaceDetector:
		{
			MpFaceDetectorResult r = {};
			st = api.faceDetDetectForVideo((MpFaceDetectorPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			for (uint32_t i = 0; i < r.detections_count && (int)i < myConfig.maxInstances; i++)
			{
				const MpDetection& d = r.detections[i];
				MpInstance inst;
				inst.conf = d.categories_count ? d.categories[0].score : 0.f;
				inst.hasBox = true;
				inst.box[0] = (float)d.bounding_box.left / w;	// MpRect is in pixels
				inst.box[1] = (float)d.bounding_box.top / h;
				inst.box[2] = (float)d.bounding_box.right / w;
				inst.box[3] = (float)d.bounding_box.bottom / h;
				inst.lm.assign((size_t)d.keypoints_count * kLmStride, 1.f);
				for (uint32_t k = 0; k < d.keypoints_count; k++)
				{
					inst.lm[(size_t)k * kLmStride + 0] = d.keypoints[k].x;
					inst.lm[(size_t)k * kLmStride + 1] = d.keypoints[k].y;
					inst.lm[(size_t)k * kLmStride + 2] = 0.f;
				}
				out.instances.push_back(std::move(inst));
			}
			api.faceDetCloseResult(&r);
			break;
		}
		case MpTaskKind::Segment:
		{
			MpImageSegmenterResult r = {};
			st = api.segSegmentForVideo((MpImageSegmenterPtr)myTask, image, nullptr, ts, &r, &msg);
			if (st != kMpOk)
				break;
			// One Info DAT row per class with its mean confidence (= fraction of the frame covered).
			std::vector<float> plane;
			int pw = 0, ph = 0;
			for (uint32_t i = 0; i < r.confidence_masks_count; i++)
			{
				MpInstance inst;
				inst.cls = (int)i;
				inst.label = i < mySegLabels.size() && !mySegLabels[i].empty() ? mySegLabels[i] : "class" + std::to_string(i);
				if (copyMask(api, r.confidence_masks[i], plane, pw, ph) && !plane.empty())
				{
					double sum = 0.0;
					for (float v : plane)
						sum += v;
					inst.conf = (float)(sum / plane.size());
				}
				out.instances.push_back(std::move(inst));
			}
			// The TOP's mask: foreground (1 - background, or the only mask), one class, or the
			// category index map.
			MpSegOutput mode = (MpSegOutput)mySegOutput.load();
			int n = (int)r.confidence_masks_count;
			if (mode == MpSegOutput::Category && r.has_category_mask)
				copyMask(api, r.category_mask, out.mask, out.maskW, out.maskH);
			else if (n > 0)
			{
				int idx = mode == MpSegOutput::Class ? std::clamp(mySegClass.load(), 0, n - 1) : 0;
				if (copyMask(api, r.confidence_masks[idx], out.mask, out.maskW, out.maskH) &&
					mode != MpSegOutput::Class && n > 1)
				{
					for (float& v : out.mask)
						v = 1.f - v;	// class 0 is background in MediaPipe's multi-mask segmenters
				}
			}
			api.segCloseResult(&r);
			break;
		}
	}
	api.imageFree(image);
	if (st != kMpOk)
	{
		out.error = mpapi::takeError(msg);
		return false;
	}
	return true;
}
