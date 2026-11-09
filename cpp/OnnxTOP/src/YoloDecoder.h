// Decodes Ultralytics end2end (NMS-free) exports -- YOLO26 / YOLOv10-style detect, seg and
// pose -- from an OnnxEngine's device outputs. No TouchDesigner dependencies.
//
//   output0 [1, N, 6 + extra]  rows: x1, y1, x2, y2, conf, class, extra...
//     seg:  extra = C mask coefficients, output1 = [1, C, ph, pw] prototypes
//     pose: extra = K * (x, y, conf) keypoints
#pragma once

#include "Kernels.h"
#include "OnnxEngine.h"

#include <string>
#include <vector>

struct Detection
{
	int					cls = 0;
	float				conf = 0.f;
	float				u1 = 0.f, v1 = 0.f, u2 = 0.f, v2 = 0.f;	// TD UV of the source image (0-1, origin bottom-left)
	std::vector<float>	keypoints;	// u, v, conf triplets (pose), empty otherwise
	float				maskFill = -1.f;	// seg: fraction of the box covered by mask > threshold
};

// Seg-only knobs. Matches onnx_yolo26_seg.py: fill = pixels above maskThreshold / box area,
// both at prototype resolution. Detections are always REPORTED; the gates only decide which
// masks reach the matte, so the report threshold can sit low for ByteTrack's recovery pass.
struct SegOptions
{
	float	maskThreshold = 0.5f;
	float	matteConf = 0.35f;
	float	matteMinFill = 0.15f;
	float	maskMargin = 0.1f;		// fade-out margin outside each box, as a fraction of its size (0 = hard crop)
	float	matteNms = 0.5f;		// a mask overlapping a more confident one by more box IoU stays out of the matte (1 = off)
	bool	instances = false;		// matte encoding: G = detection index + 1, B = class + 1
};

class YoloDecoder
{
public:
	static constexpr int kMaxDets = 64;

	YoloDecoder() = default;
	~YoloDecoder();
	YoloDecoder(const YoloDecoder&) = delete;
	YoloDecoder& operator=(const YoloDecoder&) = delete;

	// Reads output0 back to the CPU (the only readback), filters/sorts detections, and for
	// seg models decodes + composites the masks into outSurf (RGBA16Float, outW x outH,
	// mapped to the SOURCE image via lb). Non-seg models clear outSurf.
	bool decode(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int classFilter,
		int maxDets, const SegOptions& seg, cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err);

	// DETR-style detect export (e.g. onnx-community YOLO26 / RT-DETR): output0 logits [1, N, C],
	// output1 boxes [1, N, 4] normalized (cx, cy, w, h). Sigmoid + argmax per query, like
	// onnx_yolo26_obj_det.py's _parse_detr_outputs(). decode() picks this automatically.
	bool decodeDetr(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int classFilter, int maxDets,
		cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err);

	// MoveNet multipose: output0 [1, 6, 56] = 6 person slots of 17 x (y, x, score) keypoints
	// (normalized to the model input), box (ymin, xmin, ymax, xmax) and person score. Fills the
	// same detections() as YOLO pose (class 0, 17 keypoints); clears outSurf.
	bool decodeMoveNet(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int maxDets,
		cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err);

	// YuNet (OpenCV Zoo face_detection_yunet_2023mar): per-stride outputs cls_/obj_/bbox_/kps_{8,16,32}
	// decoded exactly like cv2.FaceDetectorYN (score = sqrt(cls * obj), boxes/keypoints relative
	// to grid cells), then greedy NMS. 5 keypoints (eye_r, eye_l, nose, mouth_r, mouth_l), conf 1.
	bool decodeYuNet(const OnnxEngine& engine, const Letterbox& lb, float confThresh, float nmsThresh, int maxDets,
		cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err);

	const std::vector<Detection>&	detections() const { return myDetections; }
	int								keypointCount() const { return myKeypoints; }
	bool							isSegmentation() const { return mySeg; }
	void							clear() { myDetections.clear(); }

private:
	float*					myHostDet = nullptr;	// pinned
	size_t					myHostDetBytes = 0;
	SegDet*					myHostSegDets = nullptr;	// pinned, kMaxDets
	SegDet*					myDevSegDets = nullptr;		// kMaxDets
	float*					myDevMasks = nullptr;
	size_t					myDevMasksBytes = 0;
	int*					myDevCounts = nullptr;		// kMaxDets
	int*					myHostCounts = nullptr;		// pinned, kMaxDets
	bool					myInMatte[kMaxDets] = {};	// per kept seg detection: its mask reaches the matte

	std::vector<Detection>	myDetections;
	int						myKeypoints = 0;
	bool					mySeg = false;
};
