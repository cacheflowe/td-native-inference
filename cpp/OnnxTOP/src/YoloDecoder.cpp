#include "YoloDecoder.h"

#include <algorithm>
#include <cmath>
#include <string>

YoloDecoder::~YoloDecoder()
{
	if (myHostDet)
		cudaFreeHost(myHostDet);
	if (myHostSegDets)
		cudaFreeHost(myHostSegDets);
	if (myDevSegDets)
		cudaFree(myDevSegDets);
	if (myDevMasks)
		cudaFree(myDevMasks);
	if (myDevCounts)
		cudaFree(myDevCounts);
	if (myHostCounts)
		cudaFreeHost(myHostCounts);
}

bool YoloDecoder::decode(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int classFilter,
	int maxDets, const SegOptions& segOpt, cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err)
{
	myDetections.clear();
	myKeypoints = 0;
	mySeg = false;

	const auto& outs = engine.outputs();
	const TensorSlot& det = outs[0];
	// DETR-style detect exports have separate logits + boxes outputs instead of [1, N, 6+] rows.
	if (outs.size() == 2 && det.shape.size() == 3 && outs[1].shape.size() == 3 && outs[1].shape[2] == 4 &&
		outs[1].shape[1] == det.shape[1] && det.shape[2] != 4)
		return decodeDetr(engine, lb, confThresh, classFilter, maxDets, outSurf, outW, outH, stream, err);
	if (det.shape.size() != 3 || det.shape[2] < 6 || det.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		err = "YOLO mode expects output0 float32 [1, N, 6+] (Ultralytics end2end export), got " +
			shapeString(det.shape) + " " + ortTypeName(det.type);
		launchClear(outSurf, outW, outH, 8, stream);
		return false;
	}
	const int N = (int)det.shape[1];
	const int S = (int)det.shape[2];
	const int modelW = lb.dstW, modelH = lb.dstH;

	const TensorSlot* protos = nullptr;
	int numCoef = 0, ph = 0, pw = 0;
	if (outs.size() >= 2 && outs[1].shape.size() == 4 && outs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		numCoef = (int)outs[1].shape[1];
		ph = (int)outs[1].shape[2];
		pw = (int)outs[1].shape[3];
		if (S >= 6 + numCoef)
			protos = &outs[1];
	}
	mySeg = protos != nullptr;
	myKeypoints = (!mySeg && S > 6 && (S - 6) % 3 == 0) ? (S - 6) / 3 : 0;

	// [1, N, 6+] alone also matches image-shaped outputs (e.g. depth [1, 518, 518]), which would
	// decode as garbage. A real end2end row is det (6), seg (6 + mask coefs), or pose (6 + K*3).
	// Bounds too: end2end exports keep <= 300 rows, and 133 keypoints is COCO-WholeBody.
	if ((S != 6 && !mySeg && myKeypoints == 0) || N > 1000 || myKeypoints > 133)
	{
		err = "Output0 " + shapeString(det.shape) + " doesn't look like a YOLO end2end export (rows of 6, 6 + mask "
			"coefficients, or 6 + keypoints*3) -- for image-to-image models (e.g. Depth Anything) set Output Mode to Raw";
		mySeg = false;
		launchClear(outSurf, outW, outH, 8, stream);
		return false;
	}

	// The only device->host copy in the pipeline: 300 x 38 floats ~ 45 KB for seg.
	size_t bytes = (size_t)N * S * sizeof(float);
	if (bytes > myHostDetBytes)
	{
		if (myHostDet)
			cudaFreeHost(myHostDet);
		cudaMallocHost(&myHostDet, bytes);
		myHostDetBytes = bytes;
	}
	cudaMemcpyAsync(myHostDet, det.device, bytes, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Boxes are pixel-space in some exports and normalized in others (same auto-detect as
	// onnx_yolo26_seg.py's box_max check).
	float boxMax = 0.f;
	for (int r = 0; r < N; r++)
	{
		const float* row = myHostDet + (size_t)r * S;
		if (row[4] >= confThresh)
			boxMax = std::max({ boxMax, row[0], row[1], row[2], row[3] });
	}
	const float bx = boxMax <= 1.5f ? (float)modelW : 1.f;
	const float by = boxMax <= 1.5f ? (float)modelH : 1.f;

	struct Kept { int row; float conf; };
	std::vector<Kept> kept;
	for (int r = 0; r < N; r++)
	{
		const float* row = myHostDet + (size_t)r * S;
		if (row[4] < confThresh)
			continue;
		if (classFilter >= 0 && (int)row[5] != classFilter)
			continue;
		kept.push_back({ r, row[4] });
	}
	std::sort(kept.begin(), kept.end(), [](const Kept& a, const Kept& b) { return a.conf > b.conf; });
	if ((int)kept.size() > std::min(maxDets, kMaxDets))
		kept.resize(std::min(maxDets, kMaxDets));

	auto toU = [&](float mx) { return std::clamp((mx - lb.padX) / lb.scaleX / lb.srcW, 0.f, 1.f); };
	auto toV = [&](float my) { return std::clamp(1.f - (my - lb.padY) / lb.scaleY / lb.srcH, 0.f, 1.f); };

	if (mySeg && !myDevSegDets)
	{
		cudaMallocHost(&myHostSegDets, kMaxDets * sizeof(SegDet));
		cudaMalloc(&myDevSegDets, kMaxDets * sizeof(SegDet));
		cudaMalloc(&myDevCounts, kMaxDets * sizeof(int));
		cudaMallocHost(&myHostCounts, kMaxDets * sizeof(int));
	}

	for (size_t k = 0; k < kept.size(); k++)
	{
		const float* row = myHostDet + (size_t)kept[k].row * S;
		float x1 = row[0] * bx, y1 = row[1] * by, x2 = row[2] * bx, y2 = row[3] * by;

		Detection d;
		d.cls = (int)row[5];
		d.conf = row[4];
		d.u1 = toU(x1);
		d.u2 = toU(x2);
		d.v1 = toV(y2);	// image-bottom edge -> lower v
		d.v2 = toV(y1);
		for (int i = 0; i < myKeypoints; i++)
		{
			const float* kp = row + 6 + i * 3;
			d.keypoints.push_back(toU(kp[0] * bx));
			d.keypoints.push_back(toV(kp[1] * by));
			d.keypoints.push_back(kp[2]);
		}
		myDetections.push_back(std::move(d));

		if (mySeg)
			myHostSegDets[k] = { x1, y1, x2, y2, kept[k].row };
	}

	if (!mySeg)
	{
		// Detect / pose: results live in detections(); image output is empty.
		launchClear(outSurf, outW, outH, 8, stream);
		return true;
	}

	if (!myDevMasks || myDevMasksBytes < (size_t)kMaxDets * ph * pw * sizeof(float))
	{
		// Sized for the cap once, so a changing detection count never reallocates.
		if (myDevMasks)
			cudaFree(myDevMasks);
		myDevMasksBytes = (size_t)kMaxDets * ph * pw * sizeof(float);
		cudaMalloc(&myDevMasks, myDevMasksBytes);
	}

	int numDets = (int)kept.size();
	if (numDets > 0)
	{
		cudaMemcpyAsync(myDevSegDets, myHostSegDets, numDets * sizeof(SegDet), cudaMemcpyHostToDevice, stream);
		launchProtoMasks((const float*)det.device, S, 6, myDevSegDets, numDets,
			(const float*)protos->device, numCoef, ph, pw, modelW, modelH, myDevMasks, stream);

		// Fill ratio per mask (numDets ints read back), same definition as onnx_yolo26_seg.py:
		// pixels above maskThreshold / the box's own area, both at prototype resolution.
		launchMaskCounts(myDevMasks, numDets, ph, pw, segOpt.maskThreshold, myDevCounts, stream);
		cudaMemcpyAsync(myHostCounts, myDevCounts, numDets * sizeof(int), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		const size_t plane = (size_t)ph * pw;
		for (int k = 0; k < numDets; k++)
		{
			const SegDet& sd = myHostSegDets[k];
			auto px = [&](float v, float model, int res, bool up) {
				float f = std::clamp(v / model, 0.f, 1.f) * res;
				return std::clamp(up ? (int)std::ceil(f) : (int)f, 0, res);
			};
			int area = (px(sd.x2, (float)modelW, pw, true) - px(sd.x1, (float)modelW, pw, false)) *
				(px(sd.y2, (float)modelH, ph, true) - px(sd.y1, (float)modelH, ph, false));
			Detection& d = myDetections[myDetections.size() - numDets + k];
			d.maskFill = (float)myHostCounts[k] / (float)std::max(area, 1);
			// Reported either way; only gated masks are left out of the matte.
			if (d.conf < segOpt.matteConf || d.maskFill <= segOpt.matteMinFill)
				cudaMemsetAsync(myDevMasks + k * plane, 0, plane * sizeof(float), stream);
		}
	}
	cudaError_t ce = launchComposeMatte(myDevMasks, numDets, ph, pw, lb, outSurf, outW, outH, stream);
	if (ce != cudaSuccess)
	{
		err = std::string("Mask kernel failed: ") + cudaGetErrorString(ce);
		return false;
	}
	return true;
}

bool YoloDecoder::decodeMoveNet(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int maxDets,
	cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err)
{
	constexpr int K = 17, S = K * 3 + 5;
	myDetections.clear();
	myKeypoints = 0;
	mySeg = false;
	launchClear(outSurf, outW, outH, 8, stream);

	const TensorSlot& det = engine.outputs()[0];
	const auto& sh = det.shape;
	if (sh.size() != 3 || sh[2] != S || sh[1] <= 0 || sh[1] > 64 || det.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		err = "MoveNet mode expects output0 float32 [1, N, 56] (MoveNet multipose), got " +
			shapeString(sh) + " " + ortTypeName(det.type);
		return false;
	}
	const int N = (int)sh[1];
	size_t bytes = (size_t)N * S * sizeof(float);
	if (bytes > myHostDetBytes)
	{
		if (myHostDet)
			cudaFreeHost(myHostDet);
		cudaMallocHost(&myHostDet, bytes);
		myHostDetBytes = bytes;
	}
	cudaMemcpyAsync(myHostDet, det.device, bytes, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	myKeypoints = K;

	// Coordinates are 0-1 of the model input (padding included): -> model px -> source UV.
	const float mw = (float)lb.dstW, mh = (float)lb.dstH;
	auto toU = [&](float nx) { return std::clamp((nx * mw - lb.padX) / lb.scaleX / lb.srcW, 0.f, 1.f); };
	auto toV = [&](float ny) { return std::clamp(1.f - (ny * mh - lb.padY) / lb.scaleY / lb.srcH, 0.f, 1.f); };

	struct Kept { int row; float conf; };
	std::vector<Kept> kept;
	for (int r = 0; r < N; r++)
	{
		float conf = myHostDet[(size_t)r * S + K * 3 + 4];
		if (conf >= confThresh)
			kept.push_back({ r, conf });
	}
	std::sort(kept.begin(), kept.end(), [](const Kept& a, const Kept& b) { return a.conf > b.conf; });
	if ((int)kept.size() > std::min(maxDets, kMaxDets))
		kept.resize(std::min(maxDets, kMaxDets));

	for (const Kept& k : kept)
	{
		const float* row = myHostDet + (size_t)k.row * S;
		const float* box = row + K * 3;	// ymin, xmin, ymax, xmax
		Detection d;
		d.cls = 0;
		d.conf = k.conf;
		d.u1 = toU(box[1]);
		d.u2 = toU(box[3]);
		d.v1 = toV(box[2]);	// image-bottom edge -> lower v
		d.v2 = toV(box[0]);
		d.keypoints.reserve(K * 3);
		for (int i = 0; i < K; i++)
		{
			const float* kp = row + i * 3;	// y, x, score
			d.keypoints.push_back(toU(kp[1]));
			d.keypoints.push_back(toV(kp[0]));
			d.keypoints.push_back(kp[2]);
		}
		myDetections.push_back(std::move(d));
	}
	return true;
}

namespace
{
	float iou(const float* a, const float* b)	// x1, y1, x2, y2
	{
		float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
		float iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
		float inter = ix * iy;
		float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
		return uni > 0.f ? inter / uni : 0.f;
	}
}

bool YoloDecoder::decodeYuNet(const OnnxEngine& engine, const Letterbox& lb, float confThresh, float nmsThresh,
	int maxDets, cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err)
{
	constexpr int K = 5;
	const int strides[3] = { 8, 16, 32 };
	myDetections.clear();
	myKeypoints = 0;
	mySeg = false;
	launchClear(outSurf, outW, outH, 8, stream);

	// Find cls/obj/bbox/kps per stride by output name; check shapes against the input grid.
	const auto& outs = engine.outputs();
	auto find = [&](const std::string& name) -> const TensorSlot* {
		for (const auto& o : outs)
			if (o.name == name)
				return &o;
		return nullptr;
	};
	const TensorSlot* slots[3][4] = {};
	const char* kinds[4] = { "cls_", "obj_", "bbox_", "kps_" };
	const int widths[4] = { 1, 1, 4, K * 2 };
	size_t total = 0;
	for (int si = 0; si < 3; si++)
	{
		int cells = (lb.dstW / strides[si]) * (lb.dstH / strides[si]);
		for (int k = 0; k < 4; k++)
		{
			const TensorSlot* t = find(kinds[k] + std::to_string(strides[si]));
			if (!t || t->type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || t->elementCount() != (size_t)cells * widths[k])
			{
				err = "YuNet mode expects outputs cls_/obj_/bbox_/kps_{8,16,32} (OpenCV Zoo face_detection_yunet), "
					"missing or mis-shaped: " + std::string(kinds[k]) + std::to_string(strides[si]);
				return false;
			}
			slots[si][k] = t;
			total += t->elementCount();
		}
	}

	// One readback of everything (~0.5 MB at 640x640), then decode on the CPU.
	size_t bytes = total * sizeof(float);
	if (bytes > myHostDetBytes)
	{
		if (myHostDet)
			cudaFreeHost(myHostDet);
		cudaMallocHost(&myHostDet, bytes);
		myHostDetBytes = bytes;
	}
	const float* host[3][4];
	size_t off = 0;
	for (int si = 0; si < 3; si++)
		for (int k = 0; k < 4; k++)
		{
			size_t n = slots[si][k]->elementCount();
			cudaMemcpyAsync(myHostDet + off, slots[si][k]->device, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
			host[si][k] = myHostDet + off;
			off += n;
		}
	cudaStreamSynchronize(stream);
	myKeypoints = K;

	struct Cand { float box[4]; float kps[K * 2]; float score; };
	std::vector<Cand> cands;
	for (int si = 0; si < 3; si++)
	{
		const int s = strides[si], cols = lb.dstW / s, rows = lb.dstH / s;
		const float *cls = host[si][0], *obj = host[si][1], *bb = host[si][2], *kp = host[si][3];
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < cols; c++)
			{
				int idx = r * cols + c;
				float score = std::sqrt(std::clamp(cls[idx], 0.f, 1.f) * std::clamp(obj[idx], 0.f, 1.f));
				if (score < confThresh)
					continue;
				Cand cd;
				float cx = (c + bb[idx * 4 + 0]) * s, cy = (r + bb[idx * 4 + 1]) * s;
				float w = std::exp(bb[idx * 4 + 2]) * s, h = std::exp(bb[idx * 4 + 3]) * s;
				cd.box[0] = cx - w * 0.5f;
				cd.box[1] = cy - h * 0.5f;
				cd.box[2] = cx + w * 0.5f;
				cd.box[3] = cy + h * 0.5f;
				for (int n = 0; n < K; n++)
				{
					cd.kps[n * 2] = (kp[idx * K * 2 + n * 2] + c) * s;
					cd.kps[n * 2 + 1] = (kp[idx * K * 2 + n * 2 + 1] + r) * s;
				}
				cd.score = score;
				cands.push_back(cd);
			}
	}
	std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });

	auto toU = [&](float mx) { return std::clamp((mx - lb.padX) / lb.scaleX / lb.srcW, 0.f, 1.f); };
	auto toV = [&](float my) { return std::clamp(1.f - (my - lb.padY) / lb.scaleY / lb.srcH, 0.f, 1.f); };
	std::vector<const Cand*> kept;
	const int cap = std::min(maxDets, kMaxDets);
	for (const Cand& cd : cands)
	{
		bool suppressed = false;
		for (const Cand* k : kept)
			if (iou(cd.box, k->box) > nmsThresh)
			{
				suppressed = true;
				break;
			}
		if (suppressed)
			continue;
		kept.push_back(&cd);
		Detection d;
		d.cls = 0;
		d.conf = cd.score;
		d.u1 = toU(cd.box[0]);
		d.u2 = toU(cd.box[2]);
		d.v1 = toV(cd.box[3]);
		d.v2 = toV(cd.box[1]);
		for (int n = 0; n < K; n++)
		{
			d.keypoints.push_back(toU(cd.kps[n * 2]));
			d.keypoints.push_back(toV(cd.kps[n * 2 + 1]));
			d.keypoints.push_back(1.f);
		}
		myDetections.push_back(std::move(d));
		if ((int)kept.size() >= cap)
			break;
	}
	return true;
}

bool YoloDecoder::decodeDetr(const OnnxEngine& engine, const Letterbox& lb, float confThresh, int classFilter,
	int maxDets, cudaSurfaceObject_t outSurf, int outW, int outH, cudaStream_t stream, std::string& err)
{
	myDetections.clear();
	myKeypoints = 0;
	mySeg = false;
	launchClear(outSurf, outW, outH, 8, stream);

	const TensorSlot& lg = engine.outputs()[0];
	const TensorSlot& bx = engine.outputs()[1];
	if (lg.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || bx.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		err = "DETR-style outputs must be float32, got " + std::string(ortTypeName(lg.type)) + " / " + ortTypeName(bx.type);
		return false;
	}
	const int N = (int)lg.shape[1], C = (int)lg.shape[2];
	size_t nl = (size_t)N * C, nb = (size_t)N * 4, bytes = (nl + nb) * sizeof(float);
	if (bytes > myHostDetBytes)
	{
		if (myHostDet)
			cudaFreeHost(myHostDet);
		cudaMallocHost(&myHostDet, bytes);
		myHostDetBytes = bytes;
	}
	cudaMemcpyAsync(myHostDet, lg.device, nl * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(myHostDet + nl, bx.device, nb * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	const float* boxes = myHostDet + nl;

	struct Kept { int q; int cls; float conf; };
	std::vector<Kept> kept;
	for (int q = 0; q < N; q++)
	{
		const float* l = myHostDet + (size_t)q * C;
		int best = (int)(std::max_element(l, l + C) - l);	// sigmoid is monotonic: argmax on logits
		float conf = 1.f / (1.f + std::exp(-l[best]));
		if (conf < confThresh || (classFilter >= 0 && best != classFilter))
			continue;
		kept.push_back({ q, best, conf });
	}
	std::sort(kept.begin(), kept.end(), [](const Kept& a, const Kept& b) { return a.conf > b.conf; });
	if ((int)kept.size() > std::min(maxDets, kMaxDets))
		kept.resize(std::min(maxDets, kMaxDets));

	const float mw = (float)lb.dstW, mh = (float)lb.dstH;
	auto toU = [&](float nx) { return std::clamp((nx * mw - lb.padX) / lb.scaleX / lb.srcW, 0.f, 1.f); };
	auto toV = [&](float ny) { return std::clamp(1.f - (ny * mh - lb.padY) / lb.scaleY / lb.srcH, 0.f, 1.f); };
	for (const Kept& k : kept)
	{
		const float* b = boxes + (size_t)k.q * 4;	// cx, cy, w, h (0-1 of the model input)
		Detection d;
		d.cls = k.cls;
		d.conf = k.conf;
		d.u1 = toU(b[0] - b[2] * 0.5f);
		d.u2 = toU(b[0] + b[2] * 0.5f);
		d.v1 = toV(b[1] + b[3] * 0.5f);	// image-bottom edge -> lower v
		d.v2 = toV(b[1] - b[3] * 0.5f);
		myDetections.push_back(std::move(d));
	}
	return true;
}
