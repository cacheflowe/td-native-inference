#include "Holistic.h"

#include <algorithm>
#include <cmath>

namespace holistic
{
	namespace
	{
		constexpr float kPi = 3.14159265358979f;
		constexpr float kPosePresence = 0.5f, kHandPresence = 0.5f, kFacePresence = 0.5f;
		constexpr float kWristVisibility = 0.1f;	// hand_visibility_from_hand_landmarks_from_pose
		constexpr float kPoseRoiScale = 1.25f, kBoxRoiScale = 1.25f;
		constexpr float kDuplicateIou = 0.5f, kCoveredIou = 0.3f;
		constexpr int kHeatmapKernel = 7;
		constexpr float kHeatmapMinConfidence = 0.5f;
		// wrist, pinky, index per side (left, right)
		constexpr int kHandPose[2][3] = { { 15, 17, 19 }, { 16, 18, 20 } };
		constexpr int kFacePose[] = { 0, 2, 5, 7, 8, 9, 10 };	// nose, eyes, ears, mouth corners
		constexpr int kPalm[] = { 0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18 };

		float sigmoid(float v) { return 1.f / (1.f + std::exp(-std::clamp(v, -60.f, 60.f))); }
		float normalizeRadians(float a) { return a - 2.f * kPi * std::floor((a + kPi) / (2.f * kPi)); }

		float iou(const float* a, const float* b)
		{
			float ix = std::min(a[2], b[2]) - std::max(a[0], b[0]);
			float iy = std::min(a[3], b[3]) - std::max(a[1], b[1]);
			if (ix <= 0.f || iy <= 0.f)
				return 0.f;
			float inter = ix * iy;
			return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
		}
	}

	// RectTransformationCalculator with square_long.
	Roi Tracker::transform(Roi r, float scale, float shiftY) const
	{
		if (r.angle == 0.f)
			r.y += r.h * shiftY;
		else
		{
			r.x += (-myH * r.h * shiftY * std::sin(r.angle)) / myW;
			r.y += (myH * r.h * shiftY * std::cos(r.angle)) / myH;
		}
		float side = std::max(r.w * myW, r.h * myH);
		r.w = side / myW * scale;
		r.h = side / myH * scale;
		return r;
	}

	// AlignmentPointsRectsCalculator (center x0 y0, scale point x1 y1) + RectTransformation.
	Roi Tracker::alignment(float x0, float y0, float x1, float y1, float targetDegrees, float scale, float shiftY) const
	{
		float px0 = x0 * myW, py0 = y0 * myH, px1 = x1 * myW, py1 = y1 * myH;
		float size = 2.f * std::hypot(px1 - px0, py1 - py0);
		Roi r{ x0, y0, size / myW, size / myH,
			normalizeRadians(targetDegrees * kPi / 180.f - std::atan2(-(py1 - py0), px1 - px0)) };
		return transform(r, scale, shiftY);
	}

	// LandmarkProjectionCalculator: crop-normalized -> image-normalized.
	Point Tracker::project(float nx, float ny, const Roi& r) const
	{
		float x = nx - 0.5f, y = ny - 0.5f, c = std::cos(r.angle), s = std::sin(r.angle);
		return { (c * x - s * y) * r.w + r.x, (s * x + c * y) * r.h + r.y };
	}

	// HandLandmarksToRectCalculator on the palm subset + RectTransformation (scale 2, shift -0.1).
	Roi Tracker::palmRoi(const std::vector<Point>& all) const
	{
		Point p[12];
		for (int i = 0; i < 12; i++)
			p[i] = all[(size_t)kPalm[i]];
		float x0 = p[0].x * myW, y0 = p[0].y * myH;
		float x1 = ((p[4].x + p[8].x) * 0.5f + p[6].x) * 0.5f * myW;
		float y1 = ((p[4].y + p[8].y) * 0.5f + p[6].y) * 0.5f * myH;
		float angle = normalizeRadians(kPi * 0.5f - std::atan2(-(y1 - y0), x1 - x0));
		float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
		for (const Point& q : p)
		{
			minX = std::min(minX, q.x);
			maxX = std::max(maxX, q.x);
			minY = std::min(minY, q.y);
			maxY = std::max(maxY, q.y);
		}
		float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f;
		float c = std::cos(-angle), s = std::sin(-angle);
		float pminX = 1e9f, pmaxX = -1e9f, pminY = 1e9f, pmaxY = -1e9f;
		for (const Point& q : p)
		{
			float ox = (q.x - cx) * myW, oy = (q.y - cy) * myH;
			float rx = ox * c - oy * s, ry = ox * s + oy * c;
			pminX = std::min(pminX, rx);
			pmaxX = std::max(pmaxX, rx);
			pminY = std::min(pminY, ry);
			pmaxY = std::max(pmaxY, ry);
		}
		float pcx = (pminX + pmaxX) * 0.5f, pcy = (pminY + pmaxY) * 0.5f;
		float ca = std::cos(angle), sa = std::sin(angle);
		Roi r{ (pcx * ca - pcy * sa + myW * cx) / myW, (pcx * sa + pcy * ca + myH * cy) / myH,
			(pmaxX - pminX) / myW, (pmaxY - pminY) / myH, angle };
		return transform(r, 2.f, -0.1f);
	}

	// RoiTrackingCalculator: keep last frame's landmark ROI while it agrees with the recrop ROI.
	bool Tracker::keepTracking(const Roi& prev, const std::vector<Point>& pts, const Roi& recrop,
		float rotDeg, float translation, float scale, float margin) const
	{
		float c = std::cos(-recrop.angle), s = std::sin(-recrop.angle);
		auto rx = [&](float x, float y) { return x * myW * c - y * myH * s; };
		auto ry = [&](float x, float y) { return x * myW * s + y * myH * c; };
		float diff = (prev.angle - recrop.angle) * 180.f / kPi;
		diff = std::fmod(diff + 540.f, 360.f) - 180.f;
		if (std::fabs(diff) > rotDeg)
			return false;
		if (std::fabs(rx(prev.x, prev.y) - rx(recrop.x, recrop.y)) > recrop.w * myW * translation ||
			std::fabs(ry(prev.x, prev.y) - ry(recrop.x, recrop.y)) > recrop.h * myH * translation)
			return false;
		if (std::fabs(prev.w - recrop.w) > recrop.w * scale || std::fabs(prev.h - recrop.h) > recrop.h * scale)
			return false;
		float cx = rx(recrop.x, recrop.y), cy = ry(recrop.x, recrop.y);
		float hw = recrop.w * myW * (1.f + margin) * 0.5f, hh = recrop.h * myH * (1.f + margin) * 0.5f;
		for (const Point& p : pts)
			if (!(std::fabs(rx(p.x, p.y) - cx) < hw && std::fabs(ry(p.x, p.y) - cy) < hh))
				return false;
		return true;
	}

	std::vector<Crop> Tracker::poseCrops(int width, int height, const std::vector<std::array<float, 4>>& boxes,
		const std::vector<std::string>& ids, int maxPeople)
	{
		myW = (float)std::max(1, width);
		myH = (float)std::max(1, height);
		if ((int)myPeople.size() > maxPeople)
			myPeople.resize((size_t)maxPeople);
		for (Person& p : myPeople)
			p.poseRoi = p.nextPoseRoi;
		for (size_t b = 0; b < boxes.size() && (int)myPeople.size() < maxPeople; b++)
		{
			const auto& px = boxes[b];	// x1, y1, x2, y2 px top-down
			float uv[4] = { px[0] / myW, 1.f - px[3] / myH, px[2] / myW, 1.f - px[1] / myH };
			bool covered = false, sameId = false;
			for (const Person& p : myPeople)
			{
				covered |= iou(uv, p.box) > kCoveredIou;
				sameId |= p.id == ids[b];
			}
			if (covered || sameId)
				continue;
			Person p;
			p.id = ids[b];
			Roi r{ (px[0] + px[2]) * 0.5f / myW, (px[1] + px[3]) * 0.5f / myH, (px[2] - px[0]) / myW, (px[3] - px[1]) / myH, 0.f };
			p.poseRoi = transform(r, kBoxRoiScale, 0.f);
			std::copy(uv, uv + 4, p.box);
			myPeople.push_back(std::move(p));
		}
		std::vector<Crop> crops;
		for (int i = 0; i < (int)myPeople.size(); i++)
		{
			myPeople[(size_t)i].alive = false;
			crops.push_back({ i, 0, myPeople[(size_t)i].poseRoi });
		}
		return crops;
	}

	bool Tracker::decodePose(const Crop& crop, const float* lm, float presence, const float* heatmap, const float* world)
	{
		Person& p = myPeople[(size_t)crop.person];
		if (presence < kPosePresence)
			return false;
		// TensorsToLandmarks (crop-normalized) + RefineLandmarksFromHeatmap, then projection.
		float nx[kPoseAll], ny[kPoseAll];
		for (int k = 0; k < kPoseAll; k++)
		{
			nx[k] = lm[k * 5] / kPoseInput;
			ny[k] = lm[k * 5 + 1] / kPoseInput;
			if (!heatmap)
				continue;
			int cc = (int)(nx[k] * kHeatmap), cr = (int)(ny[k] * kHeatmap);
			if (nx[k] < 0.f || ny[k] < 0.f || cc >= kHeatmap || cr >= kHeatmap)
				continue;
			int off = (kHeatmapKernel - 1) / 2;
			float sum = 0.f, wc = 0.f, wr = 0.f, mx = 0.f;
			for (int row = std::max(0, cr - off); row < std::min(kHeatmap, cr + off + 1); row++)
				for (int col = std::max(0, cc - off); col < std::min(kHeatmap, cc + off + 1); col++)
				{
					float conf = sigmoid(heatmap[(row * kHeatmap + col) * kPoseAll + k]);
					sum += conf;
					mx = std::max(mx, conf);
					wc += col * conf;
					wr += row * conf;
				}
			if (mx >= kHeatmapMinConfidence && sum > 0.f)
			{
				nx[k] = wc / kHeatmap / sum;
				ny[k] = wr / kHeatmap / sum;
			}
		}
		p.lm.fill(Landmark{});
		for (auto& w : p.world)
			w = { 0.f, 0.f, 0.f };
		p.hand[0] = p.hand[1] = p.face = false;
		const Roi& r = crop.roi;
		float c = std::cos(r.angle), s = std::sin(r.angle);
		float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f, vis = 0.f;
		for (int k = 0; k < kPose; k++)
		{
			Point q = project(nx[k], ny[k], r);
			Landmark& l = p.lm[(size_t)k];
			l = { q.x, q.y, lm[k * 5 + 2] / kPoseInput * r.w, sigmoid(lm[k * 5 + 3]) };
			vis += l.c;
			minX = std::min(minX, q.x);
			maxX = std::max(maxX, q.x);
			minY = std::min(minY, q.y);
			maxY = std::max(maxY, q.y);
			float wx = world[k * 3], wy = world[k * 3 + 1];
			p.world[(size_t)k] = { c * wx - s * wy, s * wx + c * wy, world[k * 3 + 2] };
		}
		p.score = vis / kPose;
		p.box[0] = minX;
		p.box[1] = 1.f - maxY;
		p.box[2] = maxX;
		p.box[3] = 1.f - minY;
		Point a0 = project(nx[33], ny[33], r), a1 = project(nx[34], ny[34], r);	// alignment points
		p.nextPoseRoi = alignment(a0.x, a0.y, a1.x, a1.y, 90.f, kPoseRoiScale, 0.f);
		p.alive = true;
		return true;
	}

	void Tracker::endPose()
	{
		std::vector<Person> kept;
		std::vector<size_t> order;
		for (size_t i = 0; i < myPeople.size(); i++)
			if (myPeople[i].alive)
				order.push_back(i);
		std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return myPeople[a].score > myPeople[b].score; });
		for (size_t i : order)
		{
			bool dup = false;
			for (const Person& k : kept)
				dup |= iou(myPeople[i].box, k.box) >= kDuplicateIou;
			if (!dup)
				kept.push_back(std::move(myPeople[i]));
		}
		myPeople = std::move(kept);
	}

	std::vector<Crop> Tracker::handRecropCrops() const
	{
		std::vector<Crop> crops;
		for (int i = 0; i < (int)myPeople.size(); i++)
		{
			const Person& p = myPeople[(size_t)i];
			for (int side = 0; side < 2; side++)
			{
				const Landmark& wrist = p.lm[(size_t)kHandPose[side][0]];
				if (wrist.c <= kWristVisibility)
					continue;
				const Landmark& pinky = p.lm[(size_t)kHandPose[side][1]];
				const Landmark& index = p.lm[(size_t)kHandPose[side][2]];
				// HandDetectionsFromPoseToRects: center at the estimated middle finger, size twice the
				// middle-wrist distance, rotation from wrist to middle finger; then scale 2.7, shift -0.1.
				float mx = (2.f * index.x + pinky.x) / 3.f, my = (2.f * index.y + pinky.y) / 3.f;
				float size = 2.f * std::hypot((mx - wrist.x) * myW, (my - wrist.y) * myH);
				Roi r{ mx, my, size / myW, size / myH,
					normalizeRadians(kPi * 0.5f - std::atan2(-(my - wrist.y) * myH, (mx - wrist.x) * myW)) };
				crops.push_back({ i, 1 + side, transform(r, 2.7f, -0.1f) });
			}
		}
		return crops;
	}

	Crop Tracker::handCrop(const Crop& recrop, const float* pts) const
	{
		Point a = project(pts[0] / kRecropInput, pts[1] / kRecropInput, recrop.roi);
		Point b = project(pts[2] / kRecropInput, pts[3] / kRecropInput, recrop.roi);
		Roi r = alignment(a.x, a.y, b.x, b.y, -90.f, 1.f, -0.1f);
		const auto& prev = myPeople[(size_t)recrop.person].prevHand[recrop.part - 1];
		if (prev.size() == (size_t)kHand)
		{
			Roi pr = palmRoi(prev);
			if (keepTracking(pr, prev, r, 40.f, 0.2f, 0.4f, -0.1f))
				r = pr;
		}
		return { recrop.person, recrop.part, r };
	}

	std::vector<Crop> Tracker::faceCrops() const
	{
		std::vector<Crop> crops;
		for (int i = 0; i < (int)myPeople.size(); i++)
		{
			const Person& p = myPeople[(size_t)i];
			if (std::min({ p.lm[0].c, p.lm[2].c, p.lm[5].c }) < 0.5f)	// nose and eyes
				continue;
			float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
			for (int k : kFacePose)
			{
				minX = std::min(minX, p.lm[(size_t)k].x);
				maxX = std::max(maxX, p.lm[(size_t)k].x);
				minY = std::min(minY, p.lm[(size_t)k].y);
				maxY = std::max(maxY, p.lm[(size_t)k].y);
			}
			const Landmark& re = p.lm[5];
			const Landmark& le = p.lm[2];
			// DetectionsToRects (keypoint box, rotation right eye -> left eye) + scale 2, shift -0.1.
			Roi r{ (minX + maxX) * 0.5f, (minY + maxY) * 0.5f, maxX - minX, maxY - minY,
				normalizeRadians(-std::atan2(-(le.y - re.y) * myH, (le.x - re.x) * myW)) };
			r = transform(r, 2.f, -0.1f);
			if (p.prevFace.size() == (size_t)kFace)
			{
				// FaceLandmarkLandmarksToRoi: the mesh's box, rotation 33 -> 263, scale 1.5.
				float a = 1e9f, b = -1e9f, cMin = 1e9f, d = -1e9f;
				for (const Point& q : p.prevFace)
				{
					a = std::min(a, q.x);
					b = std::max(b, q.x);
					cMin = std::min(cMin, q.y);
					d = std::max(d, q.y);
				}
				const Point& l = p.prevFace[33];
				const Point& rr = p.prevFace[263];
				Roi pr{ (a + b) * 0.5f, (cMin + d) * 0.5f, b - a, d - cMin,
					normalizeRadians(-std::atan2(-(rr.y - l.y) * myH, (rr.x - l.x) * myW)) };
				pr = transform(pr, 1.5f, 0.f);
				if (keepTracking(pr, p.prevFace, r, 15.f, 0.1f, 0.3f, -0.2f))
					r = pr;
			}
			crops.push_back({ i, 3, r });
		}
		return crops;
	}

	void Tracker::decodeHand(const Crop& crop, const float* lm, float presence, const float* world)
	{
		Person& p = myPeople[(size_t)crop.person];
		const int side = crop.part - 1;
		if (presence < kHandPresence)
			return;
		const Roi& r = crop.roi;
		float c = std::cos(r.angle), s = std::sin(r.angle);
		const int base = side == 0 ? kLeftHand0 : kRightHand0;
		// Holistic puts the hand in the body's space: its wrist on the pose's wrist.
		const auto& poseWrist = p.world[(size_t)kHandPose[side][0]];
		float w0[3] = { c * world[0] - s * world[1], s * world[0] + c * world[1], world[2] };
		auto& prev = p.prevHand[side];
		prev.resize(kHand);
		for (int k = 0; k < kHand; k++)
		{
			Point q = project(lm[k * 3] / kHandInput, lm[k * 3 + 1] / kHandInput, r);
			p.lm[(size_t)(base + k)] = { q.x, q.y, lm[k * 3 + 2] / kHandInput / 0.4f * r.w, 1.f };
			prev[(size_t)k] = q;
			float wx = world[k * 3], wy = world[k * 3 + 1];
			p.world[(size_t)(base + k)] = { c * wx - s * wy - w0[0] + poseWrist[0], s * wx + c * wy - w0[1] + poseWrist[1],
				world[k * 3 + 2] - w0[2] + poseWrist[2] };
		}
		p.hand[side] = true;
	}

	void Tracker::decodeFace(const Crop& crop, const float* lm, float presenceLogit)
	{
		Person& p = myPeople[(size_t)crop.person];
		if (sigmoid(presenceLogit) < kFacePresence)
			return;
		const Roi& r = crop.roi;
		p.prevFace.resize(kFace);
		for (int k = 0; k < kFace; k++)
		{
			Point q = project(lm[k * 3] / kFaceInput, lm[k * 3 + 1] / kFaceInput, r);
			p.lm[(size_t)(kFace0 + k)] = { q.x, q.y, lm[k * 3 + 2] / kFaceInput * r.w, 1.f };
			p.prevFace[(size_t)k] = q;
		}
		p.face = true;
	}

	void Tracker::endFrame()
	{
		for (Person& p : myPeople)
		{
			for (int side = 0; side < 2; side++)
				if (!p.hand[side])
					p.prevHand[side].clear();
			if (!p.face)
				p.prevFace.clear();
		}
	}
}
