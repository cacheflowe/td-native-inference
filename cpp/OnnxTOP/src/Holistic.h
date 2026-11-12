// MediaPipe Holistic's ROI logic and tensor decoding, for OnnxTOP's Holistic mode: Holistic's own
// landmark models (scripts/convert_holistic_onnx.py) run on rotated crops, and this class decides
// the crops and turns the outputs into landmarks. Pure CPU math, no CUDA / ORT / TD.
//
// Follows MediaPipe's graphs (mediapipe/modules/holistic_landmark, pose_landmark, hand_landmark,
// face_landmark) and calculators (AlignmentPointsRects, RectTransformation, LandmarkProjection,
// WorldLandmarkProjection, RefineLandmarksFromHeatmap, HandLandmarksToRect, RoiTracking).
// Differences: people come from a detector's boxes (not BlazePose's detector), the face recrop ROI
// comes from the pose's face points (not BlazeFace inside the pose's face crop), no landmark
// smoothing (the tracker smooths). Several people at once, each tracked from its own landmarks.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace holistic
{
	// MediaPipe NormalizedRect: center and size as fractions of the image width / height, top-down,
	// rotation in radians.
	struct Roi
	{
		float	x = 0.f, y = 0.f, w = 0.f, h = 0.f, angle = 0.f;
	};

	constexpr int kPose = 33, kPoseAll = 39, kHand = 21, kFace = 478;
	constexpr int kLeftHand0 = 33, kRightHand0 = 54, kFace0 = 75, kTotal = 553, kWorld = 75;
	constexpr int kPoseInput = 256, kRecropInput = 256, kHandInput = 224, kFaceInput = 192, kHeatmap = 64;
	// Model output sizes (floats per crop), in the converted models' output order.
	constexpr int kPoseLandmarkValues = kPoseAll * 5, kPoseWorldValues = kPoseAll * 3;
	constexpr int kHeatmapValues = kHeatmap * kHeatmap * kPoseAll;

	struct Landmark
	{
		float	x = 0.f, y = 0.f, z = 0.f;	// image-normalized, top-down; z scaled like x
		float	c = 0.f;					// pose: visibility; hands / face: 1 when present
	};

	struct Point
	{
		float	x, y;
	};

	struct Person
	{
		std::string		id;
		Roi				poseRoi;			// this frame's crop
		Roi				nextPoseRoi;		// from this frame's alignment points
		float			score = 0.f;		// mean pose visibility
		float			box[4] = {};		// pose landmark extent, TD UV (u1, v1, u2, v2)
		std::array<Landmark, kTotal>				lm{};
		std::array<std::array<float, 3>, kWorld>	world{};	// metres, hip origin, MediaPipe axes
		bool			hand[2] = {};		// left, right found this frame
		bool			face = false;
		std::vector<Point>	prevHand[2];	// last frame's landmarks, for the tracking decision
		std::vector<Point>	prevFace;
		bool			alive = false;		// pose found this frame
	};

	// One crop to run: which person, which part (0 pose, 1 left hand, 2 right hand, 3 face).
	struct Crop
	{
		int		person = 0;
		int		part = 0;
		Roi		roi;
	};

	class Tracker
	{
	public:
		void	reset() { myPeople.clear(); }

		// Pose crops for this frame: each tracked person's ROI from last frame, plus a ROI for each
		// detector box (source pixels, top-down) that no tracked person overlaps.
		std::vector<Crop>	poseCrops(int width, int height, const std::vector<std::array<float, 4>>& boxes,
								const std::vector<std::string>& ids, int maxPeople);
		// Decode one pose run (outputs: landmarks [39 x 5], presence flag, heatmap [64 x 64 x 39] or
		// null, world [39 x 3]). Returns false (person lost) when presence < 0.5.
		bool	decodePose(const Crop& crop, const float* landmarks, float presence, const float* heatmap,
					const float* world);
		// Drop lost people and duplicates (two ROIs that converged on one person).
		void	endPose();

		// Hand crops from the pose (for the ROI-refinement model) and the face crops (tracked).
		std::vector<Crop>	handRecropCrops() const;
		std::vector<Crop>	faceCrops() const;
		// The refinement model's 2 points (4 floats, 256 px crop space) -> the hand landmark crop.
		Crop	handCrop(const Crop& recrop, const float* points) const;
		void	decodeHand(const Crop& crop, const float* landmarks, float presence, const float* world);
		void	decodeFace(const Crop& crop, const float* landmarks, float presenceLogit);
		// Forget hand / face tracking for parts not found this frame.
		void	endFrame();

		const std::vector<Person>&	people() const { return myPeople; }

	private:
		Roi		transform(Roi r, float scale, float shiftY) const;
		Roi		alignment(float x0, float y0, float x1, float y1, float targetDegrees, float scale, float shiftY) const;
		Point	project(float nx, float ny, const Roi& r) const;
		Roi		palmRoi(const std::vector<Point>& pts) const;
		bool	keepTracking(const Roi& prev, const std::vector<Point>& prevPts, const Roi& recrop,
					float rotDeg, float translation, float scale, float margin) const;

		std::vector<Person>	myPeople;
		float				myW = 1.f, myH = 1.f;
	};
}
