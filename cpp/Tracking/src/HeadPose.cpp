// face_tracker_pipeline._compute_head_direction_solvepnp without OpenCV: cv2.solvePnP(ITERATIVE,
// useExtrinsicGuess) refines the pose with OpenCV's CvLevMarq from the "facing the camera at focal
// length" guess. Same model points, camera matrix, steps (lambda from 1e-3, x10 / x0.1, at most 20
// iterations, stop on a relative step below FLT_EPSILON), Rodrigues and Euler decomposition. The
// Jacobian is numeric and the 6x6 solve is Cholesky instead of OpenCV's SVD, so results agree to
// rounding, not bit for bit.
#include <algorithm>
#include <cfloat>
#include <cmath>

#include "BoxPipeline.h"

namespace tracking {

namespace {

const double kModel[5][3] = {
	{-31.5, -20.0, -10.0},	// right eye
	{31.5, -20.0, -10.0},	// left eye
	{0.0, 0.0, 0.0},	// nose tip
	{-25.0, 25.0, -12.0},	// right mouth corner
	{25.0, 25.0, -12.0},	// left mouth corner
};

void rodrigues(const double r[3], double R[3][3]) {
	const double theta = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
	if (theta < DBL_EPSILON) {
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j) R[i][j] = i == j ? 1.0 : 0.0;
		return;
	}
	const double c = std::cos(theta), s = std::sin(theta), c1 = 1.0 - c, it = 1.0 / theta;
	const double x = r[0] * it, y = r[1] * it, z = r[2] * it;
	const double rrt[3][3] = {{x * x, x * y, x * z}, {x * y, y * y, y * z}, {x * z, y * z, z * z}};
	const double rx[3][3] = {{0, -z, y}, {z, 0, -x}, {-y, x, 0}};
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) R[i][j] = c * (i == j ? 1.0 : 0.0) + c1 * rrt[i][j] + s * rx[i][j];
}

struct Problem {
	double img[5][2];
	double f, cx, cy;

	// Reprojection error (projected - observed), 10 values.
	void error(const double p[6], double e[10]) const {
		double R[3][3];
		rodrigues(p, R);
		for (int i = 0; i < 5; ++i) {
			const double* M = kModel[i];
			double X = R[0][0] * M[0] + R[0][1] * M[1] + R[0][2] * M[2] + p[3];
			double Y = R[1][0] * M[0] + R[1][1] * M[1] + R[1][2] * M[2] + p[4];
			double Z = R[2][0] * M[0] + R[2][1] * M[1] + R[2][2] * M[2] + p[5];
			const double iz = Z ? 1.0 / Z : 1.0;
			e[2 * i] = f * (X * iz) + cx - img[i][0];
			e[2 * i + 1] = f * (Y * iz) + cy - img[i][1];
		}
	}

	void jacobian(const double p[6], double J[10][6]) const {
		for (int k = 0; k < 6; ++k) {
			const double h = (k < 3 ? 1e-7 : 1e-5) * std::max(1.0, std::fabs(p[k]));
			double a[6], b[6], ea[10], eb[10];
			for (int q = 0; q < 6; ++q) a[q] = b[q] = p[q];
			a[k] += h;
			b[k] -= h;
			error(a, ea);
			error(b, eb);
			for (int i = 0; i < 10; ++i) J[i][k] = (ea[i] - eb[i]) / (2 * h);
		}
	}
};

double norm(const double* v, int n) {
	double s = 0;
	for (int i = 0; i < n; ++i) s += v[i] * v[i];
	return std::sqrt(s);
}

// Solve the symmetric positive definite A x = b (6x6). False if A isn't positive definite.
bool solve6(double A[6][6], const double b[6], double x[6]) {
	double L[6][6] = {};
	for (int i = 0; i < 6; ++i)
		for (int j = 0; j <= i; ++j) {
			double s = A[i][j];
			for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
			if (i == j) {
				if (s <= 0) return false;
				L[i][i] = std::sqrt(s);
			} else {
				L[i][j] = s / L[j][j];
			}
		}
	double y[6];
	for (int i = 0; i < 6; ++i) {
		double s = b[i];
		for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
		y[i] = s / L[i][i];
	}
	for (int i = 5; i >= 0; --i) {
		double s = y[i];
		for (int k = i + 1; k < 6; ++k) s -= L[k][i] * x[k];
		x[i] = s / L[i][i];
	}
	return true;
}

}  // namespace

bool headPoseSolvePnP(const double kp[5][2], double width, double height, double focalScale, double out[3]) {
	Problem pr;
	for (int i = 0; i < 5; ++i) {
		pr.img[i][0] = kp[i][0] * width;
		pr.img[i][1] = (1.0 - kp[i][1]) * height;
	}
	pr.f = width * focalScale;
	pr.cx = width / 2.0;
	pr.cy = height / 2.0;

	double param[6] = {0, 0, 0, 0, 0, pr.f}, prev[6];
	double err[10], J[10][6], JtJ[6][6], JtErr[6];
	int lambdaLg10 = -3, iters = 0;
	const int maxIter = 20;
	const double eps = FLT_EPSILON;

	auto step = [&]() {
		const double lambda = std::exp(lambdaLg10 * std::log(10.0));
		double A[6][6], x[6];
		for (int i = 0; i < 6; ++i)
			for (int j = 0; j < 6; ++j) A[i][j] = JtJ[i][j];
		for (int i = 0; i < 6; ++i) A[i][i] *= 1.0 + lambda;
		if (!solve6(A, JtErr, x))
			for (double& v : x) v = 0;
		for (int i = 0; i < 6; ++i) param[i] = prev[i] - x[i];
	};

	pr.error(param, err);
	pr.jacobian(param, J);
	double prevErrNorm = 0;
	for (;;) {
		for (int i = 0; i < 6; ++i) {
			JtErr[i] = 0;
			for (int r = 0; r < 10; ++r) JtErr[i] += J[r][i] * err[r];
			for (int j = 0; j < 6; ++j) {
				JtJ[i][j] = 0;
				for (int r = 0; r < 10; ++r) JtJ[i][j] += J[r][i] * J[r][j];
			}
		}
		for (int i = 0; i < 6; ++i) prev[i] = param[i];
		step();
		if (iters == 0) prevErrNorm = norm(err, 10);
		double errNorm;
		for (;;) {
			pr.error(param, err);
			errNorm = norm(err, 10);
			if (errNorm > prevErrNorm && ++lambdaLg10 <= 16) {
				step();
				continue;
			}
			break;
		}
		lambdaLg10 = std::max(lambdaLg10 - 1, -16);
		double d[6];
		for (int i = 0; i < 6; ++i) d[i] = param[i] - prev[i];
		const double prevNorm = norm(prev, 6);
		if (++iters >= maxIter || norm(d, 6) / (prevNorm + DBL_EPSILON) < eps) break;	// cv::norm(NORM_RELATIVE)
		prevErrNorm = errNorm;
		pr.error(param, err);
		pr.jacobian(param, J);
	}
	for (double v : param)
		if (!std::isfinite(v)) return false;

	double R[3][3];
	rodrigues(param, R);
	const double sy = std::sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);
	double pitch, yaw, roll;
	if (sy >= 1e-6) {
		pitch = std::atan2(R[2][1], R[2][2]);
		yaw = std::atan2(-R[2][0], sy);
		roll = std::atan2(R[1][0], R[0][0]);
	} else {
		pitch = std::atan2(-R[1][2], R[1][1]);
		yaw = std::atan2(-R[2][0], sy);
		roll = 0.0;
	}
	out[0] = pyDegrees(yaw);
	out[1] = pyDegrees(pitch);
	out[2] = pyDegrees(roll);
	return true;
}

}  // namespace tracking
