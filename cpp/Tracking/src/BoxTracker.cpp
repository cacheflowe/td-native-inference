#include "BoxTracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

// Matching numpy exactly. The Kalman matrices are 0s and 1s and the box dimensions are independent,
// so every matrix product below has at most two nonzero terms per element and rounds the same in
// any summation order; the projected covariance is diagonal, its Cholesky factor is the square
// roots, and numpy's two triangular solves (OpenBLAS) multiply by the reciprocal of the diagonal:
// gain = (b * (1 / l)) * (1 / l), verified bit for bit against numpy 2.1 (TD) and 2.5.

namespace tracking {

long long BoxTrack::nextId = 1;

namespace {

const double kStdWeightPosition = 1.0 / 20;
const double kStdWeightVelocity = 1.0 / 160;

void xyxyToXyah(const double box[4], double out[4]) {
	double w = std::max(box[2] - box[0], 1e-6);
	double h = std::max(box[3] - box[1], 1e-6);
	out[0] = box[0] + w / 2.0;
	out[1] = box[1] + h / 2.0;
	out[2] = w / h;
	out[3] = h;
}

// a @ b for 8x8 matrices, k in order (numpy's result for these sparse 0/1 products).
void matmul8(const double a[8][8], const double b[8][8], double out[8][8]) {
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j) {
			double s = 0.0;
			for (int k = 0; k < 8; ++k) s += a[i][k] * b[k][j];
			out[i][j] = s;
		}
}

struct Kalman {
	double motion[8][8], motionT[8][8];
	Kalman() {
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j) motion[i][j] = i == j ? 1.0 : 0.0;
		for (int i = 0; i < 4; ++i) motion[i][4 + i] = 1.0;
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j) motionT[i][j] = motion[j][i];
	}
};

const Kalman& kalman() {
	static Kalman k;
	return k;
}

std::vector<std::vector<double>> iouMatrix(const std::vector<std::array<double, 4>>& a, const std::vector<std::array<double, 4>>& b) {
	std::vector<std::vector<double>> out(a.size(), std::vector<double>(b.size()));
	for (size_t i = 0; i < a.size(); ++i) {
		const double areaA = (a[i][2] - a[i][0]) * (a[i][3] - a[i][1]);
		for (size_t j = 0; j < b.size(); ++j) {
			const double x1 = std::max(a[i][0], b[j][0]), y1 = std::max(a[i][1], b[j][1]);
			const double x2 = std::min(a[i][2], b[j][2]), y2 = std::min(a[i][3], b[j][3]);
			const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
			const double areaB = (b[j][2] - b[j][0]) * (b[j][3] - b[j][1]);
			const double uni = areaA + areaB - inter;
			out[i][j] = uni > 0 ? inter / uni : 0.0;
		}
	}
	return out;
}

// object_tracker._hungarian_square: shortest augmenting path with potentials. row -> col.
std::vector<int> hungarianSquare(const std::vector<std::vector<double>>& cost) {
	const int n = int(cost.size());
	const double INF = std::numeric_limits<double>::infinity();
	std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
	std::vector<int> p(n + 1, 0), way(n + 1, 0);
	for (int i = 1; i <= n; ++i) {
		p[0] = i;
		int j0 = 0;
		std::vector<double> minv(n + 1, INF);
		std::vector<char> used(n + 1, 0);
		while (true) {
			used[j0] = 1;
			const int i0 = p[j0];
			double delta = INF;
			int j1 = -1;
			for (int j = 1; j <= n; ++j) {
				if (!used[j]) {
					const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
					if (cur < minv[j]) {
						minv[j] = cur;
						way[j] = j0;
					}
					if (minv[j] < delta) {
						delta = minv[j];
						j1 = j;
					}
				}
			}
			for (int j = 0; j <= n; ++j) {
				if (used[j]) {
					u[p[j]] += delta;
					v[j] -= delta;
				} else {
					minv[j] -= delta;
				}
			}
			j0 = j1;
			if (p[j0] == 0) break;
		}
		while (j0) {
			const int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		}
	}
	std::vector<int> colOfRow(n, 0);
	for (int j = 1; j <= n; ++j)
		if (p[j] != 0) colOfRow[p[j] - 1] = j - 1;
	return colOfRow;
}

std::array<double, 4> boxOf(const BoxTrack& t) {
	std::array<double, 4> b;
	t.box(b.data());
	return b;
}

}  // namespace

// ---------------------------------------------------------------- BoxTrack

BoxTrack::BoxTrack(const double box[4], double s, std::shared_ptr<const Detection> p, int minHits_)
	: trackId(nextId++), score(s), payload(std::move(p)), minHits(minHits_) {
	double m[4];
	xyxyToXyah(box, m);
	const double std_[8] = {
		2 * kStdWeightPosition * m[3], 2 * kStdWeightPosition * m[3], 1e-2, 2 * kStdWeightPosition * m[3],
		10 * kStdWeightVelocity * m[3], 10 * kStdWeightVelocity * m[3], 1e-5, 10 * kStdWeightVelocity * m[3],
	};
	for (int i = 0; i < 4; ++i) mean[i] = m[i];
	for (int i = 4; i < 8; ++i) mean[i] = 0.0;
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; ++j) cov[i][j] = i == j ? std_[i] * std_[i] : 0.0;
	confirmed = hits >= minHits;
}

void BoxTrack::box(double out[4]) const {
	const double cx = mean[0], cy = mean[1], a = mean[2], h = mean[3];
	const double w = a * h;
	out[0] = cx - w / 2.0;
	out[1] = cy - h / 2.0;
	out[2] = cx + w / 2.0;
	out[3] = cy + h / 2.0;
}

void BoxTrack::predict() {
	const Kalman& k = kalman();
	const double h = mean[3];
	const double std_[8] = {
		kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h,
		kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h,
	};
	double m[8];
	for (int i = 0; i < 8; ++i) {
		double s = 0.0;
		for (int j = 0; j < 8; ++j) s += k.motion[i][j] * mean[j];
		m[i] = s;
	}
	double fp[8][8], fpf[8][8];
	matmul8(k.motion, cov, fp);
	matmul8(fp, k.motionT, fpf);
	for (int i = 0; i < 8; ++i) {
		mean[i] = m[i];
		for (int j = 0; j < 8; ++j) cov[i][j] = fpf[i][j] + (i == j ? std_[i] * std_[i] : 0.0);
	}
}

void BoxTrack::update(const double box[4], double s, std::shared_ptr<const Detection> p) {
	double z[4];
	xyxyToXyah(box, z);
	const double h = mean[3];
	const double std_[4] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-1, kStdWeightPosition * h};
	// Projected covariance S (diagonal), its Cholesky factor l, and the gain K (8x4).
	double l[4], inv[4];
	for (int c = 0; c < 4; ++c) {
		l[c] = std::sqrt(cov[c][c] + std_[c] * std_[c]);
		inv[c] = 1.0 / l[c];
	}
	double S[4];
	for (int c = 0; c < 4; ++c) S[c] = cov[c][c] + std_[c] * std_[c];
	double K[8][4];
	for (int r = 0; r < 8; ++r)
		for (int c = 0; c < 4; ++c) K[r][c] = (cov[r][c] * inv[c]) * inv[c];
	double innovation[4];
	for (int c = 0; c < 4; ++c) innovation[c] = z[c] - mean[c];
	double m[8];
	for (int r = 0; r < 8; ++r) {
		double sum = 0.0;
		for (int c = 0; c < 4; ++c) sum += innovation[c] * K[r][c];
		m[r] = mean[r] + sum;
	}
	// cov - (K @ S) @ K.T
	double ks[8][4];
	for (int r = 0; r < 8; ++r)
		for (int c = 0; c < 4; ++c) ks[r][c] = K[r][c] * S[c];
	for (int r = 0; r < 8; ++r)
		for (int q = 0; q < 8; ++q) {
			double sum = 0.0;
			for (int c = 0; c < 4; ++c) sum += ks[r][c] * K[q][c];
			cov[r][q] = cov[r][q] - sum;
		}
	for (int i = 0; i < 8; ++i) mean[i] = m[i];
	score = s;
	payload = std::move(p);
	lostFrames = 0;
	totalFrames += 1;
	hits += 1;
	if (hits >= minHits) confirmed = true;
}

void BoxTrack::markLost() {
	lostFrames += 1;
	totalFrames += 1;
	mean[6] = 0.0;
	mean[7] = 0.0;
}

// ---------------------------------------------------------------- assignment

std::vector<std::pair<int, int>> assignWithThreshold(const std::vector<std::vector<double>>& cost, double thresh) {
	std::vector<std::pair<int, int>> out;
	const int rows = int(cost.size()), cols = rows ? int(cost[0].size()) : 0;
	if (rows == 0 || cols == 0) return out;
	const int n = std::max(rows, cols);
	const double reject = thresh + 1.0, pad = thresh + 0.5;
	std::vector<std::vector<double>> square(n, std::vector<double>(n, pad));
	for (int r = 0; r < rows; ++r)
		for (int c = 0; c < cols; ++c) square[r][c] = cost[r][c] <= thresh ? cost[r][c] : reject;
	std::vector<int> col = hungarianSquare(square);
	for (int r = 0; r < n; ++r)
		if (r < rows && col[r] < cols && square[r][col[r]] <= thresh) out.emplace_back(r, col[r]);
	return out;
}

// ---------------------------------------------------------------- ByteTracker

const std::vector<std::shared_ptr<BoxTrack>>& ByteTracker::update(Dets dets) {
	for (auto& t : tracks) t->predict();
	if (int(dets.size()) > maxDetections) dets = capDetections(dets);
	Dets high, low;
	for (auto& d : dets)
		if (d->score >= highThresh) high.push_back(d);
	for (auto& d : dets)
		if (lowThresh <= d->score && d->score < highThresh) low.push_back(d);
	std::vector<int> all(tracks.size());
	std::iota(all.begin(), all.end(), 0);
	auto [unmatchedTracks, unmatchedHigh] = associate(all, high);
	unmatchedTracks = associate(unmatchedTracks, low).first;
	for (int i : unmatchedTracks) tracks[i]->markLost();
	for (int j : unmatchedHigh) {
		const auto& d = high[j];
		tracks.push_back(std::make_shared<BoxTrack>(d->box, d->score, d, minHits));
	}
	std::vector<std::shared_ptr<BoxTrack>> kept;
	for (auto& t : tracks)
		if (t->lostFrames <= trackBuffer) kept.push_back(t);
	tracks = std::move(kept);
	if (int(tracks.size()) > maxTracks) {
		std::stable_sort(tracks.begin(), tracks.end(), [](const auto& a, const auto& b) {
			if (a->lostFrames != b->lostFrames) return a->lostFrames < b->lostFrames;
			return -a->score < -b->score;
		});
		tracks.resize(maxTracks);
	}
	return tracks;
}

ByteTracker::Dets ByteTracker::capDetections(const Dets& dets) const {
	auto byScore = [](Dets v) {
		std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a->score > b->score; });
		return v;
	};
	Dets out;
	if (tracks.empty()) {
		out = byScore(dets);
	} else {
		std::vector<std::array<double, 4>> tb, db;
		for (auto& t : tracks) tb.push_back(boxOf(*t));
		for (auto& d : dets) db.push_back({d->box[0], d->box[1], d->box[2], d->box[3]});
		auto iou = iouMatrix(db, tb);
		Dets relevant, other;
		for (size_t i = 0; i < dets.size(); ++i) {
			double best = *std::max_element(iou[i].begin(), iou[i].end());
			(best > 0.05 ? relevant : other).push_back(dets[i]);
		}
		out = relevant;
		for (auto& d : byScore(other)) out.push_back(d);
	}
	if (int(out.size()) > maxDetections) out.resize(maxDetections);
	return out;
}

std::pair<std::vector<int>, std::vector<int>> ByteTracker::associate(const std::vector<int>& trackIdx, const Dets& dets) {
	std::vector<int> unmatchedDets;
	if (trackIdx.empty() || dets.empty()) {
		for (int c = 0; c < int(dets.size()); ++c) unmatchedDets.push_back(c);
		return {trackIdx, unmatchedDets};
	}
	std::vector<std::array<double, 4>> tb, db;
	for (int i : trackIdx) tb.push_back(boxOf(*tracks[i]));
	for (auto& d : dets) db.push_back({d->box[0], d->box[1], d->box[2], d->box[3]});
	auto cost = iouMatrix(tb, db);
	for (auto& row : cost)
		for (auto& c : row) c = 1.0 - c;
	auto matches = assignWithThreshold(cost, 1.0 - matchThresh);
	std::vector<char> rowMatched(trackIdx.size(), 0), colMatched(dets.size(), 0);
	for (auto [r, c] : matches) {
		tracks[trackIdx[r]]->update(dets[c]->box, dets[c]->score, dets[c]);
		rowMatched[r] = colMatched[c] = 1;
	}
	std::vector<int> unmatchedTracks;
	for (size_t r = 0; r < trackIdx.size(); ++r)
		if (!rowMatched[r]) unmatchedTracks.push_back(trackIdx[r]);
	for (size_t c = 0; c < dets.size(); ++c)
		if (!colMatched[c]) unmatchedDets.push_back(int(c));
	return {unmatchedTracks, unmatchedDets};
}

// ---------------------------------------------------------------- NMS (float32)

std::vector<int> nmsFloat32(const std::vector<std::shared_ptr<const Detection>>& dets, double iouThreshold) {
	const int n = int(dets.size());
	std::vector<int> keep;
	if (n == 0) return keep;
	std::vector<std::array<float, 4>> b(n);
	std::vector<float> s(n);
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < 4; ++k) b[i][k] = float(dets[i]->box[k]);
		s[i] = float(dets[i]->score);
	}
	std::vector<int> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return -s[x] < -s[y]; });
	const float threshold = float(iouThreshold);
	while (!order.empty()) {
		const int i = order[0];
		keep.push_back(i);
		if (order.size() == 1) break;
		std::vector<int> rest;
		const float areaI = (b[i][2] - b[i][0]) * (b[i][3] - b[i][1]);
		for (size_t q = 1; q < order.size(); ++q) {
			const int j = order[q];
			const float x1 = std::max(b[i][0], b[j][0]), y1 = std::max(b[i][1], b[j][1]);
			const float x2 = std::min(b[i][2], b[j][2]), y2 = std::min(b[i][3], b[j][3]);
			const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
			const float areaJ = (b[j][2] - b[j][0]) * (b[j][3] - b[j][1]);
			const float uni = areaI + areaJ - inter;
			const float iou = uni > 0 ? inter / uni : 0.0f;
			if (iou < threshold) rest.push_back(j);
		}
		order = std::move(rest);
	}
	return keep;
}

}  // namespace tracking
