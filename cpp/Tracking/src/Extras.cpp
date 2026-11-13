#include "Extras.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <unordered_map>

namespace tracking {

namespace {

std::vector<std::string> rowOf(const InfoView& v, int r) {
	std::vector<std::string> row;
	for (int c = 0; c < v.numCols(); ++c) row.emplace_back(v.cell(r, c));
	// Script DAT rows written with fewer cells read back with empty ones; trailing empties are kept
	// as the table holds them.
	return row;
}

std::string upper(std::string s) {
	for (char& ch : s) ch = char(std::toupper((unsigned char)ch));
	return s;
}

std::string lower(std::string s) {
	for (char& ch : s) ch = char(std::tolower((unsigned char)ch));
	return s;
}

// Through a pointer, so the compiler can't fold pow(x, 2.0) into x * x: Python's x ** 2 and
// x ** 0.5 call the C library's pow.
double (*volatile cpow)(double, double) = static_cast<double (*)(double, double)>(std::pow);

}  // namespace

bool parseWholeDouble(const std::string& text, double& out) {
	size_t b = 0, e = text.size();
	while (b < e && std::isspace((unsigned char)text[b])) ++b;
	while (e > b && std::isspace((unsigned char)text[e - 1])) --e;
	if (b < e && text[b] == '+') {
		++b;
		if (b < e && (text[b] == '+' || text[b] == '-')) return false;
	}
	if (b >= e) return false;
	auto r = std::from_chars(text.data() + b, text.data() + e, out);
	return r.ec == std::errc() && r.ptr == text.data() + e;
}

// ---------------------------------------------------------------- write_scores

void ClassScores::update(const InfoView* tracks, const InfoView* scores, const std::string& prefixPar, double smoothing,
		Table& out) {
	out.clear();
	if (!tracks) return;
	for (int r = 0; r < tracks->numRows(); ++r) out.push_back(rowOf(*tracks, r));
	if (!scores || scores->numRows() < 1 || out.empty()) return;
	const Row sh = rowOf(*scores, 0);
	if (sh.size() < 4 || sh[0] != "id" || sh[1] != "class" || sh[2] != "conf") return;	// not a Classify Crops Info DAT
	const std::vector<std::string> names(sh.begin() + 3, sh.end());
	const Row th = out[0];
	auto idIt = std::find(th.begin(), th.end(), "track_id");
	if (idIt == th.end()) return;	// e.g. the error row
	const std::string prefix = prefixPar.empty() ? "class" : prefixPar;
	const float s32 = float(smoothing), t32 = float(1.0 - smoothing);

	for (int r = 1; r < scores->numRows(); ++r) {
		std::vector<float> probs;
		const int n = std::min<int>(int(names.size()), scores->numCols() - 3);
		for (int c = 3; c < 3 + n; ++c) {
			double v = 0;
			parseWholeDouble(scores->cell(r, c), v);
			probs.push_back(float(v));
		}
		const std::string tid = scores->cell(r, 0);
		auto it = state_.find(tid);
		if (it == state_.end() || it->second.size() != probs.size()) {
			state_[tid] = probs;
		} else {
			for (size_t i = 0; i < probs.size(); ++i) it->second[i] = it->second[i] * s32 + probs[i] * t32;
		}
	}

	const size_t idCol = size_t(idIt - th.begin());
	auto labelIt = std::find(th.begin(), th.end(), "label");
	const long labelCol = labelIt == th.end() ? -1 : long(labelIt - th.begin());
	std::vector<std::string> present;
	std::vector<std::pair<Row, Row>> rows;
	for (size_t r = 1; r < out.size(); ++r) {
		Row row = out[r];
		const std::string tid = idCol < row.size() ? row[idCol] : "";
		present.push_back(tid);
		Row extra;
		auto it = state_.find(tid);
		if (it != state_.end()) {
			const auto& probs = it->second;
			const size_t best = probs.empty() ? 0 : size_t(std::max_element(probs.begin(), probs.end()) - probs.begin());
			// np.argmax: the first of equal maxima (max_element returns the first too).
			const std::string label = upper(best < names.size() ? names[best] : "");
			if (labelCol >= 0 && size_t(labelCol) < row.size()) {
				const std::string& cell = row[labelCol];
				const size_t close = cell.rfind('}');
				const std::string head = close == std::string::npos ? "" : cell.substr(0, close + 1);
				row[labelCol] = head + tid + " " + label;
			}
			extra.push_back(label);
			for (float v : probs) extra.push_back(fixed(double(v), 4));
		} else {
			extra.assign(1 + names.size(), "");
		}
		rows.emplace_back(std::move(row), std::move(extra));
	}
	for (auto it = state_.begin(); it != state_.end();)
		it = std::find(present.begin(), present.end(), it->first) == present.end() ? state_.erase(it) : std::next(it);

	// Insert before the trailing r, g, b color columns, like onnx_hsemotion.py.
	const bool rgb = th.size() >= 3 && th[th.size() - 3] == "r" && th[th.size() - 2] == "g" && th[th.size() - 1] == "b";
	const size_t at = rgb ? th.size() - 3 : th.size();
	Row header(th.begin(), th.begin() + at);
	header.push_back(prefix + "_label");
	for (auto& n : names) header.push_back(prefix + "_" + lower(n));
	header.insert(header.end(), th.begin() + at, th.end());
	out.clear();
	out.push_back(std::move(header));
	for (auto& [row, extra] : rows) {
		Row full(row.begin(), row.begin() + std::min(at, row.size()));
		full.insert(full.end(), extra.begin(), extra.end());
		if (row.size() > at) full.insert(full.end(), row.begin() + at, row.end());
		out.push_back(std::move(full));
	}
}

// ---------------------------------------------------------------- rtmw_convert

namespace {

const int FACE0 = 23, LEFT_HAND0 = 91, RIGHT_HAND0 = 112;
const int kPoseSource[33] = {0, FACE0 + 42, 1, FACE0 + 45, FACE0 + 39, 2, FACE0 + 36, 3, 4, FACE0 + 54, FACE0 + 48, 5, 6, 7,
	8, 9, 10, LEFT_HAND0 + 17, RIGHT_HAND0 + 17, LEFT_HAND0 + 5, RIGHT_HAND0 + 5, LEFT_HAND0 + 4, RIGHT_HAND0 + 4, 11, 12, 13,
	14, 15, 16, 19, 22, 17, 20};
const int kCount = 75;
const double kHandMinScore = 0.3, kBoneMinScore = 0.3;
struct Bone {
	int a, b;
	double metres;
};
const Bone kBones[] = {{5, 7, 0.29}, {7, 9, 0.26}, {6, 8, 0.29}, {8, 10, 0.26}, {11, 13, 0.42}, {13, 15, 0.41}, {12, 14, 0.42},
	{14, 16, 0.41}, {5, 6, 0.38}, {11, 12, 0.27}, {5, 11, 0.50}, {6, 12, 0.50}};

int sourceOf(int i) {
	if (i < 33) return kPoseSource[i];
	if (i < 54) return LEFT_HAND0 + (i - 33);
	return RIGHT_HAND0 + (i - 54);
}

const Row& rtmwHeader() {
	static Row h = [] {
		Row r = {"id", "class", "conf", "u1", "v1", "u2", "v2", "label"};
		for (int i = 0; i < kCount; ++i)
			for (const char* s : {"u", "v", "c"}) r.push_back("kp" + std::to_string(i) + "_" + s);
		for (int i = 0; i < kCount; ++i) r.push_back("kp" + std::to_string(i) + "_z");
		for (int i = 0; i < kCount; ++i)
			for (const char* s : {"x", "y", "z"}) r.push_back("w" + std::to_string(i) + "_" + s);
		return r;
	}();
	return h;
}

double num(const char* s) {
	double v = 0;
	parseWholeDouble(s, v);
	return v;
}

}  // namespace

void convertRtmw(const InfoView* info, double width, double height, Table& out) {
	out.clear();
	out.push_back(rtmwHeader());
	if (!info || info->numRows() < 2) return;
	std::unordered_map<std::string, int> index;
	for (int c = 0; c < info->numCols(); ++c) index[info->cell(0, c)] = c;	// a dict: the last duplicate wins
	auto colOf = [&](const std::string& name) {
		auto it = index.find(name);
		return it == index.end() ? -1 : it->second;
	};
	std::vector<int> kp(133), kz(133);
	for (int k = 0; k < 133; ++k) {
		kp[k] = colOf("kp" + std::to_string(k) + "_u");
		if (kp[k] < 0) return;
	}
	if (colOf("kp0_z") < 0 || colOf("crop_w") < 0) return;
	for (int k = 0; k < 133; ++k) kz[k] = colOf("kp" + std::to_string(k) + "_z");
	const int cropW = colOf("crop_w"), idCol = colOf("id"), confCol = colOf("conf");
	const int box[4] = {colOf("u1"), colOf("v1"), colOf("u2"), colOf("v2")};

	for (int r = 1; r < info->numRows(); ++r) {
		double u[133], v[133], conf[133], z[133], p[133][3];
		for (int k = 0; k < 133; ++k) {
			u[k] = num(info->cell(r, kp[k]));
			v[k] = num(info->cell(r, kp[k] + 1));
			conf[k] = num(info->cell(r, kp[k] + 2));
			z[k] = kz[k] >= 0 ? num(info->cell(r, kz[k])) : 0.0;
		}
		const double zPx = num(info->cell(r, cropW)) / 2.0;
		for (int k = 0; k < 133; ++k) {
			p[k][0] = u[k] * width;
			p[k][1] = (1.0 - v[k]) * height;
			p[k][2] = z[k] * zPx;
		}
		// _world: hip-centred metres from the median bone-length ratio.
		bool hasWorld = !(conf[11] < kBoneMinScore && conf[12] < kBoneMinScore);
		double world[133][3];
		if (hasWorld) {
			std::vector<double> ratios;
			for (const Bone& bone : kBones) {
				if (conf[bone.a] < kBoneMinScore || conf[bone.b] < kBoneMinScore) continue;
				double sum = 0;
				for (int i = 0; i < 3; ++i) sum += cpow(p[bone.a][i] - p[bone.b][i], 2.0);
				const double length = cpow(sum, 0.5);
				if (length > 1e-6) ratios.push_back(bone.metres / length);
			}
			if (ratios.empty()) {
				hasWorld = false;
			} else {
				std::sort(ratios.begin(), ratios.end());
				const size_t n = ratios.size();
				const double scale = n % 2 ? ratios[n / 2] : (ratios[n / 2 - 1] + ratios[n / 2]) / 2;
				int hips[2], nh = 0;
				for (int i : {11, 12})
					if (conf[i] >= kBoneMinScore) hips[nh++] = i;
				double root[3];
				for (int c = 0; c < 3; ++c) {
					double s = 0;
					for (int h = 0; h < nh; ++h) s += p[hips[h]][c];
					root[c] = s / nh;
				}
				for (int k = 0; k < 133; ++k)
					for (int c = 0; c < 3; ++c) world[k][c] = (p[k][c] - root[c]) * scale;
			}
		}
		bool present[kCount];
		for (int i = 0; i < kCount; ++i) present[i] = true;
		for (auto [start, end] : {std::pair<int, int>{33, 54}, {54, 75}}) {
			double sum = 0;
			for (int i = start; i < end; ++i) sum += conf[sourceOf(i)];
			if (sum / (end - start) < kHandMinScore)
				for (int i = start; i < end; ++i) present[i] = false;
		}
		Row row = {idCol >= 0 ? info->cell(r, idCol) : "0", "0", info->cell(r, confCol)};
		for (int c : box) row.push_back(info->cell(r, c));
		row.push_back("");
		for (int i = 0; i < kCount; ++i) {
			const int k = sourceOf(i);
			row.push_back(fixed(u[k], 5));
			row.push_back(fixed(v[k], 5));
			row.push_back(fixed(present[i] ? conf[k] : 0.0, 4));
		}
		for (int i = 0; i < kCount; ++i) row.push_back(present[i] ? fixed(z[sourceOf(i)], 5) : "0");
		for (int i = 0; i < kCount; ++i) {
			const bool w = hasWorld && present[i];
			for (int c = 0; c < 3; ++c) row.push_back(fixed(w ? world[sourceOf(i)][c] : 0.0, 5));
		}
		out.push_back(std::move(row));
	}
}

// ---------------------------------------------------------------- write_channels

void trackChannels(const Table& tracks, std::vector<std::pair<std::string, float>>& out) {
	out.clear();
	if (tracks.empty()) return;
	const Row& header = tracks[0];
	auto idIt = std::find(header.begin(), header.end(), "track_id");
	if (idIt == header.end()) return;
	const size_t idCol = size_t(idIt - header.begin());
	for (size_t r = 1; r < tracks.size(); ++r) {
		const Row& row = tracks[r];
		const std::string id = idCol < row.size() ? row[idCol] : "";
		const size_t n = std::min(header.size(), row.size());	// zip(header, row)
		for (size_t c = 0; c < n; ++c) {
			double v;
			if (!parseWholeDouble(row[c], v)) continue;
			std::string name = "t" + id + "_" + header[c];
			for (char& ch : name)
				if (!(std::isalnum((unsigned char)ch) || ch == '_')) ch = '_';
			out.emplace_back(std::move(name), float(v));
		}
	}
}

}  // namespace tracking
