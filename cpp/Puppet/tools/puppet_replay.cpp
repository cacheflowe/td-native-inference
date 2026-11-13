// C API over the puppet core for tests/test_puppet_cpp.py (ctypes): a rest-pose table and tracks
// tables in (tab-separated text), channel names and values out.
#include "Puppet.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#define EXPORT extern "C" __declspec(dllexport)

namespace {

class TextTable : public puppet::TableView {
public:
	explicit TextTable(const char* text) {
		std::string s(text ? text : "");
		while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();	// like str.splitlines()
		size_t pos = 0;
		while (pos <= s.size() && !s.empty()) {
			size_t end = s.find('\n', pos);
			if (end == std::string::npos) end = s.size();
			std::string line = s.substr(pos, end - pos);
			if (!line.empty() && line.back() == '\r') line.pop_back();
			std::vector<std::string> row;
			size_t c = 0;
			while (true) {
				size_t tab = line.find('\t', c);
				row.push_back(line.substr(c, tab == std::string::npos ? std::string::npos : tab - c));
				if (tab == std::string::npos) break;
				c = tab + 1;
			}
			rows_.push_back(std::move(row));
			if (end >= s.size()) break;
			pos = end + 1;
		}
		for (const auto& r : rows_) cols_ = std::max(cols_, (int)r.size());
	}
	int numRows() const override { return (int)rows_.size(); }
	int numCols() const override { return cols_; }
	const char* cell(int r, int c) const override {
		if (r < 0 || r >= (int)rows_.size() || c < 0 || c >= (int)rows_[r].size()) return "";
		return rows_[r][c].c_str();
	}

private:
	std::vector<std::vector<std::string>> rows_;
	int cols_ = 0;
};

struct Handle {
	puppet::Solver solver;
	std::string out;
};

}  // namespace

EXPORT void* pup_create() { return new Handle(); }

EXPORT void pup_destroy(void* h) { delete (Handle*)h; }

// Returns 1 when a rig is set (a hand rig: 2).
EXPORT int pup_set_rest(void* h, const char* text) {
	auto* p = (Handle*)h;
	TextTable t(text);
	if (!p->solver.setRest(&t)) return 0;
	return p->solver.isHand() ? 2 : 1;
}

// One cook; tracks NULL = no table. Returns 1 when solved, 0 when the channels were kept.
EXPORT int pup_cook(void* h, const char* tracks, const char* hand, int mirror, int overImage, double fov, double smoothing,
	int imageW, int imageH) {
	auto* p = (Handle*)h;
	puppet::Settings s;
	s.hand = hand ? hand : "any";
	s.mirror = mirror != 0;
	s.overImage = overImage != 0;
	s.fov = fov;
	s.placeSmoothing = smoothing;
	s.imageW = imageW;
	s.imageH = imageH;
	if (!tracks) return p->solver.cook(nullptr, s) ? 1 : 0;
	TextTable t(tracks);
	return p->solver.cook(&t, s) ? 1 : 0;
}

// "name\tvalue\n" per channel, values with 17 significant digits.
EXPORT const char* pup_channels(void* h) {
	auto* p = (Handle*)h;
	p->out.clear();
	const auto& names = p->solver.names();
	const auto& values = p->solver.values();
	char buf[40];
	for (size_t i = 0; i < names.size(); ++i) {
		std::snprintf(buf, sizeof(buf), "%.17g", values[i]);
		p->out += names[i] + '\t' + buf + '\n';
	}
	return p->out.c_str();
}

EXPORT const char* pup_warning(void* h) { return ((Handle*)h)->solver.warning().c_str(); }
