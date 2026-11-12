// tracker_replay.dll: a C API over the tracking core for tests/tracking_replay.py (ctypes), which feeds
// it recorded Info DATs and diffs its tables against the Python tracker's.
#include <memory>
#include <string>

#include "Common.h"
#include "Tracker.h"

using namespace tracking;

namespace {
struct Handle {
	Tracker tracker;
	Table tables[3];
	std::string text;
	explicit Handle(const char* mode) : tracker(mode) {}
};
}  // namespace

extern "C" {

__declspec(dllexport) int trk_supported(const char* mode) { return isSupportedMode(mode) ? 1 : 0; }

__declspec(dllexport) void* trk_create(const char* mode) { return new Handle(mode); }

__declspec(dllexport) void trk_destroy(void* h) { delete static_cast<Handle*>(h); }

__declspec(dllexport) void trk_reset_ids() { Tracker::resetIds(); }

__declspec(dllexport) void trk_set_par(void* h, const char* name, double value) {
	static_cast<Handle*>(h)->tracker.setPar(name, value);
}

// has_results = 0: the wrapper has no results / result_delay channels (OnnxTOP).
__declspec(dllexport) const char* trk_cook(void* hv, int table, const char* infoText, int hasInfo, int hasResults,
		double results, double resultDelay) {
	auto* h = static_cast<Handle*>(hv);
	h->tracker.setResultCounter(hasResults ? std::optional<double>(results) : std::nullopt,
		hasResults ? std::optional<double>(resultDelay) : std::nullopt);
	std::unique_ptr<TextInfo> info;
	if (hasInfo) info = std::make_unique<TextInfo>(infoText ? infoText : "");
	h->tracker.cook(TableKind(table), info.get(), h->tables[table]);
	h->text = tableText(h->tables[table]);
	return h->text.c_str();
}

__declspec(dllexport) double trk_hypot(double a, double b) { return pyHypot(a, b); }

__declspec(dllexport) const char* trk_fixed(double v, int digits) {
	static std::string s;
	s = fixed(v, digits);
	return s.c_str();
}
}
