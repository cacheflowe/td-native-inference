// The rest of the wrappers' per-frame Python, in C++:
//   ClassScores    onnxtop_tables.write_scores: a classifier's per-track class scores (HSEmotion),
//                  smoothed in float32, joined onto the tracks table
//   convertRtmw    rtmw_convert.convert: RTMW3D's Pose Crops Info DAT -> the Holistic-shaped table
//                  the rtmw tracker mode reads
//   trackChannels  wrapper_runtime.write_channels: t{id}_{column} for every numeric tracks cell
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Common.h"
#include "Tracker.h"	// InfoView

namespace tracking {

class ClassScores {
public:
	// out = the tracks table plus <prefix>_label and <prefix>_<class> columns (before r, g, b);
	// without a classifier table a plain copy.
	void update(const InfoView* tracks, const InfoView* scores, const std::string& prefix, double smoothing, Table& out);

private:
	std::map<std::string, std::vector<float>> state_;	// track id (cell text) -> smoothed scores
};

// Header only when the Info DAT has no rows or isn't a Pose Crops table with z and crop_w columns.
void convertRtmw(const InfoView* info, double width, double height, Table& out);

// (name, value) per numeric cell of each data row, in row then column order; value as TD stores
// a channel (float32 of the cell's number).
void trackChannels(const Table& tracks, std::vector<std::pair<std::string, float>>& out);

// Python's float(text): the whole string must be a number (surrounding whitespace allowed).
bool parseWholeDouble(const std::string& text, double& out);

}  // namespace tracking
