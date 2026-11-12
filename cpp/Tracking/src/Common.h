// Shared pieces of the tracker port: tables, number formatting and the few math functions whose
// results must match Python's bit for bit (the replay tests compare table text exactly).
#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace tracking {

using Row = std::vector<std::string>;
using Table = std::vector<Row>;

// Python's float(str): correctly rounded, like std::from_chars. Empty or unparsable -> 0 and false.
inline bool parseDouble(const char* s, size_t n, double& out) {
	while (n && (*s == ' ' || *s == '\t')) {
		++s;
		--n;
	}
	if (n && *s == '+') {
		++s;
		--n;
	}
	auto r = std::from_chars(s, s + n, out);
	if (r.ec == std::errc()) return true;
	out = 0.0;
	return false;
}

// f"{v:.{digits}f}": correctly rounded, "-0.0000" for small negatives, like Python.
inline std::string fixed(double v, int digits) {
	if (std::isnan(v)) return "nan";
	if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
	char buf[64];
	auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed, digits);
	return std::string(buf, r.ptr);
}

inline std::string integer(long long v) { return std::to_string(v); }

// Python's round(x) for a float: nearest integer, halves to even.
inline long long roundHalfEven(double x) {
	double r = std::nearbyint(x);	// default rounding mode: to nearest, ties to even
	return (long long)r;
}

// CPython's math.hypot (Modules/mathmodule.c vector_norm): the sum of squares carried exactly,
// so the result is correctly rounded in practice, unlike the C library's hypot.
namespace detail {
struct DoubleLength {
	double hi, lo;
};
inline DoubleLength twoSum(double a, double b) {	// |a| >= |b|
	double x = a + b;
	double z = x - a;
	return {x, b - z};
}
inline DoubleLength twoProduct(double a, double b) {	// Dekker: exact without FMA
	const double split = 134217729.0;	// 2^27 + 1
	double x = a * b;
	double ca = split * a, ah = ca - (ca - a), al = a - ah;
	double cb = split * b, bh = cb - (cb - b), bl = b - bh;
	double y = al * bl - (((x - ah * bh) - al * bh) - ah * bl);
	return {x, y};
}
}  // namespace detail

inline double pyHypot(double a, double b) {
	a = std::fabs(a);
	b = std::fabs(b);
	if (std::isinf(a) || std::isinf(b)) return INFINITY;
	if (std::isnan(a) || std::isnan(b)) return NAN;
	double mx = a > b ? a : b;
	if (mx == 0.0) return mx;
	int e;
	std::frexp(mx, &e);
	if (e < -1023) {	// subnormal range: rescale (never seen with 0-1 coordinates)
		const double m = 2.2250738585072014e-308;
		return m * pyHypot(a / m, b / m);
	}
	double scale = std::ldexp(1.0, -e);
	double csum = 1.0, frac1 = 0.0, frac2 = 0.0;
	for (double x : {a, b}) {
		x *= scale;
		auto pr = detail::twoProduct(x, x);
		auto sm = detail::twoSum(csum, pr.hi);
		csum = sm.hi;
		frac1 += pr.lo;
		frac2 += sm.lo;
	}
	double h = std::sqrt(csum - 1.0 + (frac1 + frac2));
	auto pr = detail::twoProduct(-h, h);
	auto sm = detail::twoSum(csum, pr.hi);
	csum = sm.hi;
	frac1 += pr.lo;
	frac2 += sm.lo;
	double x = csum - 1.0 + (frac1 + frac2);
	h += x / (2.0 * h);
	return h / scale;
}

inline double pyDegrees(double radians) { return radians * (180.0 / 3.141592653589793); }

// object_tracker.track_color: golden-ratio hue, colorsys.hsv_to_rgb(hue, 0.85, 1.0).
inline std::array<double, 3> trackColor(long long trackId) {
	double h = std::fmod(double(trackId) * 0.6180339887498949, 1.0);
	const double s = 0.85, v = 1.0;
	int i = int(h * 6.0);
	double f = (h * 6.0) - i;
	double p = v * (1.0 - s);
	double q = v * (1.0 - s * f);
	double t = v * (1.0 - s * (1.0 - f));
	switch (i % 6) {
		case 0: return {v, t, p};
		case 1: return {q, v, p};
		case 2: return {p, v, t};
		case 3: return {p, q, v};
		case 4: return {t, p, v};
		default: return {v, p, q};
	}
}

// object_tracker.build_label: "{#color(R, G, B);}" + text, text defaulting to "<id> <score%>%".
inline std::string buildLabel(long long trackId, double score, const std::string* text = nullptr) {
	auto c = trackColor(trackId);
	std::string out = "{#color(" + integer(roundHalfEven(c[0] * 255)) + ", " + integer(roundHalfEven(c[1] * 255)) + ", " +
		integer(roundHalfEven(c[2] * 255)) + ");}";
	if (text)
		out += *text;
	else
		out += integer(trackId) + " " + integer(roundHalfEven(score * 100)) + "%";
	return out;
}

inline void appendColor(Row& row, long long trackId) {
	auto c = trackColor(trackId);
	for (double v : c) row.push_back(fixed(v, 4));
}

// Tables as text, the way a TouchDesigner DAT shows them (tab-separated, one row per line).
inline std::string tableText(const Table& t) {
	std::string s;
	for (size_t r = 0; r < t.size(); ++r) {
		if (r) s += "\r\n";
		for (size_t c = 0; c < t[r].size(); ++c) {
			if (c) s += '\t';
			s += t[r][c];
		}
	}
	return s;
}

}  // namespace tracking
