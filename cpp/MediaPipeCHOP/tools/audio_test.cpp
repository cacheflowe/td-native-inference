// Console check of MpAudio without TouchDesigner: classifies one second of a 440 Hz sine (YAMNet has
// a "Sine wave" class) and of silence, and prints the top classes.
//   set MP_AUDIO_DLL_DIR=<repo>\plugins
//   audio_test.exe <repo>\models\mediapipe\yamnet.tflite
#include "MpAudio.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>

static int classify(mpaudio::Classifier& classifier, std::vector<float> clip, double rate, const char* what, uint64_t serial)
{
	classifier.submit(std::move(clip), rate);
	mpaudio::Result result;
	for (int i = 0; i < 500 && (!classifier.latest(result) || result.serial <= serial); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	if (result.serial <= serial)
	{
		printf("%s: no result (%s)\n", what, classifier.error().c_str());
		return 1;
	}
	std::vector<int> order(result.scores.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) { return result.scores[a] > result.scores[b]; });
	printf("%s (%zu classes, %.1f ms):", what, result.scores.size(), result.inferMs);
	for (int i = 0; i < 3 && i < (int)order.size(); i++)
		printf("  %s %.3f", result.names[order[i]].c_str(), result.scores[order[i]]);
	printf("\n");
	return 0;
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("usage: audio_test <model.tflite>\n");
		return 2;
	}
	mpaudio::Classifier classifier(argv[1]);
	for (int i = 0; i < 1000 && !classifier.loaded() && classifier.error().empty(); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	if (!classifier.loaded())
	{
		printf("load failed: %s\n", classifier.error().c_str());
		return 1;
	}
	const double rate = 48000.0;
	std::vector<float> sine((size_t)rate), silence((size_t)rate, 0.0f);
	for (size_t i = 0; i < sine.size(); i++)
		sine[i] = 0.5f * (float)std::sin(2.0 * 3.14159265358979 * 440.0 * (double)i / rate);
	int failures = classify(classifier, sine, rate, "sine 440 Hz", 0);
	failures += classify(classifier, silence, rate, "silence", 1);
	return failures;
}
