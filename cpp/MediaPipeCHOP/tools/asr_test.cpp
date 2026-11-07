// Console check of mpaudio::Transcriber without TouchDesigner: streams a 16-bit PCM mono WAV in
// 0.25 s chunks, prints every transcript update, then flushes and prints the real-time factor.
//   set MP_AUDIO_DLL_DIR=<repo>\plugins
//   asr_test.exe <model.litertlm> <speech.wav> [interim]
#include "MpAudio.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>

static bool readWav(const char* path, std::vector<float>& samples, int& rate)
{
	std::ifstream file(path, std::ios::binary);
	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (data.size() < 44 || memcmp(data.data(), "RIFF", 4) != 0)
		return false;
	size_t pos = 12;
	int channels = 1, bits = 16;
	while (pos + 8 <= data.size())
	{
		uint32_t size;
		memcpy(&size, &data[pos + 4], 4);
		if (memcmp(&data[pos], "fmt ", 4) == 0)
		{
			channels = *(int16_t*)&data[pos + 10];
			rate = *(int32_t*)&data[pos + 12];
			bits = *(int16_t*)&data[pos + 22];
		}
		else if (memcmp(&data[pos], "data", 4) == 0)
		{
			if (bits != 16)
				return false;
			const int16_t* pcm = (const int16_t*)&data[pos + 8];
			size_t frames = size / 2 / channels;
			for (size_t i = 0; i < frames; i++)
				samples.push_back(pcm[i * channels] / 32768.0f);
			return true;
		}
		pos += 8 + size + (size & 1);
	}
	return false;
}

static void print(const mpaudio::Transcript& t, double seconds)
{
	std::string text;
	for (const auto& s : t.finals)
		text += s.text + " ";
	printf("[%5.2f s] %s%s%s\n", seconds, text.c_str(), t.interim.text.empty() ? "" : "| interim: ",
		t.interim.text.c_str());
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("usage: asr_test <model.litertlm> <speech.wav> [interim]\n");
		return 2;
	}
	std::vector<float> samples;
	int rate = 0;
	if (!readWav(argv[2], samples, rate))
	{
		printf("could not read %s (16-bit PCM WAV)\n", argv[2]);
		return 2;
	}
	auto begin = std::chrono::steady_clock::now();
	mpaudio::Transcriber transcriber(argv[1], rate, argc > 3, "");
	for (int i = 0; i < 3000 && !transcriber.loaded() && transcriber.error().empty(); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	if (!transcriber.loaded())
	{
		printf("load failed: %s\n", transcriber.error().c_str());
		return 1;
	}
	auto loadedAt = std::chrono::steady_clock::now();
	printf("loaded in %.0f ms; %.2f s of audio at %d Hz\n",
		std::chrono::duration<double, std::milli>(loadedAt - begin).count(), samples.size() / (double)rate, rate);
	mpaudio::Transcript transcript;
	const size_t chunk = (size_t)(rate / 4);
	const bool realtime = std::getenv("ASR_REALTIME") != nullptr;	// pace chunks like a live input
	for (size_t at = 0; at < samples.size(); at += chunk)
	{
		transcriber.submit(samples.data() + at, std::min(chunk, samples.size() - at));
		if (realtime)
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		if (transcriber.latest(transcript))
			print(transcript, at / (double)rate);
	}
	transcriber.finishStream();
	for (int i = 0; i < 6000 && !transcriber.idle(); i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (transcriber.latest(transcript))
			print(transcript, samples.size() / (double)rate);
	}
	transcriber.latest(transcript);
	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadedAt).count();
	printf("final: ");
	print(transcript, samples.size() / (double)rate);
	printf("real-time factor %.2f (processing time / audio duration), error: '%s'\n",
		seconds / (samples.size() / (double)rate), transcriber.error().c_str());
	return transcriber.error().empty() && !transcript.finals.empty() ? 0 : 1;
}
