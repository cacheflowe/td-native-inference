// tts_test <model.litertlm> <out.wav> [text] [voice]
// Speaks the text with mpaudio::Speaker, draining its buffer in 1/60 s steps as TD would, and
// reports load time, time to first audio, underruns (steps with no audio before the line ended)
// and the voices the model offers. Set MP_AUDIO_DLL_DIR to the folder holding libmediapipe.dll.
#include "MpAudio.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>

namespace
{
	void writeWav(const char* path, const std::vector<float>& samples, int rate)
	{
		std::ofstream out(path, std::ios::binary);
		auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
		auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
		const uint32_t bytes = (uint32_t)(samples.size() * 2);
		out.write("RIFF", 4); u32(36 + bytes); out.write("WAVEfmt ", 8);
		u32(16); u16(1); u16(1); u32((uint32_t)rate); u32((uint32_t)rate * 2); u16(2); u16(16);
		out.write("data", 4); u32(bytes);
		for (float v : samples)
		{
			const int16_t s = (int16_t)(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
			out.write(reinterpret_cast<const char*>(&s), 2);
		}
	}
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::fprintf(stderr, "usage: tts_test <model.litertlm> <out.wav> [text] [voice]\n");
		return 2;
	}
	const char* text = argc > 3 ? argv[3] : "Hello from TouchDesigner. This sentence was spoken by Kokoro, running on the CPU.";
	const std::string voice = argc > 4 ? argv[4] : "";
	using clock = std::chrono::steady_clock;
	auto seconds = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };

	const auto start = clock::now();
	mpaudio::Speaker speaker(argv[1], 1.0f);
	while (!speaker.loaded() && speaker.error().empty())
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	if (!speaker.error().empty())
	{
		std::fprintf(stderr, "error: %s\n", speaker.error().c_str());
		return 1;
	}
	const auto loaded = clock::now();
	std::printf("loaded in %.2f s; %zu voices:", seconds(start, loaded), speaker.voices().size());
	for (const std::string& v : speaker.voices())
		std::printf(" %s", v.c_str());
	std::printf("\n");

	speaker.say(text, voice);
	const auto said = clock::now();
	std::vector<float> audio;
	double firstAudio = -1.0;
	int underruns = 0;
	bool started = false;
	for (;;)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(16667));
		const size_t want = (size_t)(speaker.sampleRate() / 60);
		std::vector<float> step(want, 0.0f);
		const size_t got = speaker.read(step.data(), want);
		const mpaudio::SpeakerStatus status = speaker.status();
		if (got > 0 && firstAudio < 0)
			firstAudio = seconds(said, clock::now());
		if (got > 0)
			started = true;
		if (started && got < want && status.speaking)
			underruns++;
		audio.insert(audio.end(), step.begin(), step.begin() + got);
		if (!speaker.error().empty())
		{
			std::fprintf(stderr, "error: %s\n", speaker.error().c_str());
			return 1;
		}
		if (status.spoken >= 1 && !status.speaking)
			break;
		if (seconds(said, clock::now()) > 60.0)
		{
			std::fprintf(stderr, "timed out\n");
			return 1;
		}
	}
	const double duration = (double)audio.size() / speaker.sampleRate();
	std::printf("first audio after %.2f s; %.2f s of speech at %d Hz; %d underrun steps; last chunk %.0f ms\n",
		firstAudio, duration, speaker.sampleRate(), underruns, speaker.status().synthMs);
	writeWav(argv[2], audio, speaker.sampleRate());
	std::printf("wrote %s\n", argv[2]);
	return 0;
}
