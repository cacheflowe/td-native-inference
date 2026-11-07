// mp_test -- drives MpWorker outside TouchDesigner, for checking results against the official
// Python API on identical frames.
//
//   mp_test <dll dir> <face|hands|gesture|pose|holistic|facedetector|segment> <model> <frames.rgb> <w> <h>
//           [max] [seg] [out.txt] [segoutput 0/1/2] [segclass]
//
// frames.rgb holds N packed RGB8 top-down frames of w x h. Each frame is submitted and waited
// for (so every frame is processed); the last frame's landmarks are written to out.txt as
// "instance landmark x y z vis pres" lines.
#include "MpApi.h"
#include "MpWorker.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

int main(int argc, char** argv)
{
	if (argc < 7)
	{
		printf("usage: mp_test <dll dir> <face|hands|gesture|pose> <model.task> <frames.rgb> <w> <h> [max] [seg] [out.txt]\n");
		return 1;
	}
	mpapi::setDllDir(argv[1]);
	MpTaskConfig cfg;
	const char* kinds[] = { "face", "hands", "gesture", "pose", "holistic", "facedetector", "segment" };
	for (int k = 0; k < 7; k++)
		if (!strcmp(argv[2], kinds[k]))
			cfg.kind = (MpTaskKind)k;
	cfg.modelPath = argv[3];
	int w = atoi(argv[5]), h = atoi(argv[6]);
	cfg.maxInstances = argc > 7 ? atoi(argv[7]) : 1;
	cfg.segmentation = argc > 8 && atoi(argv[8]) != 0;
	cfg.blendshapes = true;
	cfg.matrix = true;

	std::ifstream f(argv[4], std::ios::binary);
	std::vector<uint8_t> all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	size_t frameBytes = (size_t)w * h * 3;
	int frames = (int)(all.size() / frameBytes);
	if (frames < 1)
	{
		printf("no frames in %s\n", argv[4]);
		return 1;
	}

	MpWorker worker(cfg);
	if (argc > 10)
		worker.setSegOutput((MpSegOutput)atoi(argv[10]), argc > 11 ? atoi(argv[11]) : 0);
	while (worker.state() == MpWorker::State::Loading)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	if (worker.state() == MpWorker::State::Failed)
	{
		printf("load failed: %s\n", worker.loadError().c_str());
		return 1;
	}

	MpFrameResult r;
	double total = 0.0;
	int detected = 0;
	for (int i = 0; i < frames; i++)
	{
		std::vector<uint8_t> buf(all.begin() + i * frameBytes, all.begin() + (i + 1) * frameBytes);
		uint64_t before = r.seq;
		worker.submit(buf, w, h);
		while (!worker.latest(r) || r.seq == before)
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		if (!r.error.empty())
			printf("frame %d error: %s\n", i, r.error.c_str());
		total += r.inferMs;
		detected += !r.instances.empty();
	}
	printf("%s: %d frames, mean infer %.2f ms, detected in %d\n", argv[2], frames, total / frames, detected);

	for (size_t n = 0; n < r.instances.size(); n++)
	{
		const MpInstance& in = r.instances[n];
		printf("instance %zu: %zu landmarks, label '%s' cls %d conf %.4f gesture %.4f world %zu\n", n,
			in.lm.size() / kLmStride, in.label.c_str(), in.cls, in.conf, in.gestureScore, in.world.size() / 3);
		if (in.hasBox)
			printf("  box %.4f %.4f %.4f %.4f\n", in.box[0], in.box[1], in.box[2], in.box[3]);
		for (int k : { 0, 1, 4 })
			if ((size_t)k * kLmStride < in.lm.size())
				printf("  lm%d %.5f %.5f %.5f vis %.4f pres %.4f\n", k, in.lm[k * kLmStride], in.lm[k * kLmStride + 1],
					in.lm[k * kLmStride + 2], in.lm[k * kLmStride + 3], in.lm[k * kLmStride + 4]);
		if (!in.world.empty())
			printf("  world0 %.5f %.5f %.5f\n", in.world[0], in.world[1], in.world[2]);
		if (!in.scores.empty())
		{
			int best = 0;
			for (int k = 1; k < (int)in.scores.size(); k++)
				if (in.scores[k] > in.scores[best])
					best = k;
			printf("  %zu blendshapes, top %s %.4f\n", in.scores.size(), r.scoreNames[best].c_str(), in.scores[best]);
		}
		if (in.matrix.size() == 16)
			printf("  matrix row0 %.4f %.4f %.4f %.4f  row3 %.4f %.4f %.4f %.4f\n", in.matrix[0], in.matrix[1],
				in.matrix[2], in.matrix[3], in.matrix[12], in.matrix[13], in.matrix[14], in.matrix[15]);
	}
	if (!r.mask.empty())
	{
		double sum = 0;
		for (float v : r.mask)
			sum += v;
		printf("mask %dx%d mean %.4f center %.4f\n", r.maskW, r.maskH, sum / r.mask.size(),
			r.mask[(size_t)(r.maskH / 2) * r.maskW + r.maskW / 2]);
	}
	if (argc > 9)
	{
		FILE* o = fopen(argv[9], "w");
		for (size_t n = 0; n < r.instances.size(); n++)
			for (size_t k = 0; k < r.instances[n].lm.size() / kLmStride; k++)
			{
				const float* l = &r.instances[n].lm[k * kLmStride];
				fprintf(o, "%zu %zu %.6f %.6f %.6f %.6f %.6f\n", n, k, l[0], l[1], l[2], l[3], l[4]);
			}
		fclose(o);
	}
	return 0;
}
