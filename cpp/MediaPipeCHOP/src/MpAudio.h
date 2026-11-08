// MpAudio -- MediaPipe audio classification for MediaPipeCHOP.
//
// Binds the audio entry points of Google's libmediapipe.dll at runtime (no import library; same
// approach as MediaPipeTOP's MpApi, with the signatures taken from the vendored v1.2.2 headers via
// decltype) and runs the classifier on a worker thread. The DLL is CPU-only on Windows.
#pragma once

#ifndef MP_EXPORT
#define MP_EXPORT	// declarations only; nothing here is linked against
#endif
#include "mediapipe/tasks/c/components/containers/category.h"	// classification_result.h only forward-declares it
#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"
#include "mediapipe/tasks/c/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/c/core/common.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mpaudio
{
	// TD resolves a relative file parameter against the project folder. When that file doesn't
	// exist, returns the same relative path under this plugin DLL's folder if it's there (an
	// installed Plugins/<package>/models/ folder); otherwise returns `resolved` and describes both
	// places looked in via `missing`.
	std::string resolveModelPath(const char* resolved, const char* raw, std::string& missing);

	// Every class the model knows, in model order, with the newest window's scores.
	struct Result
	{
		std::vector<std::string>	names;
		std::vector<float>			scores;
		double						inferMs = 0.0;
		uint64_t					serial = 0;		// increments with every classified window
	};

	// Loads the model on its own thread, then classifies the most recently submitted clip. A clip
	// submitted while one is being classified replaces any clip still waiting (counted as dropped).
	class Classifier
	{
	public:
		explicit Classifier(const std::string& modelPath);
		~Classifier();
		Classifier(const Classifier&) = delete;
		Classifier& operator=(const Classifier&) = delete;

		void		submit(std::vector<float> mono, double sampleRate);
		bool		latest(Result& out) const;	// false until the first window is classified
		bool		loaded() const;
		std::string	error() const;
		uint64_t	dropped() const;
		const std::string&	modelPath() const { return myModelPath; }

	private:
		void		run();

		const std::string			myModelPath;
		mutable std::mutex			myMutex;
		std::condition_variable		myWake;
		bool						myStop = false;
		bool						myLoaded = false;
		std::string					myError;
		std::vector<float>			myPending;
		double						myPendingRate = 0.0;
		bool						myHasPending = false;
		uint64_t					myDropped = 0;
		Result						myResult;
		std::thread					myThread;
	};

	struct Segment
	{
		std::string	text;
		bool		final = false;
		int64_t		fromMs = 0;
		int64_t		toMs = 0;
	};

	// Everything transcribed so far: final segments (oldest first, capped) plus the current interim
	// hypothesis, if interim results are on.
	struct Transcript
	{
		std::vector<Segment>	finals;
		Segment					interim;
		double					inferMs = 0.0;	// last Transcribe call
		uint64_t				serial = 0;		// increments whenever the transcript changes
		uint64_t				totalFinals = 0;	// every final segment so far, including ones dropped from finals
	};

	// Streaming speech to text (MediaPipe AudioTranscriber: Whisper, Moonshine, Parakeet, ... as
	// .litertlm bundles). Audio must arrive in order and never be dropped, so submit() appends to
	// a queue the worker drains; backlogSeconds() shows if it falls behind real time.
	class Transcriber
	{
	public:
		Transcriber(const std::string& modelPath, int sampleRate, bool interim, const std::string& prompt);
		~Transcriber();
		Transcriber(const Transcriber&) = delete;
		Transcriber& operator=(const Transcriber&) = delete;

		void		submit(const float* samples, size_t count);
		void		finishStream();	// flush: finalizes the current utterance
		void		skip(size_t count);	// audio left out between utterances: keeps timestamps in real time
		bool		latest(Transcript& out) const;
		bool		loaded() const;
		std::string	error() const;
		double		backlogSeconds() const;
		bool		idle() const;	// loaded, nothing queued and nothing being transcribed
		const std::string&	modelPath() const { return myModelPath; }
		int			sampleRate() const { return mySampleRate; }

		static constexpr size_t kMaxFinals = 200;

	private:
		void		run();
		void		collect(const MpAudioTranscriberResult& result);

		const std::string			myModelPath;
		const int					mySampleRate;
		const bool					myInterim;
		const std::string			myPrompt;
		mutable std::mutex			myMutex;
		std::condition_variable		myWake;
		bool						myStop = false;
		bool						myLoaded = false;
		bool						myBusy = false;
		bool						myFinish = false;
		std::string					myError;
		std::vector<float>			myQueue;
		// Each flush starts a new stream whose timestamps restart at 0: segments are offset by the
		// audio of earlier streams so times count from when this transcriber started.
		uint64_t					myStreamSamples = 0;
		uint64_t					myOffsetSamples = 0;
		uint64_t					mySkipSamples = 0;	// skipped since the last stream, added before the next one
		Transcript					myTranscript;
		std::thread					myThread;
	};
}
