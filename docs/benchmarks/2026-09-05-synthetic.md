# Initial raw transcription smoke benchmark — September 5, 2026

FreeFlow's local streaming path finished a median **116 ms** after the audio
ended, versus **509 ms** for Whisper large-v3-turbo Q5 and **67 ms** for Whisper
base.en. Speech word error rates were **2.84%, 2.13%, and 2.13%**, respectively.
This is a small synthetic integration benchmark, not a real-world accuracy ranking.

| Engine / mode | Speech WER ↓ | After audio p50 ↓ | After audio p95 ↓ | Silence false positives | Failed calls |
|---|---:|---:|---:|---:|---:|
| FreeFlow Parakeet v2, production Swift streaming client | 2.84% | 116 ms | 252 ms | 0/4 | 0/28 |
| FreeFlow Parakeet v2, batch endpoint | 2.48% | 96 ms | 298 ms | 0/4 | 0/28 |
| whisper.cpp large-v3-turbo, Q5_0, batch | 2.13% | 509 ms | 646 ms | 4/4 | 0/28 |
| whisper.cpp base.en, batch | 2.13% | 67 ms | 188 ms | 4/4 | 0/28 |

Both Whisper configurations ran **without VAD**, and each emitted eight words
over the four silence trials. Silence is scored separately from speech WER.
These numbers do not describe a consumer dictation app that adds silence
filtering or cleanup. All 112 scored calls and four warmup calls succeeded.

The same 12 spoken clips were repeated twice per engine: six invented passages,
each rendered with macOS Samantha and Daniel (English (UK)). The suite also has
two synthetic silence clips. Each pass contains 108.65 seconds of audio and 282
reference words; repetitions provide 564 scored reference words per engine but
do not add new sentences or speakers. Categories are short, ordinary dictation,
fast speech, names/numbers, disfluencies, long speech, and silence.

The names/numbers category accounts for 12 of FreeFlow streaming's 16 word errors
and all 12 Whisper word errors. The scorer intentionally distinguishes written
numbers from spoken word forms. Errors outside this category total four for
FreeFlow streaming, two for FreeFlow batch, and zero for both Whisper models.
The sample is too small and too artificial to establish an accuracy advantage.

## Measurement conditions

- Apple M4 Pro, 24 GiB memory; Darwin 27.0.0, arm64; Python 3.9.6.
- `parakeet-mlx` 0.5.2, `mlx` 0.31.2, SciPy 1.18.0, aiohttp 3.14.3.
- Homebrew `whisper-cpp` 1.8.6, Metal enabled, four threads, default decoding,
  English language, no VAD or vocabulary prompts.
- FreeFlow server source SHA-256:
  `ae66a630dcc1bb40ca078c8284ae76ed4d084da9b926720b486f7a56cb8281ca`.
- Corpus SHA-256:
  `25f905580a767808a7adcfcb088f54687a2de851a158d51163b27fa8619d7052`.
- Model file SHA-256: Turbo Q5
  `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`;
  base.en `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`.
- One warmup per engine, two scored repetitions, serial calls, seeded shuffle
  `20260905`. Thirty-second request/final timeout; no retries or fallback.
- Isolated benchmark servers; Parakeet cleanup forwarding and heartbeat disabled.
  Existing app services remained running. Final machine load averages were
  3.89 / 4.29 / 3.81; this was a development machine, not a dedicated idle host.

Batch timing starts with complete audio available and includes the request and
response. Streaming uses the production Swift client, 20 ms paced audio chunks,
and measures commit-to-final after the complete file (including trailing silence).
File conversion, helper launch, cleanup, microphone capture, hotkeys, and paste
are outside the measurement. Streaming's 16→24→16 kHz resampling is part of its
path and can contribute to differences from batch. Services were already loaded;
these are not cold-start results. p95 is descriptive for this small sample.

The reusable corpus, models, and numeric JSON/Markdown reports are stored under
`~/.cache/freeflow-benchmark/`. The report is named
`results/synthetic-v1-20260905.json`. Audio and transcript content are not in the
report or this repository. Temporary conversions, replay executables, server
configuration, and server processes were removed after the run.

## Reproduce and extend

Follow the [benchmark instructions](../../local-setup/benchmark/README.md) and
use `--repeats 2` to reproduce the run shape. Keep the generated corpus rather
than regenerating it if an exact audio match matters.

Hosted Whisper and Deepgram adapters passed synthetic local HTTP tests. They
were not exercised against paid APIs because their environment credentials
were unavailable. No app history, user recordings, clipboard, or microphone was
read. No permission changes were made.

The next accuracy gate is a fixed, speaker-balanced public speech corpus, then
representative spontaneous dictation, accents, noise, and terminology. The
included LibriSpeech importer supports that step; public-corpus benchmarking
was not run in this smoke check. Separate UI tests would be needed to compare
complete products such as Apple Dictation or Wispr Flow.
