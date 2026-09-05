# FreeFlow sub-50 ms experiments — September 5, 2026

These experiments target the delay from committing the last audio sample to
receiving the final **raw transcript**. They do not measure microphone shutdown,
physical key release, cleanup, or paste. The installed app and service were not
changed. Streaming candidates run through an isolated local server; complete-clip
experiments use isolated model runners.

Across **3,258 scored calls on 414 distinct clips**, the initial screening found
configurations below 50 ms, but broader streaming validation caught accuracy
regressions. **Sub-50 ms with preserved accuracy was not achieved.** No direct Wispr
measurement or 10× accuracy advantage has been established. See the separate
[Wispr research](2026-09-05-wispr-research.md).

## Complete-clip accuracy gate: 400 public recordings

The corpus contains 200 speaker-balanced clips from each of LibriSpeech
`test-clean` and `test-other`: **73 speakers, 49.15 minutes, 7,973 reference
words**. The same clips were run through each configuration, for 2,400 scored
model calls. All succeeded. These times measure complete-clip model processing,
not streaming finalization.

| Configuration | Model p50 | Model p95 | Clean WER | Other WER | Combined WER | Exact baseline transcripts |
|---|---:|---:|---:|---:|---:|---:|
| Original bfloat16 model/decoder | 87.3 ms | 193.7 ms | 1.859% | 3.813% | 2.784% | 400/400 |
| Compiled decoder + predictor reuse | 75.7 ms | 161.3 ms | 1.859% | 3.813% | 2.784% | 400/400 |
| Above + float16 cast | 78.6 ms | 170.1 ms | 1.859% | 3.813% | 2.784% | 399/400 |
| Above + compiled encoder | 92.9 ms | 184.5 ms | 1.859% | 3.813% | 2.784% | 399/400 |
| Compiled decoder + 8-bit encoder | 56.9 ms | 143.3 ms | 1.835% | 3.839% | 2.784% | 393/400 |
| Compiled decoder + 4-bit encoder | 56.0 ms | 142.5 ms | 1.859% | 3.601% | 2.684% | 323/400 |

Decoder optimization preserved every complete transcript and reduced median
compute time by 13%. An earlier four-clip microbenchmark with three repetitions
per mode found about 20–25% improvement, illustrating the importance of corpus
composition and timing conditions. Those six modes isolated confidence removal,
combined GPU synchronization, predictor reuse, and compilation.

The quantized models reduced median compute time by approximately 35–36%.
The 8-bit model improved two clips' word-error counts and worsened two; 4-bit
improved 14 and worsened 14. The latter produced eight fewer errors overall,
but that is not a reliable accuracy gain. A paired bootstrap by speaker,
stratified by split (2,000 resamples, seed 20260905), gives a 95% interval for
the 4-bit minus baseline WER difference of **−0.275 to +0.052 percentage
points**. For 8-bit, it is −0.051 to +0.048 points. Both include zero.

The float16 runs cast already-loaded bfloat16 weights; they do not recover
higher-precision original weights. Quantization applies to encoder modules
supported by MLX, with the decoder unquantized. The compiled-encoder variant
incurs shape-dependent work and was slower on this varied corpus.

## Streaming screening

Each row below used the same three synthetic speech clips (short, ordinary
dictation, long) and one silence clip, with one scored pass. There are only 92
speech reference words, so these tests screen candidates; they cannot establish
accuracy or tail-latency claims. Each server receives a separate unscored warmup.

| Candidate | Final p50 | Final p95 | Speech WER | Decision |
|---|---:|---:|---:|---|
| Baseline, first screen | 91.0 ms | 190.2 ms | 1.09% | Reference |
| Compiled decoder | 83.1 ms | 170.6 ms | 1.09% | Some improvement |
| Compiled encoder + decoder | 76.0 ms | 178.8 ms | 1.09% | Insufficient alone |
| Optimized decoder, 2 s context, 0.5 s cadence | 67.9 ms | 73.7 ms | 1.09% | Promising window change |
| Native cache, right context 4, 0.5 s chunks | 89.2 ms | 121.7 ms | 58.70% | Reject |
| Native cache, right context 8, 0.5 s chunks | 81.3 ms | 93.0 ms | 44.57% | Reject |
| Native cache, right context 4, 0.25 s chunks | 83.4 ms | 89.9 ms | 66.30% | Reject |
| Native cache retaining full attention | 58.2 ms | 65.7 ms | 17.39% | Reject |
| Baseline, second screen | 86.4 ms | 199.2 ms | 1.09% | Reference |
| 8-bit encoder, 2 s context, 0.5 s cadence | 53.8 ms | 62.6 ms | 1.09% | Near target |
| 4-bit encoder, 2 s context, 0.5 s cadence | 55.3 ms | 70.9 ms | 1.09% | Near target |
| Compiled encoder, 2 s context, 0.5 s cadence | 69.5 ms | 121.3 ms | 1.09% | No benefit |
| Compiled encoder, 1 s context, 0.5 s cadence | 61.4 ms | 112.8 ms | 1.09% | Insufficient |
| Compiled encoder, 0.5 s context, 0.25 s cadence | 57.6 ms | 111.6 ms | 1.09% | Insufficient |
| 8-bit compiled encoder, 1 s padding buckets, 2 s context | 60.1 ms | 61.3 ms | 1.09% | No benefit |
| 4-bit compiled encoder, 1 s padding buckets, 2 s context | 57.7 ms | 71.3 ms | 1.09% | No benefit |
| 8-bit encoder, 1 s context, 0.25 s cadence | 41.4 ms | 51.6 ms | 1.09% | Broader validation |
| 4-bit encoder, 0.5 s context, 0.25 s cadence | 33.1 ms | 63.3 ms | 1.09% | Broader validation |

All non-baseline quantized/window candidates use compiled decoder steps and
predictor reuse. The original context is 10 seconds and cadence 2 seconds.
Shorter cadence does more work while the user speaks and can increase sustained
GPU use; battery, thermal behavior, and concurrent app load were not measured.
The experiments leave the existing one-second settling horizon in place.

The rejected cache rows describe **our experimental adapter and settings**,
not all possible uses of the library's streaming implementation. Native caches
change attention and finalization semantics, and this prototype was not suitable
for deployment. Padding buckets add inference-only zeros after the actual input;
the padding is processed inside the finalization timer. No candidate delays the
timer until after inference or returns a stale partial as a final result.

## Broader streaming validation

This run compares the original server with the two screened candidates on 48
clips: the first 20 imported clips from each public split, six synthetic speech
clips, and two silence clips. Public clips cover 40 speakers. These clips test
the streaming changes beyond the four-clip screen; they are not a held-out set
for every model/quantization choice because they also appear in the complete-clip
accuracy experiment. The matched run uses identical audio and clip order.

| Configuration | Final p50 | Final p95 | WER | Word errors | Calls under 50 ms |
|---|---:|---:|---:|---:|---:|
| Original server | 100.6 ms | 249.2 ms | 2.54% | 22/865 | 0/48 |
| 8-bit encoder, 1 s context, 0.25 s cadence | 46.1 ms | 89.4 ms | 5.55% | 48/865 | 34/48 |
| 4-bit encoder, 0.5 s context, 0.25 s cadence | 40.1 ms | 87.6 ms | 4.39% | 38/865 | 38/48 |

All 144 requests succeeded, and no variant hallucinated words on either silence
clip. **Both fast candidates fail the accuracy gate.** They cut median latency
by 54–60%, but increase total word errors by 118% and 73%, respectively.
Neither achieves sub-50 ms p95. Their speed must not be advertised as an
accuracy-preserving product improvement.

The 8-bit candidate worsened nine public recordings and improved none. Clean
speech WER rose from 1.35% to 6.09%; the difficult split rose from 4.48% to
6.21%. Every synthetic speech clip retained its baseline word-error count.
This is a direct example of a synthetic screen missing a real-speech regression.

The next diagnostic isolates those nine regressions (252 words, 93 seconds).
It varies retained context, partial cadence, and minimum word gap at settlement.
These are deliberately selected failures, so their scores diagnose behavior
and cannot be used as a general accuracy estimate.

| Diagnostic on the nine regressions | Word errors / 252 | Final p50 | Final p95 |
|---|---:|---:|---:|
| Original baseline, from the matched run | 3 | — | — |
| Aggressive 8-bit candidate, from the matched run | 29 | — | — |
| 8-bit, retain 10 s context, 0.25 s cadence | 7 | 95.5 ms | 121.6 ms |
| 8-bit, 2 s context, 0.5 s cadence | 11 | 64.9 ms | 83.8 ms |
| 4-bit, 1 s context, settle at ≥0.16 s word gap | 4 | 78.6 ms | 112.7 ms |
| 8-bit, retain three anchor words + ≥1 s old agreement | 18 | 44.4 ms | 74.2 ms |
| 4-bit, retain three anchor words + ≥1 s old agreement | 14 | 49.8 ms | 87.6 ms |

More context recovers many errors, but does not restore the baseline by itself.
Larger word gaps come closest, with additional latency. The prototype guarding
anchor retention and agreement age also fails the accuracy gate. These tests
implicate aggressive streaming/settlement settings; they do not establish one
isolated root cause or a production-ready replacement algorithm.

## Conservative streaming changes

Keeping the original 10 s context and 2 s partial cadence tests improvements
without the aggressive streaming settings. These two additional passes use the
same 48-clip corpus and order as the broader validation.

| Configuration | Final p50 | Final p95 | WER | Word errors |
|---|---:|---:|---:|---:|
| Original server, preceding matched run | 100.6 ms | 249.2 ms | 2.54% | 22/865 |
| Compiled decoder + predictor reuse | 96.4 ms | 222.3 ms | 2.54% | 22/865 |
| Above + 8-bit encoder | 70.7 ms | 167.5 ms | 2.77% | 24/865 |

Decoder optimization preserves every clip's word-error count. Its 13% median
complete-clip compute gain translates to only a 4% observed streaming median
gain in this single-pass comparison. This is the strongest accuracy-preserving
candidate, but the small latency difference needs repeated, interleaved runs
before making a product claim. The 8-bit variant adds two errors on one public
recording. Neither produces silence hallucinations in these four silence calls.

## Native Core ML execution

A separate Swift worker tests the same Parakeet v2 model family through
[FluidAudio](https://github.com/FluidInference/FluidAudio), configured for CPU
and Neural Engine execution. Actual device placement was not independently
profiled. These are converted Core ML weights, not bit-identical MLX weights.
The library's encoder-precision selector applies to v3; this v2 bundle must
not be described as an 8-bit encoder.

| Corpus | Scored calls | Complete-clip p50 | Complete-clip p95 | WER |
|---|---:|---:|---:|---:|
| 14 synthetic clips, two repeats | 28 | 68.1 ms | 172.2 ms | 2.13% |
| 400 public clips | 400 | 66.1 ms | 151.2 ms | 2.73% |

These client times include base64/JSON serialization, local pipe transport,
model inference, and response decoding; audio conversion and model loading are
outside the timer. Public-corpus model-only p50/p95 are 61.3/137.4 ms. Clean WER
is 1.716% and other WER 3.866%, totaling 218 errors versus the original MLX
model's 222. Four fewer errors do not establish a reliable accuracy advantage.
There were no hallucinations on the four synthetic silence trials.

This is a complete-clip execution experiment, not a native streaming adapter or
key-release-to-paste measurement. Even its warm short-clip model time is around
55–65 ms. Moving execution to Core ML alone did not reach the target. The
retained worker passed one additional scored smoke call after copying it out
of the temporary build directory; that smoke is included in the overall count,
not the table above.

The worker uses FluidAudio revision
`5c19d5e12320e22bbfb7a1877b089d2665a69add`. Its temporary dependency checkout
contains a narrow logging guard, enabled by the benchmark parent, so library
messages cannot persist transcription content. The runner checks the retained
binary against its build metadata before launch. This optional dependency is
built in a temporary Swift package; FreeFlow itself still uses Swift and Make.

## Decision and next experiment

Keep the sub-50 ms configurations out of the app. Decoder optimization merits
an upstream optional fast path because its complete transcripts match the
baseline on all 400 public clips; it is implemented here only as a benchmark
variant. The installed service remains unchanged.

The next architectural experiment should evaluate a model trained for streaming
with persistent encoder state, then measure the small amount of work remaining
at commit. Retrofitting the current full-context model's cache was not accurate
in the tested configurations. FluidAudio documents trained streaming alternatives
and their chunk constraints in its
[benchmark documentation](https://github.com/FluidInference/FluidAudio/blob/5c19d5e12320e22bbfb7a1877b089d2665a69add/Documentation/Benchmarks.md).
Published throughput and chunk sizes cannot establish finalization latency;
those alternatives still need the same accuracy gate and paced replay. This
session does not claim that an untested model will meet either target.

## Conditions and reproduction

- Apple M4 Pro, 24 GiB, arm64, Darwin 27.0.0.
- `parakeet-mlx` 0.5.2; `mlx` 0.31.2; SciPy 1.18.0; aiohttp 3.14.3;
  local STT virtualenv uses Python 3.12, orchestration uses Python 3.9.6.
- Model: `mlx-community/parakeet-tdt-0.6b-v2`; cached weights, offline mode.
- Original STT source SHA-256:
  `ae66a630dcc1bb40ca078c8284ae76ed4d084da9b926720b486f7a56cb8281ca`.
- Installed Parakeet decoder/streaming implementation SHA-256 (`parakeet.py`):
  `2a4208b051ed20356908848c76080ebee7df4a554be91fc338fbe6747b19fccb`.
- Standard-library scorer: Unicode/case/punctuation normalization, with spoken
  numbers and fillers retained. Raw CER on uppercase LibriSpeech references is
  dominated by capitalization and is not used to rank recognition.
- Percentiles use nearest rank. Tiny-screen p95 is effectively its worst call;
  no reliable p99 claim is possible from these streaming sample sizes.
- Streaming uses the production Swift client with paced 20 ms PCM chunks and
  its 24→16 kHz server resampling. The timer begins at commit after all file
  samples, including any trailing silence, and ends at the acknowledged final.
- Variant order is fixed; streaming clip order is shuffled with seed 20260905.
  This is a development machine, not a thermally controlled benchmark host.
  One GPU experiment runs at a time. Repository checks briefly overlapped the
  third screening run; the broader validation runs without those checks.
- No microphone, Accessibility, clipboard, screen context, or real-user history
  was accessed. Cleanup forwarding and heartbeat are disabled in every isolated
  server. Benchmark logs/reports contain numeric scores and public/synthetic IDs.

LibriSpeech was downloaded from [OpenSLR 12](https://www.openslr.org/12), verified
against the publisher's archive MD5s, extracted outside the repository, and
imported with seed 20260905. Original archives and extraction directories were
deleted after the selected reusable corpus was prepared. Attribution and the
CC BY 4.0 license are retained beside each selected corpus.

Run commands and settings are in the
[benchmark README](../../local-setup/benchmark/README.md). The source matrices
are `sub50-screen.json`, `sub50-screen2.json`, `sub50-screen3.json`,
`sub50-accuracy.json`, `sub50-validation.json`, `sub50-diagnostic.json`,
`sub50-guarded.json`, and `sub50-conservative.json`. Keep the prepared corpora
to preserve exact audio rather than regenerating macOS voices.

Reusable models, corpora, and numeric reports are deliberately retained under
`~/.cache/freeflow-benchmark/`. Report names in `results/` are:

- `decoder-micro-20260905.json`
- `sub50-screen-20260905.json`
- `sub50-screen2-20260905.json`
- `sub50-screen3-20260905.json`
- `sub50-accuracy-20260905.json`
- `sub50-validation-20260905.json`
- `sub50-diagnostic-20260905.json`
- `sub50-guarded-20260905.json`
- `sub50-conservative-20260905.json`
- `coreml-synthetic-20260905.json`
- `coreml-accuracy-20260905.json`
- `coreml-worker-smoke-20260905.json`

The 3,258 calls comprise 72 decoder microbenchmark calls, 72 streaming screen
calls, 2,400 complete-clip MLX calls, 144 broader streaming calls, 27 context/gap
diagnostics, 18 guarded-alignment calls, 96 conservative streaming calls, 428
Core ML corpus calls, and one retained-worker smoke call. Warmups are additional
and excluded. Repeats and reused clips are not independent accuracy samples.
The reusable Core ML worker, build metadata, resource bundles, and models are
retained under `bin/` and `models/coreml/`; the dependency checkout and build
directory were removed.

Temporary conversion files, replay executables, and experiment server processes
are removed by the runners. No experimental configuration is activated by the
app's setup or launch scripts. `make check` includes deterministic fake-model
tests of the decoder recurrence and configuration boundaries; actual local model
and Swift replay runs provide the separate integration evidence above.
