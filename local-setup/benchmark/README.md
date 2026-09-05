# Raw transcription benchmarks

Compare FreeFlow's Parakeet v2 streaming and batch paths with whisper.cpp
large-v3-turbo Q5 and base.en on identical audio with known references.
Optional adapters cover hosted Whisper on Groq and Deepgram Nova-3.
This measures raw transcription, before cleanup, voice commands, and paste.

The older `eval_models.py` measures text cleanup, and `bench_e2e.py` compares
streaming output with another transcription. Neither is a ground-truth ASR
accuracy test. This benchmark takes an explicit audio/reference manifest and
never discovers recordings or reads the app's pipeline history.

## Run the local comparison

Requirements: macOS, `python3`, `swiftc`, `ffmpeg`, `whisper-server`, and the
existing FreeFlow Python environment with cached Parakeet v2 weights (see
[local setup](../README.md)). No Python dependencies are added to the app or
benchmark. The local server uses its existing `parakeet-mlx` dependencies.

Download the two reusable Whisper model files once, outside the repository:

```bash
mkdir -p "$HOME/.cache/freeflow-benchmark/models"
curl --fail --location --output "$HOME/.cache/freeflow-benchmark/models/ggml-large-v3-turbo-q5_0.bin" \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
curl --fail --location --output "$HOME/.cache/freeflow-benchmark/models/ggml-base.en.bin" \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

Generate invented dictation with two **installed** macOS voices (use `say -v '?'`
to see available names). Output directories must be new:

```bash
python3 local-setup/benchmark/prepare_corpus.py synthetic \
  --output "$HOME/.cache/freeflow-benchmark/corpora/synthetic-v1" \
  --voice Samantha --voice 'Daniel (English (UK))'

python3 local-setup/benchmark/run_local.py \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/synthetic-v1/manifest.jsonl" \
  --output "$HOME/.cache/freeflow-benchmark/results/local-001.json" \
  --repeats 3
```

The wrapper starts three isolated servers on unused loopback ports, runs four
engine configurations serially, and stops the servers even on interruption.
It disables Parakeet's cleanup precache forwarding and idle heartbeat; it pins
the streaming parameters to the repository defaults. It leaves the installed
app and its services alone. Cached weights are required; it does not download
Parakeet implicitly. Whisper uses Metal/default decoding, four threads, no VAD,
no vocabulary hints, and no prompt. The model field labels the loaded Whisper
weights; whisper.cpp cannot select a different model per request.

Run on an idle machine, on power, and avoid dictating into FreeFlow during the
run. The wrapper does not stop existing services or other GPU workloads.
Synthetic voice generation varies with installed macOS voice versions; retain
the corpus to reproduce a run. Its audio/reference digest is recorded.

## What the numbers mean

| Metric | Definition |
|---|---|
| WER | Sum of word edit distances / total reference words across successful speech trials. Lower is better; can exceed 100%. |
| CER | Same calculation over normalized characters, including spaces. |
| Raw CER | Case/punctuation-sensitive character error rate against the supplied reference. |
| Exact rate | Fraction of successful speech trials whose normalized text matches. |
| After audio p50/p95 | Streaming: commit-to-final. Batch: request-to-final after the complete file is available. |
| Total p50/p95 | Streaming: session start through paced playback to final. Batch: complete request/response time. |
| Wall RTF | Sum of total wall times / sum of audio durations. Streaming includes playback and is normally above 1; batch below 1 runs faster than audio duration. This is not GPU compute time. |
| Failures | HTTP errors, timeouts, malformed results, or replay drift. No retries or automatic fallback hide failures. |
| Silence false positives | Empty-reference clips producing words, reported separately from speech WER. |

Normalization uses Unicode NFKC and casefolding, removes apostrophes, replaces
other punctuation with spaces, and collapses whitespace. Fillers count. Number
formatting is intentionally not normalized: “twenty two” and “22” differ. Check
the `names-numbers` category when interpreting WER. Raw CER additionally measures
formatting. Repeated clips increase timing observations, not speaker diversity.

All batch engines receive the same mono 16 kHz PCM WAV. The production Swift
streaming client receives a 24 kHz resampling of that WAV in paced 20 ms chunks;
the server resamples back to 16 kHz. That conversion is part of the app's
streaming path and can affect accuracy. References are never sent to engines.
File decoding/resampling and Swift compilation occur before timing. Batch timing
includes multipart construction, upload, queueing, inference, download, and JSON
parsing. Swift timing excludes helper process launch. It uses a monotonic clock.
“After audio” includes the entire file, including any trailing silence, and does
not claim to measure the acoustic endpoint or physical hotkey release.

Each engine gets a separately recorded unscored warmup. All scored calls are
serial, with deterministic shuffling of clip and engine order to reduce ordering
bias. These are warm service measurements, not cold model-load measurements.
Failures are excluded from success-only accuracy and latency and must be checked
before comparing scores; different success subsets are not a fair ranking.
The runner exits nonzero if any scored call fails. JSON includes per-category
summaries and numeric per-clip scores to make failures and outliers inspectable.

The tiny synthetic corpus checks integration and exposes obvious regressions.
It does **not** establish real-world dictation accuracy, accent/noise robustness,
or a reliable tail-latency percentile. Do not turn its WER into “percent accurate”
marketing. No full UI, microphone, or paste behavior is measured.

## Public speech accuracy set

For a useful next accuracy run, use at least 200 clips from each of LibriSpeech
`test-clean` and `test-other`, as separate runs. Download/extract the desired
split from [OpenSLR 12](https://www.openslr.org/12) outside the repo. LibriSpeech
contains read English speech and is CC BY 4.0; it is not conversational dictation.

```bash
python3 local-setup/benchmark/prepare_corpus.py librispeech \
  --source /path/to/LibriSpeech/test-clean \
  --output "$HOME/.cache/freeflow-benchmark/corpora/test-clean-200" \
  --limit 200 --seed 20260905
```

The importer samples across speakers with a fixed seed, copies selected FLACs,
preserves the supplied reference text, and writes attribution. It does not
download anything. Point `run_local.py` at the resulting `manifest.jsonl`.
Add separate spontaneous speech, accents, noise, terminology, and longer clips
before drawing broad conclusions. Public benchmarks may overlap model training
or model-selection data; report the exact split and corpus digest.

Custom manifests are JSONL with `id`, `audio` (relative to the manifest), `text`
(verbatim spoken reference), and optional `category`, `source`, and `license`.
An empty `text` means expected silence, not an unknown reference. Supply only
audio you have deliberately chosen for evaluation. Do not export app history
into this benchmark. Keep audio/manifests outside the tracked repository.

## Hosted engines and other tools

`benchmark.py` can target existing servers using `engines.json` or a custom
config. Select engines with repeatable `--engine ID`. `multipart` supports
OpenAI-compatible audio transcription endpoints; `deepgram` supports the
pre-recorded Listen API. Deepgram formatting is disabled and fillers enabled.
`freeflow-stream` compiles `Replay.swift` with the production realtime client and
is restricted to loopback addresses. It does not implement cloud streaming.

`engines.cloud.example.json` includes Groq Whisper large-v3, Groq Whisper Turbo,
and Deepgram Nova-3. With the relevant API keys already in your shell environment:

```bash
python3 local-setup/benchmark/benchmark.py \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/synthetic-v1/manifest.jsonl" \
  --config local-setup/benchmark/engines.cloud.example.json \
  --engine deepgram-nova-3 --allow-remote \
  --output "$HOME/.cache/freeflow-benchmark/results/deepgram-001.json"
```

`--allow-remote` explicitly permits uploading the selected corpus to the selected
HTTPS providers; their usage billing applies. Without it, only literal loopback
IPs are accepted. Keys are read only from the configured environment variable,
not `.env`, Keychain, app settings, or launchd. Redirects and inherited HTTP
proxies are disabled. Reports and progress output omit transcripts, audio,
credentials, endpoint URLs, and local paths. IDs, categories, and model labels
are user-supplied report metadata: use generic labels without private content.

Consumer apps such as Apple Dictation or Wispr Flow require a separately
controlled UI replay workflow. Engine scores here do not measure those apps.
See the [Wispr research](../../docs/benchmarks/2026-09-05-wispr-research.md)
for its published latency claims, API option, and the requirements for a fair
speed and accuracy comparison.

API references checked September 5, 2026:
[whisper.cpp server](https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/README.md),
[Parakeet v2 MLX model](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2),
[Groq STT](https://console.groq.com/docs/speech-to-text),
[Deepgram Listen](https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded).

## Verification and artifacts

`make check` runs synthetic unit tests for scoring, corpus sampling, HTTP payloads,
error accounting, and upload boundaries. No test calls a live provider.
Live engine runs are a separate manual check. The wrapper deletes its temporary
configuration and stops its servers; the runner removes converted audio and the
compiled replay helper. Reusable corpora, model caches, and explicitly named
JSON/Markdown results stay in the directory you selected. No app release or
permission changes are needed.

## Sub-50 ms experiments

The experiment runners operate independently of app deployment. They use the
existing local STT virtualenv, pinned Parakeet v2 weights, and explicit corpora.
They never enable experimental decoding in the installed app. Keep results
outside the repository, and use a new output filename on each run.

Screen streaming configurations through the production Swift replay client:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 local-setup/benchmark/experiment_matrix.py \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/synthetic-v1/manifest.jsonl" \
  --matrix local-setup/benchmark/sub50-screen2.json \
  --output "$HOME/.cache/freeflow-benchmark/results/screen-001.json" \
  --clip tts-1-1 --clip tts-1-2 --clip tts-1-6 --clip silence-2
```

Omit `--clip` to score the whole corpus; use `--repeats` for timing repeats.
`sub50-screen.json` includes the initial native-cache experiments, which had
large accuracy regressions and must not be enabled in production.
`sub50-screen3.json` explores quantization, audio padding buckets, and shorter
context/cadence. `sub50-validation.json` compares the baseline with the two
sub-50 ms screening candidates. To reproduce the 48-clip validation manifest,
combine the first 20 rows of each imported public split with synthetic IDs
`tts-1-1`, `tts-1-3`, `tts-1-4`, `tts-1-5`, `tts-2-1`, `tts-2-6`, `silence-2`,
and `silence-8`, retaining each audio file's location. Keep that manifest outside
the repository. Both runners exit nonzero on failed calls; experiment matrices
validate settings before inference and discard inherited experimental settings.
They explicitly set `STT_FAST_DECODER=0` so the original baseline remains
unmodified when the production service includes its own decoder optimization.

The 48-clip validation **rejected both sub-50 ms candidates because WER rose**.
`sub50-diagnostic.json` isolates context/cadence/gap effects on the nine recordings
that regressed in the 8-bit candidate. `sub50-guarded.json` tests an additional
prototype that retains three preceding anchor words and compares against a
hypothesis at least one second older. These selected failures are diagnostic
data, not a fresh accuracy test. None of these prototypes is production-ready.
`sub50-conservative.json` retains the original streaming settings while testing
decoder optimization and encoder quantization separately.

Test complete-clip model accuracy and compute time on the two prepared public
corpora (200 clips per split):

```bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/.freeflow-ft/venv/bin/python" \
  local-setup/benchmark/accuracy_matrix.py \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/test-clean-200/manifest.jsonl" \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/test-other-200/manifest.jsonl" \
  --matrix local-setup/benchmark/sub50-accuracy.json \
  --output "$HOME/.cache/freeflow-benchmark/results/accuracy-001.json"
```

`--limit N` limits each input manifest. This runner times complete-clip model
processing; its latency must not be labeled streaming or app latency. It holds
baseline hypotheses only in process memory to check exact equivalence. JSON
contains hashes, numeric errors, timing, settings, and exact-match flags, never
reference or generated text. Per-corpus metrics remain separate. LibriSpeech
references are uppercase; its raw CER is dominated by formatting and is not a
useful recognition metric.

Both runners use the matrix order and warm each model before scoring; streaming
clips have a deterministic shuffle. They do not randomize variant order, so
machine load and thermal drift can influence comparisons. Run one GPU benchmark
at a time. The model runner keeps all prepared audio and temporary hypotheses
in memory, suitable for these small corpora rather than arbitrarily large sets.

`experimental_decoders.py` follows the Parakeet 0.5.2 greedy recurrence and
tests confidence removal, fewer GPU synchronizations, compilation, and predictor
reuse on blank tokens. It deliberately emits unscored confidence values and is
unsuitable for consumers of token confidence. Quantization affects encoder
layers supported by MLX's quantizer; the decoder remains at its original
precision. The float16 experiment casts the already-loaded bfloat16 weights;
it does not reload higher-precision weights. The native-cache server supports
only the runner's serial sessions and is a research prototype.

The decoder derivative carries its [Apache 2.0 license](licenses/parakeet-mlx-Apache-2.0.txt).
Deterministic fake-model tests cover blank-state reuse, token timing, zero-step
termination, batch independence, and fallback. They test control flow; actual
numeric accuracy and speed require the separate local model runs above.

Measured outcomes, rejected candidates, limitations, and retained report names
are recorded in the [experiment report](../../docs/benchmarks/2026-09-05-sub50-experiments.md).

## Optional native Core ML comparison

This complete-clip benchmark uses a pinned FluidAudio dependency in a temporary
Swift package. It does not add a dependency to the FreeFlow app or change its
Make-based build. macOS 14+, Swift 6, and `ffmpeg` are required. Building downloads
the pinned dependency and its binary dependency; the first worker launch downloads
the public Parakeet v2 Core ML model bundles into the selected model directory.
Budget approximately 500 MB for model weights plus build/cache space. Audio is
sent only over local parent-owned pipes and never to a model provider.

```bash
python3 local-setup/benchmark/build_coreml.py \
  --output "$HOME/.cache/freeflow-benchmark/bin/FreeFlowCoreML"

python3 local-setup/benchmark/coreml_benchmark.py \
  --worker "$HOME/.cache/freeflow-benchmark/bin/FreeFlowCoreML" \
  --models "$HOME/.cache/freeflow-benchmark/models/coreml/parakeet-tdt-0.6b-v2" \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/test-clean-200/manifest.jsonl" \
  --manifest "$HOME/.cache/freeflow-benchmark/corpora/test-other-200/manifest.jsonl" \
  --output "$HOME/.cache/freeflow-benchmark/results/coreml-001.json"
```

Build outputs must be new. Keep the worker's adjacent JSON metadata and resource
bundles together. The builder adds a logging guard to the temporary dependency;
the runner enables it and verifies the worker hash before launch. Use the Python
runner, not the worker directly: the worker's stdout carries transcripts for
in-memory scoring and must never be redirected to a console or log file.
Reports contain numeric scores and hashes, without text or token payloads.

`--limit N` limits each manifest and `--repeats N` repeats scoring after one warmup.
Complete-clip latency includes local serialization/IPC; `model_s` isolates the
inference call. Neither measures streaming, microphone shutdown, or paste.
The v2 model does not use the library's v3 encoder-precision selector, and no
claim of 8-bit v2 weights is made. No VAD, cleanup, or personal dictionary is added.
Deterministic tests mock the worker and verify logging metadata, error handling,
and report privacy; actual model results are documented separately.
