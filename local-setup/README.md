# FreeFlow — fully local / hybrid speech-to-text setup

For synthetic latency checks and the manual microphone/paste checklist, see
[Pipeline validation](PIPELINE_VALIDATION.md).

This fork adds a **100% local** (or fast hybrid) speech-to-text + cleanup pipeline for
FreeFlow, so you don't need to pay for a hosted transcription API.

```
hold Cmd+Ctrl ─▶ Parakeet-TDT (local, MLX)  ──▶  router ──▶ pre-cleaned chunks + deterministic tail (~10ms, no model)
                  words settle while you talk;              ├─ Groq gpt-oss-20b, then the other buckets (hosted-first)
                  key-up → text in ~0.2s                    └─ local Qwen via mlx_lm — offline / quota fallback
                                                               (FORCE_LOCAL=1 puts it first, e.g. with a good fine-tune)
```

- **Transcription** runs locally with **Parakeet-TDT 0.6B** via `parakeet-mlx`
  (`stt_server.py`): it speaks both the OpenAI upload endpoint *and* the OpenAI
  Realtime WebSocket protocol FreeFlow already supports, so it transcribes **while
  you're still speaking** — at key-up only the last second is left (~0.1s, measured
  vs 0.5–0.7s for whisper.cpp, whose encoder always runs on a padded 30s window).
  whisper.cpp remains supported as an alternative (`com.freeflow.whisper-server.plist`).
- **Cleanup LLM** goes through a tiny local **router** (`router.py`) that tries a fast
  hosted model (Groq `gpt-oss-20b`, free tier), then other Groq models — each has its
  **own** per-minute token bucket, so one 429 doesn't stall you — and **falls back to a
  local model when offline / rate-limited / out of credits**. It watches Groq's
  `x-ratelimit-*` headers to skip a bucket it *knows* is empty (no wasted round trip),
  and respects the app's deadline so it never answers after FreeFlow already gave up.
  You can also run it fully local (`FORCE_LOCAL=1`, or no Groq key).
- **Local model:** either Ollama (`llama3.1:8b`, generic) or — better — the small
  **on-device fine-tuned Qwen** served by `mlx_lm` (see `HOW_IT_WORKS.md`): always
  resident, ~0.4s, no 8s cold start, and it doesn't hallucinate content from context.

Measured on an M-series Mac (2026-08): whisper 19s audio → 0.7s; Groq gpt-oss-20b
0.55s; local fine-tuned Qwen 0.45s. `python3 eval_models.py` scores them all.

Everything runs under `launchd`, so it survives reboots.

## 1. Install dependencies

```bash
python3 -m venv ~/.freeflow-ft/venv && ~/.freeflow-ft/venv/bin/pip install parakeet-mlx aiohttp scipy mlx-lm
#                                   ^ STT server + fine-tune/serve tooling (Apple Silicon)
brew install --cask ollama-app     # optional: generic local cleanup LLM if you don't fine-tune
brew install whisper-cpp           # optional: whisper.cpp instead of Parakeet
```

## 2. Download models

```bash
mkdir -p ~/.freeflow-stt/models && cd ~/.freeflow-stt/models
# Whisper large-v3-turbo, q5_0 quantized (~574MB, ~1GB RAM, near-large accuracy)
curl -L -o ggml-large-v3-turbo-q5_0.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
# Silero VAD — stops Whisper hallucinating "Thank you" on silence
curl -L -o ggml-silero-v5.1.2.bin \
  "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"

ollama pull llama3.1:8b            # local cleanup fallback model
```

## 3. Install the STT server + router

```bash
cp stt_server.py router.py start-ollama.sh ~/.freeflow-stt/
```

`stt_server.py` (port 8082) downloads `mlx-community/parakeet-tdt-0.6b-v2` on first
start (~1.2 GB; `-v3` is multilingual, see the note in the file). While you talk it
logs `settled: +38 chars (8 words, settled to 9.5s of 12.0s, window 12.0s → 123ms)`
lines; at key-up one `commit: tail 3.1s (window 13.1s) → final in 201ms` line.
`curl 127.0.0.1:8082/health` reports its MLX memory. Tests: `python3 test_stt_server.py`
(protocol contract, settling, splicing, resampling) and
`python3 test_stt_server.py --real some.wav --pace` (streams a real file at
real-time speed and prints commit→final latency).

### Router

```bash
cp router.py ~/.freeflow-stt/router.py
```

The router serves an OpenAI-compatible `/v1/chat/completions` on `127.0.0.1:11435`,
tries Groq first (`GROQ_FALLBACK_MODELS` next), then the local model. It reads the
Groq key from `GROQ_API_KEY` (set in the launchd plist below). For **fully local**,
leave the key blank or set `FORCE_LOCAL=1`. All knobs are documented at the top of
`router.py`; `curl 127.0.0.1:11435/v1/status` shows what it currently believes about
each Groq bucket. Tests: `python3 test_router.py`.

> **Why the router cares about tokens:** Groq's free tier is 8k tokens/minute *per
> model* and it reserves the request's `max_completion_tokens` up front. FreeFlow now
> sizes that budget to the transcript (instead of a flat 4096) — together with the
> multi-bucket chain that takes you from ~1 to ~8 dictations/minute before anything
> touches the local model.

## 4. launchd services

Copy the plists from `launchd/` into `~/Library/LaunchAgents/`, **edit the
placeholders** (`YOUR_USERNAME`, `YOUR_GROQ_API_KEY` — get a free key at
https://console.groq.com), then:

```bash
launchctl load ~/Library/LaunchAgents/com.freeflow.stt.plist         # Parakeet STT (:8082)
launchctl load ~/Library/LaunchAgents/com.freeflow.localmodel.plist  # fine-tuned cleanup model (:8081), if trained
launchctl load ~/Library/LaunchAgents/com.freeflow.router.plist      # (:11435)
launchctl load ~/Library/LaunchAgents/com.freeflow.ollama.plist      # optional generic fallback
# alternative STT: com.freeflow.whisper-server.plist (:8080)
```

Ollama is RAM-capped in its plist (single model, single parallel slot, 30-min idle
unload, flash attention, q8 KV cache) so it stays light on a 24GB machine. It runs
through `start-ollama.sh` (copy it next to `router.py`), which steps aside if the
Ollama **desktop app** already owns port 11434 — note that in that case the app's
defaults apply, not the plist's; the router re-arms a 30-min keep-alive after every
local call either way so the fallback stays warm.

### Optional: the on-device fine-tuned model instead of Ollama

Once you've trained an adapter (`finetune_local.sh`, see `HOW_IT_WORKS.md`), install
`launchd/com.freeflow.localmodel.plist` (mlx_lm server on `:8081`) and set in the
router plist:

```
LOCAL_URL=http://127.0.0.1:8081/v1/chat/completions
LOCAL_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit   # must equal the server's --model
LOCAL_PROMPT_FORMAT=train                              # important — see below
```

**Don't fuse the adapter into the 4-bit base.** It looks tempting (+34% decode
speed measured) but the dequantize→requantize round trip is lossy: WER 14% → 29% on
the probe set, with the model returning raw text unchanged. Fusing to fp16 and
requantizing at 8 bits keeps accuracy (14.8%) but is *not* faster (bigger weights,
memory-bound decode). Serve `--model <base-4bit> --adapter-path <adapters>`.

**Model size.** A 0.5B fine-tune on the same data decodes ~2× faster (~0.28s vs
0.43s per cleanup) but scores 21% WER with occasional garbage merges; the 1.5B stays
the default. Train one with `BASE_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
./finetune_local.sh` and compare with `eval_models.py` if you want to trade.

**Hosted first, unless you have a fine-tune you trust.** With `FORCE_LOCAL=0`
(the default) a dictation that needs the model goes to Groq gpt-oss-20b first
(~0.3–0.9s, and the better cleaner) and to the local model only offline or out of
quota; `1` puts the local model first (~0.3–0.45s, no network). Either way the
router keeps the local model resident with a 1-token touch every
`LOCAL_HEARTBEAT_S` (60) — otherwise macOS pages it out between dictations and
the first request after a break pays 7–11s. Most dictations never reach either:
see *How the pipeline stays fast*.

`LOCAL_PROMPT_FORMAT=train` makes the router re-render FreeFlow's runtime prompt into
the exact format the model was trained on. Skipping this is a silent accuracy bug:
on the probe set the same adapter goes from WER 44% / 4 context-leak hallucinations
(runtime format) to WER 14% / 0 (training format).

## 5. Point FreeFlow at it

In FreeFlow Settings (or via `defaults`/the `.settings` file), set:

- **Transcription API URL:** `http://127.0.0.1:8082/v1` (Parakeet; `:8080/v1` for whisper)
- **Realtime streaming: on** — `defaults write com.zachlatta.freeflow realtime_streaming_enabled -bool true`
  (this is what makes transcription happen while you speak; the batch upload is the fallback)
- **LLM / API base URL:** `http://127.0.0.1:11435/v1` (the router)
- **Transcription model:** any (`whisper-large-v3-turbo` is fine; the local server ignores it)
- **Post-processing model:** `openai/gpt-oss-20b` (the router maps this to the local
  model when offline)
- Any non-empty API key string works locally.

> Note: FreeFlow reads the **model** settings from `UserDefaults`
> (`defaults read com.zachlatta.freeflow …`), not from the `.settings` file — only the
> API URLs/keys live in `.settings`.

## Skipping the LLM entirely for clean short utterances

Modern STT already punctuates and capitalises. If the transcript is a short sentence
(≤ 12 words) with no fillers, self-corrections, spoken punctuation or mis-cased
vocabulary, the app pastes it straight away instead of paying the cleanup round trip
(`Sources/TranscriptFastPath.swift`). Tune or disable:

```bash
defaults write com.zachlatta.freeflow clean_transcript_fast_path_max_words -int 12   # 0 = off
```

## Verify

```bash
# transcription
curl -s http://127.0.0.1:8080/v1/audio/transcriptions -F file=@some16k.wav -F model=x
# cleanup routing (look for the X-FreeFlow-Route header)
curl -s -D - http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"clean: helo wrld"}]}'
# what the router thinks about each Groq bucket / whether the local model is warm
curl -s http://127.0.0.1:11435/v1/status
```

## How the pipeline stays fast (what to keep in mind when changing things)

Measured with `bench_e2e.py` on 8 real recordings (4–53s) on 2026-09-02, on a
machine at load average 20: commit→final **0.18–0.54s** with the key-up tail
bounded to 3–8s of audio however long the dictation, cleanup **~10ms** when the
router could assemble it deterministically (3 of 8), otherwise Groq at 0.46–0.65s;
streamed-vs-batch WER 2.8%, every difference a filler/stutter or "90"/"ninety".

* **Words settle while you talk** (`stt_server.py`, `:8082`, `com.freeflow.stt`).
  Every 2s of audio the server decodes a *window*: the unsettled audio plus 10s of
  already-settled audio in front of it. A word the last two windows agree on
  (compared without case/punctuation) that ended ≥1s before the live edge is
  settled: emitted to the app as a `completed` event and POSTed to the router's
  `/v1/precache`. At key-up the same window decode covers just the tail and is
  spliced after the settled text by *word anchor* (the last three settled words;
  TDT timestamps are quantised to 80ms, so timestamps alone duplicate or drop a
  boundary word). The audio is never cut: hard cuts at pauses were measured to
  change 2.7% of words, mostly casing and spelling at the cut. v2 (English) is the
  default because v3 dropped words / returned "" on 10/46 real tail clips.
* **The mic stream is resampled with context.** Each `append` message used to be
  resampled on its own, which put a filter transient at every message edge — 0.4%
  of words changed on real recordings, 6% on a short one. `StreamResampler` holds
  back 10ms of input so every emitted sample had real neighbours on both sides.
* **Memory is bounded, and the weights stay hot.** MLX keeps every freed buffer
  in a cache keyed by size; clips of ever-different lengths never reuse one, and
  the process reached 17 GB in nine days — which swapped the whole machine, and
  every dictation after a break paid 7–13s of page-ins for *both* models.
  `STT_CACHE_MB` (512) caps it with no measurable latency cost, `STT_WIRED_MB`
  wires the weights, and both servers are touched while idle (`STT_HEARTBEAT_S`
  45, router `LOCAL_HEARTBEAT_S` 60). `curl :8082/health` shows the numbers.
* **Deterministic cleanup first, LLM only when needed.** Parakeet already punctuates
  and capitalises, so most of what the model did was mechanical: strip "um/uh",
  convert a dictated "comma", fix a capital. `TranscriptFastPath.swift` (app, ≤60
  words) and its twin `detclean.py` (router) do that in ~0ms and *bail to the
  model* for anything needing interpretation: self-corrections and restarts ("no
  actually", "is there anything is there any way"), dictated formatting, greetings
  (email layout), quotes/brackets, repeated sentences, mis-cased vocabulary (never
  with a custom system prompt or output language). Measured against the LLM's own
  output on real transcripts: identical words on everything it accepts.
* **Settled chunks are pre-cleaned as fragments and assembled at commit.** The
  router cleans each chunk deterministically without sentence-casing or a closing
  full stop (a mid-sentence chunk must not come back as "…should be. An action."),
  and at commit `finalize_text` cases and closes the assembled whole. If every
  chunk and the tail were deterministic the router answers in ~10ms with no model
  (`route=precache`); if any piece needs judgement the *whole* dictation goes to
  the normal chain — a small base model must not judge a self-correction in
  isolation (`PRECACHE_LLM=1` re-enables that for a fine-tune you trust).
* **The router must recognise the app's request.** It parses the cleanup user
  turn to know a request *is* a cleanup; upstream changed that turn to a heredoc
  in 2026-06 and the router silently treated every real dictation as "not a
  cleanup" for two months while the bench (which used the old shape) kept
  reporting 150ms. `app_prompt.py` now pins the shape, `test_router.py` checks it
  against the Swift source, and the bench sends exactly it.
* **Context (screenshot) calls are priced correctly and never go local.** Groq
  charges ~1.8k tokens per image; counting the base64 as text predicted ~25k, so
  every context call was "predicted 429", sent to the text-only local server, and
  sat behind its warm-up for up to 10s. Images now cost `GROQ_IMAGE_TOKENS`, the
  local backend refuses them in microseconds (the app's text-only retry follows),
  and their completion budget is capped (`CONTEXT_MAX_COMPLETION_TOKENS`, 256).
* **GPU warm-up**: Apple GPUs downclock when idle (first inference 2–3× slower), so
  `/warm` (hotkey-down) pokes the local LLM with a *prefix-compatible* prompt and the
  STT server runs a tiny inference on session open.
* Things measured and rejected: parakeet-mlx native token streaming (slower than real
  time, less accurate), speculative decoding with a 0.5B draft (slower at ~45 output
  tokens), 0.5B fine-tune (2× faster, WER 21%), fusing the LoRA into 4-bit (lossy),
  cutting the audio at pauses or token gaps (2.7% of words changed), a 2s settle
  horizon (no more accurate than 1s, just later).

```bash
~/.freeflow-ft/venv/bin/python local-setup/bench_e2e.py --files 6 --label after-change
```

## Evaluate speed + accuracy of every backend

```bash
GROQ_API_KEY=... python3 eval_models.py --test ~/.freeflow-ft/probe_test.jsonl \
  --probes hallucination_probes.jsonl --parallel \
  --endpoint https://api.groq.com/openai/v1/chat/completions --endpoint-model openai/gpt-oss-20b --endpoint-label gpt-oss-20b --api-key-env GROQ_API_KEY \
  --endpoint http://127.0.0.1:8081/v1/chat/completions --endpoint-model mlx-community/Qwen2.5-1.5B-Instruct-4bit --endpoint-label local-qwen --api-key-env -
```

`--format app` (default) sends exactly what FreeFlow sends; `--format train` sends the
fine-tune's training format. `hallucination_probes.jsonl` are transcripts paired with
tempting CONTEXT (names, weekdays, file names) — the `leak` column counts outputs that
contain any of it, `ins` is the share of output words that appear in neither the raw
transcript nor the gold text.
