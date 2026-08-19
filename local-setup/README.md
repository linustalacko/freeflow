# FreeFlow — fully local / hybrid speech-to-text setup

This fork adds a **100% local** (or fast hybrid) speech-to-text + cleanup pipeline for
FreeFlow, so you don't need to pay for a hosted transcription API.

```
hold Cmd+Ctrl ─▶ Parakeet-TDT (local, MLX)  ──▶  router ──▶ local fine-tuned Qwen (~0.3s)   [FORCE_LOCAL=1]
                  streams while you talk;                   ├─ local down? Groq gpt-oss-20b, then other buckets
                  key-up → text in ~0.1s                    └─ (or Groq first / local fallback with FORCE_LOCAL=0)
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

`stt_server.py` (port 8082) downloads `mlx-community/parakeet-tdt-0.6b-v3` on first
start (~1.2 GB; `-v2` is English-only and slightly stronger there). It logs one line
per dictation: `commit: tail 1.2s → final in 96ms`. Tests: `python3 test_stt_server.py`
(protocol contract) and `python3 test_stt_server.py --real some.wav --pace`
(streams a real file at real-time speed and prints commit→final latency).

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

**Local first for speed.** `FORCE_LOCAL=1` puts the local model first (~0.3–0.45s
per cleanup, no network) with the Groq chain as the safety net; `0` = Groq first.

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

Measured with `bench_e2e.py` on real recordings (M-series, 2026-08): key-up → cleaned
text **median ≈0.15s, max ≈0.26s** (whisper + Groq era: 1.1–1.4s, sometimes 8s+).

* **Parakeet-TDT v2 via `stt_server.py`** (`:8082`, `com.freeflow.stt`): no 30-s
  encoder floor like whisper; transcribes finished segments while you speak; the
  tail at key-up costs ~0.1s. v2 (English) is the default because v3 dropped words /
  returned "" on 10/46 real tail clips (v2: 0/46). Segments only split on ≥0.8s of
  true silence after ≥3s speech, cut in the middle of the pause — more eager settings
  were measured to introduce word errors at the boundaries.
* **Cleanup on the on-device fine-tuned Qwen 1.5B** (`FORCE_LOCAL=1`, `mlx_lm` server
  `:8081`): ~7ms/token, hardware-bound — the router therefore works on generating
  *fewer tokens at key-up*: the STT server POSTs each finalized segment to the router's
  `/v1/precache`, which cleans it during speech; the app's real request then reuses the
  cleaned prefix and only the tail is generated (`precache hit:` lines in router.log).
* **Only cleanup prompts go local-first.** The context-inference call at hotkey-down
  is hosted (Groq qwen, ~0.2s): on the local model it cost 0.5s of GPU *and* evicted
  mlx_lm's single-prompt KV cache, adding ~170ms prefill to every cleanup.
* **GPU warm-up**: Apple GPUs downclock when idle (first inference 2–3× slower), so
  `/warm` (hotkey-down) pokes the local LLM with a *prefix-compatible* prompt and the
  STT server runs a tiny inference on session open.
* **Deterministic cleanup first, LLM only when needed.** Parakeet already punctuates
  and capitalises, so most of what the model did was mechanical: strip "um/uh",
  convert a dictated "comma", fix a capital. `TranscriptFastPath.swift` (app) and its
  twin `detclean.py` (router, for pre-cached chunks and the tail at commit) do that in
  ~0ms and *bail to the model* for anything needing interpretation: self-corrections
  and restarts ("no actually", "is there anything is there any way"), dictated
  formatting, greetings (email layout), quotes/brackets, repeated sentences,
  mis-cased vocabulary, or >60 words (`clean_transcript_fast_path_max_words`; 0 = off;
  never used with a custom system prompt or output language). Measured against the
  LLM's own output on real transcripts: identical words on everything it accepts.
* **Pause-less speech is pre-cleaned too**: every ~2s of speech the STT server
  transcribes the buffer-so-far and hands the router every complete sentence that
  ended ≥0.8s ago; the router matches by words (punctuation-insensitive), cleans
  only the new sentences, and rejects a chunk the model shortened by >40%. The cache
  is reset at each new dictation so nothing stale can ever match.
* Things measured and rejected: parakeet-mlx native token streaming (slower than real
  time, less accurate), speculative decoding with a 0.5B draft (slower at ~45 output
  tokens), 0.5B fine-tune (2× faster, WER 21%), fusing the LoRA into 4-bit (lossy).

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
