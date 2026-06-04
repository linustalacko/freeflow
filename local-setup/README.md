# FreeFlow — fully local / hybrid speech-to-text setup

This fork adds a **100% local** (or fast hybrid) speech-to-text + cleanup pipeline for
FreeFlow, so you don't need to pay for a hosted transcription API.

```
hold Cmd+Ctrl ─▶ whisper.cpp (local, Metal)  ──▶  router ──▶ online?  Groq gpt-oss-20b
                  /v1/audio/transcriptions                  └─ offline? local Ollama (llama3.1:8b)
```

- **Transcription** runs locally on `whisper.cpp` (Apple Silicon Metal), exposing an
  OpenAI-compatible endpoint that FreeFlow points at.
- **Cleanup LLM** goes through a tiny local **router** (`router.py`) that tries a fast
  hosted model (Groq `gpt-oss-20b`, free tier) and **falls back to a local Ollama model
  when offline / rate-limited / out of credits**. You can also run it fully local.

Everything runs under `launchd`, so it survives reboots.

## 1. Install dependencies

```bash
brew install whisper-cpp           # local STT (Metal-accelerated)
brew install --cask ollama-app     # local cleanup LLM (the formula ships without the runner; use the cask)
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

## 3. Install the router

```bash
cp router.py ~/.freeflow-stt/router.py
```

The router serves an OpenAI-compatible `/v1/chat/completions` on `127.0.0.1:11435`,
tries Groq first, falls back to local Ollama. It reads the Groq key from
`GROQ_API_KEY` (set in the launchd plist below). For **fully local**, just leave the
key blank — it always falls back to Ollama.

## 4. launchd services

Copy the three plists from `launchd/` into `~/Library/LaunchAgents/`, **edit the
placeholders** (`YOUR_USERNAME`, `YOUR_GROQ_API_KEY` — get a free key at
https://console.groq.com), then:

```bash
launchctl load ~/Library/LaunchAgents/com.freeflow.whisper-server.plist
launchctl load ~/Library/LaunchAgents/com.freeflow.ollama.plist
launchctl load ~/Library/LaunchAgents/com.freeflow.router.plist
```

Ollama is RAM-capped in its plist (single model, single parallel slot, 2-min idle
unload, flash attention, q8 KV cache) so it stays light on a 24GB machine.

## 5. Point FreeFlow at it

In FreeFlow Settings (or via `defaults`/the `.settings` file), set:

- **Transcription API URL:** `http://127.0.0.1:8080/v1`
- **LLM / API base URL:** `http://127.0.0.1:11435/v1` (the router)
- **Transcription model:** `whisper-large-v3-turbo`
- **Post-processing model:** `openai/gpt-oss-20b` (the router maps this to the local
  model when offline)
- Any non-empty API key string works locally.

> Note: FreeFlow reads the **model** settings from `UserDefaults`
> (`defaults read com.zachlatta.freeflow …`), not from the `.settings` file — only the
> API URLs/keys live in `.settings`.

## Verify

```bash
# transcription
curl -s http://127.0.0.1:8080/v1/audio/transcriptions -F file=@some16k.wav -F model=x
# cleanup routing (look for the X-FreeFlow-Route header)
curl -s -D - http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"clean: helo wrld"}]}'
```
