<p align="center">
  <img src="Resources/AppIcon-Source.png" width="128" height="128" alt="FreeFlow icon">
</p>

<h1 align="center">FreeFlow</h1>

<p align="center">
  Free and open source alternative to <a href="https://wisprflow.ai">Wispr Flow</a>, <a href="https://superwhisper.com">Superwhisper</a>, and <a href="https://monologue.to">Monologue</a>.
</p>

<p align="center">
  <b>This fork is source-only — <a href="#build-from-source">build it from source</a> to get these changes.</b><br>
  <sub>The prebuilt <a href="https://github.com/zachlatta/freeflow/releases/latest/download/FreeFlow.dmg">FreeFlow.dmg</a> is the original upstream app and does <b>not</b> include this fork's local-STT or overlay changes.</sub>
</p>

---

> ### 🍴 This is a fork
> This is a **fork of [zachlatta/freeflow](https://github.com/zachlatta/freeflow)** — all
> credit for FreeFlow goes to the original authors. This fork adds:
> - **Fully local / hybrid speech-to-text** so you don't need a paid API — local
>   `whisper.cpp` (Metal) for transcription + a small router that uses Groq's free
>   `gpt-oss-20b` when online and falls back to a local Ollama model when offline.
>   See **[`local-setup/`](local-setup/)**.
> - **Whisper VAD** so it stops hallucinating "Thank you" on silence.
> - **Overlay tweaks:** a bottom-of-screen recording pill, a center-out reactive
>   waveform that's flat when silent, fade in/out, and no processing spinner.
>
> Upstream: https://github.com/zachlatta/freeflow

<p align="center">
  <img src="Resources/demo.gif" alt="FreeFlow demo" width="600">
</p>

<p align="center">
  <i>Thank you to <a href="https://github.com/marcbodea">@marcbodea</a> for maintaining FreeFlow!</i>
</p>

## Overview

FreeFlow is a free Mac dictation app inspired by [Wispr Flow](https://wisprflow.ai/), [Superwhisper](https://superwhisper.com/), and [Monologue](https://www.monologue.to/). It gives you fast AI transcription, context-aware cleanup, and voice-driven text editing without a monthly subscription.

## Quick Start

1. [Build from source](#build-from-source) (this fork has no prebuilt release).
2. Pick a speech-to-text backend:
   - **Fully local / free** — follow [`local-setup/`](local-setup/) (`whisper.cpp` + Ollama). No API key required.
   - **Hosted** — get a free [Groq](https://console.groq.com/) key (or any OpenAI-compatible provider) and set it in Settings.
3. Hold `Fn` to talk, or tap `Command-Fn` to start and stop dictation, and have whatever you say pasted into the current text field. You can change the shortcut in Settings.

## Features

- **Custom shortcuts:** Customize both hold-to-talk and toggle dictation shortcuts. If your toggle shortcut extends your hold shortcut, you can start in hold mode and press the extra modifier keys to latch into tap mode without stopping the recording.
- **Context-aware cleanup:** FreeFlow can read nearby app context so names, terms, and phrases are spelled correctly when you dictate into email, terminals, docs, and other apps.
- **Custom vocabulary:** Add names, jargon, and project-specific words that FreeFlow should preserve during cleanup.
- **Local or hosted transcription:** Run it fully local with `whisper.cpp` + Ollama (see [`local-setup/`](local-setup/)), use Groq's free tier, or any other OpenAI-compatible provider — all configurable in settings.

## Edit Mode

Edit Mode lets you highlight existing text and transform it with a spoken instruction, like "make this shorter" or "turn this into bullets." Enable it in settings, then use your normal dictation shortcut on selected text, or choose Manual mode to require an extra modifier key.

## Privacy

There is no FreeFlow server, so FreeFlow does not store or retain your data. With the [local setup](local-setup/), **transcription never leaves your machine** (only the cleanup step optionally calls Groq, or runs fully local too). Otherwise, the only information that leaves your computer are API calls to your configured transcription and LLM provider.

## Custom Cleanup

If you'd rather keep cleanup more literal and less context-aware, you can paste this simpler prompt into the custom system prompt setting:

<details>
  <summary>Simple post-processing prompt</summary>

  <pre><code>You are a dictation post-processor. You receive raw speech-to-text output and return clean text ready to be typed into an application.

Your job:
- Remove filler words (um, uh, you know, like) unless they carry meaning.
- Fix spelling, grammar, and punctuation errors.
- When the transcript already contains a word that is a close misspelling of a name or term from the context or custom vocabulary, correct the spelling. Never insert names or terms from context that the speaker did not say.
- Preserve the speaker's intent, tone, and meaning exactly.

Output rules:
- Return ONLY the cleaned transcript text, nothing else. So NEVER output words like "Here is the cleaned transcript text:"
- If the transcription is empty, return exactly: EMPTY
- Do not add words, names, or content that are not in the transcription. The context is only for correcting spelling of words already spoken.
- Do not change the meaning of what was said.

Example:
RAW_TRANSCRIPTION: "hey um so i just wanted to like follow up on the meating from yesterday i think we should definately move the dedline to next friday becuz the desine team still needs more time to finish the mock ups and um yeah let me know if that works for you ok thanks"

Then your response would be ONLY the cleaned up text, so here your response is ONLY:
"Hey, I just wanted to follow up on the meeting from yesterday. I think we should definitely move the deadline to next Friday because the design team still needs more time to finish the mockups. Let me know if that works for you. Thanks."</code></pre>
</details>

## Using a Local Model

This fork ships a complete local/hybrid pipeline in **[`local-setup/`](local-setup/)**: local `whisper.cpp` (Metal) transcription, plus a small router that uses Groq's free `gpt-oss-20b` when online and falls back to a local Ollama model when offline. Follow that README to set it up end to end.

More generally, FreeFlow works with any OpenAI-compatible local or self-hosted provider (Ollama, LM Studio, etc.): in settings, configure the API base URL and model IDs, and set the transcription API URL separately if your STT backend differs from your LLM backend.

> Models are read from `UserDefaults` (`defaults read com.zachlatta.freeflow.dev post_processing_model`, etc.), while the API URLs and keys live in the app's `.settings` file — keep that in mind if you're scripting the config.

Local models are often slower than hosted providers, especially on cold start, long recordings, or busy hardware.

<details>
  <summary>Configure longer timeouts for local models</summary>

  FreeFlow keeps the default network timeout at 20 seconds, but you can extend it with macOS defaults:

```bash
defaults write com.zachlatta.freeflow transcription_timeout_seconds -float 120
defaults write com.zachlatta.freeflow post_processing_timeout_seconds -float 120
defaults write com.zachlatta.freeflow context_request_timeout_seconds -float 120
```

The timeout keys are:

- `transcription_timeout_seconds`: audio transcription requests
- `post_processing_timeout_seconds`: transcript cleanup and edit mode requests
- `context_request_timeout_seconds`: nearby app context requests

Only positive values are used. Remove a custom timeout to return to the 20-second default:

```bash
defaults delete com.zachlatta.freeflow transcription_timeout_seconds
defaults delete com.zachlatta.freeflow post_processing_timeout_seconds
defaults delete com.zachlatta.freeflow context_request_timeout_seconds
```

</details>

## Build from source

Requires the Xcode command-line tools (`swiftc`, `make`).

```bash
git clone https://github.com/linustalacko/freeflow.git
cd freeflow
make ARCH="$(uname -m)" CODESIGN_IDENTITY=-   # ad-hoc signed dev build
open "build/FreeFlow Dev.app"
```

This produces `build/FreeFlow Dev.app`. On first launch, grant **Microphone**, **Accessibility**, and **Input Monitoring** (the global hotkey needs them).

> **Tip:** ad-hoc signing (`CODESIGN_IDENTITY=-`) makes macOS re-prompt for Accessibility/Input-Monitoring permission on *every* rebuild. Sign with a stable identity to keep the grant across rebuilds:
> ```bash
> make ARCH="$(uname -m)" CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
> ```

## License

Licensed under the MIT license.
