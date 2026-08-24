<p align="center">
  <img src="Resources/AppIcon-Source.png" width="128" height="128" alt="FreeFlow icon">
</p>

<h1 align="center">FreeFlow — local-first fork</h1>

<p align="center">
  A Mac dictation app that runs the whole hot path on-device: streaming
  speech-to-text, deterministic cleanup, and a local model as the fallback —
  with hosted providers as a safety net rather than a dependency.
</p>

<p align="center">
  <b>Source-only — <a href="#build-from-source">build it from source</a>.</b><br>
  <sub>The upstream <a href="https://github.com/zachlatta/freeflow">prebuilt DMG</a> does <b>not</b> include any of this.</sub>
</p>

---

> ### 🍴 This is a fork
> Forked from **[zachlatta/freeflow](https://github.com/zachlatta/freeflow)** — all credit
> for FreeFlow itself goes to the original authors. This README documents what
> *this fork* adds and how the local pipeline on this machine is wired.
> Upstream is merged in regularly, so everything upstream ships is here too.

## What this fork adds

**A local pipeline that is faster than the hosted one.** Streaming Parakeet-TDT
for speech-to-text, a fine-tuned Qwen served by `mlx_lm` for cleanup, and a
router that keeps the hosted Groq chain as a fallback instead of a requirement.
Set up in **[`local-setup/`](local-setup/)**; the design rationale is in
[`local-setup/HOW_IT_WORKS.md`](local-setup/HOW_IT_WORKS.md).

**Most dictations skip the model entirely.** A deterministic cleanup layer
handles what doesn't need judgement — fillers, dictated punctuation, spoken
lists, paragraph breaks, capitalisation — in well under a millisecond, and bails
to the model for anything that needs interpreting. It runs both in the app
([`TranscriptFastPath.swift`](Sources/TranscriptFastPath.swift)) and in the
router ([`detclean.py`](local-setup/detclean.py)), pinned together by a parity
test so the two can't drift.

**The app knows where the text is going.** The frontmost app (or the tab title,
inside a browser) selects a destination profile that decides bullet style, email
layout, whether a closing period belongs, and whether the text is prose at all.

**It learns the words you keep fixing.** Corrections you type over pasted text
are captured passively and mined for repeated single-word fixes, which become
custom vocabulary automatically.

**Speculative cleanup while you talk.** The STT server posts settled sentences
to the router mid-dictation; at key-up the router substitutes the already-cleaned
prefix and only generates the tail.

## The pipeline

```
hotkey down ──► audio ──► Parakeet-TDT STT server  :8082   (streaming, ~0.1s tail at key-up)
                             │
                             ├─ settled sentences ──► router /v1/precache  (cleaned early)
                             │
hotkey up ───► final transcript
                             │
                   ┌─────────┴──────────┐
                   │  deterministic?    │  TranscriptFastPath — ~0ms, no model
                   └─────────┬──────────┘
                             │ needs judgement
                             ▼
                     router :11435 ──► local mlx_lm Qwen :8081   (FORCE_LOCAL=1)
                                   └─► Groq chain                (rate-limit aware fallback)
                             │
                             ▼
                    paste at cursor (Cmd-V, original clipboard restored)
```

Services run under `launchd` (`com.freeflow.stt`, `com.freeflow.router`,
`com.freeflow.localmodel`). The repo is the source of truth — deploy with:

```bash
cp local-setup/router.py local-setup/detclean.py ~/.freeflow-stt/ && launchctl kickstart -k gui/$UID/com.freeflow.router
```

## Destination profiles

Where the text lands changes what correct output looks like, so the frontmost
app selects a profile ([`DictationProfile.swift`](Sources/DictationProfile.swift)).
A browser resolves by tab title, so Gmail behaves like a mail client and GitHub
like a document. An unrecognised app keeps the original behaviour exactly.

| Destination | Behaviour |
|---|---|
| **Email** — Mail, Superhuman, Outlook, Gmail | Spoken greeting becomes a salutation line + blank line; `•` bullets, since mail clients render `-` literally |
| **Chat** — Slack, Discord, Messages, Telegram | Casual; markdown bullets; no closing period on a single short line; never adds a greeting or sign-off |
| **Code** — Xcode, VS Code, Cursor, Zed | Identifiers, paths, and casing preserved; no invented sentence punctuation |
| **Terminal** — Terminal, iTerm, Ghostty, Warp | Treated as a shell command: no sentence-casing, no trailing period |
| **Document** — Notion, Obsidian, Notes, Word | Full sentences, normal prose punctuation |
| **Search** — Raycast, Alfred, Spotlight | Query only, no punctuation |

The profile also skips work that can't help: terminals, editors, and launchers
take no screenshot and make no vision call, because an activity summary can't
tell you anything about a shell command.

## Dictating formatting

Spoken formatting is resolved deterministically
([`SpokenFormatting.swift`](Sources/SpokenFormatting.swift)) — instantly, and
without the model's guess at layout:

| Say | Get |
|---|---|
| "bullet point ship the release bullet point update the docs" | `- Ship the release`<br>`- Update the docs` |
| "here's the plan bullet point fix the bug bullet point ship it" | `Here's the plan:`<br>`- Fix the bug`<br>`- Ship it` |
| "numbered list write the spec, get review, merge" | `1. Write the spec`<br>`2. Get review`<br>`3. Merge` |
| "thanks for the note **new paragraph** I'll look tomorrow" | paragraph break |
| "he said **quote** this is fine **unquote**" | he said “this is fine” |

Ambiguity is treated as a reason to defer, not to guess: "we're opening **a new
line** of business", "I need **a quote** for the laptop", and "add **a bullet**
about the rollback plan" are all recognised as ordinary prose. Only genuinely
unclear readings go to the model.

> Measured on this machine: all eight formatting cases above previously went to
> the model (median 777ms round trip); they now resolve in under 1ms. On lists
> the deterministic output matches the model byte for byte — and on "new
> paragraph" it is *more* correct, because the model writes "New paragraph." out
> as literal text.

## Learned vocabulary

`InlineEditCaptureService` watches the field it just pasted into and records what
you changed. [`VocabularyLearner.swift`](Sources/VocabularyLearner.swift) mines
those for repeated single-word fixes — the same substitution twice, both sides
alphabetic, close enough to be the same intended word, not a common word — and
adds them to custom vocabulary. That biases the STT prompt and the cleanup model,
so the word stops coming out wrong.

Fixing "Aisha" → "Aysha" twice teaches it the spelling. Changing "Thursday" →
"Wednesday" twice does not, because that's a rewrite, not a misrecognition.

```bash
defaults write com.zachlatta.freeflow.dev vocabulary_auto_learn_enabled -bool false   # opt out
```

## Hidden settings

All under `com.zachlatta.freeflow.dev` for dev builds (`com.zachlatta.freeflow`
for release). Only positive values are used; `defaults delete` restores the default.

| Key | Default | What it does |
|---|---|---|
| `clean_transcript_fast_path_max_words` | `60` | Longest transcript the deterministic path will handle; `0` disables it |
| `context_wait_at_stop_ms` | `200` | How long key-up may wait for the in-flight context task; `0` never waits |
| `context_enrichment_all_apps` | `false` | Restore screenshots + vision calls for *every* app, including terminals |
| `vocabulary_auto_learn_enabled` | `true` | Learn vocabulary from repeated corrections |
| `transcription_timeout_seconds` | `20` | Audio transcription requests |
| `post_processing_timeout_seconds` | `20` | Cleanup and edit-mode requests |
| `context_request_timeout_seconds` | `20` | App-context requests |

## Build from source

Requires the Xcode command-line tools (`swiftc`, `make`).

```bash
git clone https://github.com/linustalacko/freeflow.git
cd freeflow
make ARCH="$(uname -m)" CODESIGN_IDENTITY=-   # ad-hoc signed dev build
open "build/FreeFlow Dev.app"
```

On first launch grant **Microphone**, **Accessibility**, and **Input Monitoring** —
the global hotkey needs all three.

> Ad-hoc signing makes macOS re-prompt for Accessibility on *every* rebuild. Sign
> with a stable identity to keep the grant:
> ```bash
> make ARCH="$(uname -m)" CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
> ```

## Tests and benchmarks

```bash
make check                                    # typecheck + Swift tests + plist/YAML validation
python3 local-setup/test_router.py            # router: rate limits, fallback, precache
python3 local-setup/test_detclean.py          # deterministic cleanup, pinned to the Swift output
~/.freeflow-ft/venv/bin/python local-setup/test_stt_server.py   # STT protocol (needs numpy)
```

```bash
~/.freeflow-ft/venv/bin/python local-setup/bench_e2e.py --files 8   # key-up → cleaned, end to end
~/.freeflow-ft/venv/bin/python local-setup/eval_models.py           # WER + latency per backend
```

`bench_e2e.py --target chat|email|terminal|…` measures a specific destination.
Benchmark on an idle machine — GPU contention from anything else on the box
moves these numbers far more than most code changes do.

## Debugging

| Where | What's in it |
|---|---|
| `~/.freeflow-stt/router.log` | Per-request route, deadline, latency, precache hits |
| `~/.freeflow-stt/stt-server.log` | Segment cuts, commit timing, partials posted |
| `curl 127.0.0.1:11435/v1/status` | What the router believes about each Groq bucket and the local model |
| Run Log (in-app) | Raw vs cleaned transcript, prompts, context, screenshot, corrections |
| `~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite` | Last 20 runs |

## Privacy

There is no FreeFlow server. With the local setup, audio and transcripts never
leave the machine — the hosted chain is only reached when the local model can't
meet the deadline, and `FORCE_LOCAL=1` keeps it entirely on-device.

## Edit Mode

Select text, hold the dictation shortcut, and say what to change ("make this
shorter", "turn this into bullets"). Enable it in Settings, or use Manual mode to
require an extra modifier.

## License

MIT.
