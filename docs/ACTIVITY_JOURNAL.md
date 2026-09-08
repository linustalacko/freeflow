# Optional Activity Journal

FreeFlow → Activity Journal → **Start Journal** starts a session.
**Stop Journal** stops capture and cancels current work. **Open Journal…** shows
saved summaries without starting capture. Every app launch starts stopped,
including launches with `--activity-journal`. After Start, switching from the
journal to a work app triggers the first check after a one-second settling delay.
The footer shows the last and next check times and the current processing or
skip reason; copying summaries does not replace that status.
The normal FreeFlow menu, branding, dictation shortcuts, settings and provider
choices remain intact. This feature does not start at login.

## Optional setup

From the checkout, run:

```bash
bash local-setup/install-journal-model.sh
```

This installs Ollama if needed and downloads only Qwen2.5 0.5B (about 400 MB of
weights). The installer stops and removes its older journal-specific LaunchAgent,
then uses a temporary local server for the download and shuts it down. It does
not remove other models or change dictation's independent services.

At runtime, FreeFlow starts an owned, loopback-only Ollama process only after
finding new readable screen text. It ends that process after the request,
including error/cancellation paths. Missing Ollama/model files produce a local
error; there is no automatic download or remote fallback.

## Resource limits

- Checks every 180 seconds by default; configurable from 60–900 seconds.
  Longer intervals use less energy and can miss short tasks.
- Skips sleep, lock, excluded/private windows, and busy processing before
  taking another screenshot. After the initial check, inactivity is measured
  against the whole check interval plus timer tolerance (195 seconds at the
  default interval), with a minimum two-minute window. This avoids losing
  activity between checks. There is no image queue. App switches trigger an
  extra check only while waiting for the first work app after Start.
- Captures only the active app's window, at most 1920 pixels on its longest
  edge. Screenshots are passed straight to Apple's fast OCR; no PNG encoding,
  compression or decoding round-trip.
- Compares a 64×64 thumbnail hash before OCR and a normalized text hash before
  inference. Only two hashes survive between successful samples; Stop clears them.
- OCR input is capped at 2,400 characters; the redacted model evidence is capped
  at 1,200 UTF-8 bytes. The model has a 2,048-token context, 160 output tokens,
  two CPU threads, one inference slot, a 128-token batch and no thinking.
- `keep_alive: 0` unloads the model immediately; the owned server is also terminated.
  See [Ollama's API](https://docs.ollama.com/api/chat) and
  [context configuration](https://docs.ollama.com/faq).
- Viewing/copying a day reads only that day's nearby folders (to account for
  timezone changes), rather than decoding the entire archive.

The small model and reduced OCR favor low resource use over detailed screen
understanding. Small text, images without text, and short-lived tasks may be
missed. Summaries describe what was visible; they do not prove actions or hours
worked. The export prompt makes this sampling limit explicit.

## Data and permissions

Only completed model summaries, app names, timestamps and processing metadata
are saved under the existing application-support directory:
`FreeFlow Dev/GitForWorkRaw` for the dev app. Source screenshots and OCR are
memory-only and discarded. Legacy source files from the earlier journal are
removed from its data directory when the first session starts; summaries and
unrelated files are preserved. Directory/file permissions remain 0700/0600.
Memory, swap and framework caches are managed by macOS.

Private-window detection is best effort. Password managers and user-excluded
apps are skipped. Model transport disables cloud use, redirects, proxy settings,
cookies, URL caching and persistent runtime output. No Accessibility enrichment,
selected text, or document URLs are collected by the journal.

Screen Recording permission is used only after Start. Background work never
requests permission; the explicit Allow button opens the macOS permission flow.
Copy all / Copy Today's Summaries copies only completed entries for the selected
local day; an empty day leaves the clipboard untouched. Source data is never
exported. Nothing is automatically sent to another service.

## Validation (2026-09-08)

On the development Apple Silicon Mac:

| Measurement | Original PR | Lightweight journal |
| --- | --- | --- |
| Model memory reported by Ollama | 2,261,694,545 bytes between captures | peak 414,638,406 bytes during synthetic summaries |
| Model/service between summaries | model kept warm | owned service exits; no model loaded |
| Synthetic screenshot OCR | earlier accurate-OCR first use: 29.6 seconds | fast OCR: 0.04–0.07 seconds |
| Five synthetic summaries | not measured as a matching baseline | 0.96–1.44 seconds each, including server startup/shutdown |
| Synthetic test process tree peak RSS | not measured | 650,100,736 bytes |
| Signed FreeFlow RSS after live test and Stop | not measured as a matching baseline | 54,784 KiB |

These are sampled measurements, not hard peak guarantees. Model memory and
process RSS describe different allocations and should not be added together.
The comparison of first-use OCR is not a controlled benchmark. The model-memory
reduction is approximately 82%; most importantly, the model is not resident
between checks.

Manual native-app verification: opened the journal from the menu, observed
Stopped/Start Journal on launch, used the native Start and Stop menu controls,
captured an invented work window through ScreenCaptureKit, and verified a
completed local summary with the expected export/frame topic. The model service
was absent afterward and after Stop. Only status/count metadata and synthetic
content were inspected. No raw source files were retained. Test entries and the
test application were removed afterward.

Follow-up capture fix: some real apps expose a separate 32-pixel-high toolbar
window ahead of the content window. Selecting it yielded no readable text.
Window selection now skips thin and transparent windows, preserving front-to-
back order among usable content windows. A selected private window is still
rejected; it never falls through to a public background window.

The original lightweight timer could miss activity in
the first minute of each interval, and Start from the journal itself skipped
the first sample. The activity window now covers the interval and coalescing;
a one-time app-activation retry starts capture when the user switches to work.
A 1920-point synthetic window with 14-point text lost its body text at the
1280-pixel cap. At 1600 pixels it retained text with recognition errors; the
1920-pixel cap passes body-text checks in light and dark appearances. Fast OCR
of these synthetic images took about 0.01–0.02 seconds after first use. This
raises transient image memory, while keeping the same small model and cadence.

Follow-up manual verification used the signed app at its existing install path
with existing permissions. Start from the journal showed the switch-to-work
prompt; a large invented work window then produced a completed local summary.
A subsequent scheduled check produced another summary without another Start.
The default three-minute timer was observed firing; the successful continuous
capture test used the supported 60-second interval, then restored 180 seconds.
The model service was absent after both successful samples; one between-sample
app reading was 0.0% CPU and 99,376 KiB RSS. The footer showed
check times and fixed skip/processing reasons. The final selector was also
verified on a real app with a 1728×32 toolbar above its 1728×1084 content window:
capture plus fast OCR took about 0.50 seconds and recognized 1,164 characters.
Only dimensions, character counts and duration were inspected. The signed app
then saved a new completed real-work entry; only its completion status and
timestamp were checked, and the model service had exited. Test entries, images and the
fixture app were removed. No real screen content was printed or exported.

`make check` covers stopped-by-default sessions, idle/busy/sleep gates, stale
session rejection, duplicate hashes, Unicode/input/resource bounds, runtime
ownership and cancellation with a harmless subprocess and fake readiness,
toolbar/transparent-window selection and private-window ordering, small body
text through fast OCR, summary-only storage/migration, bounded day
reads, and a private-pasteboard
round-trip. It never calls an AI provider. The existing dictation/provider/
shortcut tests also pass. No new live microphone-to-paste claim is made;
dictation's implementation is unchanged.

The optional live-model smoke uses invented text and generated graphics only:

```bash
journal_scratch="$(mktemp -d)"
swiftc -target "$(uname -m)-apple-macosx13.0" -parse-as-library \
  Sources/ActivityJournalCore.swift Sources/ActivityJournalCapture.swift \
  Sources/RawCaptureStore.swift Sources/JournalPolicy.swift \
  Sources/JournalModelRuntime.swift local-setup/JournalSmoke.swift \
  -o "$journal_scratch/journal-smoke"
"$journal_scratch/journal-smoke" --owned-runtime
rm -rf "$journal_scratch"
```
