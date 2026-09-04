# Streaming accuracy and latency validation

The September 2026 changes address sample drift, premature finalization,
unbounded audio copying, ignored short-dictation settings, and timeout retries.
They preserve the existing models, provider configuration and native Swift build.

## Reproduce without private recordings

Use the Python environment installed by the local setup (numpy, scipy, aiohttp):

```sh
make check
git diff --check
python3 local-setup/test_router.py
python3 local-setup/test_detclean.py
~/.freeflow-ft/venv/bin/python local-setup/test_stt_server.py
~/.freeflow-ft/venv/bin/python local-setup/test_realtime_client.py
~/.freeflow-ft/venv/bin/python local-setup/bench_pipeline.py --baseline-ref ecddd5e --speech
```

The benchmark synthesizes three invented utterances with macOS `say`, streams
them at real time in 1,024-sample packets, and uses one cached Parakeet v2 model
for both implementations. It disables precache and model downloads, uses only
temporary loopback listeners, and removes generated files on exit. It never
opens the app's audio directory, history database, clipboard or credentials.
Run on an idle machine and repeat before drawing statistical conclusions.

## Observed on September 4, 2026

Comparison against `ecddd5e` on the development Mac:

| Diagnostic | Before | After |
| --- | ---: | ---: |
| Three-second signal, 1,024-sample packets: sample-count error | +23 | 0 |
| Same signal: RMS difference from whole-clip resampling | 0.1123 | 0 |
| Ten-minute buffer simulation: cumulative append time | 11,421 ms | 6.08 ms |
| Ten-minute buffer simulation: retained allocation | 38.4 MB | 1.93 MB |
| 2.87-second synthetic speech: commit to final | 74.9 ms | 85.4 ms |
| 11.86-second synthetic speech: commit to final | 215.6 ms | 185.1 ms |
| 33.05-second synthetic speech: commit to final | 225.6 ms | 192.6 ms |

The buffering test simulates retaining fifteen seconds after settling; its
timing excludes inference, resampling, snapshots, microphone and paste. The
speech transcripts were identical between versions, including numbers rendered
as digits. This single run is diagnostic evidence, not an accuracy benchmark or
a statistically established latency improvement. It does not measure key-up to
paste or microphone startup and cannot establish a product percentile ranking.

## Contracts covered by regression tests

- Arbitrary packet sizes at 8, 24, 44.1 and 48 kHz match whole-clip resampling,
  including the final samples. The resampler holds sufficient filter context
  and advances on an aligned sample grid.
- Long sessions preserve all synthetic words after discarding settled audio;
  an in-flight decoding snapshot survives storage compaction and growth.
- Settled chunks use distinct IDs. The client waits for the acknowledged final
  item, including an empty tail, and does not accept a missing transcript.
- Local WebSocket peers exercise late partials, silent/malformed final events,
  close, cancellation and duplicate commit. Silent peers reach file fallback
  after an eight-second production budget; tests use a shorter injected budget.
- Short utterances honor output language, custom instructions, vocabulary and
  disabling deterministic cleanup. Simple phrases still avoid the model.
- Swift and Python cleanup preserve measurement units (`mm`), acronyms (`ER`),
  and meaningful acknowledgements (`mhm`, `hmm`) instead of deleting them as filler.
- Hosted socket contention is bounded, and timeouts are not retried with a fresh
  full budget on Python 3.9. Stale-connection retries share the remaining budget.

Final-item correlation follows the official
[Realtime transcription event contract](https://developers.openai.com/api/docs/guides/realtime-transcription).
The local server and Swift client should be updated together; older clients
still lack final-item correlation and a final-result deadline.

## Manual gate before merge

Pending: exercise the signed app with the existing microphone and permissions.
Do not launch an ad-hoc replacement that could reset system permissions.

1. Dictate a short sentence, release the shortcut while speaking its final word,
   and verify the whole sentence is pasted once.
2. Dictate continuously for at least a minute, including repeated words and
   deliberate corrections; verify boundary words, numbers and the ending.
3. Check a one-word translation, custom instruction, vocabulary capitalization,
   and spoken punctuation. Restore the original settings afterwards.
4. Repeat in a chat field and document; verify cancellation and Paste Again.
5. Verify the running STT/router source hashes and test an idle-to-first-use
   dictation. Record timings only, without exporting user content.

The scripted checks and an ad-hoc compile-and-bundle check pass. The signed app,
physical microphone, Accessibility, clipboard and paste are not verified by
those checks. No permissions, versions, release workflows or persistent data
formats change. Settled transcript snippets are removed from the STT log; no
new content logging or transmission is added.
