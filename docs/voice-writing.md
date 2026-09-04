# Voice writing

FreeFlow has two explicit writing actions, available from the menu bar. Assign
shortcuts under Settings → Dictation Shortcuts. New shortcuts start disabled so
an upgrade does not claim existing system or application shortcuts. Both actions
use tap-to-start / tap-to-finish recording; the overlay checkmark also finishes,
and Escape cancels.

## Edit the last dictation

Dictate normally, leave focus in the same field, invoke **Edit Last Dictation**,
and say, for example, “make it shorter” or “change Thursday to Friday.” FreeFlow
replaces only its last insertion. Invoke the action again and say “undo that” to
restore the preceding version. Undo stores one preceding version in memory.

Edits require the original text to be unchanged. The app checks the field,
application, document URL when available, complete text, and cursor/selection
before replacing anything. A changed destination stops insertion; a successfully
generated result remains available from **Paste Again**. Unsupported fields fail
without selecting arbitrary matching text or the whole document. Anchors reset
when FreeFlow quits. Browser space-to-NBSP conversion is accepted only because it
preserves UTF-16 offsets; other text changes are rejected.

## Draft a Gmail reply

1. Open a conversation in Gmail in the browser, expand the message to answer,
   click **Reply**, and put the cursor inside the reply body.
2. Invoke **Draft Gmail Reply** and say the intended reply, for example:
   “Say Friday afternoon works and ask whether two o'clock is convenient.”
3. Finish recording. Review the inserted reply and send it yourself.

The action inserts at the captured caret, preserving existing draft text and
signatures. It does not click Reply automatically, edit recipients, or send mail.
It requires an accessible multiline composer inside Gmail's main conversation
landmark, with a conversation URL. Inbox views, other websites, selected draft
text, unreadable fields, and unavailable or oversized thread captures fail safely.
The current implementation reads expanded text in that conversation, up to
20,000 UTF-16 units / 1,200 nodes / a 500 ms traversal budget. Collapsed messages
and attachments are not read. Browser variations still require manual testing.

## Data and latency

Writing actions use the existing streaming transcription service and configured
writing-model endpoint/fallbacks. They send the spoken instruction plus the last
insertion or captured Gmail conversation only when invoked. Thread and instruction
fields are encoded as JSON and treated as untrusted source data. These requests
have no action tools. Empty, failed, cancelled, or truncated model responses do
not fall back to pasting the spoken instruction.

Writing recordings are temporary and cleaned up. Writing prompts, thread
captures, and writing runs are not appended to the Run Log. Generated rewrites
are excluded from passive manual-correction capture. The latest result and one
undo version stay in memory. Clipboard handling honors the existing preservation
and clipboard-history preferences. Provider retention policies remain those of
the configured services.

These actions require existing Accessibility and microphone access. They do not
request new permissions or use screenshots. Chromium/Electron may be asked to
expose their Accessibility tree, as the existing inline-correction feature does.
Ordinary dictation keeps its streaming and cleanup paths; the added work is a
bounded local Accessibility insertion snapshot, with no additional model call.

## Verification

- `make check`: full type-check, deterministic tests, plist/script/YAML validation.
- `git diff --check`.
- `bash local-setup/voice-writing-smoke/run.sh`: opens an invented conversation in
  a native WebKit window and checks context scoping, exact selection, Cmd-V,
  follow-up editing, undo, and clipboard restoration. It reads only its own
  fixture process, uses no microphone or AI provider, and requests no permissions.
  The clipboard is held only in memory and restored unless the user changes it.
  Temporary binaries are removed on exit. Run it while not typing or copying.

On 2026-09-04, deterministic tests and the native smoke test passed on arm64
macOS. The synthetic native thread capture plus target validation measured 2 ms;
this is not an end-to-end dictation or model-latency benchmark.

### Before merge: manual checks still pending

Use invented email content in the actual Gmail browser. Do not attach real
messages, screenshots, recordings, prompts, or clipboard contents to the PR.

- Configure both shortcuts; tap, speak, tap again; also finish with the overlay
  and cancel with Escape. Check ordinary hold/toggle dictation still works.
- Dictate a new sentence, shorten it, change a date, and undo in the same field.
- Draft a Gmail reply from a spoken intention; verify thread relevance, language,
  tone, signature preservation, and that the message remains unsent.
- While generation is running, change focus, navigate, type, or move the caret:
  insertion must stop. Recover the completed result using Paste Again.
- Check multiline replies and emoji in the actual browser's composer. Verify
  clipboard restoration and that writing runs/rewrites are absent from the Run
  Log and manual-correction capture.
- Check a model failure/cancellation: existing text stays intact and the spoken
  instruction is not pasted. Missing permissions must produce an explanation
  without requesting Screen Recording.

Prompt structure was checked against the official
[OpenAI prompting guidance](https://developers.openai.com/api/docs/guides/prompt-engineering#message-formatting-with-markdown-and-xml).
Prompt and transport tests use synthetic fixtures and a mocked transport; they
establish request behavior, not model quality on real email.
