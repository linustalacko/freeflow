# Wispr Flow and the sub-50 ms target — September 5, 2026

Wispr Flow is a cloud dictation product that turns speech into edited writing
inside other apps. Its advertised **4× speed advantage is speaking versus
typing**, not a claim about the delay after recording stops. It also removes
fillers, handles spoken corrections, and adapts formatting to the destination.
[Product site](https://wisprflow.ai/)

## What is publicly known

| Evidence | What it supports | What it does not establish |
|---|---|---|
| Baseten customer case study: under 700 ms p99 for ASR plus Llama cleanup | A published provider claim about the combined inference pipeline | A current independent measurement from key release to pasted text, or raw ASR latency alone |
| Wispr July 9, 2026 changelog: latency down 30% since the start of the year | Wispr continues to reduce latency | An absolute latency distribution or comparable test corpus |
| Wispr CTO describes routing among speech engines and specialized formatters | Accuracy comes from a system of models and context | One disclosed English model, absolute WER, or a reproducible competitor ranking |
| Public REST and WebSocket API documentation | A potential route to a controlled direct benchmark | Access credentials, identical consumer-app behavior, or an unedited ASR output |

Baseten describes dedicated deployments on AWS, fine-tuned Llama cleanup models,
TensorRT-LLM, and its Chains orchestration. It reports that the cleanup model
generates over 100 tokens within 250 ms and the combined pipeline stays below
700 ms at p99. These are supplier-reported results without the corpus, client
timing implementation, or full latency distribution needed to reproduce them.
The 700 ms value is an upper bound; dividing it by a FreeFlow measurement does
not prove a speedup ratio.
[Baseten case study](https://www.baseten.co/resources/customers/wispr-flow/)

Wispr's CTO describes choosing different ASR engines for different languages,
combining model strengths, and training formatting models with correction
feedback. Its language-routing improvements reportedly more than halved its
own error rates, but that article does not publish an absolute, independently
reproducible WER comparison.
[Wispr language research](https://wisprflow.ai/research/supporting-languages)

In a January 2026 interview, the CTO discusses optimizing the worst user
experiences, locating network delays between models, and predicting likely
output before validating it. He describes a 250 ms cleanup-model budget for
50–100 words. These architectural lessons support measuring tail latency and
moving work into recording time, while checking that speculative output is
still correct before returning it.
[Software Engineering Radio interview](https://se-radio.net/?p=9309)

The service's documentation says transcription requires cloud processing and
describes optional context from the destination app and nearby text. A direct
benchmark should omit those fields initially and evaluate dictionary/context
assistance in a separate condition.
[Data controls](https://wisprflow.ai/data-controls)

The API recommends WebSocket streaming for latency-sensitive use. It accepts
16 kHz mono audio, incremental packets and an explicit commit, then returns
formatted text. A benchmark must time receipt of the final response after
the last audio packet, include authentication/network failures separately, and
avoid treating an API example's `total_time` as a measurement. API access was
not available during this research; Wispr was not found in the applications
directories checked. The web demo requests microphone dictation and does not
expose a documented audio-file benchmark flow.
[API introduction](https://api-docs.wisprflow.ai/introduction),
[WebSocket protocol](https://api-docs.wisprflow.ai/websocket_api),
[web demo](https://wisprflow.ai/demo)

## What “an order of magnitude better” must mean

For speed, compare **the same timing boundary and percentile** on matched clips.
Raw transcription's local final-response time and a competitor's edited,
pasted output are different endpoints. The desired product metric is key
release to correct text inserted; the current replay benchmark measures only
the raw transcription portion. Report p50, p95, p99, failure rate, cold starts,
and performance after idle. A few dozen samples cannot establish reliable p99.

For accuracy, define 10× better as **90% fewer word errors**, not ten times the
percentage of correct words. For example, a measured 3% competitor WER would
require at most 0.3% FreeFlow WER on the same held-out corpus. That is a target,
not an observed result. Use paired confidence intervals, error categories,
silence hallucinations, and failure rates; do not hide regressions behind a
single average. Wispr's deliberate filler removal and formatting need a second
evaluation against an intended final-text reference, separate from raw WER.

Public read speech is a first accuracy gate, followed by spontaneous dictation,
accents, background noise, names, numbers, technical vocabulary, corrections,
and very short utterances. Keep model selection data separate from final
validation data. No current results establish a 10× accuracy advantage.

## Engineering implications

The current FreeFlow Parakeet server repeatedly decodes overlapping windows and
performs another decode at commit. The library also computes token confidence
that this server does not consume. The experiments test removing unused work,
combining GPU synchronization, compilation, precision, quantization, shorter
windows, and native streaming caches without changing the installed service.

The native Parakeet cache is not guaranteed equivalent to full-context decoding:
the library documents that its default depth preserves exact computation only
at the first encoder layer. Its low-latency settings therefore require an
accuracy gate, not just a faster timer.
[Parakeet MLX streaming documentation](https://github.com/senstella/parakeet-mlx#streaming-transcription)

A model trained for streaming is another avenue. NVIDIA's Nemotron streaming
model offers configurations with 80 ms frames/chunks and retained state. The
80 ms chunk size is algorithmic granularity, not evidence of under-50 ms final
latency on this Mac. Evaluating another runtime/model is separate work from
changing the current model's cache settings.
[NVIDIA model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

FreeFlow's Swift stop path also waits for audio-session teardown and file
finalization, then copies the recording before requesting a final transcript.
Moving storage work out of that path may reduce product latency, but requires
capturing every final audio callback and retaining fallback behavior. No
microphone or paste timing was measured in these model experiments.

See the [local experiment results](2026-09-05-sub50-experiments.md) for measured
outcomes and rejected candidates. The installed app, permissions, provider
settings, and recording history were not modified.
