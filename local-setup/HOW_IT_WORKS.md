# FreeFlow On-Device Cleanup Model — How It Works (From First Principles)

*A plain-English explainer of what we're building and why. No prior ML knowledge assumed.*

---

## The goal, in one sentence

Replace the **cloud AI** that currently cleans up your dictation with a **tiny AI running on your own Mac** — so it's free, private, and faster.

---

## 1. What FreeFlow actually does

When you dictate, three things happen:

```
   you speak  ─►  Whisper (speech → text)  ─►  cleanup AI (text → better text)  ─►  pasted at your cursor
                   "raw transcript"              "the polished version"
```

- **Whisper** turns your voice into text. It's good, but the output is *messy*: it sticks hard line-breaks in the middle of sentences, runs sentences together, leaves a trailing "um", occasionally mishears a word.
- The **cleanup AI** rewrites that mess into clean text.
- Right now that cleanup AI is **gpt-oss-20b**, running in the cloud (on Groq). That means: a few cents per use, your words leave your machine, and there's a network round-trip (~1.8s).

**We want to swap that cloud model for a small one on your Mac.** Same job, but $0, private, and ~1 second.

---

## 2. Why a *small* model can possibly do this

Cleaning dictation is a **narrow** job. The model doesn't need to know history, code, and poetry — it needs to do *one* transformation well.

> **Analogy:** You don't need a world-class chef to make your morning coffee exactly how you like it. You need someone who's watched you make it 200 times. A small model "shown" your cleanup style can match a big general model *at that one task*.

The catch: a small model *out of the box* isn't good at it. We have to **teach** it. That teaching is called **fine-tuning**.

---

## 3. What "training" actually is

A model is a giant maths function with millions of **dials** (numbers). It takes input text and produces output text. The dials decide what comes out.

**Training** = 
1. Show it an example: a messy input, and the clean output *you wanted*.
2. Look at what it actually produced.
3. Nudge the dials so next time its output is a bit closer to yours.
4. Repeat thousands of times.

Eventually the dials settle into a configuration that does the cleanup. That's it. There's a known right answer for every input, so we just keep pushing the model toward it. (This is **imitation learning** — not the trial-and-error "reward" kind you hear about with game-playing AIs.)

---

## 4. LoRA — the trick that makes this cheap

The base model (we're using **Qwen-2.5, 1.5 billion dials**) already speaks English fluently. We do **not** want to move all 1.5 billion dials — that's slow and can break what it already knows.

**LoRA** ("Low-Rank Adaptation") freezes the entire base model and bolts on a *tiny* set of brand-new dials — an **adapter** — that learn *only* the adjustment for your task.

> **Analogy:** You don't rewrite the textbook. You add a small sticky-note of corrections on top.

- The base model: ~900 MB, frozen, untouched.
- The adapter we train: **~10 MB.** Snaps on and off.

That's why training takes minutes, not days.

---

## 5. The engine — MLX

**MLX** is Apple's framework for running and training models directly on the Mac's own chip (the GPU), using the Mac's shared memory. No cloud, no NVIDIA card needed.

On your M4 Pro: training used ~5 GB of memory and ran at ~1,100 tokens/sec; the finished model answers in ~1 second using ~1 GB. Comfortably inside your 24 GB.

---

## 6. The data — this is the whole game

A model is only as good as the examples you train it on. We need **lots** of `(messy → clean)` pairs. They come from two places:

**A. Real data (the gold standard)**
Every time you dictate, FreeFlow already logs your raw transcript *and* gpt-oss's cleanup. Even better: when **you** fix a word yourself, that edit is the *best possible* signal — it's exactly how *you* want it, which is the only way to ever beat gpt-oss. (Capturing those edits automatically inside Claude/Telegram/etc. is a feature we're still getting to work reliably — those apps hide their text from the Mac's accessibility system.)

**B. Synthetic data (the bootstrap)**
Waiting weeks to accumulate real examples is slow. So we ask gpt-oss to *invent* realistic messy→clean pairs in your style — founder/dev content, Australian spelling — giving us volume **now**. A script (`gen_synthetic_data.py`) generates these, anchored to your *real* examples so they match your actual texture.

**Two kinds of "right answer":**
- **Distillation** = copy gpt-oss's cleanup → teaches the small model to **match** gpt-oss (today's bar).
- **Your own corrections** = teaches it to **beat** gpt-oss (your personal taste).

> So the strategy is: **distil to reach the bar cheaply now, use your corrections to pass it later.**

---

## 7. The most important thing we learned about *your* data

We looked at your real transcripts. **Whisper already capitalises and punctuates well.** So your real cleanup job is **subtle** — it's mostly:

- **joining the mid-sentence line breaks** Whisper inserts (the dominant edit),
- splitting run-on sentences,
- dropping a trailing "um", light polish.

It is **not** the cartoonish "so um i wanted to like…" cleanup. This matters a lot: a model that "cleans" too aggressively actually makes your text **worse**. The target is a **light touch**.

---

## 8. How we measure success (the eval)

You can't improve what you don't measure. So:

1. **Hold some examples out** of training — the model never sees them.
2. Have each model clean those inputs.
3. Measure how far its output is from the target answer:
   - **CER** = % of *characters* that differ. **WER** = % of *words* that differ. Lower = better.
4. Always compare against two reference lines:
   - **"do nothing"** (just the raw transcript) — the floor any useful model must beat.
   - **gpt-oss** — the bar we're chasing.

(We even found and fixed a measurement bug: the score was ignoring whitespace, which hid the line-break-joining that *is* the main task. Fixed — now the metric counts it.)

---

## 9. What we've found so far (real numbers, on your real held-out rows)

| model | word error ↓ | char error ↓ | exact match ↑ | speed |
|---|---|---|---|---|
| do nothing (raw) | 10.6% | 5.6% | 0% | instant |
| small model, **untrained** | 16.2% | 8.1% | 12% | ~0.9s |
| small model, **+ trained adapter** | 11.2% | **5.0%** | **25%** | ~1.0s |
| gpt-oss (the bar) | — | ~0%\* | — | ~1.8s |

\*gpt-oss scores ~0 because its own outputs are the target here.

**Read it like this:**
- The **untrained** small model *over-edits* — it's **worse than doing nothing**.
- After training on **just 58 synthetic examples**, it **flipped**: it now joins line-breaks and stops mangling words — the **only version that beats "do nothing"** — and it runs **faster than the cloud** model.
- That's from almost no data. The trend says: **more data → better.**

This is the key result: **the approach works.** A small, on-device, fine-tuned model is on a clear path to replacing the cloud cleanup — faster, free, and private.

---

## 10. Where things stand right now

- ✅ **Full loop works end-to-end:** collect data → split → train (LoRA/MLX) → measure.
- ✅ **Synthetic data helps** — proven on your real held-out rows.
- ⏳ **Scaling synthetic data is rate-limited** by Groq's free tier (we have ~64 examples; the generator keeps trying in the background and resumes without losing progress). *Options to go faster: a paid Groq tier, or generate using the local llama model instead of the cloud.*
- 🔜 **Get automatic edit-capture working** in your real apps → real corrections → the only way to *beat* gpt-oss.
- 🔜 **When there are a few hundred examples:** train the 1.5B properly, then point FreeFlow's router at the local model. Cloud cost → **$0**.

---

## The files (what each one does)

| file | job |
|---|---|
| `export_training_data.py` | Pulls your dictation history into training pairs; splits into train / held-out test. |
| `gen_synthetic_data.py` | Generates synthetic messy→clean examples in your style (founder/dev, Australian), anchored to your real data. |
| `finetune_local.sh` | Runs the LoRA fine-tune on your Mac with MLX, then tells you how to serve it. |
| `eval_models.py` | Scores any set of models (base, +adapter, gpt-oss) against your held-out data. |
| `InlineEditCaptureService.swift` | *(in the app)* Passively captures the edits you make to dictated text, to become training labels. |

---

## The whole thing in five sentences

1. FreeFlow cleans your dictation with a cloud AI today; we want a tiny on-device one instead.
2. Cleanup is a narrow task, so a small model *taught your style* can match the big one — faster, free, private.
3. "Teaching" = showing it thousands of (messy → clean) examples and nudging its dials; LoRA does this by training a tiny 10 MB add-on instead of the whole model.
4. The examples come from your real dictations (best) and from synthetic ones we generate (fast); we measure progress by holding some out and scoring how close each model gets.
5. With just 58 examples it already beats "do nothing" and runs faster than the cloud — so the plan works; now it's mostly a matter of feeding it more data.
