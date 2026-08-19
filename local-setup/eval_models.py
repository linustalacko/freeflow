#!/usr/bin/env python3
"""
Evaluate FreeFlow post-processing models on a HELD-OUT test set, so you can see
whether a (small, on-device) model is actually good enough — and whether it's
improving as you collect more corrections.

What it measures: how close each model's cleanup gets to the text YOU kept (your
saved correction = ground truth), using word/char error rate. Lower = closer to
what you'd have written yourself.

  - The "raw" row is the do-nothing floor: how far the Whisper transcript already
    is from your gold text. Any useful model must beat it.
  - gpt-oss (if you point --endpoint at Groq) is the bar to match/beat.
  - Run the same command for 0.5B / 1.5B / 3B (base and +adapter) and ship the
    SMALLEST model whose error rate is acceptable and close to gpt-oss.

Because the test set is a stable hash-based holdout (see export_training_data.py
--split-dir), the numbers are comparable across runs — re-run after collecting
more data to watch the fine-tune hill-climb.

------------------------------------------------------------------------------
SETUP
    pip install mlx-lm            # only needed for --mlx models
    python3 export_training_data.py --split-dir ~/.freeflow-ft

EXAMPLES
    # floor + a base small model + that model with its fine-tuned adapter:
    python3 eval_models.py --test ~/.freeflow-ft/test.jsonl \
        --mlx mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --mlx mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        --mlx "mlx-community/Qwen2.5-1.5B-Instruct-4bit@$HOME/.freeflow-ft/adapters"

    # also score the current production model (Groq gpt-oss) as the bar:
    GROQ_API_KEY=... python3 eval_models.py --test ~/.freeflow-ft/test.jsonl \
        --endpoint https://api.groq.com/openai/v1/chat/completions \
        --endpoint-model openai/gpt-oss-20b --api-key-env GROQ_API_KEY \
        --endpoint-label gpt-oss-20b
------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def _levenshtein(a, b):
    """Edit distance between two sequences (lists or strings)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize(s):
    """Collapse spaces/tabs but PRESERVE newlines — joining Whisper's mid-sentence
    line breaks is the dominant real-world cleanup edit, so it must count. (WER
    still splits on all whitespace, so it stays a word-level metric; CER, which
    runs on this normalized string, becomes sensitive to line-break joining.)"""
    return re.sub(r"[ \t]+", " ", s.strip())


def wer(pred, gold):
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not g:
        return 0.0 if not p else 1.0
    return _levenshtein(p, g) / len(g)


def cer(pred, gold):
    p, g = _normalize(pred), _normalize(gold)
    if not g:
        return 0.0 if not p else 1.0
    return _levenshtein(p, g) / len(g)


_WORD = re.compile(r"[a-z0-9_/'-]+")


def _wordset(s):
    return set(_WORD.findall(s.lower()))


def insertion(pred, raw, gold):
    """Fraction of predicted words that appear in NEITHER the raw transcript nor
    the gold text — i.e. content the model made up (hallucinated names, dates,
    context leaking into the output, prompt examples being parroted)."""
    p = _wordset(pred)
    if not p:
        return 0.0
    allowed = _wordset(raw) | _wordset(gold)
    return len(p - allowed) / len(p)


def leaked(pred, forbidden):
    """True if any forbidden phrase (case-insensitive) shows up in the prediction."""
    low = pred.lower()
    return any(f.lower() in low for f in (forbidden or []))


def exact(pred, gold):
    return _normalize(pred).lower() == _normalize(gold).lower()


# ----------------------------------------------------------------------------
# prompt formats — "app" is byte-for-byte what FreeFlow sends at runtime
# (PostProcessingService.process); "train" is what export_training_data.py /
# gen_synthetic_data.py wrote, i.e. what the local fine-tune saw. Evaluating a
# model in the wrong format is a real production bug, so both are first-class.
# ----------------------------------------------------------------------------
def user_message(fmt, raw, context):
    if fmt == "app":
        return (
            "Instructions: Clean up RAW_TRANSCRIPTION and return only the cleaned transcript "
            "text without surrounding quotes. Return EMPTY if there should be no result.\n\n"
            f'CONTEXT: "{context or ""}"\n\n'
            f'RAW_TRANSCRIPTION: "{raw}"'
        )
    parts = [f"Raw transcript:\n{raw.strip()}"]
    if context and context.strip():
        parts.append("Context: context: " + context.strip())
    parts.append("Clean this up into the final text.")
    return "\n\n".join(parts)


def completion_budget(input_text, cap=4096):
    """Mirror of ModelConfiguration.completionTokenBudget in the app."""
    est = max(1, (len(input_text) + 2) // 3)
    return max(1, min(cap, max(256, 256 + est * 3)))


def request_params(model, input_text):
    """Generation params per model family, mirroring the app + router."""
    m = model.lower()
    body = {"temperature": 0.0}
    if m.startswith("openai/gpt-oss"):
        body.update({"max_completion_tokens": completion_budget(input_text),
                     "reasoning_effort": "low", "include_reasoning": False})
    elif m.startswith("qwen/"):
        body.update({"max_completion_tokens": completion_budget(input_text),
                     "reasoning_effort": "none"})
    else:
        body["max_tokens"] = completion_budget(input_text)
    return body


# ----------------------------------------------------------------------------
# model runners
# ----------------------------------------------------------------------------
class RawFloor:
    """Do-nothing baseline: emit the raw transcript unchanged."""
    label = "raw (floor)"

    def predict(self, messages, raw):
        return raw


class MLXModel:
    def __init__(self, spec):
        # spec is "MODEL_PATH" or "MODEL_PATH@ADAPTER_PATH"
        if "@" in spec:
            self.model_path, self.adapter = spec.split("@", 1)
            self.adapter = os.path.expanduser(self.adapter)
            self.label = f"{os.path.basename(self.model_path)} +adapter"
        else:
            self.model_path, self.adapter = spec, None
            self.label = os.path.basename(self.model_path) + " (base)"
        try:
            from mlx_lm import load, generate
        except ImportError:
            sys.exit("mlx-lm not installed — `pip install mlx-lm` (needed for --mlx).")
        self._generate = generate
        kwargs = {"adapter_path": self.adapter} if self.adapter else {}
        self.model, self.tokenizer = load(self.model_path, **kwargs)

    def predict(self, messages, raw):
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        # mlx-lm's generate signature has shifted across versions; try the
        # modern sampler form first, fall back to a plain call.
        try:
            from mlx_lm.sample_utils import make_sampler
            out = self._generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=completion_budget(messages[-1]["content"]),
                                 sampler=make_sampler(temp=0.0), verbose=False)
        except Exception:
            out = self._generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=completion_budget(messages[-1]["content"]),
                                 verbose=False)
        return out.strip()


class EndpointModel:
    """Any OpenAI-compatible /chat/completions endpoint (Groq, served MLX, etc.)."""
    def __init__(self, url, model, label, api_key_env=None):
        self.url, self.model, self.label = url, model, label
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not self.api_key:
            sys.exit(f"${api_key_env} not set (needed for endpoint {label}).")

        self.rate_limited = 0

    def predict(self, messages, raw):
        payload = {"model": self.model, "messages": messages}
        payload.update(request_params(self.model, messages[-1]["content"]))
        body = json.dumps(payload).encode()
        for attempt in range(6):
            req = urllib.request.Request(self.url, data=body,
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "FreeFlow-Eval/1.0"})
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                # Latency of the call that actually answered — quota waits are
                # reported in the 429 column, not smeared into p50/p95.
                self.last_latency = time.time() - t0
                return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    # Free-tier TPM bucket; wait it out so we measure the model,
                    # not the quota. Counted separately.
                    self.rate_limited += 1
                    wait = float(e.headers.get("retry-after") or 5) + 0.5
                    time.sleep(min(wait, 60))
                    continue
                raise
        raise RuntimeError("gave up after repeated 429s")


# ----------------------------------------------------------------------------
def load_test(path, fmt, system_prompt=None):
    """Rows: {input, raw, gold, forbidden}. The system prompt is taken from the
    file's first row (exported from the app) unless overridden; the user turn is
    REBUILT in `fmt` from raw+context so we test the format production uses."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            msgs = ex.get("messages") or []
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), None)
            if system_prompt is None and sys_msg:
                system_prompt = sys_msg
            raw, gold = ex["raw"], ex["gold"]
            context = ex.get("context", "")
            rows.append({
                "input": [{"role": "system", "content": system_prompt or ""},
                          {"role": "user", "content": user_message(fmt, raw, context)}],
                "raw": raw, "gold": gold, "context": context,
                "forbidden": ex.get("forbidden", []),
            })
    return rows, system_prompt


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1))
    return xs[k]


_DIFF_LOCK = threading.Lock()


def evaluate(runner, test, worst_out=None):
    wers, cers, exacts, lats, ins, leaks, errors = [], [], [], [], [], 0, 0
    diffs = []
    for ex in test:
        t0 = time.time()
        runner.last_latency = None
        try:
            pred = runner.predict(ex["input"], ex["raw"])
        except Exception as e:
            pred = f"<ERROR: {e}>"
            errors += 1
        lats.append(runner.last_latency if runner.last_latency is not None else time.time() - t0)
        w, c, x = wer(pred, ex["gold"]), cer(pred, ex["gold"]), exact(pred, ex["gold"])
        i, lk = insertion(pred, ex["raw"], ex["gold"]), leaked(pred, ex["forbidden"])
        wers.append(w); cers.append(c); exacts.append(x); ins.append(i); leaks += lk
        diffs.append((w + i + (1.0 if lk else 0.0), w, i, lk, ex["raw"], ex["context"], pred, ex["gold"]))
    n = max(1, len(test))
    if worst_out is not None:
        diffs.sort(key=lambda d: -d[0])
        with _DIFF_LOCK, open(worst_out, "a") as f:
            f.write(f"\n===== {runner.label} — worst {min(6, len(diffs))} =====\n")
            for _, w, i, lk, raw, ctx, pred, gold in diffs[:6]:
                f.write(f"[wer={w:.2f} ins={i:.2f}{' LEAK' if lk else ''}]\n"
                        f"  raw : {raw}\n" + (f"  ctx : {ctx}\n" if ctx else "") +
                        f"  pred: {pred}\n  gold: {gold}\n")
    return {
        "label": runner.label,
        "wer": sum(wers) / n,
        "cer": sum(cers) / n,
        "exact": sum(exacts) / n,
        "ins": sum(ins) / n,
        "leak": leaks,
        "errors": errors,
        "p50": _pct(lats, 0.5),
        "p95": _pct(lats, 0.95),
        "rl": getattr(runner, "rate_limited", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.expanduser("~/.freeflow-ft/test.jsonl"))
    ap.add_argument("--mlx", action="append", default=[],
                    help="MLX model, repeatable. 'PATH' or 'PATH@ADAPTER_PATH'")
    ap.add_argument("--endpoint", action="append", default=[],
                    help="OpenAI-compatible /chat/completions URL (repeatable; pair with the "
                         "Nth --endpoint-model / --endpoint-label / --api-key-env)")
    ap.add_argument("--endpoint-model", action="append", default=[], help="model name for --endpoint")
    ap.add_argument("--endpoint-label", action="append", default=[])
    ap.add_argument("--api-key-env", action="append", default=[],
                    help="env var holding the endpoint API key ('' / '-' for none)")
    ap.add_argument("--format", choices=["app", "train"], default="app",
                    help="how to build the user turn: 'app' = exactly what FreeFlow sends "
                         "in production (default); 'train' = the fine-tune's training format")
    ap.add_argument("--probes", action="append", default=[],
                    help="extra jsonl of {raw, gold, context, forbidden} hallucination probes "
                         "(repeatable). 'leak' counts predictions containing a forbidden phrase.")
    ap.add_argument("--parallel", action="store_true",
                    help="run models concurrently (each model still sequential). Use for "
                         "hosted models that sit in separate rate-limit buckets.")
    ap.add_argument("--no-floor", action="store_true", help="skip the raw-transcript floor")
    ap.add_argument("--diffs", default="/tmp/freeflow-eval-diffs.txt",
                    help="write each model's worst examples here for inspection")
    args = ap.parse_args()

    if not os.path.exists(args.test):
        sys.exit(f"test set not found: {args.test}\n"
                 f"run: python3 export_training_data.py --split-dir ~/.freeflow-ft")
    test, system_prompt = load_test(args.test, args.format)
    if not test:
        sys.exit("test set is empty — collect more gold corrections first "
                 "(the test split only draws from human-corrected rows).")
    n_probes = 0
    for pth in args.probes:
        probes, _ = load_test(pth, args.format, system_prompt=system_prompt)
        test.extend(probes)
        n_probes += len(probes)
    print(f"test set: {len(test) - n_probes} held-out gold examples + {n_probes} probes, "
          f"format={args.format}\n")
    open(args.diffs, "w").close()

    runners = []
    if not args.no_floor:
        runners.append(RawFloor())
    for spec in args.mlx:
        runners.append(MLXModel(spec))
    if len(args.endpoint) != len(args.endpoint_model):
        sys.exit("each --endpoint needs a matching --endpoint-model")
    for i, url in enumerate(args.endpoint):
        label = args.endpoint_label[i] if i < len(args.endpoint_label) else args.endpoint_model[i]
        key_env = args.api_key_env[i] if i < len(args.api_key_env) else None
        if key_env in ("", "-"):
            key_env = None
        runners.append(EndpointModel(url, args.endpoint_model[i], label, key_env))
    if not runners:
        sys.exit("nothing to evaluate — pass --mlx and/or --endpoint")

    results = [None] * len(runners)

    def run(i, r):
        print(f"  running {r.label} ...", flush=True)
        results[i] = evaluate(r, test, worst_out=args.diffs)
        print(f"  done    {r.label}", flush=True)

    if args.parallel:
        threads = [threading.Thread(target=run, args=(i, r)) for i, r in enumerate(runners)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for i, r in enumerate(runners):
            run(i, r)

    # lower WER is better; the floor stays first for reference
    print("\n" + "=" * 100)
    print(f"{'model':<34}{'p50s':>7}{'p95s':>7}{'WER↓':>8}{'CER↓':>8}{'exact↑':>8}{'ins↓':>7}{'leak':>6}{'err':>5}{'429':>5}")
    print("-" * 100)
    for res in results:
        print(f"{res['label']:<34}{res['p50']:>7.2f}{res['p95']:>7.2f}{res['wer']*100:>7.1f}%"
              f"{res['cer']*100:>7.1f}%{res['exact']*100:>7.0f}%{res['ins']*100:>6.1f}%"
              f"{res['leak']:>6}{res['errors']:>5}{res['rl']:>5}")
    print("=" * 100)
    print(f"\np50/p95 = latency per example (speed first). WER/CER = distance to the kept text. "
          f"ins = share of output words present in neither raw nor gold (made-up content). "
          f"leak = probes whose output contains a forbidden context/prompt phrase.\n"
          f"Worst-case examples per model: {args.diffs}")


if __name__ == "__main__":
    main()
