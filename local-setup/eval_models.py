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
import os
import re
import sys
import time
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


def exact(pred, gold):
    return _normalize(pred).lower() == _normalize(gold).lower()


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
                                 max_tokens=512, sampler=make_sampler(temp=0.0),
                                 verbose=False)
        except Exception:
            out = self._generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=512, verbose=False)
        return out.strip()


class EndpointModel:
    """Any OpenAI-compatible /chat/completions endpoint (Groq, served MLX, etc.)."""
    def __init__(self, url, model, label, api_key_env=None):
        self.url, self.model, self.label = url, model, label
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not self.api_key:
            sys.exit(f"${api_key_env} not set (needed for endpoint {label}).")

    def predict(self, messages, raw):
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
def load_test(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # messages WITHOUT the gold assistant turn = the model's input
            msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
            rows.append({"input": msgs, "raw": ex["raw"], "gold": ex["gold"]})
    return rows


def evaluate(runner, test, worst_out=None):
    wers, cers, exacts, lats = [], [], [], []
    diffs = []
    for ex in test:
        t0 = time.time()
        try:
            pred = runner.predict(ex["input"], ex["raw"])
        except Exception as e:
            pred = f"<ERROR: {e}>"
        lats.append(time.time() - t0)
        w, c, x = wer(pred, ex["gold"]), cer(pred, ex["gold"]), exact(pred, ex["gold"])
        wers.append(w); cers.append(c); exacts.append(x)
        diffs.append((w, ex["raw"], pred, ex["gold"]))
    n = max(1, len(test))
    if worst_out is not None:
        diffs.sort(key=lambda d: -d[0])
        with open(worst_out, "a") as f:
            f.write(f"\n===== {runner.label} — worst {min(5, len(diffs))} =====\n")
            for w, raw, pred, gold in diffs[:5]:
                f.write(f"[wer={w:.2f}]\n  raw : {raw}\n  pred: {pred}\n  gold: {gold}\n")
    return {
        "label": runner.label,
        "wer": sum(wers) / n,
        "cer": sum(cers) / n,
        "exact": sum(exacts) / n,
        "lat": sum(lats) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.expanduser("~/.freeflow-ft/test.jsonl"))
    ap.add_argument("--mlx", action="append", default=[],
                    help="MLX model, repeatable. 'PATH' or 'PATH@ADAPTER_PATH'")
    ap.add_argument("--endpoint", help="OpenAI-compatible /chat/completions URL")
    ap.add_argument("--endpoint-model", help="model name for --endpoint")
    ap.add_argument("--endpoint-label", default="endpoint")
    ap.add_argument("--api-key-env", help="env var holding the endpoint API key")
    ap.add_argument("--no-floor", action="store_true", help="skip the raw-transcript floor")
    ap.add_argument("--diffs", default="/tmp/freeflow-eval-diffs.txt",
                    help="write each model's worst examples here for inspection")
    args = ap.parse_args()

    if not os.path.exists(args.test):
        sys.exit(f"test set not found: {args.test}\n"
                 f"run: python3 export_training_data.py --split-dir ~/.freeflow-ft")
    test = load_test(args.test)
    if not test:
        sys.exit("test set is empty — collect more gold corrections first "
                 "(the test split only draws from human-corrected rows).")
    print(f"test set: {len(test)} held-out gold examples\n")
    open(args.diffs, "w").close()

    runners = []
    if not args.no_floor:
        runners.append(RawFloor())
    for spec in args.mlx:
        runners.append(MLXModel(spec))
    if args.endpoint:
        if not args.endpoint_model:
            sys.exit("--endpoint requires --endpoint-model")
        runners.append(EndpointModel(args.endpoint, args.endpoint_model,
                                     args.endpoint_label, args.api_key_env))
    if not runners:
        sys.exit("nothing to evaluate — pass --mlx and/or --endpoint")

    results = []
    for r in runners:
        print(f"  running {r.label} ...", flush=True)
        results.append(evaluate(r, test, worst_out=args.diffs))

    # lower WER is better; the floor stays first for reference
    print("\n" + "=" * 74)
    print(f"{'model':<40}{'WER↓':>8}{'CER↓':>8}{'exact↑':>9}{'s/ex':>8}")
    print("-" * 74)
    for res in results:
        print(f"{res['label']:<40}{res['wer']*100:>7.1f}%{res['cer']*100:>7.1f}%"
              f"{res['exact']*100:>8.0f}%{res['lat']:>8.2f}")
    print("=" * 74)
    print(f"\nWER/CER = avg distance to YOUR kept text (lower=better). "
          f"A model is worth shipping when its WER is well under the raw floor and "
          f"near the gpt-oss bar.\nWorst-case examples per model: {args.diffs}")


if __name__ == "__main__":
    main()
