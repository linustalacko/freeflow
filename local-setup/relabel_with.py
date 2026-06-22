#!/usr/bin/env python3
"""
Relabel a jsonl's target using a STRONGER teacher model (default gpt-oss-120b) via
the local router. This is the lever to BEAT gpt-oss-20b: the student can only be as
good as its targets, so we upgrade the targets from 20b to 120b (or, ideally, your
own corrections). Also used to build a better eval answer-key so "beating 20b"
becomes measurable (20b becomes a competitor, not the gold).

Paced + backoff for free-tier throttling. Resumable (append + skip done).

    python3 relabel_with.py --in real_test.jsonl --out real_test_120b.jsonl
    python3 relabel_with.py --in synthetic.jsonl --out synthetic_120b.jsonl --model openai/gpt-oss-120b
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROUTER = "http://127.0.0.1:11435/v1/chat/completions"


def clean_with(model, system, raw, retries=6):
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Raw transcript:\n{raw}\n\nClean this up into the final text."},
    ], "temperature": 0}).encode()
    delay = 5.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(ROUTER, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            c = (d["choices"][0]["message"].get("content") or "").strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 1.8, 60)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--pace", type=float, default=3.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l).get("raw", ""))
            except json.JSONDecodeError:
                pass

    out = open(args.out, "a")
    n = 0
    for r in rows:
        raw = r.get("raw", "")
        if not raw or raw in done:
            continue
        system = r["messages"][0]["content"]
        clean = clean_with(args.model, system, raw)
        if not clean:
            print(f"  throttled after {n} — stop; re-run to resume", file=sys.stderr)
            break
        rec = dict(r)
        rec["messages"] = [r["messages"][0], r["messages"][1],
                           {"role": "assistant", "content": clean}]
        rec["gold"] = clean
        rec["label_source"] = f"teacher:{args.model}"
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        n += 1
        print(f"  relabeled {n}", file=sys.stderr)
        time.sleep(args.pace)
    out.close()
    print(f"relabeled {n} new rows -> {args.out}")


if __name__ == "__main__":
    main()
