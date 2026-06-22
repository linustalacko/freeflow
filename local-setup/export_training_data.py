#!/usr/bin/env python3
"""
Export FreeFlow's pipeline history (Core Data SQLite) -> SFT training JSONL.

Each dictation becomes one chat example for fine-tuning the post-processing LLM:

    system : the post-processing system prompt actually used
    user   : the raw Whisper transcript + context (app, window, vocab)
    assistant (TARGET) : the CORRECTED text if you saved one (gold), else the
                         gpt-oss output (distillation target)

Read-only: opens the DB with mode=ro so it never interferes with the running app.

    python3 export_training_data.py                      # -> training.jsonl
    python3 export_training_data.py --gold-only          # only rows you corrected
    python3 export_training_data.py --split-dir DIR      # train/valid/test.jsonl

--split-dir writes three files for an HONEST eval:
    train.jsonl / valid.jsonl  -> fed to finetune_local.sh
    test.jsonl                 -> held out, read ONLY by eval_models.py
The test set is drawn from GOLD (human-corrected) rows only — those are the real
target — and membership is a stable hash of the transcript, so a given dictation
always lands in the same split even as you collect more. No leakage, and eval
numbers stay comparable across runs.

Until you have gold rows, every target falls back to the gpt-oss output — i.e.
distillation, which only teaches a small Qwen to MATCH gpt-oss. Corrected rows
are what let it BEAT gpt-oss (learn your preferences), and they're the only rows
the eval scores against.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys

DB = os.path.expanduser(
    "~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite"
)
TABLE = "ZPIPELINEHISTORYENTRY"
DEFAULT_SYSTEM = "Clean up dictated speech into polished written text."


def col_exists(conn, table, col):
    cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    return col in cols


def build_user_message(raw, app, window, vocab, summary):
    parts = [f"Raw transcript:\n{raw.strip()}"]
    ctx = []
    if app:
        ctx.append(f"app: {app}")
    if window:
        ctx.append(f"window: {window}")
    if summary and summary.strip():
        ctx.append(f"context: {summary.strip()}")
    if ctx:
        parts.append("Context: " + " | ".join(ctx))
    if vocab and vocab.strip():
        parts.append(f"Custom vocabulary: {vocab.strip()}")
    parts.append("Clean this up into the final text.")
    return "\n\n".join(parts)


def stable_bucket(key, mod=100):
    """Deterministic 0..mod-1 bucket from a string (PYTHONHASHSEED-independent)."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def load_examples(db, gold_only=False):
    if not os.path.exists(db):
        sys.exit(f"DB not found: {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    has_corrected = col_exists(conn, TABLE, "ZCORRECTEDTRANSCRIPT")

    select = [
        "ZRAWTRANSCRIPT", "ZPOSTPROCESSEDTRANSCRIPT", "ZSYSTEMPROMPT",
        "ZCONTEXTAPPNAME", "ZCONTEXTWINDOWTITLE", "ZCUSTOMVOCABULARY",
        "ZCONTEXTSUMMARY", "ZINTENT", "ZTIMESTAMP",
    ]
    if has_corrected:
        select.append("ZCORRECTEDTRANSCRIPT")

    rows = conn.execute(
        f'SELECT {", ".join(select)} FROM "{TABLE}" '
        f"WHERE ZRAWTRANSCRIPT IS NOT NULL AND ZRAWTRANSCRIPT != '' "
        f"ORDER BY ZTIMESTAMP"
    ).fetchall()
    conn.close()

    examples = []
    for r in rows:
        d = dict(zip(select, r))
        corrected = d.get("ZCORRECTEDTRANSCRIPT")
        gpt_oss = d.get("ZPOSTPROCESSEDTRANSCRIPT") or ""
        is_gold = bool(corrected and corrected.strip())
        target = (corrected if is_gold else gpt_oss).strip()
        if gold_only and not is_gold:
            continue
        if not target:
            continue
        raw = d["ZRAWTRANSCRIPT"].strip()
        system = d.get("ZSYSTEMPROMPT") or DEFAULT_SYSTEM
        user = build_user_message(
            raw, d.get("ZCONTEXTAPPNAME"), d.get("ZCONTEXTWINDOWTITLE"),
            d.get("ZCUSTOMVOCABULARY"), d.get("ZCONTEXTSUMMARY"))
        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
            "label_source": "human_correction" if is_gold else "gpt_oss_distill",
            "intent": d.get("ZINTENT"),
            "is_gold": is_gold,
            "raw": raw,
            "gold": target,
        })
    return examples, has_corrected


def training_record(ex):
    """Strip eval-only fields; keep what MLX-LM reads + provenance."""
    return {"messages": ex["messages"],
            "label_source": ex["label_source"],
            "intent": ex["intent"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training.jsonl")
    ap.add_argument("--split-dir",
                    help="write train/valid/test.jsonl into DIR (test = held-out gold)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of GOLD rows held out for the test set")
    ap.add_argument("--valid-frac", type=float, default=0.1,
                    help="fraction held out for validation during training")
    ap.add_argument("--gold-only", action="store_true",
                    help="only rows that have a saved human correction")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    examples, has_corrected = load_examples(args.db, gold_only=args.gold_only)
    n_gold = sum(e["is_gold"] for e in examples)

    if not args.split_dir:
        with open(args.out, "w") as f:
            for e in examples:
                f.write(json.dumps(training_record(e), ensure_ascii=False) + "\n")
        print(f"wrote {len(examples)} examples -> {args.out}")
        print(f"  gold (human-corrected): {n_gold}")
        print(f"  distill (gpt-oss target): {len(examples) - n_gold}")
        if not has_corrected:
            print("note: ZCORRECTEDTRANSCRIPT column not present yet.")
        return

    # ---- split mode --------------------------------------------------------
    os.makedirs(args.split_dir, exist_ok=True)
    test_pct = int(round(args.test_frac * 100))
    valid_pct = int(round(args.valid_frac * 100))
    train, valid, test = [], [], []
    for e in examples:
        b = stable_bucket(e["raw"])
        if e["is_gold"] and b < test_pct:
            test.append(e)                       # full record (has raw + gold)
        elif b < test_pct + valid_pct:
            valid.append(training_record(e))
        else:
            train.append(training_record(e))

    def dump(name, rows):
        # Skip empty splits: mlx_lm's loader tolerates a MISSING file but crashes
        # on an EMPTY one (it does data[0]). With 0 gold the test split is empty.
        path = os.path.join(args.split_dir, name)
        if rows:
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif os.path.exists(path):
            os.remove(path)

    dump("train.jsonl", train)
    dump("valid.jsonl", valid)
    dump("test.jsonl", test)
    print(f"split written to {args.split_dir}/")
    print(f"  train: {len(train)}   valid: {len(valid)}   test (held-out gold): {len(test)}")
    print(f"  total gold so far: {n_gold}")
    if len(test) < 20:
        print(f"  warning: only {len(test)} held-out gold examples — eval signal will be noisy.")
        print(f"  keep dictating + correcting; ~50+ gold gives a usable read, ~200+ is solid.")


if __name__ == "__main__":
    main()
