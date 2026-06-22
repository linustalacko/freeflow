#!/usr/bin/env python3
"""
Generate synthetic dictation-cleanup training pairs that match YOUR real texture.

Why this exists: with only ~20 real rows, there isn't enough to fine-tune. This
bootstraps volume by asking gpt-oss (via your local router) to invent new
(raw Whisper transcript -> cleaned text) pairs, *few-shot anchored to your actual
history* so the synthetic raw text looks like your real Whisper output (already
capitalised + punctuated, with mid-sentence line breaks, run-ons, trailing
fillers, casual/profane founder-and-dev content) — seasoned with Australian
spelling/vocabulary.

HONEST LIMITS
  - Targets are gpt-oss's cleanups, so this teaches a small model to MATCH
    gpt-oss (distillation), not to beat it. Your captured inline corrections are
    the only signal that beats gpt-oss / learns your personal style.
  - The synthetic raw text is gpt-oss's *imitation* of your Whisper output, not
    real ASR. Always confirm a fine-tune on a held-out set of your REAL rows.

    python3 gen_synthetic_data.py --n 20 --show            # sample, inspect
    python3 gen_synthetic_data.py --n 500 --out syn.jsonl  # scale up
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request

DB = os.path.expanduser("~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite")
ROUTER = "http://127.0.0.1:11435/v1/chat/completions"
GEN_MODEL = "openai/gpt-oss-20b"
DEFAULT_SYSTEM = "Clean up dictated speech into polished written text."

DOMAINS = [
    "giving blunt feedback to an AI coding assistant about a task it did",
    "a startup founder dictating a cold outreach / customer email",
    "thoughts on fundraising, YC, investors, or the YC network",
    "a dictated PR description, bug report, or code-review comment",
    "a casual Telegram message to a mate",
    "a product spec or feature idea for an early-stage startup",
    "reacting to outbound results — bounced emails, LinkedIn limits, lead lists",
    "a quick note-to-self or todo while working",
    "reviewing whether a sales message sounds genuine vs salesy",
    "explaining a technical decision or architecture trade-off out loud",
    "replying to a customer support ticket",
    "a Slack standup update — what I did, what's blocked",
    "dictating a tweet or a LinkedIn post",
    "replying to an investor's email",
    "leaving a voice-memo reminder to yourself",
    "a message to a co-founder about a disagreement",
    "describing the steps to reproduce a bug",
    "a design critique of a UI mockup",
    "notes from a customer discovery call",
    "a hiring note about a candidate after an interview",
    "planning the week or sprint priorities out loud",
    "a message to family or a friend about weekend plans",
    "complaining about a vendor or a tool that broke",
    "brainstorming names or taglines for a feature",
    "dictating meeting notes and action items",
    "a quick reply agreeing or disagreeing in a group chat",
    "explaining a metric or experiment result (conversion, churn, ROC)",
    "a message cancelling or rescheduling something",
    "dictating a grocery list or errands",
    "logistics for meeting up — directions, timing",
]

TOPICS = [
    "pricing and packaging", "a flaky integration test", "onboarding drop-off",
    "a demo that went badly", "Demo Day prep", "an AWS bill spike",
    "a customer who churned", "a lead-qualification feature", "email deliverability",
    "a partnership conversation", "refactoring the auth flow", "a competitor's launch",
    "hiring a first engineer", "a content or podcast idea", "runway and burn",
    "a UI redesign", "a data pipeline bug", "a sales-call follow-up",
    "LinkedIn outreach limits", "weekend plans, footy or surf", "a broken deploy",
    "conversion numbers", "a contract or legal doc", "a conference trip",
    "a mentor's advice", "a bug a user reported", "caching and performance",
    "a Notion doc to tidy up", "a tough investor question", "an ML fine-tuning run",
]

LENGTHS = [
    "a single short sentence", "a short two-sentence message",
    "a medium message of three or four sentences",
    "a long rambling paragraph with several ideas",
    "a quick fragment or phrase",
    "a multi-topic brain-dump that jumps between subjects",
]

TEXTURES = [
    "several hard line breaks in the middle of sentences",
    "a self-correction (says one thing then corrects it, e.g. 'Thursday, no actually Wednesday')",
    "a trailing 'um' and a false start",
    "a long run-on with almost no full stops",
    "a homophone / ASR slip to fix (e.g. 'to working' for 'to work')",
    "a repeated word like 'the the' or 'I I'",
    "casual profanity that should be preserved",
    "a list of items spoken as a run-on",
]

AUS = ("Use Australian SPELLING consistently in both RAW and CLEAN (organise, "
       "colour, realise, favourite, centre, cancelled, behaviour). Slang should be "
       "RARE and natural — at most an occasional 'reckon', 'heaps', 'keen', 'no "
       "worries'. Most messages have NO slang at all. Absolutely no parody / "
       "Crocodile-Dundee voice (no 'fair dinkum', 'chockers', 'plonker', 'crikey').")


def real_examples(db, k=6):
    if not os.path.exists(db):
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT ZRAWTRANSCRIPT, ZPOSTPROCESSEDTRANSCRIPT FROM ZPIPELINEHISTORYENTRY "
        "WHERE ZRAWTRANSCRIPT<>'' AND ZPOSTPROCESSEDTRANSCRIPT<>'' "
        "AND LENGTH(ZRAWTRANSCRIPT) > 80 ORDER BY LENGTH(ZRAWTRANSCRIPT) DESC"
    ).fetchall()
    conn.close()
    return [{"raw": r[0], "clean": r[1]} for r in rows[:k]]


def system_prompt(db):
    if not os.path.exists(db):
        return DEFAULT_SYSTEM
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT ZSYSTEMPROMPT FROM ZPIPELINEHISTORYENTRY WHERE ZSYSTEMPROMPT<>'' "
        "ORDER BY ZTIMESTAMP DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else DEFAULT_SYSTEM


def build_user_message(raw):
    return f"Raw transcript:\n{raw.strip()}\n\nClean this up into the final text."


def gen_prompt(anchors, scenario, k):
    shots = "\n\n".join(
        f"RAW:\n{a['raw']}\nCLEAN:\n{a['clean']}" for a in anchors)
    return f"""Here are REAL examples of a user's dictation. RAW is the raw Whisper
transcript (already capitalised and punctuated, but with mid-sentence line breaks,
run-on sentences, trailing fillers like "um", and casual/profane stream-of-thought).
CLEAN is the cleaned final text (line breaks joined, run-ons split into sentences,
fillers dropped, light grammar polish, meaning preserved).

{shots}

Now invent {k} NEW pairs in the SAME texture and style. For THIS batch:
- Scenario: {scenario['domain']}
- Loosely about: {scenario['topic']}
- Typical length: {scenario['length']}
- At least a couple should feature: {scenario['texture']}
{AUS}

CRITICAL — DIVERSITY: make the {k} examples maximally DIFFERENT from one another.
Vary the opening words, the length, the sentence structure, and the specifics. Do
NOT start two examples the same way; mix one-liners with longer paragraphs.

Rules:
- RAW must look like real Whisper output: coherent, realistic content a founder or
  developer would actually dictate; capitalised + punctuated, but with some hard
  line breaks mid-sentence, run-ons, the odd homophone/ASR slip, occasional trailing
  "um". Vary length (some short, some a long rambling paragraph).
- CLEAN must PRESERVE ALL the content and meaning of RAW. Do NOT summarise, shorten,
  or drop any sentence. Only: join the line breaks, split run-ons into sentences,
  drop pure fillers, and fix punctuation/grammar/obvious ASR slips.
- Keep the speaker's tone and any profanity; do not sanitise the meaning.
- RAW and CLEAN must differ (otherwise it teaches nothing) but say the SAME things.
- Output ONLY a JSON array of objects: [{{"raw":"...","clean":"..."}}]. No prose."""


def call_router(url, model, messages, temperature, retries=6, api_key=None):
    """Call an OpenAI-compatible endpoint (Groq router, local Ollama, or
    OpenRouter); retry with exponential backoff on errors OR empty responses."""
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "stream": False}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    delay = 5.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if content:
                return content, (data.get("usage") or {})
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 1.8, 60)
    return "", {}


def extract_json_array(text):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # tolerate trailing commas / stray text by trimming to last full object
        chunk = text[start:end + 1]
        try:
            return json.loads(chunk.replace(",]", "]"))
        except json.JSONDecodeError:
            return []


def extract_pairs(text):
    """Return [(raw, clean), ...] from a model response — handles both a JSON
    array and the markdown 'RAW: ... CLEAN: ...' format that local models emit."""
    arr = extract_json_array(text)
    if arr:
        out = []
        for o in arr:
            if isinstance(o, dict):
                out.append((str(o.get("raw", "")).strip(), str(o.get("clean", "")).strip()))
        if out:
            return out
    # Fallback: parse RAW:/CLEAN: blocks (strip bold + "Example N" headers first).
    t = re.sub(r"\*+", "", text)
    t = re.sub(r"(?im)^\s*example\s*\d+\s*$", "", t)
    out = []
    for part in re.split(r"(?im)^\s*raw\s*:\s*", t)[1:]:
        seg = re.split(r"(?im)^\s*clean\s*:\s*", part, maxsplit=1)
        if len(seg) != 2:
            seg = re.split(r"(?i)\bclean\s*:\s*", part, maxsplit=1)
        if len(seg) == 2:
            raw = seg[0].strip()
            clean = re.split(r"(?im)^\s*example\s*\d+", seg[1])[0].strip()
            if raw and clean:
                out.append((raw, clean))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="target number of pairs")
    ap.add_argument("--batch", type=int, default=10, help="pairs per gpt-oss call")
    ap.add_argument("--out", default=os.path.expanduser("~/.freeflow-ft/synthetic.jsonl"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--show", action="store_true", help="print samples to stdout")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pace", type=float, default=2.0,
                    help="seconds to sleep between calls (free-tier friendliness)")
    ap.add_argument("--base-url", default=ROUTER,
                    help="OpenAI-compatible endpoint (default router; for local: "
                         "http://127.0.0.1:11434/v1/chat/completions)")
    ap.add_argument("--model", default=GEN_MODEL,
                    help="generation model (e.g. openai/gpt-oss-120b, llama3.1:8b)")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the API key, e.g. OPENROUTER_API_KEY")
    ap.add_argument("--api-key-file", default=None,
                    help="file containing the API key (one line)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="hard spend cap: stop once measured cost reaches this")
    ap.add_argument("--price-in", type=float, default=0.0,
                    help="$ per MILLION input tokens (for budget tracking)")
    ap.add_argument("--price-out", type=float, default=0.0,
                    help="$ per MILLION output tokens (for budget tracking)")
    args = ap.parse_args()

    api_key = None
    if args.api_key_file:
        p = os.path.expanduser(args.api_key_file)
        if os.path.exists(p):
            api_key = open(p).read().strip()
    if not api_key and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    if (args.api_key_file or args.api_key_env) and not api_key:
        sys.exit("API key not found (file missing/empty or env var unset)")

    rnd = random.Random(args.seed)
    anchors = real_examples(DB)
    if len(anchors) < 2:
        sys.exit("need a few real examples in the DB to anchor generation; dictate more first")
    system = system_prompt(DB)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resume: load whatever's already on disk so throttling never loses progress.
    seen, pairs = set(), []
    if os.path.exists(args.out):
        for line in open(args.out):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("raw"):
                seen.add(rec["raw"][:60].lower())
                pairs.append(rec)
        if pairs:
            print(f"  resuming from {len(pairs)} existing pairs", file=sys.stderr)

    def record(raw, clean, domain):
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_message(raw)},
                {"role": "assistant", "content": clean},
            ],
            "label_source": "synthetic_au",
            "raw": raw, "gold": clean, "domain": domain,
        }

    fout = open(args.out, "a")
    consecutive_empty = 0
    cost = 0.0
    while len(pairs) < args.n and consecutive_empty < 10:
        scenario = {
            "domain": rnd.choice(DOMAINS),
            "topic": rnd.choice(TOPICS),
            "length": rnd.choice(LENGTHS),
            "texture": rnd.choice(TEXTURES),
        }
        domain = scenario["domain"]
        k = min(args.batch, args.n - len(pairs) + 2)
        content, usage = call_router(args.base_url, args.model,
                               [{"role": "user", "content": gen_prompt(anchors, scenario, k)}],
                               args.temperature, api_key=api_key)
        # Prefer OpenRouter's exact reported cost; fall back to price estimate.
        if usage.get("cost") is not None:
            cost += float(usage["cost"])
        else:
            cost += (usage.get("prompt_tokens", 0) * args.price_in
                     + usage.get("completion_tokens", 0) * args.price_out) / 1e6
        if not content:
            consecutive_empty += 1
            backoff = min(15 * consecutive_empty, 90)
            print(f"  throttled ({consecutive_empty}/10) — backing off {backoff}s",
                  file=sys.stderr)
            time.sleep(backoff)
            continue
        consecutive_empty = 0
        added = 0
        for raw, clean in extract_pairs(content):
            raw, clean = raw.strip(), clean.strip()
            if not raw or not clean or raw == clean:
                continue
            if len(raw) < 8 or len(clean) > 1200:
                continue
            key = raw[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            rec = record(raw, clean, domain)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            pairs.append(rec)
            added += 1
        print(f"  {len(pairs)}/{args.n} (+{added})  spent ${cost:.3f}", file=sys.stderr)
        if args.budget_usd and cost >= args.budget_usd:
            print(f"  budget ${args.budget_usd:.2f} reached (spent ${cost:.2f}) — stopping",
                  file=sys.stderr)
            break
        time.sleep(args.pace)
    fout.close()
    print(f"wrote {len(pairs)} total synthetic pairs -> {args.out}")

    if args.show:
        for p in pairs[:8]:
            print("\n────────── [" + p.get("domain", "")[:40] + "]")
            print("RAW :", p["raw"].replace("\n", "\\n "))
            print("CLEAN:", p["gold"])


if __name__ == "__main__":
    main()
