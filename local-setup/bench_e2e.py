#!/usr/bin/env python3
"""End-to-end latency benchmark for the local FreeFlow pipeline, on REAL recordings.

Simulates what the app does at key-up, per WAV in the audio dir:
  1. stream the audio to the STT server's /v1/realtime at real-time pace
     (so mid-speech segmentation behaves like it does when you talk), then commit
     → measures commit→final (what the user waits for after key-up)
  2. send the transcript through the router exactly like PostProcessingService
     (same system prompt, same user template, sized completion budget)
     → measures cleanup latency
  3. total = 1 + 2 (paste overhead is ~30ms and constant)

Usage:
  ~/.freeflow-ft/venv/bin/python local-setup/bench_e2e.py [--files N] [--no-pace] [--label X]
  env: STT_URL (default http://127.0.0.1:8082) ROUTER_URL (default http://127.0.0.1:11435)
Prints a table and appends a JSON line to /tmp/freeflow-bench.jsonl for before/after diffs.
"""
import argparse
import asyncio
import glob
import json
import os
import sqlite3
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_stt_server import run_session  # noqa: E402  (drives the realtime protocol)
from detclean import det_clean, profile_for  # noqa: E402  (mirror of the app's TranscriptFastPath)

STT_URL = os.environ.get("STT_URL", "http://127.0.0.1:8082")
ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:11435")
AUDIO_DIR = os.path.expanduser("~/Library/Application Support/FreeFlow Dev/audio")
DB = os.path.expanduser("~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite")


def app_system_prompt():
    """The exact system prompt the app used most recently (from the run log)."""
    try:
        row = sqlite3.connect(DB).execute(
            "select ZSYSTEMPROMPT from ZPIPELINEHISTORYENTRY where ZSYSTEMPROMPT is not null "
            "order by ZTIMESTAMP desc limit 1").fetchone()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    sys.exit("no system prompt in run log")


def completion_budget(text, cap=4096):
    est = max(1, (len(text) + 2) // 3)
    return max(1, min(cap, max(256, 256 + est * 3)))


def cleanup(system_prompt, transcript, context=""):
    import urllib.request
    user = ("Instructions: Clean up RAW_TRANSCRIPTION and return only the cleaned transcript "
            "text without surrounding quotes. Return EMPTY if there should be no result.\n\n"
            f'CONTEXT: "{context}"\n\nRAW_TRANSCRIPTION: "{transcript}"')
    body = {"model": "openai/gpt-oss-20b", "temperature": 0.0, "reasoning_effort": "low",
            "include_reasoning": False, "max_completion_tokens": completion_budget(transcript),
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]}
    req = urllib.request.Request(ROUTER_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-FreeFlow-Deadline-Ms": "20000"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        route = r.headers.get("X-FreeFlow-Route", "?")
        out = json.load(r)["choices"][0]["message"]["content"]
    return out, time.monotonic() - t0, route


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-cleanup", action="store_true")
    ap.add_argument("--target", default="unknown",
                    help="destination profile for the deterministic fast path "
                         "(email, chat, code, terminal, document, searchField, unknown)")
    args = ap.parse_args()
    files = sorted(glob.glob(AUDIO_DIR + "/*.wav"), key=os.path.getmtime)[-args.files:]
    port = int(STT_URL.rsplit(":", 1)[1])
    sysp = None if args.skip_cleanup else app_system_prompt()
    rows = []
    print(f"{'audio':>6} {'tail→final':>11} {'segs':>5} {'cleanup':>8} {'route':>10} {'TOTAL':>7}  transcript → cleaned")
    for f in files:
        with wave.open(f) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        dur = len(x) / 16000
        events, text, lat = await run_session(port, [x], pace=not args.no_pace)
        segs = sum(1 for e in events if e["type"].endswith("completed") and not e["_after_commit"])
        if args.skip_cleanup or not text:
            cl, ct, route = "", 0.0, "-"
        else:
            t0 = time.monotonic()
            cl = det_clean(text, profile=profile_for(args.target))  # the app skips the LLM for these (TranscriptFastPath)
            if cl is not None:
                ct, route = time.monotonic() - t0, "fast-path"
            else:
                cl, ct, route = cleanup(sysp, text)
        rows.append({"file": os.path.basename(f), "audio_s": dur, "stt_ms": lat * 1000, "segs": segs,
                     "cleanup_ms": ct * 1000, "route": route, "total_ms": (lat + ct) * 1000,
                     "text": text, "cleaned": cl})
        print(f"{dur:>5.1f}s {lat*1000:>9.0f}ms {segs:>5} {ct*1000:>6.0f}ms {route:>10} {(lat+ct)*1000:>5.0f}ms  "
              f"{text[:60]!r} → {cl[:60]!r}")
    stt = sorted(r["stt_ms"] for r in rows); tot = sorted(r["total_ms"] for r in rows)
    cle = sorted(r["cleanup_ms"] for r in rows)
    med = lambda v: v[len(v) // 2]
    print(f"\nmedian: stt {med(stt):.0f}ms  cleanup {med(cle):.0f}ms  TOTAL {med(tot):.0f}ms   "
          f"max TOTAL {max(tot):.0f}ms   ({len(rows)} files, label={args.label!r})")
    with open("/tmp/freeflow-bench.jsonl", "a") as fh:
        fh.write(json.dumps({"label": args.label, "ts": time.time(), "rows": rows}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
