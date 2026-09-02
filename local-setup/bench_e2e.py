#!/usr/bin/env python3
"""End-to-end latency + accuracy benchmark for the local FreeFlow pipeline, on REAL recordings.

Simulates what the app does, per WAV in the audio dir:
  1. stream the audio to the STT server's /v1/realtime at real-time pace (so the
     mid-speech settling behaves like it does when you talk), then commit
     → measures commit→final (what the user waits for after key-up), how many
       chunks settled while "speaking", and how the streamed transcript compares
       to a single batch transcription of the same file (WER: the accuracy cost
       of streaming, which must stay near zero)
  2. run the app's deterministic fast path (TranscriptFastPath twin); if it
     declines, send the transcript through the router exactly like
     PostProcessingService does (same system prompt, the app's request shape from
     app_prompt.py, sized completion budget) → measures cleanup latency + route
  3. total = 1 + 2 (paste overhead is ~30ms and constant)

Usage:
  ~/.freeflow-ft/venv/bin/python local-setup/bench_e2e.py [--files N] [--no-pace] [--label X] [--target chat]
  env: STT_URL (default http://127.0.0.1:8082) ROUTER_URL (default http://127.0.0.1:11435)
Prints a table and appends a JSON line to /tmp/freeflow-bench.jsonl for before/after diffs.
Benchmark on an idle machine: GPU contention from anything else on the box moves
these numbers far more than most code changes do (check `uptime` first).
"""
import argparse
import asyncio
import glob
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_stt_server import run_session  # noqa: E402  (drives the realtime protocol)
from detclean import det_clean, profile_for  # noqa: E402  (mirror of the app's TranscriptFastPath)
import app_prompt  # noqa: E402

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


def cleanup(system_prompt, transcript, context=""):
    body = app_prompt.cleanup_request(system_prompt, transcript, context)
    req = urllib.request.Request(ROUTER_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-FreeFlow-Deadline-Ms": "20000"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        route = r.headers.get("X-FreeFlow-Route", "?")
        out = json.load(r)["choices"][0]["message"]["content"]
    return out, time.monotonic() - t0, route


def batch_transcribe(path):
    """One-shot transcription of the whole file: the accuracy reference for
    streaming. The audio takes the same 16k→24k→16k resampling round trip the
    realtime socket imposes, so the comparison isolates the streaming logic."""
    import io
    from stt_server import resample
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    y = resample(resample(x, 16000, 24000), 24000, 16000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    boundary = "----freeflowbench"
    data = buf.getvalue()
    body = (("--%s\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nx\r\n" % boundary).encode()
            + ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
               "Content-Type: audio/wav\r\n\r\n" % boundary).encode() + data + ("\r\n--%s--\r\n" % boundary).encode())
    req = urllib.request.Request(STT_URL + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.load(r)["text"]
    return text, time.monotonic() - t0


def _norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref, hyp):
    """Word error rate of `hyp` against `ref` (case/punctuation-insensitive)."""
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


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
    print(f"{'audio':>6} {'tail→final':>11} {'chunks':>6} {'WERvsB':>6} {'cleanup':>8} {'route':>10} {'TOTAL':>7}  transcript → cleaned")
    for f in files:
        with wave.open(f) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        dur = len(x) / 16000
        batch, batch_s = batch_transcribe(f)
        events, text, lat = await run_session(port, [x], pace=not args.no_pace)
        chunks = sum(1 for e in events if e["type"].endswith("completed") and not e["_after_commit"])
        w_err = wer(batch, text)
        if args.skip_cleanup or not text:
            cl, ct, route = "", 0.0, "-"
        else:
            t0 = time.monotonic()
            cl = det_clean(text, profile=profile_for(args.target))  # the app skips the LLM for these (TranscriptFastPath)
            if cl is not None:
                ct, route = time.monotonic() - t0, "fast-path"
            else:
                cl, ct, route = cleanup(sysp, text)
        rows.append({"file": os.path.basename(f), "audio_s": dur, "stt_ms": lat * 1000, "chunks": chunks,
                     "batch_ms": batch_s * 1000, "wer_vs_batch": w_err,
                     "cleanup_ms": ct * 1000, "route": route, "total_ms": (lat + ct) * 1000,
                     "text": text, "batch": batch, "cleaned": cl})
        print(f"{dur:>5.1f}s {lat*1000:>9.0f}ms {chunks:>6} {w_err*100:>5.1f}% {ct*1000:>6.0f}ms {route:>10} {(lat+ct)*1000:>5.0f}ms  "
              f"{text[:50]!r} → {cl[:50]!r}")
    stt = sorted(r["stt_ms"] for r in rows); tot = sorted(r["total_ms"] for r in rows)
    cle = sorted(r["cleanup_ms"] for r in rows)
    words = sum(len(_norm(r["batch"])) for r in rows)
    errs = sum(r["wer_vs_batch"] * len(_norm(r["batch"])) for r in rows)
    med = lambda v: v[len(v) // 2]
    print(f"\nmedian: stt {med(stt):.0f}ms  cleanup {med(cle):.0f}ms  TOTAL {med(tot):.0f}ms   "
          f"max TOTAL {max(tot):.0f}ms   stream-vs-batch WER {100 * errs / max(1, words):.2f}% over {words} words"
          f"   ({len(rows)} files, label={args.label!r})")
    with open("/tmp/freeflow-bench.jsonl", "a") as fh:
        fh.write(json.dumps({"label": args.label, "ts": time.time(), "rows": rows}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
