#!/usr/bin/env python3
"""Synthetic audio/buffering benchmark; never reads dictation history.

Run with the local Python environment. --speech additionally uses macOS say and
the already-cached local Parakeet model. --baseline-ref compares a git revision.
All generated audio, imported baseline code and listeners are temporary.
"""
import argparse
import asyncio
import concurrent.futures
import importlib.util
import json
import logging
import os
import pathlib
import subprocess
import tempfile
import time
import wave

import numpy as np

import stt_server

ROOT = pathlib.Path(__file__).resolve().parent.parent
PHRASES = [
    "Please send the updated document tomorrow morning.",
    "The meeting starts at nine thirty on Thursday. Please bring the updated budget and the three invoices. "
    "We need to check the delivery date before we confirm the order. Actually, move the meeting to Friday afternoon.",
    "We need to improve the recording pipeline while preserving every word the speaker intended. "
    "The first step is to collect the audio without losing samples at the boundaries. "
    "The next step is to keep enough context for the speech model to recognize the final phrase. "
    "After that, we should remove unnecessary delays and make sure that a slow connection cannot block the application forever. "
    "Please keep the number forty two, the date September fourth, and the phrase do not delete the backup exactly as spoken. "
    "Finally, verify the result with a repeatable test before making the change available.",
]


def report(**metrics):
    print(json.dumps(metrics), flush=True)


def measure_audio(name, module):
    x = np.random.default_rng(42).normal(0, .1, 72000).astype(np.float32)
    rs = module.StreamResampler(24000)
    streamed = np.concatenate([rs.feed(x[i:i + 1024]) for i in range(0, len(x), 1024)]
                              + [rs.feed(x[:0], final=True)])
    whole = module.resample(x, 24000)
    count = min(len(streamed), len(whole))
    report(stage="resampling", version=name, sample_count_error=len(streamed) - len(whole),
           rms_error=float(np.sqrt(np.mean((streamed[:count] - whole[:count]) ** 2))))

    chunk = np.zeros(1600, dtype=np.float32)
    start, peak = time.perf_counter(), 0
    if hasattr(module, "AudioWindowBuffer"):
        buf = module.AudioWindowBuffer()
        for _ in range(6000):
            buf.append(chunk)
            buf.discard_before(max(0, buf.end - 240000))
            peak = max(peak, buf.storage.nbytes)
    else:
        buf = np.zeros(0, dtype=np.float32)
        for _ in range(6000):
            buf = np.concatenate([buf, chunk])
        peak = buf.nbytes
    report(stage="ten_minute_append", version=name, elapsed_ms=round(1000 * (time.perf_counter() - start), 2),
           retained_capacity_bytes=peak)


def synthetic_audio(scratch):
    clips = []
    for i, phrase in enumerate(PHRASES):
        aiff, wav = scratch / (str(i) + ".aiff"), scratch / (str(i) + ".wav")
        subprocess.run(["say", "-r", "175", "-o", str(aiff), phrase], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        with wave.open(str(wav)) as reader:
            clips.append(np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").astype(np.float32) / 32768)
    return clips


async def measure_speech(modules, clips, transcriber, executor):
    from aiohttp import web
    from test_stt_server import run_session
    reference = {}
    for name, module in modules:
        module.PRECACHE_URL = ""  # Benchmarks must never hand text to a hosted cleanup route.
        module.HEARTBEAT_S = 0
        runner = web.AppRunner(module.make_app(transcriber, executor))
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            for i, clip in enumerate(clips):
                events, text, latency = await run_session(port, [clip], pace=True, chunk_s=1024 / 24000)
                metrics = dict(stage="speech", version=name, case=i, audio_s=round(len(clip) / 16000, 2),
                               final_ms=round(latency * 1000, 1),
                               chunks=sum(1 for event in events if event["type"].endswith("completed")))
                if i in reference:
                    metrics["same_transcript_as_baseline"] = text == reference[i]
                else:
                    reference[i] = text
                report(**metrics)
        finally:
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="git revision to compare with this checkout")
    parser.add_argument("--speech", action="store_true", help="also measure real-time synthetic speech (~2 minutes)")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    os.environ["HF_HUB_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory(prefix="freeflow-pipeline-bench-") as directory:
        scratch = pathlib.Path(directory)
        modules = []
        if args.baseline_ref:
            source = subprocess.check_output(["git", "show", args.baseline_ref + ":local-setup/stt_server.py"], cwd=ROOT)
            path = scratch / "baseline.py"
            path.write_bytes(source)
            spec = importlib.util.spec_from_file_location("baseline", path)
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            modules.append(("baseline", baseline))
        modules.append(("current", stt_server))
        for name, module in modules:
            measure_audio(name, module)
        if args.speech:
            clips = synthetic_audio(scratch)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                transcriber = executor.submit(stt_server.Transcriber, stt_server.MODEL_ID).result()
                asyncio.run(measure_speech(modules, clips, transcriber, executor))


if __name__ == "__main__":
    main()
