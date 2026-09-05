#!/usr/bin/env python3
"""Optional Core ML model benchmark; explicit corpora and in-memory scoring only."""
import argparse
import array
import base64
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import benchmark as b


def worker_metadata(worker):
    metadata = json.loads(worker.with_suffix(".json").read_text())
    if metadata.get("library_logging_disabled") is not True or metadata.get("worker_sha256") != hashlib.sha256(worker.read_bytes()).hexdigest():
        raise ValueError("worker build metadata does not match")
    return metadata


class CoreMLClient:
    def __init__(self, worker, models):
        worker_metadata(worker)
        self.process = subprocess.Popen([str(worker), str(models)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        env={**os.environ, "FREEFLOW_BENCHMARK_SILENT": "1", "OS_ACTIVITY_MODE": "disable"})
        try:
            if not self.receive(600).get("ready"):
                raise RuntimeError("coreml_setup_failed")
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError("coreml_timeout")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("coreml_worker_closed")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError("coreml_worker_failed")
        return response

    def transcribe(self, pcm):
        started = time.perf_counter()
        request = {"pcm_f32_base64": base64.b64encode(pcm).decode("ascii")}
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()
        response = self.receive(60)
        if not isinstance(response.get("text"), str):
            raise ValueError("missing_transcript")
        response["total_s"] = time.perf_counter() - started
        return response

    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("output must be new and counts positive")
    clips = []
    for corpus, manifest in enumerate(args.manifest):
        for row in b.read_manifest(manifest)[:args.limit]:
            pcm16 = b.command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(row["audio"]),
                               "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]).stdout
            ints = array.array("h", pcm16)
            if sys.byteorder != "little":
                ints.byteswap()
            floats = array.array("f", (x / 32768 for x in ints))
            if sys.byteorder != "little":
                floats.byteswap()
            clips.append({"id": row["id"], "reference": row["text"], "corpus": corpus,
                          "category": row.get("category", "speech"), "pcm": floats.tobytes(),
                          "audio_s": len(ints) / 16000, "audio_sha256": hashlib.sha256(pcm16).hexdigest()})
    report = {"kind": "coreml-complete-clip", "model": "parakeet-tdt-0.6b-v2-coreml",
              "worker_build": worker_metadata(args.worker),
              "clip_count": len(clips), "repeats": args.repeats, "trials": []}
    print("Loading Core ML models into the isolated worker...", flush=True)
    client = CoreMLClient(args.worker, args.models)
    try:
        client.transcribe(clips[0]["pcm"])
        for repeat in range(args.repeats):
            for clip in clips:
                try:
                    response = client.transcribe(clip["pcm"])
                    trial = {"status": "ok", **b.score(clip["reference"], response["text"]),
                             "total_s": response["total_s"], "after_audio_s": response["total_s"],
                             "model_s": response["model_s"]}
                except Exception as error:
                    trial = {"status": "error", "error": type(error).__name__}
                trial.update({"clip": clip["id"], "repeat": repeat, "corpus": clip["corpus"],
                              "category": clip["category"], "audio_s": clip["audio_s"],
                              "audio_sha256": clip["audio_sha256"]})
                report["trials"].append(trial)
                count = len(report["trials"])
                if count % 25 == 0 or len(clips) < 25:
                    print("Core ML scored " + str(count) + " calls", flush=True)
        report["summary"] = b.summarize(report["trials"])
        report["corpora"] = {str(corpus): b.summarize([t for t in report["trials"] if t["corpus"] == corpus])
                             for corpus in range(len(args.manifest))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("SUMMARY " + json.dumps(report["summary"]), flush=True)
        return int(bool(report["summary"]["failures"]))
    finally:
        client.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        print("Core ML benchmark failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(2)
