#!/usr/bin/env python3
"""Run an explicit matrix of isolated experimental STT configurations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import platform

import benchmark as b
from run_local import free_port, wait_ready
from experiment_config import environment, validate_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("output must be new and repeats positive")
    rows = b.read_manifest(args.manifest)
    if args.clip:
        if set(args.clip) - {r["id"] for r in rows}:
            parser.error("a requested clip is missing from the manifest")
        rows = [r for r in rows if r["id"] in args.clip]
    if not rows:
        parser.error("no matching clips")
    matrix = validate_matrix(json.loads(args.matrix.read_text()))
    report = {"kind": "isolated-streaming-experiments", "schema": 1, "repeats": args.repeats,
              "clip_count": len(rows), "variants": [], "machine": platform.machine(),
              "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [b.HERE / "experimental_server.py", b.HERE / "experimental_decoders.py",
                                          b.HERE / "experimental_guarded.py", b.HERE / "experimental_alignment.py",
                                          b.HERE.parent / "stt_server.py"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="freeflow-matrix-") as temporary:
        temp = Path(temporary)
        clips = b.prepare(rows, temp)
        replay = b.compile_replay(temp)
        for variant in matrix:
            label = b.safe_identifier(variant["id"])
            port = free_port()
            overrides = variant.get("env", {})
            env = environment(os.environ, overrides, port)
            print("Starting " + label, flush=True)
            process = subprocess.Popen([str(Path.home() / ".freeflow-ft/venv/bin/python"),
                                        str(b.HERE / "experimental_server.py")], cwd=temp, env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            trials = []
            entry = {"id": label, "settings": overrides, "trials": trials,
                     "expected_trials": len(clips) * args.repeats}
            try:
                wait_ready(process, port)
                engine = {"id": label, "kind": "freeflow-stream", "model": "parakeet-v2",
                          "url": f"http://127.0.0.1:{port}/v1"}
                entry["warmup"] = b.transcribe(engine, clips[0], "en", 30, replay)
                if entry["warmup"]["status"] != "ok":
                    entry["setup_error"] = entry["warmup"]["error"]
                else:
                    for repeat in range(args.repeats):
                        order = list(clips)
                        random.Random(20260905 + repeat).shuffle(order)
                        for clip in order:
                            result = b.transcribe(engine, clip, "en", 30, replay)
                            result.update({"clip": clip["id"], "repeat": repeat,
                                           "category": clip.get("category", "speech"),
                                           "audio_s": clip["audio_s"], "audio_sha256": clip["audio_sha256"]})
                            trials.append(result)
                            print(label + " " + clip["id"] + ": " + result["status"], flush=True)
                    entry["summary"] = b.summarize(trials)
                    entry["under_50ms_successes"] = sum(r["status"] == "ok" and r["after_audio_s"] < .05 for r in trials)
                    entry["categories"] = {category: b.summarize([r for r in trials if r["category"] == category])
                                           for category in sorted({r["category"] for r in trials})}
                    print("SUMMARY " + label + " " + json.dumps(entry["summary"]), flush=True)
            except Exception as error:
                entry["setup_error"] = type(error).__name__
                print(label + " setup failed: " + type(error).__name__, flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                report["variants"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Finished; experimental servers and scratch files removed.", flush=True)
    return int(any(v.get("setup_error") or len(v["trials"]) != v["expected_trials"] or
                   any(t["status"] != "ok" for t in v["trials"]) for v in report["variants"]))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as error:
        print("Experiment setup failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(2)
