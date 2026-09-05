#!/usr/bin/env python3
"""Local model experiments on explicit corpora; emits scores, never transcripts.

This measures complete-clip model processing, not streaming or app latency.
Run with the local STT virtualenv. Nothing is uploaded or read from app history.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import benchmark as b
from experiment_config import environment, validate_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists() or (args.limit is not None and args.limit < 1):
        parser.error("output must be new and limit positive")
    matrix = validate_matrix(json.loads(args.matrix.read_text()), model_only=True)
    if matrix[0].get("env"):
        parser.error("first variant must be an unmodified baseline")
    parent_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment(parent_env, {}))
    logging.disable(logging.CRITICAL)
    import numpy as np
    from experimental_server import ExperimentalTranscriber

    clips = []
    for index, manifest in enumerate(args.manifest):
        for row in b.read_manifest(manifest)[:args.limit]:
            raw = b.command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(row["audio"]),
                             "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]).stdout
            clips.append({"id": row["id"], "corpus": index, "reference": row["text"],
                          "audio": np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768,
                          "audio_sha256": hashlib.sha256(raw).hexdigest()})
    baseline = {}
    report = {"kind": "complete-clip-model-experiments", "schema": 1,
              "clip_count": len(clips), "machine": platform.machine(),
              "platform": platform.system(), "model": "mlx-community/parakeet-tdt-0.6b-v2",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [b.HERE / "experimental_server.py", b.HERE / "experimental_decoders.py"]},
              "variants": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in matrix:
        overrides = variant.get("env", {})
        os.environ.clear()
        os.environ.update(environment(parent_env, overrides))
        label = b.safe_identifier(variant["id"])
        print("Starting " + label, flush=True)
        transcriber = ExperimentalTranscriber(report["model"])
        transcriber.transcribe(clips[0]["audio"])
        trials = []
        for index, clip in enumerate(clips):
            started = time.perf_counter()
            try:
                transcript = transcriber.transcribe(clip["audio"])
                elapsed = time.perf_counter() - started
                key = (clip["corpus"], clip["id"])
                if not report["variants"]:
                    baseline[key] = transcript
                trial = {"status": "ok", **b.score(clip["reference"], transcript),
                         "baseline_exact": transcript == baseline.get(key),
                         "total_s": elapsed, "after_audio_s": elapsed}
            except Exception as error:
                trial = {"status": "error", "error": type(error).__name__}
            trial.update({"clip": clip["id"], "corpus": clip["corpus"],
                          "audio_sha256": clip["audio_sha256"], "audio_s": len(clip["audio"]) / 16000})
            trials.append(trial)
            if (index + 1) % 50 == 0:
                print(label + ": " + str(index + 1) + " clips scored", flush=True)
        entry = {"id": label, "settings": overrides, "trials": trials,
                 "summary": b.summarize(trials),
                 "corpora": {str(index): b.summarize([r for r in trials if r["corpus"] == index])
                             for index in range(len(args.manifest))},
                 "baseline_exact_count": sum(r.get("baseline_exact", False) for r in trials)}
        print("SUMMARY " + label + " " + json.dumps(entry["summary"]) +
              " exact=" + str(entry["baseline_exact_count"]), flush=True)
        report["variants"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        mx = transcriber.mx
        del transcriber
        import gc
        gc.collect()
        mx.clear_cache()
    return int(any(v["summary"]["failures"] for v in report["variants"]))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print("Model experiment failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(2)
