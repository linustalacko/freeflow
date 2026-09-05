#!/usr/bin/env python3
"""Raw ASR benchmark. No app history, settings, .env, or provider discovery.

Python standard library + ffmpeg; streaming compiles the app's Swift client.
See README.md for corpus preparation, engine configuration, and metric limits.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}\Z")


def normalize(text):
    """NFKC, casefold, apostrophes removed; other punctuation separates words.

    Deliberately does not equate 'twenty two' and '22', remove fillers, or clean
    grammar. Raw CER is also reported to expose formatting differences.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("'", "").replace("’", "")
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def score(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    return {
        "word_errors": distance(ref.split(), hyp.split()), "words": len(ref.split()),
        "char_errors": distance(ref, hyp), "chars": len(ref),
        "raw_char_errors": distance(reference, hypothesis), "raw_chars": len(reference),
        "exact": ref == hyp,
        "silence_words": len(hyp.split()) if not ref else 0,
    }


def percentile(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def summarize(rows):
    successful = [r for r in rows if r["status"] == "ok"]
    speech = [r for r in successful if r["words"]]
    silence = [r for r in successful if not r["words"]]
    def rate(errors, count):
        denominator = sum(r[count] for r in speech)
        return sum(r[errors] for r in speech) / denominator if denominator else None
    total_audio = sum(r["audio_s"] for r in successful)
    return {
        "attempts": len(rows), "successes": len(successful),
        "failures": len(rows) - len(successful),
        "words": sum(r["words"] for r in speech),
        "wer": rate("word_errors", "words"), "cer": rate("char_errors", "chars"),
        "raw_cer": rate("raw_char_errors", "raw_chars"),
        "exact_rate": sum(r["exact"] for r in speech) / len(speech) if speech else None,
        "silence_trials": len(silence),
        "silence_false_positives": sum(r["silence_words"] > 0 for r in silence),
        "silence_inserted_words": sum(r["silence_words"] for r in silence),
        "after_audio_p50_s": percentile([r["after_audio_s"] for r in successful], .5),
        "after_audio_p95_s": percentile([r["after_audio_s"] for r in successful], .95),
        "total_p50_s": percentile([r["total_s"] for r in successful], .5),
        "total_p95_s": percentile([r["total_s"] for r in successful], .95),
        "wall_rtf": sum(r["total_s"] for r in successful) / total_audio if total_audio else None,
        "failure_codes": {code: sum(r.get("error") == code for r in rows)
                          for code in sorted({r["error"] for r in rows if "error" in r})},
    }


def safe_identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("IDs and categories must be short alphanumeric labels (with ._-)")
    return value


def read_manifest(path):
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = safe_identifier(row["id"])
        if identifier in seen:
            raise ValueError("duplicate corpus ID")
        seen.add(identifier)
        safe_identifier(row.get("category", "speech"))
        if not isinstance(row["text"], str):
            raise ValueError("reference text must be a string")
        audio = (path.parent / row["audio"]).resolve()
        if not audio.is_file():
            raise ValueError("manifest audio is missing")
        rows.append({**row, "audio": audio})
    if not rows:
        raise ValueError("empty corpus")
    return rows


def validate_engine(engine, allow_remote):
    safe_identifier(engine["id"])
    if engine["kind"] not in {"multipart", "freeflow-stream", "deepgram"}:
        raise ValueError("unknown engine kind")
    url = urllib.parse.urlsplit(engine["url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("engine URL must be HTTP(S), without embedded credentials")
    if url.query or url.fragment:
        raise ValueError("put API parameters in the engine fields, not the URL")
    local = url.hostname in {"127.0.0.1", "::1"}
    if not local and not allow_remote:
        raise ValueError("remote audio upload requires --allow-remote; loopback IPs work without it")
    if not local and url.scheme != "https":
        raise ValueError("remote engines require HTTPS")
    if engine["kind"] == "freeflow-stream" and not local:
        raise ValueError("Swift replay is restricted to loopback; use batch for hosted providers")
    if not isinstance(engine["model"], str) or not engine["model"].strip():
        raise ValueError("each engine needs an explicit model label")
    key_name = engine.get("api_key_env")
    if key_name and not os.environ.get(key_name):
        raise ValueError("an engine's required API key environment variable is unset")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward audio or Authorization to an unexpected destination.
        return None


def multipart(audio, model, language):
    boundary = "freeflow-" + uuid.uuid4().hex
    body = bytearray()
    for key, value in {"model": model, "language": language, "response_format": "json"}.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    body.extend(audio)
    body.extend(f'\r\n--{boundary}--\r\n'.encode())
    return bytes(body), "multipart/form-data; boundary=" + boundary


def batch(engine, audio, language, timeout):
    # File reads/resampling are outside the timer. Multipart encoding, upload,
    # queueing, inference, download, and JSON parsing are inside it. No retries.
    started = time.perf_counter()
    key = os.environ.get(engine.get("api_key_env", ""), "")
    url = engine["url"]
    if engine["kind"] == "deepgram":
        url += "?" + urllib.parse.urlencode({"model": engine["model"], "language": language,
                                             "smart_format": "false", "punctuate": "false",
                                             "filler_words": "true"})
        body, content_type = audio, "audio/wav"
    else:
        body, content_type = multipart(audio, engine["model"], language)
    headers = {"Content-Type": content_type, "User-Agent": "FreeFlow-ASR-Benchmark/1"}
    if key:
        headers["Authorization"] = ("Token " if engine["kind"] == "deepgram" else "Bearer ") + key
    request = urllib.request.Request(url, data=body, headers=headers)
    # Explicit endpoints only; ignore inherited proxy settings, including for localhost.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        payload = json.load(response)
    if engine["kind"] == "deepgram":
        text = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
    else:
        text = payload["text"]
    if not isinstance(text, str):
        raise ValueError("invalid transcription response")
    elapsed = time.perf_counter() - started
    return {"text": text, "total_s": elapsed, "after_audio_s": elapsed}


def command(args, **kwargs):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kwargs)


def prepare(rows, directory):
    """Canonical 16 kHz PCM WAV for every batch engine; 24 kHz PCM for Swift.

    Conversion runs before timing. Both files derive from the same canonical WAV.
    """
    prepared = []
    for index, row in enumerate(rows):
        wav_path, pcm_path = directory / f"{index}.wav", directory / f"{index}.pcm"
        command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(row["audio"]),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)])
        with wave.open(str(wav_path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
        if duration <= 0 or duration > 600:
            raise ValueError("clips must contain between zero and 600 seconds of audio")
        command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav_path),
                 "-ac", "1", "-ar", "24000", "-f", "s16le", str(pcm_path)])
        audio = wav_path.read_bytes()
        prepared.append({**row, "wav": audio, "pcm": pcm_path, "audio_s": duration,
                         "audio_sha256": hashlib.sha256(audio).hexdigest()})
    return prepared


def compile_replay(directory):
    output = directory / "replay"
    command(["swiftc", "-O", "-parse-as-library", "-warnings-as-errors",
             str(REPO / "Sources/RealtimeTranscriptState.swift"),
             str(REPO / "Sources/RealtimeTranscriptionService.swift"),
             str(HERE / "Replay.swift"), "-o", str(output)])
    return output


def transcribe(engine, clip, language, timeout, replay):
    started = time.perf_counter()
    try:
        if engine["kind"] == "freeflow-stream":
            request = {"pcmPath": str(clip["pcm"]), "baseURL": engine["url"],
                       "model": engine["model"], "language": language, "timeout": timeout,
                       "apiKey": os.environ.get(engine.get("api_key_env", ""), "")}
            result = command([str(replay)], input=json.dumps(request).encode(),
                             timeout=clip["audio_s"] + timeout + 10)
            result = json.loads(result.stdout)
            if "error" in result:
                return {"status": "error", "error": result["error"],
                        "attempt_s": time.perf_counter() - started}
            if abs(result["replay_s"] - clip["audio_s"]) > max(.1, clip["audio_s"] * .01):
                return {"status": "error", "error": "replay_timing_drift",
                        "attempt_s": time.perf_counter() - started}
        else:
            result = batch(engine, clip["wav"], language, timeout)
        text = result.pop("text")
        return {"status": "ok", **result, **score(clip["text"], text)}
    except urllib.error.HTTPError as error:
        code = "http_" + str(error.code)
    except (TimeoutError, subprocess.TimeoutExpired):
        code = "timeout"
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        code = "connection_error"
    except (ValueError, KeyError, IndexError, TypeError):
        code = "invalid_response"
    except subprocess.CalledProcessError:
        code = "replay_process_failed"
    return {"status": "error", "error": code, "attempt_s": time.perf_counter() - started}


def write_report(path, report):
    # A report stores numeric scores, hashes, and explicit model labels only.
    # No hypotheses, reference text, audio, API keys, URLs, or filesystem paths.
    with path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    lines = ["# Raw transcription benchmark", "",
             "Synthetic audio is a smoke test, not a real-world accuracy ranking.", "",
             "| Engine | Trials | Failures | WER | After audio p50 | After audio p95 | Silence false positives | Wall RTF |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, scale=1, suffix=""):
        return "—" if value is None else f"{value * scale:.2f}{suffix}"
    for engine in report["engines"]:
        summary = engine["summary"]
        lines.append(f'| {engine["id"]} | {summary["attempts"]} | {summary["failures"]} | '
                     f'{fmt(summary["wer"], 100, "%")} | '
                     f'{fmt(summary["after_audio_p50_s"], 1000, " ms")} | '
                     f'{fmt(summary["after_audio_p95_s"], 1000, " ms")} | '
                     f'{summary["silence_false_positives"]}/{summary["silence_trials"]} | '
                     f'{fmt(summary["wall_rtf"])} |')
    lines += ["", "WER is corpus-weighted on successful speech trials; failures and silence are separate.",
              "Streaming wall RTF includes paced audio playback; batch wall RTF is request time / audio time.",
              "After audio measures commit-to-final for streaming, and request-to-final for batch.",
              "These are warm service timings, excluding model load, file conversion, cleanup, and paste.",
              "Small samples do not establish reliable p95 latency. See JSON for categories and warmups."]
    with path.with_suffix(".md").open("x") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines[4:4 + len(report["engines"]) + 2]))


def run(args):
    if args.repeats < 1 or args.warmups < 1 or not 0 < args.timeout <= 300:
        raise ValueError("repeats/warmups must be positive; timeout must be in (0, 300]")
    output = args.output.resolve()
    if output.suffix != ".json" or output.exists() or output.with_suffix(".md").exists():
        raise ValueError("output must be a new .json path, with no existing matching .md")
    rows = read_manifest(args.manifest.resolve())
    engines = json.loads(args.config.read_text())["engines"]
    ids = [engine["id"] for engine in engines]
    if len(ids) != len(set(ids)) or (args.engine and set(args.engine) - set(ids)):
        raise ValueError("duplicate or unknown engine ID")
    engines = [e for e in engines if not args.engine or e["id"] in args.engine]
    if not engines:
        raise ValueError("no engines selected")
    for engine in engines:
        validate_engine(engine, args.allow_remote)
    report = {"schema": 1, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": {"system": platform.system(), "release": platform.release(),
                           "machine": platform.machine(), "python": platform.python_version()},
              "repeats": args.repeats, "warmups_per_engine": args.warmups, "timeout_s": args.timeout,
              "seed": args.seed, "language": args.language, "clip_count": len(rows),
              "normalization": "NFKC-casefold-alphanumeric-apostrophe-removal-v1",
              "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [HERE / "benchmark.py", HERE / "Replay.swift",
                                             REPO / "Sources/RealtimeTranscriptionService.swift",
                                             REPO / "Sources/RealtimeTranscriptState.swift"]},
              "engines": []}
    with tempfile.TemporaryDirectory(prefix="freeflow-asr-") as tmp:
        clips = prepare(rows, Path(tmp))
        report["audio_seconds_once"] = sum(c["audio_s"] for c in clips)
        digest = hashlib.sha256()
        for clip in clips:
            digest.update(json.dumps([clip["id"], clip["audio_sha256"], clip["text"],
                                      clip.get("category", "speech")], ensure_ascii=False).encode())
        report["corpus_sha256"] = digest.hexdigest()
        replay = compile_replay(Path(tmp)) if any(e["kind"] == "freeflow-stream" for e in engines) else None
        results = {e["id"]: [] for e in engines}
        warmups = {e["id"]: [] for e in engines}
        # Serial execution avoids GPU contention. Shuffle engine order per clip
        # and clip order per repetition, deterministically, to balance drift.
        rng = random.Random(args.seed)
        for engine in engines:
            print(f'Warming {engine["id"]}', flush=True)
            warm_clip = next((c for c in clips if normalize(c["text"])), clips[0])
            for _ in range(args.warmups):
                warmups[engine["id"]].append(transcribe(engine, warm_clip, args.language, args.timeout, replay))
        for repeat in range(args.repeats):
            order = list(clips)
            rng.shuffle(order)
            for clip in order:
                engine_order = list(engines)
                rng.shuffle(engine_order)
                for engine in engine_order:
                    result = transcribe(engine, clip, args.language, args.timeout, replay)
                    result.update({"clip": clip["id"], "repeat": repeat + 1,
                                   "category": clip.get("category", "speech"),
                                   "audio_s": clip["audio_s"], "audio_sha256": clip["audio_sha256"]})
                    results[engine["id"]].append(result)
                    print(f'{repeat + 1}/{args.repeats} {clip["id"]} {engine["id"]}: {result["status"]}', flush=True)
        for engine in engines:
            trials = results[engine["id"]]
            report["engines"].append({"id": engine["id"], "kind": engine["kind"], "model": engine["model"],
                "summary": summarize(trials), "warmups": warmups[engine["id"]], "trials": trials,
                "categories": {category: summarize([r for r in trials if r["category"] == category])
                               for category in sorted({r["category"] for r in trials})}})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, report)
    return 1 if any(e["summary"]["failures"] for e in report["engines"]) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "engines.json")
    parser.add_argument("--engine", action="append", help="run only these engine IDs (repeatable)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--language", default="en")
    parser.add_argument("--allow-remote", action="store_true", help="explicitly allow corpus audio uploads to configured HTTPS engines")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        # Avoid echoing provider payloads, corpus text, or local paths.
        print("Benchmark setup failed: " + (str(error) if type(error) is ValueError else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
