#!/usr/bin/env python3
"""Generate synthetic smoke audio, or import an explicitly supplied LibriSpeech split."""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import wave

from benchmark import command, safe_identifier


SENTENCES = [
    ("short", "Please move the blue folder to the desktop.", 180),
    ("dictation", "I think we should send the revised proposal tomorrow morning, after everyone has reviewed the final paragraph.", 180),
    ("fast", "Before you close the window, save the draft and check that the attachment is included in the message.", 240),
    ("names-numbers", "Marisol and Quentin will meet at twenty two Oak Street on Thursday, September seventeenth.", 165),
    ("disfluency", "Um, I wanted to, uh, change the title. Actually, keep the title and replace the second sentence.", 180),
    ("long", "The small museum stands beside a quiet river. Every morning, the caretaker opens the wooden doors and checks the lights. Visitors usually begin with the paintings upstairs, then walk through the garden behind the building. The new exhibition explains how local artists made their tools and mixed their colors. Before leaving, please return the guide to the front desk and make sure the garden gate is closed.", 190),
]


def synthetic(output, voices):
    rows = []
    for voice_index, voice in enumerate(voices):
        for index, (category, text, rate) in enumerate(SENTENCES):
            identifier = f"tts-{voice_index + 1}-{index + 1}"
            with tempfile.TemporaryDirectory(prefix="freeflow-tts-") as temp:
                aiff = Path(temp) / "speech.aiff"
                command(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text])
                command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(aiff),
                         "-af", "adelay=200,apad=pad_dur=0.3", "-ac", "1", "-ar", "16000",
                         "-c:a", "pcm_s16le", str(output / f"{identifier}.wav")])
            rows.append({"id": identifier, "audio": f"{identifier}.wav", "text": text,
                         "category": category, "source": "macOS-say-synthetic",
                         "voice": voice, "words_per_minute": rate})
    for seconds in (2, 8):
        identifier = f"silence-{seconds}"
        with wave.open(str(output / f"{identifier}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(seconds * 16000 * 2))
        rows.append({"id": identifier, "audio": f"{identifier}.wav", "text": "",
                     "category": "silence", "source": "synthetic-zero-pcm"})
    return rows


def librispeech(root, output, limit, seed):
    # Round-robin across speakers rather than taking the first chapter's clips.
    speakers = {}
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            identifier, text = line.split(" ", 1)
            safe_identifier(identifier)
            audio = transcript.parent / f"{identifier}.flac"
            if not audio.is_file():
                raise ValueError("LibriSpeech transcript has no matching FLAC")
            speakers.setdefault(identifier.split("-")[0], []).append((identifier, text, audio))
    if not speakers:
        raise ValueError("no LibriSpeech transcript files found")
    rng = random.Random(seed)
    for clips in speakers.values():
        rng.shuffle(clips)
    speaker_ids = sorted(speakers)
    rng.shuffle(speaker_ids)
    selected = []
    while any(speakers.values()) and len(selected) < limit:
        for speaker in speaker_ids:
            if speakers[speaker] and len(selected) < limit:
                selected.append(speakers[speaker].pop())
    rows = []
    for identifier, text, audio in selected:
        shutil.copyfile(audio, output / f"{identifier}.flac")
        rows.append({"id": identifier, "text": text, "audio": f"{identifier}.flac",
                     "category": "read-speech", "source": "LibriSpeech-OpenSLR-12",
                     "license": "CC-BY-4.0"})
    (output / "ATTRIBUTION.txt").write_text(
        "LibriSpeech ASR corpus, Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur.\n"
        "https://www.openslr.org/12 — CC BY 4.0. Audio copied without modification.\n"
        "https://creativecommons.org/licenses/by/4.0/\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["synthetic", "librispeech"])
    parser.add_argument("--output", required=True, type=Path, help="new directory outside the tracked tree")
    parser.add_argument("--voice", action="append", help="installed macOS voice; repeatable; default Samantha")
    parser.add_argument("--source", type=Path, help="extracted LibriSpeech split directory")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.output.exists() or args.limit < 1 or (args.mode == "librispeech" and not args.source):
        parser.error("output must be new, limit positive, and LibriSpeech requires --source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".corpus-", dir=args.output.parent) as temp:
            staged = Path(temp) / "corpus"
            staged.mkdir()
            rows = (synthetic(staged, args.voice or ["Samantha"]) if args.mode == "synthetic"
                    else librispeech(args.source, staged, args.limit, args.seed))
            (staged / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            staged.rename(args.output)
        print(f"Prepared {len(rows)} clips. References and audio stay in the selected output directory.")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Corpus preparation failed ({type(error).__name__}); check tools, voice, and input.\n")


if __name__ == "__main__":
    main()
