#!/usr/bin/env python3
"""Deterministic dictation cleanup — the Python twin of Sources/TranscriptFastPath.swift.

Keep the two in sync. Used by router.py to clean pre-cached chunks and the tail
at commit without the LLM whenever the text needs no *interpretation*:
fillers are removed, dictated commas/question marks converted, capitalisation
repaired. Returns None (→ use the model) for self-corrections/restarts, dictated
formatting, greetings (email layout), quotes/brackets, repeated sentences,
vocabulary mismatches, or long text.
"""
import re

FILLERS = {"um", "uh", "uhm", "umm", "uhh", "erm", "er", "ah", "eh", "hmm", "hm", "mhm", "mm"}
BAIL_PHRASES = [
    "no actually", "actually no", "i mean", "scratch that", "never mind", "nevermind",
    "wait", "sorry", "not that", "let me rephrase", "start over", "strike that", "correction",
    "new paragraph", "new line", "newline", "bullet", "bullets", "bullet point", "bullet points",
    "numbered list", "bullet list", "open paren", "close paren", "open bracket", "close bracket",
    "quote", "unquote", "colon", "semicolon", "underscore", "dash dash", "all caps", "slash", "hyphen",
    "period",
]
GREETINGS = ["hi", "hey", "hello", "dear", "good morning", "good afternoon", "good evening"]
PUNCT_WORDS = [(r"\s*\bcomma\b", ","), (r"\s*\bfull stop\b", "."), (r"\s*\bquestion mark\b", "?"),
               (r"\s*\bexclamation (?:mark|point)\b", "!")]
_ENDERS = ".!?"
_MARKS = ".,!?;:"


def _core(t):
    return t.strip(_MARKS).lower()


def _has_false_start(words):
    cores = [_core(w) for w in words]
    for i in range(len(cores) - 1):
        if cores[i] and cores[i] == cores[i + 1] and words[i][-1] not in _MARKS:
            return True
    for n in (2, 3):
        for i in range(len(cores) - n):
            g = cores[i:i + n]
            if not all(g):
                continue
            for j in range(i + 1, min(len(cores) - n + 1, i + n + 2)):
                if cores[j:j + n] == g:
                    return True
    return False


def _has_repeated_sentence(text):
    sents = [_core(s).strip() for s in re.split(r"[.!?]", text)]
    sents = [s for s in sents if s]
    return any(sents[i] == sents[i + 1] for i in range(len(sents) - 1))


def _vocab_mismatch(text, joined, vocabulary):
    padded = " " + text + " "
    for term in [t.strip() for t in re.split(r"[,\n]", vocabulary or "") if t.strip()]:
        if (" " + term.lower() + " ") in joined and not any((" " + term + e) in padded for e in (" ", ".", ",", "?", "!")):
            return True
    return False


def det_clean(raw, max_words=60, vocabulary=""):
    text = " ".join(l.strip() for l in (raw or "").splitlines() if l.strip()).strip()
    if not text or max_words <= 0:
        return None
    if any(c in text for c in '"([{'):
        return None
    words = text.split()
    if len(words) > max_words:
        return None
    cores = [_core(w) for w in words]
    joined = " " + " ".join(cores) + " "
    if any((" " + p + " ") in joined for p in BAIL_PHRASES):
        return None
    if any(joined.startswith(" " + g + " ") for g in GREETINGS):
        return None
    if _has_false_start(words) or _has_repeated_sentence(text):
        return None
    if _vocab_mismatch(text, joined, vocabulary):
        return None
    for pat, rep in PUNCT_WORDS:
        text = re.sub(pat, rep, text, flags=re.I)
    out = []
    for t in text.split():
        if _core(t) in FILLERS:
            if t[-1] in _ENDERS and out and out[-1][-1] not in _ENDERS:
                out[-1] += t[-1]
            continue
        out.append(t)
    text = " ".join(out)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"[,;:]([.!?])", r"\1", text)
    text = re.sub(r"([.!?])[,;:]", r"\1", text)
    text = re.sub(r"^[,;:.]\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        return None
    # capitalise first letter and after sentence end + whitespace
    chars, pending, saw_ender = [], True, False
    for ch in text:
        if ch.isspace():
            if saw_ender:
                pending = True
            chars.append(ch)
            continue
        if pending and ch.isalpha():
            chars.append(ch.upper()); pending = False
        else:
            chars.append(ch)
            if ch.isalnum():
                pending = False
        saw_ender = ch in _ENDERS
    text = "".join(chars)
    if text[-1] not in _ENDERS:
        text += "."
    return text
