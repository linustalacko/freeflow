#!/usr/bin/env python3
"""Deterministic dictation cleanup — the Python twin of Sources/TranscriptFastPath.swift
plus Sources/SpokenFormatting.swift and Sources/DictationProfile.swift.

Keep the two in sync. Used by router.py to clean pre-cached chunks and the tail
at commit without the LLM whenever the text needs no *interpretation*:
fillers are removed, dictated commas/question marks converted, dictated lists
and paragraph breaks laid out, capitalisation repaired. Returns None (→ use the
model) for self-corrections/restarts, ambiguous formatting words, greetings in a
mail app, brackets, repeated sentences, vocabulary mismatches, or long text.

The destination profile is recovered from the "Destination:" line the app
appends to its default system prompt, so the router formats for the same target
the app's own fast path would have.
"""
import re

FILLERS = {"um", "uh", "uhm", "umm", "uhh", "erm", "er", "ah", "eh", "hmm", "hm", "mhm", "mm"}
# List/break/quote words are absent on purpose: spoken_format() resolves those
# when unambiguous and reports ambiguity otherwise (same contract as the Swift).
BAIL_PHRASES = [
    "no actually", "actually no", "i mean", "scratch that", "never mind", "nevermind",
    "wait", "sorry", "not that", "let me rephrase", "start over", "strike that", "correction",
    "open paren", "close paren", "open bracket", "close bracket",
    "colon", "semicolon", "underscore", "dash dash", "all caps", "slash", "hyphen",
    "period",
]
GREETINGS = ["hi", "hey", "hello", "dear", "good morning", "good afternoon", "good evening"]
PUNCT_WORDS = [(r"\s*\bcomma\b", ","), (r"\s*\bfull stop\b", "."), (r"\s*\bquestion mark\b", "?"),
               (r"\s*\bexclamation (?:mark|point)\b", "!")]
_ENDERS = ".!?"
_MARKS = ".,!?;:"

# --- profiles (twin of DictationProfile.swift) ---------------------------------
# allows_markdown, wants_email_layout, is_casual, preserves_verbatim
PROFILES = {
    "email":       dict(allows_markdown=False, wants_email_layout=True,  is_casual=False, preserves_verbatim=False),
    "chat":        dict(allows_markdown=True,  wants_email_layout=False, is_casual=True,  preserves_verbatim=False),
    "code":        dict(allows_markdown=True,  wants_email_layout=False, is_casual=True,  preserves_verbatim=True),
    "terminal":    dict(allows_markdown=False, wants_email_layout=False, is_casual=True,  preserves_verbatim=True),
    "document":    dict(allows_markdown=True,  wants_email_layout=False, is_casual=False, preserves_verbatim=False),
    "searchField": dict(allows_markdown=False, wants_email_layout=False, is_casual=True,  preserves_verbatim=True),
    # Matches the Swift `.unknown` profile: behaves as it did before profiles existed.
    "unknown":     dict(allows_markdown=True,  wants_email_layout=True,  is_casual=False, preserves_verbatim=False),
}

# Distinctive fragment of each style hint the app appends, for recovering the target.
_HINT_MARKERS = [
    ("email", "going into an email"),
    ("chat", "going into a chat message"),
    ("code", "going into a code editor"),
    ("terminal", "going into a terminal"),
    ("document", "going into a document"),
    ("searchField", "going into a search"),
]


def profile_for(target):
    return PROFILES.get(target or "unknown", PROFILES["unknown"])


def profile_from_system_prompt(system_prompt):
    """Recover the destination profile from the app's appended `Destination:` line."""
    low = (system_prompt or "").lower()
    for target, marker in _HINT_MARKERS:
        if marker in low:
            return profile_for(target)
    return profile_for("unknown")


def bullet_marker(profile):
    return "- " if profile["allows_markdown"] else "• "


# --- spoken formatting (twin of SpokenFormatting.swift) ------------------------
LIST_INTROS = ["bullet list", "bulleted list", "bullet points", "in bullet points",
               "as bullet points", "as bullets", "as a list", "in a list"]
NUMBERED_INTROS = ["numbered list", "number list", "as a numbered list", "ordered list"]
# Longest first so "next bullet point" wins over "bullet point" at the same offset.
ITEM_MARKERS = ["next bullet point", "next bullet", "new bullet point", "new bullet",
                "bullet point", "next point", "bullet"]
NOUN_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "each", "every",
                    "another", "one", "of", "any", "some", "no", "my", "your", "our", "their"}

_AMBIGUOUS = object()  # sentinel: formatting word present but its reading is unclear


def _cores(text):
    return [w.lower().strip(_MARKS + "’'\"") for w in text.split()]


def _find_seq(needle, hay):
    if not needle or len(hay) < len(needle):
        return None
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return None


def _marker_hits(cores):
    hits, i = [], 0
    while i < len(cores):
        for m in ITEM_MARKERS:
            mw = m.split()
            if cores[i:i + len(mw)] == mw:
                hits.append((i, len(mw)))
                i += len(mw)
                break
        else:
            i += 1
    return hits


def contains_any_formatting_word(text):
    padded = " " + " ".join(_cores(text)) + " "
    every = ITEM_MARKERS + LIST_INTROS + NUMBERED_INTROS + [
        "new line", "newline", "new paragraph", "quote", "unquote", "end quote"]
    return any((" " + p + " ") in padded for p in every)


def _quote_rewrite(text):
    cores, words = _cores(text), text.split()
    if len(cores) != len(words):
        return None
    open_i = cores.index("quote") if "quote" in cores else None
    if "unquote" in cores:
        close_i, close_len = cores.index("unquote"), 1
    else:
        close_i, close_len = None, 0
        for i in range(1, len(cores)):
            if cores[i] == "quote" and cores[i - 1] == "end":
                close_i, close_len = i, 2
                break
    if open_i is None:
        return _AMBIGUOUS if close_i is not None else None
    if close_i is None or close_i <= open_i + 1:
        return _AMBIGUOUS  # "I need a quote for that" — half a pair is prose
    close_start = close_i - 1 if close_len == 2 else close_i
    if close_start <= open_i:
        return _AMBIGUOUS
    quoted = " ".join(words[open_i + 1:close_start]).strip().rstrip(",")
    if not quoted:
        return _AMBIGUOUS
    return " ".join(words[:open_i] + ["“" + quoted + "”"] + words[close_i + 1:])


def _break_rewrite(text):
    cores, words = _cores(text), text.split()
    if len(cores) != len(words):
        return None
    phrases = [(["new", "paragraph"], "\n\n"), (["new", "line"], "\n"), (["newline"], "\n")]
    out, i, did = [], 0, False
    while i < len(cores):
        for pw, rep in phrases:
            if cores[i:i + len(pw)] == pw:
                before = cores[i - 1] if i > 0 else ""
                after = cores[i + len(pw)] if i + len(pw) < len(cores) else ""
                # "a new line of business" is a noun phrase, not a line break.
                if before in NOUN_DETERMINERS or after == "of":
                    return _AMBIGUOUS
                out.append(rep)
                i += len(pw)
                did = True
                break
        else:
            out.append(words[i])
            i += 1
    if not did:
        return None
    rendered = ""
    for tok in out:
        if tok in ("\n", "\n\n"):
            rendered = rendered.rstrip().rstrip(",;:")
            rendered += tok
        else:
            if rendered and not rendered.endswith("\n"):
                rendered += " "
            rendered += tok
    rendered = rendered.strip()
    if not rendered:
        return _AMBIGUOUS
    return _capitalize_after_breaks(rendered)


def _capitalize_after_breaks(text):
    out, at_start = [], True
    for ch in text:
        if at_start and ch.isalpha():
            out.append(ch.upper())
            at_start = False
        else:
            out.append(ch)
            if not ch.isspace():
                at_start = False
        if ch == "\n":
            at_start = True
    return "".join(out)


def _list_rewrite(text, profile):
    cores, words = _cores(text), text.split()
    padded = " " + " ".join(cores) + " "
    numbered = next((p for p in NUMBERED_INTROS if (" " + p + " ") in padded), None)
    bullet_intro = next((p for p in LIST_INTROS if (" " + p + " ") in padded), None)

    hits = _marker_hits(cores)
    intro = numbered or bullet_intro
    if intro:
        # An intro's own words are not items ("bullet points" contains "bullet").
        iw = intro.split()
        istart = _find_seq(iw, cores)
        if istart is not None:
            iend = istart + len(iw)
            hits = [h for h in hits if h[0] >= iend or h[0] < istart]

    # A lone marker with no intro is someone talking *about* a bullet — prose.
    if not (numbered or bullet_intro or len(hits) >= 2):
        return None

    if not hits:
        istart = _find_seq(intro.split(), cores)
        if istart is None:
            return _AMBIGUOUS
        tail = " ".join(words[istart + len(intro.split()):]).strip(" :,-–")
        if not tail:
            return _AMBIGUOUS
        items = []
        for part in tail.split(","):
            part = part.strip()
            m = re.split(r"\s+and\s+", part, maxsplit=1, flags=re.I)
            items.extend(x.strip() for x in m)
        items = [x for x in items if x]
    else:
        items = []
        for n, (start, length) in enumerate(hits):
            s = start + length
            e = hits[n + 1][0] if n + 1 < len(hits) else len(words)
            items.append(" ".join(words[s:e]))

    cleaned = []
    for it in items:
        v = it.strip().rstrip(",;:").strip()
        if v:
            cleaned.append(v[0].upper() + v[1:])
    if len(cleaned) < 2:
        return _AMBIGUOUS

    if numbered:
        lines = ["%d. %s" % (n + 1, v) for n, v in enumerate(cleaned)]
    else:
        lines = [bullet_marker(profile) + v for v in cleaned]

    cut = None
    if intro:
        cut = _find_seq(intro.split(), cores)
    if hits:
        cut = hits[0][0] if cut is None else min(cut, hits[0][0])
    lead = ""
    if cut:
        lead = " ".join(words[:cut]).strip().rstrip(",;:-").strip()
        if lead:
            lead = lead[0].upper() + lead[1:]
            if lead[-1] not in ".!?:":
                lead += ":"
    body = "\n".join(lines)
    return (lead + "\n" + body) if lead else body


def spoken_format(text, profile):
    """Returns (text, confident, produced_list). Mirrors SpokenFormatting.apply."""
    if profile["preserves_verbatim"]:
        return (text, not contains_any_formatting_word(text), False)
    produced_list = False
    for fn, is_list in ((_quote_rewrite, False), (_break_rewrite, False), (_list_rewrite, True)):
        r = fn(text, profile) if is_list else fn(text)
        if r is _AMBIGUOUS:
            return (text, False, False)
        if r is not None:
            text = r
            produced_list = produced_list or is_list
    return (text, True, produced_list)


# --- deterministic cleanup (twin of TranscriptFastPath.swift) ------------------
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


def det_clean(raw, max_words=60, vocabulary="", profile=None):
    profile = profile or profile_for("unknown")
    text = " ".join(l.strip() for l in (raw or "").splitlines() if l.strip()).strip()
    if not text or max_words <= 0:
        return None
    if any(c in text for c in "([{"):
        return None
    words = text.split()
    if len(words) > max_words:
        return None
    cores = [_core(w) for w in words]
    joined = " " + " ".join(cores) + " "
    if any((" " + p + " ") in joined for p in BAIL_PHRASES):
        return None
    if profile["wants_email_layout"] and any(joined.startswith(" " + g + " ") for g in GREETINGS):
        return None
    if _has_false_start(words) or _has_repeated_sentence(text):
        return None
    if _vocab_mismatch(text, joined, vocabulary):
        return None

    text, confident, produced_list = spoken_format(text, profile)
    if not confident:
        return None
    # A straight double quote here came from the STT, not from spoken_format
    # (which emits curly quotes) — let the model handle those.
    if '"' in text:
        return None

    for pat, rep in PUNCT_WORDS:
        text = re.sub(pat, rep, text, flags=re.I)
    lines = []
    for line in text.split("\n"):
        out = []
        for t in line.split():
            if _core(t) in FILLERS:
                if t[-1] in _ENDERS and out and out[-1][-1] not in _ENDERS:
                    out[-1] += t[-1]
                continue
            out.append(t)
        lines.append(" ".join(out))
    text = "\n".join(lines)
    # Horizontal whitespace only: \s would eat the newlines spoken_format made.
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    text = re.sub(r"[,;:]([.!?])", r"\1", text)
    text = re.sub(r"([.!?])[,;:]", r"\1", text)
    text = re.sub(r"^[,;:.][ \t]*", "", text, flags=re.M)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return None

    if not profile["preserves_verbatim"]:
        chars, pending, saw_ender = [], True, False
        for ch in text:
            if ch.isspace():
                # A line break always starts something new, full stop or not.
                if saw_ender or ch == "\n":
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

    if profile["preserves_verbatim"] or produced_list:
        return text
    if text[-1] in _ENDERS:
        return text
    if profile["is_casual"] and not any(c in text for c in _ENDERS):
        return text
    return text + "."
