#!/usr/bin/env python3
"""
FreeFlow LLM router — sits on localhost and gives FreeFlow one stable endpoint.

  FreeFlow ──▶ this router ──▶ Groq primary model (fast, when online & under quota)
                           ├──▶ Groq fallback models (each has its OWN per-minute
                           │    token bucket, so a 429 on one ≠ a 429 on the next)
                           └──▶ local model (Ollama / mlx_lm) — free, offline, slower

Why it exists / what it fixes
-----------------------------
* Groq's free tier is per-model tokens-per-minute (8k for gpt-oss-20b) and it
  reserves `max_completion_tokens` up front, so a couple of dictations in a
  minute get 429s. We track `x-ratelimit-*` headers per model and skip a model
  we can predict will 429 — no wasted round trip — then try the next bucket.
* The app has a hard budget (`post_processing_timeout_seconds`, 20s default)
  and sends it as `X-FreeFlow-Deadline-Ms`. Every backend attempt is clamped
  to what's left of that budget, so the router never "succeeds" after the app
  has already given up and pasted the raw transcript.
* `/warm` is hit on hotkey-down. If we can predict the hosted path won't be
  usable for the next request (offline, or quota exhausted), we preload the
  local model *while the user is still speaking* so the fallback is warm.

Config (env; see launchd/com.freeflow.router.plist):
  GROQ_API_KEY            Groq key. Blank → hosted path disabled (fully local).
  GROQ_URL                default https://api.groq.com/openai/v1/chat/completions
  GROQ_FALLBACK_MODELS    comma list tried after the requested model on 429/err.
                          default: openai/gpt-oss-120b,qwen/qwen3.6-27b
  LOCAL_URL               OpenAI-compatible chat endpoint for the local model.
                          default http://127.0.0.1:11434/v1/chat/completions (Ollama)
  LOCAL_MODEL             model name sent to LOCAL_URL (default llama3.1:8b)
  LOCAL_ADAPTER_PATH      LoRA adapter dir for the mlx_lm server — REQUIRED for the
                          fine-tune to actually be used (mlx_lm ignores its own
                          --adapter-path unless the request repeats it)
  LOCAL_PROMPT_FORMAT     passthrough (default) | train — use `train` for the
                          fine-tuned mlx_lm model (it was trained on a different
                          user-turn format than the app sends; see to_train_format)
  LOCAL_KEEPALIVE_S       how long the local server keeps the model resident
                          after a request (used to decide whether to /warm). 300
  FORCE_LOCAL=1           local first, Groq chain as safety net.
  HOSTED_TIMEOUT          per-attempt timeout for Groq (s). default 8
  LOCAL_TIMEOUT           per-attempt timeout for local (s). default 120
  DEFAULT_DEADLINE_MS     used when the client sends no deadline header. 18000
  ROUTER_PORT             default 8787
"""
import http.client
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from detclean import det_clean  # deterministic cleanup twin of TranscriptFastPath.swift
except Exception:  # pragma: no cover
    def det_clean(raw, max_words=60, vocabulary=""):
        return None
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ----------------------------------------------------------------------------- config
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = os.environ.get("GROQ_API_URL") or os.environ.get("GROQ_URL") or "https://api.groq.com/openai/v1/chat/completions"
GROQ_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.6-27b").split(",") if m.strip()]
LOCAL_URL = os.environ.get("LOCAL_URL") or os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434/v1/chat/completions"
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3.1:8b")
# "passthrough": send the app's prompt as-is (generic instruct models, Ollama).
# "train": re-render into the local-setup fine-tune's training format — REQUIRED
# when LOCAL_URL is the fine-tuned mlx_lm server (see to_train_format).
LOCAL_PROMPT_FORMAT = os.environ.get("LOCAL_PROMPT_FORMAT", "passthrough").strip().lower()
# mlx_lm's server (0.31) only applies its --adapter-path when the request body
# carries "adapters": it maps the model name first and then looks the adapter up
# by the *mapped* path, which never matches — so without this every request is
# silently served by the BASE model (measured: WER 31% instead of 14%). Set it to
# the same directory as the server's --adapter-path.
LOCAL_ADAPTER_PATH = os.environ.get("LOCAL_ADAPTER_PATH", "").strip()
LOCAL_KEEPALIVE_S = float(os.environ.get("LOCAL_KEEPALIVE_S", "300"))
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "8787"))
FORCE_LOCAL = os.environ.get("FORCE_LOCAL", "").strip().lower() not in ("", "0", "false", "no")
HOSTED_TIMEOUT = float(os.environ.get("HOSTED_TIMEOUT", "8"))
LOCAL_TIMEOUT = float(os.environ.get("LOCAL_TIMEOUT", "120"))
DEFAULT_DEADLINE_MS = float(os.environ.get("DEFAULT_DEADLINE_MS", "18000"))
DEADLINE_HEADER = "X-FreeFlow-Deadline-Ms"
# Groq's WAF 403s the default python-urllib User-Agent, so we send a normal one.
UA = "FreeFlow-Router/2.0"
# Safety margin subtracted from the client's deadline: covers our own overhead
# and the app's JSON parse/paste so we hand back a usable answer, not a photo finish.
DEADLINE_MARGIN_S = 0.4
# Below this much remaining budget we don't start another backend attempt.
MIN_ATTEMPT_S = 0.5


def log(msg):
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------- rate limit tracker
class RateLimits:
    """Per-model view of Groq's sliding TPM/RPM buckets, learned from response
    headers. Enough to predict "this call will 429" without making it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}  # model -> dict(limit, remaining, seen_at, blocked_until)

    @staticmethod
    def _parse_reset(s):
        # Groq formats like "23.835s", "832ms", "1m26.4s", "2h3m".
        if not s:
            return None
        s = s.strip().lower()
        total, num = 0.0, ""
        i = 0
        while i < len(s):
            c = s[i]
            if c.isdigit() or c == ".":
                num += c
                i += 1
                continue
            unit = c
            if s[i:i + 2] == "ms":
                unit = "ms"
                i += 2
            else:
                i += 1
            if not num:
                return None
            v = float(num)
            num = ""
            total += {"h": 3600, "m": 60, "s": 1, "ms": 0.001}.get(unit, 0) * v
        if num:
            total += float(num)
        return total

    def observe(self, model, headers, status):
        h = {k.lower(): v for k, v in headers.items()}
        now = time.monotonic()
        with self._lock:
            entry = self._models.setdefault(model, {"limit": None, "remaining": None,
                                                    "seen_at": now, "blocked_until": 0.0})
            try:
                if "x-ratelimit-limit-tokens" in h:
                    entry["limit"] = int(float(h["x-ratelimit-limit-tokens"]))
                if "x-ratelimit-remaining-tokens" in h:
                    entry["remaining"] = int(float(h["x-ratelimit-remaining-tokens"]))
                    entry["seen_at"] = now
            except ValueError:
                pass
            if status == 429:
                wait = self._parse_reset(h.get("retry-after")) or self._parse_reset(
                    h.get("x-ratelimit-reset-tokens")) or 5.0
                entry["blocked_until"] = now + wait
            elif status == 200:
                entry["blocked_until"] = 0.0

    def predicted_remaining(self, model):
        with self._lock:
            entry = self._models.get(model)
            if not entry or entry["remaining"] is None or not entry["limit"]:
                return None
            # Groq's per-minute bucket refills continuously; approximate linearly.
            elapsed = time.monotonic() - entry["seen_at"]
            refill = entry["limit"] * elapsed / 60.0
            return min(entry["limit"], entry["remaining"] + refill)

    def blocked_for(self, model):
        with self._lock:
            entry = self._models.get(model)
            if not entry:
                return 0.0
            return max(0.0, entry["blocked_until"] - time.monotonic())

    def will_429(self, model, cost_tokens):
        """True when we're confident the model would reject `cost_tokens` right now."""
        if self.blocked_for(model) > 0:
            return True
        remaining = self.predicted_remaining(model)
        return remaining is not None and cost_tokens > remaining

    def snapshot(self):
        with self._lock:
            return {m: dict(e) for m, e in self._models.items()}


RATE = RateLimits()

# ------------------------------------------------------------ hosted (Groq) transport
# urllib opens a fresh TLS connection per call (~150ms to Groq). Keep one
# HTTPSConnection warm and retry once on a stale socket.
_groq_conn = None
_groq_lock = threading.Lock()
_groq_parts = urllib.parse.urlsplit(GROQ_URL)


# ---- hosted circuit breaker -------------------------------------------------
# A dead network must fail over in milliseconds. Two things defeat plain socket
# timeouts: getaddrinfo() has no timeout of its own (a DNS flap blocked a call
# for 7.8s in production), and every request re-discovers the outage. So: DNS is
# resolved in a helper thread with a hard wait, and any connect/DNS/timeout
# failure opens the breaker — hosted attempts are skipped for HOSTED_COOLDOWN_S
# (the /warm probe re-checks reachability without a user waiting).
HOSTED_COOLDOWN_S = float(os.environ.get("HOSTED_COOLDOWN_S", "20"))
DNS_TIMEOUT_S = float(os.environ.get("DNS_TIMEOUT_S", "1.0"))
_hosted_down_until = 0.0
_dns_cache = {}  # host -> (expires_at, ok)
_dns_pool = None


def hosted_breaker_open():
    return time.monotonic() < _hosted_down_until


def _trip_hosted_breaker(reason):
    global _hosted_down_until
    if not hosted_breaker_open():
        log("hosted: marking unreachable for %.0fs (%s)" % (HOSTED_COOLDOWN_S, reason))
    _hosted_down_until = time.monotonic() + HOSTED_COOLDOWN_S


def _reset_hosted_breaker():
    global _hosted_down_until
    _hosted_down_until = 0.0


def _resolvable(host, timeout=None):
    """True if `host` resolves within `timeout` seconds (cached 60s)."""
    global _dns_pool
    import concurrent.futures
    import socket
    now = time.monotonic()
    hit = _dns_cache.get(host)
    if hit and hit[0] > now:
        return hit[1]
    if _dns_pool is None:
        _dns_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="dns")
    fut = _dns_pool.submit(socket.getaddrinfo, host, 443)
    try:
        fut.result(timeout=DNS_TIMEOUT_S if timeout is None else timeout)
        ok = True
    except Exception:
        ok = False
    _dns_cache[host] = (now + (60 if ok else 5), ok)
    return ok


def _groq_post(body_bytes, headers, timeout):
    """POST to Groq reusing a pooled connection. Returns (status, headers, body)."""
    global _groq_conn
    if hosted_breaker_open():
        raise BackendError("groq", 0, "hosted breaker open (%.0fs left)" % (_hosted_down_until - time.monotonic()))
    if not _resolvable(_groq_parts.hostname):
        _trip_hosted_breaker("dns unresolvable")
        raise BackendError("groq", 0, "dns: %s unresolvable" % _groq_parts.hostname)
    with _groq_lock:
        for attempt in (0, 1):
            try:
                if _groq_conn is None:
                    cls = http.client.HTTPSConnection if _groq_parts.scheme == "https" else http.client.HTTPConnection
                    _groq_conn = cls(_groq_parts.hostname, _groq_parts.port, timeout=timeout)
                _groq_conn.timeout = timeout
                if _groq_conn.sock is not None:
                    _groq_conn.sock.settimeout(timeout)
                _groq_conn.request("POST", _groq_parts.path, body=body_bytes, headers=headers)
                resp = _groq_conn.getresponse()
                data = resp.read()
                return resp.status, dict(resp.getheaders()), data
            except (http.client.HTTPException, ConnectionError, BrokenPipeError, OSError) as e:
                # Stale keep-alive socket or genuine network error: drop it and
                # retry once; the second failure propagates.
                try:
                    _groq_conn.close()
                except Exception:
                    pass
                _groq_conn = None
                if attempt == 1 or isinstance(e, TimeoutError):
                    _trip_hosted_breaker(type(e).__name__)
                    raise


def _estimate_cost_tokens(body):
    """What Groq will count against TPM for this request: prompt (~4 chars/token)
    plus the reserved completion budget."""
    text = json.dumps(body.get("messages", []))
    prompt = len(text) / 4.0
    reserved = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    return int(prompt + reserved)


def _groq_body_for(model, body):
    """Adapt the app's payload (tuned for gpt-oss) to another Groq model."""
    b = dict(body)
    b["model"] = model
    m = model.lower()
    if m.startswith("openai/gpt-oss"):
        return b
    b.pop("include_reasoning", None)
    if m.startswith("qwen/"):
        # qwen3 thinks out loud unless told not to; we want the answer, fast.
        b["reasoning_effort"] = "none"
    else:
        b.pop("reasoning_effort", None)
    return b


class BackendError(Exception):
    def __init__(self, name, status, detail):
        super().__init__("%s: %s %s" % (name, status, detail))
        self.name, self.status, self.detail = name, status, detail


def call_groq(model, body, timeout):
    if not GROQ_KEY:
        raise BackendError("groq:" + model, 0, "no GROQ_API_KEY")
    payload = _groq_body_for(model, body)
    cost = _estimate_cost_tokens(payload)
    if RATE.will_429(model, cost):
        raise BackendError("groq:" + model, 429, "predicted (remaining≈%s, blocked %.1fs)" % (
            None if RATE.predicted_remaining(model) is None else int(RATE.predicted_remaining(model)),
            RATE.blocked_for(model)))
    headers = {"Authorization": "Bearer " + GROQ_KEY, "Content-Type": "application/json",
               "User-Agent": UA, "Connection": "keep-alive"}
    status, resp_headers, data = _groq_post(json.dumps(payload).encode(), headers, timeout)
    _reset_hosted_breaker()
    RATE.observe(model, resp_headers, status)
    if status != 200:
        raise BackendError("groq:" + model, status, data[:200].decode("utf-8", "replace"))
    return data


# ------------------------------------------------------------------ local transport
_local_warm_until = 0.0
_local_lock = threading.Lock()
_local_is_ollama = None  # unknown until the first native keep-alive touch answers


def _mark_local_warm():
    global _local_warm_until
    with _local_lock:
        _local_warm_until = time.monotonic() + LOCAL_KEEPALIVE_S


def local_is_warm():
    with _local_lock:
        return time.monotonic() < _local_warm_until


# The app's cleanup user turn (PostProcessingService.process). Captures CONTEXT
# and RAW_TRANSCRIPTION so we can re-render them for a model trained on the
# local-setup training format.
_APP_USER_RE = re.compile(
    r'^Instructions: Clean up RAW_TRANSCRIPTION.*?\n\nCONTEXT: "(?P<ctx>.*?)"\n\n'
    r'RAW_TRANSCRIPTION: "(?P<raw>.*)"\s*$', re.S)


def to_train_format(messages):
    """Rewrite the app's cleanup prompt into the format the on-device fine-tune
    was trained on (export_training_data.py / gen_synthetic_data.py):

        Raw transcript:\\n<raw>\\n\\n[Context: context: <summary>\\n\\n]Clean this up into the final text.

    Serving the 1.5B LoRA the app's runtime format instead is a real accuracy
    bug — measured WER 44%→14% and context-leak hallucinations 4→0 on the probe
    set just by fixing the format. Anything we don't recognise (edit-mode
    prompts, custom user prompts) passes through untouched."""
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") != "user" or not isinstance(last.get("content"), str):
        return messages
    m = _APP_USER_RE.match(last["content"])
    if not m:
        return messages
    parts = ["Raw transcript:\n" + m.group("raw").strip()]
    ctx = m.group("ctx").strip()
    if ctx:
        parts.append("Context: context: " + ctx)
    parts.append("Clean this up into the final text.")
    out = list(messages[:-1])
    out.append({"role": "user", "content": "\n\n".join(parts)})
    return out


def _local_body_for(body):
    payload = dict(body)
    payload["model"] = LOCAL_MODEL
    if LOCAL_ADAPTER_PATH:
        payload["adapters"] = LOCAL_ADAPTER_PATH
    if LOCAL_PROMPT_FORMAT == "train":
        payload["messages"] = to_train_format(body.get("messages", []))
    # Hosted/gpt-oss-only params the local server doesn't understand.
    for k in ("reasoning_effort", "include_reasoning", "reasoning"):
        payload.pop(k, None)
    # Ollama & mlx_lm speak `max_tokens`; keep the app's sized budget.
    if "max_completion_tokens" in payload:
        payload["max_tokens"] = payload.pop("max_completion_tokens")
    return payload


def _ollama_native_base():
    """If LOCAL_URL is an Ollama OpenAI-compat endpoint, return its native API base."""
    if "/v1/" not in LOCAL_URL:
        return None
    return LOCAL_URL.split("/v1/", 1)[0]


def _touch_ollama_keepalive():
    """Ollama re-arms a model's residency timer on every request using the
    server's default keep-alive (5m in the desktop app, whatever
    OLLAMA_KEEP_ALIVE says under launchd) — the OpenAI-compat endpoint can't
    override it. Its native /api/generate can, so after each local call we
    re-arm LOCAL_KEEPALIVE_S with an empty prompt (no generation, ~ms). Non-Ollama
    servers (mlx_lm) 404 here; that's fine, they never unload anyway."""
    global _local_is_ollama
    base = _ollama_native_base()
    if not base or _local_is_ollama is False:
        return
    payload = {"model": LOCAL_MODEL, "prompt": "", "keep_alive": "%ds" % int(LOCAL_KEEPALIVE_S)}
    req = urllib.request.Request(base + "/api/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).read()
        _local_is_ollama = True
    except urllib.error.HTTPError as e:
        if e.code == 404:  # not Ollama (e.g. mlx_lm) — never unloads, stop asking
            _local_is_ollama = False
    except Exception:
        pass


def call_local(body, timeout):
    payload = _local_body_for(body)
    req = urllib.request.Request(LOCAL_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise BackendError("local", e.code, e.read()[:200].decode("utf-8", "replace"))
    _mark_local_warm()
    threading.Thread(target=_touch_ollama_keepalive, daemon=True).start()
    return data


# ------------------------------------------------------ speculative segment pre-cleanup
# Decode speed on the local model is hardware-bound (~7ms/token), so the only way
# to make the cleanup at key-up faster is to generate FEWER tokens at key-up.
# The STT server finalizes speech segments while the user is still talking and
# POSTs each one to /v1/precache; we clean it right away (the GPU is idle for us
# during speech). When the app's real cleanup request arrives, the raw transcript
# it carries begins with those exact segment texts (the app joins segments with a
# single space), so we substitute the cached cleaned text for the longest matching
# raw prefix and only run the model on the remainder (usually the last few
# seconds). Zero app changes; falls back to a normal full cleanup on any mismatch.
PRECACHE_ENABLED = os.environ.get("PRECACHE", "1").strip().lower() not in ("0", "false", "no")
# Chunks (pre-cached segments and the tail at commit) are cleaned deterministically
# when they need no interpretation (see detclean.py) — ~0ms and no risk of a small
# model "summarizing" a clause it sees without context. The LLM handles the rest.
DET_CLEAN = os.environ.get("DET_CLEAN", "1").strip().lower() not in ("0", "false", "no")


_DEFAULT_PROMPT_HEAD = "You are a literal dictation cleanup layer"
_VOCAB_MARKER = "Use these spellings exactly in the output when relevant:\n"


def _vocab_from_system_prompt(system_prompt):
    """The app appends the user's custom vocabulary to the system prompt; the
    deterministic cleaner needs it to notice mis-cased terms."""
    if not system_prompt or _VOCAB_MARKER not in system_prompt:
        return ""
    return system_prompt.split(_VOCAB_MARKER, 1)[1].strip()


def clean_chunk(text, system_prompt, context="", timeout=60):
    """Deterministic when safe, else the local model. Returns (cleaned, how).
    Deterministic only under the app's default system prompt — a custom prompt
    means the user wants the model's judgement, so we give it to them."""
    if DET_CLEAN and (system_prompt or "").lstrip().startswith(_DEFAULT_PROMPT_HEAD):
        d = det_clean(text, vocabulary=_vocab_from_system_prompt(system_prompt))
        if d is not None:
            return d, "det"
    out = sanitize_cleanup_output(_local_generate(_cleanup_messages_for(text, system_prompt, context),
                                                  _sized_budget(text), timeout=timeout))
    return ("" if out.strip().upper() == "EMPTY" else out), "llm"
PRECACHE_TTL_S = float(os.environ.get("PRECACHE_TTL_S", "300"))
_precache_lock = threading.Lock()
_precache = []  # list of dicts: {raw, done(Event), cleaned, ts}


def _cleanup_messages_for(raw, system_prompt, context=""):
    parts = ["Raw transcript:\n" + raw.strip()]
    if context:
        parts.append("Context: context: " + context)
    parts.append("Clean this up into the final text.")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n\n".join(parts)}]


def _local_generate(messages, max_tokens, timeout=60):
    """Run the local model on already-train-formatted messages; return content string."""
    payload = {"model": LOCAL_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if LOCAL_ADAPTER_PATH:
        payload["adapters"] = LOCAL_ADAPTER_PATH
    req = urllib.request.Request(LOCAL_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    _mark_local_warm()
    return (data["choices"][0]["message"]["content"] or "").strip()


def _sized_budget(text):
    est = max(1, (len(text) + 2) // 3)
    return max(1, min(4096, max(256, 256 + est * 3)))


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def _norm_words(text):
    """Punctuation/case-insensitive word list, for prefix matching."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _cut_after_words(text, n):
    """Character offset in `text` right after its n-th word (as _norm_words counts)."""
    if n <= 0:
        return 0
    k = 0
    for m in _WORD_RE.finditer(text):
        k += 1
        if k == n:
            return m.end()
    return len(text)


def precache_reset():
    with _precache_lock:
        _precache.clear()


def precache_segment(raw):
    """Called for each finalized chunk of speech (a pause-cut segment, or the
    settled sentences of a partial transcript). Whatever prefix of it is already
    cached is reused; only the NEW words are cleaned, in the background."""
    raw = (raw or "").strip()
    if not PRECACHE_ENABLED or not raw or not _last_cleanup_system:
        return False
    # Incremental: consume anything already cached at the front of this text.
    prefix_clean, rest = _consume_precache(raw, time.monotonic() + 3.0, keep=True)
    if not rest:
        return True  # everything already cached
    entry = {"raw": rest, "words": _norm_words(rest), "done": threading.Event(), "cleaned": None,
             "ts": time.monotonic()}
    if len(entry["words"]) < 2:
        return False
    with _precache_lock:
        _precache.append(entry)
        cutoff = time.monotonic() - PRECACHE_TTL_S
        _precache[:] = [e for e in _precache if e["ts"] >= cutoff][-24:]

    def work():
        t0 = time.monotonic()
        try:
            out, how = clean_chunk(rest, _last_cleanup_system)
            # A chunk is cleaned without the rest of the dictation as context, and a
            # small model occasionally "summarizes" it. Legit cleanup only removes
            # fillers/restarts, so a big word loss means the chunk is unusable —
            # leave the entry failed and the real request cleans everything.
            raw_n, out_n = len(_norm_words(rest)), len(_norm_words(out))
            if raw_n >= 4 and out_n < 0.6 * raw_n:
                log("precache: chunk rejected (kept %d/%d words) — will clean at commit" % (out_n, raw_n))
                return
            entry["cleaned"] = out
            log("precache(%s): %d chars → %d chars in %dms" % (how, len(rest), len(out), (time.monotonic() - t0) * 1000))
        except Exception as e:
            log("precache failed: %r" % (e,))
        finally:
            entry["done"].set()
    threading.Thread(target=work, daemon=True).start()
    return True


def _consume_precache(raw, deadline_at, keep=False):
    """Return (cleaned_prefix, remainder_raw). Greedily matches cached chunks at
    the front of `raw`, word by word (punctuation/case-insensitive — the final
    transcript may differ from a partial by a comma or two). `keep` leaves the
    entries in the cache (used when a later partial re-covers the same words)."""
    cleaned_parts, rest = [], raw.strip()
    while rest:
        rest_words = _norm_words(rest)
        with _precache_lock:
            candidates = [e for e in _precache
                          if e["words"] and len(e["words"]) <= len(rest_words)
                          and rest_words[:len(e["words"])] == e["words"]]
        if not candidates:
            break
        best = max(candidates, key=lambda e: len(e["words"]))
        wait = max(0.0, deadline_at - time.monotonic() - 0.3)
        if not best["done"].wait(timeout=wait) or best["cleaned"] is None:
            break  # not ready / failed → clean the rest normally
        if best["cleaned"]:
            cleaned_parts.append(best["cleaned"])
        rest = rest[_cut_after_words(rest, len(best["words"])):].lstrip(" ,;:.!?").strip()
        if not keep:
            with _precache_lock:
                if best in _precache:
                    _precache.remove(best)
    return " ".join(cleaned_parts), rest


_ECHO_LINES = ("clean this up into the final text.", "clean this up into the final text",
               "raw transcript:", "here is the cleaned transcript:", "cleaned transcript:")


def sanitize_cleanup_output(text):
    """Small models sometimes echo the instruction line(s) of the training
    prompt back before/instead of the answer. Strip those; the caller treats an
    empty result as a backend failure (→ next backend / raw transcript)."""
    if not text:
        return text
    lines = [l for l in text.strip().splitlines()]
    while lines and lines[0].strip().lower() in _ECHO_LINES:
        lines.pop(0)
    while lines and lines[-1].strip().lower() in _ECHO_LINES:
        lines.pop()
    return "\n".join(lines).strip()


def _sanitized_local_response(data):
    """Apply sanitize_cleanup_output to a chat-completion JSON payload (bytes)."""
    try:
        obj = json.loads(data)
        msg = obj["choices"][0]["message"]
        cleaned = sanitize_cleanup_output(msg.get("content") or "")
        if not cleaned:
            raise BackendError("local", 0, "cleanup output was only an instruction echo")
        if cleaned != (msg.get("content") or "").strip():
            log("sanitized local output (instruction echo removed)")
            msg["content"] = cleaned
            return json.dumps(obj).encode()
        return data
    except BackendError:
        raise
    except Exception:
        return data


def call_local_cleanup_with_precache(body, timeout, deadline_at):
    """Local cleanup that reuses pre-cleaned segments. Falls back to call_local."""
    m = _APP_USER_RE.match(body["messages"][-1]["content"])
    if not (PRECACHE_ENABLED and m and LOCAL_PROMPT_FORMAT == "train"):
        return call_local(body, timeout)
    raw, ctx = m.group("raw"), m.group("ctx").strip()
    prefix_clean, rest = _consume_precache(raw, deadline_at)
    if not prefix_clean and rest == raw.strip():
        return call_local(body, timeout)  # nothing cached; normal path (keeps its own params)
    sys_msg = next((x.get("content") for x in body.get("messages", []) if x.get("role") == "system"), "") or ""
    t0 = time.monotonic()
    tail_clean, how = "", "-"
    if rest:
        tail_clean, how = clean_chunk(rest, sys_msg, ctx, timeout=timeout)
    content = " ".join(p for p in (prefix_clean, tail_clean) if p).strip() or "EMPTY"
    log("precache hit: prefix %d chars reused, tail %d chars cleaned (%s) in %dms" % (
        len(raw) - len(rest), len(rest), how, (time.monotonic() - t0) * 1000))
    return json.dumps({"id": "precache", "object": "chat.completion", "model": LOCAL_MODEL,
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": content}}]}).encode()


# The system prompt of the most recent cleanup request. Used to warm the local
# model with a prompt that SHARES ITS PREFIX with the next real request: mlx_lm's
# server caches the KV of the last prompt only, so a warm-up with a different
# prompt (e.g. "hi") would evict the cache and cost ~170ms of prefill on the
# next cleanup instead of saving anything.
_SYSTEM_PROMPT_CACHE_FILE = os.path.expanduser("~/.freeflow-stt/last_system_prompt.txt")
try:
    with open(_SYSTEM_PROMPT_CACHE_FILE) as _f:
        _last_cleanup_system = _f.read() or None
except Exception:
    _last_cleanup_system = None


def _remember_cleanup_system(sys_msg):
    """Persist so a router restart can still warm the local model with the right
    prefix before the first dictation (otherwise that one pays a full prefill)."""
    global _last_cleanup_system
    if sys_msg == _last_cleanup_system:
        return
    _last_cleanup_system = sys_msg
    try:
        with open(_SYSTEM_PROMPT_CACHE_FILE, "w") as f:
            f.write(sys_msg)
    except Exception:
        pass


def warm_local(force=False):
    """Preload the local model / spin the GPU up. `force` = also when the model is
    already resident (Apple GPUs downclock when idle; the first inference after a
    quiet minute is 2-3× slower, so we poke it while the user is still speaking)."""
    if local_is_warm() and not force:
        return
    msgs = [{"role": "user", "content": "Raw transcript:\nhi\n\nClean this up into the final text."}]
    if _last_cleanup_system:
        msgs.insert(0, {"role": "system", "content": _last_cleanup_system})
    tiny = {"model": LOCAL_MODEL, "messages": msgs, "max_tokens": 1}
    t0 = time.monotonic()
    try:
        call_local(tiny, timeout=60)
        log("warm: local model %s ready in %.0fms" % (LOCAL_MODEL, (time.monotonic() - t0) * 1000))
    except Exception as e:
        log("warm: local preload failed: %r" % (e,))


def hosted_reachable():
    """Cheap probe used by /warm (nobody is waiting on it). Also the way the
    breaker gets re-closed while the network is back but no request has proven it."""
    if not GROQ_KEY:
        return False
    if not _resolvable(_groq_parts.hostname):
        _trip_hosted_breaker("dns unresolvable")
        return False
    try:
        probe = urllib.request.Request(GROQ_URL.replace("/chat/completions", "/models"),
                                       headers={"Authorization": "Bearer " + GROQ_KEY, "User-Agent": UA})
        urllib.request.urlopen(probe, timeout=1.5)
        _reset_hosted_breaker()
        return True
    except urllib.error.HTTPError:
        _reset_hosted_breaker()  # reachable, just not 200
        return True
    except Exception as e:
        _trip_hosted_breaker(type(e).__name__)
        return False


TYPICAL_REQUEST_TOKENS = 2000  # ~1.5k prompt (system prompt dominates) + sized completion


def warm_if_needed():
    """Called on hotkey-down. Preload the local model iff we predict the hosted
    chain won't serve the next request quickly."""
    if FORCE_LOCAL:
        warm_local(force=True)
        return
    if local_is_warm():
        return
    if not hosted_reachable():
        log("warm: hosted unreachable → preloading local")
        warm_local()
        return
    for m in [None] + GROQ_FALLBACK_MODELS:
        # None = "the primary model, whichever the app asks for" — we don't know
        # its name here, so use the last-seen primary if any.
        model = m or _last_primary_model
        if model and not RATE.will_429(model, TYPICAL_REQUEST_TOKENS):
            return  # at least one hosted bucket has room; no need to spend RAM
    log("warm: all hosted buckets look exhausted → preloading local")
    warm_local()


_last_primary_model = None


# ------------------------------------------------------------------------ routing
def is_cleanup_request(body):
    """True for FreeFlow's dictation-cleanup prompt (the thing the on-device model
    is fine-tuned for). Context inference / edit mode / anything else → False."""
    msgs = body.get("messages") or []
    if not msgs or msgs[-1].get("role") != "user" or not isinstance(msgs[-1].get("content"), str):
        return False
    return _APP_USER_RE.match(msgs[-1]["content"]) is not None


def backend_chain(primary_model, local_first):
    hosted = [("groq:" + primary_model, lambda b, t, m=primary_model: call_groq(m, b, t))]
    for m in GROQ_FALLBACK_MODELS:
        if m != primary_model:
            hosted.append(("groq:" + m, lambda b, t, m=m: call_groq(m, b, t)))
    local = [("local", call_local)]
    return local + hosted if local_first else hosted + local


def route(body, deadline_at):
    """Try backends in order until one answers inside the deadline.
    Returns (route_name, response_bytes) or raises BackendError for the last failure."""
    global _last_primary_model, _last_cleanup_system
    primary = body.get("model") or "openai/gpt-oss-20b"
    _last_primary_model = primary
    cleanup = is_cleanup_request(body)
    if cleanup:
        sys_msg = next((m.get("content") for m in body.get("messages", []) if m.get("role") == "system"), None)
        if isinstance(sys_msg, str):
            _remember_cleanup_system(sys_msg)
    # FORCE_LOCAL means "the on-device model does the CLEANUP". Other prompts
    # (the context-inference call at hotkey-down, edit mode) go hosted-first: the
    # local model isn't tuned for them, they'd occupy the single-threaded local
    # server right before the cleanup arrives, and they'd evict its prefix cache.
    local_first = FORCE_LOCAL and (cleanup or not GROQ_KEY)
    last_err = None
    for name, fn in backend_chain(primary, local_first):
        remaining = deadline_at - time.monotonic()
        if remaining < MIN_ATTEMPT_S:
            log("deadline: %.2fs left, not trying %s" % (remaining, name))
            break
        per_attempt = LOCAL_TIMEOUT if name == "local" else HOSTED_TIMEOUT
        timeout = max(MIN_ATTEMPT_S, min(per_attempt, remaining))
        t0 = time.monotonic()
        try:
            if name == "local" and cleanup:
                data = _sanitized_local_response(call_local_cleanup_with_precache(body, timeout, deadline_at))
            else:
                data = fn(body, timeout)
            log("route=%s ms=%d" % (name, (time.monotonic() - t0) * 1000))
            return name, data
        except BackendError as e:
            last_err = e
            log("%s failed status=%s in %dms (%s) -> next" % (
                name, e.status, (time.monotonic() - t0) * 1000, e.detail.replace("\n", " ")[:120]))
        except Exception as e:  # timeouts, connection errors, bad JSON from backend
            last_err = BackendError(name, 0, "%s: %s" % (type(e).__name__, e))
            log("%s failed (%s) in %dms -> next" % (name, type(e).__name__, (time.monotonic() - t0) * 1000))
    raise last_err or BackendError("router", 0, "no backends configured")


# ------------------------------------------------------------------------ server
def models_payload():
    data = [{"id": m, "object": "model", "owned_by": "groq"}
            for m in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"] + GROQ_FALLBACK_MODELS]
    seen, uniq = set(), []
    for d in data:
        if d["id"] not in seen:
            seen.add(d["id"])
            uniq.append(d)
    uniq.append({"id": LOCAL_MODEL, "object": "model", "owned_by": "local"})
    return {"object": "list", "data": uniq}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # we log what matters ourselves

    def _send(self, code, data, extra=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            log("client hung up before response could be written (%d)" % code)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p.endswith("/models"):
            self._send(200, json.dumps(models_payload()).encode())
        elif p.endswith("/status"):
            self._send(200, json.dumps({
                "force_local": FORCE_LOCAL, "local_model": LOCAL_MODEL, "local_url": LOCAL_URL,
                "local_warm": local_is_warm(), "fallback_models": GROQ_FALLBACK_MODELS,
                "hosted_breaker_open": hosted_breaker_open(),
                "rate_limits": RATE.snapshot()}, default=str).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        p = self.path.rstrip("/")
        if p.endswith("/warm"):
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            threading.Thread(target=warm_if_needed, daemon=True).start()
            self._send(200, b'{"status":"warming"}')
            return
        if p.endswith("/precache"):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                req = json.loads(raw)
            except Exception:
                req = {}
            if req.get("reset"):
                precache_reset()  # a new dictation started; nothing old can apply
            seg = req.get("raw", "")
            ok = precache_segment(seg) if seg else False
            self._send(200, json.dumps({"queued": bool(ok)}).encode())
            return
        if "/chat/completions" not in p:
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except Exception:
            self._send(400, b'{"error":"bad json"}')
            return
        try:
            deadline_ms = float(self.headers.get(DEADLINE_HEADER, DEFAULT_DEADLINE_MS))
        except ValueError:
            deadline_ms = DEFAULT_DEADLINE_MS
        deadline_at = time.monotonic() + max(0.0, deadline_ms / 1000.0 - DEADLINE_MARGIN_S)
        log("request model=%s deadline_ms=%s max_completion_tokens=%s est_cost=%d" % (
            body.get("model"), self.headers.get(DEADLINE_HEADER, "none(default)"),
            body.get("max_completion_tokens", body.get("max_tokens", "unset")), _estimate_cost_tokens(body)))
        try:
            name, data = route(body, deadline_at)
            self._send(200, data, {"X-FreeFlow-Route": name})
        except BackendError as e:
            code = 504 if (deadline_at - time.monotonic()) < MIN_ATTEMPT_S else 502
            self._send(code, json.dumps({"error": {"message": "router: all backends failed; last: %s" % e}}).encode())


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    srv.daemon_threads = True
    log("freeflow-router listening on 127.0.0.1:%d hosted=%s fallbacks=%s local=%s@%s force_local=%s" % (
        LISTEN_PORT, "on" if GROQ_KEY else "off", ",".join(GROQ_FALLBACK_MODELS), LOCAL_MODEL, LOCAL_URL, FORCE_LOCAL))
    srv.serve_forever()


if __name__ == "__main__":
    main()
