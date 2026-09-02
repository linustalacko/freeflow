#!/usr/bin/env python3
"""
FreeFlow local speech-to-text server — Parakeet-TDT (NVIDIA) on Apple Silicon via
parakeet-mlx. Replaces whisper-server for the hot path.

Two endpoints on one port (FreeFlow derives both from "Transcription API URL"):

  POST /v1/audio/transcriptions   OpenAI-style multipart upload → {"text": ...}
                                  (used as the fallback / by eval_models.py)
  WS   /v1/realtime               OpenAI Realtime *transcription* protocol as
                                  spoken by RealtimeTranscriptionService.swift:
                                    ← session.update, input_audio_buffer.append
                                      (base64 PCM16 LE mono, 24 kHz), input_audio_buffer.commit
                                    → input_audio_buffer.committed,
                                      conversation.item.input_audio_transcription.completed
                                      {item_id, transcript}, error

How the streaming path stays fast (and accurate)
------------------------------------------------
* Parakeet-TDT is ~0.01× real time with no fixed floor (whisper pads every clip
  to 30s): 15s of audio → ~0.17s, 60s → ~0.6s.
* While you talk, every PARTIAL_EVERY_S of audio we transcribe a *window*: the
  not-yet-settled audio plus LEFT_CONTEXT_S of already-settled audio before it,
  so the words we care about always have full left context. Words that the last
  two windows agree on, that ended ≥ SETTLE_HORIZON_S before the window end, and
  that are followed by a small inter-word gap are *settled*: emitted to the app
  as a `completed` event and POSTed to the router's /v1/precache so cleanup can
  start early. The audio is never cut, so nothing is transcribed without context
  (cutting at pauses was measured to change ~2.7% of words vs batch, mostly
  casing/spelling at the cut).
* At key-up only the unsettled tail (typically 2–5s) plus its left context is
  transcribed and spliced after the settled text by timestamp: ~0.1–0.2s,
  independent of how long you spoke and whether you ever paused. (The old
  design cut segments only on ≥0.8s pauses; this speaker almost never pauses, so
  95% of every dictation was transcribed at key-up — up to 10s for a 3-minute
  dictation.)
* Memory: MLX's buffer cache is capped (STT_CACHE_MB). Uncapped, every
  different clip length left new buffers behind and the process grew to 17 GB
  in nine days, swapped the whole machine, and every dictation after a break
  paid 7–13s of page-ins. The model weights are wired (STT_WIRED_MB) and
  touched every STT_HEARTBEAT_S while idle so they cannot be paged out either.

Client contract subtleties (see RealtimeTranscriptionService.swift):
* The app appends every `completed` transcript to its final text (joined by a
  single space), and resolves the await on the FIRST `completed` it receives
  after it sent `commit`. So once we have received `commit` we must emit exactly
  ONE `completed`, carrying everything not yet emitted. Partial results that
  finish after the commit are discarded — their words are covered by the tail.

Config (env):
  STT_MODEL            default mlx-community/parakeet-tdt-0.6b-v2 (English; -v3 = multilingual, see note)
  STT_PORT             default 8082
  STT_PARTIAL_EVERY_S  seconds of audio between mid-speech windows (2.0)
  STT_LEFT_CONTEXT_S   settled audio kept in front of every window (10.0)
  STT_SETTLE_HORIZON_S a word must have ended this long before the window end to settle (1.0)
  STT_SETTLE_GAP_S     minimum inter-word gap at a settle point (0.0: any agreed word
                       boundary — the commit splice anchors on words, not on time)
  STT_MIN_SETTLE_S     do not settle chunks shorter than this (1.0)
  STT_CACHE_MB         MLX buffer-cache cap (512); STT_WIRED_MB wired weights budget (2048)
  STT_HEARTBEAT_S      idle keep-warm interval (45; 0 = off)
"""
import asyncio
import base64
import collections
import concurrent.futures
import io
import json
import logging
import os
import sys
import time
import uuid
import wave

import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*web.AppKey.*")

log = logging.getLogger("freeflow-stt")

# v2 (English) is used by default on purpose: measured on real recordings, v3
# (multilingual) returned "" / dropped the trailing phrase for 10 of 46 tail clips
# that started mid-word or had certain lengths; v2 failed on 0 of 46 at identical
# speed. Set STT_MODEL=mlx-community/parakeet-tdt-0.6b-v3 for non-English dictation.
MODEL_ID = os.environ.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
PORT = int(os.environ.get("STT_PORT", "8082"))
PARTIAL_EVERY_S = float(os.environ.get("STT_PARTIAL_EVERY_S", "2.0"))
LEFT_CONTEXT_S = float(os.environ.get("STT_LEFT_CONTEXT_S", "10.0"))
SETTLE_HORIZON_S = float(os.environ.get("STT_SETTLE_HORIZON_S", "1.0"))
SETTLE_GAP_S = float(os.environ.get("STT_SETTLE_GAP_S", "0.0"))
MIN_SETTLE_S = float(os.environ.get("STT_MIN_SETTLE_S", "1.0"))
CACHE_MB = int(os.environ.get("STT_CACHE_MB", "512"))
WIRED_MB = int(os.environ.get("STT_WIRED_MB", "2048"))
HEARTBEAT_S = float(os.environ.get("STT_HEARTBEAT_S", "45"))
TARGET_SR = 16000
# Where settled text is pushed so the LLM router can pre-clean it while the user
# is still talking (router.py /v1/precache). Empty = off.
PRECACHE_URL = os.environ.get("ROUTER_PRECACHE_URL", "http://127.0.0.1:11435/v1/precache")

Word = collections.namedtuple("Word", "text start end")


def source_sha():
    """Short hash of this file, reported by /health and the startup line so a
    deployed copy that drifted from the repo is visible (local-setup/deploy.sh)."""
    import hashlib
    try:
        return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
    except Exception:
        return "unknown"


SOURCE_SHA = source_sha()


class ServerState:
    """Mutable runtime state (aiohttp freezes the app mapping at startup)."""

    def __init__(self):
        self.sessions = 0                 # open realtime sockets
        self.last_work = time.monotonic() # last time the GPU thread got a job
        self.http = None                  # ClientSession for /v1/precache
        self.heartbeat = None


# ------------------------------------------------------------------ audio utils
def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def resample(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class StreamResampler:
    """Resample a continuous stream that arrives in messages. resample_poly is
    stateless, so resampling each message on its own puts a filter transient at
    every message edge — ten glitches a second in the audio the model hears.
    Here each message is resampled together with the tail of the previous one,
    and the last `overlap` input samples are held back until the next message
    (or flush) so both edges of every emitted sample had real neighbours."""

    def __init__(self, sr_in, sr_out=TARGET_SR, overlap=240):
        self.sr_in, self.sr_out = sr_in, sr_out
        self.overlap = overlap
        self.pending = np.zeros(0, dtype=np.float32)   # input not yet emitted
        self.context = np.zeros(0, dtype=np.float32)   # already-emitted input kept as left context

    def _out_len(self, n_in):
        return int(round(n_in * self.sr_out / self.sr_in))

    def feed(self, x, final=False):
        if self.sr_in == self.sr_out:
            return x
        self.pending = np.concatenate([self.pending, x])
        emit_n = len(self.pending) if final else len(self.pending) - self.overlap
        if emit_n <= 0:
            return np.zeros(0, dtype=np.float32)
        block = np.concatenate([self.context, self.pending])
        y = resample(block, self.sr_in, self.sr_out)
        skip = self._out_len(len(self.context))
        out = y[skip:skip + self._out_len(emit_n)]
        emitted, self.pending = self.pending[:emit_n], self.pending[emit_n:]
        self.context = np.concatenate([self.context, emitted])[-self.overlap:]
        return out.astype(np.float32)


def decode_wav(data: bytes):
    """PCM WAV → (float32 mono, sample_rate). FreeFlow writes 16 kHz mono int16."""
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError("unsupported WAV sample width %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def words_from_tokens(tokens, offset=0.0):
    """Merge sub-word tokens into words. A token that starts with a space begins a
    word; punctuation and continuation pieces attach to the word before them.
    Times are shifted by `offset` (the window start within the session)."""
    words = []
    for t in tokens:
        text, start, end = t.text, t.start + offset, t.end + offset
        if words and not text.startswith(" "):
            prev = words[-1]
            words[-1] = Word(prev.text + text, prev.start, max(prev.end, end))
        else:
            words.append(Word(text, start, end))
    return words


def words_text(words):
    return "".join(w.text for w in words).strip()


def norm_word(text):
    return text.strip().lower().strip(".,!?;:\"'“”‘’")


def agreed_settle_count(prev, cur, window_end, settled_time):
    """LocalAgreement: how many leading words of `cur` may be settled now. Words
    are compared without case/punctuation (a comma that appears with more right
    context must not block settling), must have ended ≥ SETTLE_HORIZON_S before
    the window end, and the settle point must follow an inter-word gap of at
    least SETTLE_GAP_S and cover at least MIN_SETTLE_S of new audio."""
    if not prev or not cur:
        return 0
    n = 0
    while n < len(prev) and n < len(cur) and norm_word(prev[n].text) == norm_word(cur[n].text):
        n += 1
    best = 0
    for i in range(n):
        if cur[i].end > window_end - SETTLE_HORIZON_S:
            break
        next_start = cur[i + 1].start if i + 1 < len(cur) else window_end
        if next_start - cur[i].end >= SETTLE_GAP_S:
            best = i + 1
    if best and cur[best - 1].end - settled_time < MIN_SETTLE_S:
        return 0
    return best


def splice_tail(words, settled_words, cut):
    """Words of the commit window that come after the settled text. Anchors on
    the last settled words (text match, near the cut time) so a boundary word
    the final decode re-tokenised is neither dropped nor duplicated; falls back
    to the timestamp when no anchor is found."""
    texts = [norm_word(w.text) for w in words]
    for k in (3, 2, 1):
        anchor = [norm_word(w.text) for w in settled_words[-k:]]
        if len(anchor) < k:
            continue
        # The match closest to the cut wins: a repeated word ("the … the") must
        # not pull the splice later and drop what lies between.
        hits = [(abs(words[i + k - 1].end - cut), i + k) for i in range(len(texts) - k + 1)
                if texts[i:i + k] == anchor and abs(words[i + k - 1].end - cut) <= 1.5]
        if hits:
            return words[min(hits)[1]:]
    tail = [w for w in words if w.start >= cut - 0.1]
    if tail and settled_words and tail[0].start < cut + 0.05 \
            and norm_word(tail[0].text) == norm_word(settled_words[-1].text):
        tail = tail[1:]
    return tail


# ------------------------------------------------------------------ model
class Transcriber:
    def __init__(self, model_id):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        self.mx, self.get_logmel = mx, get_logmel
        # Bound the buffer cache BEFORE the first allocation: uncapped, MLX keeps
        # every freed buffer, and clips of ever-different lengths never reuse
        # one — measured 13 GB of dead cache after 21 real dictations.
        if CACHE_MB >= 0:
            mx.set_cache_limit(CACHE_MB * 2 ** 20)
        if WIRED_MB > 0:
            try:
                limit = min(WIRED_MB * 2 ** 20, int(mx.device_info()["max_recommended_working_set_size"]))
                mx.set_wired_limit(limit)
            except Exception as e:  # older MLX / no Metal
                log.warning("could not set wired limit: %r", e)
        t0 = time.time()
        self.model = from_pretrained(model_id)
        log.info("loaded %s in %.1fs", model_id, time.time() - t0)
        # first call compiles kernels; do it now, not on the user's first dictation
        self.transcribe(np.zeros(TARGET_SR, dtype=np.float32))
        self.transcribe(np.zeros(TARGET_SR * 5, dtype=np.float32))
        log.info("warm (active %.0fMB, cache %.0fMB)", mx.get_active_memory() / 2 ** 20, mx.get_cache_memory() / 2 ** 20)

    def _generate(self, x: np.ndarray):
        mel = self.get_logmel(self.mx.array(x), self.model.preprocessor_config)
        return self.model.generate(mel)[0]

    def transcribe(self, x: np.ndarray) -> str:
        if len(x) < int(0.15 * TARGET_SR):
            return ""
        return self._generate(x).text.strip()

    def transcribe_tokens(self, x: np.ndarray):
        """Sub-word tokens with timestamps (seconds, relative to the clip start)."""
        if len(x) < int(0.15 * TARGET_SR):
            return []
        return list(self._generate(x).tokens)

    def memory_stats(self):
        return {"active_mb": round(self.mx.get_active_memory() / 2 ** 20),
                "cache_mb": round(self.mx.get_cache_memory() / 2 ** 20),
                "peak_mb": round(self.mx.get_peak_memory() / 2 ** 20)}


# ------------------------------------------------------------------ realtime session
class RealtimeSession:
    """State for one /v1/realtime socket."""

    def __init__(self, ws, app):
        self.ws = ws
        self.app = app
        self.loop = asyncio.get_running_loop()
        self.rate = 24000
        self.resampler = StreamResampler(self.rate)
        self.buf = np.zeros(0, dtype=np.float32)   # the whole session, 16 kHz
        self.settled_samples = 0                    # audio before this index has been emitted
        self.settled_chars = 0
        self.settled_chunks = 0
        self.settled_words = []                     # last few settled words (commit splice anchor)
        self.prev_hyp = None                        # unsettled words of the previous window
        self.audio_since_partial = 0.0
        self.partial_inflight = False
        self.committed = False
        self.item_id = "item_" + uuid.uuid4().hex[:12]
        self.t_start = time.monotonic()
        self.t_commit = None

    async def send(self, obj):
        try:
            await self.ws.send_str(json.dumps(obj))
        except Exception:
            pass

    def on_session_update(self, ev):
        try:
            self.rate = int(ev["session"]["audio"]["input"]["format"].get("rate", 24000))
        except Exception:
            self.rate = 24000
        self.resampler = StreamResampler(self.rate)

    # ---- audio in ------------------------------------------------------------
    def on_append(self, ev):
        x = self.resampler.feed(pcm16_to_float(base64.b64decode(ev.get("audio", ""))))
        self.buf = np.concatenate([self.buf, x])
        self.audio_since_partial += len(x) / TARGET_SR
        if (not self.committed and not self.partial_inflight
                and self.audio_since_partial >= PARTIAL_EVERY_S):
            self.audio_since_partial = 0.0
            # Nothing new to hear (the user is thinking): skip the window.
            recent = self.buf[-int(PARTIAL_EVERY_S * TARGET_SR):]
            if len(recent) and float(np.max(np.abs(recent))) < 1e-4:
                return
            self._schedule_partial()

    def _window(self):
        start = max(0, self.settled_samples - int(LEFT_CONTEXT_S * TARGET_SR))
        return start, len(self.buf), self.buf[start:]

    def _schedule_partial(self):
        self.partial_inflight = True
        start, end, audio = self._window()
        t0 = time.monotonic()
        fut = self.loop.run_in_executor(self.app["executor"], self._tokens_safe, audio)
        fut.add_done_callback(lambda f: self.loop.call_soon_threadsafe(self._on_partial, f, start, end, t0))
        self.app["state"].last_work = time.monotonic()

    def _tokens_safe(self, audio):
        try:
            return self.app["transcriber"].transcribe_tokens(audio)
        except Exception as e:  # surfaced to the client as an error event at commit
            log.exception("transcribe failed")
            return e

    def _unsettled_words(self, tokens, start_idx):
        """Words of a window hypothesis that lie after the settled point — anchored
        on the last settled words, because TDT timestamps are quantised to 80ms
        and a re-decoded boundary word can land on either side of the cut."""
        return splice_tail(words_from_tokens(tokens, start_idx / TARGET_SR), self.settled_words,
                           self.settled_samples / TARGET_SR)

    def _on_partial(self, fut, start_idx, end_idx, t0):
        self.partial_inflight = False
        if self.committed:
            return  # its words are covered by the tail; emitting now would break the one-final contract
        toks = fut.result()
        if isinstance(toks, Exception) or toks is None:
            return
        window_end = end_idx / TARGET_SR
        cur = self._unsettled_words(toks, start_idx)
        n = agreed_settle_count(self.prev_hyp, cur, window_end, self.settled_samples / TARGET_SR)
        self.prev_hyp = cur
        if n <= 0:
            return
        settled, rest = cur[:n], cur[n:]
        gap_end = rest[0].start if rest else window_end
        cut_time = min(settled[-1].end + (gap_end - settled[-1].end) / 2.0, settled[-1].end + 0.5)
        self.settled_samples = max(self.settled_samples, int(cut_time * TARGET_SR))
        self.settled_words = (self.settled_words + settled)[-8:]
        self.prev_hyp = rest
        text = words_text(settled)
        self.settled_chars += len(text)
        self.settled_chunks += 1
        log.info("settled: +%d chars (%d words, settled to %.1fs of %.1fs, window %.1fs → %.0fms) %r",
                 len(text), n, cut_time, window_end, (end_idx - start_idx) / TARGET_SR,
                 (time.monotonic() - t0) * 1000, text if len(text) <= 48 else text[:22] + "…" + text[-22:])
        if text:
            asyncio.ensure_future(self.send({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": self.item_id, "transcript": text}))
            asyncio.ensure_future(precache(self.app, text))

    # ---- commit --------------------------------------------------------------
    def on_commit(self):
        self.committed = True
        self.t_commit = time.monotonic()
        held = self.resampler.feed(np.zeros(0, dtype=np.float32), final=True)
        if len(held):
            self.buf = np.concatenate([self.buf, held])
        start, end, audio = self._window()
        fut = self.loop.run_in_executor(self.app["executor"], self._tokens_safe, audio)
        fut.add_done_callback(lambda f: self.loop.call_soon_threadsafe(self._on_final, f, start, end))
        self.app["state"].last_work = time.monotonic()

    def _on_final(self, fut, start_idx, end_idx):
        res = fut.result()
        if isinstance(res, Exception):
            asyncio.ensure_future(self.send({"type": "error", "error": {
                "code": "transcription_failed", "message": str(res)}}))
            return
        cut = self.settled_samples / TARGET_SR
        tail = splice_tail(words_from_tokens(res, start_idx / TARGET_SR), self.settled_words, cut)
        transcript = words_text(tail)
        log.info("commit: tail %.1fs (window %.1fs) → final in %.0fms (%d chars; session %.1fs, %d chunks/%d chars settled early)",
                 (end_idx - self.settled_samples) / TARGET_SR, (end_idx - start_idx) / TARGET_SR,
                 (time.monotonic() - self.t_commit) * 1000, len(transcript),
                 time.monotonic() - self.t_start, self.settled_chunks, self.settled_chars)
        asyncio.ensure_future(self.send({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": self.item_id, "transcript": transcript}))


async def precache(app, text, reset=False):
    """Fire-and-forget: hand settled text to the router for pre-cleanup. `reset` =
    new dictation started; the router drops anything cached from the previous one."""
    if not PRECACHE_URL:
        return
    try:
        state = app["state"]
        session = state.http
        if session is None:
            from aiohttp import ClientSession, ClientTimeout
            session = state.http = ClientSession(timeout=ClientTimeout(total=2))
        async with session.post(PRECACHE_URL, json={"raw": text, "reset": reset}) as r:
            await r.read()
    except Exception as e:
        log.debug("precache post failed: %r", e)


# ------------------------------------------------------------------ keep-warm
async def heartbeat(app):
    """Touch the weights while idle so macOS never pages them out. Under memory
    pressure a model that sat idle for ten minutes cost 7–13s of page-ins on the
    next dictation; a 0.5s clip every HEARTBEAT_S keeps every weight recently
    used for ~15ms of GPU time."""
    if HEARTBEAT_S <= 0:
        return
    state = app["state"]
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        if state.sessions or time.monotonic() - state.last_work < HEARTBEAT_S * 0.9:
            continue
        app["executor"].submit(app["transcriber"].transcribe, np.zeros(int(0.5 * TARGET_SR), dtype=np.float32))
        state.last_work = time.monotonic()


async def start_background(app):
    app["state"].heartbeat = asyncio.ensure_future(heartbeat(app))


async def stop_background(app):
    state = app["state"]
    if state.heartbeat:
        state.heartbeat.cancel()
    if state.http:
        await state.http.close()


# ------------------------------------------------------------------ HTTP handlers
async def handle_realtime(request):
    from aiohttp import web, WSMsgType
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)
    app = request.app
    state = app["state"]
    sess = RealtimeSession(ws, app)
    state.sessions += 1
    log.info("realtime: session open")
    # Apple GPUs downclock when idle: the first inference after a quiet minute
    # is 2-3× slower. Poke the model now — the user is still speaking, so this
    # is free — instead of paying it at commit time.
    app["executor"].submit(app["transcriber"].transcribe, np.zeros(int(0.6 * TARGET_SR), dtype=np.float32))
    state.last_work = time.monotonic()
    asyncio.ensure_future(precache(app, "", reset=True))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
                continue
            try:
                ev = json.loads(msg.data)
            except Exception:
                continue
            t = ev.get("type")
            if t == "session.update":
                sess.on_session_update(ev)
            elif t == "input_audio_buffer.append":
                sess.on_append(ev)
            elif t == "input_audio_buffer.commit":
                await sess.send({"type": "input_audio_buffer.committed", "item_id": sess.item_id})
                sess.on_commit()
    finally:
        state.sessions -= 1
    log.info("realtime: session closed")
    return ws


async def handle_transcriptions(request):
    from aiohttp import web
    reader = await request.multipart()
    audio_bytes, fmt = None, "json"
    async for part in reader:
        if part.name == "file":
            audio_bytes = await part.read(decode=False)
        elif part.name == "response_format":
            fmt = (await part.text()).strip()
    if not audio_bytes:
        return web.json_response({"error": "no file"}, status=400)
    try:
        x, sr = decode_wav(audio_bytes)
    except Exception as e:
        return web.json_response({"error": "unsupported audio (PCM WAV only): %s" % e}, status=400)
    x = resample(x, sr)
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    request.app["state"].last_work = time.monotonic()
    text = await loop.run_in_executor(request.app["executor"], request.app["transcriber"].transcribe, x)
    log.info("batch: %.1fs audio → %d chars in %.0fms", len(x) / TARGET_SR, len(text), (time.monotonic() - t0) * 1000)
    if fmt == "text":
        return web.Response(text=text)
    return web.json_response({"text": text, "duration": len(x) / TARGET_SR})


async def handle_models(request):
    from aiohttp import web
    return web.json_response({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"},
                                                        {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "local"}]})


async def handle_health(request):
    from aiohttp import web
    body = {"ok": True, "model": MODEL_ID, "sessions": request.app["state"].sessions, "source_sha": SOURCE_SHA}
    tr = request.app.get("transcriber")
    if tr is not None and hasattr(tr, "memory_stats"):
        try:
            body["memory"] = tr.memory_stats()
        except Exception:
            pass
    return web.json_response(body)


def make_app(transcriber, executor=None):
    from aiohttp import web
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["executor"] = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    app["transcriber"] = transcriber
    app["state"] = ServerState()
    app.add_routes([
        web.get("/v1/realtime", handle_realtime),
        web.get("/realtime", handle_realtime),
        web.post("/v1/audio/transcriptions", handle_transcriptions),
        web.post("/inference", handle_transcriptions),
        web.get("/v1/models", handle_models),
        web.get("/health", handle_health),
    ])
    app.on_startup.append(start_background)
    app.on_cleanup.append(stop_background)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    from aiohttp import web
    # MLX keeps per-thread default device/stream state; touching the model from
    # a thread other than the one that initialised it fails under launchd with
    # "There is no Stream(gpu, 0) in current thread". So ONE worker thread owns
    # MLX end to end: it loads the model and runs every transcription.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    transcriber = executor.submit(Transcriber, MODEL_ID).result()
    app = make_app(transcriber, executor)
    log.info("freeflow-stt %s listening on 127.0.0.1:%d (model=%s partial=%.1fs left_ctx=%.0fs horizon=%.1fs gap=%.2fs cache=%dMB wired=%dMB heartbeat=%.0fs)",
             SOURCE_SHA, PORT, MODEL_ID, PARTIAL_EVERY_S, LEFT_CONTEXT_S, SETTLE_HORIZON_S, SETTLE_GAP_S, CACHE_MB, WIRED_MB, HEARTBEAT_S)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None, access_log=None)


if __name__ == "__main__":
    main()
