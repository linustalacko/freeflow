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

Why this is faster than whisper-server
--------------------------------------
* whisper's encoder always runs on a padded 30s window, so it costs ~0.5s even for
  a 1.4s utterance. Parakeet-TDT is ~0.01× real time with no floor: 7s → 0.14s,
  26s → 0.39s (measured, M-series).
* On the realtime socket we transcribe *while you're still talking*: whenever a
  natural pause ends a segment (>= MIN_SEGMENT_S of speech followed by >= PAUSE_S
  of silence) that segment is transcribed in the background at full batch
  quality. At key-up only the short tail is left (~0.1s). No pauses → the whole
  buffer is transcribed at commit, still 2-4× faster than whisper.
  (parakeet-mlx's native token-streaming mode was measured to be both slower than
  real time on 250ms chunks and markedly less accurate, so it is not used.)

Client contract subtleties (see RealtimeTranscriptionService.swift):
* The app appends every `completed` transcript to its final text, and resolves
  the await on the FIRST `completed` it receives after it sent `commit`. So once
  we've received `commit` we must emit exactly ONE `completed` carrying every
  not-yet-emitted segment plus the tail. Segment results that finish after the
  commit are held and merged into that final event.

Config (env):
  STT_MODEL      default mlx-community/parakeet-tdt-0.6b-v2 (English; -v3 = multilingual, see note)
  STT_PORT       default 8082
  STT_PAUSE_S    silence that closes a segment mid-speech (default 0.8)
  STT_MIN_SEGMENT_S  minimum speech before a pause may close a segment (default 3.0)
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
PAUSE_S = float(os.environ.get("STT_PAUSE_S", "0.8"))
MIN_SEGMENT_S = float(os.environ.get("STT_MIN_SEGMENT_S", "3.0"))
# Pause-less speech never triggers a segment cut, so every PARTIAL_EVERY_S of
# speech we also transcribe the buffer-so-far and hand the router every COMPLETE
# sentence that ended >= PARTIAL_SETTLE_S ago (words that old don't change when
# more audio arrives). The router pre-cleans just the new sentences.
PARTIAL_EVERY_S = float(os.environ.get("STT_PARTIAL_EVERY_S", "2.0"))
PARTIAL_SETTLE_S = float(os.environ.get("STT_PARTIAL_SETTLE_S", "0.8"))
TARGET_SR = 16000
FRAME_S = 0.02  # VAD frame
# Where finalized mid-speech segments are pushed so the LLM router can pre-clean
# them while the user is still talking (router.py /v1/precache). Empty = off.
PRECACHE_URL = os.environ.get("ROUTER_PRECACHE_URL", "http://127.0.0.1:11435/v1/precache")


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


class EnergyVAD:
    """Cheap pause detector: RMS per 20ms frame against an adaptive threshold
    derived from the distribution of recent frames — p20 (≈ the room) and p90
    (≈ this speaker's peaks) over the last ~10s. Only used to find *long* pauses
    for opportunistic pre-transcription and to pace partials — a miss just means
    more work at commit; a false split only affects punctuation.
    (An earlier noise-floor tracker learned "noise" from soft speech frames and
    ended up above the median speech level, marking most speech as silence.)"""

    def __init__(self, sr=TARGET_SR):
        self.frame = int(sr * FRAME_S)
        self.history = collections.deque(maxlen=int(10.0 / FRAME_S))
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.had_speech = False
        self._thr = 0.004

    def threshold(self):
        return self._thr * (0.7 if self.in_speech else 1.0)

    def _update_threshold(self):
        if len(self.history) < 25:
            return
        arr = np.fromiter(self.history, dtype=np.float32, count=len(self.history))
        p20, p90 = np.percentile(arr, [20, 90])
        # room × 2.5 (but never above 40% of the speaker's peaks — in continuous
        # speech p20 is itself speech), or 15% of the peaks; floor 0.003
        self._thr = max(0.003, p90 * 0.15, min(p20 * 2.5, p90 * 0.4))

    def feed(self, x: np.ndarray):
        """Returns list of per-frame booleans (speech?) for complete frames."""
        out = []
        n = len(x) // self.frame
        for i in range(n):
            f = x[i * self.frame:(i + 1) * self.frame]
            rms = float(np.sqrt(np.mean(f * f)))
            self.history.append(rms)
            if (self.speech_frames + self.silence_frames) % 5 == 0:
                self._update_threshold()
            speech = rms > self.threshold()
            if speech:
                self.speech_frames += 1
                self.silence_frames = 0
                self.had_speech = True
            else:
                self.silence_frames += 1
            self.in_speech = speech
            out.append(speech)
        return out


# ------------------------------------------------------------------ model
class Transcriber:
    def __init__(self, model_id):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        self.mx, self.get_logmel = mx, get_logmel
        t0 = time.time()
        self.model = from_pretrained(model_id)
        log.info("loaded %s in %.1fs", model_id, time.time() - t0)
        # first call compiles kernels; do it now, not on the user's first dictation
        self.transcribe(np.zeros(TARGET_SR, dtype=np.float32))
        self.transcribe(np.zeros(TARGET_SR * 5, dtype=np.float32))
        log.info("warm")

    def transcribe(self, x: np.ndarray) -> str:
        if len(x) < int(0.15 * TARGET_SR):
            return ""
        mel = self.get_logmel(self.mx.array(x), self.model.preprocessor_config)
        res = self.model.generate(mel)[0]
        return res.text.strip()

    def settled_sentences(self, x: np.ndarray, settle_s: float) -> str:
        """Transcribe and return only the leading complete sentences whose last
        token ended at least `settle_s` before the end of the clip."""
        if len(x) < int(1.0 * TARGET_SR):
            return ""
        mel = self.get_logmel(self.mx.array(x), self.model.preprocessor_config)
        res = self.model.generate(mel)[0]
        limit = len(x) / TARGET_SR - settle_s
        keep = []
        for sent in res.sentences:
            text = sent.text.strip()
            if not text or not sent.tokens:
                continue
            if sent.tokens[-1].end > limit or text[-1] not in ".!?":
                break
            keep.append(text)
        return " ".join(keep)


# ------------------------------------------------------------------ realtime session
class RealtimeSession:
    """State for one /v1/realtime socket."""

    def __init__(self, ws, app):
        self.ws = ws
        self.app = app
        self.loop = asyncio.get_running_loop()
        self.rate = 24000
        self.buf = np.zeros(0, dtype=np.float32)   # unsegmented 16 kHz audio
        self.vad = EnergyVAD()
        self.seg_speech_s = 0.0
        self.committed = False
        self.held = []          # segment transcripts finished after commit
        self.pending = 0        # segment jobs in flight
        self.item_id = "item_" + uuid.uuid4().hex[:12]
        self.t_start = time.monotonic()
        self.emitted_chars = 0
        self.speech_since_partial = 0.0
        self.partial_inflight = False
        self.last_partial_text = ""

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

    def on_append(self, ev):
        x = pcm16_to_float(base64.b64decode(ev.get("audio", "")))
        x = resample(x, self.rate)
        self.buf = np.concatenate([self.buf, x])
        flags = self.vad.feed(x)
        self.seg_speech_s += sum(flags) * FRAME_S
        # Close a segment on a long pause after enough speech. Cut in the MIDDLE of
        # the silence run so the segment ends in silence and the tail starts in
        # silence — Parakeet is fragile on clips that begin mid-word (measured:
        # words dropped / empty output), so never let a cut land on speech.
        if (not self.committed and self.seg_speech_s >= MIN_SEGMENT_S
                and self.vad.silence_frames * FRAME_S >= PAUSE_S):
            cut = len(self.buf) - (self.vad.silence_frames * self.vad.frame) // 2
            if cut > int(MIN_SEGMENT_S * TARGET_SR):
                seg, self.buf = self.buf[:cut], self.buf[cut:]
                self.seg_speech_s = 0.0
                self.speech_since_partial = 0.0
                self.last_partial_text = ""
                self.vad.silence_frames = 0
                self._schedule(seg, final=False)
                return
        # Pause-less speech: periodically pre-transcribe settled sentences.
        self.speech_since_partial += sum(flags) * FRAME_S
        if (PRECACHE_URL and not self.committed and not self.partial_inflight
                and self.speech_since_partial >= PARTIAL_EVERY_S):
            self.speech_since_partial = 0.0
            self.partial_inflight = True
            snapshot = self.buf.copy()
            fut = self.loop.run_in_executor(self.app["executor"], self._partial_safe, snapshot)
            fut.add_done_callback(lambda f: self.loop.call_soon_threadsafe(self._on_partial, f))

    def _partial_safe(self, audio):
        try:
            return self.app["transcriber"].settled_sentences(audio, PARTIAL_SETTLE_S)
        except Exception as e:
            log.exception("partial transcribe failed")
            return ""

    def _on_partial(self, fut):
        self.partial_inflight = False
        text = fut.result() or ""
        if self.committed or not text or text == self.last_partial_text:
            return
        self.last_partial_text = text
        log.info("partial: %d chars of settled sentences → precache", len(text))
        asyncio.ensure_future(precache(self.app, text))

    def on_commit(self):
        self.committed = True
        tail, self.buf = self.buf, np.zeros(0, dtype=np.float32)
        self._schedule(tail, final=True)

    def _schedule(self, audio, final):
        self.pending += 1
        fut = self.loop.run_in_executor(self.app["executor"], self._transcribe_safe, audio)
        fut.add_done_callback(lambda f: self.loop.call_soon_threadsafe(self._on_result, f, final, len(audio)))

    def _transcribe_safe(self, audio):
        try:
            return self.app["transcriber"].transcribe(audio)
        except Exception as e:  # surfaced to the client as an error event
            log.exception("transcribe failed")
            return e

    def _on_result(self, fut, final, n_samples):
        self.pending -= 1
        res = fut.result()
        if isinstance(res, Exception):
            asyncio.ensure_future(self.send({"type": "error", "error": {
                "code": "transcription_failed", "message": str(res)}}))
            return
        text = res
        if final:
            # Executor is single-threaded FIFO, so every earlier segment job has
            # already run and either emitted or been parked in `held`.
            parts = [t for t in self.held if t] + ([text] if text else [])
            transcript = " ".join(parts).strip()
            self.held = []
            log.info("commit: tail %.1fs → final in %.0fms (%d chars, session %.1fs)",
                     n_samples / TARGET_SR, (time.monotonic() - self.t_commit) * 1000,
                     len(transcript), time.monotonic() - self.t_start)
            asyncio.ensure_future(self.send({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": self.item_id, "transcript": transcript}))
        elif self.committed:
            self.held.append(text)  # merge into the single post-commit event
        else:
            log.info("segment %.1fs → %d chars (mid-speech)", n_samples / TARGET_SR, len(text))
            if text:
                asyncio.ensure_future(self.send({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": self.item_id, "transcript": text}))
                asyncio.ensure_future(precache(self.app, text))


async def precache(app, text, reset=False):
    """Fire-and-forget: hand finalized text (a pause-cut segment or the settled
    sentences of a partial) to the router for pre-cleanup. `reset` = new
    dictation started; the router drops anything cached from the previous one."""
    if not PRECACHE_URL:
        return
    try:
        session = app.get("http")
        if session is None:
            from aiohttp import ClientSession, ClientTimeout
            session = app["http"] = ClientSession(timeout=ClientTimeout(total=2))
        async with session.post(PRECACHE_URL, json={"raw": text, "reset": reset}) as r:
            await r.read()
    except Exception as e:
        log.debug("precache post failed: %r", e)


# ------------------------------------------------------------------ HTTP handlers
async def handle_realtime(request):
    from aiohttp import web, WSMsgType
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)
    sess = RealtimeSession(ws, request.app)
    log.info("realtime: session open")
    # Apple GPUs downclock when idle: the first inference after a quiet minute
    # is 2-3× slower. Poke the model now — the user is still speaking, so this
    # is free — instead of paying it at commit time.
    request.app["executor"].submit(request.app["transcriber"].transcribe,
                                   np.zeros(int(0.6 * TARGET_SR), dtype=np.float32))
    asyncio.ensure_future(precache(request.app, "", reset=True))
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
            sess.t_commit = time.monotonic()
            await sess.send({"type": "input_audio_buffer.committed", "item_id": sess.item_id})
            sess.on_commit()
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
    return web.json_response({"ok": True, "model": MODEL_ID})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    from aiohttp import web
    app = web.Application(client_max_size=64 * 1024 * 1024)
    # MLX keeps per-thread default device/stream state; touching the model from
    # a thread other than the one that initialised it fails under launchd with
    # "There is no Stream(gpu, 0) in current thread". So ONE worker thread owns
    # MLX end to end: it loads the model and runs every transcription.
    app["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    app["transcriber"] = app["executor"].submit(Transcriber, MODEL_ID).result()
    app.add_routes([
        web.get("/v1/realtime", handle_realtime),
        web.get("/realtime", handle_realtime),
        web.post("/v1/audio/transcriptions", handle_transcriptions),
        web.post("/inference", handle_transcriptions),
        web.get("/v1/models", handle_models),
        web.get("/health", handle_health),
    ])
    log.info("freeflow-stt listening on 127.0.0.1:%d (model=%s pause=%.1fs min_segment=%.1fs)", PORT, MODEL_ID, PAUSE_S, MIN_SEGMENT_S)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None, access_log=None)


if __name__ == "__main__":
    main()
