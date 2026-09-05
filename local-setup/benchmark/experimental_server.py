#!/usr/bin/env python3
"""Isolated experiment server. Never used by the app's deployment scripts."""
import asyncio
import base64
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["ROUTER_PRECACHE_URL"] = ""
os.environ["STT_HEARTBEAT_S"] = "0"
import stt_server as stt
import numpy as np

from experimental_decoders import install


class ExperimentalTranscriber(stt.Transcriber):
    def __init__(self, model_id):
        super().__init__(model_id)
        dtype = os.environ.get("EXP_DTYPE", "bfloat16")
        if dtype != "bfloat16":
            if dtype not in {"float16", "float32"}:
                raise ValueError("unsupported experiment dtype")
            self.model.set_dtype(getattr(self.mx, dtype))
        bits = int(os.environ.get("EXP_QUANTIZE_BITS", "0"))
        if bits:
            if bits not in {4, 8}:
                raise ValueError("unsupported experiment quantization")
            import mlx.nn as nn
            nn.quantize(self.model.encoder, bits=bits)
        install(self.model, os.environ.get("EXP_DECODER", "baseline"))
        if os.environ.get("EXP_COMPILE_ENCODER") == "1":
            original = self.model.encoder.__call__
            compiled = self.mx.compile(original)
            # Special methods resolve on the class, not the instance. A proxy
            # delegates attributes used by native streaming attention switching.
            class EncoderProxy:
                def __call__(self, *args, **kwargs):
                    return compiled(*args, **kwargs)
                def __getattr__(self, name):
                    return getattr(original.__self__, name)
            self.model.encoder = EncoderProxy()

    def _generate(self, audio):
        # Stabilize encoder shapes for compilation. Padding is computed inside
        # inference timing; it never changes when the replay's audio ends.
        bucket = round(float(os.environ.get("EXP_PAD_BUCKET_S", "0")) * stt.TARGET_SR)
        if bucket and len(audio) % bucket:
            audio = np.pad(audio, (0, bucket - len(audio) % bucket))
        return super()._generate(audio)


class CachedSession(stt.RealtimeSession):
    """Exercise parakeet-mlx's native state cache behind FreeFlow's wire contract.

    All model operations run on the existing single inference worker. This
    experimental endpoint accepts one session at a time; the runner is serial.
    """
    def __init__(self, ws, app):
        super().__init__(ws, app)
        self.pending = np.zeros(0, dtype=np.float32)
        self.stream = None
        self.native_error = False
        self.chunk_samples = int(float(os.environ.get("EXP_CHUNK_S", "0.5")) * stt.TARGET_SR)
        self.right_context = int(os.environ.get("EXP_RIGHT_CONTEXT", "4"))
        self.depth = int(os.environ.get("EXP_DEPTH", "1"))

    def _add(self, audio):
        if self.native_error:
            return
        try:
            transcriber = self.app["transcriber"]
            if self.stream is None:
                self.stream = transcriber.model.transcribe_stream(
                    context_size=(128, self.right_context), depth=self.depth,
                    keep_original_attention=os.environ.get("EXP_KEEP_ATTENTION") == "1")
                self.stream.__enter__()
            self.stream.add_audio(transcriber.mx.array(audio))
        except Exception:
            self.native_error = True

    def on_append(self, event):
        if self.committed:
            return
        audio = self.resampler.feed(stt.pcm16_to_float(base64.b64decode(event.get("audio", ""))))
        self.pending = np.concatenate([self.pending, audio])
        while len(self.pending) >= self.chunk_samples:
            chunk = self.pending[:self.chunk_samples].copy()
            self.pending = self.pending[self.chunk_samples:]
            self.app["executor"].submit(self._add, chunk)

    def _finish(self, audio):
        try:
            # Flush the last actual samples, including sub-hop/subsampling tails.
            # Padding is processed after commit, never counted as user speech.
            pad = int(float(os.environ.get("EXP_FINAL_PAD_S", "0.08")) * stt.TARGET_SR)
            self._add(np.pad(audio, (0, max(pad, 1280 - len(audio)))))
            if self.native_error or self.stream is None:
                raise RuntimeError("native_stream_failed")
            return self.stream.result.text
        finally:
            if self.stream is not None:
                self.stream.__exit__(None, None, None)
                self.stream = None

    def on_commit(self):
        if self.committed:
            return
        self.committed = True
        held = self.resampler.feed(np.zeros(0, dtype=np.float32), final=True)
        audio = np.concatenate([self.pending, held])
        future = self.loop.run_in_executor(self.app["executor"], self._finish, audio)
        future.add_done_callback(lambda result: self.loop.call_soon_threadsafe(self._final, result))

    def _final(self, future):
        try:
            transcript = future.result()
            event = {"type": "conversation.item.input_audio_transcription.completed",
                     "item_id": self.item_id, "transcript": transcript}
        except Exception:
            event = {"type": "error", "error": {"code": "native_stream_failed", "message": "experimental decoder failed"}}
        asyncio.ensure_future(self.send(event))


if __name__ == "__main__":
    stt.Transcriber = ExperimentalTranscriber
    if os.environ.get("EXP_SESSION") == "cached":
        stt.RealtimeSession = CachedSession
    elif os.environ.get("EXP_SESSION") == "guarded":
        from experimental_guarded import GuardedSession
        stt.RealtimeSession = GuardedSession
    # The experiment runner records only scores. Suppress provider/log payloads.
    logging.disable(logging.CRITICAL)
    stt.main()
