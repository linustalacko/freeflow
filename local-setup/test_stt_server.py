#!/usr/bin/env python3
"""Tests for stt_server.py.

  python3 test_stt_server.py            # protocol/contract tests with a fake transcriber (fast)
  python3 test_stt_server.py --real F.wav [--pace]
                                        # also stream a real 16 kHz WAV through a real
                                        # Parakeet model (--pace = send at real-time speed) and
                                        # report commit→final latency vs the batch endpoint

Run with the venv that has aiohttp (+ parakeet-mlx for --real):
  ~/.freeflow-ft/venv/bin/python local-setup/test_stt_server.py
"""
import asyncio
import base64
import concurrent.futures
import json
import os
import sys
import time
import unittest
import wave

import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*web.AppKey.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stt_server  # noqa: E402

from aiohttp import web, ClientSession, WSMsgType  # noqa: E402


class FakeTranscriber:
    """Deterministic stand-in: returns 'seg<N>' for the Nth call and sleeps
    `delay` so we can create in-flight jobs at commit time."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []

    def transcribe(self, x):
        if not np.any(x):
            return ""  # the session-open GPU warm-up (all zeros) — not a real call
        self.calls.append(len(x))
        time.sleep(self.delay)
        if len(x) < int(0.15 * stt_server.TARGET_SR):
            return ""
        return "seg%d" % len(self.calls)

    def settled_sentences(self, x, settle_s):
        # partial pre-transcription: report a sentence for every 3s of audio,
        # never counting toward the segment numbering above
        self.partials = getattr(self, "partials", 0) + 1
        n = int(len(x) / stt_server.TARGET_SR // 3)
        return " ".join("Partial sentence %d." % i for i in range(1, n + 1))


def make_app(transcriber):
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    app["transcriber"] = transcriber
    app.add_routes([web.get("/v1/realtime", stt_server.handle_realtime),
                    web.post("/v1/audio/transcriptions", stt_server.handle_transcriptions)])
    return app


def pcm16_b64(x_float, sr_out=24000, sr_in=16000):
    y = stt_server.resample(x_float.astype(np.float32), sr_in, sr_out)
    return base64.b64encode((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def tone_burst(seconds, sr=16000, amp=0.3):
    """Loud 'speech-like' signal the energy VAD counts as speech."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t) * (0.85 + 0.15 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def silence(seconds, sr=16000):
    return np.zeros(int(seconds * sr), dtype=np.float32)


async def run_session(port, chunks_16k, pace=False, chunk_s=0.1):
    """Drive the protocol like RealtimeTranscriptionService.swift. Returns
    (events, final_transcript_client_view, commit_to_final_s)."""
    events, final_parts, t_commit, t_final = [], [], None, None
    async with ClientSession() as cs:
        async with cs.ws_connect("http://127.0.0.1:%d/v1/realtime?intent=transcription" % port) as ws:
            await ws.send_str(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "x"}, "turn_detection": None}}}}))

            async def reader():
                nonlocal t_final
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    ev = json.loads(msg.data)
                    ev["_after_commit"] = t_commit is not None
                    events.append(ev)
                    if ev["type"] == "conversation.item.input_audio_transcription.completed":
                        final_parts.append(ev["transcript"])
                        if t_commit is not None:
                            t_final = time.monotonic()
                            return
            rt = asyncio.ensure_future(reader())
            step = int(chunk_s * 16000)
            for x in chunks_16k:
                for i in range(0, len(x), step):
                    await ws.send_str(json.dumps({"type": "input_audio_buffer.append", "audio": pcm16_b64(x[i:i + step])}))
                    if pace:
                        await asyncio.sleep(chunk_s)
                    else:
                        await asyncio.sleep(0)
            t_commit = time.monotonic()
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.wait_for(rt, timeout=60)
    return events, " ".join(p for p in final_parts if p).strip(), (t_final - t_commit) if t_final else None


class ContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeTranscriber(delay=0.0)
        self.precached = []
        async def fake_precache(request):
            self.precached.append((await request.json())["raw"])
            return web.json_response({"queued": True})
        self.router = web.Application()
        self.router.add_routes([web.post("/v1/precache", fake_precache)])
        self.router_runner = web.AppRunner(self.router)
        await self.router_runner.setup()
        rsite = web.TCPSite(self.router_runner, "127.0.0.1", 0)
        await rsite.start()
        stt_server.PRECACHE_URL = "http://127.0.0.1:%d/v1/precache" % rsite._server.sockets[0].getsockname()[1]
        self.app = make_app(self.fake)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        if self.app.get("http"):
            await self.app["http"].close()
        await self.runner.cleanup()
        await self.router_runner.cleanup()
        self.app["executor"].shutdown(wait=False)

    async def test_no_pause_means_single_final_completed_with_everything(self):
        events, text, _ = await run_session(self.port, [tone_burst(2.0)])
        completed = [e for e in events if e["type"].endswith("completed")]
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0]["_after_commit"])
        self.assertEqual(text, "seg1")
        self.assertTrue(any(e["type"] == "input_audio_buffer.committed" for e in events))

    async def test_pause_splits_segment_mid_speech_then_tail_at_commit(self):
        # 3.5s speech, 1.2s pause (> PAUSE_S after > MIN_SEGMENT_S) → segment 1 emitted
        # mid-speech; then 1.5s more speech → tail at commit.
        events, text, _ = await run_session(self.port, [tone_burst(3.5), silence(1.2), tone_burst(1.5)])
        completed = [e for e in events if e["type"].endswith("completed")]
        self.assertEqual(len(completed), 2, [e["type"] for e in events])
        self.assertFalse(completed[0]["_after_commit"], "first segment must be emitted before commit")
        self.assertTrue(completed[1]["_after_commit"])
        self.assertEqual(text, "seg1 seg2")
        await asyncio.sleep(0.2)
        pushed = [p for p in self.precached if p]  # drop the session-open reset
        self.assertIn("seg1", pushed, "mid-speech segment must be pushed to the router precache")
        self.assertNotIn("seg2", pushed, "the tail must not be precached")
        # segment 1 must contain roughly the first burst (3.5s minus the 0.3s guard),
        # tail the rest — nothing lost, nothing duplicated
        total = sum(self.fake.calls)
        self.assertAlmostEqual(total / 16000, 3.5 + 1.2 + 1.5, delta=0.15)

    async def test_inflight_segment_at_commit_is_merged_into_one_post_commit_event(self):
        # Slow transcriber: the mid-speech segment job is still running when commit
        # arrives. Client contract: exactly ONE completed after commit, carrying both.
        self.fake.delay = 0.6
        events, text, _ = await run_session(self.port, [tone_burst(3.5), silence(1.2), tone_burst(0.8)])
        after = [e for e in events if e["type"].endswith("completed") and e["_after_commit"]]
        before = [e for e in events if e["type"].endswith("completed") and not e["_after_commit"]]
        self.assertEqual(len(before), 0, "segment finished after commit must not be emitted separately")
        self.assertEqual(len(after), 1)
        self.assertEqual(text, "seg1 seg2")

    async def test_pauseless_speech_pushes_settled_partials(self):
        # 7s of continuous "speech": no segment cut, but partials should be posted
        # (every ~3s of speech) and the final commit must still be a single event.
        events, text, _ = await run_session(self.port, [tone_burst(7.0)], pace=False)
        await asyncio.sleep(0.3)
        partials = [p for p in self.precached if p.startswith("Partial sentence")]
        self.assertGreaterEqual(len(partials), 1, self.precached)
        completed = [e for e in events if e["type"].endswith("completed")]
        self.assertEqual(len(completed), 1)
        self.assertEqual(text, "seg1")

    async def test_short_tail_yields_empty_final_but_still_resolves(self):
        events, text, _ = await run_session(self.port, [tone_burst(3.5), silence(1.2), silence(0.05)])
        after = [e for e in events if e["type"].endswith("completed") and e["_after_commit"]]
        self.assertEqual(len(after), 1)
        self.assertEqual(text, "seg1")

    async def test_batch_endpoint_wav(self):
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((tone_burst(1.0) * 32767).astype("<i2").tobytes())
        from aiohttp import FormData
        fd = FormData()
        fd.add_field("file", buf.getvalue(), filename="a.wav", content_type="audio/wav")
        fd.add_field("model", "x")
        fd.add_field("response_format", "verbose_json")
        async with ClientSession() as cs:
            async with cs.post("http://127.0.0.1:%d/v1/audio/transcriptions" % self.port, data=fd) as r:
                self.assertEqual(r.status, 200)
                body = await r.json()
        self.assertEqual(body["text"], "seg1")
        self.assertAlmostEqual(body["duration"], 1.0, delta=0.01)


async def real_run(path, pace):
    print("loading model …")
    tr = stt_server.Transcriber(stt_server.MODEL_ID)
    app = make_app(tr)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    t0 = time.monotonic(); batch = tr.transcribe(x); tb = time.monotonic() - t0
    events, text, lat = await run_session(port, [x], pace=pace)
    segs = [e for e in events if e["type"].endswith("completed")]
    print("\naudio %.1fs  batch=%.2fs  realtime commit→final=%.3fs  (%d completed events, %d mid-speech)" % (
        len(x) / 16000, tb, lat, len(segs), sum(1 for e in segs if not e["_after_commit"])))
    print("  batch   :", batch[:200])
    print("  realtime:", text[:200])
    await runner.cleanup()


if __name__ == "__main__":
    if "--real" in sys.argv:
        i = sys.argv.index("--real")
        asyncio.run(real_run(sys.argv[i + 1], "--pace" in sys.argv))
    else:
        unittest.main(verbosity=2)
