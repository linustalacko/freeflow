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
import io
import json
import os
import re
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

from aiohttp import web, ClientSession, WSMsgType, FormData  # noqa: E402

SR = 16000


class FakeToken:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def word_tone(k, seconds=0.4, sr=SR, amp=0.3):
    """The k-th test word: a tone whose frequency encodes k, so a fake transcriber
    can name it from the audio alone (any window, any offset)."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * (200 + 20 * k) * t)).astype(np.float32)


def silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def speech(words, gap=0.15, lead=0.0):
    """`words` consecutive test words separated by `gap` seconds of silence."""
    parts = [silence(lead)] if lead else []
    for k in words:
        parts += [word_tone(k), silence(gap)]
    return np.concatenate(parts)


def tone_burst(seconds, sr=SR, amp=0.3):
    """Loud 'speech-like' signal (kept for the batch test)."""
    return speech(list(range(int(seconds / 0.55) + 1)))[: int(seconds * sr)]


class FakeTranscriber:
    """Deterministic stand-in for Parakeet: finds tone runs in the audio and
    names each by its frequency (see word_tone), with real timestamps. `delay`
    simulates GPU time so tests can create in-flight jobs at commit time."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []      # audio lengths of transcribe_tokens calls
        self.pokes = 0       # all-zero clips (session-open poke / heartbeat)

    def _runs(self, x):
        frame = int(0.02 * SR)
        n = len(x) // frame
        rms = np.array([float(np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2))) for i in range(n)])
        runs, start = [], None
        for i, v in enumerate(rms):
            if v > 0.02 and start is None:
                start = i
            elif v <= 0.02 and start is not None:
                runs.append((start * frame, i * frame))
                start = None
        if start is not None:
            runs.append((start * frame, n * frame))
        return [r for r in runs if r[1] - r[0] >= int(0.1 * SR)]

    def transcribe_tokens(self, x):
        if not np.any(x):
            self.pokes += 1
            return []
        self.calls.append(len(x))
        if self.delay:
            time.sleep(self.delay)
        toks = []
        for a, b in self._runs(x):
            seg = x[a:b]
            spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freq = float(np.argmax(spectrum)) * SR / len(seg)
            k = int(round((freq - 200) / 20.0))
            toks.append(FakeToken(" w%d" % k, a / SR, b / SR))
        return toks

    def transcribe(self, x):
        if not np.any(x):
            self.pokes += 1
            return ""
        return stt_server.words_text(stt_server.words_from_tokens(self.transcribe_tokens(x)))


def make_app(transcriber):
    return stt_server.make_app(transcriber, concurrent.futures.ThreadPoolExecutor(max_workers=1))


def pcm16_b64(x_float, sr_out=24000, sr_in=SR):
    y = stt_server.resample(x_float.astype(np.float32), sr_in, sr_out)
    return base64.b64encode((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()).decode()


async def run_session(port, chunks_16k, pace=False, chunk_s=0.1, speed=1.0):
    """Drive the protocol like RealtimeTranscriptionService.swift. Returns
    (events, final_transcript_client_view, commit_to_final_s). `pace` sends the
    audio at real-time speed divided by `speed`. Like the mic, the 24 kHz stream
    is one continuous signal that is merely *sent* in pieces."""
    events, final_parts, t_commit, t_final = [], [], None, None
    stream24 = stt_server.resample(np.concatenate(chunks_16k).astype(np.float32), SR, 24000)
    async with ClientSession() as cs:
        async with cs.ws_connect("http://127.0.0.1:%d/v1/realtime?intent=transcription" % port) as ws:
            await ws.send_str(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "x"}, "turn_detection": None}}}}))

            async def reader():
                nonlocal t_final
                committed_item = None
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    ev = json.loads(msg.data)
                    ev["_after_commit"] = t_commit is not None
                    events.append(ev)
                    if ev["type"] == "input_audio_buffer.committed":
                        committed_item = ev.get("item_id")
                    if ev["type"] == "conversation.item.input_audio_transcription.completed":
                        final_parts.append(ev["transcript"])
                        if t_commit is not None and committed_item is not None and ev.get("item_id") == committed_item:
                            t_final = time.monotonic()
                            return
            rt = asyncio.ensure_future(reader())
            step = int(chunk_s * 24000)
            for i in range(0, len(stream24), step):
                piece = (np.clip(stream24[i:i + step], -1, 1) * 32767).astype("<i2").tobytes()
                await ws.send_str(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(piece).decode()}))
                if pace:
                    await asyncio.sleep(chunk_s / speed)
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
        self.saved = {k: getattr(stt_server, k) for k in ("HEARTBEAT_S", "LEFT_CONTEXT_S", "MIN_SETTLE_S")}
        stt_server.HEARTBEAT_S = 0
        self.app = make_app(self.fake)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()
        await self.router_runner.cleanup()
        self.app["executor"].shutdown(wait=False)
        for k, v in self.saved.items():
            setattr(stt_server, k, v)

    @staticmethod
    def completed(events, after_commit):
        return [e for e in events if e["type"].endswith("completed") and e["_after_commit"] == after_commit]

    async def test_short_speech_means_single_final_completed_with_everything(self):
        events, text, _ = await run_session(self.port, [speech([1, 2, 3])])
        self.assertEqual(len(self.completed(events, False)), 0)
        self.assertEqual(len(self.completed(events, True)), 1)
        self.assertEqual(text, "w1 w2 w3")
        self.assertTrue(any(e["type"] == "input_audio_buffer.committed" for e in events))

    async def test_continuous_speech_settles_words_mid_speech_and_only_the_tail_at_commit(self):
        # 16 words ≈ 8.8s of pause-less speech: words the last two windows agree
        # on (and that ended ≥1s before the window end) are emitted while the
        # user is still talking; the final event carries only what is left.
        words = list(range(16))
        events, text, _ = await run_session(self.port, [speech(words)], pace=True, speed=8)
        mid = self.completed(events, False)
        self.assertGreaterEqual(len(mid), 1, [e["type"] for e in events])
        self.assertEqual(len(self.completed(events, True)), 1)
        self.assertEqual(text, " ".join("w%d" % k for k in words), "nothing lost, nothing duplicated")
        final = self.completed(events, True)[0]["transcript"]
        final_id = self.completed(events, True)[0]["item_id"]
        self.assertNotIn(final_id, [e["item_id"] for e in mid])
        self.assertEqual(len({e["item_id"] for e in mid}), len(mid))
        self.assertLess(len(final.split()), len(words), "the tail must not re-transcribe settled words")
        await asyncio.sleep(0.2)
        pushed = [p for p in self.precached if p]
        self.assertEqual(pushed, [e["transcript"] for e in mid], "every settled chunk is precached, in order")
        self.assertNotIn(final, pushed, "the tail must not be precached")
        # the commit window is bounded: left context + tail, not the whole session
        self.assertLess(self.fake.calls[-1] / SR, stt_server.LEFT_CONTEXT_S + 5.0)

    async def test_partial_finishing_after_commit_is_discarded(self):
        # Slow transcriber: a mid-speech window is still running when commit
        # arrives. Client contract: exactly ONE completed after commit, carrying
        # everything not yet emitted — the late partial must not be emitted.
        self.fake.delay = 0.25
        words = list(range(10))
        events, text, _ = await run_session(self.port, [speech(words)], pace=True, speed=8)
        after = self.completed(events, True)
        self.assertEqual(len(after), 1)
        self.assertEqual(text, " ".join("w%d" % k for k in words))

    async def test_long_session_keeps_every_word_after_discarding_old_audio(self):
        words = list(range(90))
        events, text, _ = await run_session(self.port, [speech(words)], pace=True, speed=8)
        self.assertGreater(len(self.completed(events, False)), 5)
        self.assertEqual(text, " ".join("w%d" % k for k in words))
        self.assertLess(max(self.fake.calls) / SR, stt_server.LEFT_CONTEXT_S + 7)

    async def test_silence_only_windows_are_skipped(self):
        events, text, _ = await run_session(self.port, [speech([1, 2]), silence(6.0)], pace=True, speed=8)
        # 8s of audio would schedule ~3 windows; the digital-silence ones cost nothing
        self.assertLessEqual(len(self.fake.calls), 3)
        self.assertEqual(text, "w1 w2")

    async def test_short_tail_yields_empty_final_but_still_resolves(self):
        events, text, _ = await run_session(self.port, [silence(0.05)])
        after = self.completed(events, True)
        self.assertEqual(len(after), 1)
        self.assertEqual(text, "")

    async def test_heartbeat_touches_the_model_only_while_idle(self):
        stt_server.HEARTBEAT_S = 0.15
        state = self.app["state"]
        state.last_work = time.monotonic() - 10
        task = asyncio.ensure_future(stt_server.heartbeat(self.app))
        try:
            await asyncio.sleep(0.5)
            self.assertGreaterEqual(self.fake.pokes, 1, "idle → weights are touched")
            # an open session suspends it (the user is dictating; the GPU is busy for them)
            state.sessions = 1
            state.last_work = time.monotonic() - 10
            before = self.fake.pokes
            await asyncio.sleep(0.4)
            self.assertEqual(self.fake.pokes, before, "no heartbeat while a session is open")
        finally:
            task.cancel()
            state.sessions = 0

    async def test_batch_endpoint_wav(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((speech([4, 5]) * 32767).astype("<i2").tobytes())
        fd = FormData()
        fd.add_field("file", buf.getvalue(), filename="a.wav", content_type="audio/wav")
        fd.add_field("model", "x")
        fd.add_field("response_format", "verbose_json")
        async with ClientSession() as cs:
            async with cs.post("http://127.0.0.1:%d/v1/audio/transcriptions" % self.port, data=fd) as r:
                self.assertEqual(r.status, 200)
                body = await r.json()
        self.assertEqual(body["text"], "w4 w5")
        self.assertAlmostEqual(body["duration"], 1.1, delta=0.01)

    async def test_health_reports_memory(self):
        async with ClientSession() as cs:
            async with cs.get("http://127.0.0.1:%d/health" % self.port) as r:
                body = await r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["sessions"], 0)
        self.assertEqual(body["source_sha"], stt_server.source_sha(), "health reports which source is running")


class SettleAndSpliceTests(unittest.TestCase):
    W = stt_server.Word

    def test_agreement_ignores_punctuation_and_needs_horizon(self):
        prev = [self.W(" so", 0.0, 0.2), self.W(" the", 0.3, 0.4), self.W(" deploy", 0.5, 0.9), self.W(" is", 1.0, 1.1)]
        cur = [self.W(" So", 0.0, 0.2), self.W(" the", 0.3, 0.4), self.W(" deploy,", 0.5, 0.9), self.W(" is", 1.0, 1.1),
               self.W(" done", 1.2, 1.5)]
        # window ends at 2.6s: "is" ended at 1.1 (≥1s before) → settle 4 words; "done" too recent
        self.assertEqual(stt_server.agreed_settle_count(prev, cur, 2.6, 0.0), 4)
        # nothing agreed → nothing settles; a too-short span → nothing either
        self.assertEqual(stt_server.agreed_settle_count(prev, [self.W(" no", 0.0, 0.2)], 2.6, 0.0), 0)
        self.assertEqual(stt_server.agreed_settle_count(prev, cur, 2.6, 0.5), 0)

    def test_splice_anchors_on_words_not_time(self):
        settled = [self.W(" hello", 0.0, 0.4), self.W(" there", 0.5, 0.9), self.W(" world", 1.0, 1.4)]
        # the commit decode re-tokenises the boundary ("world" → "world!") and shifts it 0.3s
        commit = [self.W(" hello", 0.0, 0.4), self.W(" there", 0.5, 0.9), self.W(" world!", 1.1, 1.7),
                  self.W(" and", 1.8, 2.0), self.W(" more", 2.1, 2.4)]
        tail = stt_server.splice_tail(commit, settled, cut=1.5)
        self.assertEqual(stt_server.words_text(tail), "and more")
        # no anchor found → timestamp fallback, with the boundary duplicate dropped
        tail = stt_server.splice_tail([self.W(" world", 1.45, 1.7), self.W(" and", 1.8, 2.0)], settled[-1:], cut=1.5)
        self.assertEqual(stt_server.words_text(tail), "and")
        self.assertEqual(stt_server.words_text(stt_server.splice_tail(commit, [], cut=1.5)), "and more")

    def test_splice_prefers_the_anchor_match_closest_to_the_cut(self):
        settled = [self.W(" and", 0.5, 0.7), self.W(" the", 0.8, 0.9)]
        commit = [self.W(" and", 0.5, 0.7), self.W(" the", 0.8, 0.9), self.W(" thing", 1.0, 1.3),
                  self.W(" and", 1.4, 1.6), self.W(" the", 1.7, 1.8), self.W(" rest", 1.9, 2.2)]
        tail = stt_server.splice_tail(commit, settled, cut=0.95)
        self.assertEqual(stt_server.words_text(tail), "thing and the rest", "the later repeat must not swallow 'thing'")


class StreamResamplerTests(unittest.TestCase):
    def test_arbitrary_packet_sizes_preserve_phase_and_sample_count(self):
        for rate in (24000, 44100, 48000, 8000):
            rng = np.random.default_rng(42)
            x = rng.normal(0, 0.1, rate + 137).astype(np.float32)
            whole = stt_server.resample(x, rate)
            for sizes in ([1024], [511], [1, 7, 1024, 83, 3000]):
                with self.subTest(rate=rate, sizes=sizes):
                    rs = stt_server.StreamResampler(rate)
                    parts, pos, i = [], 0, 0
                    while pos < len(x):
                        size = sizes[i % len(sizes)]
                        parts.append(rs.feed(x[pos:pos + size]))
                        pos += size
                        i += 1
                    parts.append(rs.feed(np.zeros(0, dtype=np.float32), final=True))
                    streamed = np.concatenate(parts)
                    self.assertEqual(len(streamed), len(whole))
                    np.testing.assert_allclose(streamed, whole, atol=1e-6)
                    self.assertEqual(len(rs.feed(np.zeros(0, dtype=np.float32), final=True)), 0)

    def test_chunked_output_matches_whole_clip_resampling(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(24000 * 3).astype(np.float32) * 0.1
        whole = stt_server.resample(x, 24000, 16000)
        rs = stt_server.StreamResampler(24000)
        parts = [rs.feed(x[i:i + 2400]) for i in range(0, len(x), 2400)]
        parts.append(rs.feed(np.zeros(0, dtype=np.float32), final=True))
        streamed = np.concatenate(parts)
        self.assertEqual(len(streamed), len(whole))
        # everything but the very last few samples (no right context at flush) is identical
        self.assertTrue(np.allclose(streamed[:-200], whole[:-200], atol=1e-4))
        # stateless per-message resampling is what we are replacing: visibly different
        naive = np.concatenate([stt_server.resample(x[i:i + 2400], 24000, 16000) for i in range(0, len(x), 2400)])
        self.assertFalse(np.allclose(naive[:-200], whole[:-200], atol=1e-4))

    def test_same_rate_is_passthrough(self):
        rs = stt_server.StreamResampler(16000)
        x = np.ones(100, dtype=np.float32)
        self.assertIs(rs.feed(x), x)


class AudioWindowBufferTests(unittest.TestCase):
    def test_long_session_retains_context_without_losing_samples(self):
        buf = stt_server.AudioWindowBuffer()
        chunk = np.arange(1024, dtype=np.float32)
        end = 0
        for _ in range(10000):
            buf.append(chunk + end)
            end += len(chunk)
            buf.discard_before(max(0, end - 160000))
        self.assertEqual(buf.end, end)
        self.assertEqual(buf.start, end - 160000)
        np.testing.assert_array_equal(buf.snapshot(buf.start), np.arange(end - 160000, end, dtype=np.float32))
        self.assertLessEqual(buf.storage.nbytes, 4 * 1024 * 1024)

    def test_inflight_snapshot_survives_compaction_and_growth(self):
        buf = stt_server.AudioWindowBuffer()
        original = np.arange(300000, dtype=np.float32)
        buf.append(original)
        snapshot = buf.snapshot(0)
        buf.discard_before(299000)
        buf.append(np.ones(1000000, dtype=np.float32))
        np.testing.assert_array_equal(snapshot, original)
        self.assertEqual(buf.end, 1300000)
        self.assertEqual(buf.snapshot(299000)[999], original[-1])


class WordMergeTests(unittest.TestCase):
    def test_subword_and_punctuation_tokens_attach_to_their_word(self):
        toks = [FakeToken(" inv", 0.0, 0.2), FakeToken("ite", 0.2, 0.4), FakeToken("?", 0.4, 0.5), FakeToken(" Ok", 0.9, 1.1)]
        words = stt_server.words_from_tokens(toks, offset=10.0)
        self.assertEqual([w.text for w in words], [" invite?", " Ok"])
        self.assertAlmostEqual(words[0].start, 10.0)
        self.assertAlmostEqual(words[0].end, 10.5)
        self.assertEqual(stt_server.words_text(words), "invite? Ok")


async def real_run(path, pace):
    print("loading model …")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    tr = executor.submit(stt_server.Transcriber, stt_server.MODEL_ID).result()
    app = stt_server.make_app(tr, executor)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    t0 = time.monotonic(); batch = tr.transcribe(x); tb = time.monotonic() - t0
    events, text, lat = await run_session(port, [x], pace=pace)
    segs = [e for e in events if e["type"].endswith("completed")]
    print("\naudio %.1fs  batch=%.2fs  realtime commit→final=%.3fs  (%d completed events, %d mid-speech)" % (
        len(x) / SR, tb, lat, len(segs), sum(1 for e in segs if not e["_after_commit"])))
    print("  batch   :", batch[:200])
    print("  realtime:", text[:200])
    await runner.cleanup()


if __name__ == "__main__":
    if "--real" in sys.argv:
        i = sys.argv.index("--real")
        asyncio.run(real_run(sys.argv[i + 1], "--pace" in sys.argv))
    else:
        unittest.main(verbosity=2)
