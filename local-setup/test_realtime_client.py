#!/usr/bin/env python3
"""Exercise the production Swift WebSocket client against local scripted peers.

No microphone, permission prompts, credentials or AI providers. The compiled
probe and local listener are removed when the test finishes.
Run with the local Python environment (aiohttp required).
"""
import asyncio
import pathlib
import subprocess
import tempfile

from aiohttp import web

ROOT = pathlib.Path(__file__).resolve().parent.parent


async def peer(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    scenario = request.query["scenario"]
    async for message in ws:
        event = message.json()
        if event["type"] != "input_audio_buffer.commit":
            continue
        if scenario == "closed":
            await ws.close()
            break
        await ws.send_json({"type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "earlier", "transcript": "Keep"})
        await ws.send_json({"type": "input_audio_buffer.committed", "item_id": "final"})
        if scenario == "malformed":
            await ws.send_json({"type": "conversation.item.input_audio_transcription.completed",
                                "item_id": "final"})
        if scenario not in ("ordered", "duplicate"):
            continue
        await ws.send_json({"type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "inflight", "transcript": "every"})
        await asyncio.sleep(0.12)
        await ws.send_json({"type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "final", "transcript": "word."})
    return ws


async def run(probe):
    app = web.Application()
    app.router.add_get("/v1/realtime", peer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        for scenario in ("ordered", "silent", "malformed", "closed", "cancel", "duplicate"):
            proc = await asyncio.create_subprocess_exec(
                str(probe), "http://127.0.0.1:%d?scenario=%s" % (port, scenario), scenario)
            try:
                code = await asyncio.wait_for(proc.wait(), timeout=5)
                if code:
                    raise AssertionError("Swift probe failed: " + scenario)
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="freeflow-realtime-test-") as scratch:
        probe = pathlib.Path(scratch) / "probe"
        subprocess.run(["swiftc", "-parse-as-library", "-warnings-as-errors", "-o", str(probe),
                        str(ROOT / "Sources/RealtimeTranscriptState.swift"),
                        str(ROOT / "Sources/RealtimeTranscriptionService.swift"),
                        str(ROOT / "local-setup/RealtimeClientProbe.swift")], check=True)
        asyncio.run(run(probe))
