#!/usr/bin/env python3
"""
FreeFlow LLM router — sits on localhost and gives FreeFlow one stable endpoint.

  FreeFlow ──▶ this router ──▶ try Groq gpt-oss (fast, when online)
                           └──▶ on offline / error / timeout ──▶ local Ollama llama3.1:8b

Only the post-processing/context LLM goes through here. Transcription talks to
whisper-server directly. Groq key comes from the GROQ_API_KEY env (set in the
launchd plist).
"""
import json
import os
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3.1:8b")
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "8787"))
# FORCE_LOCAL=1 → skip the hosted provider entirely and use the local model
# (e.g. the on-device fine-tuned Qwen). Unset/0 → normal Groq-first behaviour.
FORCE_LOCAL = os.environ.get("FORCE_LOCAL", "").strip().lower() not in ("", "0", "false", "no")

# A short timeout so "no internet" falls through to local quickly instead of hanging.
HOSTED_TIMEOUT = float(os.environ.get("HOSTED_TIMEOUT", "12"))
LOCAL_TIMEOUT = float(os.environ.get("LOCAL_TIMEOUT", "120"))
# Groq's WAF 403s the default python-urllib User-Agent, so we send a normal one.
UA = "FreeFlow-Router/1.0"

MODELS = {
    "object": "list",
    "data": [
        {"id": "openai/gpt-oss-20b", "object": "model", "owned_by": "groq"},
        {"id": "openai/gpt-oss-120b", "object": "model", "owned_by": "groq"},
        {"id": LOCAL_MODEL, "object": "model", "owned_by": "ollama"},
    ],
}


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def call_groq(body):
    headers = {
        "Authorization": "Bearer " + GROQ_KEY,
        "Content-Type": "application/json",
        "User-Agent": UA,
    }
    return _post(GROQ_URL, body, headers, HOSTED_TIMEOUT)


def warm_local_if_offline():
    # Quick reachability probe of the hosted provider; if it fails, trigger an
    # Ollama model load (empty generate) so it's resident before cleanup runs.
    try:
        probe = urllib.request.Request(
            GROQ_URL.replace("/chat/completions", "/models"),
            headers={"Authorization": "Bearer " + GROQ_KEY, "User-Agent": UA})
        urllib.request.urlopen(probe, timeout=1.5)
        return  # hosted path is up; no local preload needed
    except Exception:
        pass
    try:
        load = json.dumps({"model": LOCAL_MODEL, "prompt": "", "stream": False}).encode()
        req = urllib.request.Request(
            OLLAMA_URL.replace("/v1/chat/completions", "/api/generate"),
            data=load, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
        print("warm: preloaded %s (hosted unreachable)" % LOCAL_MODEL, flush=True)
    except Exception as e:
        print("warm: local preload failed: %r" % e, flush=True)


def call_local(body):
    payload = dict(body)
    payload["model"] = LOCAL_MODEL
    # Strip hosted/gpt-oss-only params the local model doesn't need.
    for k in ("reasoning_effort", "include_reasoning", "reasoning", "max_completion_tokens"):
        payload.pop(k, None)
    return _post(OLLAMA_URL, payload, {"Content-Type": "application/json"}, LOCAL_TIMEOUT)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet; launchd captures stderr if we print() explicitly

    def _send(self, code, data, route=None):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if route:
            self.send_header("X-FreeFlow-Route", route)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, json.dumps(MODELS).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        # /warm: called by FreeFlow on hotkey-down. If the hosted provider looks
        # unreachable (offline), preload the local model NOW — while the user is
        # still speaking — so cleanup doesn't pay the model-load cold start.
        if self.path.rstrip("/").endswith("/warm"):
            threading.Thread(target=warm_local_if_offline, daemon=True).start()
            self._send(200, b'{"status":"warming"}')
            return

        if "/chat/completions" not in self.path:
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except Exception:
            self._send(400, b'{"error":"bad json"}')
            return

        # Backend order. FORCE_LOCAL → local first with Groq as a safety net (so a
        # dead local model never breaks dictation); otherwise Groq first, local
        # fallback. Each is tried in turn on ANY failure (offline, 4xx, 5xx, timeout).
        backends = ([("local", call_local), ("groq", call_groq)] if FORCE_LOCAL
                    else [("groq", call_groq), ("local", call_local)])
        for name, fn in backends:
            try:
                resp = fn(body)
                print("route=%s" % name, flush=True)
                self._send(200, resp, route=name)
                return
            except Exception as e:
                print("%s failed (%s) -> next backend" % (name, type(e).__name__), flush=True)
        self._send(502, json.dumps({"error": {"message": "router: all backends unavailable"}}).encode())


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    print("freeflow-router listening on 127.0.0.1:%d (hosted=%s local=%s)" % (LISTEN_PORT, GROQ_URL, LOCAL_MODEL), flush=True)
    srv.serve_forever()
