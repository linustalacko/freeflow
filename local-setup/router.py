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
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "llama3.1:8b")
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "8787"))

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

        # 1) Try Groq (online + fast). 2) Fall back to local on ANY failure
        #    (offline, 4xx like revoked-key/out-of-credits, 5xx, timeout).
        try:
            resp = call_groq(body)
            print("route=groq model=%s" % body.get("model"), flush=True)
            self._send(200, resp, route="groq")
            return
        except Exception as e:
            print("groq failed (%s) -> falling back to local" % type(e).__name__, flush=True)

        try:
            resp = call_local(body)
            print("route=local model=%s" % LOCAL_MODEL, flush=True)
            self._send(200, resp, route="local")
        except Exception as e:
            print("local also failed: %r" % e, flush=True)
            self._send(502, json.dumps({"error": {"message": "router: hosted and local both unavailable"}}).encode())


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    print("freeflow-router listening on 127.0.0.1:%d (hosted=%s local=%s)" % (LISTEN_PORT, GROQ_URL, LOCAL_MODEL), flush=True)
    srv.serve_forever()
