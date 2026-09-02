#!/usr/bin/env python3
"""Regression tests for router.py — run with `python3 local-setup/test_router.py`.

Spins up fake "Groq" and "local" HTTP servers on localhost so the routing,
rate-limit prediction and deadline logic can be exercised without a network."""
import json
import os
import sys
import threading
import time
import unittest
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class FakeBackend(ThreadingHTTPServer):
    """Scriptable OpenAI-compatible endpoint. `script[model]` is a callable
    (body) -> (status, headers, json_body) or a static tuple."""

    def __init__(self):
        self.script = {}
        self.requests = []
        self.native = []
        self.lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.daemon_threads = True
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return "http://127.0.0.1:%d/v1/chat/completions" % self.server_address[1]

    def seen(self, model):
        with self.lock:
            return [r for r in self.requests if r.get("model") == model]


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.endswith("/api/generate"):
            with self.server.lock:
                self.server.native.append(body)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        with self.server.lock:
            self.server.requests.append(body)
        entry = self.server.script.get(body.get("model"), (404, {}, {"error": "no such model"}))
        status, headers, payload = entry(body) if callable(entry) else entry
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def ok(text):
    return (200, {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7000"},
            {"choices": [{"message": {"role": "assistant", "content": text}}]})


def rate_limited(retry_after="30"):
    return (429, {"retry-after": retry_after, "x-ratelimit-limit-tokens": "8000",
                  "x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "30s"},
            {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}})


GROQ = FakeBackend()
LOCAL = FakeBackend()

os.environ.update({
    "GROQ_API_KEY": "test-key",
    "GROQ_URL": GROQ.url,
    "GROQ_FALLBACK_MODELS": "fallback-a,fallback-b",
    "LOCAL_URL": LOCAL.url,
    "LOCAL_MODEL": "local-model",
    "HOSTED_TIMEOUT": "3",
    "LOCAL_TIMEOUT": "3",
    "ROUTER_PORT": "0",
})
sys.path.insert(0, HERE)
import router  # noqa: E402  (must import after env is set)
import app_prompt  # noqa: E402

ROUTER = ThreadingHTTPServer(("127.0.0.1", 0), router.Handler)
ROUTER.daemon_threads = True
threading.Thread(target=ROUTER.serve_forever, daemon=True).start()
ROUTER_URL = "http://127.0.0.1:%d/v1" % ROUTER.server_address[1]


def post(payload, deadline_ms=None, path="/chat/completions"):
    headers = {"Content-Type": "application/json"}
    if deadline_ms is not None:
        headers[router.DEADLINE_HEADER] = str(deadline_ms)
    req = urllib.request.Request(ROUTER_URL + path, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.getheaders()), json.loads(r.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read()), time.monotonic() - t0


DEFAULT_SYS = "You are a literal dictation cleanup layer for short messages.\n\nHard contract: ..."


def cleanup_payload(model="primary", text="um so hello world", context="", system=DEFAULT_SYS):
    """Byte-for-byte the shape PostProcessingService.process sends (see app_prompt.py)."""
    body = app_prompt.cleanup_request(system, text, context, model=model)
    body["max_completion_tokens"] = 400
    return body


def image_payload(model="ctx-model", max_completion_tokens=512):
    """The context-inference call AppContextService makes at hotkey-down."""
    return {"model": model, "temperature": 0.2, "max_completion_tokens": max_completion_tokens,
            "messages": [{"role": "system", "content": "You are a context synthesis assistant."},
                         {"role": "user", "content": [
                             {"type": "text", "text": "Analyze the screenshot plus metadata."},
                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 120000}}]}]}


def app_payload(model="primary", text="um so hello world"):
    return {"model": model, "temperature": 0, "max_completion_tokens": 400,
            "reasoning_effort": "low", "include_reasoning": False,
            "messages": [{"role": "system", "content": "clean it"},
                         {"role": "user", "content": "RAW_TRANSCRIPTION: " + text}]}


class RouterTests(unittest.TestCase):
    def setUp(self):
        GROQ.script.clear()
        LOCAL.script.clear()
        with GROQ.lock:
            GROQ.requests.clear()
        with LOCAL.lock:
            LOCAL.requests.clear()
            LOCAL.native.clear()
        router.RATE = router.RateLimits()
        router.FORCE_LOCAL = False
        router._local_warm_until = 0.0
        router._last_cleanup_system = None
        with router._precache_lock:
            router._precache.clear()
        router.LOCAL_PROMPT_FORMAT = "passthrough"
        router.DET_CLEAN = False  # most tests exercise the LLM plumbing; see test_deterministic_*
        router.PRECACHE_LLM = True  # the LLM plumbing tests pre-clean chunks with the fake model
        router._local_last_call = 0.0
        router._reset_hosted_breaker()
        router._dns_cache.clear()

    def test_primary_success(self):
        GROQ.script["primary"] = ok("Hello world.")
        status, headers, body, _ = post(app_payload())
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:primary")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello world.")

    def test_429_falls_through_to_next_groq_bucket_not_local(self):
        GROQ.script["primary"] = rate_limited()
        GROQ.script["fallback-a"] = ok("From fallback A.")
        status, headers, body, _ = post(app_payload())
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:fallback-a")
        self.assertEqual(LOCAL.requests, [], "local must not be hit while a hosted bucket has room")

    def test_predicted_429_skips_round_trip(self):
        GROQ.script["primary"] = rate_limited(retry_after="30")
        GROQ.script["fallback-a"] = ok("A")
        post(app_payload())
        self.assertEqual(len(GROQ.seen("primary")), 1)
        # Second request inside the retry-after window: primary must be skipped
        # without a network call, fallback-a serves.
        status, headers, _, _ = post(app_payload())
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:fallback-a")
        self.assertEqual(len(GROQ.seen("primary")), 1, "primary should not be retried while blocked")

    def test_low_remaining_tokens_predicts_429(self):
        # First call succeeds but reports the bucket nearly empty; second call
        # (cost ≈ prompt + 400 reserved) should be routed around primary.
        GROQ.script["primary"] = (200, {"x-ratelimit-limit-tokens": "8000",
                                        "x-ratelimit-remaining-tokens": "50"},
                                  {"choices": [{"message": {"content": "first"}}]})
        GROQ.script["fallback-a"] = ok("second")
        post(app_payload())
        _, headers, _, _ = post(app_payload())
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:fallback-a")
        self.assertEqual(len(GROQ.seen("primary")), 1)

    def test_all_hosted_fail_then_local_gets_adapted_payload(self):
        GROQ.script["primary"] = rate_limited()
        GROQ.script["fallback-a"] = rate_limited()
        GROQ.script["fallback-b"] = (500, {}, {"error": "boom"})
        LOCAL.script["local-model"] = ok("local answer")
        status, headers, body, _ = post(app_payload())
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
        sent = LOCAL.requests[-1]
        self.assertEqual(sent["model"], "local-model")
        self.assertNotIn("reasoning_effort", sent)
        self.assertNotIn("include_reasoning", sent)
        self.assertNotIn("max_completion_tokens", sent)
        self.assertEqual(sent["max_tokens"], 400, "sized completion budget must be preserved")
        self.assertTrue(router.local_is_warm())

    def test_local_call_rearms_ollama_keepalive(self):
        GROQ.script["primary"] = rate_limited()
        GROQ.script["fallback-a"] = rate_limited()
        GROQ.script["fallback-b"] = rate_limited()
        LOCAL.script["local-model"] = ok("local answer")
        post(app_payload())
        for _ in range(20):
            if LOCAL.native:
                break
            time.sleep(0.05)
        self.assertEqual(len(LOCAL.native), 1, "expected one native /api/generate keep-alive touch")
        self.assertEqual(LOCAL.native[0]["model"], "local-model")
        self.assertEqual(LOCAL.native[0]["prompt"], "")
        self.assertEqual(LOCAL.native[0]["keep_alive"], "%ds" % int(router.LOCAL_KEEPALIVE_S))

    def test_fallback_payload_adaptation_per_model(self):
        b = router._groq_body_for("qwen/qwen3.6-27b", app_payload())
        self.assertEqual(b["reasoning_effort"], "none")
        self.assertNotIn("include_reasoning", b)
        b = router._groq_body_for("openai/gpt-oss-120b", app_payload())
        self.assertEqual(b["reasoning_effort"], "low")
        self.assertIn("include_reasoning", b)
        b = router._groq_body_for("groq/compound-mini", app_payload())
        self.assertNotIn("reasoning_effort", b)

    def test_deadline_bounds_total_time_and_returns_504(self):
        def slow(body):
            time.sleep(2.5)
            return ok("too late")
        GROQ.script["primary"] = slow
        GROQ.script["fallback-a"] = slow
        LOCAL.script["local-model"] = slow
        status, _, body, elapsed = post(app_payload(), deadline_ms=1500)
        self.assertEqual(status, 504)
        self.assertLess(elapsed, 2.2, "router must give up inside the client's budget, not after")
        self.assertEqual(LOCAL.requests, [])

    def test_force_local_routes_only_cleanup_prompts_local_first(self):
        router.FORCE_LOCAL = True
        LOCAL.script["local-model"] = ok("local")
        GROQ.script["ctx-model"] = ok("hosted")
        GROQ.script["primary"] = ok("hosted")
        # context-inference style prompt → hosted first even under FORCE_LOCAL
        ctx = {"model": "ctx-model", "messages": [{"role": "system", "content": "You infer activity."},
                                                  {"role": "user", "content": "Analyze the context ... App: Safari"}]}
        _, headers, _, _ = post(ctx)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:ctx-model")
        self.assertEqual(LOCAL.requests, [])
        # cleanup prompt → local first, and its system prompt is remembered for warm-up
        _, headers, _, _ = post(cleanup_payload())
        self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
        self.assertEqual(router._last_cleanup_system, DEFAULT_SYS)
        # warm-up under FORCE_LOCAL always pokes local, with a prefix-compatible prompt
        with LOCAL.lock:
            LOCAL.requests.clear()
        router._local_warm_until = time.monotonic() + 9999  # "already warm" must not skip it
        post({}, path="/warm")
        for _ in range(20):
            if LOCAL.requests:
                break
            time.sleep(0.05)
        self.assertEqual(len(LOCAL.requests), 1)
        self.assertEqual(LOCAL.requests[0]["messages"][0], {"role": "system", "content": DEFAULT_SYS})
        self.assertTrue(LOCAL.requests[0]["messages"][1]["content"].startswith("Raw transcript:"))
        self.assertEqual(LOCAL.requests[0]["max_tokens"], 1)

    def test_force_local_tries_local_first_with_groq_safety_net(self):
        router.FORCE_LOCAL = True
        LOCAL.script["local-model"] = (500, {}, {"error": "model crashed"})
        GROQ.script["primary"] = ok("groq saved it")
        status, headers, _, _ = post(cleanup_payload())
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:primary")
        self.assertEqual(len(LOCAL.requests), 1)

    def test_warm_endpoint_preloads_local_only_when_hosted_is_exhausted(self):
        LOCAL.script["local-model"] = ok("")
        GROQ.script["primary"] = ok("fine")
        post(app_payload())  # teaches the router the primary model + its headroom
        status, _, _, _ = post({}, path="/warm")
        self.assertEqual(status, 200)
        time.sleep(0.5)
        self.assertEqual(LOCAL.requests, [], "hosted has room → don't burn RAM preloading")

        router.RATE = router.RateLimits()
        GROQ.script["primary"] = rate_limited("60")
        GROQ.script["fallback-a"] = rate_limited("60")
        GROQ.script["fallback-b"] = rate_limited("60")
        LOCAL.script["local-model"] = ok("l")
        post(app_payload())  # every hosted bucket now known-blocked
        router._local_warm_until = 0.0
        with LOCAL.lock:
            LOCAL.requests.clear()
        post({}, path="/warm")
        time.sleep(0.8)
        self.assertEqual(len(LOCAL.requests), 1, "all buckets blocked → preload local")
        self.assertEqual(LOCAL.requests[0]["max_tokens"], 1)

    def test_train_format_translation(self):
        app_user = ('Instructions: Clean up RAW_TRANSCRIPTION and return only the cleaned transcript '
                    'text without surrounding quotes. Return EMPTY if there should be no result.\n\n'
                    'CONTEXT: "User is in Slack with Dana."\n\n'
                    'RAW_TRANSCRIPTION: "um so hello\nworld"')
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": app_user}]
        out = router.to_train_format(msgs)
        self.assertEqual(out[0], msgs[0], "system prompt must be untouched (train used the same one)")
        self.assertEqual(out[1]["content"],
                         "Raw transcript:\num so hello\nworld\n\n"
                         "Context: context: User is in Slack with Dana.\n\n"
                         "Clean this up into the final text.")
        # no context → no Context line
        no_ctx = app_user.replace('CONTEXT: "User is in Slack with Dana."', 'CONTEXT: ""')
        out = router.to_train_format([msgs[0], {"role": "user", "content": no_ctx}])
        self.assertNotIn("Context:", out[1]["content"])
        # unrecognised prompts (edit mode, custom) pass through untouched
        edit = [msgs[0], {"role": "user", "content": "Transform SELECTED_TEXT according to VOICE_COMMAND ..."}]
        self.assertEqual(router.to_train_format(edit), edit)
        # and _local_body_for only applies it when configured
        router.LOCAL_PROMPT_FORMAT = "train"
        try:
            b = router._local_body_for({"model": "x", "messages": msgs, "max_completion_tokens": 300})
            self.assertTrue(b["messages"][1]["content"].startswith("Raw transcript:"))
        finally:
            router.LOCAL_PROMPT_FORMAT = "passthrough"
        b = router._local_body_for({"model": "x", "messages": msgs})
        self.assertEqual(b["messages"], msgs)

    def _local_echo_cleaner(self):
        """Fake local model: 'cleans' by upper-casing the Raw transcript block."""
        def clean(body):
            u = body["messages"][-1]["content"]
            raw = u.split("Raw transcript:\n", 1)[1].split("\n\n", 1)[0]
            return ok("<" + raw.upper() + ">")
        return clean

    def test_precache_prefix_is_reused_and_only_tail_is_generated(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        # a first normal cleanup teaches the router the system prompt
        post(cleanup_payload(text="hello there"))
        # STT server pushes two finalized segments while the user is still talking
        for seg in ("first segment here", "second segment too"):
            status, _, body, _ = post({"raw": seg}, path="/precache")
            self.assertEqual(status, 200)
            self.assertTrue(body["queued"])
        for _ in range(40):
            with router._precache_lock:
                if all(e["done"].is_set() for e in router._precache):
                    break
            time.sleep(0.05)
        with LOCAL.lock:
            LOCAL.requests.clear()
        # the app's real request: segments joined by single spaces + a tail
        full = "first segment here second segment too and the tail"
        status, headers, body, _ = post(cleanup_payload(text=full))
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
        self.assertEqual(body["choices"][0]["message"]["content"],
                         "<FIRST SEGMENT HERE> <SECOND SEGMENT TOO> <AND THE TAIL>")
        # only the tail went to the model at key-up
        self.assertEqual(len(LOCAL.requests), 1)
        self.assertIn("Raw transcript:\nand the tail", LOCAL.requests[0]["messages"][-1]["content"])
        # cache entries are consumed
        with router._precache_lock:
            self.assertEqual(router._precache, [])

    def test_precache_mismatch_falls_back_to_full_cleanup(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        post(cleanup_payload(text="warm up"))
        post({"raw": "something else entirely"}, path="/precache")
        time.sleep(0.3)
        with LOCAL.lock:
            LOCAL.requests.clear()
        _, _, body, _ = post(cleanup_payload(text="totally different words"))
        self.assertEqual(body["choices"][0]["message"]["content"], "<TOTALLY DIFFERENT WORDS>")
        self.assertEqual(len(LOCAL.requests), 1)
        self.assertIn("totally different words", LOCAL.requests[0]["messages"][-1]["content"])

    def test_precache_ignored_before_any_cleanup_request(self):
        router.FORCE_LOCAL = True
        _, _, body, _ = post({"raw": "no system prompt known yet"}, path="/precache")
        self.assertFalse(body["queued"])

    def _wait_precache_done(self):
        for _ in range(60):
            with router._precache_lock:
                if router._precache and all(e["done"].is_set() for e in router._precache):
                    return
            time.sleep(0.05)

    def test_partials_chain_incrementally_and_match_despite_punctuation(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        post(cleanup_payload(text="warm up"))
        with LOCAL.lock:
            LOCAL.requests.clear()
        # partial 1: first sentence; partial 2: covers sentence 1 again + sentence 2
        post({"raw": "I really do not like the colour on it."}, path="/precache")
        self._wait_precache_done()
        post({"raw": "I really do not like the colour on it. And then I do not like the text."}, path="/precache")
        self._wait_precache_done()
        # only the NEW words of partial 2 were cleaned
        cleaned_inputs = [r["messages"][-1]["content"] for r in LOCAL.requests]
        self.assertEqual(len(cleaned_inputs), 2)
        self.assertIn("Raw transcript:\nAnd then I do not like the text.", cleaned_inputs[1])
        with LOCAL.lock:
            LOCAL.requests.clear()
        # final transcript: same words, slightly different punctuation, plus a tail
        final = "I really do not like the colour on it, and then I do not like the text. Can we fix it"
        _, headers, body, _ = post(cleanup_payload(text=final))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
        self.assertEqual(body["choices"][0]["message"]["content"],
                         "<I REALLY DO NOT LIKE THE COLOUR ON IT.> <AND THEN I DO NOT LIKE THE TEXT.> <CAN WE FIX IT>")
        self.assertEqual(len(LOCAL.requests), 1)
        self.assertIn("Raw transcript:\nCan we fix it", LOCAL.requests[0]["messages"][-1]["content"])

    def test_precache_reset_drops_previous_dictation(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        post(cleanup_payload(text="warm up"))
        post({"raw": "I think we should ship it."}, path="/precache")
        self._wait_precache_done()
        post({"raw": "", "reset": True}, path="/precache")  # new dictation begins
        with LOCAL.lock:
            LOCAL.requests.clear()
        _, _, body, _ = post(cleanup_payload(text="I think we should ship it tomorrow"))
        self.assertEqual(body["choices"][0]["message"]["content"], "<I THINK WE SHOULD SHIP IT TOMORROW>")
        self.assertEqual(len(LOCAL.requests), 1, "stale entry must not be reused after reset")

    def test_precache_chunk_that_lost_words_is_rejected(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = ok("Just execute.")  # a "summary" of a long chunk
        post(cleanup_payload(text="warm up"))
        post({"raw": "Just do it. I'm about to sleep for an hour, so just execute on your own and test."}, path="/precache")
        self._wait_precache_done()
        with router._precache_lock:
            self.assertEqual(len(router._precache), 1)
            self.assertIsNone(router._precache[0]["cleaned"], "over-shortened chunk must be rejected")
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        with LOCAL.lock:
            LOCAL.requests.clear()
        _, _, body, _ = post(cleanup_payload(text="Just do it. I'm about to sleep for an hour, so just execute on your own and test. Bye"))
        self.assertTrue(body["choices"][0]["message"]["content"].startswith("<JUST DO IT."))
        self.assertEqual(len(LOCAL.requests), 1, "falls back to one full cleanup")

    def test_deterministic_chunk_cleanup_skips_the_model(self):
        router.DET_CLEAN = True
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        post(cleanup_payload(text="warm up"))
        with LOCAL.lock:
            LOCAL.requests.clear()
        post({"raw": "Um, so the first part is done."}, path="/precache")
        self._wait_precache_done()
        self.assertEqual(LOCAL.requests, [], "a filler-only chunk is cleaned deterministically")
        _, _, body, _ = post(cleanup_payload(text="Um, so the first part is done. Uh, and here is the tail"))
        self.assertEqual(body["choices"][0]["message"]["content"], "So the first part is done. And here is the tail.")
        self.assertEqual(LOCAL.requests, [], "tail without interpretation needs → no model call at commit")
        # a tail that needs interpretation still goes to the model
        _, _, body, _ = post(cleanup_payload(text="Um, so the first part is done. Thursday no actually Wednesday"))
        self.assertEqual(len(LOCAL.requests), 1)
        self.assertIn("Thursday no actually Wednesday", LOCAL.requests[0]["messages"][-1]["content"])
        # a custom system prompt → never deterministic (user wants the model's judgement)
        with LOCAL.lock:
            LOCAL.requests.clear()
        post({"raw": "", "reset": True}, path="/precache")
        post(cleanup_payload(text="Um, so hello there friend", system="My own rules: shout everything."))
        self.assertEqual(len(LOCAL.requests), 1)

    def test_precache_full_match_needs_no_generation(self):
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = self._local_echo_cleaner()
        post(cleanup_payload(text="warm up"))
        post({"raw": "only one segment"}, path="/precache")
        time.sleep(0.3)
        with LOCAL.lock:
            LOCAL.requests.clear()
        _, _, body, _ = post(cleanup_payload(text="only one segment"))
        self.assertEqual(body["choices"][0]["message"]["content"], "<ONLY ONE SEGMENT>")
        self.assertEqual(LOCAL.requests, [], "fully cached → zero model calls at key-up")

    def test_local_calls_carry_adapter_path_when_configured(self):
        router.LOCAL_ADAPTER_PATH = "/tmp/adapters"
        try:
            b = router._local_body_for({"model": "x", "messages": []})
            self.assertEqual(b["adapters"], "/tmp/adapters")
        finally:
            router.LOCAL_ADAPTER_PATH = ""
        self.assertNotIn("adapters", router._local_body_for({"model": "x", "messages": []}))

    def test_instruction_echo_is_stripped_or_treated_as_failure(self):
        san = router.sanitize_cleanup_output
        self.assertEqual(san("Clean this up into the final text.\nSpeed test."), "Speed test.")
        self.assertEqual(san("Speed test.\nClean this up into the final text."), "Speed test.")
        self.assertEqual(san("Clean this up into the final text."), "")
        self.assertEqual(san("Normal output."), "Normal output.")
        # end to end: a pure echo from local must fall through to the next backend
        router.FORCE_LOCAL = True
        router.LOCAL_PROMPT_FORMAT = "train"
        LOCAL.script["local-model"] = ok("Clean this up into the final text.")
        GROQ.script["primary"] = ok("groq answer")
        _, headers, body, _ = post(cleanup_payload(text="that is definitely not the same"))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:primary")
        self.assertEqual(body["choices"][0]["message"]["content"], "groq answer")

    def test_unreachable_hosted_trips_breaker_and_fails_over_fast(self):
        # Point the hosted URL at a host that cannot resolve; the FIRST request
        # must fail over to local quickly (DNS wait is bounded), and the SECOND
        # must skip hosted without even trying (breaker open).
        saved = router._groq_parts
        router._groq_parts = urllib.parse.urlsplit("https://does-not-exist.invalid/openai/v1/chat/completions")
        try:
            LOCAL.script["local-model"] = ok("local answer")
            t0 = time.monotonic()
            _, headers, _, _ = post(app_payload())
            first = time.monotonic() - t0
            self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
            self.assertLess(first, router.DNS_TIMEOUT_S + 1.5, "DNS failure must be bounded")
            self.assertTrue(router.hosted_breaker_open())
            t0 = time.monotonic()
            _, headers, _, _ = post(app_payload())
            self.assertEqual(headers.get("X-FreeFlow-Route"), "local")
            self.assertLess(time.monotonic() - t0, 0.5, "breaker open → no hosted attempt at all")
        finally:
            router._groq_parts = saved
            router._reset_hosted_breaker()
            router._dns_cache.clear()

    # ---- the app's request shape --------------------------------------------
    def test_app_prompt_matches_swift_source(self):
        """app_prompt.cleanup_user_message must be what PostProcessingService.process
        sends, or the router silently stops recognising cleanups (it did, for two
        months, after upstream switched to the heredoc format)."""
        swift = open(os.path.join(HERE, "..", "Sources", "PostProcessingService.swift")).read()
        marker = 'let userMessage = """\nInstructions: Clean up RAW_TRANSCRIPTION'
        start = swift.index(marker) + len('let userMessage = """\n')
        block = swift[start:]
        block = block[:block.index('\n"""')]
        rendered = block.replace("\\(contextSummary)", "CTX").replace("\\(transcript)", "RAW")
        self.assertEqual(rendered, app_prompt.cleanup_user_message("RAW", "CTX"))

    def test_both_cleanup_prompt_formats_are_recognised(self):
        raw = "um so\nhello world"
        for content in (app_prompt.cleanup_user_message(raw, "In Slack"),
                        app_prompt.legacy_cleanup_user_message(raw, "In Slack")):
            self.assertEqual(router.parse_cleanup_user(content), ("In Slack", raw))
            self.assertTrue(router.is_cleanup_request({"messages": [{"role": "user", "content": content}]}))
        self.assertFalse(router.is_cleanup_request({"messages": [{"role": "user", "content": "Analyze the context"}]}))
        self.assertFalse(router.is_cleanup_request(image_payload()))
        # the train-format rewrite sees through the heredoc too
        out = router.to_train_format([{"role": "system", "content": "S"},
                                      {"role": "user", "content": app_prompt.cleanup_user_message(raw, "")}])
        self.assertEqual(out[1]["content"], "Raw transcript:\num so\nhello world\n\nClean this up into the final text.")

    # ---- image (context) requests --------------------------------------------
    def test_images_are_priced_per_image_not_per_base64_char(self):
        cost = router._estimate_cost_tokens(image_payload())
        self.assertLess(cost, router.GROQ_IMAGE_TOKENS + 512 + 200)
        self.assertGreater(cost, router.GROQ_IMAGE_TOKENS)
        self.assertTrue(router._has_image(image_payload()))
        self.assertFalse(router._has_image(cleanup_payload()))

    def test_local_refuses_image_requests_instantly(self):
        GROQ.script["ctx-model"] = rate_limited("60")
        GROQ.script["fallback-a"] = rate_limited("60")
        GROQ.script["fallback-b"] = rate_limited("60")
        LOCAL.script["local-model"] = ok("should never be asked")
        t0 = time.monotonic()
        status, _, _, _ = post(image_payload())
        self.assertIn(status, (502, 504))
        self.assertLess(time.monotonic() - t0, 1.0, "the local refusal must not wait on anything")
        self.assertEqual(LOCAL.requests, [], "a text-only server must never see an image request")

    def test_image_request_completion_budget_is_capped(self):
        GROQ.script["ctx-model"] = ok("Two sentences.")
        _, headers, _, _ = post(image_payload(max_completion_tokens=512))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:ctx-model")
        self.assertEqual(GROQ.requests[-1]["max_completion_tokens"], router.CONTEXT_MAX_COMPLETION_TOKENS)
        # cleanup budgets are the app's business
        GROQ.script["primary"] = ok("x")
        post(cleanup_payload())
        self.assertEqual(GROQ.requests[-1]["max_completion_tokens"], 400)

    # ---- deterministic precache (no model at all) ----------------------------
    def test_deterministic_precache_answers_without_any_model(self):
        router.DET_CLEAN = True
        router.PRECACHE_LLM = False
        GROQ.script["primary"] = ok("hosted")
        post(cleanup_payload(text="teach the system prompt"))  # hosted-first; teaches the router the prompt
        for seg in ("Um, so the first part is done.", "and the second part too,"):
            post({"raw": seg}, path="/precache")
        self._wait_precache_done()
        with GROQ.lock:
            GROQ.requests.clear()
        full = "Um, so the first part is done. and the second part too, uh and here is the tail"
        status, headers, body, _ = post(cleanup_payload(text=full))
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-FreeFlow-Route"), "precache")
        self.assertEqual(body["choices"][0]["message"]["content"],
                         "So the first part is done. And the second part too, and here is the tail.")
        self.assertEqual(GROQ.requests, [])
        self.assertEqual(LOCAL.requests, [])
        with router._precache_lock:
            self.assertEqual(router._precache, [], "used entries are consumed")

    def test_precache_chunk_needing_the_model_is_left_for_commit(self):
        router.DET_CLEAN = True
        router.PRECACHE_LLM = False
        GROQ.script["primary"] = ok("Let's meet Wednesday.")
        post(cleanup_payload(text="teach"))
        post({"raw": "let's meet Thursday no actually Wednesday"}, path="/precache")
        self._wait_precache_done()
        with router._precache_lock:
            self.assertIsNone(router._precache[0]["cleaned"])
        self.assertEqual(LOCAL.requests, [], "PRECACHE_LLM=0: the small model must not judge a self-correction")
        with GROQ.lock:
            GROQ.requests.clear()
        _, headers, body, _ = post(cleanup_payload(text="let's meet Thursday no actually Wednesday and that's it"))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "groq:primary", "cleaned whole, by the normal chain")
        self.assertIn("Thursday no actually Wednesday and that's it", GROQ.requests[0]["messages"][-1]["content"])
        self.assertEqual(body["choices"][0]["message"]["content"], "Let's meet Wednesday.")

    def test_fragment_assembly_keeps_sentence_flow(self):
        # Chunks settle mid-sentence. Cleaning each as a sentence would give
        # "…should be. An action." — fragments are cased and closed as a whole.
        router.DET_CLEAN = True
        router.PRECACHE_LLM = False
        GROQ.script["primary"] = ok("hosted")
        post(cleanup_payload(text="teach"))
        for seg in ("hold up the pendant should be", "an action. or you should see her"):
            post({"raw": seg}, path="/precache")
        self._wait_precache_done()
        _, headers, body, _ = post(cleanup_payload(text="hold up the pendant should be an action. or you should see her um holding it out"))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "precache")
        self.assertEqual(body["choices"][0]["message"]["content"],
                         "Hold up the pendant should be an action. Or you should see her holding it out.")
        # a chat destination: no closing period on a single short line either way
        chat_sys = DEFAULT_SYS + "\n\nDestination:\n- The text is going into a chat message. Keep it casual."
        post(cleanup_payload(text="teach", system=chat_sys))
        post({"raw": "sounds good"}, path="/precache")
        self._wait_precache_done()
        _, headers, body, _ = post(cleanup_payload(text="sounds good to me", system=chat_sys))
        self.assertEqual(headers.get("X-FreeFlow-Route"), "precache")
        self.assertEqual(body["choices"][0]["message"]["content"], "Sounds good to me")

    def test_local_heartbeat_touches_the_model_only_when_idle(self):
        LOCAL.script["local-model"] = ok("")
        saved = router.LOCAL_HEARTBEAT_S
        router.LOCAL_HEARTBEAT_S = 0.2
        try:
            router._local_last_call = 0.0
            self.assertTrue(router.heartbeat_tick())
            self.assertEqual(len(LOCAL.requests), 1)
            self.assertEqual(LOCAL.requests[0]["max_tokens"], 1)
            self.assertFalse(router.heartbeat_tick(), "just touched → nothing to do")
            self.assertEqual(len(LOCAL.requests), 1)
        finally:
            router.LOCAL_HEARTBEAT_S = saved

    def test_parse_reset_formats(self):
        p = router.RateLimits._parse_reset
        self.assertAlmostEqual(p("23.835s"), 23.835)
        self.assertAlmostEqual(p("832ms"), 0.832)
        self.assertAlmostEqual(p("1m26.4s"), 86.4)
        self.assertAlmostEqual(p("2h3m"), 7380)
        self.assertAlmostEqual(p("6"), 6)
        self.assertIsNone(p(""))

    def test_status_reports_running_source_and_policy(self):
        req = urllib.request.Request(ROUTER_URL + "/status")
        with urllib.request.urlopen(req, timeout=5) as r:
            status = json.loads(r.read())
        self.assertEqual(status["source_sha"], router.source_sha())
        for key in ("force_local", "precache_llm", "local_heartbeat_s", "local_warm"):
            self.assertIn(key, status)

    def test_models_endpoint_lists_fallbacks_and_local(self):
        req = urllib.request.Request(ROUTER_URL + "/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        for expected in ("fallback-a", "fallback-b", "local-model"):
            self.assertIn(expected, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
