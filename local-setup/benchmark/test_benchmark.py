import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

import benchmark as b
import prepare_corpus


class MetricsTests(unittest.TestCase):
    def test_normalization_keeps_spoken_content(self):
        self.assertEqual(b.normalize("DON’T stop—café!"), "dont stop café")
        self.assertEqual(b.score("The blue bird", "the BLUE bird.")["word_errors"], 0)
        self.assertEqual(b.score("um twenty two", "22")["word_errors"], 3)

    def test_insertions_deletions_and_substitution(self):
        self.assertEqual(b.score("a b c", "a x c d")["word_errors"], 2)
        self.assertEqual(b.score("a b c", "a")["word_errors"], 2)
        self.assertEqual(b.score("a", "b c d")["word_errors"], 3)

    def test_weighted_wer_and_failed_requests_are_separate(self):
        rows = [{"status": "ok", "audio_s": 1, "total_s": .1, "after_audio_s": .1,
                 **b.score("a", "x")},
                {"status": "ok", "audio_s": 2, "total_s": .2, "after_audio_s": .2,
                 **b.score("a b c d e f g h i", "a b c d e f g h i")},
                {"status": "error", "error": "http_429", "audio_s": 1}]
        result = b.summarize(rows)
        self.assertEqual(result["wer"], .1)  # not the per-clip mean of .5
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["after_audio_p95_s"], .2)
        self.assertEqual(result["failure_codes"], {"http_429": 1})

    def test_silence_hallucinations_do_not_pollute_speech_denominator(self):
        result = b.summarize([{"status": "ok", "audio_s": 2, "total_s": .1,
                               "after_audio_s": .1, **b.score("", "Thank you") }])
        self.assertIsNone(result["wer"])
        self.assertEqual(result["silence_false_positives"], 1)
        self.assertEqual(result["silence_inserted_words"], 2)
        self.assertIsNone(b.summarize([])["after_audio_p50_s"])
        self.assertIsNone(b.summarize([{"status": "error", "error": "timeout"}])["wer"])


class PrivacyTests(unittest.TestCase):
    def test_remote_requires_opt_in_and_https(self):
        engine = {"id": "hosted", "kind": "multipart", "model": "example", "url": "https://example.com/v1"}
        with self.assertRaises(ValueError):
            b.validate_engine(engine, False)
        b.validate_engine(engine, True)
        for url in ["http://example.com/v1", "https://key@example.com/v1", "https://example.com/?key=secret"]:
            with self.assertRaises(ValueError):
                b.validate_engine({**engine, "url": url}, True)
        with self.assertRaises(ValueError):
            b.validate_engine({**engine, "kind": "freeflow-stream"}, True)

    def test_missing_key_fails_before_audio_upload(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            b.validate_engine({"id": "local", "kind": "multipart", "model": "example",
                               "url": "http://127.0.0.1/v1", "api_key_env": "TEST_KEY"}, False)

    def test_duplicate_manifest_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.wav").write_bytes(b"synthetic")
            row = {"id": "one", "audio": "a.wav", "text": "Synthetic reference"}
            manifest = root / "manifest.jsonl"
            manifest.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                b.read_manifest(manifest)


@contextlib.contextmanager
def server(status=200, payload=None, redirect=None):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *_):
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/transcribe", seen
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


class EndpointTests(unittest.TestCase):
    def test_multipart_upload_and_scoring_without_content_in_result(self):
        with server(payload={"text": "The blue bird."}) as (url, seen):
            result = b.transcribe({"id": "mock", "kind": "multipart", "url": url, "model": "mock"},
                                  {"wav": b"synthetic-audio", "text": "the blue bird"}, "en", 2, None)
        self.assertEqual(result["word_errors"], 0)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("text", result)
        self.assertIn(b"synthetic-audio", seen[0][1])
        self.assertNotIn(b"the blue bird", seen[0][1])  # reference never sent

    def test_deepgram_response_shape(self):
        payload = {"results": {"channels": [{"alternatives": [{"transcript": "blue bird"}]}]}}
        with server(payload=payload) as (url, seen):
            result = b.batch({"kind": "deepgram", "url": url, "model": "mock"}, b"audio", "en", 2)
        self.assertEqual(result["text"], "blue bird")
        self.assertIn("smart_format=false", seen[0][0])
        self.assertEqual(seen[0][1], b"audio")

    def test_errors_are_counted_without_echoing_response(self):
        with server(429, {"error": "synthetic-secret"}) as (url, seen):
            result = b.transcribe({"kind": "multipart", "url": url, "model": "mock"},
                                  {"wav": b"audio", "text": "reference"}, "en", 2, None)
        self.assertEqual(result["error"], "http_429")
        self.assertEqual(len(seen), 1)  # no hidden retry
        self.assertNotIn("synthetic-secret", json.dumps(result))

    def test_redirect_does_not_forward_audio(self):
        with server(payload={"text": "unexpected"}) as (target, seen):
            with server(307, {}, target) as (url, _):
                with self.assertRaises(urllib.error.HTTPError):
                    b.batch({"kind": "multipart", "url": url, "model": "mock"}, b"audio", "en", 2)
            self.assertEqual(seen, [])

    def test_missing_transcript_is_a_failure_not_a_silent_success(self):
        with server(payload={"unrelated": "synthetic text"}) as (url, _):
            result = b.transcribe({"kind": "multipart", "url": url, "model": "mock"},
                                  {"wav": b"audio", "text": "reference"}, "en", 2, None)
        self.assertEqual(result["error"], "invalid_response")

    def test_report_does_not_contain_reference_or_hypothesis(self):
        with server(payload={"text": "invented response phrase"}) as (url, _):
            result = b.transcribe({"kind": "multipart", "url": url, "model": "mock"},
                                  {"wav": b"synthetic audio", "text": "invented reference phrase"}, "en", 2, None)
        result.update({"audio_s": 1, "clip": "fixture", "category": "synthetic"})
        report = {"engines": [{"id": "mock", "summary": b.summarize([result]), "trials": [result]}]}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "result.json"
            b.write_report(path, report)
            content = path.read_text() + path.with_suffix(".md").read_text()
            for private in ["invented response phrase", "invented reference phrase", "synthetic audio", url]:
                self.assertNotIn(private, content)


class CorpusTests(unittest.TestCase):
    def test_librispeech_selection_covers_speakers_and_preserves_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            for speaker in range(3):
                lines = []
                for index in range(3):
                    identifier = f"{speaker}-1-{index}"
                    (source / f"{identifier}.flac").write_bytes(b"synthetic fixture")
                    lines.append(identifier + " THE BLUE BIRD")
                (source / f"{speaker}.trans.txt").write_text("\n".join(lines))
            rows = prepare_corpus.librispeech(source, output, 3, 1)
            self.assertEqual(len({r["id"].split("-")[0] for r in rows}), 3)
            self.assertTrue(all(r["text"] == "THE BLUE BIRD" for r in rows))


if __name__ == "__main__":
    unittest.main()
