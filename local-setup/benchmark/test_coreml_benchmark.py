import array
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import coreml_benchmark as c


class CoreMLBoundaryTests(unittest.TestCase):
    def test_unverified_or_logging_enabled_worker_is_never_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / "worker"
            worker.write_bytes(b"synthetic executable")
            digest = hashlib.sha256(worker.read_bytes()).hexdigest()
            for metadata in [{"library_logging_disabled": False, "worker_sha256": digest},
                             {"library_logging_disabled": True, "worker_sha256": "wrong"}]:
                worker.with_suffix(".json").write_text(json.dumps(metadata))
                with patch.object(c.subprocess, "Popen") as spawn, self.assertRaises(ValueError):
                    c.CoreMLClient(worker, Path(temporary))
                spawn.assert_not_called()
            worker.with_suffix(".json").write_text(json.dumps({
                "library_logging_disabled": True, "worker_sha256": digest}))
            self.assertEqual(c.worker_metadata(worker)["worker_sha256"], digest)

    def test_worker_errors_are_sanitized(self):
        client = c.CoreMLClient.__new__(c.CoreMLClient)
        client.process = SimpleNamespace(stdout=io.BytesIO(b'{"error":"invented private payload"}\n'))
        with patch.object(c.select, "select", return_value=([client.process.stdout], [], [])):
            with self.assertRaisesRegex(RuntimeError, "^coreml_worker_failed$"):
                client.receive(1)

    def test_timeout_is_counted_as_an_error(self):
        client = c.CoreMLClient.__new__(c.CoreMLClient)
        client.process = SimpleNamespace(stdout=io.BytesIO())
        with patch.object(c.select, "select", return_value=([], [], [])):
            with self.assertRaises(TimeoutError):
                client.receive(1)

    def test_report_and_worker_request_exclude_reference_and_payloads(self):
        reference = "invented reference phrase"
        hypothesis = "invented response phrase"
        client = Mock()
        response = {"text": hypothesis, "tokens": [{"text": "invented token"}],
                    "model_s": .02, "total_s": .03}
        client.transcribe.side_effect = [response, response, RuntimeError("invented private error")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "report.json"
            args = ["coreml_benchmark.py", "--worker", str(root / "worker"),
                    "--models", str(root / "models"), "--manifest", str(root / "manifest"),
                    "--output", str(output), "--repeats", "2"]
            rows = [{"id": "fixture", "audio": root / "audio.wav", "text": reference}]
            with patch.object(c.sys, "argv", args), patch.object(c.b, "read_manifest", return_value=rows), \
                    patch.object(c.b, "command", return_value=SimpleNamespace(stdout=array.array("h", [0, 1]).tobytes())), \
                    patch.object(c, "worker_metadata", return_value={"library_logging_disabled": True}), \
                    patch.object(c, "CoreMLClient", return_value=client), contextlib.redirect_stdout(io.StringIO()) as console:
                self.assertEqual(c.main(), 1)
            content = output.read_text() + console.getvalue()
            for value in [reference, hypothesis, "invented token", "invented private error", str(root)]:
                self.assertNotIn(value, content)
            report = json.loads(output.read_text())
            self.assertEqual(report["summary"]["failures"], 1)
            self.assertEqual(report["trials"][0]["word_errors"], 1)
            self.assertTrue(all(isinstance(call.args[0], bytes) for call in client.transcribe.call_args_list))
            client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
