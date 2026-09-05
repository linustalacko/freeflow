import unittest

from experiment_config import environment, validate_matrix


class ConfigTests(unittest.TestCase):
    def test_baseline_cannot_inherit_experiments_or_provider_forwarding(self):
        result = environment({"EXP_DECODER": "compiled-cache", "STT_MODEL": "other-model",
                              "STT_PORT": "8082", "STT_HEARTBEAT_S": "45", "STT_FAST_DECODER": "1",
                              "ROUTER_PRECACHE_URL": "http://example.invalid", "PATH": "test-path"}, {}, 1234)
        self.assertNotIn("EXP_DECODER", result)
        self.assertEqual(result["STT_MODEL"], "mlx-community/parakeet-tdt-0.6b-v2")
        self.assertEqual(result["STT_PORT"], "1234")
        self.assertEqual(result["ROUTER_PRECACHE_URL"], "")
        self.assertEqual(result["STT_HEARTBEAT_S"], "0")
        self.assertEqual(result["STT_FAST_DECODER"], "0")
        self.assertEqual(result["PATH"], "test-path")

    def test_unknown_or_unbounded_settings_fail_before_execution(self):
        for key, value in [("STT_MODEL", "other"), ("EXP_UNKNOWN", "private-value"),
                           ("EXP_DECODER", "other"), ("STT_PARTIAL_EVERY_S", "0"),
                           ("EXP_CHUNK_S", "nan"), ("EXP_DEPTH", "1.5"),
                           ("EXP_QUANTIZE_BITS", 4)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_matrix([{"id": "one", "env": {key: value}}])

    def test_duplicate_variants_and_ignored_model_settings_fail(self):
        with self.assertRaises(ValueError):
            validate_matrix([{"id": "one"}, {"id": "one"}])
        with self.assertRaises(ValueError):
            validate_matrix([{"id": "one", "env": {"STT_LEFT_CONTEXT_S": "2"}}], model_only=True)
        self.assertTrue(validate_matrix([{"id": "one", "env": {"EXP_QUANTIZE_BITS": "4"}}]))
