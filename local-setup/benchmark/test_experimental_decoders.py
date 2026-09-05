"""Deterministic decoder control-flow tests; no MLX, model weights, or providers."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experimental_decoders import install


class Array:
    dtype = "synthetic"

    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        result, value = [], self.value
        while isinstance(value, list):
            result.append(len(value))
            value = value[0] if value else None
        return tuple(result)

    def __getitem__(self, key):
        def select(value, keys):
            if not keys:
                return value
            head, *tail = keys
            if isinstance(head, slice):
                return [select(v, tail) for v in value[head]]
            return select(value[head], tail)
        return Array(select(self.value, key if isinstance(key, tuple) else (key,)))

    def astype(self, _):
        return self

    def tolist(self):
        return self.value

    def __int__(self):
        return int(self.value)


def flatten(value):
    if isinstance(value, list):
        return [v for item in value for v in flatten(item)]
    return [value]


def argmax(array):
    values = flatten(array.value)
    return Array(max(range(len(values)), key=values.__getitem__))


class Greedy:
    pass


@dataclass
class Token:
    id: int
    text: str
    start: float
    duration: float
    confidence: float


class Model:
    vocabulary = ["one", "two"]
    durations = [0, 1]
    max_symbols = 2
    time_ratio = .08

    def __init__(self, decisions):
        self.decisions = decisions
        self.predict_calls = 0

    def decoder(self, token, hidden):
        self.predict_calls += 1
        return Array([0]), (Array(1), Array(2))

    def joint(self, feature, output):
        step = flatten(feature.value)[0]
        token, duration = self.decisions[step]
        scores = [int(i == token) for i in range(3)] + [int(i == duration) for i in range(2)]
        return Array([[[scores]]])

    def decode_greedy(self, *args, **kwargs):
        return "original"


class DecoderTests(unittest.TestCase):
    def setUp(self):
        mx = SimpleNamespace(array=Array, argmax=argmax, compile=lambda f: f,
                             stack=lambda arrays: Array([a.value for a in arrays]))
        modules = {"mlx": SimpleNamespace(core=mx), "mlx.core": mx,
                   "parakeet_mlx": SimpleNamespace(tokenizer=SimpleNamespace(
                       decode=lambda ids, vocabulary: vocabulary[ids[0]])),
                   "parakeet_mlx.alignment": SimpleNamespace(AlignedToken=Token),
                   "parakeet_mlx.parakeet": SimpleNamespace(Greedy=Greedy)}
        self.modules = patch.dict("sys.modules", modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_blank_cache_preserves_tokens_and_resets_after_emission(self):
        # Blank, blank, word, blank, word: predictor is reused until a word
        # changes the recurrent state, then must be recomputed exactly once.
        for mode in ["no-confidence", "fused-sync", "compiled", "cached-predictor", "compiled-cache"]:
            with self.subTest(mode=mode):
                model = Model([(2, 1), (2, 1), (0, 1), (2, 1), (1, 1)])
                install(model, mode)
                result, state = model.decode_greedy(Array([[[i] for i in range(5)]]),
                                                    config=SimpleNamespace(decoding=Greedy()))
                self.assertEqual([t.id for t in result[0]], [0, 1])
                self.assertEqual([t.start for t in result[0]], [.16, .32])
                self.assertTrue(all(t.confidence == 0 for t in result[0]))
                self.assertEqual(model.predict_calls, 2 if "cache" in mode else 5)
                self.assertIsNotNone(state[0])

    def test_zero_duration_guard_terminates_and_lengths_are_respected(self):
        model = Model([(2, 0), (0, 1), (1, 1)])
        install(model, "compiled-cache")
        result, _ = model.decode_greedy(Array([[[0], [1], [2]]]), Array([2]),
                                        config=SimpleNamespace(decoding=Greedy()))
        self.assertEqual([t.id for t in result[0]], [0])
        self.assertEqual(model.predict_calls, 1)

    def test_batch_state_is_independent_and_caller_lists_are_preserved(self):
        model = Model([(0, 1)])
        install(model, "compiled-cache")
        last, hidden = [None, None], [None, None]
        result, state = model.decode_greedy(Array([[[0]], [[0]]]), last_token=last,
                                            hidden_state=hidden, config=SimpleNamespace(decoding=Greedy()))
        self.assertEqual([[t.id for t in row] for row in result], [[0], [0]])
        self.assertEqual(model.predict_calls, 2)
        self.assertEqual(last, [None, None])
        self.assertEqual(hidden, [None, None])

    def test_baseline_and_unsupported_decoding_keep_original(self):
        model = Model([])
        original = model.decode_greedy
        install(model, "baseline")
        self.assertEqual(model.decode_greedy, original)
        install(model, "compiled-cache")
        self.assertEqual(model.decode_greedy(None, config=SimpleNamespace(decoding="beam")), "original")
        with self.assertRaises(ValueError):
            install(model, "unknown")


if __name__ == "__main__":
    unittest.main()
