from types import SimpleNamespace
import unittest

from experimental_alignment import aged_hypothesis, anchor_window_start


class AlignmentTests(unittest.TestCase):
    def test_short_window_keeps_three_complete_anchor_words(self):
        words = [SimpleNamespace(start=t) for t in [6.0, 7.0, 8.0, 9.0]]
        self.assertAlmostEqual(anchor_window_start(10, .5, words), 6.76)
        self.assertEqual(anchor_window_start(10, 10, words), 0)
        self.assertEqual(anchor_window_start(0, .5, []), 0)

    def test_fast_repeated_hypotheses_do_not_count_as_mature_agreement(self):
        history = [(1.0, ["old synthetic words"]), (1.75, ["new synthetic words"])]
        self.assertEqual(aged_hypothesis(history, 1.9, 1), [])
        self.assertEqual(aged_hypothesis(history, 2, 1), ["old synthetic words"])
        self.assertEqual(aged_hypothesis(history, 2.75, 1), ["new synthetic words"])
