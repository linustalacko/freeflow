"""Benchmark-only context and agreement guards; no app activation."""
from collections import deque
import os

import stt_server as stt
from experimental_alignment import aged_hypothesis, anchor_window_start


class GuardedSession(stt.RealtimeSession):
    def __init__(self, ws, app):
        super().__init__(ws, app)
        self.hypothesis_history = deque()
        self.agreement_age = float(os.environ.get("EXP_AGREEMENT_AGE_S", "1"))
        original_discard = self.buf.discard_before

        def retain_anchor(requested_start):
            # Parent calls discard before replacing settled_words, so this
            # conservatively retains the previous anchor too. No whole-session
            # audio accumulation is introduced by this retention rule.
            safe_start = self._anchor_start()
            original_discard(min(requested_start, safe_start))
        self.buf.discard_before = retain_anchor

    def _anchor_start(self):
        return int(anchor_window_start(self.settled_samples / stt.TARGET_SR,
                                       stt.LEFT_CONTEXT_S, self.settled_words) * stt.TARGET_SR)

    def _window(self):
        start = max(self.buf.start, self._anchor_start())
        return start, self.buf.end, self.buf.snapshot(start)

    def _on_partial(self, future, start_idx, end_idx, t0):
        if not self.committed:
            tokens = future.result()
            if tokens is not None and not isinstance(tokens, Exception):
                end = end_idx / stt.TARGET_SR
                previous = aged_hypothesis(self.hypothesis_history, end, self.agreement_age)
                self.prev_hyp = stt.splice_tail(previous, self.settled_words,
                                               self.settled_samples / stt.TARGET_SR)
                self.hypothesis_history.append((end, stt.words_from_tokens(tokens, start_idx / stt.TARGET_SR)))
                while len(self.hypothesis_history) > 1 and self.hypothesis_history[1][0] < end - self.agreement_age - .5:
                    self.hypothesis_history.popleft()
        super()._on_partial(future, start_idx, end_idx, t0)
