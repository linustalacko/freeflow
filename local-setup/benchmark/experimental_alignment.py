"""Pure helpers for an experimental guard on aggressive streaming windows."""


def anchor_window_start(cut, context, settled_words, padding=.24):
    start = max(0.0, cut - context)
    if settled_words:
        start = min(start, max(0.0, settled_words[-min(3, len(settled_words))].start - padding))
    return start


def aged_hypothesis(history, window_end, minimum_age):
    eligible = [words for end, words in history if end <= window_end - minimum_age + 1e-9]
    return eligible[-1] if eligible else []
