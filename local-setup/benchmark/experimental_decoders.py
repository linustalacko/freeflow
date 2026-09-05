"""Benchmark-only Parakeet TDT decoding experiments; no production activation.

Greedy recurrence follows parakeet-mlx 0.5.2 (Apache-2.0):
https://github.com/senstella/parakeet-mlx/blob/master/parakeet_mlx/parakeet.py
Modified for FreeFlow experiments: omit confidence, fuse host synchronization,
optionally compile steps, and reuse the predictor after blank tokens.
License copy: licenses/parakeet-mlx-Apache-2.0.txt
These variants omit unused entropy confidence. Token confidence is explicitly
0 (unscored); do not use this module for consumers requiring confidence values.
"""
from types import MethodType


def install(model, mode):
    import mlx.core as mx
    from parakeet_mlx.alignment import AlignedToken
    from parakeet_mlx import tokenizer
    from parakeet_mlx.parakeet import Greedy

    original = model.decode_greedy
    if mode == "baseline":
        return original
    if mode not in {"no-confidence", "fused-sync", "compiled", "cached-predictor", "compiled-cache"}:
        raise ValueError("unknown decoder experiment")

    def predict(token, state, feature):
        output, (hidden, cell) = model.decoder(token, state)
        return output.astype(feature.dtype), (hidden.astype(feature.dtype), cell.astype(feature.dtype))

    def joint(feature, output):
        logits = model.joint(feature, output)[0, 0]
        return mx.stack([mx.argmax(logits[:, :len(model.vocabulary) + 1]),
                         mx.argmax(logits[:, len(model.vocabulary) + 1:])])

    def combined(feature, token, state):
        output, hidden = predict(token, state, feature)
        return joint(feature, output), hidden

    combined_step = mx.compile(combined) if mode == "compiled" else combined
    predictor = mx.compile(predict) if mode == "compiled-cache" else predict
    joint_step = mx.compile(joint) if mode == "compiled-cache" else joint

    def decode(self, features, lengths=None, last_token=None, hidden_state=None, *, config):
        if not isinstance(config.decoding, Greedy):
            return original(features, lengths, last_token, hidden_state, config=config)
        batch_count, steps, *_ = features.shape
        hidden_state = list(hidden_state) if hidden_state is not None else [None] * batch_count
        last_token = list(last_token) if last_token is not None else [None] * batch_count
        lengths = lengths if lengths is not None else mx.array([steps] * batch_count)
        results = []
        for batch in range(batch_count):
            hypothesis, step, new_symbols = [], 0, 0
            feature, length = features[batch:batch + 1], int(lengths[batch])
            cached = None
            while step < length:
                current = feature[:, step:step + 1]
                token = mx.array([[last_token[batch]]]) if last_token[batch] is not None else None
                if mode in {"cached-predictor", "compiled-cache"}:
                    if cached is None:
                        cached = predictor(token, hidden_state[batch], current)
                    output, next_hidden = cached
                    choices = joint_step(current, output)
                else:
                    choices, next_hidden = combined_step(current, token, hidden_state[batch])
                if mode == "no-confidence":
                    pred_token, decision = int(choices[0]), int(choices[1])
                else:
                    pred_token, decision = choices.tolist()
                duration = self.durations[decision]
                if pred_token != len(self.vocabulary):
                    hypothesis.append(AlignedToken(pred_token,
                        text=tokenizer.decode([pred_token], self.vocabulary),
                        start=step * self.time_ratio, duration=duration * self.time_ratio,
                        confidence=0.0))
                    last_token[batch] = pred_token
                    hidden_state[batch] = next_hidden
                    cached = None
                step += duration
                new_symbols += 1
                if duration:
                    new_symbols = 0
                elif self.max_symbols is not None and new_symbols >= self.max_symbols:
                    step += 1
                    new_symbols = 0
            results.append(hypothesis)
        return results, hidden_state

    model.decode_greedy = MethodType(decode, model)
    return original
