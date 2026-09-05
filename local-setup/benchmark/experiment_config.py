"""Explicit experiment settings; inherited shell settings cannot alter a baseline."""
import math

from benchmark import safe_identifier


CHOICES = {
    "EXP_DECODER": {"baseline", "no-confidence", "fused-sync", "compiled", "cached-predictor", "compiled-cache"},
    "EXP_DTYPE": {"bfloat16", "float16", "float32"},
    "EXP_QUANTIZE_BITS": {"0", "4", "8"},
    "EXP_COMPILE_ENCODER": {"0", "1"},
    "EXP_KEEP_ATTENTION": {"0", "1"},
    "EXP_SESSION": {"cached", "guarded"},
}
RANGES = {
    "EXP_CHUNK_S": (.02, 10), "EXP_RIGHT_CONTEXT": (0, 256), "EXP_DEPTH": (1, 24),
    "EXP_FINAL_PAD_S": (0, 5), "EXP_PAD_BUCKET_S": (0, 5), "EXP_AGREEMENT_AGE_S": (.25, 3),
    "STT_PARTIAL_EVERY_S": (.02, 10), "STT_LEFT_CONTEXT_S": (0, 30),
    "STT_SETTLE_HORIZON_S": (0, 10), "STT_SETTLE_GAP_S": (0, 10),
    "STT_MIN_SETTLE_S": (0, 10),
}
INTEGERS = {"EXP_RIGHT_CONTEXT", "EXP_DEPTH"}
MODEL_SETTINGS = {"EXP_DECODER", "EXP_DTYPE", "EXP_QUANTIZE_BITS", "EXP_COMPILE_ENCODER", "EXP_PAD_BUCKET_S"}
DEFAULTS = {"STT_MODEL": "mlx-community/parakeet-tdt-0.6b-v2", "STT_PARTIAL_EVERY_S": "2",
            "STT_LEFT_CONTEXT_S": "10", "STT_SETTLE_HORIZON_S": "1", "STT_SETTLE_GAP_S": "0",
            "STT_MIN_SETTLE_S": "1", "STT_CACHE_MB": "512", "STT_WIRED_MB": "2048",
            "STT_FAST_DECODER": "0"}


def validate_matrix(matrix, model_only=False):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a nonempty list")
    seen = set()
    for variant in matrix:
        identifier = safe_identifier(variant["id"])
        if identifier in seen:
            raise ValueError("duplicate variant ID")
        seen.add(identifier)
        settings = variant.get("env", {})
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        for key, value in settings.items():
            if not isinstance(value, str) or (model_only and key not in MODEL_SETTINGS):
                raise ValueError("invalid experiment setting")
            if key in CHOICES:
                if value not in CHOICES[key]:
                    raise ValueError("unknown setting choice")
            elif key in RANGES:
                number = float(value)
                low, high = RANGES[key]
                if not math.isfinite(number) or not low <= number <= high:
                    raise ValueError("setting outside experiment range")
                if key in INTEGERS and str(int(number)) != value:
                    raise ValueError("setting must be an integer")
            else:
                raise ValueError("unsupported experiment setting")
    return matrix


def environment(parent, settings, port=None):
    env = {k: v for k, v in parent.items() if not k.startswith(("EXP_", "STT_"))}
    env.update(DEFAULTS)
    env.update(settings)
    env.update({"ROUTER_PRECACHE_URL": "", "STT_HEARTBEAT_S": "0",
                "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    if port is not None:
        env["STT_PORT"] = str(port)
    return env
