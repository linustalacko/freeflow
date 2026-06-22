#!/usr/bin/env bash
#
# Fine-tune a small Qwen LoRA for FreeFlow's post-processing, ON THIS MAC, then
# serve it locally. No cloud, no API keys, no GPU rental — all on the M4 Pro.
#
#   1. export FreeFlow history -> train/valid/test JSONL (export_training_data.py)
#   2. LoRA fine-tune a small Qwen on-device             (MLX-LM)
#   3. serve it as a local OpenAI endpoint               (mlx_lm.server)
#   4. point the router at it                            (LOCAL_MODEL / OLLAMA_URL)
#
# One-time setup:  pip install mlx-lm
# Run:             bash finetune_local.sh
#
# FIND THE SMALLEST MODEL THAT'S GOOD ENOUGH — run the ladder and eval each:
#   for M in Qwen2.5-0.5B-Instruct-4bit Qwen2.5-1.5B-Instruct-4bit Qwen2.5-3B-Instruct-4bit; do
#     BASE_MODEL="mlx-community/$M" ADAPTERS="$HOME/.freeflow-ft/adapters-$M" bash finetune_local.sh
#   done
# then compare them with eval_models.py (see that script). Ship the smallest whose
# WER-to-your-gold is near the gpt-oss bar. On a 24GB M4 Pro even 7B fits fine, so
# pick on QUALITY + LATENCY, not memory — smaller is faster and the task is narrow.
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# mlx-lm needs Python 3.12 (system python may be 3.14, which has no mlx wheels).
# Default to the project venv created during setup; override with PYTHON=...
PYTHON="${PYTHON:-$HOME/.freeflow-ft/venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"

BASE_MODEL="${BASE_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"  # 4-bit; start small
WORK="${WORK:-$HOME/.freeflow-ft}"
ADAPTERS="${ADAPTERS:-$WORK/adapters}"
ITERS="${ITERS:-600}"
PORT="${PORT:-8080}"
MIN_EXAMPLES="${MIN_EXAMPLES:-200}"

mkdir -p "$WORK"

echo "==> 1/4  export FreeFlow history -> train/valid/test (test = held-out gold)"
"$PYTHON" "$HERE/export_training_data.py" --split-dir "$WORK"
TRAIN_N=$(wc -l < "$WORK/train.jsonl" | tr -d ' ')
VALID_N=$(wc -l < "$WORK/valid.jsonl" | tr -d ' ')
echo "    train=$TRAIN_N valid=$VALID_N  (test.jsonl is held out — eval_models.py only)"
if [ "$TRAIN_N" -lt "$MIN_EXAMPLES" ]; then
  echo ""
  echo "!!  Only $TRAIN_N training examples — too few to fine-tune well (want ~500+,"
  echo "    ideally with human corrections). Keep dictating + correcting inline and"
  echo "    re-run. Exiting before wasting a training run."
  exit 1
fi

echo "==> 2/4  LoRA fine-tune on-device (MLX) — runs on the Mac's GPU"
echo "    base: $BASE_MODEL  ->  adapter: $ADAPTERS"
"$PYTHON" -m mlx_lm lora \
  --model "$BASE_MODEL" \
  --train \
  --data "$WORK" \
  --adapter-path "$ADAPTERS" \
  --batch-size 1 \
  --num-layers 8 \
  --iters "$ITERS" \
  --steps-per-eval 100

echo "==> 3/4  quick sanity generation with the new adapter"
"$PYTHON" -m mlx_lm generate \
  --model "$BASE_MODEL" --adapter-path "$ADAPTERS" \
  --max-tokens 120 \
  --prompt "Clean up this dictation: so um i wanted to like follow up on the the thing we discussed yesterday about pricing" || true

cat <<EOF

==> 4/4  measure it, then serve it

  # Did it actually improve? Compare base vs +adapter vs the gpt-oss bar:
  python3 "$HERE/eval_models.py" --test "$WORK/test.jsonl" \\
    --mlx "$BASE_MODEL" \\
    --mlx "$BASE_MODEL@$ADAPTERS"
  # (add --endpoint https://api.groq.com/openai/v1/chat/completions \\
  #       --endpoint-model openai/gpt-oss-20b --api-key-env GROQ_API_KEY \\
  #       --endpoint-label gpt-oss-20b   to score the current production bar too)

  # Serve the fine-tuned model as an OpenAI-compatible endpoint on :$PORT
  "$PYTHON" -m mlx_lm server --model "$BASE_MODEL" --adapter-path "$ADAPTERS" --port $PORT

  # then point the router at it (in the launchd plist / env):
  #   OLLAMA_URL = http://127.0.0.1:$PORT/v1/chat/completions
  #   LOCAL_MODEL = $BASE_MODEL
  # and force-local (skip Groq) if you want it as the primary post-processor.

Inference cost from here on: \$0. Private. On-device. Re-run whenever you've
collected more corrections to hill-climb the model.
EOF
