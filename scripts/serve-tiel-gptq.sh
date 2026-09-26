#!/usr/bin/env bash
set -euo pipefail

# Set MODEL_DIR to the directory containing this release.
MODEL_DIR="${MODEL_DIR:-/models/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP}"
IMAGE="${VLLM_XPU_IMAGE:-vllm-xpu-arc:local}"

docker run --rm --name tiel-coder \
  --device /dev/dri --ipc host --shm-size 4g \
  -p 8000:8000 \
  -v "${MODEL_DIR}:/models/tiel-coder:ro" \
  "$IMAGE" serve /models/tiel-coder \
  --chat-template /models/tiel-coder/chat_template.jinja \
  --served-model-name tiel-coder \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.97 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --mm-processor-kwargs '{"max_pixels":4194304}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000 \
  --enable-prompt-tokens-details
