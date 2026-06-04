#!/usr/bin/env bash
set -euo pipefail

# 启动本地 OpenAI 兼容 vLLM 服务。mean_best 独立维护该入口。

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the HF-format checkpoint directory}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}
PYTHON_BIN=${PYTHON_BIN:-}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}
DTYPE=${DTYPE:-bfloat16}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

echo "[serve] MODEL_PATH=${MODEL_PATH}"
echo "[serve] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[serve] HOST=${HOST} PORT=${PORT} TP_SIZE=${TP_SIZE} DP_SIZE=${DP_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  ${EXTRA_VLLM_ARGS}
