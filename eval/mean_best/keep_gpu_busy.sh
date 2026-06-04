#!/usr/bin/env bash
set -euo pipefail

# 持续生成压测脚本：启动或复用 vLLM，循环读取数据集 prompt 发请求。
# 不写 generations/scored/summary，vLLM 日志默认丢到 /dev/null。
#
# 示例：
#   TP_SIZE=4 DP_SIZE=2 bash eval/mean_best/keep_gpu_busy.sh /cfs_turbo/jiaxicao/ckpt/hf/Qwen3-8B

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

fail() {
  echo "[busy] ERROR: $*" >&2
  exit 1
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo python
  fi
}

abs_path() {
  local path="$1"

  if [[ -d "${path}" ]]; then
    (cd "${path}" && pwd)
    return
  fi

  if [[ -e "${path}" ]]; then
    local dir
    dir="$(cd "$(dirname "${path}")" && pwd)"
    printf '%s/%s\n' "${dir}" "$(basename "${path}")"
    return
  fi

  return 1
}

has_hf_weights() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  compgen -G "${dir}/*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "${dir}/pytorch_model*.bin" >/dev/null 2>&1 && return 0
  compgen -G "${dir}/model*.bin" >/dev/null 2>&1 && return 0
  return 1
}

is_hf_weight_file() {
  case "$(basename "$1")" in
    *.safetensors|pytorch_model*.bin|model*.bin) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_hf_model_path() {
  local path="$1"

  if [[ -f "${path}" ]]; then
    is_hf_weight_file "${path}" || fail "unsupported weight file: ${path}"
    path="$(dirname "${path}")"
  fi

  has_hf_weights "${path}" || fail "expected an HF model dir or weight file, got: ${path}"
  echo "${path}"
}

endpoint_ready() {
  curl -sf "${BASE_URL}/models" >/dev/null 2>&1
}

wait_for_endpoint() {
  echo "[busy] waiting for ${BASE_URL}/models (up to ${SERVE_WAIT_SECS}s)"
  local deadline=$((SECONDS + SERVE_WAIT_SECS))

  while true; do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      fail "vLLM exited before endpoint became ready"
    fi

    endpoint_ready && return

    if (( SECONDS >= deadline )); then
      fail "server not ready within ${SERVE_WAIT_SECS}s"
    fi

    sleep 5
  done
}

stop_server() {
  [[ -n "${SERVER_PID:-}" ]] || return

  echo "[busy] stopping vLLM server (pid ${SERVER_PID})"
  if [[ -n "${SERVER_PGID:-}" ]]; then
    kill -- "-${SERVER_PGID}" 2>/dev/null || true
    sleep 2
    kill -9 -- "-${SERVER_PGID}" 2>/dev/null || true
  else
    pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi

  wait "${SERVER_PID}" 2>/dev/null || true
}

start_or_reuse_server() {
  if [[ "${SKIP_SERVER_SETUP}" == "1" ]]; then
    wait_for_endpoint
    return
  fi

  endpoint_ready && fail "${BASE_URL}/models is already reachable. Set SKIP_SERVER_SETUP=1 or choose another PORT."

  echo "[busy] starting vLLM with tp=${TP_SIZE} dp=${DP_SIZE}; logs -> ${SERVER_LOG}"
  if command -v setsid >/dev/null 2>&1; then
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${MODEL_NAME}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    HOST="${HOST}" PORT="${PORT}" TP_SIZE="${TP_SIZE}" DP_SIZE="${DP_SIZE}" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" DTYPE="${DTYPE}" \
      setsid bash "${HERE}/serve_vllm.sh" >"${SERVER_LOG}" 2>&1 &
    SERVER_PGID=$!
  else
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${MODEL_NAME}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    HOST="${HOST}" PORT="${PORT}" TP_SIZE="${TP_SIZE}" DP_SIZE="${DP_SIZE}" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" DTYPE="${DTYPE}" \
      bash "${HERE}/serve_vllm.sh" >"${SERVER_LOG}" 2>&1 &
    SERVER_PGID=""
  fi

  SERVER_PID=$!
  wait_for_endpoint
}

if [[ "$#" -ne 1 || -z "${1:-}" ]]; then
  cat >&2 <<USAGE
Usage:
  TP_SIZE=4 DP_SIZE=2 bash eval/mean_best/keep_gpu_busy.sh <hf-model-path>

Required:
  <hf-model-path>  HF model directory or one weight file inside it.

Useful overrides:
  DATASET_NAME=apigen DATASET_FILE=/path/to/test.jsonl
  PORT=8000 TP_SIZE=4 DP_SIZE=1 CONCURRENCY=8 NUM_SAMPLES=16
  TEMPERATURE=0.6 TOP_P=0.95 TOP_K=-1 MAX_TOKENS=8192
  SKIP_SERVER_SETUP=1 BASE_URL=http://127.0.0.1:8000/v1
USAGE
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-"$(pick_python)"}
WEIGHT_PATH="$(abs_path "$1")" || fail "missing model path: $1"
MODEL_PATH="$(resolve_hf_model_path "${WEIGHT_PATH}")"

DATASET_NAME=${DATASET_NAME:-apigen}
DATASET_FILE=${DATASET_FILE:-"${REPO_ROOT}/datasets/${DATASET_NAME}/test.jsonl"}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BASE_URL=${BASE_URL:-"http://${HOST}:${PORT}/v1"}
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-18944}
DTYPE=${DTYPE:-bfloat16}
SERVE_WAIT_SECS=${SERVE_WAIT_SECS:-900}
SKIP_SERVER_SETUP=${SKIP_SERVER_SETUP:-0}
SERVER_LOG=${SERVER_LOG:-/dev/null}

NUM_SAMPLES=${NUM_SAMPLES:-16}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MAX_TOKENS=${MAX_TOKENS:-8192}
CONCURRENCY=${CONCURRENCY:-8}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
GENERATION_SEED=${GENERATION_SEED:-42}
LOG_EVERY=${LOG_EVERY:-20}

[[ -f "${DATASET_FILE}" ]] || fail "missing DATASET_FILE=${DATASET_FILE}"

echo "=================================================================="
echo "[busy] data file   : ${DATASET_FILE}"
echo "[busy] model path  : ${MODEL_PATH}"
echo "[busy] model name  : ${MODEL_NAME}"
echo "[busy] endpoint    : ${BASE_URL}"
echo "[busy] parallel    : tp=${TP_SIZE} dp=${DP_SIZE}"
echo "[busy] samples     : ${NUM_SAMPLES}"
echo "[busy] decoding    : temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} max_tokens=${MAX_TOKENS}"
echo "[busy] output      : no eval artifacts; vLLM log=${SERVER_LOG}"
echo "=================================================================="

trap stop_server EXIT
start_or_reuse_server

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
"${PYTHON_BIN}" "${HERE}/keep_gpu_busy.py" \
  --data-path "${DATASET_FILE}" \
  --model "${MODEL_NAME}" \
  --base-url "${BASE_URL}" \
  --num-samples "${NUM_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --seed "${GENERATION_SEED}" \
  --log-every "${LOG_EVERY}"
