#!/usr/bin/env bash
set -euo pipefail

# 单数据集 mean@k / best@k 评估流程：
# 1. 读取数据集、权重、采样和 vLLM 参数；
# 2. 将传入权重解析成 vLLM 可加载的 HuggingFace 模型目录；
# 3. 启动本地 vLLM 服务，或复用已有 OpenAI 兼容服务；
# 4. 调用 evaluate_mean_best.py 生成、打分并汇总。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

fail() {
  echo "[eval] ERROR: $*" >&2
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

safe_name() {
  local name="$1"
  name="${name// /_}"
  name="${name//\//_}"
  name="${name//:/_}"
  echo "${name}"
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

infer_label() {
  local path="$1"

  if [[ -n "${STEP:-}" ]]; then
    echo "step_${STEP}"
  elif [[ "${path}" =~ global_step_([0-9]+) ]]; then
    STEP="${BASH_REMATCH[1]}"
    echo "step_${STEP}"
  elif [[ -f "${path}" ]]; then
    basename "$(dirname "${path}")"
  else
    basename "${path}"
  fi
}

find_actor_dir() {
  local dir="$1"

  # 支持三类常见输入：actor 目录、global_step_* 目录、训练 run 根目录。
  if [[ "$(basename "${dir}")" == "actor" ]]; then
    echo "${dir}"
  elif [[ "$(basename "${dir}")" == "huggingface" && "$(basename "$(dirname "${dir}")")" == "actor" ]]; then
    dirname "${dir}"
  elif [[ -d "${dir}/actor" ]]; then
    echo "${dir}/actor"
  elif [[ -n "${STEP:-}" && -d "${dir}/global_step_${STEP}/actor" ]]; then
    echo "${dir}/global_step_${STEP}/actor"
  else
    return 1
  fi
}

merge_actor_dir() {
  local actor_dir="$1"

  [[ -d "${actor_dir}" ]] || fail "missing actor checkpoint: ${actor_dir}"

  if [[ "${FORCE_MERGE}" == "1" ]]; then
    rm -rf "${MERGED_DIR}"
  fi

  if has_hf_weights "${MERGED_DIR}"; then
    echo "[eval] reusing merged HF model: ${MERGED_DIR}" >&2
  else
    echo "[eval] merging actor checkpoint -> ${MERGED_DIR}" >&2
    PYTHON_BIN="${PYTHON_BIN}" \
    ACTOR_CKPT="${actor_dir}" \
    TARGET_DIR="${MERGED_DIR}" \
      bash "${HERE}/merge_fsdp_actor_to_hf.sh" >&2
  fi

  echo "${MERGED_DIR}"
}

resolve_model_path() {
  local path="${WEIGHT_PATH_ABS}"

  # 单个权重文件按“所在目录是 HF 模型目录”处理。
  if [[ -f "${path}" ]]; then
    is_hf_weight_file "${path}" || fail "unsupported weight file: ${path}"
    local model_dir
    model_dir="$(dirname "${path}")"
    has_hf_weights "${model_dir}" || fail "no HF weights in ${model_dir}"
    echo "${model_dir}"
    return
  fi

  [[ -d "${path}" ]] || fail "weight path must be a file or directory: ${path}"

  if [[ "${FORCE_MERGE}" != "1" ]]; then
    if has_hf_weights "${path}"; then
      echo "${path}"
      return
    fi
  fi

  if [[ "${FORCE_MERGE}" != "1" ]]; then
    if has_hf_weights "${path}/huggingface"; then
      echo "${path}/huggingface"
      return
    fi
  fi

  local actor_dir
  actor_dir="$(find_actor_dir "${path}")" || {
    fail "cannot resolve ${path}; pass an HF dir, weight file, actor dir, or run dir with STEP set"
  }
  merge_actor_dir "${actor_dir}"
}

endpoint_ready() {
  curl -sf "${BASE_URL}/models" >/dev/null 2>&1
}

wait_for_endpoint() {
  echo "[eval] waiting for ${BASE_URL}/models (up to ${SERVE_WAIT_SECS}s)"

  local deadline=$((SECONDS + SERVE_WAIT_SECS))
  while true; do
    # 如果服务是本脚本启动的，进程提前退出时直接给出日志尾部。
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[eval] ERROR: vLLM exited early. Tail of ${SERVER_LOG}:" >&2
      tail -n 40 "${SERVER_LOG}" >&2 || true
      exit 1
    fi

    endpoint_ready && return

    if (( SECONDS >= deadline )); then
      echo "[eval] ERROR: server not ready within ${SERVE_WAIT_SECS}s" >&2
      [[ -n "${SERVER_LOG:-}" ]] && tail -n 40 "${SERVER_LOG}" >&2 || true
      exit 1
    fi

    sleep 5
  done
}

stop_server() {
  [[ -n "${SERVER_PID:-}" ]] || return

  echo "[eval] stopping vLLM server (pid ${SERVER_PID})"
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

  MODEL_PATH="$(resolve_model_path)"
  echo "[eval] model path  : ${MODEL_PATH}"

  SERVER_LOG="${LOG_DIR}/vllm_${RUN_LABEL}.log"
  echo "[eval] starting vLLM (log: ${SERVER_LOG})"

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

run_evaluator() {
  local extra_args=()

  [[ -n "${METRIC_KS}" ]] && extra_args+=(--metric-ks "${METRIC_KS}")
  [[ -n "${LIMIT}" ]] && extra_args+=(--limit "${LIMIT}")
  [[ "${OVERWRITE}" == "1" ]] && extra_args+=(--overwrite)

  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

  "${PYTHON_BIN}" "${HERE}/evaluate_mean_best.py" \
    --dataset "${DATASET_NAME}" \
    --data-path "${DATASET_FILE}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --seed "${GENERATION_SEED}" \
    --concurrency "${CONCURRENCY}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --retries "${RETRIES}" \
    "${extra_args[@]}"
}

DATASET_NAME=${DATASET_NAME:?Set DATASET_NAME}
WEIGHT_PATH=${WEIGHT_PATH:?Pass the required <weight-path> argument to eval_*.sh}

[[ -z "${MODEL_PATH_OVERRIDE:-}" ]] || {
  fail "MODEL_PATH_OVERRIDE is no longer supported; pass eval_*.sh <weight-path>"
}

PYTHON_BIN=${PYTHON_BIN:-"$(pick_python)"}
WEIGHT_PATH_ABS="$(abs_path "${WEIGHT_PATH}")" || fail "missing weight path: ${WEIGHT_PATH}"
if [[ -z "${RUN_LABEL:-}" ]]; then
  RUN_LABEL="$(infer_label "${WEIGHT_PATH_ABS}")"
fi
RUN_LABEL="$(safe_name "${RUN_LABEL}")"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
DATASET_FILE=${DATASET_FILE:-"${REPO_ROOT}/datasets/${DATASET_NAME}/test.jsonl"}
EVAL_ROOT=${EVAL_ROOT:-"${REPO_ROOT}/eval_outputs/mean_best/${DATASET_NAME}/${RUN_LABEL}"}
OUT_DIR="${EVAL_ROOT}"
LOG_DIR="${EVAL_ROOT}/logs"
MERGED_DIR="${EVAL_ROOT}/merged_hf"

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
FORCE_MERGE=${FORCE_MERGE:-0}

NUM_SAMPLES=${NUM_SAMPLES:-16}
METRIC_KS=${METRIC_KS:-}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MAX_TOKENS=${MAX_TOKENS:-8192}
CONCURRENCY=${CONCURRENCY:-8}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
RETRIES=${RETRIES:-3}
GENERATION_SEED=${GENERATION_SEED:-42}
LIMIT=${LIMIT:-}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
[[ -f "${DATASET_FILE}" ]] || fail "missing DATASET_FILE=${DATASET_FILE}"

echo "=================================================================="
echo "[eval] dataset     : ${DATASET_NAME}"
echo "[eval] data file   : ${DATASET_FILE}"
echo "[eval] weight path : ${WEIGHT_PATH_ABS}"
echo "[eval] run label   : ${RUN_LABEL}"
echo "[eval] model name  : ${MODEL_NAME}"
echo "[eval] python      : ${PYTHON_BIN}"
echo "[eval] endpoint    : ${BASE_URL}"
echo "[eval] parallel    : tp=${TP_SIZE} dp=${DP_SIZE}"
echo "[eval] samples     : ${NUM_SAMPLES} (ks=${METRIC_KS:-auto})"
echo "[eval] decoding    : temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} max_tokens=${MAX_TOKENS}"
echo "[eval] eval root   : ${EVAL_ROOT}"
echo "=================================================================="

trap stop_server EXIT
start_or_reuse_server
run_evaluator

echo "[eval] DONE. outputs: ${OUT_DIR}"
