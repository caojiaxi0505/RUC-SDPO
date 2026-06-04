#!/usr/bin/env bash
set -euo pipefail

# 将 verl FSDP actor checkpoint 合并成 HuggingFace 格式，供 vLLM 加载。

ACTOR_CKPT=${ACTOR_CKPT:?Set ACTOR_CKPT to a verl global_step_*/actor checkpoint directory}
TARGET_DIR=${TARGET_DIR:?Set TARGET_DIR to the output HuggingFace model directory}
EXTRA_MERGE_ARGS=${EXTRA_MERGE_ARGS:-}
PYTHON_BIN=${PYTHON_BIN:-}

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_CKPT}" \
  --target_dir "${TARGET_DIR}" \
  ${EXTRA_MERGE_ARGS}
