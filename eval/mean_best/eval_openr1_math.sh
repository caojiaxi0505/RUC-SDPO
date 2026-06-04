#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_entrypoint.sh"

# 默认关闭 Qwen3 thinking，使 mean_best 评测与训练 validation 的 non-thinking 设置对齐。
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  export CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'
fi

run_mean_best_entrypoint openr1_math 8192 "$@"
