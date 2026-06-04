#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_entrypoint.sh"

# TACO 代码生成在 Qwen3 thinking mode 下很容易把 token 花在 <think> 中，导致代码块被截断。
# 默认显式关闭 thinking，使评测与训练 validation 的 non-thinking 设置对齐。
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  export CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'
fi

run_mean_best_entrypoint taco 8192 "$@"
