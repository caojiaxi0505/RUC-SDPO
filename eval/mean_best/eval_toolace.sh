#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_entrypoint.sh"

# ToolACE 属于 tool-use 评测，默认关闭 Qwen3 thinking，使输出更贴近训练 validation 的 non-thinking 设置。
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  export CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'
fi

run_mean_best_entrypoint toolace 8192 "$@"
