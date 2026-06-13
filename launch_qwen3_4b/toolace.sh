#!/usr/bin/env bash
# ToolACE (Qwen3-4B)：串行执行 grpo / sdpo / ruc-sdpo-grpo(sd0.01) / ruc-sdpo-grpo(sd0.02)。
# 全量训练（train_max_samples=-1，约 9326 条）；每 5 step 做一次 validation（采样 64 条）；保存全部权重。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_suite toolace -1
