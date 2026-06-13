#!/usr/bin/env bash
# MedMCQA (Qwen3-8B)：串行执行 grpo / sdpo / ruc-sdpo-grpo(sd0.01) / ruc-sdpo-grpo(sd0.02)。
# 训练 9600 条样本；每 5 step 做一次 validation（采样 64 条）；保存全部权重。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

run_suite medmcqa 9600
