#!/usr/bin/env bash
# launch_qwen3_8b 公共配置与串行运行逻辑（基于 Qwen3-8B）。
#
# 用法：数据集脚本先 source 本文件，再调用：
#     run_suite <dataset> <train_max_samples>
# 会按下面顺序“串行”执行 4 个实验（前一个结束后才开始下一个）：
#   1) grpo                  —— GRPO baseline
#   2) sdpo                  —— 原始 SDPO/JSD baseline
#   3) ruc-sdpo-grpo sd0.01  —— GRPO 主损失 + RUC-SDPO 辅助蒸馏，auxiliary_coef(sd系数)=0.01
#   4) ruc-sdpo-grpo sd0.02  —— 同上，auxiliary_coef(sd系数)=0.02
#
# 说明：故意不开启 `set -e`，这样单个实验失败时仍会继续跑后续实验，
#       最后统一打印每个实验的成功/失败情况，避免整夜串行任务因一个失败而全部中断。

set -uo pipefail

LAUNCH_8B_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${LAUNCH_8B_DIR}/.." && pwd)}"
export PROJECT_DIR

# ===== 基础模型 =====
export MODEL_PATH="${MODEL_PATH:-/cfs_turbo/jiaxicao/ckpt/hf/Qwen3-8B}"
export MODEL_TAG="${MODEL_TAG:-Qwen3-8B}"
export SEED="${SEED:-1}"

# ===== 输出目录隔离 =====
# 历史 Qwen3-8B 实验已落盘在 outputs/ 下，这里改用独立的 outputs_qwen3_8b/，
# 既不影响历史结果，又因为目录为空使 resume_mode=auto 自然从头训练。
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs_qwen3_8b}"

# ===== 通用 validation / 保存超参 =====
# 每 5 个 step 做一次 validation，同时每 5 个 step 保存一次权重
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
# max_actor_ckpt_to_keep=null 表示保留全部 checkpoint（“权重都保存下来”）
export MAX_CKPTS="${MAX_CKPTS:-null}"
# 每次 validation 采样 64 条样本
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"

# 记录本次 suite 各实验结果，结束时统一打印
declare -a SUITE_RESULTS=()

# run_one <method> <experiment_name> [auxiliary_coef]
# 通过 env 在子进程里设置本次实验“特有”的环境变量，避免污染其他实验。
run_one() {
  local method="$1"
  local experiment_name="$2"
  local aux_coef="${3:-}"

  # 本次实验特有的环境变量
  local -a run_env=("EXPERIMENT_NAME=${experiment_name}")

  # 只有 ruc-sdpo-grpo 需要自蒸馏相关配置（复用 Qwen3-8B 最优配置）
  if [[ "${method}" == "ruc-sdpo-grpo" ]]; then
    run_env+=(
      "AUXILIARY_COEF=${aux_coef}"              # sd 系数
      "RANDOM_SELECT_SOLUTION=True"             # random：随机选择 sibling demonstration
      "SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS=True" # nonself：从所有非自身 sibling response 中选
      "DISTILLATION_TEACHER_POLICY=ema_policy"  # teacher_ema：EMA teacher
      "JF_POLICY=ema_policy"                    # jf_ema：用 EMA policy 估计 J_f
      "DISTILLATION_TOPK=100"                   # topk=100
      "ALPHA=0.5"                               # alpha=0.5
      "UPLIFT_AGGREGATION=uid"                  # uid：同一 prompt 共享 uplift 权重
      "UPLIFT_NUM_SAMPLES=1"                    # k1：每个 prompt 额外采样 1 次估计 J_f
    )
  fi

  echo "######################################################################"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] 开始实验"
  echo "#   dataset    : ${SUITE_DATASET}"
  echo "#   method     : ${method}"
  echo "#   experiment : ${experiment_name}"
  [[ -n "${aux_coef}" ]] && echo "#   sd(aux_coef): ${aux_coef}"
  echo "######################################################################"

  env "${run_env[@]}" bash "${PROJECT_DIR}/launch/train.sh" "${method}" "${SUITE_DATASET}"
  local status=$?

  if [[ ${status} -eq 0 ]]; then
    SUITE_RESULTS+=("OK    ${experiment_name}")
  else
    SUITE_RESULTS+=("FAIL  ${experiment_name} (exit=${status})")
    echo "!!! 实验失败: ${experiment_name} (exit=${status})，继续执行后续实验 !!!"
  fi
}

# run_suite <dataset> <train_max_samples>
# 串行执行 grpo / sdpo / ruc-sdpo-grpo(sd0.01) / ruc-sdpo-grpo(sd0.02)
run_suite() {
  export SUITE_DATASET="$1"
  export TRAIN_MAX_SAMPLES="$2"

  local base="${SUITE_DATASET}-${MODEL_TAG}-seed${SEED}"
  local ruc_suffix="random-nonself-uid-k1-jf_ema-teacher_ema-alpha0p5"

  echo "====================================================================="
  echo "数据集 ${SUITE_DATASET} 串行实验开始 (train_max_samples=${TRAIN_MAX_SAMPLES})"
  echo "====================================================================="

  # 1) GRPO baseline
  run_one grpo "grpo-${base}"

  # 2) SDPO / JSD baseline
  run_one sdpo "sdpo-${base}"

  # 3) RUC-SDPO-GRPO, sd 系数 = 0.01
  run_one ruc-sdpo-grpo "ruc-sdpo-grpo-${base}-${ruc_suffix}-sd0p01" 0.01

  # 4) RUC-SDPO-GRPO, sd 系数 = 0.02
  run_one ruc-sdpo-grpo "ruc-sdpo-grpo-${base}-${ruc_suffix}-sd0p02" 0.02

  echo "===================== ${SUITE_DATASET} 实验汇总 ====================="
  local r
  for r in "${SUITE_RESULTS[@]}"; do
    echo "  ${r}"
  done
  echo "====================================================================="
}
