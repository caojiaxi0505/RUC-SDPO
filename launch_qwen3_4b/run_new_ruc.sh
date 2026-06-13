#!/usr/bin/env bash
# launch_qwen3_4b 新 RUC 方案入口：只执行新的 all-failed + Wilson gate + shrinkage RUC-SDPO-GRPO。
#
# 已完成的 grpo / sdpo / ruc-sdpo-grpo(sd0.01) / ruc-sdpo-grpo(sd0.02) 不会在这里重复执行。
#
# 用法：
#   bash launch_qwen3_4b/run_new_ruc.sh
#   bash launch_qwen3_4b/run_new_ruc.sh openr1_math medmcqa
#   UPLIFT_NUM_SAMPLES=2 bash launch_qwen3_4b/run_new_ruc.sh apigen
#
# 说明：故意不开启 `set -e`，单个数据集失败时继续后续数据集，最后统一打印汇总。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(openr1_math medmcqa apigen toolace)
fi

train_max_samples_for_dataset() {
  case "$1" in
    openr1_math|medmcqa|apigen)
      echo 9600
      ;;
    toolace)
      echo -1
      ;;
    *)
      return 1
      ;;
  esac
}

# 新算法默认配置；调用方仍可通过环境变量覆盖。
export AUXILIARY_COEF="${AUXILIARY_COEF:-0.02}"
export RANDOM_SELECT_SOLUTION="${RANDOM_SELECT_SOLUTION:-True}"
export SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS="${SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS:-True}"
export DISTILLATION_TEACHER_POLICY="${DISTILLATION_TEACHER_POLICY:-ema_policy}"
export JF_POLICY="${JF_POLICY:-ema_policy}"
export DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
export ALPHA="${ALPHA:-0.5}"
export UPLIFT_AGGREGATION="${UPLIFT_AGGREGATION:-uid}"
export UPLIFT_NUM_SAMPLES="${UPLIFT_NUM_SAMPLES:-1}"
export UPLIFT_ALL_FAILED_ONLY="${UPLIFT_ALL_FAILED_ONLY:-True}"
export UPLIFT_CONFIDENCE_GATE="${UPLIFT_CONFIDENCE_GATE:-wilson}"
export UPLIFT_MIN_UPLIFT="${UPLIFT_MIN_UPLIFT:-0.02}"
export UPLIFT_DELTA="${UPLIFT_DELTA:-0.1}"
export UPLIFT_Z="${UPLIFT_Z:-null}"
export UPLIFT_SHRINKAGE="${UPLIFT_SHRINKAGE:-jeffreys}"
export UPLIFT_ACTIVE_ONLY_DISTILLATION="${UPLIFT_ACTIVE_ONLY_DISTILLATION:-True}"
export GRAD_DIAGNOSTICS_ENABLE="${GRAD_DIAGNOSTICS_ENABLE:-False}"
export GRAD_DIAGNOSTICS_EVERY_N_STEPS="${GRAD_DIAGNOSTICS_EVERY_N_STEPS:-20}"

declare -a NEW_RUC_RESULTS=()

label_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//\//_}"
  echo "${value}"
}

run_new_one() {
  local experiment_name="$1"

  local -a run_env=(
    "EXPERIMENT_NAME=${experiment_name}"
    "AUXILIARY_COEF=${AUXILIARY_COEF}"
    "RANDOM_SELECT_SOLUTION=${RANDOM_SELECT_SOLUTION}"
    "SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS=${SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS}"
    "DISTILLATION_TEACHER_POLICY=${DISTILLATION_TEACHER_POLICY}"
    "JF_POLICY=${JF_POLICY}"
    "DISTILLATION_TOPK=${DISTILLATION_TOPK}"
    "ALPHA=${ALPHA}"
    "UPLIFT_AGGREGATION=${UPLIFT_AGGREGATION}"
    "UPLIFT_NUM_SAMPLES=${UPLIFT_NUM_SAMPLES}"
    "UPLIFT_ALL_FAILED_ONLY=${UPLIFT_ALL_FAILED_ONLY}"
    "UPLIFT_CONFIDENCE_GATE=${UPLIFT_CONFIDENCE_GATE}"
    "UPLIFT_MIN_UPLIFT=${UPLIFT_MIN_UPLIFT}"
    "UPLIFT_DELTA=${UPLIFT_DELTA}"
    "UPLIFT_Z=${UPLIFT_Z}"
    "UPLIFT_SHRINKAGE=${UPLIFT_SHRINKAGE}"
    "UPLIFT_ACTIVE_ONLY_DISTILLATION=${UPLIFT_ACTIVE_ONLY_DISTILLATION}"
    "GRAD_DIAGNOSTICS_ENABLE=${GRAD_DIAGNOSTICS_ENABLE}"
    "GRAD_DIAGNOSTICS_EVERY_N_STEPS=${GRAD_DIAGNOSTICS_EVERY_N_STEPS}"
  )

  echo "######################################################################"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] 开始新 RUC 实验"
  echo "#   dataset    : ${SUITE_DATASET}"
  echo "#   method     : ruc-sdpo-grpo"
  echo "#   experiment : ${experiment_name}"
  echo "#   sd(aux_coef): ${AUXILIARY_COEF}"
  echo "#   gate       : all_failed=${UPLIFT_ALL_FAILED_ONLY}, ${UPLIFT_CONFIDENCE_GATE}, tau=${UPLIFT_MIN_UPLIFT}, delta=${UPLIFT_DELTA}, shrinkage=${UPLIFT_SHRINKAGE}"
  echo "#   active-only: ${UPLIFT_ACTIVE_ONLY_DISTILLATION}"
  echo "#   grad debug : enable=${GRAD_DIAGNOSTICS_ENABLE}, every=${GRAD_DIAGNOSTICS_EVERY_N_STEPS}"
  echo "######################################################################"

  env "${run_env[@]}" bash "${PROJECT_DIR}/launch/train.sh" ruc-sdpo-grpo "${SUITE_DATASET}"
  local status=$?

  if [[ ${status} -eq 0 ]]; then
    NEW_RUC_RESULTS+=("OK    ${experiment_name}")
  else
    NEW_RUC_RESULTS+=("FAIL  ${experiment_name} (exit=${status})")
    echo "!!! 新 RUC 实验失败: ${experiment_name} (exit=${status})，继续执行后续数据集 !!!"
  fi
}

for ds in "${DATASETS[@]}"; do
  train_max_samples="$(train_max_samples_for_dataset "${ds}")"
  status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "!!! 跳过未知数据集: ${ds} !!!"
    NEW_RUC_RESULTS+=("SKIP  ${ds} (unknown dataset)")
    continue
  fi

  export SUITE_DATASET="${ds}"
  export TRAIN_MAX_SAMPLES="${train_max_samples}"

  base="${SUITE_DATASET}-${MODEL_TAG}-seed${SEED}"
  delta_label="$(label_value "${UPLIFT_DELTA}")"
  tau_label="$(label_value "${UPLIFT_MIN_UPLIFT}")"
  alpha_label="$(label_value "${ALPHA}")"
  aux_label="$(label_value "${AUXILIARY_COEF}")"
  ruc_suffix="allfail-${UPLIFT_CONFIDENCE_GATE}-d${delta_label}-tau${tau_label}-${UPLIFT_SHRINKAGE}-activeonly-random-nonself-${UPLIFT_AGGREGATION}-k${UPLIFT_NUM_SAMPLES}-jf_${JF_POLICY}-teacher_${DISTILLATION_TEACHER_POLICY}-alpha${alpha_label}"
  experiment_name="ruc-sdpo-grpo-new-${base}-${ruc_suffix}-sd${aux_label}"

  echo "====================================================================="
  echo "数据集 ${SUITE_DATASET} 新 RUC 实验开始 (train_max_samples=${TRAIN_MAX_SAMPLES})"
  echo "====================================================================="

  run_new_one "${experiment_name}"
done

echo "===================== Qwen3-4B 新 RUC 实验汇总 ====================="
for r in "${NEW_RUC_RESULTS[@]}"; do
  echo "  ${r}"
done
echo "====================================================================="
