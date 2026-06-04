#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash launch/train.sh <method> <dataset> [hydra_overrides...]

Methods:
  grpo
  sdpo
  ruc-sdpo
  ruc-sdpo-grpo

Datasets:
  apigen
  toolace
  openr1_math
  scienceqa
  medmcqa
  taco

Common environment overrides:
  PROJECT_DIR, OUTPUT_ROOT, EXPERIMENT_NAME
  MODEL_PATH, MODEL_TAG, SEED
  NUM_GPUS, CUDA_VISIBLE_DEVICES
  TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE, ROLLOUT_N, VAL_N
  LR, LR_WARMUP_STEPS, TOTAL_EPOCHS, TEST_FREQ, SAVE_FREQ
  TRAIN_MAX_SAMPLES, VAL_MAX_SAMPLES, DATA_SEED, DATA_SHUFFLE
  JF_POLICY, UPLIFT_AGGREGATION, UPLIFT_NUM_SAMPLES
  AUXILIARY_COEF, ALPHA, DISTILLATION_TOPK
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

METHOD="$1"
DATASET_INPUT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

case "${METHOD}" in
  grpo)
    CONFIG_NAME="baseline_grpo"
    ;;
  sdpo|ruc-sdpo|ruc-sdpo-grpo)
    CONFIG_NAME="sdpo"
    ;;
  *)
    echo "ERROR: unsupported method: ${METHOD}" >&2
    usage
    exit 1
    ;;
esac

DATASET_NAME="$(basename "${DATASET_INPUT}")"
case "${DATASET_NAME}" in
  apigen|toolace|openr1_math|scienceqa|medmcqa|taco)
    ;;
  *)
    echo "ERROR: unsupported dataset: ${DATASET_INPUT}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${DATASET_INPUT}" = /* ]]; then
  DATASET_DIR="${DATASET_INPUT}"
elif [[ "${DATASET_INPUT}" == datasets/* ]]; then
  DATASET_DIR="${PROJECT_DIR}/${DATASET_INPUT}"
else
  DATASET_DIR="${PROJECT_DIR}/datasets/${DATASET_NAME}"
fi

TRAIN_FILE="${DATASET_DIR}/train.jsonl"
VAL_FILE="${DATASET_DIR}/test.jsonl"
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  echo "ERROR: missing ${TRAIN_FILE} or ${VAL_FILE}" >&2
  echo "Run the dataset conversion script before launching training." >&2
  exit 1
fi

MODEL_TAG="${MODEL_TAG:?Set MODEL_TAG to a short model name used in EXPERIMENT_NAME}"
SEED="${SEED:?Set SEED to the experiment seed, for example SEED=1}"
DATA_SEED="${DATA_SEED:-${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${METHOD}-${DATASET_NAME}-${MODEL_TAG}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/ckpt/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${EXPERIMENT_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${ROLLOUT_DIR}" "${TENSORBOARD_DIR}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base HF model path}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_N="${VAL_N:-16}"
LR="${LR:-2e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18944}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
MAX_CKPTS="${MAX_CKPTS:-20}"
# 快速迭代数据集默认从训练集随机采样 6400 条；设为 -1 可恢复全量训练。
case "${DATASET_NAME}" in
  apigen|toolace|openr1_math|medmcqa|taco)
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-6400}"
    ;;
esac
# 快速迭代默认只从 test.jsonl 中采样 64 条做 validation；设为 -1 可恢复全量验证。
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
DATA_SHUFFLE="${DATA_SHUFFLE:-True}"

COMMON_ARGS=(
  "vars.dir=${PROJECT_DIR}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "critic.model.path=${MODEL_PATH}"
  "data.train_files=[\"${TRAIN_FILE}\"]"
  "data.val_files=[\"${VAL_FILE}\"]"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.shuffle=${DATA_SHUFFLE}"
  "data.seed=${DATA_SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
  "critic.model.fsdp_config.seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "algorithm.rollout_correction.rollout_is=token"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_CKPTS}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.rollout_data_dir=${ROLLOUT_DIR}/train"
  "trainer.validation_data_dir=${ROLLOUT_DIR}/val"
  'trainer.logger=["console","tensorboard"]'
)

[[ -n "${TRAIN_MAX_SAMPLES:-}" ]] && COMMON_ARGS+=("data.train_max_samples=${TRAIN_MAX_SAMPLES}")
COMMON_ARGS+=("data.val_max_samples=${VAL_MAX_SAMPLES}")

DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
ALPHA="${ALPHA:-0.5}"
JF_POLICY="${JF_POLICY:-ema_policy}"
UPLIFT_NUM_SAMPLES="${UPLIFT_NUM_SAMPLES:-1}"
UPLIFT_AGGREGATION="${UPLIFT_AGGREGATION:-uid}"
REWARD_UPPER_BOUND="${REWARD_UPPER_BOUND:-1.0}"
AUXILIARY_COEF="${AUXILIARY_COEF:-0.02}"

METHOD_ARGS=()
case "${METHOD}" in
  grpo)
    ;;
  sdpo)
    METHOD_ARGS+=(
      "actor_rollout_ref.actor.self_distillation.objective=jsd"
      "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
      "actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}"
      "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
      "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.enable=False"
    )
    ;;
  ruc-sdpo)
    METHOD_ARGS+=(
      "actor_rollout_ref.actor.self_distillation.objective=ruc-sdpo"
      "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
      "actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}"
      "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
      "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.enable=True"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.jf_policy=${JF_POLICY}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.num_samples=${UPLIFT_NUM_SAMPLES}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.aggregation=${UPLIFT_AGGREGATION}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.reward_upper_bound=${REWARD_UPPER_BOUND}"
    )
    ;;
  ruc-sdpo-grpo)
    METHOD_ARGS+=(
      "actor_rollout_ref.actor.self_distillation.objective=ruc-sdpo-grpo"
      "actor_rollout_ref.actor.self_distillation.auxiliary_coef=${AUXILIARY_COEF}"
      "actor_rollout_ref.actor.self_distillation.auxiliary_base_loss_mode=vanilla"
      "actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK}"
      "actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}"
      "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True"
      "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.enable=True"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.jf_policy=${JF_POLICY}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.num_samples=${UPLIFT_NUM_SAMPLES}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.aggregation=${UPLIFT_AGGREGATION}"
      "actor_rollout_ref.actor.self_distillation.uplift_calibration.reward_upper_bound=${REWARD_UPPER_BOUND}"
    )
    ;;
esac

[[ -n "${RANDOM_SELECT_SOLUTION:-}" ]] && METHOD_ARGS+=(
  "actor_rollout_ref.actor.self_distillation.random_select_solution=${RANDOM_SELECT_SOLUTION}"
)
[[ -n "${SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS:-}" ]] && METHOD_ARGS+=(
  "actor_rollout_ref.actor.self_distillation.sample_from_all_non_self_solutions=${SAMPLE_FROM_ALL_NON_SELF_SOLUTIONS}"
)

echo "=================================================================="
echo "experiment : ${EXPERIMENT_NAME}"
echo "method     : ${METHOD}"
echo "dataset    : ${DATASET_NAME}"
echo "seed       : ${SEED}"
echo "data seed  : ${DATA_SEED}"
echo "model      : ${MODEL_PATH}"
echo "gpus       : ${CUDA_VISIBLE_DEVICES}"
echo "ckpt dir   : ${CKPT_DIR}"
echo "log file   : ${LOG_FILE}"
echo "=================================================================="

cd "${PROJECT_DIR}"
bash training/verl_training.sh \
  "${EXPERIMENT_NAME}" \
  "${CONFIG_NAME}" \
  "${DATASET_DIR}" \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
