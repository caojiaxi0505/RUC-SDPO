#!/usr/bin/env bash

# Common positional-argument handling for mean@k / best@k dataset wrappers.

mean_best_usage() {
  local script_name="$1"
  local dataset_name="$2"
  local default_max_tokens="$3"

  cat >&2 <<USAGE
Usage:
  bash eval/mean_best/${script_name} <weight-path>

Required:
  <weight-path>  Local HF model directory, a weight file inside that directory,
                 a verl global_step_*/actor checkpoint directory, or a run
                 directory together with STEP.

Dataset defaults:
  DATASET_NAME=${dataset_name}
  MAX_TOKENS=${default_max_tokens}

Useful overrides:
  EVAL_ROOT=/path/to/output
  STEP=630                    # only needed when <weight-path> is a run directory
  SKIP_SERVER_SETUP=1         # reuse BASE_URL, but <weight-path> is still required
  PORT=8001 NUM_SAMPLES=16 TEMPERATURE=0.6 TOP_P=0.95 TOP_K=-1
  METRIC_KS=1,2,4,8,16
USAGE
}

run_mean_best_entrypoint() {
  local dataset_name="$1"
  local default_max_tokens="$2"
  shift 2

  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local script_name
  script_name="$(basename "$0")"

  if [[ "$#" -ne 1 || -z "${1:-}" ]]; then
    mean_best_usage "${script_name}" "${dataset_name}" "${default_max_tokens}"
    exit 2
  fi

  export DATASET_NAME="${DATASET_NAME:-${dataset_name}}"
  export MAX_TOKENS="${MAX_TOKENS:-${default_max_tokens}}"
  export WEIGHT_PATH="$1"

  bash "${here}/_run_one_dataset.sh"
}
