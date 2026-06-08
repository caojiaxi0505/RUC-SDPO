#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

HOST="${TACO_VERIFIER_HOST:-127.0.0.1}"
PORT="${TACO_VERIFIER_PORT:-18080}"
WORKERS="${TACO_VERIFIER_WORKERS:-}"
MAX_TEST_CASES="${TACO_VERIFIER_MAX_TEST_CASES:-}"
BACKEND="${TACO_VERIFIER_BACKEND:-local}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --backend "${BACKEND}"
)

if [[ -n "${WORKERS}" ]]; then
  ARGS+=(--workers "${WORKERS}")
fi

if [[ -n "${MAX_TEST_CASES}" ]]; then
  ARGS+=(--max-test-cases "${MAX_TEST_CASES}")
fi

if [[ "${BACKEND}" == "ags" ]]; then
  : "${TACO_AGS_TEMPLATE:=${AGS_TEMPLATE:-}}"
  if [[ -z "${TACO_AGS_TEMPLATE}" ]]; then
    echo "[taco-verifier] ERROR: TACO_AGS_TEMPLATE or AGS_TEMPLATE is required when TACO_VERIFIER_BACKEND=ags." >&2
    exit 1
  fi
  ARGS+=(--ags-template "${TACO_AGS_TEMPLATE}")
  [[ -n "${TACO_AGS_SANDBOXES:-}" ]] && ARGS+=(--ags-sandboxes "${TACO_AGS_SANDBOXES}")
  [[ -n "${TACO_AGS_IMAGE:-}" ]] && ARGS+=(--ags-image "${TACO_AGS_IMAGE}")
  [[ -n "${TACO_AGS_IMAGE_REGISTRY_TYPE:-}" ]] && ARGS+=(--ags-image-registry-type "${TACO_AGS_IMAGE_REGISTRY_TYPE}")
  [[ -n "${TACO_AGS_CPUS:-}" ]] && ARGS+=(--ags-cpus "${TACO_AGS_CPUS}")
  [[ -n "${TACO_AGS_MEMORY_MB:-}" ]] && ARGS+=(--ags-memory-mb "${TACO_AGS_MEMORY_MB}")
  [[ -n "${TACO_AGS_SANDBOX_TIMEOUT:-}" ]] && ARGS+=(--ags-sandbox-timeout "${TACO_AGS_SANDBOX_TIMEOUT}")
  [[ -n "${TACO_AGS_COMMAND_TIMEOUT_BUFFER:-}" ]] && ARGS+=(--ags-command-timeout-buffer "${TACO_AGS_COMMAND_TIMEOUT_BUFFER}")
  [[ -n "${TACO_AGS_PYTHON_BIN:-}" ]] && ARGS+=(--ags-python-bin "${TACO_AGS_PYTHON_BIN}")
  [[ "${TACO_AGS_ALLOW_INTERNET:-0}" == "1" || "${TACO_AGS_ALLOW_INTERNET:-}" == "true" ]] && ARGS+=(--ags-allow-internet)
  [[ -n "${TACO_AGS_CUSTOM_CONFIG:-}" ]] && ARGS+=(--ags-custom-config "${TACO_AGS_CUSTOM_CONFIG}")
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m verl.utils.reward_score.feedback.taco_batch_verifier_server "${ARGS[@]}"
