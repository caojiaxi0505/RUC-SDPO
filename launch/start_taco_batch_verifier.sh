#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

HOST="${TACO_VERIFIER_HOST:-127.0.0.1}"
PORT="${TACO_VERIFIER_PORT:-18080}"
WORKERS="${TACO_VERIFIER_WORKERS:-}"
MAX_TEST_CASES="${TACO_VERIFIER_MAX_TEST_CASES:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${WORKERS}" ]]; then
  ARGS+=(--workers "${WORKERS}")
fi

if [[ -n "${MAX_TEST_CASES}" ]]; then
  ARGS+=(--max-test-cases "${MAX_TEST_CASES}")
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m verl.utils.reward_score.feedback.taco_batch_verifier_server "${ARGS[@]}"
