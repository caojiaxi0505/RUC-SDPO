#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_entrypoint.sh"

run_mean_best_entrypoint toolace 8192 "$@"
