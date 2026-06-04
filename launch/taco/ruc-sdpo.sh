# 实验说明：TACO 上的 RUC-SDPO，只使用 reward-uplift calibration 加权蒸馏。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PROJECT_DIR

bash "${PROJECT_DIR}/launch/train.sh" ruc-sdpo taco "$@"
