# 实验说明：ToolACE 上的 RUC-SDPO-GRPO，GRPO 主损失结合 RUC-SDPO 辅助蒸馏。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PROJECT_DIR

bash "${PROJECT_DIR}/launch/train.sh" ruc-sdpo-grpo toolace "$@"
