# 实验说明：OpenR1-Math 上的 GRPO baseline，用于方法主轴对照。
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PROJECT_DIR

bash "${PROJECT_DIR}/launch/train.sh" grpo openr1_math "$@"
