#!/usr/bin/env bash
# launch_qwen3_8b 总入口：串行执行四个数据集的全部训练。
#
# 默认依次运行：openr1_math.sh -> medmcqa.sh -> apigen.sh -> toolace.sh，
# 每个数据集内部又串行执行 grpo / sdpo / ruc-sdpo-grpo(sd0.01) / ruc-sdpo-grpo(sd0.02)，
# 因此总共串行跑 4 数据集 × 4 方法 = 16 个训练。
#
# 用法：
#   bash launch_qwen3_8b/run_all.sh                 # 跑全部 4 个数据集
#   bash launch_qwen3_8b/run_all.sh apigen toolace  # 只跑指定数据集（按给定顺序）
#   SEED=2 bash launch_qwen3_8b/run_all.sh          # 环境变量会透传给各数据集脚本
#
# 说明：故意不开启 `set -e`，单个数据集脚本异常退出也会继续后续数据集，最后打印总汇总。
# 提示：整体耗时很长，建议放到后台运行，例如：
#   nohup bash launch_qwen3_8b/run_all.sh > run_all_qwen3_8b.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 数据集执行顺序：命令行传参则用传参，否则默认全部四个
if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(openr1_math medmcqa apigen toolace)
fi

declare -a ALL_RESULTS=()

for ds in "${DATASETS[@]}"; do
  ds_script="${SCRIPT_DIR}/${ds}.sh"
  if [[ ! -f "${ds_script}" ]]; then
    echo "!!! 跳过未知数据集脚本: ${ds_script} !!!"
    ALL_RESULTS+=("SKIP  ${ds}.sh (not found)")
    continue
  fi

  echo "#####################################################################"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] 开始数据集: ${ds}"
  echo "#####################################################################"

  bash "${ds_script}"
  status=$?

  if [[ ${status} -eq 0 ]]; then
    ALL_RESULTS+=("OK    ${ds}.sh")
  else
    ALL_RESULTS+=("FAIL  ${ds}.sh (exit=${status})")
    echo "!!! 数据集脚本异常退出: ${ds}.sh (exit=${status})，继续后续数据集 !!!"
  fi
done

echo "===================== 全部数据集执行汇总 ====================="
for r in "${ALL_RESULTS[@]}"; do
  echo "  ${r}"
done
echo "============================================================="
echo "注意：各方法(grpo/sdpo/ruc-sdpo-grpo)的成功/失败详情见每个数据集自己的汇总输出。"
