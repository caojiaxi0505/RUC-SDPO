#!/usr/bin/env bash
set -euo pipefail

# 停止 keep_gpu_busy.sh 启动的请求 client 和本地 vLLM 服务。
# 用法：
#   bash eval/mean_best/keep_gpu_free.sh
#   FORCE_KILL=0 bash eval/mean_best/keep_gpu_free.sh  # 只发送 TERM，不发送 KILL

FORCE_KILL=${FORCE_KILL:-1}
TERM_WAIT_SECS=${TERM_WAIT_SECS:-3}

patterns=(
  "eval/mean_best/keep_gpu_busy.py"
  "eval/mean_best/keep_gpu_busy.sh"
  "eval/mean_best/serve_vllm.sh"
  "vllm.entrypoints.openai.api_server"
  "VLLM::Worker"
)

kill_pattern() {
  local signal="$1"
  local pattern="$2"

  if pgrep -af "${pattern}" >/dev/null 2>&1; then
    echo "[free] ${signal} ${pattern}"
    pkill "-${signal}" -f "${pattern}" 2>/dev/null || true
  fi
}

echo "[free] stopping keep_gpu_busy client and vLLM server"
for pattern in "${patterns[@]}"; do
  kill_pattern TERM "${pattern}"
done

sleep "${TERM_WAIT_SECS}"

if [[ "${FORCE_KILL}" == "1" ]]; then
  for pattern in "${patterns[@]}"; do
    kill_pattern KILL "${pattern}"
  done
fi

echo "[free] remaining related processes:"
if ! ps -ef | grep -E 'keep_gpu_busy|serve_vllm|vllm.entrypoints.openai.api_server|VLLM::Worker' | grep -v grep; then
  echo "[free] none"
fi
