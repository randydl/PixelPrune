#!/bin/bash
# PixelPrune vLLM 启动包装脚本（用于"挂载仓库 + 直接启动"的快速迭代流程）。
#
# 为什么需要它：PixelPrune 通过 setup.py 里声明的 `vllm.general_plugins` entry point
# 自动加载。entry point 由**已安装发行版**的元数据提供，因此仅挂载仓库 + 设置
# PYTHONPATH 是不够的 —— vLLM 启动时不会发现插件，补丁不会被应用。
#
# 用法：
#   docker run --gpus all --ipc=host -p 18000:8000 \
#     -v /mnt/models:/models \
#     -v /mnt/projects/PixelPrune:/pixelprune \
#     -e PIXELPRUNE_ENABLED=true \
#     --entrypoint /pixelprune/scripts/vllm_entrypoint.sh \
#     vllm/vllm-openai:v0.29.0-cu129 \
#     /models/Qwen/Qwen3.5-2B --served-model-name Qwen3.5-2B ...
#
# 所有参数原样透传给 `vllm serve`。
set -euo pipefail

PP_SRC="${PIXELPRUNE_SRC:-/pixelprune}"

if [ ! -d "$PP_SRC" ]; then
    echo "[pixelprune] 找不到 $PP_SRC —— 请确认已用 -v 挂载 PixelPrune 仓库" >&2
    exit 1
fi

# 可编辑安装：改代码立即生效，无需重建镜像。
# 幂等 —— 已安装则跳过，避免每次启动重复构建。
if ! python3 -c "import pixelprune" 2>/dev/null; then
    echo "[pixelprune] 安装 $PP_SRC ..."
    pip install -e "$PP_SRC" --no-deps -q
fi

echo "[pixelprune] ENABLED=${PIXELPRUNE_ENABLED:-<unset>} \
METHOD=${PIXELPRUNE_METHOD:-pred_2d} \
METRIC=${PIXELPRUNE_METRIC:-max} \
THRESHOLD=${PIXELPRUNE_THRESHOLD:-0.0} \
VERBOSE=${PIXELPRUNE_VERBOSE:-false}"

exec vllm serve "$@"
