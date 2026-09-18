#!/bin/bash
# 启动带 PixelPrune 插件的 vLLM 服务（适配 vLLM >= 0.29）。
# 容器内先 `pip install -e /pixelprune`（注册 vllm.general_plugins entry point），
# 随后 `vllm serve` 启动时插件自动应用 monkey-patch。
sudo docker run --gpus '"device=7"' \
  --privileged --ipc=host -p 18000:8000 \
  -e PIXELPRUNE_ENABLED=true \
  -e PIXELPRUNE_VERBOSE=true \
  -v /nas_train/app.e0016372/models:/models \
  -v /nas_train/app.e0016372/projects/PixelPrune:/pixelprune \
  --entrypoint bash \
  vllm/vllm-openai:v0.29.0-cu129 -c \
  'pip install -e /pixelprune --no-build-isolation -q && exec vllm serve /models/Qwen/Qwen3.5-0.8B \
    --served-model-name Qwen3.5-0.8B \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.25 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3'
