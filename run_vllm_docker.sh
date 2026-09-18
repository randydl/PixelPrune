#!/bin/bash

sudo docker run --gpus '"device=7"' \
  --privileged --ipc=host -p 18000:8000 \
  -v /nas_train/app.e0016372/models:/models \
  -v /nas_train/app.e0016372/projects/PixelPrune:/pixelprune \
  vllm/vllm-openai:v0.29.0-cu129 /models/Qwen/Qwen3.5-0.8B \
  --served-model-name Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.25 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
