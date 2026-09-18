#!/bin/bash

sudo docker run --gpus all \
  --ipc=host \
  -p 18000:8000 \
  -v /nas_train/app.e0016372/models:/models:ro \
  -e PIXELPRUNE_ENABLED=true \
  -e PIXELPRUNE_VERBOSE=true \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  pixelprune-vllm:0.29.0 /models/Qwen/Qwen3.5-0.8B \
  --served-model-name Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --max-num-seqs 256 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.25 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
