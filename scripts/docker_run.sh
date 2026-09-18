#!/bin/bash

MODEL_NAME="Qwen3.8-27B-FP8"
MODEL_PATH="/models/Qwen/${MODEL_NAME}"

docker run --gpus all \
  --privileged --ipc=host -p 18000:8000 \
  -v /nas_train/app.e0016372/models:/models:ro \
  -e PIXELPRUNE_ENABLED=true \
  -e PIXELPRUNE_VERBOSE=true \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  pixelprune-vllm:0.29.0 "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --max-num-seqs 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.25 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  # --mm-encoder-tp-mode data
