#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${1:-pixelprune-vllm:0.29.0}"
BASE="${2:-vllm/vllm-openai:v0.29.0-cu129}"

echo "Building $IMAGE (base: $BASE) ..."

sudo docker build \
    --build-arg VLLM_IMAGE="$BASE" \
    -f docker/Dockerfile -t "$IMAGE" .

echo "Done: $IMAGE"

sudo docker run --rm --entrypoint python3 "$IMAGE" -c \
    'import pixelprune; print("pixelprune", pixelprune.__version__, "ok")'
