FROM vllm/vllm-openai:v0.29.0-cu129

COPY setup.py /opt/pixelprune/setup.py
COPY pixelprune /opt/pixelprune/pixelprune

RUN pip install /opt/pixelprune --no-deps --no-build-isolation

ENV PIXELPRUNE_ENABLED=false \
    PIXELPRUNE_STRICT=true \
    PIXELPRUNE_METHOD=pred_2d \
    PIXELPRUNE_METRIC=max \
    PIXELPRUNE_THRESHOLD=0.0 \
    PIXELPRUNE_VERBOSE=false
