"""
vLLM Qwen3.5 monkey-patch (vLLM >= 0.29): 使用 PixelPrune 选择器计算 keep_indices。

Qwen3.5 在 vLLM 中通过 qwen3_5.py 定义，继承自 Qwen3VLForConditionalGeneration，
共享同一个 Qwen3VLMultiModalProcessor 和 Qwen3_VisionTransformer。

与 Qwen3-VL 的关键差异（vLLM 0.29）：
- ``Qwen3_5ForConditionalGeneration.get_mrope_input_positions`` 被覆盖为返回
  扁平位置 (arange 广播到 (3, seq))，因此 **不需要** 像 Qwen3-VL 那样用
  keep_indices 修正空间 mrope —— 占位符数量已在 processor 中按裁剪结果设置，
  arange(len(input_tokens)) 自然正确。
- ``supports_multimodal_pruning = True``，但仅在启用 video pruning 时才会
  走 EVS / recompute_mrope_positions 路径；默认配置下图像不受影响。
- 共享 ``_parse_and_validate_image_input`` / ``_process_image_input`` /
  ``get_encoder_cudagraph_config`` / VisionTransformer.forward，均通过 patch
  父类生效。

继承关系：
  Qwen3_5ForConditionalGeneration → Qwen3VLForConditionalGeneration
  Qwen3_5MoeForConditionalGeneration → Qwen3_5ForConditionalGeneration

环境变量：PIXELPRUNE_ENABLED, PIXELPRUNE_METHOD, PIXELPRUNE_METRIC,
PIXELPRUNE_THRESHOLD, PIXELPRUNE_VERBOSE。在创建 vLLM LLM 前调用 apply_patches()。
"""

from __future__ import annotations

try:
    from vllm.model_executor.models import qwen3_5
except ImportError:
    import importlib
    qwen3_5 = importlib.import_module("vllm.model_executor.models.qwen3_5")

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:
    logger = None

# Reuse patch functions from qwen3_vl_vllm
from .qwen3_vl_vllm import (
    apply_patches as _apply_qwen3vl_patches,
    _parse_image_input,
    _process_image_input,
    _get_encoder_cudagraph_config,
    _patch,
)


def apply_patches() -> None:
    """对 vLLM Qwen3.5 应用 PixelPrune monkey-patch。

    1. 先 apply qwen3_vl 的 patch（patch 父类 + Processor + VisionTransformer）
    2. 再显式 patch Qwen3.5 自身的类（确保不依赖 Python MRO 隐式继承）
    3. **不** patch ``get_mrope_input_positions`` —— Qwen3.5 使用扁平位置，
       裁剪后的占位符数量已能由 arange(len(input_tokens)) 正确覆盖。
    """
    if getattr(qwen3_5.Qwen3_5ForConditionalGeneration,
               "__pixelprune_patched__", False):
        return

    # Step 1: Patch 父类（Qwen3VLForConditionalGeneration、
    #          Qwen3VLMultiModalProcessor、Qwen3_VisionTransformer）
    _apply_qwen3vl_patches()

    # Step 2: 显式 patch Qwen3_5ForConditionalGeneration 上的图像处理方法。
    # 这些方法 Qwen3.5 未覆盖，通过 MRO 已指向 patched 版本；显式 patch 确保
    # 即使未来覆盖也能生效。
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "_parse_and_validate_image_input", _parse_image_input)
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "_process_image_input", _process_image_input)
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "get_encoder_cudagraph_config", _get_encoder_cudagraph_config)

    # Step 3: 同样 patch MoE 变体
    if hasattr(qwen3_5, "Qwen3_5MoeForConditionalGeneration"):
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "_parse_and_validate_image_input", _parse_image_input)
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "_process_image_input", _process_image_input)
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "get_encoder_cudagraph_config", _get_encoder_cudagraph_config)

    qwen3_5.Qwen3_5ForConditionalGeneration.__pixelprune_patched__ = True
    if logger:
        logger.info("Qwen3.5 PixelPrune (vLLM) patches applied successfully")


__all__ = ["apply_patches"]
