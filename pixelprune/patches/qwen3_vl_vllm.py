"""
vLLM Qwen3-VL monkey-patch (vLLM >= 0.29): 使用 PixelPrune 选择器计算
keep_indices，在 ViT 输入层裁剪 patch，并同步修正 mrope 位置与占位符数量。

适配 vLLM 0.29 的主要变化：
- ``Qwen3_VisionTransformer.forward`` 改为接收 ``encoder_metadata`` 字典
  （由 ``prepare_encoder_metadata`` 计算），不再内联计算 cu_seqlens 等。
- ``Qwen3VLMultiModalProcessor._call_hf_processor`` 被替换为
  ``_apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs)``，
  返回 BatchFeature。
- ``get_mrope_input_positions`` 委托给静态方法 ``_get_mrope_input_positions``，
  mm_features 为 ``MultiModalFeatureSpec`` 列表，``.data`` 为
  ``MultiModalKwargsItem``。
- 默认 ``cudagraph_mm_encoder=False``；为安全起见，启用 PixelPrune 时强制
  关闭 encoder CUDA graph（输出 token 数随 keep_indices 变化，无法固定 shape）。

环境变量：PIXELPRUNE_ENABLED, PIXELPRUNE_METHOD, PIXELPRUNE_METRIC,
PIXELPRUNE_THRESHOLD, PIXELPRUNE_VERBOSE。在创建 vLLM LLM 前调用 apply_patches()。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Mapping, Sequence

import numpy as np
import torch

from pixelprune.core import (
    compute_merged_keep_indices,
    merged_indices_to_patch_indices,
)

try:
    from vllm.model_executor.models import qwen3_vl
except ImportError:
    import importlib
    qwen3_vl = importlib.import_module("vllm.model_executor.models.qwen3_vl")

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:
    logger = None

from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement, PromptUpdate

try:
    from transformers.feature_extraction_utils import BatchFeature
except Exception:
    BatchFeature = None


# ---------------------------------------------------------------------------
# Pixel value normalization (aligned with qwen3_vl_hf.py)
# ---------------------------------------------------------------------------


def normalize_pixel_values_for_selector(pixel_values: torch.Tensor) -> torch.Tensor:
    """将 PixelValues 归一化到 [0,1] 以供 PatchSelector 使用。

    Qwen3-VL 的 pixel_values shape 为 (total_patches, C*T*H*W)，
    每个 patch 包含 T=2 帧，两帧内容相同（静态图像）。
    取第 0 帧并做逆标准化 x*0.5+0.5 还原到 [0,1]。
    """
    C, T, H, W = 3, 2, 16, 16
    pv_reshaped = pixel_values.view(-1, C, T, H, W)
    frame_0 = pv_reshaped[:, :, 0, :, :]
    normalized = (frame_0 * 0.5) + 0.5
    return normalized.reshape(pixel_values.shape[0], -1)


def _unwrap_data(x: Any) -> Any:
    """从 MultiModalFieldElem / 包装对象中取出原始张量。"""
    if x is None:
        return None
    return getattr(x, "data", x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_packed_by_indices(
    tensor: torch.Tensor,
    grid_thw: Any,
    keep_indices: List[torch.Tensor],
) -> torch.Tensor:
    """从 packed tensor 中按每张图的 keep_indices 选择 token。"""
    selected = []
    offset = 0
    grid_list = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    for seq_idx, (t, h, w) in enumerate(grid_list):
        t, h, w = int(t), int(h), int(w)
        seq_length = t * h * w
        indices = keep_indices[seq_idx].to(tensor.device)
        selected.append(tensor[offset : offset + seq_length][indices])
        offset += seq_length
    return torch.cat(selected)


# ======================= Patched Vision Transformer Forward =======================


def _vt_forward(
    self: Any,
    x: torch.Tensor,
    grid_thw: Any,
    *,
    keep_indices: List[torch.Tensor] | None = None,
    encoder_metadata: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Patched Qwen3_VisionTransformer.forward (vLLM 0.29)：

    - 复用 ``prepare_encoder_metadata`` 计算全量 pos_embed / rotary / cu_seqlens。
    - 若提供 keep_indices，则在 patch_embed 之后按 patch 级索引裁剪
      hidden_states / pos_embeds / rotary，并基于裁剪后的长度重算
      cu_seqlens / sequence_lengths / max_seqlen。
    """
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    if isinstance(grid_thw, list):
        grid_thw_list = grid_thw
    else:
        grid_thw_list = grid_thw.tolist()

    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    hidden_states = self.patch_embed(hidden_states)

    if encoder_metadata is None:
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

    pos_embeds = encoder_metadata["pos_embeds"]
    rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
    rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]

    has_prune = keep_indices is not None

    if has_prune:
        # PixelPrune: 按每张图的 patch 级 keep_indices 裁剪 ViT 输入。
        hidden_states = _select_packed_by_indices(
            hidden_states, grid_thw_list, keep_indices
        )
        pos_embeds = _select_packed_by_indices(
            pos_embeds, grid_thw_list, keep_indices
        )
        rotary_pos_emb_cos = _select_packed_by_indices(
            rotary_pos_emb_cos, grid_thw_list, keep_indices
        )
        rotary_pos_emb_sin = _select_packed_by_indices(
            rotary_pos_emb_sin, grid_thw_list, keep_indices
        )

        # 基于裁剪后的每图 patch 数重算 attention 元数据。
        lens = [int(len(idx)) for idx in keep_indices]
        cu_seqlens = np.array([0] + list(np.cumsum(lens)), dtype=np.int32)
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, self.device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            self.device,
            fp8_padded_hidden_size=getattr(self, "fp8_padded_hidden_size", None),
        )
    else:
        cu_seqlens = encoder_metadata["cu_seqlens"]
        sequence_lengths = encoder_metadata.get("sequence_lengths")
        max_seqlen = encoder_metadata["max_seqlen"]

    hidden_states = hidden_states + pos_embeds
    hidden_states = hidden_states.unsqueeze(1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    output = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
    return output


# ======================= Patched Model Functions =======================


def _parse_image_input(self: Any, **kwargs: object) -> Any:
    """Patched _parse_and_validate_image_input: 透传 keep_indices 字段。"""
    pixel_values = kwargs.pop("pixel_values", None)
    image_embeds = kwargs.pop("image_embeds", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    keep_indices = _unwrap_data(kwargs.pop("keep_indices", None))

    if pixel_values is None and image_embeds is None:
        return None

    if pixel_values is not None:
        result = {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if keep_indices is not None:
            result["keep_indices"] = keep_indices
        return result

    return {
        "type": "image_embeds",
        "image_embeds": image_embeds,
        "image_grid_thw": image_grid_thw,
    }


def _process_image_input(self: Any, image_input: Any) -> tuple:
    """Patched _process_image_input: 将 keep_indices 传给 VisionTransformer。"""
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2
    keep_indices = image_input.get("keep_indices")
    if keep_indices is not None:
        keep_indices = _unwrap_data(keep_indices)

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if keep_indices is not None and self.use_data_parallel:
            raise RuntimeError(
                "PixelPrune (keep_indices) 不支持 data parallel 模式"
            )
        if self.use_data_parallel:
            from vllm.model_executor.models.vision import (
                run_dp_sharded_mrope_vision_model,
            )
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        image_embeds = self.visual(
            pixel_values, grid_thw=grid_thw, keep_indices=keep_indices
        )

    merge_size = self.visual.spatial_merge_size
    merge_length = merge_size * merge_size
    if keep_indices is None:
        sizes = (grid_thw.prod(-1) // merge_length).tolist()
    else:
        sizes = [len(idx) // merge_length for idx in keep_indices]
    return image_embeds.split(sizes)


def _get_mrope_input_positions(
    self: Any,
    input_tokens: list,
    mm_features: list,
) -> tuple:
    """Patched get_mrope_input_positions: 支持 keep_indices 修正空间位置。

    vLLM 0.29 中 ``_iter_mm_grid_hw`` 对 image 仍返回全量
    ``actual_num_tokens = llm_grid_h * llm_grid_w``，但 PixelPrune 的占位符
    只有 ``num_kept`` 个 token，因此必须用 keep_indices 重算空间坐标并按
    裁剪后的数量推进 ``st``。
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    merge_length = spatial_merge_size * spatial_merge_size

    image_features = sorted(
        [f for f in mm_features if f.modality == "image"],
        key=lambda f: f.mm_position.offset,
    )
    mm_feature_map = {f.mm_position.offset: f for f in image_features}

    llm_pos_ids_list = []
    st = 0
    for (
        offset,
        llm_grid_h,
        llm_grid_w,
        actual_num_tokens,
    ) in qwen3_vl.Qwen3VLForConditionalGeneration._iter_mm_grid_hw(
        input_tokens,
        mm_features,
        video_token_id=self.config.video_token_id,
        vision_start_token_id=self.config.vision_start_token_id,
        vision_end_token_id=self.config.vision_end_token_id,
        spatial_merge_size=spatial_merge_size,
    ):
        # Skip frames with 0 tokens (EVS)
        if actual_num_tokens == 0:
            continue

        text_len = offset - st
        st_idx = (
            int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
        )

        # Text positions
        if text_len > 0:
            text_positions = np.broadcast_to(
                np.arange(text_len), (3, text_len)
            ) + st_idx
            llm_pos_ids_list.append(text_positions)
            st_idx += text_len

        # Check if this is an image with keep_indices
        mm_feature = mm_feature_map.get(offset)
        if mm_feature is not None:
            ki_data = _unwrap_data(
                mm_feature.data.get("keep_indices") if mm_feature.data else None
            )
            if ki_data is not None:
                # PixelPrune: use keep_indices to compute correct spatial positions
                if isinstance(ki_data, torch.Tensor):
                    keep_indices_np = ki_data.cpu().numpy()
                else:
                    keep_indices_np = np.asarray(ki_data)

                # keep_indices 是 patch 级；每个 merged token 对应
                # merge_length 个连续 patch，取每块第 0 个 patch // merge_length
                # 得到 merged 级索引。
                merged_indices = keep_indices_np[::merge_length] // merge_length
                num_kept_tokens = len(merged_indices)
                llm_w = llm_grid_w

                # 从 merged 索引恢复 (t, h, w) 空间坐标
                token_t = merged_indices // (llm_grid_h * llm_w)
                token_hw = merged_indices % (llm_grid_h * llm_w)
                token_h = token_hw // llm_w
                token_w = token_hw % llm_w

                frame_positions = np.stack(
                    [token_t + st_idx, token_h + st_idx, token_w + st_idx],
                    axis=0,
                )
                llm_pos_ids_list.append(frame_positions)
                st = offset + num_kept_tokens
                continue

        # Default: handle normally (including lumped video placeholders)
        expected_tokens_per_frame = llm_grid_h * llm_grid_w
        if actual_num_tokens > expected_tokens_per_frame:
            # Lumped placeholder
            num_logical_frames = actual_num_tokens // expected_tokens_per_frame
            remainder = actual_num_tokens % expected_tokens_per_frame
            for _ in range(num_logical_frames):
                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                st_idx = int(llm_pos_ids_list[-1].max() + 1)
                text_len = 0
            if remainder > 0:
                full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(full_grid[:, :remainder] + text_len + st_idx)
        else:
            grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
            llm_pos_ids_list.append(grid_indices + text_len + st_idx)

        st = offset + actual_num_tokens

    # Trailing text
    if st < len(input_tokens):
        st_idx = (
            int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
        )
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
        )

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    if llm_positions.shape[1] != len(input_tokens):
        raise ValueError(
            f"Position count mismatch: got {llm_positions.shape[1]}, "
            f"expected {len(input_tokens)}"
        )
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    return torch.from_numpy(llm_positions), mrope_position_delta


def _get_encoder_cudagraph_config(self: Any) -> Any:
    """Patched get_encoder_cudagraph_config: 启用 PixelPrune 时禁用 encoder
    CUDA graph（输出 token 数随 keep_indices 变化，无法匹配固定 budget）。"""
    cfg = _orig(
        qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config"
    )(self)
    if os.environ.get("PIXELPRUNE_ENABLED", "").lower() in ("1", "true", "yes"):
        try:
            cfg.modalities = []
        except Exception:
            pass
    return cfg


# ======================= Patched Processor =======================


def _mmp_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "__init__")(self, *args, **kwargs)
    self._pixelprune_enabled = (
        os.environ.get("PIXELPRUNE_ENABLED", "true").lower()
        in ("1", "true", "yes")
    )


def _mmp_apply_hf_processor_main(
    self: Any,
    mm_items: Any,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Any:
    """Patched _apply_hf_processor_main: 计算 keep_indices 并注入到输出。"""
    out = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "_apply_hf_processor_main")(
        self, mm_items, hf_processor_mm_kwargs
    )
    if self._pixelprune_enabled:
        pixel_values = out.get("pixel_values")
        image_grid_thw = out.get("image_grid_thw")
        if pixel_values is not None and image_grid_thw is not None:
            spatial_merge_size = (
                self.info.get_hf_config().vision_config.spatial_merge_size
            )
            block_size = spatial_merge_size ** 2

            # Normalize pixel values (aligned with qwen3_vl_hf.py)
            pixel_values_norm = normalize_pixel_values_for_selector(pixel_values)

            t0 = time.perf_counter()
            merged_keep_indices = compute_merged_keep_indices(
                pixel_values_norm,
                image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
            selector_ms = (time.perf_counter() - t0) * 1000

            keep_indices = merged_indices_to_patch_indices(
                merged_keep_indices, block_size, pixel_values.device
            )
            out["keep_indices"] = keep_indices

            _verbose_log(image_grid_thw, merged_keep_indices,
                         block_size, selector_ms)
    return out


def _verbose_log(
    image_grid_thw: torch.Tensor,
    merged_keep_indices: List[torch.Tensor],
    block_size: int,
    selector_ms: float,
) -> None:
    """PIXELPRUNE_VERBOSE=true 时打印每张图的 pruning 统计。"""
    VERBOSE = os.environ.get("PIXELPRUNE_VERBOSE", "false").lower() in (
        "true", "1", "yes",
    )
    if not VERBOSE:
        return

    grid_list = image_grid_thw.tolist()
    num_images = len(grid_list)
    total_org = 0
    total_new = 0

    for i, ((t, h, w), ki) in enumerate(zip(grid_list, merged_keep_indices)):
        t, h, w = int(t), int(h), int(w)
        org_patches = t * h * w
        new_patches = len(ki) * block_size
        org_merged = org_patches // block_size
        new_merged = len(ki)
        ratio = new_merged / org_merged if org_merged > 0 else 1.0
        total_org += org_merged
        total_new += new_merged
        sys.stdout.write(
            f"[PixelPrune][img {i}/{num_images}] "
            f"grid=({t},{h},{w}) "
            f"merged: {org_merged} -> {new_merged} "
            f"(retain {ratio:.1%})\n"
        )

    overall_ratio = total_new / total_org if total_org > 0 else 1.0
    sys.stdout.write(
        f"[PixelPrune][total] images={num_images}, "
        f"merged: {total_org} -> {total_new} (retain {overall_ratio:.1%}), "
        f"method={os.environ.get('PIXELPRUNE_METHOD', 'pred_2d')}, "
        f"metric={os.environ.get('PIXELPRUNE_METRIC', 'max')}, "
        f"threshold={os.environ.get('PIXELPRUNE_THRESHOLD', '0.0')}, "
        f"selector={selector_ms:.1f}ms\n"
    )
    sys.stdout.flush()


def _mmp_fields(
    self: Any, hf_inputs: Any, hf_processor_mm_kwargs: Mapping[str, object]
) -> Mapping[str, Any]:
    """Patched _get_mm_fields_config: 注册 keep_indices 字段。"""
    cfg = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "_get_mm_fields_config")(
        self, hf_inputs, hf_processor_mm_kwargs
    )
    cfg = dict(cfg)
    if "keep_indices" in hf_inputs:
        cfg["keep_indices"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    return cfg


def _mmp_prompt_updates(
    self: Any,
    mm_items: Any,
    hf_processor_mm_kwargs: Mapping[str, Any],
    out_mm_kwargs: Any,
) -> Sequence[PromptUpdate]:
    """Patched _get_prompt_updates: 根据 keep_indices 调整 image token 数量。"""
    ups = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "_get_prompt_updates")(
        self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
    )
    if not self._pixelprune_enabled:
        return ups

    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    merge_length = image_processor.merge_size ** 2

    def get_image_replacement_prune(item_idx: int) -> list:
        out_item = out_mm_kwargs["image"][item_idx]
        ki = _unwrap_data(out_item.get("keep_indices"))
        if ki is not None:
            num_tokens = len(ki) // merge_length
        else:
            grid_thw = _unwrap_data(out_item["image_grid_thw"])
            num_tokens = int(grid_thw.prod()) // merge_length
        return [hf_processor.image_token_id] * num_tokens

    new_ups = []
    for u in ups:
        if u.modality == "image":
            new_ups.append(PromptReplacement(
                modality="image", target=u.target,
                replacement=get_image_replacement_prune,
            ))
        else:
            new_ups.append(u)
    return new_ups


# ======================= Monkey-patch infrastructure =======================


def _orig(cls: type, name: str) -> Any:
    return getattr(cls, f"__pixelprune_orig_{name}__", None)


def _save_orig_once(cls: type, name: str) -> None:
    key = f"__pixelprune_orig_{name}__"
    if not hasattr(cls, key):
        setattr(cls, key, getattr(cls, name))


def _patch(cls: type, name: str, fn: Any) -> None:
    _save_orig_once(cls, name)
    setattr(cls, name, fn)


_PATCHES = [
    (qwen3_vl.Qwen3VLMultiModalProcessor, "__init__", _mmp_init),
    (qwen3_vl.Qwen3VLMultiModalProcessor, "_apply_hf_processor_main",
     _mmp_apply_hf_processor_main),
    (qwen3_vl.Qwen3VLMultiModalProcessor, "_get_mm_fields_config", _mmp_fields),
    (qwen3_vl.Qwen3VLMultiModalProcessor, "_get_prompt_updates",
     _mmp_prompt_updates),
    (qwen3_vl.Qwen3VLForConditionalGeneration,
     "_parse_and_validate_image_input", _parse_image_input),
    (qwen3_vl.Qwen3VLForConditionalGeneration, "_process_image_input",
     _process_image_input),
    (qwen3_vl.Qwen3VLForConditionalGeneration, "get_mrope_input_positions",
     _get_mrope_input_positions),
    (qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config",
     _get_encoder_cudagraph_config),
    (qwen3_vl.Qwen3_VisionTransformer, "forward", _vt_forward),
]


def apply_patches() -> None:
    """对 vLLM Qwen3-VL 应用 PixelPrune monkey-patch。"""
    if getattr(qwen3_vl.Qwen3VLMultiModalProcessor, "__pixelprune_patched__", False):
        return
    for cls, name, fn in _PATCHES:
        _patch(cls, name, fn)
    qwen3_vl.Qwen3VLMultiModalProcessor.__pixelprune_patched__ = True
    if logger:
        logger.info("Qwen3-VL PixelPrune (vLLM) patches applied successfully")
