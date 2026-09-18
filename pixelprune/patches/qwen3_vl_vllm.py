"""
vLLM Qwen3-VL monkey-patch：使用 PixelPrune 选择器计算 keep_indices，支持多种方法。

环境变量：PIXELPRUNE_ENABLED, PIXELPRUNE_METHOD, PIXELPRUNE_METRIC, PIXELPRUNE_THRESHOLD, PIXELPRUNE_VERBOSE
在创建 vLLM LLM 前调用 apply_patches()。
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
    from transformers import BatchFeature
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
    return getattr(x, "data", x)


def _pixelprune_enabled() -> bool:
    return os.environ.get("PIXELPRUNE_ENABLED", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_packed_by_indices(
    tensor: torch.Tensor,
    grid_thw: torch.Tensor | list,
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
    keep_indices: List[torch.Tensor] | None = None,
    *,
    encoder_metadata: dict | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Patched VisionTransformer.forward: 支持 keep_indices 在 ViT 输入层裁剪 patch。

    - 使用 MMEncoderAttention 计算 sequence_lengths / max_seqlen / cu_seqlens
    - block.forward 传 sequence_lengths 参数

    encoder_metadata 是 vLLM 新版（>= 0.29）引入的容器，用于在 eager 路径与
    encoder CUDA graph 之间复用 ViT 的 kernel 预计算量（pos_embeds /
    rotary_pos_emb_* / cu_seqlens / max_seqlen / sequence_lengths）。

    剪枝场景下必须忽略它：CUDA graph 变体会把 cu_seqlens padding 到
    max_frames_per_batch，而剪枝后序列长度已变，复用会留下幽灵零长度序列。
    且走 CUDA graph 时 keep_indices 本就不会传到这里（那条路径绕过
    _process_image_input），因此恒按 None 处理。
    """
    # 无剪枝：完全交回原始实现，保留 encoder_metadata / CUDA graph 语义。
    # 注意 encoder_metadata 仅存在于新版 vLLM，故仅在非 None 时显式传参，
    # 保证旧版 schema 不会收到未知关键字参数。
    if keep_indices is None:
        orig = _orig(qwen3_vl.Qwen3_VisionTransformer, "forward")
        if encoder_metadata is not None:
            return orig(self, x, grid_thw, encoder_metadata=encoder_metadata)
        return orig(self, x, grid_thw, **kwargs)

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    if isinstance(grid_thw, list):
        grid_thw_list = grid_thw
        grid_thw_np = np.array(grid_thw, dtype=np.int32)
    else:
        grid_thw_list = grid_thw.tolist()
        grid_thw_np = grid_thw.numpy() if hasattr(grid_thw, "numpy") else np.array(grid_thw.cpu())

    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    hidden_states = self.patch_embed(hidden_states)
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
    rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

    has_prune = keep_indices is not None

    if has_prune:
        # Select patches according to keep_indices
        hidden_states = _select_packed_by_indices(hidden_states, grid_thw_list, keep_indices)
        pos_embeds_sel = _select_packed_by_indices(pos_embeds, grid_thw_list, keep_indices)
        rotary_pos_emb_cos = _select_packed_by_indices(rotary_pos_emb_cos, grid_thw_list, keep_indices)
        rotary_pos_emb_sin = _select_packed_by_indices(rotary_pos_emb_sin, grid_thw_list, keep_indices)
        hidden_states = hidden_states + pos_embeds_sel

        lens = [len(idx) for idx in keep_indices]
        cu_seqlens = np.array([0] + list(np.cumsum(lens)), dtype=np.int32)
    else:
        hidden_states = hidden_states + pos_embeds
        cu_seqlens = np.repeat(
            grid_thw_np[:, 1] * grid_thw_np[:, 2], grid_thw_np[:, 0]
        ).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

    sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
        self.attn_backend, cu_seqlens, self.device
    )
    max_seqlen = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
        dtype=torch.int32,
    )
    # FP8 ViT attention 下 cu_seqlens 需按 padding 后的 hidden size 缩放。
    # 该参数仅存在于新版 vLLM；旧版没有，故条件传参以保持兼容。
    _fp8_kwargs = {}
    if getattr(self, "fp8_padded_hidden_size", None) is not None:
        _fp8_kwargs["fp8_padded_hidden_size"] = self.fp8_padded_hidden_size
    cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
        self.attn_backend,
        cu_seqlens,
        self.hidden_size,
        self.tp_size,
        self.device,
        **_fp8_kwargs,
    )

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
            deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](hidden_states)
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
        _assert_no_evs_conflict(self)

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if keep_indices is not None and self.use_data_parallel:
            raise RuntimeError(
                "PixelPrune (keep_indices) 不支持 data parallel 模式"
            )
        if self.use_data_parallel:
            from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
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


def _assert_no_evs_conflict(self: Any) -> None:
    """PixelPrune 与 EVS 视频剪枝不兼容，同时启用时立刻报错。

    EVS 开启后 `embed_multimodal` 会无条件调用 `_postprocess_image_embeds_evs`，
    该方法按**完整** grid 切分 image embedding 并拼接 mrope 位置通道；而 PixelPrune
    已把 embedding 序列缩短，两者长度对不上会直接崩。与其在深层报一个难以定位的
    形状错误，不如在这里给出可操作的提示。
    """
    if getattr(self, "is_multimodal_pruning_enabled", False):
        raise RuntimeError(
            "PixelPrune 与 EVS 视频剪枝不兼容：EVS 的 _postprocess_image_embeds_evs "
            "会按完整 grid 处理已被 PixelPrune 缩短的 image embedding 序列。"
            "请移除 --mm-processor-kwargs 中的 video_pruning_rate 后重试。"
        )


def _encoder_cudagraph_config(self: Any) -> Any:
    """PixelPrune 启用时，把 image 模态从 encoder CUDA graph 中排除。

    该路径由 encoder_cudagraph_manager 直接驱动 ViT，完全绕过
    `_process_image_input`，因此 keep_indices 会被忽略，与本补丁在
    `_get_prompt_updates` 中算出的 token 数不一致。

    vLLM 自身对 EVS 视频剪枝已做了同样处理（上游 `is_multimodal_pruning_enabled`
    分支），这里是对图像剪枝的对应扩展。视频模态保持不变，避免连带损失性能。
    """
    cfg = _orig(
        qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config"
    )(self)
    if _pixelprune_enabled():
        cfg.modalities = [m for m in cfg.modalities if m != "image"]
        if not cfg.modalities:
            cfg.max_frames_per_video = 1
    return cfg


def _get_mrope_input_positions(
    self: Any,
    input_tokens: list,
    mm_features: list,
) -> tuple:
    """Patched get_mrope_input_positions: 支持 keep_indices 修正空间位置。

    - 使用 _iter_mm_grid_hw 静态方法 (4 参数, 返回 4 元组)
    - 当有 keep_indices 时，用实际空间坐标替代默认的 grid_indices
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    merge_length = spatial_merge_size * spatial_merge_size

    image_features = sorted(
        [f for f in mm_features if f.modality == "image"],
        key=lambda f: f.mm_position.offset,
    )
    mm_feature_map = {f.mm_position.offset: f for f in image_features}

    # Delegate to original static method for iteration
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
        st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0

        # Text positions
        if text_len > 0:
            text_positions = np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            llm_pos_ids_list.append(text_positions)
            st_idx += text_len

        # Check if this is an image with keep_indices
        mm_feature = mm_feature_map.get(offset)
        if mm_feature is not None:
            ki_data = _unwrap_data(mm_feature.data.get("keep_indices"))
            if ki_data is not None:
                # PixelPrune: use keep_indices to compute correct spatial positions
                keep_indices_tensor = ki_data
                if isinstance(keep_indices_tensor, torch.Tensor):
                    keep_indices_np = keep_indices_tensor.cpu().numpy()
                else:
                    keep_indices_np = np.array(keep_indices_tensor)

                # Convert patch-level keep_indices to merged-level token indices
                # keep_indices are patch-level (block_size consecutive per merged token)
                merged_indices = keep_indices_np[::merge_length] // merge_length
                num_kept_tokens = len(merged_indices)
                llm_w = llm_grid_w

                # Recover spatial (t, h, w) coordinates from merged indices
                token_t = merged_indices // (llm_grid_h * llm_w)
                token_hw = merged_indices % (llm_grid_h * llm_w)
                token_h = token_hw // llm_w
                token_w = token_hw % llm_w

                frame_positions = np.stack([
                    token_t + st_idx,
                    token_h + st_idx,
                    token_w + st_idx,
                ], axis=0)
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
        st_idx = int(llm_pos_ids_list[-1].max() + 1) if llm_pos_ids_list else 0
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


# ======================= vLLM 版本适配 =======================


# HF processor hook 的名称随 vLLM 版本变化：
#   旧版（<= 0.18 系列）：_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs)
#   新版（>= 0.29）：      _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs)
# 两者都返回 transformers.BatchFeature，且该返回值会被原样交给 _get_mm_fields_config，
# 因此在此基础上注入 keep_indices 的机制在新旧版完全一致。
_HF_PROC_HOOKS = ("_apply_hf_processor_main", "_call_hf_processor")
_HF_PROC_HOOK: str | None = None


def _hf_proc_hook() -> str:
    """返回当前 vLLM 版本可用的 HF processor hook 名称（结果缓存）。"""
    global _HF_PROC_HOOK
    if _HF_PROC_HOOK is None:
        for name in _HF_PROC_HOOKS:
            if hasattr(qwen3_vl.Qwen3VLMultiModalProcessor, name):
                _HF_PROC_HOOK = name
                break
        else:
            from vllm import __version__ as _vllm_version
            raise RuntimeError(
                f"PixelPrune: Qwen3VLMultiModalProcessor 上找不到任何可用的 HF "
                f"processor hook（已尝试 {list(_HF_PROC_HOOKS)}）；"
                f"当前 vllm={_vllm_version}，该版本可能不受支持。"
            )
    return _HF_PROC_HOOK


# ======================= Patched Processor =======================


def _mmp_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "__init__")(self, *args, **kwargs)
    self._pixelprune_enabled = _pixelprune_enabled()


def _mmp_apply_hf(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Patched HF processor hook：计算 keep_indices 并注入到输出。

    用 *args/**kwargs 透传以兼容新旧两种 hook 签名（见 _hf_proc_hook）。
    """
    hook = _hf_proc_hook()
    out = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, hook)(self, *args, **kwargs)
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
    self: Any, hf_inputs: Any, hf_mm_kwargs: Mapping[str, object]
) -> Mapping[str, Any]:
    """Patched _get_mm_fields_config: 注册 keep_indices 字段。"""
    cfg = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, "_get_mm_fields_config")(
        self, hf_inputs, hf_mm_kwargs
    )
    cfg = dict(cfg)
    if "keep_indices" in hf_inputs:
        # keep_on_cpu=True 是必须的：vLLM 在前缀缓存命中时会用
        # strip_covered_mm_data 剥离非 keep_on_cpu 字段。若 keep_indices 被剥掉，
        # _get_mrope_input_positions 会回退到按完整 grid 计算位置，导致
        # "Position count mismatch" 并使 EngineCore 崩溃(多图+前缀缓存命中时复现)。
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
                modality="image", target=u.target, replacement=get_image_replacement_prune
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
        if not hasattr(cls, name):
            from vllm import __version__ as _vllm_version
            raise AttributeError(
                f"PixelPrune: {cls.__name__} 上不存在 {name!r}，无法打补丁。"
                f"当前 vllm={_vllm_version}，该 API 可能已在新版本中改名或移除。"
            )
        setattr(cls, key, getattr(cls, name))


def _patch(cls: type, name: str, fn: Any) -> None:
    _save_orig_once(cls, name)
    setattr(cls, name, fn)


def _build_patches() -> List[tuple]:
    """构造补丁清单。

    延迟到调用时才解析 HF processor hook 名称，避免 import 期即因版本不兼容而失败。
    """
    patches = [
        (qwen3_vl.Qwen3VLMultiModalProcessor, "__init__", _mmp_init),
        (qwen3_vl.Qwen3VLMultiModalProcessor, _hf_proc_hook(), _mmp_apply_hf),
        (qwen3_vl.Qwen3VLMultiModalProcessor, "_get_mm_fields_config", _mmp_fields),
        (qwen3_vl.Qwen3VLMultiModalProcessor, "_get_prompt_updates", _mmp_prompt_updates),
        (qwen3_vl.Qwen3VLForConditionalGeneration, "_parse_and_validate_image_input", _parse_image_input),
        (qwen3_vl.Qwen3VLForConditionalGeneration, "_process_image_input", _process_image_input),
        (qwen3_vl.Qwen3VLForConditionalGeneration, "get_mrope_input_positions", _get_mrope_input_positions),
        (qwen3_vl.Qwen3_VisionTransformer, "forward", _vt_forward),
    ]
    # encoder CUDA graph 是较新版本才有的配置项，旧版没有；缺失时跳过即可，
    # 不影响其余补丁（这些是必需的，缺失会由 _save_orig_once 报错）。
    if hasattr(qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config"):
        patches.append((
            qwen3_vl.Qwen3VLForConditionalGeneration,
            "get_encoder_cudagraph_config",
            _encoder_cudagraph_config,
        ))
    return patches


def _verify_patches(patches: List[tuple]) -> None:
    """自检：确认每个补丁函数确实已挂到目标类上。

    这是防止"静默失效"的关键一环 —— 若补丁没打上，服务会照常启动并返回
    未剪枝的结果，使用者无从察觉。
    """
    missing = [
        f"{cls.__name__}.{name}"
        for cls, name, fn in patches
        if getattr(cls, name, None) is not fn
    ]
    if missing:
        raise RuntimeError(
            f"PixelPrune: 以下补丁未能生效: {missing}。"
            f"剪枝不会发生，请检查 vLLM 版本兼容性。"
        )


def apply_patches() -> None:
    """对 vLLM Qwen3-VL 应用 PixelPrune monkey-patch。"""
    if getattr(qwen3_vl.Qwen3VLMultiModalProcessor, "__pixelprune_patched__", False):
        return
    patches = _build_patches()
    for cls, name, fn in patches:
        _patch(cls, name, fn)
    _verify_patches(patches)
    qwen3_vl.Qwen3VLMultiModalProcessor.__pixelprune_patched__ = True
    if logger:
        logger.info(
            "Qwen3-VL PixelPrune (vLLM) patches applied successfully (hf_proc_hook=%s)",
            _hf_proc_hook(),
        )