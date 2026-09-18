from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("PIXELPRUNE_ENABLED", "").lower() in ("true", "1", "yes")


def _strict() -> bool:
    """补丁失败时是否直接报错终止进程。

    默认 True。静默降级会让服务正常启动并返回**未剪枝**的结果，而使用者无从察觉——
    在评估场景下这会产出被误标为 PixelPrune 的 baseline 数值，因此默认必须响亮失败。
    仅在明确知晓后果时才用 PIXELPRUNE_STRICT=false 换取"带病运行"。
    """
    return os.environ.get("PIXELPRUNE_STRICT", "true").lower() not in ("false", "0", "no")


def _warn(msg: str) -> None:
    try:
        from vllm.logger import init_logger
        init_logger(__name__).warning(msg)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(msg)


def maybe_apply_patches() -> None:
    """vLLM 调用的入口函数。未启用时立即返回，启用时才懒加载真正的 patch。

    启用状态下若补丁失败，默认抛错（见 _strict），避免"以为开了剪枝、其实没开"。
    """
    if not _enabled():
        return

    # Qwen3.5 的 apply_patches 会同时 patch 共享父类 Qwen3-VL；
    # 若 vllm 版本不含 qwen3_5 模型，则退回到只 patch Qwen3-VL。
    try:
        from .qwen3_5_vllm import apply_patches
    except ImportError:
        try:
            from .qwen3_vl_vllm import apply_patches
        except ImportError:
            _warn("PixelPrune: 当前 vLLM 不含 Qwen3-VL/Qwen3.5 模型，跳过")
            return

    try:
        apply_patches()
    except Exception as e:
        msg = (
            f"PixelPrune vLLM plugin failed: {e}\n"
            f"视觉 token 剪枝不会生效，推理结果等同于未剪枝。"
            f"若确认要在此状态下继续运行，设置 PIXELPRUNE_STRICT=false 可降级为警告。"
        )
        if _strict():
            raise RuntimeError(msg) from e
        _warn(msg)