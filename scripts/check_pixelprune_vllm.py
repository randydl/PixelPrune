"""验证 PixelPrune 的 vLLM 补丁是否真正生效。

补丁失效是**静默**的：服务照常启动、照常返回结果，只是没有剪枝。本脚本在启动
推理前做一次静态自检，确认每个补丁函数确实挂到了目标类上。

用法：
    PIXELPRUNE_ENABLED=true python3 scripts/check_pixelprune_vllm.py

退出码：0 = 全部生效，1 = 存在未生效的补丁。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PIXELPRUNE_ENABLED", "true")


def main() -> int:
    import vllm

    from pixelprune.patches import vllm_bootstrap
    from pixelprune.patches.qwen3_vl_vllm import _build_patches, _hf_proc_hook

    print(f"vllm = {vllm.__version__}")

    try:
        vllm_bootstrap.maybe_apply_patches()
    except Exception as e:
        print(f"\n补丁应用失败：\n{e}")
        return 1

    print(f"HF processor hook = {_hf_proc_hook()}\n")

    failed = []
    for cls, name, fn in _build_patches():
        actual = getattr(cls, name, None)
        if actual is fn:
            print(f"  PATCHED   {cls.__name__}.{name}")
        else:
            failed.append(f"{cls.__name__}.{name}")
            print(f"  ORIGINAL  {cls.__name__}.{name}")
            print(f"            期望 {fn.__name__}，实际 {getattr(actual, '__name__', actual)}")

    if failed:
        print(f"\n以下补丁未生效：{failed}")
        print("剪枝不会发生，请检查 vLLM 版本兼容性。")
        return 1

    print(f"\n全部 {len(_build_patches())} 个补丁均已生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
