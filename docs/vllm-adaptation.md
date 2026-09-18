# PixelPrune 适配新版 vLLM（≥ 0.29）原理详解

> 对应 commit：`适配新版本vllm`（`147c35e`）
> 涉及文件：`pixelprune/patches/qwen3_vl_vllm.py`、`pixelprune/patches/qwen3_5_vllm.py`、`pixelprune/patches/vllm_bootstrap.py`、`docker/Dockerfile.vllm`、`scripts/vllm_entrypoint.sh`、`scripts/check_pixelprune_vllm.py`

本文档说明 PixelPrune 在从 vLLM `0.18.x` 升级到 `0.29.x` 时遇到的所有 API 变化、对应的兼容策略，以及背后的设计动机。阅读前建议先了解 PixelPrune 的基本工作原理（在 ViT 输入层按 `keep_indices` 裁剪 patch）。

---

## 0. 背景：PixelPrune 如何介入 vLLM

PixelPrune 对 vLLM 的适配采用 **monkey-patch**（运行时方法替换）而非 fork 源码。它通过 `setup.py` 声明的 entry point 自动加载：

```python
entry_points={
    "vllm.general_plugins": [
        "pixelprune = pixelprune.patches.vllm_bootstrap:maybe_apply_patches",
    ],
}
```

vLLM 启动时会枚举所有已安装发行版的 `vllm.general_plugins` entry point 并调用它们。`maybe_apply_patches()` 在启用时（`PIXELPRUNE_ENABLED=true`）懒加载真正的 patch 模块，把 Qwen3-VL / Qwen3.5 的若干关键方法替换成 PixelPrune 版本。

被 patch 的方法横跨三个层次：

| 层次 | 类 | 方法 | 作用 |
|------|------|------|------|
| Processor | `Qwen3VLMultiModalProcessor` | HF processor hook | 调 HF processor，注入 `keep_indices` |
| Processor | `Qwen3VLMultiModalProcessor` | `_get_mm_fields_config` | 注册 `keep_indices` 为多模态字段 |
| Processor | `Qwen3VLMultiModalProcessor` | `_get_prompt_updates` | 按 `keep_indices` 调整 image token 数 |
| Model | `Qwen3VLForConditionalGeneration` | `_parse_and_validate_image_input` | 透传 `keep_indices` |
| Model | `Qwen3VLForConditionalGeneration` | `_process_image_input` | 把 `keep_indices` 传给 ViT |
| Model | `Qwen3VLForConditionalGeneration` | `get_mrope_input_positions` | 按裁剪后坐标修正 mrope 位置 |
| Model | `Qwen3VLForConditionalGeneration` | `get_encoder_cudagraph_config` *(新)* | 把 image 排除出 encoder CUDA graph |
| ViT | `Qwen3_VisionTransformer` | `forward` | 在输入层裁剪 patch |

升级到 vLLM 0.29 后，这些方法中有多个签名变化、新增参数或改名，下面逐项展开。

---

## 1. HF processor hook 改名（最关键的 API 变化）

### 1.1 变化内容

PixelPrune 注入 `keep_indices` 的核心切入点是 **HF processor 调用之后** 的钩子。vLLM 在这个版本里把该方法重命名并改了签名：

| 版本 | 方法名 | 签名 |
|------|--------|------|
| ≤ 0.18.x | `_call_hf_processor` | `(self, prompt, mm_data, mm_kwargs, tok_kwargs)` |
| ≥ 0.29 | `_apply_hf_processor_main` | `(self, mm_items, hf_processor_mm_kwargs)` |

两个版本的返回值都是 `transformers.BatchFeature`，且会被原样交给 `_get_mm_fields_config`。因此"在返回值里塞 `keep_indices`"的注入机制本身在新旧版完全一致，**唯一需要适配的就是方法名和调用签名**。

### 1.2 兼容策略：运行时探测 + 缓存 + 透传

```python
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
            raise RuntimeError(...)  # 两个都找不到 → 明确报错
    return _HF_PROC_HOOK
```

**设计要点：**

1. **按优先级探测**：列表里新版名在前，优先命中新版；旧版环境自动回退到旧名。
2. **结果缓存**：`_HF_PROC_HOOK` 第一次解析后缓存，避免每次 patch 都 `hasattr`。
3. **找不到就报错**：两个 hook 都不存在时，明确抛出带 vLLM 版本号的 `RuntimeError`，而不是静默跳过。

打补丁的函数从固定签名的 `_mmp_call_hf(self, prompt, mm_data, mm_kwargs, tok_kwargs)` 改成用 `*args/**kwargs` 透传的 `_mmp_apply_hf`：

```python
def _mmp_apply_hf(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Patched HF processor hook：计算 keep_indices 并注入到输出。"""
    hook = _hf_proc_hook()
    out = _orig(qwen3_vl.Qwen3VLMultiModalProcessor, hook)(self, *args, **kwargs)
    if self._pixelprune_enabled:
        pixel_values = out.get("pixel_values")
        image_grid_thw = out.get("image_grid_thw")
        if pixel_values is not None and image_grid_thw is not None:
            ...  # 计算 keep_indices 并注入 out
    return out
```

`*args/**kwargs` 透传让同一个函数能同时适配新旧两种调用签名——调用方（vLLM 内部）按哪种签名传参，都原样转给原始实现。

### 1.3 延迟构造补丁清单

补丁清单从模块级常量 `_PATCHES` 改成了函数 `_build_patches()`，**在 `apply_patches()` 调用时才解析 hook 名**：

```python
def _build_patches() -> List[tuple]:
    patches = [
        ...
        (qwen3_vl.Qwen3VLMultiModalProcessor, _hf_proc_hook(), _mmp_apply_hf),
        ...
    ]
```

为什么不能在 import 期就构造？因为 `_hf_proc_hook()` 内部可能 `raise RuntimeError`，若在模块顶层执行，会导致 **整个插件 import 失败**，连"未启用剪枝"的正常 vLLM 启动都会被波及。延迟到 `apply_patches()`（仅启用时调用）后，错误只发生在真正需要剪枝的场景。

---

## 2. `encoder_metadata` 与 encoder CUDA graph

### 2.1 新机制概述

vLLM 0.29 在视觉模型中引入了 **encoder CUDA graph**：把 ViT 的前向也做 CUDA graph capture，以减少 eager 路径的 kernel launch 开销。配套引入了 `encoder_metadata` 容器，用于在 **eager 路径** 与 **encoder CUDA graph 路径** 之间复用 ViT 的预计算量：

- `pos_embeds`
- `rotary_pos_emb_cos / rotary_pos_emb_sin`
- `cu_seqlens`
- `max_seqlen`
- `sequence_lengths`

`VisionTransformer.forward` 的新签名因此多了一个关键字参数：

```python
def forward(self, x, grid_thw, *, encoder_metadata=None, **kwargs):
```

### 2.2 为什么剪枝必须忽略 `encoder_metadata`

这是本次适配最微妙的一点。原因有两层：

1. **CUDA graph 路径会 padding `cu_seqlens`**：encoder CUDA graph 要求固定形状，因此 `cu_seqlens` 会被 padding 到 `max_frames_per_batch`。而剪枝后序列长度已经变短，直接复用 padding 后的 `cu_seqlens` 会留下"幽灵零长度序列"，导致 attention 计算错乱。
2. **CUDA graph 路径根本不会传 `keep_indices`**：encoder CUDA graph 由 `encoder_cudagraph_manager` 直接驱动 ViT，**完全绕过** `_process_image_input`，因此 `keep_indices` 在这条路径上根本拿不到。也就是说，即便我们想在这条路径里剪枝，也做不到。

结论：**剪枝路径下 `encoder_metadata` 恒按 `None` 处理**，自己重新计算 `cu_seqlens` / `max_seqlen` / `sequence_lengths`。

### 2.3 代码实现

`_vt_forward` 的新签名和分流逻辑：

```python
def _vt_forward(
    self, x, grid_thw,
    keep_indices=None,
    *,
    encoder_metadata=None,   # 新版才有
    **kwargs,
):
    # 无剪枝：交回原始实现，按需透传 encoder_metadata（旧版没有该参数）
    if keep_indices is None:
        orig = _orig(qwen3_vl.Qwen3_VisionTransformer, "forward")
        if encoder_metadata is not None:
            return orig(self, x, grid_thw, encoder_metadata=encoder_metadata)
        return orig(self, x, grid_thw, **kwargs)

    # 剪枝：自己计算 cu_seqlens / max_seqlen / sequence_lengths，忽略 encoder_metadata
    ...
```

注意无剪枝分支里的条件透传：`encoder_metadata` 仅在新版存在，旧版 schema 不认识这个关键字参数。因此用 `if encoder_metadata is not None` 守卫，保证旧版 vLLM 不会收到未知 kwargs。

### 2.4 把 image 模态排除出 encoder CUDA graph

光在 `forward` 里忽略 `encoder_metadata` 还不够——因为 encoder CUDA graph 路径会绕过 `_process_image_input`，导致 `keep_indices` 被忽略，进而 ViT 输出长度与 `_get_prompt_updates` 里算出的 token 数对不上。为此新增了对 `get_encoder_cudagraph_config` 的 patch：

```python
def _encoder_cudagraph_config(self: Any) -> Any:
    """PixelPrune 启用时把 image 模态从 encoder CUDA graph 中排除。"""
    cfg = _orig(
        qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config"
    )(self)
    if _pixelprune_enabled():
        cfg.modalities = [m for m in cfg.modalities if m != "image"]
        if not cfg.modalities:
            cfg.max_frames_per_video = 1
    return cfg
```

**设计要点：**

- 只排除 `image`，**不动 `video`**。video 走 EVS（vLLM 自带的视频剪枝）有自己的处理，连带排除会损失性能。
- vLLM 上游对 EVS 视频剪枝已做了同样处理（`is_multimodal_pruning_enabled` 分支），这里是对 **图像剪枝** 的对应扩展。
- 该方法是较新版本才有的，旧版没有，因此用 `hasattr` 守卫，缺失时跳过补丁（见 §5.1）。

---

## 3. FP8 ViT attention 的新参数

### 3.1 变化内容

vLLM 0.29 给 `MMEncoderAttention.maybe_recompute_cu_seqlens` 增加了 `fp8_padded_hidden_size` 参数。FP8 ViT attention 下，`cu_seqlens` 需要按 **padding 后的 hidden size** 进行缩放，否则 FP8 kernel 的序列边界会算错。

### 3.2 兼容策略：条件传参

```python
_fp8_kwargs = {}
if getattr(self, "fp8_padded_hidden_size", None) is not None:
    _fp8_kwargs["fp8_padded_hidden_size"] = self.fp8_padded_hidden_size
cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
    self.attn_backend, cu_seqlens, self.hidden_size, self.tp_size, self.device,
    **_fp8_kwargs,
)
```

`fp8_padded_hidden_size` 是新版才有的实例属性，旧版没有。用 `getattr(self, ..., None)` 探测，仅在存在时才传该关键字参数，保证旧版不会收到未知 kwargs。这是贯穿整个适配过程的通用模式——**对"仅新版有的参数/方法"，一律用 `getattr`/`hasattr` 守卫 + 条件传参**。

---

## 4. `MultiModalFieldConfig.batched` 新增 `keep_on_cpu`

`_mmp_fields` 里给 `keep_indices` 的字段配置从：

```python
cfg["keep_indices"] = MultiModalFieldConfig.batched("image")
```

改为：

```python
cfg["keep_indices"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
```

新版 vLLM 的 `MultiModalFieldConfig.batched` 支持 `keep_on_cpu` 参数。`keep_indices` 是一组长度不等的 `torch.Tensor` 列表，需要在不同设备间随多模态输入一起搬运。设置 `keep_on_cpu=True` 让它在调度阶段保持在 CPU，由模型侧按需搬到 GPU，避免在多卡 broadcast 阶段就一次性占满显存或触发不必要的 D2D 拷贝。该参数旧版不识别但传了也不会报错（`batched` 用 `**kwargs` 接收），因此无需条件守卫。

---

## 5. monkey-patch 基础设施的健壮化

### 5.1 `_save_orig_once`：属性不存在即报错

旧实现里，若目标方法在新版本中改名/移除，`_save_orig_once` 会静默不存原函数，后续 `_orig()` 返回 `None`，patch 看似挂上实则调用即崩。新实现增加了存在性检查：

```python
def _save_orig_once(cls: type, name: str) -> None:
    key = f"__pixelprune_orig_{name}__"
    if not hasattr(cls, key):
        if not hasattr(cls, name):
            from vllm import __version__ as _vllm_version
            raise AttributeError(
                f"PixelPrune: {cls.__name__} 上不存在 {name!r}，无法打补丁 "
                f"(vllm={_vllm_version}，该 API 可能已在新版本中改名或移除)。"
            )
        setattr(cls, key, getattr(cls, name))
```

**关键改进**：把"找不到目标方法"从"静默继续"变成"立即报错并带上 vLLM 版本号"。这是防止静默失效的第一道防线。

### 5.2 `_build_patches`：按可用性动态构造

补丁清单改为延迟构造，并对"仅新版有的方法"用 `hasattr` 守卫：

```python
def _build_patches() -> List[tuple]:
    patches = [
        # 必需补丁：缺失会由 _save_orig_once 报错
        (qwen3_vl.Qwen3VLMultiModalProcessor, "__init__", _mmp_init),
        (qwen3_vl.Qwen3VLMultiModalProcessor, _hf_proc_hook(), _mmp_apply_hf),
        ...
        (qwen3_vl.Qwen3_VisionTransformer, "forward", _vt_forward),
    ]
    # 可选补丁：encoder CUDA graph 仅较新版本才有，缺失时跳过
    if hasattr(qwen3_vl.Qwen3VLForConditionalGeneration, "get_encoder_cudagraph_config"):
        patches.append((
            qwen3_vl.Qwen3VLForConditionalGeneration,
            "get_encoder_cudagraph_config",
            _encoder_cudagraph_config,
        ))
    return patches
```

区分两类补丁：
- **必需补丁**（processor hook、`forward` 等）：缺失 → `_save_orig_once` 报错。
- **可选补丁**（`get_encoder_cudagraph_config`）：缺失 → 跳过，不影响其余功能。

### 5.3 `_verify_patches`：apply 后自检

即便 `_save_orig_once` 不报错，patch 也可能因为类继承/MRO 等原因没真正挂上。新增 `_verify_patches` 在 apply 之后逐项核对：

```python
def _verify_patches(patches: List[tuple]) -> None:
    """自检：确认每个补丁函数已挂到目标类上，防止静默失效。"""
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
```

用 `is` 而非 `==` 比较函数身份，确保挂上去的就是 PixelPrune 的函数对象。

### 5.4 `apply_patches` 收尾日志带 hook 名

```python
logger.info(
    "Qwen3-VL PixelPrune (vLLM) patches applied successfully (hf_proc_hook=%s)",
    _hf_proc_hook(),
)
```

日志里带出实际命中的 hook 名，便于在多版本环境排查。

---

## 6. `vllm_bootstrap`：从"静默降级"到"严格失败"

### 6.1 为什么要改

旧实现里补丁失败只打一条 warning，服务照常启动。这会带来一个**极危险的静默失败**：用户以为开了剪枝，实际没开，服务正常返回 **未剪枝** 的结果。在评估场景下，这会产出被误标为 "PixelPrune" 的 baseline 数值，污染整个实验结论，且难以察觉。

### 6.2 新行为

引入 `PIXELPRUNE_STRICT`（默认 `true`）：

```python
def _strict() -> bool:
    return os.environ.get("PIXELPRUNE_STRICT", "true").lower() not in ("false", "0", "no")
```

`maybe_apply_patches` 捕获到异常时：

```python
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
```

默认直接 `raise` 终止进程，让用户立刻发现问题。仅当显式设置 `PIXELPRUNE_STRICT=false` 时才降级为 warning——这是给"明确知晓后果、只需要带病运行调试"的场景留的逃生口。

### 6.3 三层防线汇总

至此，防止静默失效形成三层防线：

1. **`_save_orig_once`**：目标方法不存在 → import/patch 期报错。
2. **`_verify_patches`**：补丁没挂上 → apply 后立即报错。
3. **`PIXELPRUNE_STRICT`**：apply 整体抛异常 → 启动期终止进程（默认开启）。

---

## 7. 配套自检脚本 `scripts/check_pixelprune_vllm.py`

为了让用户在 **启动推理之前** 就能确认补丁生效，新增了静态自检脚本：

```python
def main() -> int:
    import vllm
    from pixelprune.patches import vllm_bootstrap
    from pixelprune.patches.qwen3_vl_vllm import _build_patches, _hf_proc_hook

    print(f"vllm = {vllm.__version__}")
    vllm_bootstrap.maybe_apply_patches()   # 失败会抛错
    print(f"HF processor hook = {_hf_proc_hook()}\n")

    for cls, name, fn in _build_patches():
        actual = getattr(cls, name, None)
        if actual is fn:
            print(f"  PATCHED   {cls.__name__}.{name}")
        else:
            ...  # 报告未生效
    return 0 if not failed else 1
```

用法：

```bash
PIXELPRUNE_ENABLED=true python3 scripts/check_pixelprune_vllm.py
```

退出码 `0` = 全部生效，`1` = 存在未生效补丁。可在 CI 或部署脚本里作为前置 gate。

---

## 8. 部署形态适配：entry point 与 Docker

### 8.1 entry point 必须来自已安装发行版

vLLM 通过 `vllm.general_plugins` entry point 发现插件，而 entry point 的元数据由 **已安装的发行版** 提供。**仅挂载仓库 + 设置 `PYTHONPATH` 不够**——vLLM 启动时不会发现插件，补丁不会被应用。

这带来了两种部署形态：

#### 形态 A：派生镜像（固化部署）

`docker/Dockerfile.vllm`：

```dockerfile
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0-cu129
FROM ${VLLM_IMAGE}

COPY setup.py /opt/pixelprune/setup.py
COPY pixelprune /opt/pixelprune/pixelprune

RUN pip install /opt/pixelprune --no-deps -q

ENV PIXELPRUNE_ENABLED=false
```

- 基于 `vllm/vllm-openai:v0.29.0-cu129`。
- `pip install --no-deps`：pixelprune 仅依赖 numpy，vLLM 镜像中已具备，避免引入多余依赖。
- 默认 `PIXELPRUNE_ENABLED=false`，显式设为 false 以便 `docker inspect` 时可见；运行时通过 `-e PIXELPRUNE_ENABLED=true` 启用。

#### 形态 B：挂载仓库 + 可编辑安装（开发迭代）

`scripts/vllm_entrypoint.sh`：

```bash
PP_SRC="${PIXELPRUNE_SRC:-/pixelprune}"

# 幂等：已安装则跳过
if ! python3 -c "import pixelprune" 2>/dev/null; then
    pip install -e "$PP_SRC" --no-deps -q
fi

exec vllm serve "$@"
```

- 挂载仓库后用 `pip install -e` 做可编辑安装，让 entry point 元数据写入 site-packages，同时代码改动即时生效。
- 幂等检查避免每次启动重复构建。
- 所有参数原样透传给 `vllm serve`。

### 8.2 形态选择

| 场景 | 形态 |
|------|------|
| 生产固化部署、版本可追溯 | A（派生镜像） |
| 改代码即时生效、调试迭代 | B（挂载 + `pip install -e`） |

---

## 9. Qwen3.5 的继承式 patch

`qwen3_5_vllm.py` 适配了 Qwen3.5。Qwen3.5 在 vLLM 中继承自 Qwen3-VL：

```
Qwen3_5ForConditionalGeneration       → Qwen3VLForConditionalGeneration
Qwen3_5MoeForConditionalGeneration    → Qwen3_5ForConditionalGeneration
```

两者共享同一个 `Qwen3VLMultiModalProcessor` 和 `Qwen3_VisionTransformer`。因此 Qwen3.5 的 `apply_patches` 策略是：

1. **先 apply Qwen3-VL 的 patch**：patch 父类 + Processor + VisionTransformer（共享组件）。
2. **再显式 patch Qwen3_5 自身方法**：虽然通过 MRO 这些方法已指向 patched 版本，但显式 patch 确保：
   - 即使 Qwen3.5 未来覆盖这些方法，patch 仍然生效；
   - 代码意图清晰，不依赖隐式继承。
3. **同样 patch MoE 变体**（`Qwen3_5MoeForConditionalGeneration`，若存在）。

本次适配中，`qwen3_5_vllm.py` 唯一的改动是把对 `_mmp_call_hf` 的引用改名成 `_mmp_apply_hf`，以匹配 §1 的 hook 改名。

---

## 10. EVS 视频剪枝互斥检测

vLLM 自带 EVS（Early Visual Streaming）视频剪枝，通过 `--mm-processor-kwargs '{"video_pruning_rate": ...}'` 开启，开启后 `is_multimodal_pruning_enabled` 为 True。PixelPrune 与 EVS **不兼容**：

- EVS 开启后 `embed_multimodal` 会无条件调用 `_postprocess_image_embeds_evs`，该方法按 **完整** grid 切分 image embedding 并拼接 mrope 位置通道；
- 而 PixelPrune 已把 embedding 序列缩短，两者长度对不上会直接崩。

与其在深层报一个难以定位的形状错误，不如在入口处立即给出可操作的提示。新增 `_assert_no_evs_conflict`，在 `_process_image_input` 检测到 `keep_indices` 时调用：

```python
def _assert_no_evs_conflict(self: Any) -> None:
    if getattr(self, "is_multimodal_pruning_enabled", False):
        raise RuntimeError(
            "PixelPrune 与 EVS 视频剪枝不兼容：... 请移除 video_pruning_rate 后重试。"
        )
```

注意：EVS 影响的是 **视频** 模态，PixelPrune 剪的是 **图像** 模态。两者本可共存，但 EVS 的 `_postprocess_image_embeds_evs` 对 image embedding 也按完整 grid 处理，才导致冲突。这是 vLLM 上游实现细节，未来可能解耦。

---

## 11. 兼容策略模式总结

整个适配过程反复用到以下几种模式，可复用到未来版本升级：

| 模式 | 适用场景 | 写法 |
|------|----------|------|
| 运行时探测 + 缓存 | 方法改名 | `hasattr` 逐个试，命中后缓存到模块级变量 |
| `*args/**kwargs` 透传 | 签名变化 | patch 函数用可变参数，原样转给原始实现 |
| `getattr` 守卫 + 条件传参 | 新增可选参数 | `if getattr(self, "x", None) is not None: kwargs["x"] = ...` |
| `hasattr` 守卫 + 跳过补丁 | 新增可选方法 | `if hasattr(cls, "m"): patches.append(...)` |
| 延迟构造 | 探测可能抛错 | 把补丁清单从模块常量改成函数，调用时才解析 |
| 严格失败 + 自检 | 防止静默失效 | apply 后核对函数身份，失败默认 `raise` |
| 入口互斥检测 | 与上游功能冲突 | 在最早能判断的入口处显式报错并给出操作建议 |

核心思想一句话：**把"硬编码的 API 名/签名"换成"运行时探测 + 透传"，把"打不上补丁也照常跑"换成"严格失败 + 自检"**。

---

## 12. 环境变量速查

| 变量 | 默认 | 作用 |
|------|------|------|
| `PIXELPRUNE_ENABLED` | `true` | 是否启用剪枝（vLLM 插件入口判断） |
| `PIXELPRUNE_STRICT` | `true` | 补丁失败时是否终止进程（`false` 降级为 warning） |
| `PIXELPRUNE_METHOD` | `pred_2d` | 剪枝方法 |
| `PIXELPRUNE_METRIC` | `max` | 剪枝度量 |
| `PIXELPRUNE_THRESHOLD` | `0.0` | 剪枝阈值 |
| `PIXELPRUNE_VERBOSE` | `false` | 打印每张图的 pruning 统计 |

---

## 13. 验证清单

升级 vLLM 版本后，建议按以下顺序验证：

1. **静态自检**：`PIXELPRUNE_ENABLED=true python3 scripts/check_pixelprune_vllm.py`，确认所有补丁 `PATCHED`。
2. **启动日志**：确认出现 `patches applied successfully (hf_proc_hook=...)`，且 hook 名符合该版本预期。
3. **功能验证**：用一张已知图像跑推理，开启 `PIXELPRUNE_VERBOSE=true`，确认 `[PixelPrune]` 统计输出且 retain 比例合理。
4. **互斥检测**：同时开启 EVS，确认 `_assert_no_evs_conflict` 报错。
5. **性能对比**：对比未剪枝 / 剪枝的吞吐与显存，确认剪枝生效。
