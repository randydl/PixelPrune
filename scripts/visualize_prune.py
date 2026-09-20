#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PixelPrune debug visualizer (pred_2d / max only)
================================================

Run the pred_2d selector with the `max` metric on an image and visualize
the original vs. the PixelPrune-reconstructed token grid.

Six panels (row 1: 1/2/3, row 2: 4/5/6):
    1. Original image (resized)
    2. Keep mask: white = kept, black = pruned
    3. Kept-token overlay on the original image (bright = kept, dim = pruned)
    4. Token grid: average color of each merged token
    5. After PixelPrune: grid reconstructed from kept tokens via LOCO-I
    6. Reconstruction error heatmap

Two preprocessing modes:
    - processor mode (default): use a HuggingFace AutoProcessor (exact model input).
    - manual mode (--no-processor): built-in smart_resize + patchify.

Examples:
    python scripts/visualize_prune.py
    python scripts/visualize_prune.py --threshold 0.05 --outdir debug_vis
    python scripts/visualize_prune.py --no-processor --max-pixels 1003520
    python scripts/visualize_prune.py --image a.png b.jpg --outdir debug_vis
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pixelprune.core import compute_merged_keep_indices  # noqa: E402


# Constants (Qwen3-VL)
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
BLOCK_SIZE = SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE  # 4
MERGE_PIXEL = PATCH_SIZE * SPATIAL_MERGE_SIZE          # 32 px per merged token

DEFAULT_PROC_PATH = "/nas_train/app.e0016372/models/Qwen/Qwen3.5-0.8B"
DEFAULT_IMAGE = "assets/doc.jpg"


# --- Preprocessing ---

def smart_resize(h: int, w: int, factor: int = MERGE_PIXEL,
                 min_pixels: int = 65536, max_pixels: int = 16777216) -> Tuple[int, int]:
    """Qwen3VL smart_resize: snap dims to a multiple of `factor`."""
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def preprocess_manual(img, max_pixels: int, min_pixels: int):
    """No processor: smart_resize + patchify, returns patches in [0,1]."""
    from PIL import Image
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB")

    w, h = img.size
    h_b, w_b = smart_resize(h, w, factor=MERGE_PIXEL,
                            min_pixels=min_pixels, max_pixels=max_pixels)
    img_r = img.resize((w_b, h_b), Image.BICUBIC)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0  # (Hb, Wb, 3) in [0,1]

    nh, nw = h_b // PATCH_SIZE, w_b // PATCH_SIZE
    patches = arr.reshape(nh, PATCH_SIZE, nw, PATCH_SIZE, 3)
    patches = patches.transpose(0, 2, 4, 1, 3)  # (nh, nw, 3, PH, PW)
    patches = patches.reshape(nh * nw, 3 * PATCH_SIZE * PATCH_SIZE)
    patches01 = torch.from_numpy(patches).contiguous()

    grid_thw = torch.tensor([[1, nh, nw]], dtype=torch.long)
    return patches01, grid_thw, img_r


def preprocess_processor(img, proc_path: str):
    """Use a real HuggingFace processor, then de-normalize to [0,1]."""
    from PIL import Image
    from transformers import AutoProcessor
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")

    proc = AutoProcessor.from_pretrained(proc_path)
    out = proc(images=[img], return_tensors="pt")
    pixel_values = out["pixel_values"]    # (N, C*T*PH*PW) in [-1,1]
    grid_thw = out["image_grid_thw"]      # (1,3)

    C, T, H, W = 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
    pv = pixel_values.view(-1, C, T, H, W)[:, :, 0, :, :]  # take frame 0
    pv = pv * 0.5 + 0.5                                    # de-normalize -> [0,1]
    patches01 = pv.reshape(pv.shape[0], -1).contiguous()
    return patches01, grid_thw, img


def resolve_processor_path(arg: Optional[str]) -> Optional[str]:
    if arg:
        return arg
    env = os.environ.get("PIXELPRUNE_PROC_PATH")
    if env:
        return env
    if os.path.isdir(DEFAULT_PROC_PATH):
        return DEFAULT_PROC_PATH
    return None


# --- LOCO-I reconstruction (mirrors pred_2d's predictor) ---

def loco_reconstruct(grid: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Reconstruct the full merged grid from kept tokens using LOCO-I (A=left, B=up, C=up-left)."""
    mh, mw, _ = grid.shape
    rec = grid.clone()
    g = grid

    def eq(a, b):
        return torch.equal(a, b)

    for r in range(mh):
        for c in range(mw):
            if keep[r, c] or (r == 0 and c == 0):
                rec[r, c] = g[r, c]
                continue
            if c > 0:
                A = rec[r, c - 1]
            elif r > 0:
                A = rec[r - 1, 0]
            else:
                A = g[r, c]
            B = rec[r - 1, c] if r > 0 else A
            Cn = rec[r - 1, c - 1] if (r > 0 and c > 0) else B
            if r > 0 and c > 0:
                cb = eq(Cn, B)
                ca = eq(Cn, A)
                pred = B if (ca and not cb) else A
            else:
                pred = A
            rec[r, c] = pred
    return rec


# --- Token color grid ---

def merged_color_grid(patches01: torch.Tensor, grid_thw: torch.Tensor
                      ) -> Tuple[torch.Tensor, int, int]:
    """Aggregate patch pixels into a merged-token average-color grid (mh, mw, 3)."""
    _, h, w = grid_thw[0].tolist()
    mh, mw = int(h) // SPATIAL_MERGE_SIZE, int(w) // SPATIAL_MERGE_SIZE
    n_merged = mh * mw
    merged = patches01.view(n_merged, BLOCK_SIZE, 3, PATCH_SIZE, PATCH_SIZE)
    colors = merged.mean(dim=(1, 3, 4))  # (n_merged, 3)
    return colors.view(mh, mw, 3), mh, mw


# --- Visualization ---

def visualize_one(
    image_path: str,
    threshold: float,
    use_processor: bool,
    proc_path: Optional[str],
    max_pixels: int,
    min_pixels: int,
    outdir: Optional[str],
    show: bool,
) -> dict:
    import matplotlib
    if outdir and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1. preprocess ----
    if use_processor:
        if proc_path is None:
            print(f"[warn] no processor found, falling back to manual mode: {image_path}")
            patches01, grid_thw, img_ref = preprocess_manual(
                image_path, max_pixels, min_pixels)
        else:
            patches01, grid_thw, img_ref = preprocess_processor(image_path, proc_path)
    else:
        patches01, grid_thw, img_ref = preprocess_manual(
            image_path, max_pixels, min_pixels)

    # ---- 2. run pred_2d / max ----
    keep_indices = compute_merged_keep_indices(
        patches01,
        grid_thw,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        method="pred_2d",
        metric="max",
        threshold=threshold,
    )[0]

    merged_colors, mh, mw = merged_color_grid(patches01, grid_thw)
    keep_mask = torch.zeros(mh * mw, dtype=torch.bool)
    keep_mask[keep_indices.long()] = True
    keep_mask = keep_mask.view(mh, mw)

    n_total = mh * mw
    n_keep = int(keep_mask.sum().item())
    ratio = n_keep / n_total if n_total else 1.0

    # ---- 3. reconstruct + error ----
    recon = loco_reconstruct(merged_colors, keep_mask)
    err = (merged_colors - recon).abs().mean(dim=-1)  # (mh, mw)
    err_np = err.numpy()

    # ---- 4. plot (row1: original/keep mask/overlay, row2: token grid/recon/error); figure sized to shared aspect ratio (h_px x w_px) ----
    h_px, w_px = mh * MERGE_PIXEL, mw * MERGE_PIXEL
    ar = w_px / h_px  # width / height of every panel
    panel_h = 2.8  # inches per panel row height
    panel_w = panel_h * ar
    fig_w = 3 * panel_w
    fig_h = 2 * panel_h + 1.0
    fig_w = float(min(max(fig_w, 9.0), 22.0))
    fig_h = float(min(max(fig_h, 5.0), 14.0))
    fig, axes = plt.subplots(
        2, 3, figsize=(fig_w, fig_h), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    orig_arr = np.asarray(
        img_ref.convert("RGB") if hasattr(img_ref, "convert") else img_ref)
    keep_np = keep_mask.numpy().astype(bool)

    # panel 3: keep mask (panel 2) overlaid on the original (panel 1) at 80% opacity.
    keep_tiled = np.kron(
        keep_np.astype(np.float32),
        np.ones((MERGE_PIXEL, MERGE_PIXEL), dtype=np.float32),
    )  # (h_px, w_px)
    mask_rgb = np.stack([keep_tiled] * 3, axis=-1)  # white/black
    base = orig_arr[:h_px, :w_px].astype(np.float32)
    if base.max() > 1.5:
        base = base / 255.0
    overlay = base * 0.8 + mask_rgb * 0.2

    # (1) Original
    axes[0, 0].imshow(orig_arr)
    axes[0, 0].set_title("1. Original (resized)")
    axes[0, 0].axis("off")

    # (2) Keep mask
    axes[0, 1].imshow(keep_np.astype(np.uint8), cmap="gray",
                       interpolation="nearest", vmin=0, vmax=1)
    axes[0, 1].set_title(f"2. Keep mask (white=kept, {n_keep}/{n_total})")
    axes[0, 1].axis("off")

    # (3) Kept-token overlay on the original image
    axes[0, 2].imshow(overlay, interpolation="nearest")
    axes[0, 2].set_title("3. Kept-token overlay (bright=kept)")
    axes[0, 2].axis("off")

    # (4) Token grid
    axes[1, 0].imshow(merged_colors.numpy(), interpolation="nearest")
    axes[1, 0].set_title(f"4. Token grid (merged {mh}x{mw})")
    axes[1, 0].axis("off")

    # (5) After PixelPrune
    axes[1, 1].imshow(recon.numpy(), interpolation="nearest")
    axes[1, 1].set_title("5. After PixelPrune (LOCO-I recon)")
    axes[1, 1].axis("off")

    # (6) Recon error
    axes[1, 2].imshow(err_np, cmap="hot", interpolation="nearest",
                      vmin=0, vmax=max(float(err_np.max()), 1e-6))
    axes[1, 2].set_title("6. Recon error (|orig - recon| mean)")
    axes[1, 2].axis("off")

    title = (f"{os.path.basename(image_path)}  |  pred_2d / max  tau={threshold}  |  "
             f"kept {n_keep}/{n_total} = {ratio*100:.2f}% "
             f"(pruned {100*(1-ratio):.2f}%)  |  merged grid {mh}x{mw}")
    fig.suptitle(title, fontsize=12)

    stats = {
        "image": image_path, "method": "pred_2d", "metric": "max",
        "threshold": threshold, "merged_grid": [mh, mw],
        "total_tokens": n_total, "kept_tokens": n_keep,
        "retain_ratio": ratio, "compression": 1 - ratio,
        "mean_recon_err": float(err.mean().item()),
        "max_recon_err": float(err.max().item()),
    }
    print(f"[stats] {stats}")

    # ---- 5. save / show ----
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        suffix = f"_pred2d_max_t{threshold}".replace(".", "p")
        out_path = os.path.join(outdir, f"{stem}{suffix}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[save] {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return stats


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PixelPrune debug visualizer (pred_2d / max only): "
                    "original vs. PixelPrune-reconstructed token grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", nargs="*", default=[DEFAULT_IMAGE],
                   help=f"image path(s); default {DEFAULT_IMAGE}")
    p.add_argument("--threshold", type=float,
                   default=float(os.environ.get("PIXELPRUNE_THRESHOLD", "0.0") or "0.0"),
                   help="similarity threshold tau (max metric)")
    p.add_argument("--no-processor", action="store_true",
                   help="skip HuggingFace processor; use built-in smart_resize+patchify")
    p.add_argument("--processor-path", default=None,
                   help="local processor path (auto-detected by default)")
    p.add_argument("--max-pixels", type=int, default=16777216,
                   help="max_pixels in manual mode (controls downsampling)")
    p.add_argument("--min-pixels", type=int, default=65536,
                   help="min_pixels in manual mode")
    p.add_argument("--outdir", default=None,
                   help="directory to save figures; if unset, nothing is saved")
    p.add_argument("--show", action="store_true",
                   help="open an interactive window (use --outdir on headless hosts)")
    return p


def main():
    args = build_parser().parse_args()
    proc_path = resolve_processor_path(args.processor_path)
    use_proc = (not args.no_processor)

    all_stats: List[dict] = []
    for img_path in args.image:
        if not os.path.exists(img_path):
            print(f"[skip] file not found: {img_path}")
            continue
        try:
            s = visualize_one(
                image_path=img_path,
                threshold=args.threshold,
                use_processor=use_proc,
                proc_path=proc_path,
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                outdir=args.outdir,
                show=args.show,
            )
            all_stats.append(s)
        except Exception as e:
            print(f"[error] {img_path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_stats) > 1:
        avg_ratio = float(np.mean([s["retain_ratio"] for s in all_stats]))
        avg_comp = float(np.mean([s["compression"] for s in all_stats]))
        print(f"\n[summary] {len(all_stats)} images | "
              f"avg retain {avg_ratio*100:.2f}% | avg pruned {avg_comp*100:.2f}%")


if __name__ == "__main__":
    main()
