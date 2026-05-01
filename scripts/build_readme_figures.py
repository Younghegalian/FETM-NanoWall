#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ASSETS = ROOT / "docs" / "assets"
SAMPLE = ROOT / "runs" / "sample_001"
TRANSPORT = SAMPLE / "transport_lambda_0p10_stride4_dir256" / "transport_fields.npz"


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    _copy_png(SAMPLE / "cropped_input.png", ASSETS / "fetm_sem_crop.png")
    _copy_png(SAMPLE / "height_preview.png", ASSETS / "fetm_height_preview.png")
    _copy_png(SAMPLE / "visualization_3d" / "surface_3d_preview.png", ASSETS / "fetm_surface_preview.png")

    plotly_surface = Path.home() / "Downloads" / "newplot.png"
    if plotly_surface.exists():
        _copy_png(plotly_surface, ASSETS / "fetm_plotly_surface.png")

    _write_transport_preview(TRANSPORT, ASSETS / "fetm_transport_projection_preview.png")
    _write_accessibility_surface(
        domain_path=SAMPLE / "domain.npz",
        transport_path=TRANSPORT,
        out_path=ASSETS / "fetm_accessibility_surface.png",
    )
    return 0


def _copy_png(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def _write_transport_preview(transport_path: Path, out_path: Path) -> None:
    data = np.load(transport_path, allow_pickle=False)
    solid = data["mask_solid"]
    accessibility = data["accessibility"]
    vis_ang = data["vis_ang"]
    source_scatter = data["source_scatter_fraction"]
    source_lost = data["source_lost_fraction"]
    source_error = data["source_conservation_error"]

    void = ~solid
    panels = [
        ("source scatter fraction max z", np.nanmax(np.where(void, source_scatter, np.nan), axis=0), "magma", False),
        ("accessibility max z", np.nanmax(np.where(void, accessibility, np.nan), axis=0), "viridis", False),
        ("solid column height proxy", solid.sum(axis=0), "gray", False),
        ("source lost fraction max z", np.nanmax(np.where(void, source_lost, np.nan), axis=0), "plasma", False),
        ("angular visibility max z", np.nanmax(np.where(void, vis_ang, np.nan), axis=0), "cividis", False),
        ("source conservation error max z", np.nanmax(np.where(void, source_error, np.nan), axis=0), "cubehelix", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), dpi=170)
    for ax, (title, arr, cmap, log_scale) in zip(axes.flat, panels):
        arr = np.asarray(arr, dtype=np.float64)
        if log_scale:
            vmax = np.nanmax(arr)
            floor = vmax * 1e-8 if vmax > 0 else 1e-12
            arr = np.log10(np.maximum(arr, floor))
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_accessibility_surface(domain_path: Path, transport_path: Path, out_path: Path) -> None:
    domain = np.load(domain_path, allow_pickle=False)
    transport = np.load(transport_path, allow_pickle=False)
    meta = _load_meta(transport)
    stride = int(meta["xy_stride"])
    dx = float(meta["dx_um"])

    height = np.asarray(domain["height_um"], dtype=np.float32)
    height_ds = _block_max_downsample(height, stride)
    solid = transport["mask_solid"].astype(bool)
    accessibility = transport["accessibility"]
    access_xy = np.nanmax(np.where(~solid, accessibility, np.nan), axis=0)
    access_xy = np.nan_to_num(access_xy, nan=0.0)

    ny, nx = height_ds.shape
    x = (np.arange(nx, dtype=np.float32) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float32) + 0.5) * dx
    x_grid, y_grid = np.meshgrid(x, y)

    fig = plt.figure(figsize=(9, 7), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=float(np.nanmin(access_xy)), vmax=float(np.nanmax(access_xy)))
    colors = plt.cm.viridis(norm(access_xy))
    ax.plot_surface(
        x_grid,
        y_grid,
        height_ds,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(access_xy)
    fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08, label="accessibility")
    ax.set_title("accessibility mapped on nanowall height surface", pad=14)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("height (um)")
    ax.view_init(elev=52, azim=-128)
    ax.set_box_aspect((1, 1, 0.72))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_meta(npz: np.lib.npyio.NpzFile) -> dict:
    import json

    raw = str(np.asarray(npz["metadata_json"]).reshape(-1)[0])
    return json.loads(raw)


def _block_max_downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr.copy()
    h, w = arr.shape
    out_h = int(np.ceil(h / stride))
    out_w = int(np.ceil(w / stride))
    padded = np.full((out_h * stride, out_w * stride), np.nan, dtype=np.float32)
    padded[:h, :w] = arr
    blocks = padded.reshape(out_h, stride, out_w, stride)
    return np.nanmax(blocks, axis=(1, 3)).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
