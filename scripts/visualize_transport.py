#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create projection previews for transport_fields.npz.")
    parser.add_argument("--transport", required=True, help="Path to transport_fields.npz.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.transport)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(path, allow_pickle=False)
    scatter = data["phi_scatter"]
    surface = data["phi_surface"]
    solid = data["mask_solid"]
    accessibility = data["accessibility"]
    vis_ang = data["vis_ang"]
    d_min = data["d_min_um"]

    void = ~solid
    d_plot = np.where(d_min >= 0, d_min, np.nan)
    panels = [
        ("scatter xy sum", scatter.sum(axis=0), "magma", True),
        ("surface xy sum", surface.sum(axis=0), "viridis", True),
        ("solid column height proxy", solid.sum(axis=0), "gray", False),
        ("accessibility max z", np.nanmax(np.where(void, accessibility, np.nan), axis=0), "plasma", False),
        ("angular visibility max z", np.nanmax(np.where(void, vis_ang, np.nan), axis=0), "cividis", False),
        ("min wall distance min z", np.nanmin(d_plot, axis=0), "cubehelix", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=170)
    for ax, (title, arr, cmap, log_scale) in zip(axes.flat, panels):
        arr = np.asarray(arr, dtype=np.float64)
        if log_scale:
            arr = np.log10(np.maximum(arr, np.nanmax(arr) * 1e-8 if np.nanmax(arr) > 0 else 1e-12))
            title = f"log10 {title}"
        im = ax.imshow(arr, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
