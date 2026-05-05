#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot side projections of voxel wall hit counts.")
    parser.add_argument("case_dir", help="Directory containing particle_hits_summary.json and face_hits.u64.")
    parser.add_argument("--out", help="Output PNG path. Defaults to CASE_DIR/wall_hit_side_projection.png.")
    parser.add_argument("--chunk-z", type=int, default=32, help="Z chunk size used while summing the memmap.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    summary = json.loads((case_dir / "particle_hits_summary.json").read_text(encoding="utf-8"))
    nz, ny, nx = [int(v) for v in summary["shape_zyx"]]
    dx_um = float(summary["dx_um"])
    hits_path = Path(summary.get("outputs", {}).get("face_hits_u64", case_dir / "face_hits.u64"))
    if not hits_path.is_absolute():
        hits_path = ROOT / hits_path

    hits = np.memmap(hits_path, dtype=np.uint64, mode="r", shape=(6, nz, ny, nx))
    xz = np.zeros((nz, nx), dtype=np.float64)
    yz = np.zeros((nz, ny), dtype=np.float64)
    for z0 in range(0, nz, args.chunk_z):
        z1 = min(nz, z0 + args.chunk_z)
        block = hits[:, z0:z1, :, :]
        xz[z0:z1, :] = block.sum(axis=(0, 2), dtype=np.float64)
        yz[z0:z1, :] = block.sum(axis=(0, 3), dtype=np.float64)

    out = Path(args.out) if args.out else case_dir / "wall_hit_side_projection.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    positive = np.concatenate([xz[xz > 0], yz[yz > 0]])
    if positive.size:
        norm = LogNorm(vmin=max(float(np.percentile(positive, 1)), 1.0), vmax=float(positive.max()))
    else:
        norm = None

    extent_xz = [0, nx * dx_um, 0, nz * dx_um]
    extent_yz = [0, ny * dx_um, 0, nz * dx_um]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180, constrained_layout=True)
    panels = [
        (axes[0], xz, extent_xz, "wall hits projected to X-Z", "x (um)"),
        (axes[1], yz, extent_yz, "wall hits projected to Y-Z", "y (um)"),
    ]
    for ax, arr, extent, title, xlabel in panels:
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="inferno", norm=norm, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("z (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="hit count")

    fig.savefig(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
