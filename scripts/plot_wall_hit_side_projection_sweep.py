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
    parser = argparse.ArgumentParser(description="Plot sweep side projections with a shared color scale.")
    parser.add_argument("case_dirs", nargs="+", help="Case directories containing particle_hits_summary.json.")
    parser.add_argument("--out", help="Output summary PNG path.")
    parser.add_argument("--chunk-z", type=int, default=32, help="Z chunk size used while summing the memmap.")
    parser.add_argument("--vmin", type=float, help="Shared log-scale minimum. Defaults to global 1st percentile.")
    parser.add_argument("--vmax", type=float, help="Shared log-scale maximum. Defaults to global max.")
    parser.add_argument(
        "--normalize-by-particles",
        action="store_true",
        help="Divide each projection by the number of simulated particles in that case.",
    )
    parser.add_argument(
        "--write-case-plots",
        action="store_true",
        help="Also overwrite each case_dir/wall_hit_side_projection.png using the shared color scale.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = [
        _load_case(Path(case_dir), args.chunk_z, normalize_by_particles=args.normalize_by_particles)
        for case_dir in args.case_dirs
    ]
    positives = np.concatenate(
        [
            values[values > 0]
            for case in cases
            for values in (np.asarray(case["xz"]), np.asarray(case["yz"]))
        ]
    )
    positive_max = float(positives.max())
    vmin_default = float(np.percentile(positives, 1.0))
    vmin = float(args.vmin) if args.vmin is not None else max(vmin_default, np.finfo(float).tiny)
    vmax = float(args.vmax) if args.vmax is not None else positive_max
    norm = LogNorm(vmin=vmin, vmax=vmax)
    colorbar_label = "hits per particle" if args.normalize_by_particles else "hit count"

    default_name = (
        "wall_hit_side_projection_per_particle_summary.png"
        if args.normalize_by_particles
        else "wall_hit_side_projection_summary.png"
    )
    case_plot_name = (
        "wall_hit_side_projection_per_particle.png"
        if args.normalize_by_particles
        else "wall_hit_side_projection.png"
    )
    out = Path(args.out) if args.out else _common_parent([case["case_dir"] for case in cases]) / default_name
    _plot_summary(cases, norm, out, colorbar_label)
    if args.write_case_plots:
        for case in cases:
            _plot_case(case, norm, case["case_dir"] / case_plot_name, colorbar_label)

    print(out)
    return 0


def _load_case(case_dir: Path, chunk_z: int, normalize_by_particles: bool) -> dict[str, object]:
    summary = json.loads((case_dir / "particle_hits_summary.json").read_text(encoding="utf-8"))
    nz, ny, nx = [int(v) for v in summary["shape_zyx"]]
    dx_um = float(summary["dx_um"])
    hits_path = Path(summary.get("outputs", {}).get("face_hits_u64", case_dir / "face_hits.u64"))
    if not hits_path.is_absolute():
        hits_path = ROOT / hits_path

    hits = np.memmap(hits_path, dtype=np.uint64, mode="r", shape=(6, nz, ny, nx))
    xz = np.zeros((nz, nx), dtype=np.float64)
    yz = np.zeros((nz, ny), dtype=np.float64)
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        block = hits[:, z0:z1, :, :]
        xz[z0:z1, :] = block.sum(axis=(0, 2), dtype=np.float64)
        yz[z0:z1, :] = block.sum(axis=(0, 3), dtype=np.float64)

    n_particle = int(summary["n_particle"])
    if normalize_by_particles:
        xz /= float(n_particle)
        yz /= float(n_particle)

    lambda_um = summary.get("lambda_um")
    label = f"lambda = {float(lambda_um):.2f} um" if lambda_um is not None else case_dir.name
    if normalize_by_particles:
        label = f"{label}\nN = {n_particle}"
    return {
        "case_dir": case_dir,
        "label": label,
        "dx_um": dx_um,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "xz": xz,
        "yz": yz,
    }


def _plot_summary(cases: list[dict[str, object]], norm: LogNorm, out: Path, colorbar_label: str) -> None:
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(10.8, 2.6 * len(cases)),
        dpi=180,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    last_image = None
    for row, case in enumerate(cases):
        last_image = _draw_panel(axes[row, 0], case, "xz", norm)
        last_image = _draw_panel(axes[row, 1], case, "yz", norm)
        axes[row, 0].set_ylabel(f"{case['label']}\nz (um)")
    fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.024, pad=0.015, label=colorbar_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _plot_case(case: dict[str, object], norm: LogNorm, out: Path, colorbar_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180, constrained_layout=True)
    image = _draw_panel(axes[0], case, "xz", norm)
    image = _draw_panel(axes[1], case, "yz", norm)
    fig.suptitle(str(case["label"]), fontsize=14)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label=colorbar_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _draw_panel(ax: plt.Axes, case: dict[str, object], plane: str, norm: LogNorm):
    arr = np.asarray(case[plane], dtype=np.float64)
    dx_um = float(case["dx_um"])
    if plane == "xz":
        extent = [0, int(case["nx"]) * dx_um, 0, int(case["nz"]) * dx_um]
        title = "X-Z projection"
        xlabel = "x (um)"
    else:
        extent = [0, int(case["ny"]) * dx_um, 0, int(case["nz"]) * dx_um]
        title = "Y-Z projection"
        xlabel = "y (um)"
    plot_arr = np.maximum(arr, norm.vmin)
    image = ax.imshow(plot_arr, origin="lower", aspect="auto", cmap="inferno", norm=norm, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if plane == "yz":
        ax.set_ylabel("z (um)")
    return image


def _common_parent(paths: list[Path]) -> Path:
    resolved = [path.resolve() for path in paths]
    common = Path(os.path.commonpath([str(path) for path in resolved]))
    return common if common.is_dir() else common.parent


if __name__ == "__main__":
    raise SystemExit(main())
