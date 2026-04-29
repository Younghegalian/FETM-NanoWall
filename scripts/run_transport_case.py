#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a transport case and automatically export ParaView files."
    )
    parser.add_argument("--domain", default="runs/sample_001/domain.npz")
    parser.add_argument("--results-root", default="runs/sample_001")
    parser.add_argument("--out-dir", help="Optional explicit output directory.")
    parser.add_argument("--lambda-um", type=float, default=0.10)
    parser.add_argument("--xy-stride", type=int, required=True)
    parser.add_argument("--n-dir", type=int, required=True)
    parser.add_argument("--z-padding-um", type=float, default=0.20)
    parser.add_argument("--max-dist-factor", type=float, default=6.0)
    parser.add_argument("--max-reflect", type=int, default=4)
    parser.add_argument("--n-thread", type=int, default=8)
    parser.add_argument("--no-box-reflect", action="store_true")
    parser.add_argument("--rebuild-kernel", action="store_true")
    parser.add_argument("--keep-kernel-buffers", action="store_true")
    parser.add_argument("--skip-paraview-voxel-mesh", action="store_true")
    parser.add_argument("--skip-paraview-height-mesh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args)
    cmd = [
        sys.executable,
        "-m",
        "nano_transport.run_transport",
        "--domain",
        args.domain,
        "--out-dir",
        str(out_dir),
        "--lambda-um",
        f"{args.lambda_um:.12g}",
        "--xy-stride",
        str(args.xy_stride),
        "--z-padding-um",
        f"{args.z_padding_um:.12g}",
        "--n-dir",
        str(args.n_dir),
        "--max-dist-factor",
        f"{args.max_dist_factor:.12g}",
        "--max-reflect",
        str(args.max_reflect),
        "--n-thread",
        str(args.n_thread),
    ]
    if not args.no_box_reflect:
        cmd.append("--use-box-reflect")
    if args.rebuild_kernel:
        cmd.append("--rebuild-kernel")
    if args.keep_kernel_buffers:
        cmd.append("--keep-kernel-buffers")
    if args.skip_paraview_voxel_mesh:
        cmd.append("--skip-paraview-voxel-mesh")
    if args.skip_paraview_height_mesh:
        cmd.append("--skip-paraview-height-mesh")

    return subprocess.run(cmd, cwd=ROOT).returncode


def _default_out_dir(args: argparse.Namespace) -> Path:
    label = f"{args.lambda_um:.2f}".replace("-", "m").replace(".", "p")
    name = f"transport_lambda_{label}_stride{args.xy_stride}_dir{args.n_dir}"
    return Path(args.results_root) / name


if __name__ == "__main__":
    raise SystemExit(main())
