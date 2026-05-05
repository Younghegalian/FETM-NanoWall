#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_transport.voxelize import voxelize_height_domain


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V_MEAN_UM_S = 370353425.4688162


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a transport case and automatically export ParaView files."
    )
    parser.add_argument("--domain", default="runs/sample_001/domain.npz")
    parser.add_argument("--results-root", default="runs/sample_001")
    parser.add_argument("--out-dir", help="Optional explicit output directory.")
    parser.add_argument("--lambda-um", type=float, default=0.10)
    parser.add_argument("--v-mean-um-s", type=float, default=DEFAULT_V_MEAN_UM_S)
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
    parser.add_argument(
        "--reuse-height-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache domain_height_surface.vtk per domain/stride and link it into each case.",
    )
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
        "--v-mean-um-s",
        f"{args.v_mean_um_s:.12g}",
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
    if args.skip_paraview_height_mesh or args.reuse_height_mesh:
        cmd.append("--skip-paraview-height-mesh")

    started_at = datetime.now().astimezone()
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    ended_at = datetime.now().astimezone()
    runtime = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "returncode": result.returncode,
        "command": cmd,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.returncode == 0 and args.reuse_height_mesh and not args.skip_paraview_height_mesh:
        height_mesh = _ensure_cached_height_mesh(args)
        paraview_dir = out_dir / "paraview"
        paraview_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(height_mesh, paraview_dir / "domain_height_surface.vtk")
        _record_reused_height_mesh(paraview_dir / "paraview_export_metadata.json", height_mesh)
    (out_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return result.returncode


def _default_out_dir(args: argparse.Namespace) -> Path:
    label = f"{args.lambda_um:.2f}".replace("-", "m").replace(".", "p")
    name = f"transport_lambda_{label}_stride{args.xy_stride}_dir{args.n_dir}"
    return Path(args.results_root) / name


def _ensure_cached_height_mesh(args: argparse.Namespace) -> Path:
    cache_dir = Path(args.results_root) / f"domain_height_stride{args.xy_stride}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = cache_dir / "domain_height_surface.vtk"
    metadata_path = cache_dir / "domain_height_surface_metadata.json"
    domain_path = _resolve_path(args.domain)
    if not mesh_path.exists():
        meta = _write_height_surface_mesh(
            mesh_path,
            domain_path=domain_path,
            xy_stride=int(args.xy_stride),
            z_padding_um=0.0,
        )
        metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return mesh_path


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def _write_height_surface_mesh(path: Path, domain_path: Path, xy_stride: int, z_padding_um: float) -> dict[str, object]:
    domain = voxelize_height_domain(str(domain_path), xy_stride=xy_stride, z_padding_um=z_padding_um)
    height = domain.height_um.astype(np.float32)
    ny, nx = height.shape
    y = (np.arange(ny, dtype=np.float32) + 0.5) * domain.dx_um
    x = (np.arange(nx, dtype=np.float32) + 0.5) * domain.dx_um
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel(), height.ravel()]).astype(np.float32)

    faces = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            a = row * nx + col
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    cells = np.asarray(faces, dtype=np.int32)

    with path.open("wb") as file:
        _write_line(file, "# vtk DataFile Version 3.0")
        _write_line(file, "height surface")
        _write_line(file, "BINARY")
        _write_line(file, "DATASET POLYDATA")
        _write_line(file, f"POINTS {points.shape[0]} float")
        _write_be_f32(file, points)
        _write_line(file, "")
        _write_line(file, f"POLYGONS {cells.shape[0]} {cells.shape[0] * 4}")
        polys = np.empty((cells.shape[0], 4), dtype=">i4")
        polys[:, 0] = 3
        polys[:, 1:] = cells.astype(">i4", copy=False)
        file.write(polys.tobytes(order="C"))
        _write_line(file, "")
        _write_line(file, f"POINT_DATA {points.shape[0]}")
        _write_line(file, "SCALARS height_um float 1")
        _write_line(file, "LOOKUP_TABLE default")
        _write_be_f32(file, height.ravel(order="C"))
        _write_line(file, "")

    return {
        "source_domain": str(domain_path),
        "xy_stride": xy_stride,
        "shape_yx": [ny, nx],
        "n_points": int(points.shape[0]),
        "n_triangles": int(cells.shape[0]),
        "dx_um": float(domain.dx_um),
        "vtk": str(path),
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _record_reused_height_mesh(metadata_path: Path, height_mesh: Path) -> None:
    meta = {}
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta["domain_height_surface_vtk"] = str(height_mesh)
    meta["domain_height_surface_reused"] = True
    metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _write_line(file, text: str) -> None:
    file.write(text.encode("ascii"))
    file.write(b"\n")


def _write_be_f32(file, arr: np.ndarray) -> None:
    file.write(np.asarray(arr, dtype=">f4").tobytes(order="C"))


if __name__ == "__main__":
    raise SystemExit(main())
