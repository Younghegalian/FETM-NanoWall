from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from nano_transport.voxelize import voxelize_height_domain


DEFAULT_V_MEAN_UM_S = 370353425.4688162


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run first-event probabilistic transport on a voxelized nanodomain."
    )
    parser.add_argument("--domain", required=True, help="Input domain.npz from SEM pipeline.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--lambda-um", type=float, required=True, help="Mean free path in um.")
    parser.add_argument(
        "--v-mean-um-s",
        type=float,
        default=DEFAULT_V_MEAN_UM_S,
        help="Mean molecular speed in um/s for kinetic_contact_rate_s_inv.",
    )
    parser.add_argument("--xy-stride", type=int, default=8, help="Downsampling stride for x/y.")
    parser.add_argument("--z-padding-um", type=float, default=0.2, help="Void padding above max height.")
    parser.add_argument("--n-dir", type=int, default=64, help="Fibonacci direction count.")
    parser.add_argument("--max-dist-factor", type=float, default=6.0)
    parser.add_argument("--max-reflect", type=int, default=2)
    parser.add_argument("--use-box-reflect", action="store_true")
    parser.add_argument("--n-thread", type=int, default=4, help="C++ worker thread count.")
    parser.add_argument(
        "--export-paraview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export ParaView VTI/VTK files after transport finishes.",
    )
    parser.add_argument(
        "--paraview-dir",
        help="ParaView output directory. Defaults to <out-dir>/paraview.",
    )
    parser.add_argument("--skip-paraview-voxel-mesh", action="store_true")
    parser.add_argument("--skip-paraview-height-mesh", action="store_true")
    parser.add_argument(
        "--export-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write transport_metrics.json with scalar integrals and averages.",
    )
    parser.add_argument(
        "--metrics-no-percentiles",
        action="store_true",
        help="Skip percentile metrics for faster summaries on very large domains.",
    )
    parser.add_argument(
        "--keep-kernel-buffers",
        action="store_true",
        help="Keep intermediate .f32/.u8 files produced by the C++ kernel.",
    )
    parser.add_argument("--kernel", default="transport_cpp/transport_dda", help="Compiled C++ kernel path.")
    parser.add_argument("--rebuild-kernel", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = Path(args.kernel)
    if not kernel.is_absolute():
        kernel = Path(__file__).resolve().parents[1] / kernel
    kernel = _windows_exe_path(kernel)
    if args.rebuild_kernel or not kernel.exists():
        build_kernel(kernel)

    domain = voxelize_height_domain(
        args.domain,
        xy_stride=args.xy_stride,
        z_padding_um=args.z_padding_um,
    )
    mask_path = out_dir / "mask_solid.u8"
    domain.mask_solid.astype(np.uint8).ravel(order="C").tofile(mask_path)

    nz, ny, nx = domain.shape_zyx
    cmd = [
        str(kernel),
        str(mask_path),
        str(nx),
        str(ny),
        str(nz),
        f"{domain.dx_um:.12g}",
        f"{args.lambda_um:.12g}",
        str(args.n_dir),
        f"{args.max_dist_factor:.12g}",
        str(args.max_reflect),
        "1" if args.use_box_reflect else "0",
        str(out_dir),
        str(args.n_thread),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    kernel_meta = _parse_kernel_meta(result.stdout)

    shape = (nz, ny, nx)
    fields = {
        "mask_solid": domain.mask_solid.astype(np.bool_),
        "accessibility": _read_f32(out_dir / "accessibility.f32", shape),
        "vis_ang": _read_f32(out_dir / "vis_ang.f32", shape),
        "d_min_um": _read_f32(out_dir / "d_min_um.f32", shape),
        "source_scatter_fraction": _read_f32(out_dir / "source_scatter_fraction.f32", shape),
        "source_escape_fraction": _read_f32(out_dir / "source_escape_fraction.f32", shape),
        "source_lost_fraction": _read_f32(out_dir / "source_lost_fraction.f32", shape),
        "source_probability_sum": _read_f32(out_dir / "source_probability_sum.f32", shape),
        "source_conservation_error": _read_f32(out_dir / "source_conservation_error.f32", shape),
    }
    kinetic_contact_rate = fields["accessibility"].astype(np.float32) * np.float32(args.v_mean_um_s / args.lambda_um)
    fields["kinetic_contact_rate_s_inv"] = kinetic_contact_rate
    fields["time_weighted_accessibility_s_inv"] = kinetic_contact_rate
    meta = {
        "source_domain": str(args.domain),
        "source_mode": "uniform_void",
        "primary_field": "accessibility",
        "primary_field_definition": "source-voxel direct surface-arrival fraction; source_scatter_fraction + accessibility + source_escape_fraction + source_lost_fraction ~= 1",
        "shape_zyx": list(shape),
        "xy_stride": args.xy_stride,
        "dx_um": domain.dx_um,
        "z_max_um": domain.z_max_um,
        "lambda_um": args.lambda_um,
        "v_mean_um_s": args.v_mean_um_s,
        "kinetic_contact_rate_scale_s_inv": args.v_mean_um_s / args.lambda_um,
        "kinetic_contact_rate_definition": "kinetic_contact_rate_s_inv = accessibility * v_mean_um_s / lambda_um",
        "time_weighted_accessibility_definition": "Deprecated alias of kinetic_contact_rate_s_inv.",
        "n_dir": args.n_dir,
        "max_dist_factor": args.max_dist_factor,
        "max_reflect": args.max_reflect,
        "use_box_reflect": args.use_box_reflect,
        "n_thread": args.n_thread,
        **kernel_meta,
    }
    npz_path = out_dir / "transport_fields.npz"
    metadata_path = out_dir / "metadata.json"
    np.savez_compressed(
        npz_path,
        **fields,
        metadata_json=np.array([json.dumps(meta, indent=2, sort_keys=True)]),
    )
    metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    metrics_meta = None
    if args.export_metrics:
        metrics_meta = summarize_metrics(
            transport_npz=npz_path,
            out_path=out_dir / "transport_metrics.json",
            no_percentiles=args.metrics_no_percentiles,
        )

    paraview_meta = None
    if args.export_paraview:
        paraview_dir = Path(args.paraview_dir) if args.paraview_dir else out_dir / "paraview"
        paraview_meta = export_paraview(
            transport_npz=npz_path,
            domain_npz=Path(args.domain),
            out_dir=paraview_dir,
            skip_voxel_mesh=args.skip_paraview_voxel_mesh,
            skip_height_mesh=args.skip_paraview_height_mesh,
        )

    if not args.keep_kernel_buffers:
        cleanup_kernel_buffers(out_dir)

    output = {
        "transport_npz": str(npz_path),
        "metadata": str(metadata_path),
        **meta,
    }
    if metrics_meta is not None:
        output["metrics"] = metrics_meta
    if paraview_meta is not None:
        output["paraview"] = paraview_meta
    print(json.dumps(output, indent=2))
    return 0


def build_kernel(kernel: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "transport_cpp" / "transport_dda.cpp"
    kernel = _windows_exe_path(kernel)
    kernel.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_clangxx(), "-O3", "-std=c++17", str(source), "-o", str(kernel)]
    subprocess.run(cmd, check=True)


def _windows_exe_path(path: Path) -> Path:
    if os.name == "nt" and path.suffix == "":
        return path.with_suffix(".exe")
    return path


def _clangxx() -> str:
    bundled = Path(r"C:\Program Files\LLVM\bin\clang++.exe")
    return str(bundled) if os.name == "nt" and bundled.exists() else "clang++"


def export_paraview(
    transport_npz: Path,
    domain_npz: Path,
    out_dir: Path,
    skip_voxel_mesh: bool,
    skip_height_mesh: bool,
) -> dict:
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_paraview.py"
    cmd = [
        sys.executable,
        str(script),
        "--transport",
        str(transport_npz),
        "--out-dir",
        str(out_dir),
        "--domain",
        str(domain_npz),
    ]
    if skip_voxel_mesh:
        cmd.append("--skip-voxel-mesh")
    if skip_height_mesh:
        cmd.append("--skip-height-mesh")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def summarize_metrics(transport_npz: Path, out_path: Path, no_percentiles: bool) -> dict:
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_transport_metrics.py"
    cmd = [
        sys.executable,
        str(script),
        "--transport",
        str(transport_npz),
        "--out",
        str(out_path),
    ]
    if no_percentiles:
        cmd.append("--no-percentiles")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def cleanup_kernel_buffers(out_dir: Path) -> None:
    for name in (
        "accessibility.f32",
        "vis_ang.f32",
        "d_min_um.f32",
        "source_scatter_fraction.f32",
        "source_escape_fraction.f32",
        "source_lost_fraction.f32",
        "source_probability_sum.f32",
        "source_conservation_error.f32",
        "mask_solid.u8",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def _read_f32(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} floats; expected {expected}")
    return arr.reshape(shape)


def _parse_kernel_meta(stdout: str) -> dict:
    meta = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            if "." in value or "e" in value.lower():
                meta[key] = float(value)
            else:
                meta[key] = int(value)
        except ValueError:
            meta[key] = value
    return meta


if __name__ == "__main__":
    raise SystemExit(main())
