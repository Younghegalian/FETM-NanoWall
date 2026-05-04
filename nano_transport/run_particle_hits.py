from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from nano_transport.voxelize import voxelize_height_domain


FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a stripped particle gas-wall collision simulation on a voxelized nano-domain."
    )
    parser.add_argument("--domain", required=True, help="Input domain.npz from the SEM pipeline.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--xy-stride", type=int, default=8, help="Downsampling stride for x/y.")
    parser.add_argument("--z-padding-um", type=float, default=0.2, help="Void padding above max height.")
    parser.add_argument(
        "--n-particle",
        type=int,
        default=None,
        help="Number of simulated gas particles. Defaults to the ppm-based ideal-gas count.",
    )
    parser.add_argument("--ppm", type=float, default=50e-6, help="Gas mole fraction used to compute n when --n-particle is omitted.")
    parser.add_argument(
        "--init-mode",
        choices=("uniform", "top"),
        default="uniform",
        help="Initial/reinjection particle distribution: full void domain or void cells above wall height.",
    )
    parser.add_argument(
        "--wall-height-um",
        type=float,
        default=None,
        help="Lower z bound for --init-mode top. Defaults to the max reconstructed height.",
    )
    parser.add_argument("--steps", type=int, default=150000, help="Number of simulation steps.")
    parser.add_argument("--dt-s", type=float, default=3e-12, help="Simulation time step in seconds.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps to skip before counting wall hits.")
    parser.add_argument("--temp-k", type=float, default=298.0, help="Gas temperature in K.")
    parser.add_argument("--pressure-pa", type=float, default=1.01e5, help="Gas pressure in Pa.")
    parser.add_argument("--molecular-diameter-m", type=float, default=3.7e-10, help="Molecular diameter in m.")
    parser.add_argument("--molar-mass-kg-mol", type=float, default=46e-3, help="Molar mass in kg/mol.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--kernel", default="transport_cpp/particle_hits", help="Compiled C++ kernel path.")
    parser.add_argument("--rebuild-kernel", action="store_true", help="Rebuild the C++ kernel before running.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = Path(args.kernel)
    if not kernel.is_absolute():
        kernel = Path(__file__).resolve().parents[1] / kernel
    if args.rebuild_kernel or not kernel.exists():
        build_kernel(kernel)

    domain = voxelize_height_domain(
        args.domain,
        xy_stride=args.xy_stride,
        z_padding_um=args.z_padding_um,
    )
    mask_path = out_dir / "mask_solid.u8"
    domain.mask_solid.astype(np.uint8).ravel(order="C").tofile(mask_path)

    constants = gas_constants(
        temp_k=args.temp_k,
        pressure_pa=args.pressure_pa,
        molecular_diameter_m=args.molecular_diameter_m,
        molar_mass_kg_mol=args.molar_mass_kg_mol,
    )
    nz, ny, nx = domain.shape_zyx
    n_particle = int(args.n_particle) if args.n_particle is not None else particle_count_from_ppm(
        mask_solid=domain.mask_solid,
        dx_um=domain.dx_um,
        ppm=args.ppm,
        molar_volume_m3_mol=constants["molar_volume_m3_mol"],
    )
    top_min_z_um = float(args.wall_height_um) if args.wall_height_um is not None else float(np.nanmax(domain.height_um))
    cmd = [
        str(kernel),
        str(mask_path),
        str(nx),
        str(ny),
        str(nz),
        f"{domain.dx_um:.12g}",
        str(n_particle),
        str(args.steps),
        f"{args.dt_s:.12g}",
        str(args.warmup_steps),
        f"{constants['sigma_um_s']:.12g}",
        f"{constants['lambda_um']:.12g}",
        str(args.seed),
        args.init_mode,
        f"{top_min_z_um:.12g}",
        str(out_dir),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    kernel_meta = _parse_kernel_meta(result.stdout)

    face_hits_path = out_dir / "face_hits.u64"
    face_hits = np.fromfile(face_hits_path, dtype=np.uint64)
    expected = 6 * int(np.prod(domain.shape_zyx))
    if face_hits.size != expected:
        raise ValueError(f"{face_hits_path} has {face_hits.size} values; expected {expected}")
    face_hits = face_hits.reshape((6, nz, ny, nx))

    vtk_path = out_dir / "wall_hit_count.vtk"
    face_count = write_wall_hit_vtk(vtk_path, domain.mask_solid, face_hits, domain.dx_um)
    wall_area_um2 = float(face_count * domain.dx_um * domain.dx_um)
    warmup_steps = int(kernel_meta["warmup_steps"])
    simulated_time_s = float(kernel_meta["simulated_time_s"])
    total_hits = int(kernel_meta["total_hits"])
    collision_rate = float(kernel_meta["collision_rate_s_inv"])

    summary = {
        "source_domain": str(args.domain),
        "shape_zyx": [int(v) for v in domain.shape_zyx],
        "dx_um": float(domain.dx_um),
        "xy_stride": int(args.xy_stride),
        "z_padding_um": float(args.z_padding_um),
        "n_particle": n_particle,
        "n_particle_source": "explicit" if args.n_particle is not None else "ppm",
        "ppm": float(args.ppm),
        "init_mode": args.init_mode,
        "wall_height_um": top_min_z_um,
        "steps": int(args.steps),
        "dt_s": float(args.dt_s),
        "warmup_steps": warmup_steps,
        "simulated_time_s": simulated_time_s,
        "total_hits": total_hits,
        "collision_rate_s_inv": collision_rate,
        "wall_surface_face_count": int(face_count),
        "wall_area_um2": wall_area_um2,
        "area_averaged_collision_rate_um2_s_inv": collision_rate / wall_area_um2 if wall_area_um2 > 0 else 0.0,
        "temp_k": float(args.temp_k),
        "pressure_pa": float(args.pressure_pa),
        "molecular_diameter_m": float(args.molecular_diameter_m),
        "molar_mass_kg_mol": float(args.molar_mass_kg_mol),
        "seed": int(args.seed),
        "lambda_um": float(constants["lambda_um"]),
        "sigma_um_s": float(constants["sigma_um_s"]),
        "v_mean_um_s": float(constants["v_mean_um_s"]),
        "molar_volume_m3_mol": float(constants["molar_volume_m3_mol"]),
        "outputs": {
            "wall_hit_count_vtk": str(vtk_path),
            "hit_curve_csv": str(out_dir / "hit_curve.csv"),
            "face_hits_u64": str(face_hits_path),
            "mask_solid_u8": str(mask_path),
        },
        "kernel": kernel_meta,
    }
    summary_path = out_dir / "particle_hits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


def build_kernel(kernel: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "transport_cpp" / "particle_hits.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["clang++", "-O3", "-std=c++17", str(source), "-o", str(kernel)]
    subprocess.run(cmd, check=True)


def gas_constants(
    temp_k: float,
    pressure_pa: float,
    molecular_diameter_m: float,
    molar_mass_kg_mol: float,
) -> dict[str, float]:
    kb = 1.380649e-23
    na = 6.02214076e23
    gas_constant = 8.314462618
    molecule_mass_kg = molar_mass_kg_mol / na
    sigma_m_s = float(np.sqrt(kb * temp_k / molecule_mass_kg))
    v_mean_m_s = float(np.sqrt(8.0 * kb * temp_k / (np.pi * molecule_mass_kg)))
    lambda_m = float(kb * temp_k / (np.sqrt(2.0) * np.pi * molecular_diameter_m**2 * pressure_pa))
    return {
        "sigma_um_s": sigma_m_s * 1e6,
        "v_mean_um_s": v_mean_m_s * 1e6,
        "lambda_um": lambda_m * 1e6,
        "molar_volume_m3_mol": gas_constant * temp_k / pressure_pa,
    }


def particle_count_from_ppm(mask_solid: np.ndarray, dx_um: float, ppm: float, molar_volume_m3_mol: float) -> int:
    na = 6.02214076e23
    if ppm <= 0:
        raise ValueError("--ppm must be positive")
    void_voxels = int(np.count_nonzero(~mask_solid))
    voxel_volume_m3 = (dx_um * 1e-6) ** 3
    air_volume_m3 = void_voxels * voxel_volume_m3
    n = round(ppm * air_volume_m3 * na / molar_volume_m3_mol)
    return max(1, int(n))


def write_wall_hit_vtk(path: Path, solid: np.ndarray, face_hits: np.ndarray, dx_um: float) -> int:
    face_count = sum(int(np.count_nonzero(_exposed_face_mask(solid, face))) for face in FACE_NAMES)
    point_count = face_count * 4
    with path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("particle wall hit count\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write(f"POINTS {point_count} float\n")

        for face in FACE_NAMES:
            mask = _exposed_face_mask(solid, face)
            coords = np.argwhere(mask)
            points = _face_points(face, coords, dx_um)
            for point in points.reshape(-1, 3):
                file.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")

        file.write(f"POLYGONS {face_count} {face_count * 5}\n")
        offset = 0
        for face in FACE_NAMES:
            n_face = int(np.count_nonzero(_exposed_face_mask(solid, face)))
            for _ in range(n_face):
                file.write(f"4 {offset} {offset + 1} {offset + 2} {offset + 3}\n")
                offset += 4

        file.write(f"CELL_DATA {face_count}\n")
        file.write("SCALARS hit_count unsigned_long 1\n")
        file.write("LOOKUP_TABLE default\n")
        for face_idx, face in enumerate(FACE_NAMES):
            mask = _exposed_face_mask(solid, face)
            for value in face_hits[face_idx][mask]:
                file.write(f"{int(value)}\n")
    return face_count


def _exposed_face_mask(solid: np.ndarray, face: str) -> np.ndarray:
    exposed = np.zeros_like(solid, dtype=bool)
    if face == "x-":
        exposed[:, :, 1:] = solid[:, :, 1:] & ~solid[:, :, :-1]
    elif face == "x+":
        exposed[:, :, :-1] = solid[:, :, :-1] & ~solid[:, :, 1:]
    elif face == "y-":
        exposed[:, 1:, :] = solid[:, 1:, :] & ~solid[:, :-1, :]
    elif face == "y+":
        exposed[:, :-1, :] = solid[:, :-1, :] & ~solid[:, 1:, :]
    elif face == "z-":
        exposed[1:, :, :] = solid[1:, :, :] & ~solid[:-1, :, :]
    elif face == "z+":
        exposed[:-1, :, :] = solid[:-1, :, :] & ~solid[1:, :, :]
    else:
        raise ValueError(f"unknown face: {face}")
    return exposed


def _face_points(face: str, coords_zyx: np.ndarray, dx_um: float) -> np.ndarray:
    if coords_zyx.size == 0:
        return np.empty((0, 4, 3), dtype=np.float32)

    z = coords_zyx[:, 0].astype(np.float32)
    y = coords_zyx[:, 1].astype(np.float32)
    x = coords_zyx[:, 2].astype(np.float32)
    x0 = x * dx_um
    x1 = (x + 1.0) * dx_um
    y0 = y * dx_um
    y1 = (y + 1.0) * dx_um
    z0 = z * dx_um
    z1 = (z + 1.0) * dx_um

    points = np.empty((coords_zyx.shape[0], 4, 3), dtype=np.float32)
    if face == "x-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x0, y0, z1])
        points[:, 2] = np.column_stack([x0, y1, z1])
        points[:, 3] = np.column_stack([x0, y1, z0])
    elif face == "x+":
        points[:, 0] = np.column_stack([x1, y0, z0])
        points[:, 1] = np.column_stack([x1, y1, z0])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x1, y0, z1])
    elif face == "y-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x1, y0, z0])
        points[:, 2] = np.column_stack([x1, y0, z1])
        points[:, 3] = np.column_stack([x0, y0, z1])
    elif face == "y+":
        points[:, 0] = np.column_stack([x0, y1, z0])
        points[:, 1] = np.column_stack([x0, y1, z1])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x1, y1, z0])
    elif face == "z-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x0, y1, z0])
        points[:, 2] = np.column_stack([x1, y1, z0])
        points[:, 3] = np.column_stack([x1, y0, z0])
    elif face == "z+":
        points[:, 0] = np.column_stack([x0, y0, z1])
        points[:, 1] = np.column_stack([x1, y0, z1])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x0, y1, z1])
    else:
        raise ValueError(f"unknown face: {face}")
    return points


def _parse_kernel_meta(stdout: str) -> dict[str, float | int | str]:
    meta: dict[str, float | int | str] = {}
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
