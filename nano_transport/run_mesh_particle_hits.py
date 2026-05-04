from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from nano_transport.voxelize import voxelize_height_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gas particle wall-hit simulation on an isotropic height-field isosurface."
    )
    parser.add_argument("--domain", required=True, help="Input domain.npz from the SEM pipeline.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--xy-stride", type=int, default=8, help="Downsampling stride for x/y before meshing.")
    parser.add_argument(
        "--max-triangle-edge-um",
        type=float,
        default=0.05,
        help="Target maximum x/y/z cell size for isosurface extraction.",
    )
    parser.add_argument("--z-padding-um", type=float, default=0.2, help="Void padding above max height.")
    parser.add_argument("--n-particle", type=int, default=None, help="Particle count. Defaults to ppm-based count.")
    parser.add_argument("--ppm", type=float, default=50e-6, help="Gas mole fraction used when --n-particle is omitted.")
    parser.add_argument(
        "--init-mode",
        choices=("uniform", "top"),
        default="uniform",
        help="Initial/reinjection distribution: full void region or z above wall height.",
    )
    parser.add_argument("--wall-height-um", type=float, default=None, help="Lower z bound for --init-mode top.")
    parser.add_argument("--steps", type=int, default=150000, help="Number of simulation steps.")
    parser.add_argument("--total-time-s", type=float, default=None, help="Set steps from this simulated duration and --dt-s.")
    parser.add_argument("--dt-s", type=float, default=3e-12, help="Simulation time step in seconds.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps to skip before counting wall hits.")
    parser.add_argument("--curve-interval-steps", type=int, default=1, help="Write hit_curve.csv every N counted steps.")
    parser.add_argument("--skip-vtk", action="store_true", help="Skip writing the large per-face VTK file.")
    parser.add_argument("--temp-k", type=float, default=298.0, help="Gas temperature in K.")
    parser.add_argument("--pressure-pa", type=float, default=1.01e5, help="Gas pressure in Pa.")
    parser.add_argument("--molecular-diameter-m", type=float, default=3.7e-10, help="Molecular diameter in m.")
    parser.add_argument("--molar-mass-kg-mol", type=float, default=46e-3, help="Molar mass in kg/mol.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--kernel", default="transport_cpp/mesh_particle_hits", help="Compiled C++ kernel path.")
    parser.add_argument("--mesh-kernel", default="transport_cpp/height_isosurface", help="Compiled C++ surface mesh builder path.")
    parser.add_argument("--mesh-cache-dir", default=None, help="Directory for reusable surface meshes. Defaults next to domain.npz.")
    parser.add_argument("--force-remesh", action="store_true", help="Ignore cached surface mesh and rebuild it.")
    parser.add_argument("--rebuild-kernel", action="store_true", help="Rebuild the C++ kernel before running.")
    return parser


def main(argv: list[str] | None = None) -> int:
    total_wall_start = time.perf_counter()
    args = build_parser().parse_args(argv)
    if args.max_triangle_edge_um <= 0:
        raise ValueError("--max-triangle-edge-um must be positive")
    if args.total_time_s is not None:
        if args.total_time_s <= 0:
            raise ValueError("--total-time-s must be positive")
        args.steps = int(np.ceil(args.total_time_s / args.dt_s))
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.curve_interval_steps <= 0:
        raise ValueError("--curve-interval-steps must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = Path(args.kernel)
    if not kernel.is_absolute():
        kernel = Path(__file__).resolve().parents[1] / kernel
    mesh_kernel = Path(args.mesh_kernel)
    if not mesh_kernel.is_absolute():
        mesh_kernel = Path(__file__).resolve().parents[1] / mesh_kernel
    if args.rebuild_kernel or not kernel.exists():
        build_kernel(kernel)
    if args.rebuild_kernel or not mesh_kernel.exists():
        build_mesh_kernel(mesh_kernel)

    domain = voxelize_height_domain(args.domain, xy_stride=args.xy_stride, z_padding_um=args.z_padding_um)
    base_height = np.asarray(domain.height_um, dtype=np.float32)
    mesh_dx_um = min(float(args.max_triangle_edge_um), float(domain.dx_um))
    height, dx_um = resample_height_bilinear_spacing(
        base_height,
        source_dx_um=float(domain.dx_um),
        target_dx_um=mesh_dx_um,
    )
    ny, nx = height.shape
    iso_dz_um = dx_um
    mesh_wall_start = time.perf_counter()
    mesh_cache = load_or_build_mesh_cache(
        args=args,
        mesh_kernel=mesh_kernel,
        height=height,
        nx=nx,
        ny=ny,
        dx_um=dx_um,
        z_max_um=float(domain.z_max_um),
        dz_um=iso_dz_um,
    )
    mesh_wall_s = time.perf_counter() - mesh_wall_start
    height_path = mesh_cache["height_path"]
    vertices_path = mesh_cache["vertices_path"]
    faces_path = mesh_cache["faces_path"]
    mesh_meta = mesh_cache["mesh_builder"]
    n_vertex = int(mesh_meta["n_vertex"])
    n_face = int(mesh_meta["n_face"])
    if vertices_path.stat().st_size != n_vertex * 3 * np.dtype(np.float32).itemsize:
        raise ValueError("surface vertex file size does not match mesh metadata")
    if faces_path.stat().st_size != n_face * 3 * np.dtype(np.int32).itemsize:
        raise ValueError("surface mesh builder output size does not match its metadata")

    constants = gas_constants(
        temp_k=args.temp_k,
        pressure_pa=args.pressure_pa,
        molecular_diameter_m=args.molecular_diameter_m,
        molar_mass_kg_mol=args.molar_mass_kg_mol,
    )
    n_particle = int(args.n_particle) if args.n_particle is not None else particle_count_from_ppm(
        mask_solid=domain.mask_solid,
        dx_um=float(domain.dx_um),
        ppm=args.ppm,
        molar_volume_m3_mol=constants["molar_volume_m3_mol"],
    )
    wall_height_um = float(args.wall_height_um) if args.wall_height_um is not None else float(np.nanmax(height))
    domain_x_um = float(nx * dx_um)
    domain_y_um = float(ny * dx_um)
    bin_size_um = max(min(float(args.max_triangle_edge_um), 0.02), 1e-6)

    cmd = [
        str(kernel),
        str(height_path),
        str(nx),
        str(ny),
        f"{dx_um:.12g}",
        f"{domain.z_max_um:.12g}",
        str(vertices_path),
        str(faces_path),
        str(n_vertex),
        str(n_face),
        f"{domain_x_um:.12g}",
        f"{domain_y_um:.12g}",
        f"{bin_size_um:.12g}",
        str(n_particle),
        str(args.steps),
        f"{args.dt_s:.12g}",
        str(args.warmup_steps),
        f"{constants['sigma_um_s']:.12g}",
        f"{constants['lambda_um']:.12g}",
        str(args.seed),
        args.init_mode,
        f"{wall_height_um:.12g}",
        str(out_dir),
        str(args.curve_interval_steps),
    ]
    kernel_wall_start = time.perf_counter()
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    kernel_wall_s = time.perf_counter() - kernel_wall_start
    kernel_meta = _parse_kernel_meta(result.stdout)

    hits_path = out_dir / "surface_face_hits.u64"
    if hits_path.stat().st_size != n_face * np.dtype(np.uint64).itemsize:
        actual = hits_path.stat().st_size // np.dtype(np.uint64).itemsize
        raise ValueError(f"{hits_path} has {actual} values; expected {n_face}")

    vtk_path = out_dir / "mesh_wall_hit_count.vtk"
    vtk_wall_start = time.perf_counter()
    if args.skip_vtk:
        area_um2 = float(mesh_meta["wall_area_um2"]) if "wall_area_um2" in mesh_meta else None
        vtk_output = None
    else:
        vertices = np.memmap(vertices_path, dtype=np.float32, mode="r").reshape(-1, 3)
        faces = np.memmap(faces_path, dtype=np.int32, mode="r").reshape(-1, 3)
        face_hits = np.memmap(hits_path, dtype=np.uint64, mode="r")
        area_um2 = write_triangle_hit_vtk(vtk_path, vertices, faces, face_hits)
        vtk_output = str(vtk_path)
    vtk_wall_s = time.perf_counter() - vtk_wall_start
    total_hits = int(kernel_meta["total_hits"])
    simulated_time_s = float(kernel_meta["simulated_time_s"])
    collision_rate = float(kernel_meta["collision_rate_s_inv"])
    total_wall_s = time.perf_counter() - total_wall_start
    particle_steps = int(args.steps) * int(n_particle)

    summary = {
        "source_domain": str(args.domain),
        "surface_model": "isotropic_marching_tetrahedra_height_field",
        "source_shape_yx": [int(base_height.shape[0]), int(base_height.shape[1])],
        "shape_yx": [int(ny), int(nx)],
        "dx_um": dx_um,
        "source_dx_um": float(domain.dx_um),
        "height_spacing_ratio": float(dx_um / float(domain.dx_um)),
        "iso_dz_um": float(iso_dz_um),
        "z_max_um": float(domain.z_max_um),
        "xy_stride": int(args.xy_stride),
        "max_triangle_edge_um": float(args.max_triangle_edge_um),
        "n_vertex": n_vertex,
        "n_face": n_face,
        "z_padding_um": float(args.z_padding_um),
        "n_particle": n_particle,
        "n_particle_source": "explicit" if args.n_particle is not None else "ppm",
        "ppm": float(args.ppm),
        "init_mode": args.init_mode,
        "wall_height_um": wall_height_um,
        "steps": int(args.steps),
        "total_time_s_requested": None if args.total_time_s is None else float(args.total_time_s),
        "dt_s": float(args.dt_s),
        "curve_interval_steps": int(args.curve_interval_steps),
        "skip_vtk": bool(args.skip_vtk),
        "warmup_steps": int(kernel_meta["warmup_steps"]),
        "simulated_time_s": simulated_time_s,
        "total_hits": total_hits,
        "collision_rate_s_inv": collision_rate,
        "wall_area_um2": area_um2,
        "area_averaged_collision_rate_um2_s_inv": collision_rate / area_um2 if area_um2 else None,
        "temp_k": float(args.temp_k),
        "pressure_pa": float(args.pressure_pa),
        "molecular_diameter_m": float(args.molecular_diameter_m),
        "molar_mass_kg_mol": float(args.molar_mass_kg_mol),
        "lambda_um": float(constants["lambda_um"]),
        "sigma_um_s": float(constants["sigma_um_s"]),
        "v_mean_um_s": float(constants["v_mean_um_s"]),
        "molar_volume_m3_mol": float(constants["molar_volume_m3_mol"]),
        "seed": int(args.seed),
        "timing_s": {
            "total_wall": total_wall_s,
            "mesh_cache_or_build": mesh_wall_s,
            "kernel_wall": kernel_wall_s,
            "vtk_write": vtk_wall_s,
            "particle_steps_per_wall_s": particle_steps / kernel_wall_s if kernel_wall_s > 0 else None,
        },
        "outputs": {
            "mesh_wall_hit_count_vtk": vtk_output,
            "surface_face_hits_u64": str(hits_path),
            "hit_curve_csv": str(out_dir / "hit_curve.csv"),
            "height_um_f32": str(height_path),
            "surface_vertices_f32": str(vertices_path),
            "surface_faces_i32": str(faces_path),
        },
        "mesh_cache": {
            "hit": bool(mesh_cache["cache_hit"]),
            "key": str(mesh_cache["cache_key"]),
            "dir": str(mesh_cache["cache_dir"]),
            "meta_json": str(mesh_cache["meta_path"]),
        },
        "mesh_builder": mesh_meta,
        "kernel": kernel_meta,
    }
    summary_path = out_dir / "mesh_particle_hits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


def build_kernel(kernel: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "transport_cpp" / "mesh_particle_hits.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["clang++", "-O3", "-std=c++17", str(source), "-o", str(kernel)], check=True)


def build_mesh_kernel(mesh_kernel: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "transport_cpp" / "height_isosurface.cpp"
    mesh_kernel.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["clang++", "-O3", "-std=c++17", str(source), "-o", str(mesh_kernel)], check=True)


def run_mesh_builder(
    mesh_kernel: Path,
    height_path: Path,
    nx: int,
    ny: int,
    dx_um: float,
    z_max_um: float,
    dz_um: float,
    vertices_path: Path,
    faces_path: Path,
) -> dict[str, float | int | str]:
    cmd = [
        str(mesh_kernel),
        str(height_path),
        str(nx),
        str(ny),
        f"{dx_um:.12g}",
        f"{z_max_um:.12g}",
        f"{dz_um:.12g}",
        str(vertices_path),
        str(faces_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return _parse_kernel_meta(result.stdout)


def load_or_build_mesh_cache(
    *,
    args: argparse.Namespace,
    mesh_kernel: Path,
    height: np.ndarray,
    nx: int,
    ny: int,
    dx_um: float,
    z_max_um: float,
    dz_um: float,
) -> dict[str, object]:
    cache_root = Path(args.mesh_cache_dir) if args.mesh_cache_dir else Path(args.domain).resolve().parent / "mesh_cache"
    identity = mesh_cache_identity(args, height, nx, ny, dx_um, z_max_um, dz_um)
    cache_key = mesh_cache_key(identity)
    cache_dir = cache_root / cache_key
    height_path = cache_dir / "height_um.f32"
    vertices_path = cache_dir / "surface_vertices.f32"
    faces_path = cache_dir / "surface_faces.i32"
    meta_path = cache_dir / "surface_mesh_meta.json"

    cache_hit = False
    mesh_meta: dict[str, float | int | str] | None = None
    if not args.force_remesh and meta_path.exists() and height_path.exists() and vertices_path.exists() and faces_path.exists():
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta.get("identity") == identity:
            mesh_meta = cached_meta["mesh_builder"]
            cache_hit = True

    if mesh_meta is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        height.ravel(order="C").tofile(height_path)
        mesh_meta = run_mesh_builder(
            mesh_kernel=mesh_kernel,
            height_path=height_path,
            nx=nx,
            ny=ny,
            dx_um=dx_um,
            z_max_um=z_max_um,
            dz_um=dz_um,
            vertices_path=vertices_path,
            faces_path=faces_path,
        )
        meta_path.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "mesh_builder": mesh_meta,
                    "files": {
                        "height_um_f32": str(height_path),
                        "surface_vertices_f32": str(vertices_path),
                        "surface_faces_i32": str(faces_path),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return {
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "cache_dir": cache_dir,
        "height_path": height_path,
        "vertices_path": vertices_path,
        "faces_path": faces_path,
        "meta_path": meta_path,
        "mesh_builder": mesh_meta,
    }


def mesh_cache_identity(
    args: argparse.Namespace,
    height: np.ndarray,
    nx: int,
    ny: int,
    dx_um: float,
    z_max_um: float,
    dz_um: float,
) -> dict[str, object]:
    height_c = np.ascontiguousarray(height, dtype=np.float32)
    height_hash = hashlib.sha256(height_c.tobytes()).hexdigest()
    return {
        "surface_model": "isotropic_marching_tetrahedra_height_field",
        "domain": str(Path(args.domain).resolve()),
        "xy_stride": int(args.xy_stride),
        "z_padding_um": float(args.z_padding_um),
        "nx": int(nx),
        "ny": int(ny),
        "dx_um": float(dx_um),
        "dz_um": float(dz_um),
        "z_max_um": float(z_max_um),
        "height_sha256": height_hash,
    }


def mesh_cache_key(identity: dict[str, object]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def gas_constants(temp_k: float, pressure_pa: float, molecular_diameter_m: float, molar_mass_kg_mol: float) -> dict[str, float]:
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
    return max(1, int(round(ppm * air_volume_m3 * na / molar_volume_m3_mol)))


def resample_height_bilinear_spacing(
    height: np.ndarray,
    source_dx_um: float,
    target_dx_um: float,
) -> tuple[np.ndarray, float]:
    ny, nx = height.shape
    domain_x_um = nx * source_dx_um
    domain_y_um = ny * source_dx_um
    out_nx = max(2, int(np.ceil(domain_x_um / target_dx_um)))
    out_ny = max(2, int(np.ceil(domain_y_um / target_dx_um)))
    dx_um = max(domain_x_um / out_nx, domain_y_um / out_ny)
    y_um = (np.arange(out_ny, dtype=np.float32) + 0.5) * dx_um
    x_um = (np.arange(out_nx, dtype=np.float32) + 0.5) * dx_um
    y = np.clip(y_um / source_dx_um - 0.5, 0.0, float(ny - 1))
    x = np.clip(x_um / source_dx_um - 0.5, 0.0, float(nx - 1))
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    wy = y - y0
    wx = x - x0
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    h00 = height[y0[:, None], x0[None, :]]
    h10 = height[y0[:, None], x1[None, :]]
    h01 = height[y1[:, None], x0[None, :]]
    h11 = height[y1[:, None], x1[None, :]]
    refined = (
        (1.0 - wx[None, :]) * (1.0 - wy[:, None]) * h00
        + wx[None, :] * (1.0 - wy[:, None]) * h10
        + (1.0 - wx[None, :]) * wy[:, None] * h01
        + wx[None, :] * wy[:, None] * h11
    )
    return refined.astype(np.float32), float(dx_um)


def write_triangle_hit_vtk(path: Path, vertices: np.ndarray, faces: np.ndarray, face_hits: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    with path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("isotropic marching tetrahedra height-field particle wall hit count\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write(f"POINTS {vertices.shape[0]} float\n")
        for point in vertices:
            file.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        file.write(f"POLYGONS {faces.shape[0]} {faces.shape[0] * 4}\n")
        for a, b, c in faces:
            file.write(f"3 {int(a)} {int(b)} {int(c)}\n")
        file.write(f"CELL_DATA {faces.shape[0]}\n")
        file.write("SCALARS hit_count unsigned_long 1\n")
        file.write("LOOKUP_TABLE default\n")
        for value in face_hits:
            file.write(f"{int(value)}\n")
    return float(np.sum(area))


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
