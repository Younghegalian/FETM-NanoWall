from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write legacy VTK files from saved per-face hit counts.")
    parser.add_argument(
        "--sweep-dir",
        default="runs/sample_001/fullres_pressure_sweep_100ppm_10us",
        help="Pressure sweep directory containing lambda_* case folders.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing VTK files.")
    return parser.parse_args()


def load_summary(case_dir: Path) -> dict:
    with (case_dir / "mesh_particle_hits_summary.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def write_summary(case_dir: Path, summary: dict) -> None:
    with (case_dir / "mesh_particle_hits_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def write_vtk(
    vtk_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_hits: np.ndarray,
    chunk_size: int = 500_000,
) -> None:
    with vtk_path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("isotropic marching tetrahedra height-field particle wall hit count\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write(f"POINTS {vertices.shape[0]} float\n")
        for start in range(0, vertices.shape[0], chunk_size):
            stop = min(start + chunk_size, vertices.shape[0])
            np.savetxt(file, vertices[start:stop], fmt="%.9g")

        file.write(f"POLYGONS {faces.shape[0]} {faces.shape[0] * 4}\n")
        for start in range(0, faces.shape[0], chunk_size):
            stop = min(start + chunk_size, faces.shape[0])
            cells = np.empty((stop - start, 4), dtype=np.int64)
            cells[:, 0] = 3
            cells[:, 1:] = faces[start:stop]
            np.savetxt(file, cells, fmt="%d")

        file.write(f"CELL_DATA {faces.shape[0]}\n")
        file.write("SCALARS hit_count unsigned_long 1\n")
        file.write("LOOKUP_TABLE default\n")
        for start in range(0, face_hits.shape[0], chunk_size):
            stop = min(start + chunk_size, face_hits.shape[0])
            np.savetxt(file, face_hits[start:stop], fmt="%d")


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    case_dirs = sorted(path for path in sweep_dir.glob("lambda_*") if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No lambda_* case directories found in {sweep_dir}")

    first_summary = load_summary(case_dirs[0])
    outputs = first_summary["outputs"]
    n_vertex = int(first_summary["n_vertex"])
    n_face = int(first_summary["n_face"])
    vertices = np.memmap(outputs["surface_vertices_f32"], dtype=np.float32, mode="r").reshape(n_vertex, 3)
    faces = np.memmap(outputs["surface_faces_i32"], dtype=np.int32, mode="r").reshape(n_face, 3)

    for case_dir in case_dirs:
        summary = load_summary(case_dir)
        hits_path = Path(summary["outputs"]["surface_face_hits_u64"])
        if not hits_path.is_absolute():
            hits_path = Path.cwd() / hits_path
        vtk_path = case_dir / "mesh_wall_hit_count.vtk"
        if vtk_path.exists() and not args.force:
            print(f"[skip] {case_dir.name}: {vtk_path} already exists")
            continue

        expected_size = n_face * np.dtype(np.uint64).itemsize
        if hits_path.stat().st_size != expected_size:
            raise ValueError(f"{hits_path} size mismatch; expected {expected_size} bytes")

        print(f"[write] {case_dir.name}: {vtk_path}")
        started = time.perf_counter()
        face_hits = np.memmap(hits_path, dtype=np.uint64, mode="r")
        write_vtk(vtk_path, vertices, faces, face_hits)
        elapsed = time.perf_counter() - started

        summary["outputs"]["mesh_wall_hit_count_vtk"] = str(vtk_path)
        summary["outputs"]["mesh_wall_hit_count_vtk_postprocessed"] = str(vtk_path)
        summary.setdefault("postprocessing", {})["vtk_from_saved_face_hits"] = {
            "elapsed_s": elapsed,
            "path": str(vtk_path),
            "source_face_hits_u64": str(hits_path),
        }
        write_summary(case_dir, summary)
        print(f"[done] {case_dir.name}: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
