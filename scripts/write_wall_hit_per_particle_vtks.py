#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano_transport.run_particle_hits import FACE_NAMES, _exposed_face_mask, _face_points, count_exposed_faces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write wall-hit VTK files normalized by simulated particle count.")
    parser.add_argument("case_dirs", nargs="+", help="Particle-hit case directories.")
    parser.add_argument(
        "--name",
        default="wall_hit_count_per_particle.vtk",
        help="Output VTK filename inside each case directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for case_dir_arg in args.case_dirs:
        case_dir = Path(case_dir_arg)
        summary_path = case_dir / "particle_hits_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        nz, ny, nx = [int(v) for v in summary["shape_zyx"]]
        dx_um = float(summary["dx_um"])
        n_particle = int(summary["n_particle"])
        if n_particle <= 0:
            raise ValueError(f"{case_dir}: n_particle must be positive")

        mask_path = _resolve_path(summary.get("outputs", {}).get("mask_solid_u8", case_dir / "mask_solid.u8"))
        hits_path = _resolve_path(summary.get("outputs", {}).get("face_hits_u64", case_dir / "face_hits.u64"))
        solid = np.fromfile(mask_path, dtype=np.uint8).reshape((nz, ny, nx)).astype(bool)
        face_hits = np.memmap(hits_path, dtype=np.uint64, mode="r", shape=(6, nz, ny, nx))

        out_path = case_dir / args.name
        face_count = write_vtk(out_path, solid, face_hits, dx_um, n_particle)
        if face_count != count_exposed_faces(solid):
            raise ValueError(f"{case_dir}: exposed face count changed while writing VTK")

        summary.setdefault("outputs", {})["wall_hit_count_per_particle_vtk"] = str(out_path)
        summary.setdefault("postprocessing", {})["wall_hit_count_per_particle_vtk"] = {
            "source_face_hits_u64": str(hits_path),
            "normalization": "face hit count divided by n_particle",
            "n_particle": n_particle,
            "scalar_name": "hit_count_per_particle",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(out_path)
    return 0


def write_vtk(path: Path, solid: np.ndarray, face_hits: np.ndarray, dx_um: float, n_particle: int) -> int:
    face_count = count_exposed_faces(solid)
    point_count = face_count * 4
    with path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("particle wall hit count per particle\n")
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
        file.write("SCALARS hit_count_per_particle float 1\n")
        file.write("LOOKUP_TABLE default\n")
        for face_idx, face in enumerate(FACE_NAMES):
            mask = _exposed_face_mask(solid, face)
            values = np.asarray(face_hits[face_idx][mask], dtype=np.float64) / float(n_particle)
            for value in values:
                file.write(f"{value:.9g}\n")
    return face_count


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
