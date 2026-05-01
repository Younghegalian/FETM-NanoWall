#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_transport.voxelize import voxelize_height_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the reconstructed height surface with SEM texture coordinates."
    )
    parser.add_argument("--domain", required=True, help="Path to domain.npz.")
    parser.add_argument("--texture", required=True, help="SEM crop image to map onto the height surface.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--stride", type=int, default=2, help="XY stride for mesh resolution.")
    parser.add_argument("--z-padding-um", type=float, default=0.2, help="Voxelizer z padding; kept for metadata parity.")
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=1.0,
        help="Visual-only z multiplier applied to exported vertices.",
    )
    parser.add_argument(
        "--no-flip-v",
        action="store_true",
        help="Do not flip OBJ/VTK v texture coordinate. Default follows OBJ bottom-left texture convention.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domain_path = Path(args.domain)
    texture_path = Path(args.texture)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(args.stride))
    z_exaggeration = float(args.z_exaggeration)
    flip_v = not bool(args.no_flip_v)

    domain = voxelize_height_domain(
        str(domain_path),
        xy_stride=stride,
        z_padding_um=float(args.z_padding_um),
    )
    height = domain.height_um.astype(np.float32) * z_exaggeration
    ny, nx = height.shape
    x = (np.arange(nx, dtype=np.float32) + 0.5) * domain.dx_um
    y = (np.arange(ny, dtype=np.float32) + 0.5) * domain.dx_um
    x_grid, y_grid = np.meshgrid(x, y)

    vertices = np.column_stack([x_grid.ravel(), y_grid.ravel(), height.ravel()]).astype(np.float32)
    uv = _build_uv(ny, nx, flip_v=flip_v)
    faces = _build_faces(ny, nx)

    texture_out = out_dir / "sem_texture.png"
    obj_path = out_dir / "domain_height_sem_textured.obj"
    mtl_path = out_dir / "domain_height_sem_textured.mtl"
    vtk_path = out_dir / "domain_height_sem_textured.vtk"
    metadata_path = out_dir / "textured_height_metadata.json"

    texture_size = _copy_texture(texture_path, texture_out)
    _write_mtl(mtl_path, texture_out.name)
    _write_obj(obj_path, mtl_path.name, vertices, uv, faces)
    _write_vtk(vtk_path, vertices, uv, faces, height.ravel(order="C"))

    metadata = {
        "source_domain": str(domain_path),
        "source_texture": str(texture_path),
        "texture_png": str(texture_out),
        "texture_size_px": list(texture_size),
        "stride": stride,
        "shape_yx": [ny, nx],
        "n_points": int(vertices.shape[0]),
        "n_triangles": int(faces.shape[0]),
        "dx_um": float(domain.dx_um),
        "z_exaggeration": z_exaggeration,
        "flip_v": flip_v,
        "obj": str(obj_path),
        "mtl": str(mtl_path),
        "vtk": str(vtk_path),
        "paraview_note": "Open the VTK or OBJ surface, then apply sem_texture.png as the texture if ParaView does not load it automatically.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


def _build_uv(ny: int, nx: int, *, flip_v: bool) -> np.ndarray:
    u = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    v = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    if flip_v:
        v = 1.0 - v
    u_grid, v_grid = np.meshgrid(u, v)
    return np.column_stack([u_grid.ravel(), v_grid.ravel()]).astype(np.float32)


def _build_faces(ny: int, nx: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            a = row * nx + col
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(faces, dtype=np.int32)


def _copy_texture(src: Path, dst: Path) -> tuple[int, int]:
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        rgb.save(dst)
        return rgb.size


def _write_mtl(path: Path, texture_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "newmtl sem_texture",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.000000",
                "illum 1",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_obj(path: Path, mtl_name: str, vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# FETM-NanoWall SEM-textured height surface\n")
        file.write(f"mtllib {mtl_name}\n")
        file.write("usemtl sem_texture\n")
        for x, y, z in vertices:
            file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uv:
            file.write(f"vt {u:.9g} {v:.9g}\n")
        for a, b, c in faces:
            # OBJ indices are 1-based. Vertex and texture-coordinate indices match.
            file.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")


def _write_vtk(path: Path, vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray, height: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("FETM SEM-textured height surface\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")
        file.write(f"POINTS {vertices.shape[0]} float\n")
        for x, y, z in vertices:
            file.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
        file.write(f"POLYGONS {faces.shape[0]} {faces.shape[0] * 4}\n")
        for a, b, c in faces:
            file.write(f"3 {a} {b} {c}\n")
        file.write(f"POINT_DATA {vertices.shape[0]}\n")
        file.write("TEXTURE_COORDINATES sem_uv 2 float\n")
        for u, v in uv:
            file.write(f"{u:.9g} {v:.9g}\n")
        file.write("SCALARS height_um float 1\n")
        file.write("LOOKUP_TABLE default\n")
        for value in height:
            file.write(f"{value:.9g}\n")


if __name__ == "__main__":
    raise SystemExit(main())
