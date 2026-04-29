#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano_transport.voxelize import voxelize_height_domain


FLOAT_FIELDS = ("phi_total", "phi_scatter", "phi_surface", "accessibility", "vis_ang", "d_min_um")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export transport and nano-domain geometry for ParaView.")
    parser.add_argument("--transport", required=True, help="Path to transport_fields.npz.")
    parser.add_argument("--out-dir", required=True, help="Output directory for ParaView files.")
    parser.add_argument("--domain", help="Optional domain.npz for a triangulated height surface.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(FLOAT_FIELDS),
        help="Transport scalar fields to write to VTI.",
    )
    parser.add_argument("--skip-voxel-mesh", action="store_true", help="Do not export exact voxel boundary mesh.")
    parser.add_argument("--skip-height-mesh", action="store_true", help="Do not export triangulated height mesh.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transport_path = Path(args.transport)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(transport_path, allow_pickle=False)
    meta = _load_metadata(data)
    dx_um = float(meta["dx_um"])
    shape = tuple(int(v) for v in data["mask_solid"].shape)
    nz, ny, nx = shape

    fields = {name: np.asarray(data[name], dtype=np.float32) for name in args.fields if name in data.files}
    missing = sorted(set(args.fields) - set(fields))
    if missing:
        raise KeyError(f"transport file does not contain fields: {missing}")
    mask_solid = np.asarray(data["mask_solid"], dtype=np.uint8)

    volume_path = out_dir / "transport_fields.vti"
    _write_vti_cell_data(
        volume_path,
        shape_zyx=shape,
        spacing_um=(dx_um, dx_um, dx_um),
        float_fields=fields,
        uint8_fields={"mask_solid": mask_solid},
    )

    outputs: dict[str, object] = {
        "source_transport": str(transport_path),
        "shape_zyx": [nz, ny, nx],
        "dx_um": dx_um,
        "volume_vti": str(volume_path),
        "volume_fields": sorted([*fields.keys(), "mask_solid"]),
        "cell_data_layout": "arrays are voxel/cell data with order [z, y, x]; ParaView coordinates use x/y/z in um",
    }

    if not args.skip_voxel_mesh:
        voxel_mesh_path = out_dir / "domain_solid_voxel_surface.vtk"
        face_count = _write_voxel_boundary_mesh(voxel_mesh_path, mask_solid.astype(bool), dx_um)
        outputs["domain_solid_voxel_surface_vtk"] = str(voxel_mesh_path)
        outputs["domain_solid_voxel_surface_faces"] = face_count

    domain_path = Path(args.domain) if args.domain else _domain_from_metadata(meta)
    if domain_path is not None and not args.skip_height_mesh:
        height_mesh_path = out_dir / "domain_height_surface.vtk"
        height_meta = _write_height_surface_mesh(
            height_mesh_path,
            domain_path=domain_path,
            xy_stride=int(meta["xy_stride"]),
            z_padding_um=0.0,
        )
        outputs["domain_height_surface_vtk"] = str(height_mesh_path)
        outputs["domain_height_surface"] = height_meta

    metadata_path = out_dir / "paraview_export_metadata.json"
    metadata_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" in data.files:
        raw = str(np.asarray(data["metadata_json"]).reshape(-1)[0])
        return json.loads(raw)
    raise KeyError("transport file does not contain metadata_json")


def _domain_from_metadata(meta: dict) -> Path | None:
    source = meta.get("source_domain")
    if not source:
        return None
    path = Path(str(source))
    return path if path.exists() else None


def _write_vti_cell_data(
    path: Path,
    shape_zyx: tuple[int, int, int],
    spacing_um: tuple[float, float, float],
    float_fields: dict[str, np.ndarray],
    uint8_fields: dict[str, np.ndarray],
) -> None:
    nz, ny, nx = shape_zyx
    data_blocks: list[bytes] = []
    arrays: list[tuple[str, str, int]] = []
    offset = 0

    for name, arr in float_fields.items():
        block = _array_payload(np.asarray(arr, dtype="<f4"))
        arrays.append((name, "Float32", offset))
        data_blocks.append(block)
        offset += len(block)
    for name, arr in uint8_fields.items():
        block = _array_payload(np.asarray(arr, dtype=np.uint8))
        arrays.append((name, "UInt8", offset))
        data_blocks.append(block)
        offset += len(block)

    with path.open("wb") as file:
        _write_line(file, '<?xml version="1.0"?>')
        _write_line(file, '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">')
        _write_line(
            file,
            f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" '
            f'Origin="0 0 0" Spacing="{spacing_um[0]:.12g} {spacing_um[1]:.12g} {spacing_um[2]:.12g}">',
        )
        _write_line(file, f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">')
        _write_line(file, '      <CellData Scalars="phi_total">')
        for name, vtk_type, vtk_offset in arrays:
            _write_line(file, f'        <DataArray type="{vtk_type}" Name="{name}" format="appended" offset="{vtk_offset}"/>')
        _write_line(file, "      </CellData>")
        _write_line(file, "    </Piece>")
        _write_line(file, "  </ImageData>")
        _write_line(file, '  <AppendedData encoding="raw">')
        file.write(b"_")
        for block in data_blocks:
            file.write(block)
        _write_line(file, "")
        _write_line(file, "  </AppendedData>")
        _write_line(file, "</VTKFile>")


def _array_payload(arr: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(arr.ravel(order="C"))
    size_header = np.array([contiguous.nbytes], dtype="<u8").tobytes()
    return size_header + contiguous.tobytes(order="C")


def _write_voxel_boundary_mesh(path: Path, solid: np.ndarray, dx_um: float) -> int:
    with path.open("wb") as file:
        face_count = _count_boundary_faces(solid)
        point_count = face_count * 4
        _write_legacy_polydata_header(file, "solid voxel boundary", point_count, face_count)

        point_offset = 0
        face_offset = 0
        for points in _iter_boundary_face_points(solid, dx_um):
            n_face = points.shape[0]
            _write_be_f32(file, points.reshape(-1, 3))
            point_offset += n_face * 4
        _write_line(file, "")
        _write_line(file, f"POLYGONS {face_count} {face_count * 5}")

        for points in _iter_boundary_face_points(solid, dx_um):
            n_face = points.shape[0]
            polys = np.empty((n_face, 5), dtype=">i4")
            base = face_offset + np.arange(n_face, dtype=np.int32) * 4
            polys[:, 0] = 4
            polys[:, 1] = base
            polys[:, 2] = base + 1
            polys[:, 3] = base + 2
            polys[:, 4] = base + 3
            file.write(polys.tobytes(order="C"))
            face_offset += n_face * 4
        _write_line(file, "")
    return int(face_count)


def _count_boundary_faces(solid: np.ndarray) -> int:
    nz, ny, nx = solid.shape
    count = 0
    count += int(np.count_nonzero(solid[:, :, 0]))
    count += int(np.count_nonzero(solid[:, :, nx - 1]))
    if nx > 1:
        count += int(np.count_nonzero(solid[:, :, 1:] != solid[:, :, :-1]))
    count += int(np.count_nonzero(solid[:, 0, :]))
    count += int(np.count_nonzero(solid[:, ny - 1, :]))
    if ny > 1:
        count += int(np.count_nonzero(solid[:, 1:, :] != solid[:, :-1, :]))
    count += int(np.count_nonzero(solid[0, :, :]))
    count += int(np.count_nonzero(solid[nz - 1, :, :]))
    if nz > 1:
        count += int(np.count_nonzero(solid[1:, :, :] != solid[:-1, :, :]))
    return count


def _iter_boundary_face_points(solid: np.ndarray, dx_um: float) -> Iterable[np.ndarray]:
    nz, ny, nx = solid.shape
    directions = [
        ("x-", solid[:, :, 0], (0, 0, 0)),
        ("x+", solid[:, :, nx - 1], (0, 0, nx - 1)),
        ("y-", solid[:, 0, :], (0, 0, 0)),
        ("y+", solid[:, ny - 1, :], (0, ny - 1, 0)),
        ("z-", solid[0, :, :], (0, 0, 0)),
        ("z+", solid[nz - 1, :, :], (nz - 1, 0, 0)),
    ]

    for axis, mask, origin in directions:
        points = _boundary_points_for_mask(axis, mask, origin, dx_um)
        if points.size:
            yield points

    if nx > 1:
        trans = solid[:, :, 1:] != solid[:, :, :-1]
        left_solid = trans & solid[:, :, :-1]
        right_solid = trans & solid[:, :, 1:]
        for points in (
            _boundary_points_for_mask("x+", left_solid, (0, 0, 0), dx_um),
            _boundary_points_for_mask("x-", right_solid, (0, 0, 1), dx_um),
        ):
            if points.size:
                yield points
    if ny > 1:
        trans = solid[:, 1:, :] != solid[:, :-1, :]
        low_solid = trans & solid[:, :-1, :]
        high_solid = trans & solid[:, 1:, :]
        for points in (
            _boundary_points_for_mask("y+", low_solid, (0, 0, 0), dx_um),
            _boundary_points_for_mask("y-", high_solid, (0, 1, 0), dx_um),
        ):
            if points.size:
                yield points
    if nz > 1:
        trans = solid[1:, :, :] != solid[:-1, :, :]
        low_solid = trans & solid[:-1, :, :]
        high_solid = trans & solid[1:, :, :]
        for points in (
            _boundary_points_for_mask("z+", low_solid, (0, 0, 0), dx_um),
            _boundary_points_for_mask("z-", high_solid, (1, 0, 0), dx_um),
        ):
            if points.size:
                yield points


def _boundary_points_for_mask(axis: str, mask: np.ndarray, origin_zyx: tuple[int, int, int], dx_um: float) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 4, 3), dtype=np.float32)
    z = coords[:, 0] + origin_zyx[0]
    if mask.ndim == 3:
        y = coords[:, 1] + origin_zyx[1]
        x = coords[:, 2] + origin_zyx[2]
    elif axis.startswith("x"):
        y = coords[:, 1] + origin_zyx[1]
        x = np.full(coords.shape[0], origin_zyx[2], dtype=np.int64)
    elif axis.startswith("y"):
        y = np.full(coords.shape[0], origin_zyx[1], dtype=np.int64)
        x = coords[:, 1] + origin_zyx[2]
    else:
        y = coords[:, 0] + origin_zyx[1]
        x = coords[:, 1] + origin_zyx[2]
        z = np.full(coords.shape[0], origin_zyx[0], dtype=np.int64)

    x0 = x.astype(np.float32) * dx_um
    x1 = (x.astype(np.float32) + 1.0) * dx_um
    y0 = y.astype(np.float32) * dx_um
    y1 = (y.astype(np.float32) + 1.0) * dx_um
    z0 = z.astype(np.float32) * dx_um
    z1 = (z.astype(np.float32) + 1.0) * dx_um

    points = np.empty((coords.shape[0], 4, 3), dtype=np.float32)
    if axis == "x-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x0, y0, z1])
        points[:, 2] = np.column_stack([x0, y1, z1])
        points[:, 3] = np.column_stack([x0, y1, z0])
    elif axis == "x+":
        points[:, 0] = np.column_stack([x1, y0, z0])
        points[:, 1] = np.column_stack([x1, y1, z0])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x1, y0, z1])
    elif axis == "y-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x1, y0, z0])
        points[:, 2] = np.column_stack([x1, y0, z1])
        points[:, 3] = np.column_stack([x0, y0, z1])
    elif axis == "y+":
        points[:, 0] = np.column_stack([x0, y1, z0])
        points[:, 1] = np.column_stack([x0, y1, z1])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x1, y1, z0])
    elif axis == "z-":
        points[:, 0] = np.column_stack([x0, y0, z0])
        points[:, 1] = np.column_stack([x0, y1, z0])
        points[:, 2] = np.column_stack([x1, y1, z0])
        points[:, 3] = np.column_stack([x1, y0, z0])
    elif axis == "z+":
        points[:, 0] = np.column_stack([x0, y0, z1])
        points[:, 1] = np.column_stack([x1, y0, z1])
        points[:, 2] = np.column_stack([x1, y1, z1])
        points[:, 3] = np.column_stack([x0, y1, z1])
    else:
        raise ValueError(f"unknown face axis: {axis}")
    return points


def _write_height_surface_mesh(
    path: Path,
    domain_path: Path,
    xy_stride: int,
    z_padding_um: float,
) -> dict[str, object]:
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
    }


def _write_legacy_polydata_header(file, title: str, point_count: int, face_count: int) -> None:
    _write_line(file, "# vtk DataFile Version 3.0")
    _write_line(file, title)
    _write_line(file, "BINARY")
    _write_line(file, "DATASET POLYDATA")
    _write_line(file, f"POINTS {point_count} float")


def _write_be_f32(file, arr: np.ndarray) -> None:
    file.write(np.asarray(arr, dtype=">f4").tobytes(order="C"))


def _write_line(file, text: str) -> None:
    file.write(text.encode("utf-8") + b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
