#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a nano-domain height field as 3D previews.")
    parser.add_argument("--domain", required=True, help="Path to domain.npz.")
    parser.add_argument("--out-dir", required=True, help="Output directory for 3D artifacts.")
    parser.add_argument("--stride", type=int, default=2, help="Grid stride for preview/mesh export.")
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=1.0,
        help="Visual z scale multiplier. Metadata keeps the true height units.",
    )
    parser.add_argument("--title", default="Nano Domain Surface", help="HTML plot title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domain_path = Path(args.domain)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(domain_path, allow_pickle=False)
    height_um = np.asarray(data["height_um"], dtype=np.float32)
    pixel_x = float(data["pixel_size_um_x"])
    pixel_y = float(data["pixel_size_um_y"])
    stride = max(1, int(args.stride))

    height_ds = height_um[::stride, ::stride]
    height_vis = height_ds * float(args.z_exaggeration)
    y = np.arange(height_ds.shape[0], dtype=np.float32) * pixel_y * stride
    x = np.arange(height_ds.shape[1], dtype=np.float32) * pixel_x * stride

    html_path = out_dir / "surface_3d.html"
    ply_path = out_dir / "surface_mesh.ply"
    metadata_path = out_dir / "surface_3d_metadata.json"

    _write_plotly_html(
        html_path=html_path,
        title=args.title,
        x=x,
        y=y,
        height_um=height_ds,
        height_vis=height_vis,
        z_exaggeration=float(args.z_exaggeration),
    )
    _write_mesh_ply(ply_path, x=x, y=y, z=height_ds)

    metadata = {
        "source_domain": str(domain_path),
        "height_shape": list(height_um.shape),
        "export_shape": list(height_ds.shape),
        "stride": stride,
        "pixel_size_um_x": pixel_x,
        "pixel_size_um_y": pixel_y,
        "width_um": float((height_um.shape[1] - 1) * pixel_x),
        "height_um_y_extent": float((height_um.shape[0] - 1) * pixel_y),
        "z_min_um": float(np.nanmin(height_um)),
        "z_max_um": float(np.nanmax(height_um)),
        "z_exaggeration": float(args.z_exaggeration),
        "html": str(html_path),
        "mesh_ply": str(ply_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


def _write_plotly_html(
    html_path: Path,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    height_um: np.ndarray,
    height_vis: np.ndarray,
    z_exaggeration: float,
) -> None:
    x_grid, y_grid = np.meshgrid(x, y)
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=height_vis,
                surfacecolor=height_um,
                colorscale="Viridis",
                colorbar={"title": "height (um)"},
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": "white",
                        "project_z": True,
                    }
                },
            )
        ]
    )
    z_title = "height (um)" if z_exaggeration == 1.0 else f"height x{z_exaggeration:g} (um)"
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x (um)",
            "yaxis_title": "y (um)",
            "zaxis_title": z_title,
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.35, "y": -1.65, "z": 1.05}},
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)


def _write_mesh_ply(ply_path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    rows, cols = z.shape
    x_grid, y_grid = np.meshgrid(x, y)
    vertices = np.column_stack([x_grid.ravel(), y_grid.ravel(), z.ravel()])
    faces = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col
            b = a + 1
            c = a + cols
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.export(ply_path)


if __name__ == "__main__":
    raise SystemExit(main())
