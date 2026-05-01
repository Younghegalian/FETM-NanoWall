#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create interactive 3D cross-section transport views.")
    parser.add_argument("--transport", required=True, help="Path to transport_fields.npz.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--clip-low", type=float, default=2.0, help="Lower percentile for color clipping.")
    parser.add_argument("--clip-high", type=float, default=98.0, help="Upper percentile for color clipping.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transport_path = Path(args.transport)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(transport_path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"][0]))
    dx = float(meta["dx_um"])

    fields = _build_fields(data)
    html_3d = out_dir / "transport_3d_slices.html"
    html_stack = out_dir / "transport_z_stack.html"
    _write_orthogonal_slice_viewer(html_3d, fields, data["mask_solid"], dx, args.clip_low, args.clip_high)
    _write_z_stack_viewer(html_stack, fields, dx, args.clip_low, args.clip_high)

    print(json.dumps({"slices_3d": str(html_3d), "z_stack": str(html_stack)}, indent=2))
    return 0


def _build_fields(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    accessibility = data["accessibility"].astype(np.float32)
    vis_ang = data["vis_ang"].astype(np.float32)
    d_min = data["d_min_um"].astype(np.float32)
    source_scatter = data["source_scatter_fraction"].astype(np.float32)
    source_lost = data["source_lost_fraction"].astype(np.float32)
    source_error = data["source_conservation_error"].astype(np.float32)
    solid = data["mask_solid"].astype(bool)
    void = ~solid
    return {
        "source scatter fraction": np.where(void, source_scatter, np.nan),
        "accessibility": np.where(void, accessibility, np.nan),
        "source lost fraction": np.where(void, source_lost, np.nan),
        "source conservation error": np.where(void, source_error, np.nan),
        "angular visibility": np.where(void, vis_ang, np.nan),
        "minimum wall distance (um)": np.where((d_min >= 0) & void, d_min, np.nan),
    }


def _write_orthogonal_slice_viewer(
    html_path: Path,
    fields: dict[str, np.ndarray],
    solid: np.ndarray,
    dx: float,
    clip_low: float,
    clip_high: float,
) -> None:
    nz, ny, nx = solid.shape
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    x = np.arange(nx, dtype=np.float32) * dx
    y = np.arange(ny, dtype=np.float32) * dx
    z = np.arange(nz, dtype=np.float32) * dx
    xx, yy = np.meshgrid(x, y)
    xx_xz, zz_xz = np.meshgrid(x, z)
    yy_yz, zz_yz = np.meshgrid(y, z)

    fig = go.Figure()
    visibility = []
    buttons = []
    trace_count_per_field = 3
    for field_idx, (name, arr) in enumerate(fields.items()):
        cmin, cmax = _color_limits(arr, clip_low, clip_high)
        visible = field_idx == 0
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=np.full_like(xx, z[iz]),
                surfacecolor=arr[iz, :, :],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                name=f"{name} z-slice",
                showscale=visible,
                colorbar={"title": name},
                visible=visible,
                opacity=0.92,
            )
        )
        fig.add_trace(
            go.Surface(
                x=xx_xz,
                y=np.full_like(xx_xz, y[iy]),
                z=zz_xz,
                surfacecolor=arr[:, iy, :],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                name=f"{name} y-slice",
                showscale=False,
                visible=visible,
                opacity=0.92,
            )
        )
        fig.add_trace(
            go.Surface(
                x=np.full_like(yy_yz, x[ix]),
                y=yy_yz,
                z=zz_yz,
                surfacecolor=arr[:, :, ix],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                name=f"{name} x-slice",
                showscale=False,
                visible=visible,
                opacity=0.92,
            )
        )
        vis = [False] * (len(fields) * trace_count_per_field + 1)
        start = field_idx * trace_count_per_field
        for j in range(trace_count_per_field):
            vis[start + j] = True
        vis[-1] = True
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": vis},
                    {"title": f"Transport 3D slices: {name}"},
                ],
            }
        )
        visibility.append(visible)

    top = _solid_top_surface(solid, dx)
    top_y, top_x = np.indices(top.shape)
    fig.add_trace(
        go.Surface(
            x=top_x.astype(np.float32) * dx,
            y=top_y.astype(np.float32) * dx,
            z=top,
            colorscale=[[0, "rgba(160,160,160,0.25)"], [1, "rgba(160,160,160,0.25)"]],
            showscale=False,
            name="solid top",
            opacity=0.22,
            visible=True,
        )
    )
    fig.update_layout(
        title=f"Transport 3D slices: {next(iter(fields))}",
        scene={
            "xaxis_title": "x (um)",
            "yaxis_title": "y (um)",
            "zaxis_title": "z (um)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.35, "y": -1.65, "z": 1.05}},
        },
        updatemenus=[{"buttons": buttons, "x": 0.01, "y": 0.98, "xanchor": "left", "yanchor": "top"}],
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)


def _write_z_stack_viewer(
    html_path: Path,
    fields: dict[str, np.ndarray],
    dx: float,
    clip_low: float,
    clip_high: float,
) -> None:
    name, arr = next(iter(fields.items()))
    nz = arr.shape[0]
    cmin, cmax = _color_limits(arr, clip_low, clip_high)
    frames = []
    for iz in range(nz):
        frames.append(go.Frame(data=[go.Heatmap(z=arr[iz], zmin=cmin, zmax=cmax, colorscale="Viridis")], name=str(iz)))
    fig = go.Figure(
        data=[go.Heatmap(z=arr[0], zmin=cmin, zmax=cmax, colorscale="Viridis", colorbar={"title": name})],
        frames=frames,
    )
    steps = [
        {
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"{i}",
            "method": "animate",
        }
        for i in range(nz)
    ]
    fig.update_layout(
        title=f"Z cross-section stack: {name}",
        xaxis_title="x index",
        yaxis_title="y index",
        yaxis={"scaleanchor": "x"},
        sliders=[{"active": 0, "steps": steps, "currentvalue": {"prefix": f"z slice, dz={dx:.4g} um: "}}],
        margin={"l": 40, "r": 10, "t": 50, "b": 40},
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)


def _solid_top_surface(solid: np.ndarray, dx: float) -> np.ndarray:
    nz, ny, nx = solid.shape
    top_idx = np.zeros((ny, nx), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            filled = np.flatnonzero(solid[:, y, x])
            top_idx[y, x] = float(filled.max()) if filled.size else 0.0
    return top_idx * dx


def _color_limits(arr: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    cmin = float(np.percentile(finite, low))
    cmax = float(np.percentile(finite, high))
    if cmax <= cmin:
        cmax = cmin + 1.0
    return cmin, cmax


if __name__ == "__main__":
    raise SystemExit(main())
