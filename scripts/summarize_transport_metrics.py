#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_PERCENTILES = (5.0, 50.0, 95.0, 99.0)
FIELD_ALIASES = {
    "kinetic_contact_rate_s_inv": ("kinetic_contact_rate_s_inv", "time_weighted_accessibility_s_inv"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize scalar metrics from FETM transport fields.")
    parser.add_argument("--transport", required=True, help="Path to transport_fields.npz.")
    parser.add_argument(
        "--out",
        help="Output JSON path. Defaults to <transport parent>/transport_metrics.json.",
    )
    parser.add_argument(
        "--no-percentiles",
        action="store_true",
        help="Skip percentile metrics for faster summaries on very large domains.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transport_path = Path(args.transport)
    out_path = Path(args.out) if args.out else transport_path.with_name("transport_metrics.json")
    metrics = summarize_transport_metrics(
        transport_path,
        include_percentiles=not bool(args.no_percentiles),
    )
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"metrics": str(out_path), **metrics["summary"]}, indent=2, sort_keys=True))
    return 0


def summarize_transport_metrics(transport_path: Path, *, include_percentiles: bool = True) -> dict[str, Any]:
    data = np.load(transport_path, allow_pickle=False)
    meta = _read_metadata(data)
    mask_solid = np.asarray(data["mask_solid"], dtype=bool)
    mask_void = ~mask_solid
    dx_um = float(meta.get("dx_um", 1.0))
    voxel_volume_um3 = dx_um**3
    nz, ny, nx = mask_solid.shape
    projected_width_um = nx * dx_um
    projected_height_um = ny * dx_um
    projected_area_um2 = projected_width_um * projected_height_um

    n_voxel = int(mask_solid.size)
    n_solid = int(np.count_nonzero(mask_solid))
    n_void = int(np.count_nonzero(mask_void))
    geometry = {
        "shape_zyx": list(mask_solid.shape),
        "dx_um": dx_um,
        "voxel_volume_um3": voxel_volume_um3,
        "projected_width_um": projected_width_um,
        "projected_height_um": projected_height_um,
        "projected_area_um2": projected_area_um2,
        "z_extent_um": nz * dx_um,
        "n_voxel": n_voxel,
        "n_void": n_void,
        "n_solid": n_solid,
        "domain_volume_um3": float(n_voxel * voxel_volume_um3),
        "void_volume_um3": float(n_void * voxel_volume_um3),
        "solid_volume_um3": float(n_solid * voxel_volume_um3),
        "void_fraction": float(n_void / n_voxel) if n_voxel else 0.0,
        "solid_fraction": float(n_solid / n_voxel) if n_voxel else 0.0,
    }

    fields: dict[str, Any] = {}
    for name in (
        "accessibility",
        "kinetic_contact_rate_s_inv",
        "vis_ang",
        "d_min_um",
        "source_scatter_fraction",
        "source_escape_fraction",
        "source_lost_fraction",
        "source_probability_sum",
        "source_conservation_error",
    ):
        source_name = _resolve_field_name(name, data.files)
        if source_name is None:
            continue
        arr = np.asarray(data[source_name], dtype=np.float64)
        field_info: dict[str, Any] = {
            "all": _stats(
                arr,
                voxel_volume_um3,
                projected_area_um2,
                include_percentiles=include_percentiles,
            ),
            "void": _stats(
                arr,
                voxel_volume_um3,
                projected_area_um2,
                mask=mask_void,
                include_percentiles=include_percentiles,
            ),
            "solid": _stats(
                arr,
                voxel_volume_um3,
                projected_area_um2,
                mask=mask_solid,
                include_percentiles=include_percentiles,
            ),
        }
        if name == "d_min_um":
            valid = mask_void & np.isfinite(arr) & (arr >= 0.0)
            field_info["valid_void"] = _stats(
                arr,
                voxel_volume_um3,
                projected_area_um2,
                mask=valid,
                include_percentiles=include_percentiles,
            )
        fields[name] = field_info
        del arr

    scatter_sum = float(meta.get("scatter_sum", fields.get("source_scatter_fraction", {}).get("void", {}).get("mean", 0.0)))
    accessibility_sum = float(
        meta.get(
            "accessibility_sum",
            fields.get("accessibility", {}).get("void", {}).get("mean", 0.0),
        )
    )
    lost_mass = float(meta.get("lost_mass", 0.0))
    escape_mass = float(meta.get("escape_mass", 0.0))
    probability_sum = scatter_sum + accessibility_sum + lost_mass + escape_mass

    summary = {
        "transport": str(transport_path),
        "source_mode": meta.get("source_mode", "unknown"),
        "lambda_um": meta.get("lambda_um"),
        "xy_stride": meta.get("xy_stride"),
        "n_dir": meta.get("n_dir"),
        "source_scatter_fraction_global": scatter_sum,
        "accessibility_global": accessibility_sum,
        "source_escape_fraction_global": escape_mass,
        "source_lost_fraction_global": lost_mass,
        "lost_mass": lost_mass,
        "escape_mass": escape_mass,
        "probability_sum": probability_sum,
        "void_mean_source_scatter_fraction": fields.get("source_scatter_fraction", {}).get("void", {}).get("mean"),
        "void_mean_accessibility": fields.get("accessibility", {}).get("void", {}).get("mean"),
        "void_accessibility_areal_integral_um": fields.get("accessibility", {}).get("void", {}).get("areal_integral_um"),
        "void_mean_kinetic_contact_rate_s_inv": fields.get("kinetic_contact_rate_s_inv", {})
        .get("void", {})
        .get("mean"),
        "void_kinetic_contact_rate_areal_integral_um_s_inv": fields.get(
            "kinetic_contact_rate_s_inv", {}
        )
        .get("void", {})
        .get("areal_integral_um"),
        "void_mean_source_escape_fraction": fields.get("source_escape_fraction", {}).get("void", {}).get("mean"),
        "void_mean_source_lost_fraction": fields.get("source_lost_fraction", {}).get("void", {}).get("mean"),
        "void_mean_angle_fraction": fields.get("vis_ang", {}).get("void", {}).get("mean"),
        "void_angle_fraction_areal_integral_um": fields.get("vis_ang", {}).get("void", {}).get("areal_integral_um"),
        "valid_void_mean_min_wall_distance_um": fields.get("d_min_um", {}).get("valid_void", {}).get("mean"),
        "void_source_probability_sum_min": fields.get("source_probability_sum", {}).get("void", {}).get("min"),
        "void_source_probability_sum_max": fields.get("source_probability_sum", {}).get("void", {}).get("max"),
        "void_source_probability_sum_mean": fields.get("source_probability_sum", {}).get("void", {}).get("mean"),
        "void_source_conservation_error_max": fields.get("source_conservation_error", {}).get("void", {}).get("max"),
        "void_source_conservation_error_mean": fields.get("source_conservation_error", {}).get("void", {}).get("mean"),
    }

    return {
        "summary": summary,
        "geometry": geometry,
        "probability_balance": {
            "scatter_sum": scatter_sum,
            "accessibility_sum": accessibility_sum,
            "lost_mass": lost_mass,
            "escape_mass": escape_mass,
            "probability_sum": probability_sum,
            "probability_error_abs": abs(probability_sum - 1.0),
        },
        "fields": fields,
        "notes": {
            "source_budget_fields": "source_scatter_fraction + accessibility + source_escape_fraction + source_lost_fraction should be approximately 1 for each void source voxel.",
            "volume_weighted_integral_um3": "sum(field * voxel_volume_um3). This is useful for scalar source fields such as accessibility.",
            "areal_integral_um": "sum(field * voxel_volume_um3) / projected_area_um2. This is the preferred thin-film/substrate-normalized integral for scalar fields such as accessibility.",
            "areal_sum_per_um2": "sum(field) / projected_area_um2. This is a diagnostic count-normalized field sum.",
            "accessibility": "Mean direct free-flight surface-arrival probability over sampled directions.",
            "kinetic_contact_rate_s_inv": "Kinetic Contact Rate, KCR = accessibility * v_mean/lambda. This is a rate field, not a normalized probability.",
            "source_probability_sum": "Per-source scatter + surface + escape + lost fractions. Void values should be approximately 1.",
            "source_conservation_error": "Absolute error |source_probability_sum - 1| for each source voxel.",
            "vis_ang": "Fraction of sampled directions that reached a solid surface.",
            "d_min_um": "Minimum ray distance from a void voxel to a solid wall; valid_void excludes negative/no-hit values.",
        },
    }


def _read_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    raw = np.asarray(data["metadata_json"]).ravel()[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _resolve_field_name(requested: str, available: list[str]) -> str | None:
    for candidate in FIELD_ALIASES.get(requested, (requested,)):
        if candidate in available:
            return candidate
    return None


def _stats(
    arr: np.ndarray,
    voxel_volume_um3: float,
    projected_area_um2: float,
    *,
    mask: np.ndarray | None = None,
    include_percentiles: bool,
) -> dict[str, Any]:
    values = arr[mask] if mask is not None else arr.ravel()
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "min": None,
            "max": None,
            "nonzero_count": 0,
            "nonzero_fraction": 0.0,
            "volume_weighted_integral_um3": 0.0,
            "areal_integral_um": 0.0,
            "areal_sum_per_um2": 0.0,
            "mean_per_um3": None,
        }

    total = float(np.sum(values, dtype=np.float64))
    square_sum = float(np.sum(values * values, dtype=np.float64))
    volume_weighted_integral_um3 = total * voxel_volume_um3
    nonzero_count = int(np.count_nonzero(values))
    out: dict[str, Any] = {
        "count": count,
        "sum": total,
        "mean": float(total / count),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "nonzero_count": nonzero_count,
        "nonzero_fraction": float(nonzero_count / count),
        "volume_weighted_integral_um3": float(volume_weighted_integral_um3),
        "areal_integral_um": float(volume_weighted_integral_um3 / projected_area_um2) if projected_area_um2 > 0 else None,
        "areal_sum_per_um2": float(total / projected_area_um2) if projected_area_um2 > 0 else None,
        "mean_per_um3": float(total / (count * voxel_volume_um3)),
        "square_sum": square_sum,
    }
    if total > 0.0 and square_sum > 0.0:
        effective_count = (total * total) / square_sum
        out["self_weighted_mean"] = float(square_sum / total)
        out["effective_active_count"] = float(effective_count)
        out["effective_active_volume_um3"] = float(effective_count * voxel_volume_um3)
        out["effective_active_fraction"] = float(effective_count / count)
        out["density_self_weighted_mean_per_um3"] = float(square_sum / (total * voxel_volume_um3))
    else:
        out["self_weighted_mean"] = None
        out["effective_active_count"] = 0.0
        out["effective_active_volume_um3"] = 0.0
        out["effective_active_fraction"] = 0.0
        out["density_self_weighted_mean_per_um3"] = None
    if include_percentiles:
        percentiles = np.percentile(values, DEFAULT_PERCENTILES)
        for q, value in zip(DEFAULT_PERCENTILES, percentiles):
            out[f"p{int(q):02d}"] = float(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
