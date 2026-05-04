from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Knudsen number against accessibility and collision rate.")
    parser.add_argument("--sweep-dir", required=True, help="Particle pressure sweep directory.")
    parser.add_argument(
        "--access-summary",
        default="runs/sample_001/mfp_sweep_stride2_dir256_summary.json",
        help="MFP sweep summary containing areal accessibility values.",
    )
    parser.add_argument(
        "--geometry-metrics",
        default="runs/sample_001/transport_lambda_0p10_stride2_dir256/transport_metrics.json",
        help="Transport metrics JSON containing void volume.",
    )
    parser.add_argument(
        "--mesh-meta",
        default="runs/sample_001/mesh_cache/737db1fd1880bb9242c4d931/surface_mesh_meta.json",
        help="Surface mesh metadata containing wall area.",
    )
    return parser.parse_args()


def read_accessibility(path: Path) -> dict[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {round(float(case["lambda_um"]), 8): case for case in data["cases"]}


def load_characteristic_length(geometry_metrics: Path, mesh_meta: Path) -> float:
    geometry = json.loads(geometry_metrics.read_text(encoding="utf-8"))["geometry"]
    mesh = json.loads(mesh_meta.read_text(encoding="utf-8"))
    void_volume_um3 = float(geometry["void_volume_um3"])
    wall_area_um2 = float(mesh["mesh_builder"]["wall_area_um2"])
    return 4.0 * void_volume_um3 / wall_area_um2


def build_rows(sweep_dir: Path, access_by_lambda: dict[float, dict], lc_um: float) -> list[dict]:
    report_path = sweep_dir / "pressure_sweep_report.csv"
    rows = []
    with report_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            lambda_um = float(row["lambda_um"])
            access = access_by_lambda[round(lambda_um, 8)]
            n_particle = int(row["n_particle"])
            collision_rate = float(row["collision_rate_s_inv"])
            rows.append(
                {
                    "case": row["case"],
                    "lambda_um": lambda_um,
                    "pressure_pa": float(row["pressure_pa"]),
                    "kn_hydraulic_4v_awall": lambda_um / lc_um,
                    "areal_accessibility_um": float(access["void_accessibility_areal_integral_um"]),
                    "accessibility_global": float(access["accessibility_global"]),
                    "n_particle": n_particle,
                    "collision_rate_s_inv": collision_rate,
                    "collision_rate_per_particle_s_inv": collision_rate / n_particle,
                    "hits_per_particle": float(row["total_hits"]) / n_particle,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict]) -> None:
    kn = [row["kn_hydraulic_4v_awall"] for row in rows]
    rate = [row["collision_rate_per_particle_s_inv"] / 1e8 for row in rows]
    access = [row["areal_accessibility_um"] for row in rows]
    lambdas = [row["lambda_um"] for row in rows]

    fig, ax_rate = plt.subplots(figsize=(6.8, 4.4), dpi=180)
    ax_rate.plot(kn, rate, "o-", color="#176B87", linewidth=2.0, label="collision per particle")
    ax_rate.set_xlabel("Kn = lambda / (4 V_void / A_wall)")
    ax_rate.set_ylabel("collision rate per particle (1e8 s^-1)", color="#176B87")
    ax_rate.tick_params(axis="y", labelcolor="#176B87")
    ax_rate.grid(True, alpha=0.25)

    ax_access = ax_rate.twinx()
    ax_access.plot(kn, access, "s--", color="#C47F2C", linewidth=2.0, label="areal accessibility")
    ax_access.set_ylabel("areal accessibility (um)", color="#C47F2C")
    ax_access.tick_params(axis="y", labelcolor="#C47F2C")

    for x, y, lambda_um in zip(kn, rate, lambdas):
        ax_rate.annotate(f"lambda={lambda_um:g}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    access_by_lambda = read_accessibility(Path(args.access_summary))
    lc_um = load_characteristic_length(Path(args.geometry_metrics), Path(args.mesh_meta))
    rows = build_rows(sweep_dir, access_by_lambda, lc_um)
    if not rows:
        raise ValueError(f"No rows found in {sweep_dir / 'pressure_sweep_report.csv'}")
    write_csv(sweep_dir / "knudsen_accessibility_collision.csv", rows)
    plot(sweep_dir / "knudsen_accessibility_collision.png", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
