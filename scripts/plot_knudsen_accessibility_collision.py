from __future__ import annotations

import argparse
import csv
import json
import math
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
        default=None,
        help="Optional fallback surface mesh metadata containing wall area.",
    )
    parser.add_argument(
        "--v-mean-um-s",
        type=float,
        default=370353425.4688162,
        help="Mean molecular speed used for kinetic contact rate.",
    )
    return parser.parse_args()


def read_accessibility(path: Path) -> dict[float, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {round(float(case["lambda_um"]), 8): case for case in data["cases"]}


def load_void_volume(geometry_metrics: Path) -> float:
    geometry = json.loads(geometry_metrics.read_text(encoding="utf-8"))["geometry"]
    return float(geometry["void_volume_um3"])


def mesh_wall_area(mesh_meta: str | None) -> float | None:
    if not mesh_meta:
        return None
    mesh = json.loads(Path(mesh_meta).read_text(encoding="utf-8"))
    return float(mesh["mesh_builder"]["wall_area_um2"])


def build_rows(
    sweep_dir: Path,
    access_by_lambda: dict[float, dict],
    void_volume_um3: float,
    fallback_wall_area_um2: float | None,
    default_v_mean_um_s: float,
) -> list[dict]:
    report_path = sweep_dir / "pressure_sweep_report.csv"
    rows = []
    with report_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            lambda_um = float(row["lambda_um"])
            access = access_by_lambda[round(lambda_um, 8)]
            n_particle = int(row["n_particle"])
            collision_rate = float(row["collision_rate_s_inv"])
            wall_area_um2 = float(row.get("wall_area_um2") or fallback_wall_area_um2)
            lc_um = 4.0 * void_volume_um3 / wall_area_um2
            v_mean_um_s = float(row.get("v_mean_um_s") or default_v_mean_um_s)
            accessibility_global = float(access["accessibility_global"])
            void_mean_kcr_s_inv = v_mean_um_s / lambda_um * accessibility_global
            rows.append(
                {
                    "case": row["case"],
                    "lambda_um": lambda_um,
                    "pressure_pa": float(row["pressure_pa"]),
                    "kn_hydraulic_4v_awall": lambda_um / lc_um,
                    "areal_accessibility_um": float(access["void_accessibility_areal_integral_um"]),
                    "accessibility_global": accessibility_global,
                    "v_mean_um_s": v_mean_um_s,
                    "void_mean_kinetic_contact_rate_s_inv": void_mean_kcr_s_inv,
                    "void_kinetic_contact_time_ns": _contact_time_ns(void_mean_kcr_s_inv),
                    "void_contact_probability_0p1ns": _contact_probability(void_mean_kcr_s_inv, 0.1e-9),
                    "void_contact_probability_1ns": _contact_probability(void_mean_kcr_s_inv, 1.0e-9),
                    "n_particle": n_particle,
                    "collision_rate_s_inv": collision_rate,
                    "collision_rate_per_particle_s_inv": collision_rate / n_particle,
                    "hits_per_particle": float(row["total_hits"]) / n_particle,
                }
            )
    return rows


def _contact_time_ns(rate_s_inv: float) -> float:
    return 1.0e9 / rate_s_inv if rate_s_inv > 0.0 else math.inf


def _contact_probability(rate_s_inv: float, window_s: float) -> float:
    return 1.0 - math.exp(-rate_s_inv * window_s) if rate_s_inv > 0.0 else math.nan


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


def plot_kcr(path: Path, rows: list[dict]) -> None:
    kn = [row["kn_hydraulic_4v_awall"] for row in rows]
    rate = [row["collision_rate_per_particle_s_inv"] / 1e9 for row in rows]
    kcr = [row["void_mean_kinetic_contact_rate_s_inv"] / 1e9 for row in rows]
    ratio = [r / k if k > 0.0 else math.nan for r, k in zip(rate, kcr)]
    lambdas = [row["lambda_um"] for row in rows]

    blue = "#176B87"
    rust = "#A34E2D"
    gray = "#5F6C7B"

    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.5),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.12], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    ax.fill_between(kn, kcr, rate, color="#CFE2EA", alpha=0.55, linewidth=0)
    ax.plot(
        kn,
        rate,
        color=blue,
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.4,
        label="KWFS collision per particle",
    )
    ax.plot(
        kn,
        kcr,
        color=rust,
        marker="s",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.7,
        linewidth=2.2,
        linestyle=(0, (4, 2)),
        label="void-mean KCR",
    )
    ax.set_xscale("log")
    ax.set_ylabel(r"rate ($10^9$ s$^{-1}$)")
    ax.set_title("Void-Mean KCR vs Particle Collision Rate", loc="left", fontsize=12.5, pad=10)
    ax.text(
        0.0,
        1.01,
        "same voxel void domain; KCR averaged over void cells",
        transform=ax.transAxes,
        fontsize=8.5,
        color=gray,
        va="bottom",
    )
    ax.grid(True, which="major", axis="both", color="#E1E6EA", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#EEF2F4", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_ylim(0.0, max(max(rate), max(kcr)) * 1.22)

    for idx, (x, y, lambda_um) in enumerate(zip(kn, rate, lambdas)):
        offset = (5, 8) if idx % 2 == 0 else (5, -14)
        ax.annotate(
            rf"$\lambda={lambda_um:g}\ \mu m$",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.2,
            color=gray,
        )

    ax_ratio.axhline(1.0, color="#A0A8B0", linewidth=1.1, linestyle=(0, (3, 3)))
    ax_ratio.plot(
        kn,
        ratio,
        color="#3F4D5A",
        marker="D",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=1.9,
    )
    ax_ratio.set_ylabel("sim / KCR")
    ax_ratio.set_xlabel(r"$Kn = \lambda / (4V_{\mathrm{void}}/A_{\mathrm{wall}})$")
    ax_ratio.grid(True, which="major", axis="both", color="#E1E6EA", linewidth=0.8)
    ax_ratio.set_ylim(0.8, max(ratio) * 1.16)
    ax_ratio.set_xticks(kn)
    ax_ratio.set_xticklabels([f"{x:.3g}" for x in kn])

    for axis in (ax, ax_ratio):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#C8D0D6")
        axis.spines["bottom"].set_color("#C8D0D6")
        axis.tick_params(colors="#2F3A44", labelsize=8.2)

    fig.subplots_adjust(left=0.095, right=0.975, top=0.9, bottom=0.12, hspace=0.08)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    access_by_lambda = read_accessibility(Path(args.access_summary))
    void_volume_um3 = load_void_volume(Path(args.geometry_metrics))
    rows = build_rows(sweep_dir, access_by_lambda, void_volume_um3, mesh_wall_area(args.mesh_meta), args.v_mean_um_s)
    if not rows:
        raise ValueError(f"No rows found in {sweep_dir / 'pressure_sweep_report.csv'}")
    write_csv(sweep_dir / "knudsen_accessibility_collision.csv", rows)
    plot(sweep_dir / "knudsen_accessibility_collision.png", rows)
    plot_kcr(sweep_dir / "knudsen_kcr_collision.png", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
