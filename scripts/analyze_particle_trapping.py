#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("lambda_0p01", "lambda_0p05", "lambda_0p10", "lambda_0p20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-particle wall-hit trapping diagnostics.")
    parser.add_argument("--sweep-dir", required=True, help="KWFS pressure sweep directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    rows = []
    distributions = []
    for case in CASES:
        case_dir = sweep_dir / case
        summary_path = case_dir / "particle_hits_summary.json"
        hit_path = case_dir / "particle_hit_counts.u32"
        burst_path = case_dir / "particle_max_wall_burst_counts.u32"
        if not summary_path.exists() or not hit_path.exists() or not burst_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n_particle = int(summary["n_particle"])
        hits = _read_u32(hit_path, n_particle)
        bursts = _read_u32(burst_path, n_particle)
        row = {
            "case": case,
            "lambda_um": float(summary["lambda_um"]),
            "n_particle": n_particle,
            "total_hits": int(summary["total_hits"]),
            **_prefixed("hit", summarize_counts(hits, total=int(summary["total_hits"]))),
            **_prefixed("max_burst", summarize_counts(bursts)),
        }
        rows.append(row)
        distributions.append((case, float(summary["lambda_um"]), hits))

    if not rows:
        raise ValueError(f"No particle diagnostics found in {sweep_dir}")

    write_csv(sweep_dir / "particle_trapping_report.csv", rows)
    plot_distribution(sweep_dir / "particle_hit_distribution.png", distributions)
    plot_concentration(sweep_dir / "particle_hit_concentration.png", rows)
    print(json.dumps({"sweep_dir": str(sweep_dir), "rows": rows}, indent=2))
    return 0


def _read_u32(path: Path, expected_size: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.uint32)
    if values.size != expected_size:
        raise ValueError(f"{path} has {values.size} values; expected {expected_size}")
    return values


def summarize_counts(values: np.ndarray, *, total: int | None = None) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(arr)
    count = int(arr.size)
    value_sum = float(arr.sum(dtype=np.float64))
    out: dict[str, float | int | bool] = {
        "sum": value_sum,
        "mean": float(value_sum / count),
        "std": float(np.std(arr)),
        "min": float(sorted_values[0]),
        "p50": float(np.quantile(sorted_values, 0.50)),
        "p90": float(np.quantile(sorted_values, 0.90)),
        "p95": float(np.quantile(sorted_values, 0.95)),
        "p99": float(np.quantile(sorted_values, 0.99)),
        "p999": float(np.quantile(sorted_values, 0.999)),
        "max": float(sorted_values[-1]),
        "zero_fraction": float(np.count_nonzero(arr == 0.0) / count),
        "gini": gini(sorted_values),
    }
    if total is not None:
        out["sum_matches_total"] = int(value_sum) == int(total)
    if value_sum > 0.0:
        top_1 = max(1, int(np.ceil(0.01 * count)))
        top_5 = max(1, int(np.ceil(0.05 * count)))
        top_10 = max(1, int(np.ceil(0.10 * count)))
        out["top_1pct_fraction"] = float(sorted_values[-top_1:].sum(dtype=np.float64) / value_sum)
        out["top_5pct_fraction"] = float(sorted_values[-top_5:].sum(dtype=np.float64) / value_sum)
        out["top_10pct_fraction"] = float(sorted_values[-top_10:].sum(dtype=np.float64) / value_sum)
    else:
        out["top_1pct_fraction"] = 0.0
        out["top_5pct_fraction"] = 0.0
        out["top_10pct_fraction"] = 0.0
    return out


def gini(sorted_values: np.ndarray) -> float:
    values = np.asarray(sorted_values, dtype=np.float64)
    total = float(values.sum(dtype=np.float64))
    n = values.size
    if n == 0 or total <= 0.0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * values, dtype=np.float64)) / (n * total) - (n + 1.0) / n)


def _prefixed(prefix: str, values: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(path: Path, distributions: list[tuple[str, float, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    colors = ("#176B87", "#A34E2D", "#5B7F3A", "#6E5C9A")
    for color, (_case, lambda_um, hits) in zip(colors, distributions):
        sorted_hits = np.sort(hits.astype(np.float64))
        rank = (np.arange(sorted_hits.size, dtype=np.float64) + 1.0) / sorted_hits.size
        normalized = sorted_hits / max(float(np.mean(sorted_hits)), 1.0)
        ax.plot(rank, normalized, color=color, linewidth=2.0, label=rf"$\lambda={lambda_um:g}\ \mu m$")
    ax.set_yscale("log")
    ax.set_xlabel("particle rank fraction")
    ax.set_ylabel("hits per particle / mean")
    ax.set_title("Per-Particle Hit Distribution", loc="left", fontsize=12.0, fontweight="semibold")
    ax.grid(True, which="major", color="#E1E6EA", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#EEF2F4", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8.5)
    _polish_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_concentration(path: Path, rows: list[dict]) -> None:
    lambdas = [row["lambda_um"] for row in rows]
    top1 = [100.0 * float(row["hit_top_1pct_fraction"]) for row in rows]
    gini_pct = [100.0 * float(row["hit_gini"]) for row in rows]
    burst = [float(row["max_burst_p99"]) for row in rows]

    fig, (ax, ax_burst) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08},
    )
    ax.plot(lambdas, top1, "o-", color="#176B87", linewidth=2.2, label="top 1% hit share")
    ax.plot(lambdas, gini_pct, "D-.", color="#3F4D5A", linewidth=1.9, label="hit-count Gini x100")
    ax.set_xscale("log")
    ax.set_ylabel("hit concentration (%)")
    ax.set_title("Particle Trapping Diagnostics", loc="left", fontsize=12.0, fontweight="semibold")
    ax.grid(True, which="major", color="#E1E6EA", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_ylim(0.0, max(max(top1), max(gini_pct)) * 1.25)

    ax_burst.plot(lambdas, burst, "s--", color="#A34E2D", linewidth=2.0, label="p99 wall-hit burst")
    ax_burst.set_xscale("log")
    ax_burst.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax_burst.set_ylabel("burst count")
    ax_burst.grid(True, which="major", color="#E1E6EA", linewidth=0.8)
    ax_burst.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_burst.set_ylim(0.0, max(burst) * 1.2)

    _polish_axes(ax)
    _polish_axes(ax_burst)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.92, bottom=0.12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _polish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8D0D6")
    ax.spines["bottom"].set_color("#C8D0D6")
    ax.tick_params(colors="#2F3A44", labelsize=8.2)


if __name__ == "__main__":
    raise SystemExit(main())
