from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


CASES = (
    ("lambda_0p01", 676443.032930288),
    ("lambda_0p05", 135288.6065860576),
    ("lambda_0p10", 67644.3032930288),
    ("lambda_0p20", 33822.1516465144),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full-resolution pressure sweep.")
    parser.add_argument("--domain", default="runs/sample_001/domain.npz")
    parser.add_argument("--out-root", default="runs/sample_001/fullres_pressure_sweep_100ppm_10us")
    parser.add_argument("--ppm", type=float, default=100e-6)
    parser.add_argument("--n-particle", type=int, default=None, help="Fixed particle count for all pressures.")
    parser.add_argument("--total-time-s", type=float, default=1e-5)
    parser.add_argument("--dt-s", type=float, default=3e-12)
    parser.add_argument("--write-vtk", action="store_true", help="Write per-face VTK hit-count files.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--kernel", default="transport_cpp/particle_hits")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Rerun cases even if their summary exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "sweep_run.log"

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"[config] ppm={args.ppm:.12g} total_time_s={args.total_time_s:.12g} dt_s={args.dt_s:.12g} write_vtk={args.write_vtk}\n")
        for case, pressure_pa in CASES:
            case_dir = out_root / case
            summary_path = case_dir / "particle_hits_summary.json"
            if summary_path.exists() and not args.force:
                log.write(f"[skip] {case}: summary exists\n")
                log.flush()
                continue

            cmd = [
                args.python,
                "-m",
                "nano_transport.run_particle_hits",
                "--domain",
                args.domain,
                "--out-dir",
                str(case_dir),
                "--xy-stride",
                "2",
                "--z-padding-um",
                "0.2",
                "--ppm",
                f"{args.ppm:.12g}",
                "--total-time-s",
                f"{args.total_time_s:.12g}",
                "--dt-s",
                f"{args.dt_s:.12g}",
                "--warmup-steps",
                str(args.warmup_steps),
                "--pressure-pa",
                f"{pressure_pa:.12g}",
                "--kernel",
                args.kernel,
                "--seed",
                str(args.seed),
                "--escape-reinject-mode",
                "boundary",
            ]
            if args.n_particle is not None:
                cmd.extend(["--n-particle", str(args.n_particle)])
            if not args.write_vtk:
                cmd.append("--skip-vtk")
            log.write(f"[run] {case} pressure_pa={pressure_pa:.12g}\n")
            log.flush()
            started = time.perf_counter()
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            log.write(result.stdout)
            log.write(f"\n[done] {case} exit={result.returncode} wall_s={elapsed:.3f}\n")
            log.flush()
            if result.returncode != 0:
                return result.returncode

            write_report(out_root)

        write_report(out_root)
        log.write(f"=== sweep end {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return 0


def write_report(out_root: Path) -> None:
    rows = []
    for case, _pressure_pa in CASES:
        case_dir = out_root / case
        summary_path = case_dir / "particle_hits_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hits_path = case_dir / "face_hits.u64"
        hit_sum = None
        if hits_path.exists():
            hits = np.memmap(hits_path, dtype=np.uint64, mode="r")
            hit_sum = int(hits.sum())
        timing = summary.get("timing_s", {})
        particle_diag = summary.get("particle_diagnostics", {})
        hit_counts = particle_diag.get("hit_counts", {})
        max_burst = particle_diag.get("max_wall_burst_counts", {})
        rows.append(
            {
                "case": case,
                "lambda_um": summary["lambda_um"],
                "pressure_pa": summary["pressure_pa"],
                "ppm": summary["ppm"],
                "n_particle": summary["n_particle"],
                "steps": summary["steps"],
                "dt_s": summary["dt_s"],
                "simulated_time_s": summary["simulated_time_s"],
                "total_hits": summary["total_hits"],
                "total_escapes": summary.get("total_escapes"),
                "total_stuck_resets": summary.get("total_stuck_resets"),
                "total_bg_scatters": summary.get("total_bg_scatters"),
                "collision_rate_s_inv": summary["collision_rate_s_inv"],
                "area_averaged_collision_rate_um2_s_inv": summary["area_averaged_collision_rate_um2_s_inv"],
                "wall_area_um2": summary["wall_area_um2"],
                "wall_surface_face_count": summary.get("wall_surface_face_count"),
                "v_mean_um_s": summary.get("v_mean_um_s"),
                "kernel_wall_s": timing.get("kernel_wall"),
                "particle_steps_per_wall_s": timing.get("particle_steps_per_wall_s"),
                "hit_sum_check": hit_sum,
                "particle_hit_mean": hit_counts.get("mean"),
                "particle_hit_p50": hit_counts.get("p50"),
                "particle_hit_p90": hit_counts.get("p90"),
                "particle_hit_p99": hit_counts.get("p99"),
                "particle_hit_p999": hit_counts.get("p999"),
                "particle_hit_max": hit_counts.get("max"),
                "particle_hit_gini": hit_counts.get("gini"),
                "particle_hit_top_1pct_fraction": hit_counts.get("top_1pct_fraction"),
                "particle_hit_top_5pct_fraction": hit_counts.get("top_5pct_fraction"),
                "particle_max_burst_p99": max_burst.get("p99"),
                "particle_max_burst_max": max_burst.get("max"),
            }
        )
    if not rows:
        return
    report_path = out_root / "pressure_sweep_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
