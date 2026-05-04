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
    parser = argparse.ArgumentParser(description="Run the full-resolution 100 ppm pressure sweep for 10 us.")
    parser.add_argument("--domain", default="runs/sample_001/domain.npz")
    parser.add_argument("--out-root", default="runs/sample_001/fullres_pressure_sweep_100ppm_10us")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--kernel", default="/private/tmp/mesh_particle_hits_3d_test")
    parser.add_argument("--mesh-kernel", default="/private/tmp/height_isosurface_test")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--curve-interval-steps", type=int, default=10000)
    parser.add_argument("--force", action="store_true", help="Rerun cases even if their summary exists.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "sweep_run.log"

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for case, pressure_pa in CASES:
            case_dir = out_root / case
            summary_path = case_dir / "mesh_particle_hits_summary.json"
            if summary_path.exists() and not args.force:
                log.write(f"[skip] {case}: summary exists\n")
                log.flush()
                continue

            cmd = [
                args.python,
                "-m",
                "nano_transport.run_mesh_particle_hits",
                "--domain",
                args.domain,
                "--out-dir",
                str(case_dir),
                "--xy-stride",
                "2",
                "--max-triangle-edge-um",
                "0.05",
                "--z-padding-um",
                "0.2",
                "--ppm",
                "100e-6",
                "--total-time-s",
                "1e-5",
                "--dt-s",
                "3e-12",
                "--curve-interval-steps",
                str(args.curve_interval_steps),
                "--skip-vtk",
                "--pressure-pa",
                f"{pressure_pa:.12g}",
                "--kernel",
                args.kernel,
                "--mesh-kernel",
                args.mesh_kernel,
                "--seed",
                str(args.seed),
            ]
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
        summary_path = case_dir / "mesh_particle_hits_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        hits_path = case_dir / "surface_face_hits.u64"
        hit_sum = None
        if hits_path.exists():
            hits = np.memmap(hits_path, dtype=np.uint64, mode="r")
            hit_sum = int(hits.sum())
        timing = summary.get("timing_s", {})
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
                "collision_rate_s_inv": summary["collision_rate_s_inv"],
                "area_averaged_collision_rate_um2_s_inv": summary["area_averaged_collision_rate_um2_s_inv"],
                "wall_area_um2": summary["wall_area_um2"],
                "mesh_cache_hit": summary["mesh_cache"]["hit"],
                "kernel_wall_s": timing.get("kernel_wall"),
                "particle_steps_per_wall_s": timing.get("particle_steps_per_wall_s"),
                "hit_sum_check": hit_sum,
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
