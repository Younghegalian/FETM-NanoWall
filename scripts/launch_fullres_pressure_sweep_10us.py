from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path("runs/sample_001/fullres_pressure_sweep_100ppm_10us")
    root.mkdir(parents=True, exist_ok=True)
    launcher_log = root / "launcher.log"
    pid_path = root / "sweep.pid"
    cmd = [sys.executable, "scripts/run_fullres_pressure_sweep_10us.py"]
    with launcher_log.open("ab") as log:
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"started pid={process.pid}")
    print(f"pid_file={pid_path}")
    print(f"log={root / 'sweep_run.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
