#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import numpy as np
    import PIL
    import torch
    from depth_anything_3.api import DepthAnything3
    from nano_sem_domain.pipeline import run_pipeline

    print(f"numpy={np.__version__}")
    print(f"pillow={PIL.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"mps_available={mps}")
    print(f"DepthAnything3={DepthAnything3.__name__}")
    print(f"pipeline={run_pipeline.__name__}")
    print("env_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
