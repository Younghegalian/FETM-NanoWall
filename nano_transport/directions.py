from __future__ import annotations

import numpy as np


def fibonacci_sphere(n_dir: int) -> np.ndarray:
    if n_dir <= 0:
        raise ValueError("n_dir must be positive")
    idx = np.arange(n_dir, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * (idx + 0.5) / n_dir)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * idx
    dirs = np.column_stack([np.cos(theta) * radius, np.sin(theta) * radius, z])
    return dirs.astype(np.float32)
