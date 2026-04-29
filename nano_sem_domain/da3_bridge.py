from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from nano_sem_domain.config import DepthConfig


def run_depth_anything(
    image_path: str | Path,
    config: DepthConfig,
    export_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    _configure_macos_runtime()
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "Depth-Anything-3 is not installed. Run `pip install -e tools/Depth-Anything-3` "
            "or pass --depth-npy for an already computed depth map."
        ) from exc

    device = _resolve_device(config.device, torch)
    model = _load_model(DepthAnything3, config.model).to(device)
    da3_export_dir = Path(export_dir) / "da3_debug" if config.export_da3_debug and export_dir else None
    prediction = model.inference(
        image=[str(image_path)],
        process_res=config.process_res,
        process_res_method=config.process_res_method,
        export_dir=str(da3_export_dir) if da3_export_dir else None,
        export_format="mini_npz-depth_vis" if da3_export_dir else "mini_npz",
    )
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    metadata = {
        "model": config.model,
        "device": str(device),
        "process_res": config.process_res,
        "process_res_method": config.process_res_method,
        "depth_shape": list(depth.shape),
    }
    if hasattr(prediction, "conf"):
        metadata["has_confidence"] = True
    return depth, metadata


def _load_model(depth_anything_cls, model: str):
    if "/" in model or Path(model).exists():
        return depth_anything_cls.from_pretrained(model)
    return depth_anything_cls(model_name=model)


def _resolve_device(device: str, torch_module):
    if device != "auto":
        return torch_module.device(device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _configure_macos_runtime() -> None:
    # Local macOS wheels can load multiple OpenMP runtimes through torch/open3d/sklearn.
    # This keeps DA3 import usable in the project venv. Prefer a Linux/CUDA box for
    # production-scale inference.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
