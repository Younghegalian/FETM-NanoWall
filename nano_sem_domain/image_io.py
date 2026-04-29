from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from nano_sem_domain.config import Crop


def load_image_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def crop_image(image: np.ndarray, crop_px: Crop | None) -> np.ndarray:
    if crop_px is None:
        return image.copy()
    x, y, width, height = crop_px
    if width <= 0 or height <= 0:
        raise ValueError("crop width and height must be positive")
    h, w = image.shape[:2]
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    if x < 0 or y < 0 or x >= w or y >= h or x2 <= x or y2 <= y:
        raise ValueError(f"crop {crop_px} is outside image size {(w, h)}")
    return image[y:y2, x:x2].copy()


def save_rgb(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_height_preview(height_um: np.ndarray, path: str | Path) -> None:
    arr = np.asarray(height_um, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        scaled = np.zeros(arr.shape, dtype=np.uint8)
    else:
        vmin = float(np.nanpercentile(arr[finite], 1.0))
        vmax = float(np.nanpercentile(arr[finite], 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
        scaled = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    rgb = np.stack([scaled, scaled, scaled], axis=-1)
    save_rgb(rgb, path)
