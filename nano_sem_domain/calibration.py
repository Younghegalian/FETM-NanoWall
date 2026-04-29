from __future__ import annotations

from math import hypot

import numpy as np

from nano_sem_domain.config import CalibrationConfig, Crop


def compute_pixel_size_um(image_rgb: np.ndarray, config: CalibrationConfig) -> float:
    if config.pixel_size_um is not None:
        if config.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        return float(config.pixel_size_um)

    if config.ruler_length_um is None or config.ruler_length_um <= 0:
        raise ValueError("Provide calibration.pixel_size_um or a positive ruler_length_um")

    if config.ruler_line_px is not None:
        x1, y1, x2, y2 = config.ruler_line_px
        length_px = hypot(x2 - x1, y2 - y1)
        if length_px <= 0:
            raise ValueError("ruler_line_px has zero length")
        return float(config.ruler_length_um) / length_px

    if config.ruler_bbox_px is not None:
        length_px = detect_scale_bar_length_px(
            image_rgb,
            config.ruler_bbox_px,
            config.ruler_threshold,
        )
        return float(config.ruler_length_um) / length_px

    raise ValueError(
        "Calibration needs one of pixel_size_um, ruler_line_px, or ruler_bbox_px"
    )


def detect_scale_bar_length_px(
    image_rgb: np.ndarray,
    bbox_px: Crop,
    threshold_fraction: float = 0.85,
) -> float:
    x, y, width, height = bbox_px
    patch = image_rgb[y : y + height, x : x + width]
    if patch.size == 0:
        raise ValueError(f"ruler_bbox_px {bbox_px} is empty")

    gray = patch.astype(np.float32).mean(axis=2)
    gmin = float(np.min(gray))
    gmax = float(np.max(gray))
    if gmax <= gmin:
        raise ValueError("ruler_bbox_px has no intensity contrast")

    bright_cut = gmin + threshold_fraction * (gmax - gmin)
    dark_cut = gmax - threshold_fraction * (gmax - gmin)
    bright = gray >= bright_cut
    dark = gray <= dark_cut

    bright_run = _scale_bar_candidate_run(bright)
    dark_run = _scale_bar_candidate_run(dark)
    best = max(bright_run, dark_run)
    if best <= 1:
        raise ValueError(
            "Could not detect a scale bar in ruler_bbox_px; use ruler_line_px instead"
        )
    return float(best)


def output_pixel_size_um(
    source_pixel_size_um: float,
    crop_shape_hw: tuple[int, int],
    depth_shape_hw: tuple[int, int],
) -> tuple[float, float]:
    crop_h, crop_w = crop_shape_hw
    depth_h, depth_w = depth_shape_hw
    if depth_h <= 0 or depth_w <= 0:
        raise ValueError("depth shape must be positive")
    return (
        float(source_pixel_size_um) * crop_w / depth_w,
        float(source_pixel_size_um) * crop_h / depth_h,
    )


def _longest_horizontal_run(mask: np.ndarray) -> int:
    best = 0
    for row in mask:
        padded = np.concatenate([[False], row.astype(bool), [False]])
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2]
        if starts.size:
            best = max(best, int(np.max(ends - starts)))
    return best


def _scale_bar_candidate_run(mask: np.ndarray) -> int:
    coverage = float(np.mean(mask))
    if coverage <= 0.0001 or coverage >= 0.45:
        return 0
    return _longest_horizontal_run(mask)
