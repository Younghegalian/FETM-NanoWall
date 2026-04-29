from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from nano_sem_domain.config import BiasConfig, HeightConfig


@dataclass
class BiasCorrectionResult:
    corrected: np.ndarray
    plane: np.ndarray
    base_mask: np.ndarray
    plane_coefficients: tuple[float, float, float]


@dataclass
class HeightScaleResult:
    height_um: np.ndarray
    scale_factor: float
    scale_reference: float


@dataclass
class EdgeLockResult:
    height_um: np.ndarray
    edge_mask: np.ndarray
    edge_score: np.ndarray
    edge_height_um: float | None


def depth_to_height_like(depth: np.ndarray, invert_depth: bool = False) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("depth must be a 2D array")
    height = -arr if invert_depth else arr.copy()
    finite = np.isfinite(height)
    if np.any(finite):
        height = height - np.nanpercentile(height[finite], 1.0)
    return height.astype(np.float32)


def correct_planar_bias(height_like: np.ndarray, config: BiasConfig) -> BiasCorrectionResult:
    arr = np.asarray(height_like, dtype=np.float32)
    if not config.enabled:
        return BiasCorrectionResult(
            corrected=arr.copy(),
            plane=np.zeros_like(arr, dtype=np.float32),
            base_mask=np.isfinite(arr),
            plane_coefficients=(0.0, 0.0, 0.0),
        )

    base_mask = _low_region_mask(arr, config.base_percentile, config.tile_size_px)
    method = config.surface_method.lower()
    if method == "polynomial":
        coef = _fit_surface_robust(
            arr,
            base_mask,
            degree=config.surface_degree,
            min_samples=config.min_samples,
            max_samples=config.max_samples,
            mad_sigma=config.mad_sigma,
            iterations=config.iterations,
        )
        plane = _surface_from_coefficients(arr.shape, coef, degree=config.surface_degree).astype(np.float32)
    elif method == "tile":
        coef = np.array([], dtype=np.float64)
        plane = _tile_base_surface(arr, base_mask, config.tile_size_px).astype(np.float32)
    else:
        raise ValueError("bias.surface_method must be 'polynomial' or 'tile'")
    corrected = arr - plane

    finite_base = base_mask & np.isfinite(corrected)
    if np.any(finite_base):
        corrected = corrected - np.nanpercentile(corrected[finite_base], config.zero_percentile)
    if config.clip_negative:
        corrected = np.maximum(corrected, 0.0)
    if config.base_lock_enabled:
        corrected, base_mask = _lock_base_floor(
            corrected,
            method=config.base_lock_method,
            percentile=config.base_lock_percentile,
            tile_size=config.tile_size_px,
        )
    return BiasCorrectionResult(
        corrected=corrected.astype(np.float32),
        plane=plane,
        base_mask=base_mask,
        plane_coefficients=tuple(float(v) for v in coef),
    )


def scale_height(corrected: np.ndarray, config: HeightConfig) -> HeightScaleResult:
    arr = np.asarray(corrected, dtype=np.float32)
    finite = np.isfinite(arr)
    positive = finite & (arr > 0)
    if not np.any(positive):
        return HeightScaleResult(arr.copy(), 1.0, 0.0)

    reference = float(np.nanpercentile(arr[positive], config.scale_percentile))
    if reference <= 1e-12 or config.target_height_um is None:
        scale = 1.0
    else:
        scale = float(config.target_height_um) / reference

    height_um = arr * scale
    if config.cap_height_um is not None:
        height_um = np.minimum(height_um, float(config.cap_height_um))
    return HeightScaleResult(height_um.astype(np.float32), scale, reference)


def lock_edge_height(
    height_um: np.ndarray,
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    config: HeightConfig,
) -> EdgeLockResult:
    arr = np.asarray(height_um, dtype=np.float32)
    if not config.edge_lock_enabled:
        return EdgeLockResult(
            height_um=arr.copy(),
            edge_mask=np.zeros(arr.shape, dtype=bool),
            edge_score=np.zeros(arr.shape, dtype=np.float32),
            edge_height_um=None,
        )

    target = config.edge_lock_height_um
    if target is None:
        target = config.target_height_um
    if target is None:
        raise ValueError("height.edge_lock_height_um or height.target_height_um is required")
    target = float(target)

    gray = _resized_gray(image_rgb, arr.shape)
    score = _local_brightness_score(gray, config.edge_lock_background_sigma_px)
    candidates = np.isfinite(arr) & ~base_mask
    if not np.any(candidates):
        edge_mask = np.zeros(arr.shape, dtype=bool)
    else:
        cutoff = float(np.nanpercentile(score[candidates], config.edge_lock_percentile))
        edge_mask = candidates & (score >= cutoff)
        edge_mask = _clean_edge_mask(
            edge_mask,
            close_px=config.edge_lock_close_px,
            dilate_px=config.edge_lock_dilate_px,
        )
        edge_mask &= candidates

    locked = arr.copy()
    if config.edge_lock_cap_to_height:
        locked = np.minimum(locked, target)
    if np.any(edge_mask):
        locked[edge_mask] = _edge_locked_values(
            arr[edge_mask],
            target=target,
            mode=config.edge_lock_mode,
            negative_tolerance=config.edge_lock_negative_tolerance,
        )
    locked[base_mask] = 0.0
    return EdgeLockResult(
        height_um=locked.astype(np.float32),
        edge_mask=edge_mask,
        edge_score=score.astype(np.float32),
        edge_height_um=target,
    )


def _low_region_mask(arr: np.ndarray, percentile: float, tile_size: int) -> np.ndarray:
    finite = np.isfinite(arr)
    mask = np.zeros(arr.shape, dtype=bool)
    h, w = arr.shape
    tile = max(8, int(tile_size))
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            sub = arr[y0 : y0 + tile, x0 : x0 + tile]
            sub_finite = np.isfinite(sub)
            if not np.any(sub_finite):
                continue
            cutoff = np.nanpercentile(sub[sub_finite], percentile)
            mask[y0 : y0 + tile, x0 : x0 + tile] |= sub <= cutoff
    if np.count_nonzero(mask) < 10 and np.any(finite):
        cutoff = np.nanpercentile(arr[finite], percentile)
        mask = finite & (arr <= cutoff)
    return mask & finite


def _fit_surface_robust(
    arr: np.ndarray,
    initial_mask: np.ndarray,
    degree: int,
    min_samples: int,
    max_samples: int,
    mad_sigma: float,
    iterations: int,
) -> np.ndarray:
    degree = int(degree)
    if degree < 1 or degree > 4:
        raise ValueError("bias.surface_degree must be between 1 and 4")
    finite = np.isfinite(arr)
    mask = initial_mask & finite
    if np.count_nonzero(mask) < min_samples and np.any(finite):
        cutoff = np.nanpercentile(arr[finite], 20.0)
        mask = finite & (arr <= cutoff)

    y_idx, x_idx = np.nonzero(mask)
    z = arr[y_idx, x_idx].astype(np.float64)
    if z.size < 3:
        return np.array([0.0, 0.0, float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0])

    if z.size > max_samples:
        rng = np.random.default_rng(42)
        pick = rng.choice(z.size, size=max_samples, replace=False)
        y_idx = y_idx[pick]
        x_idx = x_idx[pick]
        z = z[pick]

    x_norm, y_norm = _normalized_xy(arr.shape, x_idx, y_idx)
    design = _polynomial_design(x_norm, y_norm, degree)
    keep = np.ones(z.shape, dtype=bool)
    coef = np.array([0.0, 0.0, float(np.nanmedian(z))])
    for _ in range(max(1, iterations)):
        coef = np.linalg.lstsq(design[keep], z[keep], rcond=None)[0]
        residual = z - design @ coef
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
        sigma = max(1.4826 * mad, 1e-9)
        new_keep = np.abs(residual) <= mad_sigma * sigma
        if np.count_nonzero(new_keep) < 3:
            break
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return coef


def _surface_from_coefficients(
    shape: tuple[int, int],
    coef: np.ndarray,
    degree: int,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    x_norm, y_norm = _normalized_xy(shape, xx.ravel(), yy.ravel())
    surface = _polynomial_design(x_norm, y_norm, degree) @ coef
    return surface.reshape(shape)


def _polynomial_design(x_norm: np.ndarray, y_norm: np.ndarray, degree: int) -> np.ndarray:
    terms = []
    for total_degree in range(1, degree + 1):
        for x_power in range(total_degree, -1, -1):
            y_power = total_degree - x_power
            terms.append((x_norm ** x_power) * (y_norm ** y_power))
    terms.append(np.ones_like(x_norm))
    return np.column_stack(terms)


def _tile_base_surface(
    arr: np.ndarray,
    base_mask: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    h, w = arr.shape
    tile = max(16, int(tile_size))
    rows = int(np.ceil(h / tile))
    cols = int(np.ceil(w / tile))
    control = np.empty((rows, cols), dtype=np.float32)
    finite = np.isfinite(arr)

    global_base = float(np.nanpercentile(arr[finite], 10.0)) if np.any(finite) else 0.0
    for row in range(rows):
        for col in range(cols):
            y0 = row * tile
            x0 = col * tile
            sub = arr[y0 : min(y0 + tile, h), x0 : min(x0 + tile, w)]
            sub_base = base_mask[y0 : min(y0 + tile, h), x0 : min(x0 + tile, w)]
            vals = sub[sub_base & np.isfinite(sub)]
            if vals.size >= 3:
                control[row, col] = float(np.nanmedian(vals))
                continue
            vals = sub[np.isfinite(sub)]
            control[row, col] = float(np.nanpercentile(vals, 10.0)) if vals.size else global_base

    image = Image.fromarray(control, mode="F")
    resized = image.resize((w, h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def _lock_base_floor(
    corrected: np.ndarray,
    method: str,
    percentile: float,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(corrected, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr.copy(), finite
    method = method.lower()
    if method == "global":
        floor = np.full(arr.shape, float(np.nanpercentile(arr[finite], percentile)), dtype=np.float32)
    elif method == "tile":
        floor = _tile_percentile_surface(arr, percentile=percentile, tile_size=tile_size)
    else:
        raise ValueError("bias.base_lock_method must be 'global' or 'tile'")
    locked = np.maximum(arr - floor, 0.0)
    base_mask = finite & (arr <= floor)
    return locked.astype(np.float32), base_mask


def _resized_gray(image_rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").convert("L")
    image = image.resize((shape[1], shape[0]), resample=Image.Resampling.BICUBIC)
    gray = np.asarray(image, dtype=np.float32)
    low = float(np.nanpercentile(gray, 1.0))
    high = float(np.nanpercentile(gray, 99.0))
    if high <= low:
        return np.zeros(shape, dtype=np.float32)
    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def _local_brightness_score(gray: np.ndarray, sigma_px: float) -> np.ndarray:
    sigma = max(0.0, float(sigma_px))
    if sigma <= 0.0:
        score = gray.copy()
    else:
        background = ndimage.gaussian_filter(gray, sigma=sigma)
        score = gray - background
    low = float(np.nanpercentile(score, 1.0))
    high = float(np.nanpercentile(score, 99.0))
    if high <= low:
        return np.zeros(gray.shape, dtype=np.float32)
    return np.clip((score - low) / (high - low), 0.0, 1.0)


def _clean_edge_mask(edge_mask: np.ndarray, close_px: int, dilate_px: int) -> np.ndarray:
    mask = edge_mask.astype(bool)
    if close_px > 0:
        structure = _disk_structure(close_px)
        mask = ndimage.binary_closing(mask, structure=structure)
    if dilate_px > 0:
        structure = _disk_structure(dilate_px)
        mask = ndimage.binary_dilation(mask, structure=structure)
    return mask


def _edge_locked_values(
    values: np.ndarray,
    target: float,
    mode: str,
    negative_tolerance: float,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "constant":
        return np.full(values.shape, target, dtype=np.float32)
    if mode != "max_band":
        raise ValueError("height.edge_lock_mode must be 'constant' or 'max_band'")

    vals = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.full(vals.shape, target, dtype=np.float32)

    low = float(np.nanpercentile(vals[finite], 5.0))
    high = float(np.nanpercentile(vals[finite], 99.5))
    if high <= low:
        normalized = np.ones(vals.shape, dtype=np.float32)
    else:
        normalized = np.clip((vals - low) / (high - low), 0.0, 1.0)

    tol = max(0.0, float(negative_tolerance))
    lower = target * (1.0 - tol)
    locked = lower + normalized * (target - lower)
    return np.clip(locked, lower, target).astype(np.float32)


def _disk_structure(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _tile_percentile_surface(
    arr: np.ndarray,
    percentile: float,
    tile_size: int,
) -> np.ndarray:
    h, w = arr.shape
    tile = max(16, int(tile_size))
    rows = int(np.ceil(h / tile))
    cols = int(np.ceil(w / tile))
    control = np.empty((rows, cols), dtype=np.float32)
    finite = np.isfinite(arr)
    global_floor = float(np.nanpercentile(arr[finite], percentile)) if np.any(finite) else 0.0

    for row in range(rows):
        for col in range(cols):
            y0 = row * tile
            x0 = col * tile
            sub = arr[y0 : min(y0 + tile, h), x0 : min(x0 + tile, w)]
            vals = sub[np.isfinite(sub)]
            control[row, col] = float(np.nanpercentile(vals, percentile)) if vals.size else global_floor

    image = Image.fromarray(control, mode="F")
    resized = image.resize((w, h), resample=Image.Resampling.BICUBIC)
    floor = np.asarray(resized, dtype=np.float32)
    floor = np.maximum(floor, 0.0)
    upper = float(np.nanmax(arr[finite])) if np.any(finite) else global_floor
    return np.minimum(floor, upper)


def _normalized_xy(
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    x_norm = (x.astype(np.float64) / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (y.astype(np.float64) / max(h - 1, 1)) * 2.0 - 1.0
    return x_norm, y_norm
