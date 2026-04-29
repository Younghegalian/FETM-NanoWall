from __future__ import annotations

from pathlib import Path

import numpy as np

from nano_sem_domain.calibration import compute_pixel_size_um, output_pixel_size_um
from nano_sem_domain.config import DomainConfig, load_domain_config
from nano_sem_domain.da3_bridge import run_depth_anything
from nano_sem_domain.depth_processing import (
    correct_planar_bias,
    depth_to_height_like,
    lock_edge_height,
    scale_height,
)
from nano_sem_domain.image_io import crop_image, load_image_rgb, save_height_preview, save_json, save_rgb


def run_pipeline(
    config: DomainConfig | str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    depth_npy: str | Path | None = None,
) -> dict:
    cfg = load_domain_config(config) if not isinstance(config, DomainConfig) else config
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_image_rgb(image_path)
    source_pixel_size_um = compute_pixel_size_um(source, cfg.calibration)
    cropped = crop_image(source, cfg.crop_px)
    cropped_path = output_dir / "cropped_input.png"
    save_rgb(cropped, cropped_path)

    if depth_npy is not None:
        raw_depth = np.load(depth_npy).astype(np.float32)
        depth_metadata = {"source": str(depth_npy), "mode": "precomputed_npy"}
    else:
        raw_depth, depth_metadata = run_depth_anything(cropped_path, cfg.depth, output_dir)
        depth_metadata["mode"] = "depth_anything_3"

    raw_depth_path = output_dir / "raw_depth.npy"
    np.save(raw_depth_path, raw_depth.astype(np.float32))

    height_like = depth_to_height_like(raw_depth, invert_depth=cfg.depth.invert_depth)
    bias = correct_planar_bias(height_like, cfg.bias)
    scaled = scale_height(bias.corrected, cfg.height)
    edge_locked = lock_edge_height(scaled.height_um, cropped, bias.base_mask, cfg.height)
    pixel_size_x, pixel_size_y = output_pixel_size_um(
        source_pixel_size_um,
        crop_shape_hw=cropped.shape[:2],
        depth_shape_hw=raw_depth.shape,
    )

    height_path = output_dir / "height_um.npy"
    preview_path = output_dir / "height_preview.png"
    domain_path = output_dir / "domain.npz"
    metadata_path = output_dir / "metadata.json"

    np.save(height_path, edge_locked.height_um)
    save_height_preview(edge_locked.height_um, preview_path)

    metadata = {
        "source_image": str(image_path),
        "source_shape_hw": list(source.shape[:2]),
        "crop_px": list(cfg.crop_px) if cfg.crop_px is not None else None,
        "crop_shape_hw": list(cropped.shape[:2]),
        "depth_shape_hw": list(raw_depth.shape),
        "source_pixel_size_um": source_pixel_size_um,
        "pixel_size_um_x": pixel_size_x,
        "pixel_size_um_y": pixel_size_y,
        "height_scale_factor": scaled.scale_factor,
        "height_scale_reference": scaled.scale_reference,
        "target_height_um": cfg.height.target_height_um,
        "edge_lock_enabled": cfg.height.edge_lock_enabled,
        "edge_lock_height_um": edge_locked.edge_height_um,
        "edge_lock_mode": cfg.height.edge_lock_mode,
        "edge_lock_negative_tolerance": cfg.height.edge_lock_negative_tolerance,
        "edge_lock_percentile": cfg.height.edge_lock_percentile,
        "edge_lock_fraction": float(np.mean(edge_locked.edge_mask)),
        "bias_surface_method": cfg.bias.surface_method,
        "bias_surface_degree": cfg.bias.surface_degree,
        "base_lock_enabled": cfg.bias.base_lock_enabled,
        "base_lock_method": cfg.bias.base_lock_method,
        "base_lock_percentile": cfg.bias.base_lock_percentile,
        "bias_plane_coefficients": list(bias.plane_coefficients),
        "depth": depth_metadata,
        "config": cfg.to_dict(),
    }
    save_json(metadata, metadata_path)

    np.savez_compressed(
        domain_path,
        height_um=edge_locked.height_um.astype(np.float32),
        base_mask=bias.base_mask.astype(np.bool_),
        edge_mask=edge_locked.edge_mask.astype(np.bool_),
        edge_score=edge_locked.edge_score.astype(np.float32),
        bias_plane=bias.plane.astype(np.float32),
        raw_depth=raw_depth.astype(np.float32),
        pixel_size_um_x=np.float32(pixel_size_x),
        pixel_size_um_y=np.float32(pixel_size_y),
        metadata_json=np.array([metadata_path.read_text(encoding="utf-8")]),
    )

    return {
        "cropped_input": str(cropped_path),
        "raw_depth": str(raw_depth_path),
        "height_um": str(height_path),
        "height_preview": str(preview_path),
        "domain": str(domain_path),
        "metadata": str(metadata_path),
    }
