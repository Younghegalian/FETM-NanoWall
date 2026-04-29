from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelDomain:
    mask_solid: np.ndarray
    height_um: np.ndarray
    dx_um: float
    z_max_um: float
    xy_stride: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.mask_solid.shape)


def voxelize_height_domain(
    domain_npz: str,
    xy_stride: int = 8,
    z_padding_um: float = 0.2,
    dx_um: float | None = None,
    z_max_um: float | None = None,
) -> VoxelDomain:
    data = np.load(domain_npz, allow_pickle=False)
    height_full = np.asarray(data["height_um"], dtype=np.float32)
    pixel_x = float(data["pixel_size_um_x"])
    pixel_y = float(data["pixel_size_um_y"])
    if abs(pixel_x - pixel_y) / max(pixel_x, pixel_y) > 1e-6:
        raise ValueError("Only square in-plane pixels are supported in this first transport kernel")

    stride = max(1, int(xy_stride))
    height = _block_max_downsample(height_full, stride)
    dx = float(dx_um) if dx_um is not None else pixel_x * stride
    if dx <= 0:
        raise ValueError("dx_um must be positive")

    height_max = float(np.nanmax(height))
    z_extent = float(z_max_um) if z_max_um is not None else height_max + float(z_padding_um)
    z_extent = max(z_extent, height_max + dx)
    nz = int(np.ceil(z_extent / dx)) + 1
    z_centers = (np.arange(nz, dtype=np.float32) + 0.5) * dx
    mask_solid = z_centers[:, None, None] <= height[None, :, :]
    mask_solid[0, :, :] = True  # flat substrate layer
    return VoxelDomain(
        mask_solid=mask_solid.astype(np.bool_),
        height_um=height.astype(np.float32),
        dx_um=dx,
        z_max_um=float(nz * dx),
        xy_stride=stride,
    )


def _block_max_downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return arr.copy()
    h, w = arr.shape
    out_h = int(np.ceil(h / stride))
    out_w = int(np.ceil(w / stride))
    padded = np.full((out_h * stride, out_w * stride), np.nan, dtype=np.float32)
    padded[:h, :w] = arr
    blocks = padded.reshape(out_h, stride, out_w, stride)
    return np.nanmax(blocks, axis=(1, 3)).astype(np.float32)
