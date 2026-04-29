from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


Crop = tuple[int, int, int, int]


@dataclass
class CalibrationConfig:
    pixel_size_um: float | None = None
    ruler_length_um: float | None = None
    ruler_line_px: tuple[float, float, float, float] | None = None
    ruler_bbox_px: Crop | None = None
    ruler_threshold: float = 0.85


@dataclass
class DepthConfig:
    model: str = "depth-anything/DA3MONO-LARGE"
    device: str = "auto"
    process_res: int = 756
    process_res_method: str = "upper_bound_resize"
    invert_depth: bool = False
    export_da3_debug: bool = False


@dataclass
class BiasConfig:
    enabled: bool = True
    surface_method: str = "polynomial"
    surface_degree: int = 1
    base_percentile: float = 12.0
    tile_size_px: int = 64
    min_samples: int = 2000
    max_samples: int = 200000
    mad_sigma: float = 2.5
    iterations: int = 4
    zero_percentile: float = 1.0
    base_lock_enabled: bool = True
    base_lock_method: str = "global"
    base_lock_percentile: float = 20.0
    clip_negative: bool = True


@dataclass
class HeightConfig:
    target_height_um: float | None = 1.7
    scale_percentile: float = 99.5
    cap_height_um: float | None = None
    edge_lock_enabled: bool = False
    edge_lock_height_um: float | None = None
    edge_lock_mode: str = "constant"
    edge_lock_negative_tolerance: float = 0.03
    edge_lock_percentile: float = 82.0
    edge_lock_background_sigma_px: float = 18.0
    edge_lock_close_px: int = 1
    edge_lock_dilate_px: int = 1
    edge_lock_cap_to_height: bool = True


@dataclass
class DomainConfig:
    crop_px: Crop | None = None
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    height: HeightConfig = field(default_factory=HeightConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_or_none(value: Any, length: int, name: str) -> tuple | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} numbers")
    return tuple(value)


def load_domain_config(path: str | Path) -> DomainConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = data.get("calibration", {})
    depth = data.get("depth", {})
    bias = data.get("bias", {})
    height = data.get("height", {})

    return DomainConfig(
        crop_px=_tuple_or_none(data.get("crop_px"), 4, "crop_px"),
        calibration=CalibrationConfig(
            pixel_size_um=calibration.get("pixel_size_um"),
            ruler_length_um=calibration.get("ruler_length_um"),
            ruler_line_px=_tuple_or_none(calibration.get("ruler_line_px"), 4, "ruler_line_px"),
            ruler_bbox_px=_tuple_or_none(calibration.get("ruler_bbox_px"), 4, "ruler_bbox_px"),
            ruler_threshold=float(calibration.get("ruler_threshold", 0.85)),
        ),
        depth=DepthConfig(**depth),
        bias=BiasConfig(**bias),
        height=HeightConfig(**height),
    )
