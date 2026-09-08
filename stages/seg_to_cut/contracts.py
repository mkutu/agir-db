"""Immutable input-validation results used by later cutout processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PixelBoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class WorldBoundingBox:
    """Four ordered bounding-box corners in one projected CRS."""

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]
    crs: str

    @property
    def polygon(self) -> tuple[tuple[float, float], ...]:
        """Return perimeter order for planar polygon-area calculations."""
        return (self.top_left, self.top_right, self.bottom_right, self.bottom_left)


@dataclass(frozen=True)
class DetectionInput:
    image_id: str
    bounding_box_id: int
    normalized_bbox: tuple[float, float, float, float]
    pixel_bbox: PixelBoundingBox
    class_id: int
    species_id: str
    cultivar_id: str | None = None
    world_bbox: WorldBoundingBox | None = None

    @property
    def identity(self) -> tuple[str, int]:
        return (self.image_id, self.bounding_box_id)


@dataclass(frozen=True)
class AreaMetricInput:
    """Per-cutout estimate used to finalize batch/category area metrics."""

    image_id: str
    bounding_box_id: int
    species_id: str
    cultivar_id: str | None
    bbox_area_cm2: float | None

    @property
    def identity(self) -> tuple[str, int]:
        return (self.image_id, self.bounding_box_id)


@dataclass(frozen=True)
class ValidatedImageInput:
    image_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    detections: tuple[DetectionInput, ...]


@dataclass(frozen=True)
class BatchValidationResult:
    images: tuple[ValidatedImageInput, ...]
    known_class_ids: frozenset[int]

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def detection_count(self) -> int:
        return sum(len(image.detections) for image in self.images)


@dataclass(frozen=True)
class CleanupResult:
    """Cleaned local mask and diagnostics for one detection."""

    mask: NDArray[np.uint8]
    removed_components: int
    remaining_components: int
    removed_pixels: int
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None
