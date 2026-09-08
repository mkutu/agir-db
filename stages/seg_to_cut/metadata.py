"""Reproducible measurements of cleaned masks and original RGB crops.

These helpers return JSON-compatible cutout_props fields. The processor/writer
integration and run-report emission are handled by later implementation steps.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import skimage
from numpy.typing import NDArray
from skimage.measure import blur_effect
from skimage.morphology import convex_hull_image

from .config import SegToCutConfig, parse_config
from .contracts import PixelBoundingBox

SIDES = ("top", "bottom", "left", "right")
BLUR_H_SIZE = 11


def measurement_provenance(config: SegToCutConfig) -> dict[str, Any]:
    """Return the measurement contract to include in a future run report."""
    return {
        "cutout_version": config.cutout_version,
        "libraries": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
        },
        "rgb_statistics": {
            "pixel_selection": "whole_unmasked_crop",
            "channel_order": "RGB",
            "scale_divisor": 255,
            "std_ddof": 0,
            "input": "before_jpeg_encoding",
        },
        "blur_effect": {
            "algorithm": "skimage.measure.blur_effect",
            "pixel_selection": "whole_unmasked_crop",
            "input": "before_jpeg_encoding",
            "h_size": BLUR_H_SIZE,
            "channel_axis": -1,
            "reduce_func": "max",
            "higher_is_blurrier": True,
            "undefined": None,
        },
        "edge_cut": {
            "band_width_px": config.border_width_px,
            "threshold": config.edge_threshold,
            "fraction": "foreground_pixels / actual_band_pixels",
            "comparison": "strictly_greater",
        },
        "solidity": {
            "algorithm": "skimage.morphology.convex_hull_image",
            "offset_coordinates": True,
            "include_borders": True,
            "scope": "all_cleaned_foreground",
        },
        "component_connectivity": 8,
    }


def calculate_crop_properties(rgb_crop: NDArray[np.uint8]) -> dict[str, Any]:
    """Measure every RGB pixel before masking or JPEG encoding.

    Callers loading through OpenCV must convert BGR to RGB before calling.
    Blur is undefined for spatial dimensions under three pixels or any non-finite
    directional score; JSON receives null.
    """
    if (
        not isinstance(rgb_crop, np.ndarray)
        or rgb_crop.dtype != np.uint8
        or rgb_crop.ndim != 3
        or rgb_crop.shape[2] != 3
        or min(rgb_crop.shape[:2]) == 0
    ):
        raise ValueError("rgb_crop must be a non-empty HxWx3 uint8 RGB array")

    normalized = rgb_crop.astype(np.float64) / 255.0
    blur = None
    if min(rgb_crop.shape[:2]) >= 3:
        # Inspect both directional values so max cannot hide a NaN depending on
        # axis order. NumPy suppresses only the expected undefined divisions.
        with np.errstate(divide="ignore", invalid="ignore"):
            directional = blur_effect(
                rgb_crop, channel_axis=-1, h_size=BLUR_H_SIZE, reduce_func=None
            )
        if np.all(np.isfinite(directional)):
            blur = float(np.max(directional))
    return {
        "cropout_rgb_mean": normalized.mean(axis=(0, 1)).tolist(),
        "cropout_rgb_std": normalized.std(axis=(0, 1), ddof=0).tolist(),
        "blur_effect": blur,
    }


def calculate_mask_properties(
    cleaned_mask: NDArray[np.uint8],
    *,
    expected_class_id: int,
    pixel_bbox: PixelBoundingBox,
    image_width: int,
    image_height: int,
    config: SegToCutConfig,
) -> dict[str, Any]:
    """Measure a non-empty class-coded mask after border-sweep cleanup.

    Each side uses its own band, clipped to the corresponding crop dimension.
    Bands can overlap; occupancy uses the actual number of pixels in each band.
    Solidity counts pixels in a single raster hull across all retained components.
    """
    # Validate direct dataclass callers as well as YAML-loaded configurations.
    parse_config(vars(config))
    if (
        not isinstance(cleaned_mask, np.ndarray)
        or cleaned_mask.dtype != np.uint8
        or cleaned_mask.ndim != 2
        or cleaned_mask.size == 0
    ):
        raise ValueError("cleaned_mask must be a non-empty 2D uint8 array")
    if (
        isinstance(expected_class_id, bool)
        or not isinstance(expected_class_id, int)
        or not 1 <= expected_class_id <= 255
    ):
        raise ValueError("expected_class_id must be an integer in 1..255")
    if np.any((cleaned_mask != 0) & (cleaned_mask != expected_class_id)):
        raise ValueError("cleaned_mask contains an unexpected class value")
    target = cleaned_mask == expected_class_id
    if not target.any():
        raise ValueError("empty foreground must be skipped before measuring")
    coordinates = (
        image_width,
        image_height,
        pixel_bbox.xmin,
        pixel_bbox.ymin,
        pixel_bbox.xmax,
        pixel_bbox.ymax,
    )
    if any(isinstance(v, bool) or not isinstance(v, int) for v in coordinates):
        raise ValueError("image dimensions and pixel coordinates must be integers")
    if not (
        0 <= pixel_bbox.xmin < pixel_bbox.xmax <= image_width
        and 0 <= pixel_bbox.ymin < pixel_bbox.ymax <= image_height
        and cleaned_mask.shape == (pixel_bbox.height, pixel_bbox.width)
    ):
        raise ValueError("mask dimensions and clipped source bounding box must agree")

    band = config.border_width_px
    fractions = {
        "top": float(target[:band, :].mean()),
        "bottom": float(target[-band:, :].mean()),
        "left": float(target[:, :band].mean()),
        "right": float(target[:, -band:].mean()),
    }
    flagged = [side for side in SIDES if fractions[side] > config.edge_threshold]
    source_boundary = {
        "top": pixel_bbox.ymin == 0,
        "bottom": pixel_bbox.ymax == image_height,
        "left": pixel_bbox.xmin == 0,
        "right": pixel_bbox.xmax == image_width,
    }
    placement = "unrestricted"
    if len(flagged) == 1:
        placement = f"{flagged[0]}_edge_only"
    elif len(flagged) == 2 and set(flagged) not in ({"top", "bottom"}, {"left", "right"}):
        placement = f"{flagged[0]}_{flagged[1]}_corner_only"
    elif len(flagged) >= 2:
        placement = "unsuitable"

    component_count, _ = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)
    hull = convex_hull_image(target, offset_coordinates=True, include_borders=True)
    return {
        "extends_border": bool(
            target[0, :].any() or target[-1, :].any() or target[:, 0].any() or target[:, -1].any()
        ),
        "edge_cut": {
            "flagged": bool(flagged),
            "threshold": config.edge_threshold,
            "band_width_px": band,
            "flagged_sides": flagged,
            "source_image_sides": [side for side in flagged if source_boundary[side]],
            "detection_box_truncation_sides": [
                side for side in flagged if not source_boundary[side]
            ],
            "synthetic_placement": placement,
            "plant_fraction": fractions,
        },
        "num_components": int(component_count - 1),
        "solidity": float(np.count_nonzero(target) / np.count_nonzero(hull)),
    }


def calculate_cutout_properties(
    rgb_crop: NDArray[np.uint8],
    cleaned_mask: NDArray[np.uint8],
    *,
    expected_class_id: int,
    pixel_bbox: PixelBoundingBox,
    image_width: int,
    image_height: int,
    config: SegToCutConfig,
) -> dict[str, Any]:
    """Combine mask and appearance fields for a validated, cleaned cutout."""
    properties = calculate_mask_properties(
        cleaned_mask,
        expected_class_id=expected_class_id,
        pixel_bbox=pixel_bbox,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    appearance = calculate_crop_properties(rgb_crop)
    if rgb_crop.shape[:2] != cleaned_mask.shape:
        raise ValueError("RGB crop and cleaned mask dimensions must agree")
    return {**properties, **appearance}
