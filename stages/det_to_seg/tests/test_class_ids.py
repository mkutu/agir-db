"""Tests for det_to_seg's class_ids -- CSV row parsing edge cases."""

import csv

import pytest

from stages.det_to_seg.class_ids import (
    ClassIdResolutionError,
    build_class_id_index,
)

CATALOG = {
    "species": {
        "GLMA4": {"class_id": 17},
        "AMPA": {"class_id": 28},
    }
}


def _write_georeferenced_csv(path, rows):
    fieldnames = ["image_id", "bounding_box_id", "species_id", "cultivar_id"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_class_id_index_skips_zero_detection_placeholder_row(tmp_path):
    # A zero-detection placeholder row (bounding_box_id=="" -- see
    # jpg_to_det's export_predictions and species.NO_DETECTIONS_METHOD) has
    # no real bounding box to index and no species_id in the catalog --
    # must be skipped, not raise, or it would take down the whole batch.
    csv_path = tmp_path / "batch_georeferenced.csv"
    _write_georeferenced_csv(csv_path, [
        {"image_id": "NC_1", "bounding_box_id": "0", "species_id": "GLMA4", "cultivar_id": ""},
        {"image_id": "NC_2", "bounding_box_id": "", "species_id": "", "cultivar_id": ""},
    ])

    index = build_class_id_index(csv_path, CATALOG)

    assert index.get("NC_1", 0) == 17
    assert index.get("NC_2", 0) is None  # never indexed, and never looked up for real


def test_build_class_id_index_raises_on_real_row_missing_species(tmp_path):
    # A real (non-placeholder) row with an unresolvable species_id must
    # still raise -- only the zero-detection placeholder marker is skipped.
    csv_path = tmp_path / "batch_georeferenced.csv"
    _write_georeferenced_csv(csv_path, [
        {"image_id": "NC_1", "bounding_box_id": "0", "species_id": "UNKNOWN_CODE", "cultivar_id": ""},
    ])

    with pytest.raises(ClassIdResolutionError):
        build_class_id_index(csv_path, CATALOG)
