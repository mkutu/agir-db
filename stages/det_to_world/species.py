"""
Species assignment logic (folded in from the former assign_species stage).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Detections farther than this from every zone polygon (in the shapefile's
# own CRS units — meters, for the UTM shapefiles this pipeline uses) are
# tagged UNKNOWN_SPECIES_ID/UNKNOWN_CLASS_ID instead of taking the nearest
# polygon's species. A normal boundary case (point just outside a polygon
# edge due to rounding) lands within a meter or two; anything past this is
# more likely a wrong shapefile, a georeferencing problem, or a real survey
# gap — logged as a warning (see assign_spatial()) since it no longer fails
# the batch.
DEFAULT_MAX_NEAREST_DISTANCE_M = 5.0


class SpeciesAssignmentError(Exception):
    """Base class for species-assignment failures that should fail the batch."""


class UnknownSpeciesCodeError(SpeciesAssignmentError):
    """Raised when a species_id/cultivar_id assigned from the shapefile has no
    matching entry in the species catalog."""


# jpg_to_det's YOLO detector emits classname="color_checker" (class id 1)
# for the physical color-reference card alongside classname="plant" (class
# id 0) for real detections — see stages/jpg_to_det/detector.py. Zone
# shapefiles have no polygon for a color-checker card, so running it through
# the normal spatial join just assigns whatever species zone the card
# physically happens to sit in (and can even get tagged UNKNOWN_* below if
# it lands outside every zone). The species catalog already carries a
# dedicated entry for exactly this case
# (species_catalog.generated.json -> species.COLORCHECKER, class_id 28), so
# color-checker rows are assigned that fixed identity directly instead of
# ever going through the shapefile.
COLOR_CHECKER_CLASSNAME = "color_checker"
COLOR_CHECKER_SPECIES_ID = "COLORCHECKER"
COLOR_CHECKER_CLASS_ID = "28"
COLOR_CHECKER_ASSIGNMENT_METHOD = "color_checker_class"

# Real (non-color-checker) detections that can't be given a real species —
# either because remap_rows() never got world coordinates for them at all
# (no ASFM grid for the image, or every bbox corner missed the grid surface
# even after nudging — see stages/det_to_world/remapper.py), or because
# they *did* georeference but landed farther than max_nearest_distance_m
# from every zone polygon (see assign_spatial() below). Both used to either
# vanish from the output CSV entirely or fail the whole batch; now they're
# kept and tagged with the species catalog's dedicated "unknown plant"
# entry (species_catalog.generated.json -> species.PLANT, class_id 27)
# instead.
UNKNOWN_SPECIES_ID = "PLANT"
UNKNOWN_CLASS_ID = "27"
UNKNOWN_NOT_GEOREFERENCED_METHOD = "unknown_not_georeferenced"
UNKNOWN_TOO_FAR_METHOD = "unknown_too_far"

# jpg_to_det's export_predictions() (and backfill_legacy_metadata.py's own
# conversion path) emit one placeholder row per zero-detection image --
# bounding_box_id=="" -- so every image is represented in the georeferenced
# CSV even with nothing detected. Unlike UNKNOWN_SPECIES_ID above, this is
# deliberately not a species assignment at all (species_id left unset/None,
# not "PLANT"): there's no real detection here to call an unidentified
# plant. Never participates in the spatial join or the monoculture code.
NO_DETECTIONS_METHOD = "no_detections"


# Optional per-zone attributes, joined in and renamed only when the
# shapefile's own columns actually have them — a shapefile missing any of
# these produces no corresponding output column, same as before any of
# them existed. comm_name and cultc_id/disp_name are mutually exclusive in
# practice (ordinary species zones vs. cultivar seasons, where every zone
# is the same species but a different cultivar); class_id has shown up in
# every zone shapefile checked so far but isn't assumed universal.
_OPTIONAL_ZONE_ATTRS = {
    "comm_name": "species_name",
    "cultc_id": "cultivar_id",
    "disp_name": "cultivar_name",
    "class_id": "class_id",
}


def _assign_fixed_identity(
    rows: pd.DataFrame,
    present_attrs: list[str],
    species_id: str,
    class_id: str,
    assignment_method: str,
    species_name: str,
) -> pd.DataFrame:
    """Stamp fixed species identity onto non-spatial rows.

    Add class_id and species_name only when their zone attributes are present.
    """
    rows = rows.copy()
    rows["species_id"] = species_id
    rows["assignment_method"] = assignment_method
    if "class_id" in present_attrs:
        rows["class_id"] = class_id
    if "comm_name" in present_attrs:
        rows["species_name"] = species_name
    return rows


def assign_spatial(
    dets: pd.DataFrame,
    shapefile: str,
    max_nearest_distance_m: float = DEFAULT_MAX_NEAREST_DISTANCE_M,
) -> pd.DataFrame:
    """Assign species to detections via spatial join against a zone shapefile.

    See _OPTIONAL_ZONE_ATTRS for the extra per-zone columns carried through
    when present in the shapefile (species_name, cultivar_id/cultivar_name,
    class_id).

    Detections outside every zone polygon fall back to the nearest one, but
    only within max_nearest_distance_m — see DEFAULT_MAX_NEAREST_DISTANCE_M.
    Farther than that, the detection is tagged UNKNOWN_SPECIES_ID/
    UNKNOWN_CLASS_ID (assignment_method=UNKNOWN_TOO_FAR_METHOD) instead of
    failing the batch.

    Color-checker rows (classname == COLOR_CHECKER_CLASSNAME) never
    participate in the join — see the COLOR_CHECKER_* constants above.
    Rows with no world coordinates at all (remap_rows() couldn't
    georeference them — no grid for the image, or every bbox corner missed
    the grid surface) are likewise excluded and tagged
    UNKNOWN_NOT_GEOREFERENCED_METHOD. Zero-detection placeholder rows
    (bounding_box_id=="") are excluded before either of those checks and
    tagged NO_DETECTIONS_METHOD instead — see that constant above.
    """
    zones = gpd.read_file(shapefile)
    present_attrs = [column for column in _OPTIONAL_ZONE_ATTRS if column in zones.columns]

    if "bounding_box_id" in dets.columns:
        placeholder_mask = dets["bounding_box_id"] == ""
    else:
        placeholder_mask = pd.Series(False, index=dets.index)
    placeholders = _assign_fixed_identity(
        dets[placeholder_mask], present_attrs,
        None, None, NO_DETECTIONS_METHOD, None,
    )
    dets = dets[~placeholder_mask]

    if "classname" in dets.columns:
        color_checker_mask = dets["classname"] == COLOR_CHECKER_CLASSNAME
    else:
        color_checker_mask = pd.Series(False, index=dets.index)
    color_checkers = _assign_fixed_identity(
        dets[color_checker_mask], present_attrs,
        COLOR_CHECKER_SPECIES_ID, COLOR_CHECKER_CLASS_ID, COLOR_CHECKER_ASSIGNMENT_METHOD, "colorchecker",
    )
    dets = dets[~color_checker_mask]

    if "world_centroid_x" in dets.columns:
        ungeoreferenced_mask = dets["world_centroid_x"].isna()
    else:
        ungeoreferenced_mask = pd.Series(True, index=dets.index)
    ungeoreferenced = _assign_fixed_identity(
        dets[ungeoreferenced_mask], present_attrs,
        UNKNOWN_SPECIES_ID, UNKNOWN_CLASS_ID, UNKNOWN_NOT_GEOREFERENCED_METHOD, "unknown",
    )
    dets = dets[~ungeoreferenced_mask]

    excluded = [frame for frame in (placeholders, color_checkers, ungeoreferenced) if not frame.empty]

    if dets.empty:
        return pd.concat(excluded).sort_index() if excluded else dets

    gdf = gpd.GeoDataFrame(
        dets,
        geometry=gpd.points_from_xy(dets.world_centroid_x, dets.world_centroid_y),
        crs=dets["crs"].iloc[0],
    ).to_crs(zones.crs)

    zone_columns = ["species", "geometry"] + present_attrs

    # perform a spatial join... "does this point fall inside this polygon?"
    joined = gpd.sjoin(gdf, zones[zone_columns], how="left", predicate="within")
    joined["assignment_method"] = "spatial_join"

    # if any unmatched detections, assign nearest polygon's species (and
    # whatever optional attrs are present) as fallback, within
    # max_nearest_distance_m — anything farther gets tagged UNKNOWN below.
    too_far_index = pd.Index([])
    unmatched = joined["species"].isna()
    if unmatched.any():
        nearest = gpd.sjoin_nearest(
            gdf[unmatched], zones[zone_columns], how="left", distance_col="nearest_distance_m"
        )
        # sjoin_nearest returns one row per tied nearest neighbor (e.g. a point
        # equidistant from two zones), so it can return more rows than inputs —
        # keep only the first match per point so the assignment aligns 1:1.
        nearest = nearest[~nearest.index.duplicated(keep="first")]

        too_far = nearest[nearest["nearest_distance_m"] > max_nearest_distance_m]
        within_threshold = nearest[nearest["nearest_distance_m"] <= max_nearest_distance_m]
        too_far_index = too_far.index

        if not too_far.empty:
            offenders = ", ".join(
                f"{row.image_id}:{row.bounding_box_id} ({row.nearest_distance_m:.2f}m)"
                for row in too_far.itertuples()
            )
            logger.warning(
                "%d detection(s) fall more than %sm outside every zone polygon in %s; "
                "assigned %s (class_id %s) instead of failing the batch: %s",
                len(too_far), max_nearest_distance_m, shapefile,
                UNKNOWN_SPECIES_ID, UNKNOWN_CLASS_ID, offenders,
            )

        joined.loc[within_threshold.index, "species"] = within_threshold["species"]
        joined.loc[within_threshold.index, "assignment_method"] = "nearest_polygon"
        for column in present_attrs:
            joined.loc[within_threshold.index, column] = within_threshold[column]

    rename_map = {"species": "species_id", **{c: _OPTIONAL_ZONE_ATTRS[c] for c in present_attrs}}
    result = joined.rename(columns=rename_map)

    if len(too_far_index) > 0:
        result.loc[too_far_index, "species_id"] = UNKNOWN_SPECIES_ID
        result.loc[too_far_index, "assignment_method"] = UNKNOWN_TOO_FAR_METHOD
        if "class_id" in present_attrs:
            result.loc[too_far_index, "class_id"] = UNKNOWN_CLASS_ID
        if "comm_name" in present_attrs:
            result.loc[too_far_index, "species_name"] = "unknown"

    if not excluded:
        return result
    # Restore original row order — sjoin preserves gdf's (already
    # color-checker/ungeoreferenced-filtered) index, so re-interleaving by
    # index puts excluded rows back where they were in the input.
    return pd.concat([result] + excluded).sort_index()


def assign_monoculture(dets: pd.DataFrame, species_code: str) -> pd.DataFrame:
    """Assign a single species code to all detections (monoculture batch, no spatial join).

    Color-checker rows (classname == COLOR_CHECKER_CLASSNAME) are excluded from
    the monoculture species code and get the fixed COLOR_CHECKER_* identity
    instead — see the COLOR_CHECKER_* constants above. Zero-detection
    placeholder rows (bounding_box_id=="") are likewise excluded and tagged
    NO_DETECTIONS_METHOD instead of the monoculture code.
    """
    dets = dets.copy()
    dets["species_id"] = species_code
    dets["assignment_method"] = "monoculture_config"
    if "classname" in dets.columns:
        color_checker_mask = dets["classname"] == COLOR_CHECKER_CLASSNAME
        dets.loc[color_checker_mask, "species_id"] = COLOR_CHECKER_SPECIES_ID
        dets.loc[color_checker_mask, "assignment_method"] = COLOR_CHECKER_ASSIGNMENT_METHOD
    if "bounding_box_id" in dets.columns:
        placeholder_mask = dets["bounding_box_id"] == ""
        dets.loc[placeholder_mask, "species_id"] = None
        dets.loc[placeholder_mask, "assignment_method"] = NO_DETECTIONS_METHOD
    return dets


# species_catalog.generated.json fields to carry onto every row, renamed with
# a species_/cultivar_ prefix so they can't collide with the shapefile-sourced
# columns above (species_name, cultivar_id, cultivar_name, class_id) — those
# stay as the season's own human-curated labels; these are the canonical
# reference-data columns, namespaced so a consumer can tell the two apart and
# pick whichever it wants (mirrors the existing cultivar-over-species
# fallback in scripts/job/visualize.py's _detection_name_label).
_SPECIES_CATALOG_FIELDS = {
    "common_name": "species_common_name",
    "family": "species_family",
    "genus": "species_genus",
    "growth_habit": "species_growth_habit",
    "category": "species_category",
    "hex": "species_hex",
    "r": "species_r",
    "g": "species_g",
    "b": "species_b",
}
_CULTIVAR_CATALOG_FIELDS = {
    "display_name": "cultivar_display_name",
    "line_name": "cultivar_line_name",
    "registered": "cultivar_registered",
    "hex": "cultivar_hex",
    "r": "cultivar_r",
    "g": "cultivar_g",
    "b": "cultivar_b",
}


def load_catalog(path: str | Path) -> Dict[str, Any]:
    """Load species_catalog.generated.json (see orchestrator/species_catalog.py).

    Plain file read — never opens a DB connection. Stages must stay DB-free;
    the DB-derived flat file is the only reference data they consume.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_with_catalog(dets: pd.DataFrame, catalog: Dict[str, Any]) -> pd.DataFrame:
    """Attach species/cultivar reference columns (see _SPECIES_CATALOG_FIELDS /
    _CULTIVAR_CATALOG_FIELDS) by looking up species_id (USDA_symbol) and, when
    present, cultivar_id (cultivar_class_id) in *catalog*.

    A species_id/cultivar_id with no catalog entry (e.g. a shapefile code that
    predates the current reference data, or a zone shapefile out of sync with
    the reference tables) means the zone data and the catalog have drifted
    apart — that needs a person to reconcile, so this raises
    UnknownSpeciesCodeError rather than silently blanking the enrichment
    columns.
    """
    dets = dets.copy()
    species_catalog = catalog.get("species", {})
    cultivar_catalog = catalog.get("cultivars", {})

    unmatched_species = set(dets["species_id"].dropna().unique()) - set(species_catalog)
    if unmatched_species:
        message = f"species_catalog has no entry for species_id(s): {sorted(unmatched_species)}"
        logger.error(message)
        raise UnknownSpeciesCodeError(message)

    for src_field, out_col in _SPECIES_CATALOG_FIELDS.items():
        dets[out_col] = dets["species_id"].map(
            lambda species_id: species_catalog.get(species_id, {}).get(src_field)
        )

    if "cultivar_id" in dets.columns:
        present_cultivar_ids = set(dets["cultivar_id"].dropna().unique())
        unmatched_cultivars = present_cultivar_ids - set(cultivar_catalog)
        if unmatched_cultivars:
            message = f"species_catalog has no entry for cultivar_id(s): {sorted(unmatched_cultivars)}"
            logger.error(message)
            raise UnknownSpeciesCodeError(message)
        for src_field, out_col in _CULTIVAR_CATALOG_FIELDS.items():
            dets[out_col] = dets["cultivar_id"].map(
                lambda cultivar_id: (
                    cultivar_catalog.get(cultivar_id, {}).get(src_field)
                    if pd.notna(cultivar_id) else None
                )
            )

    return dets
