import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from stages.det_to_world.species import (
    COLOR_CHECKER_CLASS_ID,
    COLOR_CHECKER_SPECIES_ID,
    NO_DETECTIONS_METHOD,
    UNKNOWN_CLASS_ID,
    UNKNOWN_SPECIES_ID,
    UnknownSpeciesCodeError,
    assign_monoculture,
    assign_spatial,
    enrich_with_catalog,
    load_catalog,
)

ZONE_CRS = "EPSG:32617"


@pytest.fixture
def shapefile(tmp_path):
    """Two non-overlapping rectangular zone polygons with distinct species codes."""
    zones = gpd.GeoDataFrame(
        {"species": ["ABUTH", "BETVU"]},
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        crs=ZONE_CRS,
    )
    path = tmp_path / "zones.shp"
    zones.to_file(path)
    return path


@pytest.fixture
def common_name_shapefile(tmp_path):
    """Two zone polygons with a common-name attribute (e.g. cover_crops_2025_2026)."""
    zones = gpd.GeoDataFrame(
        {
            "species": ["ABUTH", "BETVU"],
            "comm_name": ["Velvetleaf", "Sugarbeet"],
        },
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        crs=ZONE_CRS,
    )
    path = tmp_path / "common_name_zones.shp"
    zones.to_file(path)
    return path


@pytest.fixture
def class_id_shapefile(tmp_path):
    """Zone polygons that also carry a class_id attribute, alongside comm_name."""
    zones = gpd.GeoDataFrame(
        {
            "species": ["ABUTH", "BETVU"],
            "comm_name": ["Velvetleaf", "Sugarbeet"],
            "class_id": ["75", "76"],
        },
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        crs=ZONE_CRS,
    )
    path = tmp_path / "class_id_zones.shp"
    zones.to_file(path)
    return path


@pytest.fixture
def cultivar_shapefile(tmp_path):
    """Two zone polygons, same species, distinct cultivars (e.g. peanuts_2026)."""
    zones = gpd.GeoDataFrame(
        {
            "species": ["ARHY", "ARHY"],
            "cultc_id": ["107", "102"],
            "disp_name": ["Peanut - EXP-OLEIC-001", "Peanut - TifNV-HG"],
        },
        geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)],
        crs=ZONE_CRS,
    )
    path = tmp_path / "cultivar_zones.shp"
    zones.to_file(path)
    return path


@pytest.fixture
def geo_df():
    """Three detections: two inside zones, one just outside both (~2.8m from the
    nearest zone 1 corner, well within the default nearest-fallback threshold)."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001", "IMG_0001", "IMG_0001"],
            "bounding_box_id": [0, 1, 2],
            "classname": ["weed", "weed", "weed"],
            "conf": [0.95, 0.90, 0.85],
            "xmin": [0.1, 0.4, 0.7],
            "ymin": [0.1, 0.4, 0.7],
            "xmax": [0.2, 0.5, 0.8],
            "ymax": [0.2, 0.5, 0.8],
            # inside zone 1, inside zone 2, just outside both (nearest: zone 1)
            "world_centroid_x": [5.0, 25.0, 12.0],
            "world_centroid_y": [5.0, 25.0, 12.0],
            "world_tl_x": [4.0, 24.0, 11.0],
            "world_tl_y": [4.0, 24.0, 11.0],
            "world_tr_x": [6.0, 26.0, 13.0],
            "world_tr_y": [4.0, 24.0, 11.0],
            "world_bl_x": [4.0, 24.0, 11.0],
            "world_bl_y": [6.0, 26.0, 13.0],
            "world_br_x": [6.0, 26.0, 13.0],
            "world_br_y": [6.0, 26.0, 13.0],
            "crs": [ZONE_CRS, ZONE_CRS, ZONE_CRS],
        }
    )


@pytest.fixture
def far_geo_df():
    """Single detection ~99m from the nearest zone polygon — beyond any
    reasonable nearest-fallback threshold."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001"],
            "bounding_box_id": [0],
            "classname": ["weed"],
            "conf": [0.95],
            "xmin": [0.1],
            "ymin": [0.1],
            "xmax": [0.2],
            "ymax": [0.2],
            "world_centroid_x": [100.0],
            "world_centroid_y": [100.0],
            "world_tl_x": [99.0],
            "world_tl_y": [99.0],
            "world_tr_x": [101.0],
            "world_tr_y": [99.0],
            "world_bl_x": [99.0],
            "world_bl_y": [101.0],
            "world_br_x": [101.0],
            "world_br_y": [101.0],
            "crs": [ZONE_CRS],
        }
    )


@pytest.fixture
def det_df():
    """Basic detection rows as produced by jpg_to_det — no world coordinate columns."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001", "IMG_0001"],
            "bounding_box_id": [0, 1],
            "classname": ["weed", "weed"],
            "conf": [0.95, 0.90],
            "xmin": [0.1, 0.4],
            "ymin": [0.1, 0.4],
            "xmax": [0.2, 0.5],
            "ymax": [0.2, 0.5],
        }
    )


def test_assign_spatial_within(geo_df, shapefile):
    """Centroids inside a zone polygon get that zone's species and spatial_join method."""
    result = assign_spatial(geo_df, str(shapefile))

    # test bbox 0 and 1
    inside = result[result["bounding_box_id"].isin([0, 1])]

    # validate species ids
    assert set(inside["species_id"]) == {"ABUTH", "BETVU"}

    # validate spaital join
    assert (inside["assignment_method"] == "spatial_join").all()


def test_assign_spatial_nearest_fallback(geo_df, shapefile):
    """Centroid outside all zone polygons falls back to the nearest polygon."""
    result = assign_spatial(geo_df, str(shapefile))

    # test bbox 2
    outside = result[result["bounding_box_id"] == 2]
    assert outside["species_id"].notna().all()
    assert (outside["assignment_method"] == "nearest_polygon").all()


def test_assign_spatial_tags_unknown_when_nearest_zone_beyond_default_threshold(far_geo_df, shapefile):
    """A detection ~99m from every zone polygon exceeds the 5m default and is
    tagged UNKNOWN instead of failing the batch."""
    result = assign_spatial(far_geo_df, str(shapefile))

    assert result.loc[0, "species_id"] == UNKNOWN_SPECIES_ID
    assert result.loc[0, "assignment_method"] == "unknown_too_far"


def test_assign_spatial_tags_unknown_when_nearest_zone_beyond_custom_threshold(geo_df, shapefile):
    """A custom, stricter max_nearest_distance_m can flag a point normally within
    tolerance, tagging it UNKNOWN instead of failing the batch."""
    result = assign_spatial(geo_df, str(shapefile), max_nearest_distance_m=1.0)

    outside = result[result["bounding_box_id"] == 2].iloc[0]
    assert outside["species_id"] == UNKNOWN_SPECIES_ID
    assert outside["assignment_method"] == "unknown_too_far"

    # the two in-zone detections are unaffected by the stricter threshold
    inside = result[result["bounding_box_id"].isin([0, 1])]
    assert set(inside["species_id"]) == {"ABUTH", "BETVU"}


def test_assign_spatial_tags_unknown_class_id_when_present(far_geo_df, class_id_shapefile):
    """A too-far detection gets UNKNOWN_CLASS_ID when the shapefile carries
    class_id for its zones, same as any other zone attribute."""
    result = assign_spatial(far_geo_df, str(class_id_shapefile))

    assert result.loc[0, "class_id"] == UNKNOWN_CLASS_ID


def test_assign_spatial_nearest_fallback_within_custom_threshold(far_geo_df, shapefile):
    """Raising max_nearest_distance_m lets a farther point still fall back normally."""
    result = assign_spatial(far_geo_df, str(shapefile), max_nearest_distance_m=200.0)

    assert result.loc[0, "species_id"] == "BETVU"
    assert result.loc[0, "assignment_method"] == "nearest_polygon"


def test_assign_spatial_no_cultivar_columns_when_shapefile_lacks_them(geo_df, shapefile):
    """Shapefiles without cultc_id (most seasons) produce no cultivar columns at all."""
    result = assign_spatial(geo_df, str(shapefile))

    cultivar_cols = [c for c in result.columns if c.startswith("cultivar_")]
    assert cultivar_cols == [], f"unexpected cultivar columns: {cultivar_cols}"


def test_assign_spatial_within_assigns_species_name(geo_df, common_name_shapefile):
    """Zones with a comm_name attribute get species_name alongside species_id."""
    result = assign_spatial(geo_df, str(common_name_shapefile))

    inside = result[result["bounding_box_id"].isin([0, 1])]
    assert set(inside["species_name"]) == {"Velvetleaf", "Sugarbeet"}
    # comm_name shapefiles never have cultivar columns
    assert "cultivar_id" not in result.columns
    assert "cultivar_name" not in result.columns


def test_assign_spatial_carries_class_id_when_present(geo_df, class_id_shapefile):
    """class_id is joined through independently of species_name/cultivar."""
    result = assign_spatial(geo_df, str(class_id_shapefile))

    inside = result[result["bounding_box_id"].isin([0, 1])]
    assert set(inside["class_id"]) == {"75", "76"}


def test_assign_spatial_no_class_id_column_when_shapefile_lacks_it(geo_df, shapefile):
    result = assign_spatial(geo_df, str(shapefile))
    assert "class_id" not in result.columns


@pytest.fixture
def geo_df_with_color_checker():
    """Same layout as geo_df, plus a color-checker row interleaved between
    the two in-zone detections — sitting inside zone 1 (would normally match
    ABUTH/class_id 75) — to verify it gets the fixed colorchecker identity
    instead, and that original row order survives the split/rejoin."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001"] * 4,
            "bounding_box_id": [0, 1, 2, 3],
            "classname": ["weed", "color_checker", "weed", "weed"],
            "conf": [0.95, 0.99, 0.90, 0.85],
            "xmin": [0.1, 0.3, 0.4, 0.7],
            "ymin": [0.1, 0.3, 0.4, 0.7],
            "xmax": [0.2, 0.35, 0.5, 0.8],
            "ymax": [0.2, 0.35, 0.5, 0.8],
            # inside zone 1, inside zone 1, inside zone 2, just outside both
            "world_centroid_x": [5.0, 6.0, 25.0, 12.0],
            "world_centroid_y": [5.0, 6.0, 25.0, 12.0],
            "world_tl_x": [4.0, 5.0, 24.0, 11.0],
            "world_tl_y": [4.0, 5.0, 24.0, 11.0],
            "world_tr_x": [6.0, 7.0, 26.0, 13.0],
            "world_tr_y": [4.0, 5.0, 24.0, 11.0],
            "world_bl_x": [4.0, 5.0, 24.0, 11.0],
            "world_bl_y": [6.0, 7.0, 26.0, 13.0],
            "world_br_x": [6.0, 7.0, 26.0, 13.0],
            "world_br_y": [6.0, 7.0, 26.0, 13.0],
            "crs": [ZONE_CRS] * 4,
        }
    )


@pytest.fixture
def far_color_checker_df():
    """A color-checker ~99m from every zone polygon — must get the
    COLORCHECKER identity, not UNKNOWN, since color-checker rows never enter
    the spatial join at all (contrast with far_geo_df, a plant at the same
    distance, which gets tagged UNKNOWN instead)."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001"],
            "bounding_box_id": [0],
            "classname": ["color_checker"],
            "conf": [0.99],
            "xmin": [0.1],
            "ymin": [0.1],
            "xmax": [0.2],
            "ymax": [0.2],
            "world_centroid_x": [100.0],
            "world_centroid_y": [100.0],
            "world_tl_x": [99.0],
            "world_tl_y": [99.0],
            "world_tr_x": [101.0],
            "world_tr_y": [99.0],
            "world_bl_x": [99.0],
            "world_bl_y": [101.0],
            "world_br_x": [101.0],
            "world_br_y": [101.0],
            "crs": [ZONE_CRS],
        }
    )


def test_assign_spatial_color_checker_excluded_from_zone_species(geo_df_with_color_checker, class_id_shapefile):
    """A color-checker centroid sitting inside a zone polygon gets the fixed
    COLORCHECKER identity, not that zone's species/class_id — and the real
    plant detections around it are unaffected."""
    result = assign_spatial(geo_df_with_color_checker, str(class_id_shapefile))

    checker = result[result["bounding_box_id"] == 1].iloc[0]
    assert checker["species_id"] == COLOR_CHECKER_SPECIES_ID
    assert checker["class_id"] == COLOR_CHECKER_CLASS_ID
    assert checker["assignment_method"] == "color_checker_class"

    plants = result[result["bounding_box_id"].isin([0, 2])]
    assert set(plants["species_id"]) == {"ABUTH", "BETVU"}
    assert set(plants["class_id"]) == {"75", "76"}


def test_assign_spatial_preserves_row_order_with_color_checker(geo_df_with_color_checker, shapefile):
    """Splitting color-checker rows out for exclusion and rejoining them
    afterward must not reshuffle the batch's row order."""
    result = assign_spatial(geo_df_with_color_checker, str(shapefile))
    assert list(result["bounding_box_id"]) == [0, 1, 2, 3]


def test_assign_spatial_color_checker_skips_zone_too_far_check(far_color_checker_df, shapefile):
    """A color-checker this far from every zone would get tagged UNKNOWN if
    it were a plant (see
    test_assign_spatial_tags_unknown_when_nearest_zone_beyond_default_threshold);
    since it never enters the join, it keeps its COLORCHECKER identity instead."""
    result = assign_spatial(far_color_checker_df, str(shapefile))
    assert result.loc[0, "species_id"] == COLOR_CHECKER_SPECIES_ID
    assert result.loc[0, "assignment_method"] == "color_checker_class"


def test_assign_spatial_all_color_checkers(far_color_checker_df, shapefile):
    """A batch that's entirely color-checker rows (no plants at all) still
    resolves without ever touching the shapefile's zones."""
    result = assign_spatial(far_color_checker_df, str(shapefile))
    assert len(result) == 1
    assert result.loc[0, "species_id"] == COLOR_CHECKER_SPECIES_ID


@pytest.fixture
def geo_df_with_ungeoreferenced():
    """Same layout as geo_df, plus a fourth row with no world coordinates at
    all — as remap_rows() now produces for a detection it couldn't
    georeference (no grid for the image, or every bbox corner missed the
    grid surface even after nudging)."""
    return pd.DataFrame(
        {
            "image_id": ["IMG_0001"] * 4,
            "bounding_box_id": [0, 1, 2, 3],
            "classname": ["weed", "weed", "weed", "weed"],
            "conf": [0.95, 0.90, 0.85, 0.80],
            "xmin": [0.1, 0.4, 0.7, 0.5],
            "ymin": [0.1, 0.4, 0.7, 0.5],
            "xmax": [0.2, 0.5, 0.8, 0.6],
            "ymax": [0.2, 0.5, 0.8, 0.6],
            "world_centroid_x": [5.0, 25.0, 12.0, None],
            "world_centroid_y": [5.0, 25.0, 12.0, None],
            "world_tl_x": [4.0, 24.0, 11.0, None],
            "world_tl_y": [4.0, 24.0, 11.0, None],
            "world_tr_x": [6.0, 26.0, 13.0, None],
            "world_tr_y": [4.0, 24.0, 11.0, None],
            "world_bl_x": [4.0, 24.0, 11.0, None],
            "world_bl_y": [6.0, 26.0, 13.0, None],
            "world_br_x": [6.0, 26.0, 13.0, None],
            "world_br_y": [6.0, 26.0, 13.0, None],
            "crs": [ZONE_CRS, ZONE_CRS, ZONE_CRS, None],
        }
    )


def test_assign_spatial_tags_unknown_for_ungeoreferenced_rows(geo_df_with_ungeoreferenced, shapefile):
    """A row with no world coordinates gets tagged UNKNOWN instead of being
    dropped or crashing the spatial join; georeferenced rows are unaffected."""
    result = assign_spatial(geo_df_with_ungeoreferenced, str(shapefile))

    missing = result[result["bounding_box_id"] == 3].iloc[0]
    assert missing["species_id"] == UNKNOWN_SPECIES_ID
    assert missing["assignment_method"] == "unknown_not_georeferenced"

    georeferenced = result[result["bounding_box_id"].isin([0, 1, 2])]
    assert set(georeferenced["species_id"]) == {"ABUTH", "BETVU"}


def test_assign_spatial_preserves_row_order_with_ungeoreferenced(geo_df_with_ungeoreferenced, shapefile):
    result = assign_spatial(geo_df_with_ungeoreferenced, str(shapefile))
    assert list(result["bounding_box_id"]) == [0, 1, 2, 3]


def test_assign_spatial_tags_unknown_when_no_world_columns_at_all(det_df, shapefile):
    """A batch where remap_rows() produced zero world coordinates for any
    row (e.g. every image lacked a grid) still resolves, tagging every row
    UNKNOWN rather than crashing on a missing world_centroid_x column."""
    result = assign_spatial(det_df, str(shapefile))

    assert (result["species_id"] == UNKNOWN_SPECIES_ID).all()
    assert (result["assignment_method"] == "unknown_not_georeferenced").all()


def test_assign_spatial_ungeoreferenced_color_checker_keeps_color_checker_identity(shapefile):
    """A color-checker detection that never got georeferenced still gets the
    COLORCHECKER identity, not UNKNOWN -- classname is checked first."""
    df = pd.DataFrame(
        {
            "image_id": ["IMG_0001"],
            "bounding_box_id": [0],
            "classname": ["color_checker"],
            "conf": [0.99],
            "xmin": [0.1],
            "ymin": [0.1],
            "xmax": [0.2],
            "ymax": [0.2],
        }
    )
    result = assign_spatial(df, str(shapefile))

    assert result.loc[0, "species_id"] == COLOR_CHECKER_SPECIES_ID
    assert result.loc[0, "assignment_method"] == "color_checker_class"


@pytest.fixture
def geo_df_with_placeholder(geo_df):
    """geo_df plus a fourth row: a zero-detection placeholder (see jpg_to_det's
    export_predictions), bounding_box_id=="" and no world coordinates at
    all -- distinct from a real detection that failed to georeference."""
    placeholder = pd.DataFrame([{
        "image_id": "IMG_0002", "bounding_box_id": "", "classname": "",
        "conf": "", "xmin": "", "ymin": "", "xmax": "", "ymax": "",
    }])
    return pd.concat([geo_df, placeholder], ignore_index=True)


def test_assign_spatial_tags_placeholder_row_no_detections(geo_df_with_placeholder, shapefile):
    """A zero-detection placeholder row gets NO_DETECTIONS_METHOD with no
    species assigned -- not UNKNOWN_SPECIES_ID, which would misleadingly
    claim an unidentified plant exists in an image with no detections."""
    result = assign_spatial(geo_df_with_placeholder, str(shapefile))

    placeholder_row = result[result["bounding_box_id"] == ""].iloc[0]
    assert placeholder_row["assignment_method"] == NO_DETECTIONS_METHOD
    assert pd.isna(placeholder_row["species_id"]) or placeholder_row["species_id"] in (None, "")

    real = result[result["bounding_box_id"].isin([0, 1, 2])]
    assert set(real["species_id"]) == {"ABUTH", "BETVU"}


def test_assign_spatial_placeholder_never_reaches_spatial_join(geo_df_with_placeholder, shapefile):
    """Placeholder rows are excluded before the join entirely -- they never
    get a real zone's species_id even if the join would otherwise assign one."""
    result = assign_spatial(geo_df_with_placeholder, str(shapefile))

    placeholder_row = result[result["bounding_box_id"] == ""].iloc[0]
    assert placeholder_row["assignment_method"] != "spatial_join"
    assert placeholder_row["assignment_method"] != "nearest_polygon"


def test_assign_spatial_within_assigns_cultivar(geo_df, cultivar_shapefile):
    """Zones with a cultc_id attribute get cultivar_id/cultivar_name alongside species_id."""
    result = assign_spatial(geo_df, str(cultivar_shapefile))

    inside = result[result["bounding_box_id"].isin([0, 1])]
    assert set(inside["species_id"]) == {"ARHY"}
    assert set(inside["cultivar_id"]) == {"107", "102"}
    assert set(inside["cultivar_name"]) == {"Peanut - EXP-OLEIC-001", "Peanut - TifNV-HG"}
    # cultivar shapefiles never have a comm_name column
    assert "species_name" not in result.columns


def test_assign_spatial_nearest_fallback_assigns_cultivar(geo_df, cultivar_shapefile):
    """Nearest-polygon fallback carries cultivar columns too, not just species."""
    result = assign_spatial(geo_df, str(cultivar_shapefile))

    outside = result[result["bounding_box_id"] == 2]
    assert (outside["assignment_method"] == "nearest_polygon").all()
    assert outside["cultivar_id"].notna().all()
    assert outside["cultivar_name"].notna().all()


def test_assign_monoculture_sets_species_and_method(det_df):
    """Monoculture -> detections receive given species code and monoculture_config method."""
    result = assign_monoculture(det_df, "BETVU")

    assert (result["species_id"] == "BETVU").all()
    assert (result["assignment_method"] == "monoculture_config").all()


def test_assign_monoculture_no_world_columns(det_df):
    """Monoculture output contains no world coordinate columns."""
    result = assign_monoculture(det_df, "BETVU")

    # no "world_" columns
    world_cols = [c for c in result.columns if c.startswith("world_")]
    assert world_cols == [], f"unexpected world columns: {world_cols}"


def test_assign_monoculture_excludes_color_checker(det_df):
    """A color-checker row doesn't get the monoculture species code — it
    gets the fixed COLORCHECKER identity instead."""
    det_df = det_df.copy()
    det_df.loc[1, "classname"] = "color_checker"

    result = assign_monoculture(det_df, "BETVU")

    plant = result.loc[0]
    assert plant["species_id"] == "BETVU"
    assert plant["assignment_method"] == "monoculture_config"

    checker = result.loc[1]
    assert checker["species_id"] == COLOR_CHECKER_SPECIES_ID
    assert checker["assignment_method"] == "color_checker_class"


def test_assign_monoculture_excludes_placeholder_row(det_df):
    """A zero-detection placeholder row doesn't get the monoculture species
    code either — it gets NO_DETECTIONS_METHOD with no species assigned."""
    det_df = det_df.copy()
    det_df["bounding_box_id"] = det_df["bounding_box_id"].astype(object)
    det_df.loc[1, "bounding_box_id"] = ""

    result = assign_monoculture(det_df, "BETVU")

    plant = result.loc[0]
    assert plant["species_id"] == "BETVU"
    assert plant["assignment_method"] == "monoculture_config"

    placeholder = result.loc[1]
    assert placeholder["assignment_method"] == NO_DETECTIONS_METHOD
    assert pd.isna(placeholder["species_id"]) or placeholder["species_id"] in (None, "")


@pytest.fixture
def catalog():
    """Minimal species_catalog.generated.json-shaped dict (see orchestrator/species_catalog.py)."""
    return {
        "species": {
            "ARHY": {
                "common_name": "peanut",
                "family": "Fabaceae",
                "genus": "Arachis",
                "growth_habit": "forb/herb",
                "category": "cash crop",
                "hex": "#a5482f",
                "r": 165,
                "g": 72,
                "b": 47,
            },
            "COLORCHECKER": {
                "common_name": "colorchecker",
                "family": "Colorchecker",
                "genus": "Colorchecker",
                "growth_habit": "colorchecker",
                "category": "colorchecker",
                "class_id": 28,
                "hex": "#e73b58",
                "r": 231,
                "g": 59,
                "b": 88,
            },
        },
        "cultivars": {
            "107": {
                "display_name": "Peanut - EXP-OLEIC-001",
                "line_name": "oleic",
                "registered": 0,
                "hex": "#f105f2",
                "r": 241,
                "g": 5,
                "b": 242,
            },
            "102": {
                "display_name": "Peanut - TifNV-HG",
                "line_name": "high-oleic",
                "registered": 1,
                "hex": "#0f52f1",
                "r": 15,
                "g": 82,
                "b": 241,
            },
        },
    }


def test_load_catalog_reads_json_file(tmp_path):
    path = tmp_path / "species_catalog.generated.json"
    path.write_text(json.dumps({"species": {}, "cultivars": {}}), encoding="utf-8")

    assert load_catalog(path) == {"species": {}, "cultivars": {}}


def test_enrich_with_catalog_adds_species_columns(catalog):
    dets = pd.DataFrame({"species_id": ["ARHY"], "assignment_method": ["monoculture_config"]})

    result = enrich_with_catalog(dets, catalog)

    assert result.loc[0, "species_common_name"] == "peanut"
    assert result.loc[0, "species_family"] == "Fabaceae"
    assert (result.loc[0, "species_r"], result.loc[0, "species_g"], result.loc[0, "species_b"]) == (165, 72, 47)


def test_enrich_with_catalog_populates_color_checker_identity(catalog):
    """species_id=COLOR_CHECKER_SPECIES_ID (as set by assign_spatial/assign_monoculture
    for color-checker rows) resolves against the catalog's own COLORCHECKER entry."""
    dets = pd.DataFrame({
        "species_id": [COLOR_CHECKER_SPECIES_ID],
        "assignment_method": ["color_checker_class"],
    })

    result = enrich_with_catalog(dets, catalog)

    assert result.loc[0, "species_common_name"] == "colorchecker"
    assert (result.loc[0, "species_r"], result.loc[0, "species_g"], result.loc[0, "species_b"]) == (231, 59, 88)


def test_enrich_with_catalog_raises_on_unmatched_species(catalog):
    dets = pd.DataFrame({"species_id": ["UNKNOWN_CODE"], "assignment_method": ["monoculture_config"]})

    with pytest.raises(UnknownSpeciesCodeError, match="UNKNOWN_CODE"):
        enrich_with_catalog(dets, catalog)


def test_enrich_with_catalog_raises_on_unmatched_cultivar(catalog):
    dets = pd.DataFrame({
        "species_id": ["ARHY"],
        "cultivar_id": ["UNKNOWN_CULTIVAR"],
    })

    with pytest.raises(UnknownSpeciesCodeError, match="UNKNOWN_CULTIVAR"):
        enrich_with_catalog(dets, catalog)


def test_enrich_with_catalog_adds_cultivar_columns_only_when_present(catalog):
    dets = pd.DataFrame({
        "species_id": ["ARHY", "ARHY"],
        "cultivar_id": ["107", None],
    })

    result = enrich_with_catalog(dets, catalog)

    assert result.loc[0, "cultivar_display_name"] == "Peanut - EXP-OLEIC-001"
    assert pd.isna(result.loc[1, "cultivar_display_name"])


def test_enrich_with_catalog_no_cultivar_columns_when_shapefile_lacked_them(catalog):
    dets = pd.DataFrame({"species_id": ["ARHY"]})

    result = enrich_with_catalog(dets, catalog)

    cultivar_cols = [c for c in result.columns if c.startswith("cultivar_")]
    assert cultivar_cols == []


def test_enrich_with_catalog_after_assign_spatial(geo_df, cultivar_shapefile, catalog):
    """End-to-end: assign_spatial's cultivar_id feeds straight into enrichment."""
    assigned = assign_spatial(geo_df, str(cultivar_shapefile))

    result = enrich_with_catalog(assigned, catalog)

    exp_oleic_rows = result[result["cultivar_id"] == "107"]
    assert (exp_oleic_rows["cultivar_display_name"] == "Peanut - EXP-OLEIC-001").all()
    assert (exp_oleic_rows["species_common_name"] == "peanut").all()
