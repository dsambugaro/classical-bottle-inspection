"""InspectionParams: defaults mirror config.py and dict conversions are lossless."""

from __future__ import annotations

import pytest

from bottle_inspection import config, features, preprocessing
from bottle_inspection.params import InspectionParams


def cells(by_region: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """The per-region cell sizes of config (HOG_CELL / LBP_CELL) with `auto` shared by auto_1..3."""
    return {
        region: by_region["auto" if region in config.AUTO_ROIS else region]
        for region in config.REGIONS
    }


def test_defaults_equal_config_constants() -> None:
    """Every default comes from the constants used by the CLI."""
    p = InspectionParams.defaults()
    assert p.localization.size == config.LOCALIZATION_SIZE
    assert p.localization.close_size == config.MASK_CLOSE_SIZE
    assert p.localization.open_size == config.MASK_OPEN_SIZE
    assert p.localization.profile_smoothing == config.PROFILE_SMOOTHING
    assert p.localization.column_fraction == config.COLUMN_PROFILE_FRACTION
    assert p.localization.row_fraction == config.ROW_PROFILE_FRACTION
    assert p.localization.top_fraction == config.TOP_PROFILE_FRACTION
    assert p.localization.bottle_width == config.BOTTLE_WIDTH
    assert p.normalization.enabled == config.NORMALIZE_LIGHTING
    assert p.normalization.reference_patch == config.REFERENCE_PATCH
    assert p.bottle.crop_size == config.BOTTLE_SIZE
    assert p.rois.bands() == config.ROIS
    assert p.rois.sizes() == config.ROI_SIZE
    assert p.hog.orientations == config.HOG_ORIENTATIONS
    assert p.auto_rois.bands == ()
    assert p.hog.cells() == cells(config.HOG_CELL)
    assert p.hog.block == features.HOG_BLOCK
    assert p.color.hsv_bins == config.HSV_HISTOGRAM_BINS
    assert p.hog.cell_bottle == config.HOG_CELL["bottle"]
    assert p.lbp.points == config.LBP_POINTS
    assert p.lbp.radius == config.LBP_RADIUS
    assert p.lbp.cells() == cells(config.LBP_CELL)
    assert p.region_bands() == {**config.ROIS, "bottle": (0.0, 1.0)}
    assert list(p.features.groups) == config.FEATURE_GROUPS


def test_round_trips() -> None:
    """Nested and flat dicts convert back to equal params (also with learned auto bands)."""
    p = InspectionParams.defaults()
    assert InspectionParams.from_dict(p.to_dict()) == p
    assert InspectionParams.from_dict(p.to_flat_dict()) == p
    p = p.with_auto_bands(((0.0, 0.25), (0.25, 0.5), (0.5, 1.0)))
    assert InspectionParams.from_dict(p.to_dict()) == p
    assert InspectionParams.from_dict(p.to_flat_dict()) == p


def test_partial_update_keeps_other_values() -> None:
    """A partial dict changes only the given keys."""
    p = InspectionParams.from_dict({"rois.cap": [0.1, 0.2], "localization": {"close_size": 21}})
    assert p.rois.cap == (0.1, 0.2)
    assert p.localization.close_size == 21
    assert p.rois.label == config.ROIS["label"]


@pytest.mark.parametrize(
    "data",
    [{"rois.unknown": 1}, {"nope.x": 1}, {"localization.close_size": 1.5}, {"rois.cap": [1]}],
)
def test_invalid_dicts_raise(data: dict) -> None:
    """Unknown keys, wrong types and wrong lengths are rejected."""
    with pytest.raises((KeyError, TypeError, ValueError)):
        InspectionParams.from_dict(data)
