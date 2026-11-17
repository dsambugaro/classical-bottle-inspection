"""Inspection parameters as one immutable value (used by the web interface).

The pipeline functions read their ad hoc values from `config.py`. They also accept an
optional keyword argument `params: InspectionParams | None`; when it is None they use
`InspectionParams.defaults()`, which is built from the very same constants, so the CLI
uses the configured values. The web interface passes the values of a profile instead.

This module also holds the helpers shared by the CLI and the web interface to describe a
trained model: the extra metadata saved in `model.joblib`.
"""

from __future__ import annotations

import dataclasses
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from importlib import import_module, metadata
from typing import Any

from . import config

Range = tuple[float, float]
"""(start, end) pair of relative coordinates."""

Rect = tuple[float, float, float, float]
"""(x0, y0, x1, y1) in relative coordinates."""

Size = tuple[int, int]
"""(width, height) in pixels."""

Hsv = tuple[int, int, int]
"""(h, s, v) in OpenCV ranges (H 0..180, S and V 0..255)."""


def _module_constant(module: str, name: str) -> Any:
    """Reads a constant defined outside config.py (imported lazily: those modules import this one)."""
    return getattr(import_module(f"{__package__}.{module}"), name)


@dataclass(frozen=True)
class LocalizationParams:
    """Background subtraction and detection of the bottle box."""

    size: Size = field(default_factory=lambda: config.LOCALIZATION_SIZE)
    close_size: int = field(default_factory=lambda: config.MASK_CLOSE_SIZE)
    open_size: int = field(default_factory=lambda: config.MASK_OPEN_SIZE)
    profile_smoothing: int = field(default_factory=lambda: config.PROFILE_SMOOTHING)
    column_fraction: float = field(default_factory=lambda: config.COLUMN_PROFILE_FRACTION)
    bottle_width: float = field(default_factory=lambda: config.BOTTLE_WIDTH)
    top_fraction: float = field(default_factory=lambda: config.TOP_PROFILE_FRACTION)
    row_fraction: float = field(default_factory=lambda: config.ROW_PROFILE_FRACTION)


@dataclass(frozen=True)
class NormalizationParams:
    """Photometric normalization by the reference wall patch."""

    enabled: bool = field(default_factory=lambda: config.NORMALIZE_LIGHTING)
    reference_patch: Rect = field(default_factory=lambda: config.REFERENCE_PATCH)


@dataclass(frozen=True)
class BottleParams:
    """Canonical size of the detected bottle box."""

    crop_size: Size = field(default_factory=lambda: config.BOTTLE_SIZE)


@dataclass(frozen=True)
class WideAreaParams:
    """Wide fixed area of the image around the bottle (does not use the bottle box)."""

    rect: Rect = field(default_factory=lambda: config.WIDE_AREA_RECT)
    crop_size: Size = field(default_factory=lambda: config.WIDE_AREA_SIZE)


@dataclass(frozen=True)
class WholeImageParams:
    """Whole image, without localization."""

    crop_size: Size = field(default_factory=lambda: config.WHOLE_IMAGE_SIZE)


@dataclass(frozen=True)
class RoiParams:
    """Ad hoc ROIs: vertical band inside the bottle box (0 = top, 1 = bottom) and
    canonical resize size of each one."""

    cap: Range = field(default_factory=lambda: config.ROIS["cap"])
    label: Range = field(default_factory=lambda: config.ROIS["label"])
    body: Range = field(default_factory=lambda: config.ROIS["body"])
    cap_size: Size = field(default_factory=lambda: config.ROI_SIZE["cap"])
    label_size: Size = field(default_factory=lambda: config.ROI_SIZE["label"])
    body_size: Size = field(default_factory=lambda: config.ROI_SIZE["body"])

    def bands(self) -> dict[str, Range]:
        """The vertical band of each ROI, in the same order as config.ROIS."""
        return {"cap": self.cap, "label": self.label, "body": self.body}

    def sizes(self) -> dict[str, Size]:
        """The canonical size of each ROI, in the same order as config.ROI_SIZE."""
        return {"cap": self.cap_size, "label": self.label_size, "body": self.body_size}


@dataclass(frozen=True)
class AutoRoiParams:
    """Automatic regions: bands of the bottle box learned from the training images."""

    bands: tuple[Range, ...] = ()
    """(y0, y1) of each automatic region inside the bottle box, top to bottom. Empty
    until they are learned (regions.learn_auto_bands); a trained model stores them."""


def _region_cells(group: Any) -> dict[str, Size]:
    """The cell size of every region (the automatic regions share `cell_auto`)."""
    return {
        region: getattr(group, "cell_auto" if region in config.AUTO_ROIS else f"cell_{region}")
        for region in config.REGIONS
    }


@dataclass(frozen=True)
class HogParams:
    """HOG descriptor settings (they define the feature vector length)."""

    orientations: int = field(default_factory=lambda: config.HOG_ORIENTATIONS)
    cell_cap: Size = field(default_factory=lambda: config.HOG_CELL["cap"])
    cell_label: Size = field(default_factory=lambda: config.HOG_CELL["label"])
    cell_body: Size = field(default_factory=lambda: config.HOG_CELL["body"])
    cell_auto: Size = field(default_factory=lambda: config.HOG_CELL["auto"])
    cell_bottle: Size = field(default_factory=lambda: config.HOG_CELL["bottle"])
    cell_wide: Size = field(default_factory=lambda: config.HOG_CELL["wide"])
    cell_image: Size = field(default_factory=lambda: config.HOG_CELL["image"])
    block: Size = field(default_factory=lambda: _module_constant("features", "HOG_BLOCK"))

    def cells(self) -> dict[str, Size]:
        """The HOG cell size of each region."""
        return _region_cells(self)


@dataclass(frozen=True)
class LbpParams:
    """LBP descriptor settings (they define the feature vector length)."""

    points: int = field(default_factory=lambda: config.LBP_POINTS)
    radius: int = field(default_factory=lambda: config.LBP_RADIUS)
    cell_cap: Size = field(default_factory=lambda: config.LBP_CELL["cap"])
    cell_label: Size = field(default_factory=lambda: config.LBP_CELL["label"])
    cell_body: Size = field(default_factory=lambda: config.LBP_CELL["body"])
    cell_auto: Size = field(default_factory=lambda: config.LBP_CELL["auto"])
    cell_bottle: Size = field(default_factory=lambda: config.LBP_CELL["bottle"])
    cell_wide: Size = field(default_factory=lambda: config.LBP_CELL["wide"])
    cell_image: Size = field(default_factory=lambda: config.LBP_CELL["image"])

    def cells(self) -> dict[str, Size]:
        """The LBP cell size of each region."""
        return _region_cells(self)


@dataclass(frozen=True)
class ColorParams:
    """HSV histogram settings."""

    hsv_bins: Hsv = field(default_factory=lambda: config.HSV_HISTOGRAM_BINS)


@dataclass(frozen=True)
class FeatureParams:
    """Feature groups concatenated into the feature vector of a model, in order.

    The default lists every extracted group; a trained model stores only the groups of
    its own vector (see config.FEATURE_VECTORS)."""

    groups: tuple[str, ...] = field(default_factory=lambda: tuple(config.FEATURE_GROUPS))


@dataclass(frozen=True)
class InspectionParams:
    """Every configurable value the preprocessing and feature functions need."""

    localization: LocalizationParams = field(default_factory=LocalizationParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    bottle: BottleParams = field(default_factory=BottleParams)
    wide: WideAreaParams = field(default_factory=WideAreaParams)
    image: WholeImageParams = field(default_factory=WholeImageParams)
    rois: RoiParams = field(default_factory=RoiParams)
    auto_rois: AutoRoiParams = field(default_factory=AutoRoiParams)
    hog: HogParams = field(default_factory=HogParams)
    lbp: LbpParams = field(default_factory=LbpParams)
    color: ColorParams = field(default_factory=ColorParams)
    features: FeatureParams = field(default_factory=FeatureParams)

    @staticmethod
    def defaults() -> InspectionParams:
        """The values of the constants in config.py (exactly what the CLI uses)."""
        return _cached_defaults()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Plain nested dict ({group: {name: value}}) with tuples turned into lists."""
        return {
            group.name: {
                item.name: _to_plain(getattr(getattr(self, group.name), item.name))
                for item in dataclasses.fields(getattr(self, group.name))
            }
            for group in dataclasses.fields(self)
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat dict keyed by dot paths, e.g. {"rois.cap": [0.08, 0.19], ...}."""
        return {
            f"{group}.{name}": value
            for group, values in self.to_dict().items()
            for name, value in values.items()
        }

    @staticmethod
    def from_dict(data: dict[str, Any], base: InspectionParams | None = None) -> InspectionParams:
        """Builds params from a nested or flat (dot path) dict.

        Keys that are absent keep the value of `base` (defaults when None).

        Raises:
            KeyError: if a group or parameter name is unknown.
        """
        nested: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                nested.setdefault(key, {}).update(value)
            else:
                group, _, name = key.partition(".")
                nested.setdefault(group, {})[name] = value
        current = base if base is not None else InspectionParams.defaults()
        groups: dict[str, Any] = {}
        for group in dataclasses.fields(current):
            group_value = getattr(current, group.name)
            changes = nested.pop(group.name, {})
            known = {item.name: item for item in dataclasses.fields(group_value)}
            unknown = set(changes) - set(known)
            if unknown:
                raise KeyError(f"unknown parameters in '{group.name}': {sorted(unknown)}")
            converted = {
                name: _from_plain(value, getattr(group_value, name))
                for name, value in changes.items()
            }
            groups[group.name] = dataclasses.replace(group_value, **converted)
        if nested:
            raise KeyError(f"unknown parameter groups: {sorted(nested)}")
        return InspectionParams(**groups)

    def auto_bands(self) -> dict[str, Range]:
        """Band of each automatic region, keyed by name (empty until they are learned)."""
        return dict(zip(config.AUTO_ROIS, self.auto_rois.bands))

    def region_bands(self) -> dict[str, Range]:
        """Vertical band, inside the bottle box, of every available region: the ad hoc
        ROIs, the automatic regions (once learned) and the whole bottle."""
        return {**self.rois.bands(), **self.auto_bands(), config.WHOLE_BOTTLE: (0.0, 1.0)}

    def region_sizes(self) -> dict[str, Size]:
        """Canonical size of every available region. An automatic region is a slice of
        the BOTTLE_SIZE crop, so its size follows from its band."""
        width, height = self.bottle.crop_size
        auto = {
            name: (width, round((y1 - y0) * height)) for name, (y0, y1) in self.auto_bands().items()
        }
        return {
            **self.rois.sizes(),
            **auto,
            config.WHOLE_BOTTLE: self.bottle.crop_size,
            config.WIDE_AREA: self.wide.crop_size,
            config.WHOLE_IMAGE: self.image.crop_size,
        }

    def feature_groups(self) -> list[str]:
        """The feature groups that can be extracted with these parameters, in
        config.FEATURE_GROUPS order (the automatic ones only once their bands are learned)."""
        regions = self.region_sizes()
        return [g for g in config.FEATURE_GROUPS if g.rpartition("_")[0] in regions]

    def with_auto_bands(self, bands: tuple[Range, ...]) -> InspectionParams:
        """A copy with the learned bands of the automatic regions."""
        return dataclasses.replace(self, auto_rois=AutoRoiParams(bands=tuple(bands)))

    def with_feature_groups(self, groups: tuple[str, ...]) -> InspectionParams:
        """A copy whose feature vector is made of `groups`, in order."""
        return dataclasses.replace(self, features=FeatureParams(groups=groups))

    def with_normalization(self, enabled: bool) -> InspectionParams:
        """A copy with normalization.enabled set to `enabled`."""
        return dataclasses.replace(
            self, normalization=dataclasses.replace(self.normalization, enabled=enabled)
        )


@cache
def _cached_defaults() -> InspectionParams:
    """InspectionParams built from config.py, computed once (config holds constants)."""
    return InspectionParams()


def resolve(params: InspectionParams | None) -> InspectionParams:
    """Returns `params`, or the defaults from config.py when it is None."""
    return params if params is not None else _cached_defaults()


def _to_plain(value: Any) -> Any:
    """Turns tuples into lists, recursively, for JSON."""
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _from_plain(value: Any, reference: Any) -> Any:
    """Converts a JSON value to the type of `reference` (lists become tuples)."""
    if isinstance(reference, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected a sequence, got {value!r}")
        if reference and len(value) != len(reference) and not isinstance(reference[0], str):
            raise ValueError(f"expected {len(reference)} values, got {len(value)}")
        if reference:
            return tuple(_from_plain(item, reference[0]) for item in value)
        return tuple(tuple(item) if isinstance(item, list) else item for item in value)
    if isinstance(reference, bool):
        if not isinstance(value, bool):
            raise TypeError(f"expected a bool, got {value!r}")
        return value
    if isinstance(reference, int):
        if isinstance(value, bool) or not float(value).is_integer():
            raise TypeError(f"expected an integer, got {value!r}")
        return int(value)
    if isinstance(reference, float):
        if isinstance(value, bool):
            raise TypeError(f"expected a number, got {value!r}")
        return float(value)
    return value


# =============================================================================
# Trained model metadata shared by the CLI and the web interface
# =============================================================================


ARTIFACT_FORMAT_VERSION: int = 1
"""Version of the model.joblib contents: the fitted pipeline, the background model, the
classes and the training metadata of `training_metadata` (parameters, feature vector, best
hyperparameters, library versions). A model of another version is not loaded."""


def library_versions() -> dict[str, str]:
    """Versions of the libraries that determine features and model loading."""
    import cv2

    def installed(distribution: str) -> str:
        """Installed version of a distribution, or "unknown"."""
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "unknown"

    return {
        "python": platform.python_version(),
        "numpy": installed("numpy"),
        "scikit-learn": installed("scikit-learn"),
        "scikit-image": installed("scikit-image"),
        "opencv": str(cv2.__version__),
    }


def training_metadata(
    normalize: bool,
    best_params: dict[str, Any],
    feature_groups: dict[str, int],
    params: InspectionParams | None = None,
) -> dict[str, Any]:
    """Extra keys saved in model.joblib next to model, background and classes.

    Args:
        normalize: whether the run used the photometric normalization.
        best_params: hyperparameters chosen by the grid search.
        feature_groups: number of values of each feature group of the model's vector, in
            vector order. They are saved as `params.features.groups`, so the model can be
            fed with exactly the vector it was trained on.
        params: parameters used in training (defaults from config.py when None).
    """
    used = resolve(params).with_normalization(normalize).with_feature_groups(tuple(feature_groups))
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "params": used.to_dict(),
        "best_params": dict(best_params),
        "feature_length": int(sum(feature_groups.values())),
        "feature_groups": dict(feature_groups),
        "library_versions": library_versions(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
