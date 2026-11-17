"""Parameter catalog (§7): the single definition the profile editor is rendered from.

Each entry describes one key of `InspectionParams` (dot path) or of the verdict settings:
type, limits, whether it is editable or locked by training, i18n keys and the name of the
constant in config.py. `validate_params` checks a flat parameter dict against the catalog
and, optionally, against the locked values of a trained model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from bottle_inspection.config import AUTO_ROIS, HOG_CELL, LBP_CELL
from bottle_inspection.params import InspectionParams

from .classes import CLASS_NAMES

ParamType = Literal[
    "int", "float", "bool", "range", "ranges", "rect", "size", "int3", "enum", "multiselect"
]
"""`int3` is a fixed triple of integers (the HSV histogram bins); `ranges` is a list of
(start, end) pairs of any length (the learned bands of the automatic regions)."""

Mode = Literal["editable", "locked"]


@dataclass(frozen=True)
class CatalogEntry:
    """One parameter of the catalog."""

    key: str
    group: str
    type: ParamType
    mode: Mode
    config_constant: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    affects_preview: bool = False
    options: list[str] | None = None
    constraints: list[str] = field(default_factory=list)
    nullable: bool = False

    @property
    def label_key(self) -> str:
        """i18n key of the label."""
        return f"params.{self.key}.label"

    @property
    def help_key(self) -> str:
        """i18n key of the help text."""
        return f"params.{self.key}.help"


GROUPS: list[str] = [
    "localization",
    "normalization",
    "bottle",
    "rois",
    "auto_rois",
    "hog",
    "lbp",
    "color",
    "features",
    "verdict",
]
"""Groups in display order."""

REL = {"min": 0.0, "max": 1.0, "step": 0.01, "unit": "relative"}


def _entries() -> list[CatalogEntry]:
    """Every catalog entry, in display order."""
    e = CatalogEntry
    return [
        # 7.1 Localization (background subtraction, Otsu threshold, bottle box)
        e("localization.size", "localization", "size", "locked", "LOCALIZATION_SIZE", unit="px"),
        e(
            "localization.close_size",
            "localization",
            "int",
            "editable",
            "MASK_CLOSE_SIZE",
            min=1,
            max=101,
            step=2,
            unit="px",
            affects_preview=True,
        ),
        e(
            "localization.open_size",
            "localization",
            "int",
            "editable",
            "MASK_OPEN_SIZE",
            min=1,
            max=51,
            step=2,
            unit="px",
            affects_preview=True,
        ),
        e(
            "localization.profile_smoothing",
            "localization",
            "int",
            "editable",
            "PROFILE_SMOOTHING",
            min=1,
            max=101,
            step=2,
            unit="px",
            affects_preview=True,
        ),
        e(
            "localization.column_fraction",
            "localization",
            "float",
            "editable",
            "COLUMN_PROFILE_FRACTION",
            min=0.05,
            max=0.95,
            step=0.01,
            unit="fraction",
            affects_preview=True,
        ),
        e(
            "localization.bottle_width",
            "localization",
            "float",
            "editable",
            "BOTTLE_WIDTH",
            min=0.02,
            max=1.0,
            step=0.005,
            unit="relative",
            affects_preview=True,
        ),
        e(
            "localization.top_fraction",
            "localization",
            "float",
            "editable",
            "TOP_PROFILE_FRACTION",
            min=0.05,
            max=0.95,
            step=0.01,
            unit="fraction",
            affects_preview=True,
        ),
        e(
            "localization.row_fraction",
            "localization",
            "float",
            "editable",
            "ROW_PROFILE_FRACTION",
            min=0.05,
            max=0.95,
            step=0.01,
            unit="fraction",
            affects_preview=True,
        ),
        # 7.2 Photometric normalization
        e(
            "normalization.enabled",
            "normalization",
            "bool",
            "editable",
            "NORMALIZE_LIGHTING",
            affects_preview=True,
        ),
        e(
            "normalization.reference_patch",
            "normalization",
            "rect",
            "editable",
            "REFERENCE_PATCH",
            affects_preview=True,
            constraints=["x0 < x1", "y0 < y1"],
            **REL,
        ),
        # 7.3 Bottle crop
        e("bottle.crop_size", "bottle", "size", "locked", "BOTTLE_SIZE", unit="px"),
        # 7.4 ROIs
        *[
            e(
                f"rois.{roi}",
                "rois",
                "range",
                "editable",
                f'ROIS["{roi}"]',
                affects_preview=True,
                constraints=["start < end"],
                **REL,
            )
            for roi in ("cap", "label", "body")
        ],
        *[
            e(f"rois.{roi}_size", "rois", "size", "locked", f'ROI_SIZE["{roi}"]', unit="px")
            for roi in ("cap", "label", "body")
        ],
        # 7.4b Automatic regions: learned from the training images, never edited
        e("auto_rois.bands", "auto_rois", "ranges", "locked", "-", **REL),
        # 7.5 HOG
        e("hog.orientations", "hog", "int", "locked", "HOG_ORIENTATIONS", unit="bins"),
        *[
            e(f"hog.cell_{region}", "hog", "size", "locked", f'HOG_CELL["{region}"]', unit="px")
            for region in HOG_CELL
        ],
        e("hog.block", "hog", "size", "locked", "features.HOG_BLOCK", unit="cells"),
        # LBP
        e("lbp.points", "lbp", "int", "locked", "LBP_POINTS", unit="points"),
        e("lbp.radius", "lbp", "int", "locked", "LBP_RADIUS", unit="px"),
        *[
            e(f"lbp.cell_{region}", "lbp", "size", "locked", f'LBP_CELL["{region}"]', unit="px")
            for region in LBP_CELL
        ],
        # 7.6 Colour
        e("color.hsv_bins", "color", "int3", "locked", "HSV_HISTOGRAM_BINS", unit="bins"),
        # 7.7 Feature groups
        e("features.groups", "features", "multiselect", "locked", "FEATURE_GROUPS"),
        # 7.8 Verdict (stored in profile.verdict, not in the params)
        e(
            "verdict.reject_classes",
            "verdict",
            "multiselect",
            "editable",
            "OK_CLASS",
            options=list(CLASS_NAMES),
        ),
        e(
            "verdict.min_confidence",
            "verdict",
            "float",
            "editable",
            "-",
            min=0.0,
            max=1.0,
            step=0.01,
            nullable=True,
        ),
        e(
            "verdict.low_confidence_verdict",
            "verdict",
            "enum",
            "editable",
            "-",
            options=["reject", "review"],
        ),
    ]


CATALOG: list[CatalogEntry] = _entries()
CATALOG_BY_KEY: dict[str, CatalogEntry] = {entry.key: entry for entry in CATALOG}
PARAM_KEYS: list[str] = [entry.key for entry in CATALOG if entry.group != "verdict"]
"""Keys stored in profile.params (every InspectionParams field)."""
EDITABLE_KEYS: list[str] = [k for k in PARAM_KEYS if CATALOG_BY_KEY[k].mode == "editable"]
LOCKED_KEYS: list[str] = [k for k in PARAM_KEYS if CATALOG_BY_KEY[k].mode == "locked"]

VERDICT_DEFAULTS: dict[str, Any] = {
    "reject_classes": [name for name in CLASS_NAMES if name != "bottle_no_defect"],
    "min_confidence": None,
    "low_confidence_verdict": "review",
}


def catalog_json() -> dict[str, Any]:
    """The catalog as served by GET /api/params/catalog."""
    defaults = InspectionParams.defaults().to_flat_dict()
    entries = []
    for entry in CATALOG:
        item = asdict(entry)
        item["label_key"], item["help_key"] = entry.label_key, entry.help_key
        if entry.group == "verdict":
            item["default"] = VERDICT_DEFAULTS[entry.key.split(".", 1)[1]]
        else:
            item["default"] = defaults[entry.key]
        if entry.key == "features.groups":
            item["options"] = list(defaults["features.groups"])
        entries.append(item)
    return {"groups": GROUPS, "entries": entries}


# =============================================================================
# Validation
# =============================================================================


@dataclass(frozen=True)
class ParamError:
    """One validation error; `code` is translated by the frontend (REQ-L4)."""

    key: str
    code: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        return {"key": self.key, "code": self.code, "params": self.params}


def _is_number(value: Any) -> bool:
    """True for int/float but not bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_numbers(entry: CatalogEntry, values: list[Any], integer: bool) -> list[ParamError]:
    """Type and min/max checks of a list of numbers."""
    errors = []
    for value in values:
        if not _is_number(value) or (integer and not float(value).is_integer()):
            return [ParamError(entry.key, "param.invalid_type", {"expected": entry.type})]
        if entry.min is not None and value < entry.min:
            errors.append(ParamError(entry.key, "param.below_min", {"min": entry.min}))
        if entry.max is not None and value > entry.max:
            errors.append(ParamError(entry.key, "param.above_max", {"max": entry.max}))
    return errors[:1]


TYPE_LENGTHS: dict[str, int] = {"range": 2, "rect": 4, "size": 2, "int3": 3}


def _validate_ranges(entry: CatalogEntry, value: Any) -> list[ParamError]:
    """Checks a variable-length list of (start, end) pairs, each one inside min..max."""
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in value
    ):
        return [ParamError(entry.key, "param.invalid_type", {"expected": "ranges"})]
    for item in value:
        errors = _check_numbers(entry, item, integer=False)
        if errors:
            return errors
        if not item[0] < item[1]:
            return [ParamError(entry.key, "param.range_order")]
    return []


def validate_entry(entry: CatalogEntry, value: Any) -> list[ParamError]:
    """Checks one value against its catalog entry (types, limits, single-field constraints)."""
    if value is None:
        return [] if entry.nullable else [ParamError(entry.key, "param.required")]
    kind = entry.type
    if kind == "bool":
        ok = isinstance(value, bool)
        return [] if ok else [ParamError(entry.key, "param.invalid_type", {"expected": "bool"})]
    if kind in ("int", "float"):
        return _check_numbers(entry, [value], integer=kind == "int")
    if kind == "enum":
        if value not in (entry.options or []):
            return [ParamError(entry.key, "param.invalid_option", {"options": entry.options})]
        return []
    if kind == "multiselect":
        options = entry.options
        if entry.key == "features.groups":
            options = list(InspectionParams.defaults().features.groups)
        if not isinstance(value, list) or any(v not in (options or []) for v in value):
            return [ParamError(entry.key, "param.invalid_option", {"options": options})]
        if len(set(value)) != len(value):
            return [ParamError(entry.key, "param.duplicate_option")]
        return []
    if kind == "ranges":
        return _validate_ranges(entry, value)
    if not isinstance(value, list) or len(value) != TYPE_LENGTHS[kind]:
        return [ParamError(entry.key, "param.invalid_length", {"length": TYPE_LENGTHS[kind]})]
    errors = _check_numbers(entry, value, integer=kind in ("size", "int3"))
    if errors:
        return errors
    if kind == "size" and any(v <= 0 for v in value):
        return [ParamError(entry.key, "param.below_min", {"min": 1})]
    if kind == "range" and not value[0] < value[1]:
        return [ParamError(entry.key, "param.range_order")]
    if kind == "rect" and not (value[0] < value[2] and value[1] < value[3]):
        return [ParamError(entry.key, "param.rect_order")]
    return []


def _cell_scope(region: str) -> str:
    """Suffix of the `hog.cell_*` key of a region (the automatic regions share `auto`)."""
    return "auto" if region in AUTO_ROIS else region


def validate_params(
    flat: dict[str, Any], training: dict[str, Any] | None = None
) -> list[ParamError]:
    """Validates a complete flat param dict; with `training`, locked keys must match it."""
    errors: list[ParamError] = []
    for key in flat:
        if key not in CATALOG_BY_KEY or key.startswith("verdict."):
            errors.append(ParamError(key, "param.unknown"))
    for key in PARAM_KEYS:
        if key not in flat:
            errors.append(ParamError(key, "param.required"))
            continue
        entry_errors = validate_entry(CATALOG_BY_KEY[key], flat[key])
        errors.extend(entry_errors)
        if (
            not entry_errors
            and training is not None
            and CATALOG_BY_KEY[key].mode == "locked"
            and flat[key] != training.get(key)
        ):
            errors.append(ParamError(key, "param.locked", {"training": training.get(key)}))
    if errors:
        return errors
    params = to_params(flat)
    for region, size in params.region_sizes().items():
        cell, block = params.hog.cells()[region], params.hog.block
        if any(s // c < b for s, c, b in zip(size, cell, block)):
            errors.append(
                ParamError(
                    f"hog.cell_{_cell_scope(region)}", "param.hog_block_fit", {"roi": region}
                )
            )
    return errors


def validate_verdict(verdict: dict[str, Any]) -> list[ParamError]:
    """Validates the verdict settings of a profile."""
    errors: list[ParamError] = []
    unknown = set(verdict) - set(VERDICT_DEFAULTS)
    errors += [ParamError(f"verdict.{k}", "param.unknown") for k in sorted(unknown)]
    for name in VERDICT_DEFAULTS:
        key = f"verdict.{name}"
        if name not in verdict:
            errors.append(ParamError(key, "param.required"))
        else:
            errors += validate_entry(CATALOG_BY_KEY[key], verdict[name])
    return errors


def ordered_reject_classes(classes: list[str]) -> list[str]:
    """Reject classes sorted in id order."""
    return [name for name in CLASS_NAMES if name in classes]


def to_params(flat: dict[str, Any]) -> InspectionParams:
    """InspectionParams from a validated flat dict."""
    return InspectionParams.from_dict({k: v for k, v in flat.items() if k in PARAM_KEYS})


def diff_keys(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> list[str]:
    """Keys whose values differ between two flat dicts."""
    return [key for key in keys if a.get(key) != b.get(key)]
