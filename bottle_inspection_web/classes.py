"""The 7 classes, always listed with their exact identifiers and in id order."""

from __future__ import annotations

from typing import Any

from bottle_inspection.config import CLASSES, OK_CLASS
from bottle_inspection.didactic import CLASS_COLORS

CLASS_NAMES: list[str] = list(CLASSES)
"""bottle_no_defect, bottle_other_brand, defect_low_fill, defect_no_cap, defect_loose_cap,
defect_debris, defect_label_damage (ids 0..6)."""

DEFAULT_REJECT_CLASSES: list[str] = [name for name in CLASS_NAMES if name != OK_CLASS]
"""Every class except bottle_no_defect rejects by default."""


def class_list() -> list[dict[str, Any]]:
    """The 7 classes with id, name, default verdict and chart colour (upper-case hex)."""
    return [
        {
            "id": class_id,
            "name": name,
            "default_verdict": "pass" if name == OK_CLASS else "reject",
            "color": CLASS_COLORS[name].upper(),
            "description_key": f"classes.{name}",
        }
        for class_id, name in enumerate(CLASS_NAMES)
    ]


def ordered_scores(values: dict[str, float], fill: float = 0.0) -> dict[str, float]:
    """All 7 classes in id order; classes missing from `values` get `fill`."""
    return {name: float(values.get(name, fill)) for name in CLASS_NAMES}


def ordered_counts(values: dict[str, int]) -> dict[str, int]:
    """All 7 classes in id order with integer counts (0 when absent)."""
    return {name: int(values.get(name, 0)) for name in CLASS_NAMES}
