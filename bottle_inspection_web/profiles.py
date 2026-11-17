"""Profiles (recipes): seeding, derived status, versions and diffs (§6, §8.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from bottle_inspection.config import FEATURE_VECTORS

from .db.models import Profile, ProfileVersion, TrainedModel, as_utc, utcnow
from .models.discovery import split_model_ref
from .params_catalog import (
    EDITABLE_KEYS,
    LOCKED_KEYS,
    PARAM_KEYS,
    VERDICT_DEFAULTS,
    diff_keys,
)

PROFILE_COLORS: list[str] = [
    "#2457C5",
    "#0B7A75",
    "#7A4CC2",
    "#B5470F",
    "#1E7B34",
    "#A61E4D",
    "#5C6B7A",
    "#8F6A00",
]
"""The 8 preset accent colours of profiles."""

ALGORITHM_SHORT: dict[str, str] = {"svm": "SVM", "knn": "KNN", "random_forest": "RF"}
"""Short algorithm names used in the names of the seed profiles."""

VERSIONED_FIELDS: tuple[str, ...] = ("model_ref", "params", "background_source", "verdict")
"""A saved change to any of these increments the profile version (REQ-PR1)."""

BACKGROUND_SOURCES: tuple[str, ...] = ("model", "captured", "uploaded")


@dataclass
class ProfileStatus:
    """Derived status of a profile (never stored)."""

    status: str
    can_analyse: bool
    modified_keys: list[str] = field(default_factory=list)
    locked_mismatch: list[str] = field(default_factory=list)
    reason: str | None = None
    model_updated_on: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "status": self.status,
            "can_analyse": self.can_analyse,
            "modified_keys": self.modified_keys,
            "locked_mismatch": self.locked_mismatch,
            "reason": self.reason,
            "model_updated_on": iso(self.model_updated_on),
        }


def iso(value: datetime | None) -> str | None:
    """ISO-8601 UTC string with a Z suffix."""
    value = as_utc(value)
    if value is None:
        return None
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def derive_status(profile: Profile, model: TrainedModel | None) -> ProfileStatus:
    """ready / modified / model_missing / incompatible (§8.2)."""
    if model is None or model.missing or model.load_error:
        reason = "profile.reason.model_error" if model and model.load_error else None
        return ProfileStatus(
            "model_missing", False, reason=reason or "profile.reason.model_missing"
        )
    training = model.training_params_json
    locked = diff_keys(profile.params_json, training, LOCKED_KEYS)
    length_mismatch = (
        profile.feature_length is not None
        and model.feature_length
        and profile.feature_length != model.feature_length
    )
    updated_on = model.updated_on if model.updated_on else None
    if locked or length_mismatch:
        return ProfileStatus(
            "incompatible",
            False,
            locked_mismatch=locked or ["feature_length"],
            reason="profile.reason.incompatible",
            model_updated_on=updated_on,
        )
    modified = diff_keys(profile.params_json, training, EDITABLE_KEYS)
    if profile.background_source != "model":
        modified.append("background_source")
    status = "modified" if modified else "ready"
    return ProfileStatus(status, True, modified_keys=modified, model_updated_on=updated_on)


def snapshot(profile: Profile) -> dict[str, Any]:
    """The versioned settings of a profile."""
    return {
        "model_ref": profile.model_ref,
        "algorithm": profile.algorithm,
        "params": dict(profile.params_json),
        "background_source": profile.background_source,
        "background_path": profile.background_path,
        "verdict": dict(profile.verdict_json),
        "feature_length": profile.feature_length,
    }


def record_version(session: Session, profile: Profile) -> None:
    """Stores the snapshot of the profile's current version."""
    session.add(
        ProfileVersion(
            profile_id=profile.id, version=profile.version, snapshot_json=snapshot(profile)
        )
    )


def snapshot_diff(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-level differences between two snapshots (params compared key by key)."""
    changes = []
    for key in ("model_ref", "background_source"):
        if a.get(key) != b.get(key):
            changes.append({"key": key, "from": a.get(key), "to": b.get(key)})
    params_a, params_b = a.get("params", {}), b.get("params", {})
    for key in PARAM_KEYS:
        if params_a.get(key) != params_b.get(key):
            changes.append({"key": key, "from": params_a.get(key), "to": params_b.get(key)})
    verdict_a, verdict_b = a.get("verdict", {}), b.get("verdict", {})
    for name in VERDICT_DEFAULTS:
        if verdict_a.get(name) != verdict_b.get(name):
            changes.append(
                {"key": f"verdict.{name}", "from": verdict_a.get(name), "to": verdict_b.get(name)}
            )
    return changes


def unique_name(session: Session, base: str) -> str:
    """`base`, or `base (2)`, `base (3)`... when the name is taken (deleted profiles too)."""
    taken = set(session.scalars(select(Profile.name)))
    if base not in taken:
        return base[:60]
    n = 2
    while f"{base} ({n})"[:60] in taken:
        n += 1
    return f"{base} ({n})"[:60]


SCOPE_DESCRIPTIONS: dict[str, str] = {
    "rois": "ROIs: one descriptor over the three hand-picked regions (cap, label, body).",
    "auto_rois": "Automatic regions: one descriptor over the three regions learned from the "
    "training images.",
    "bottle": "Whole bottle: one descriptor over the whole detected bottle (no ROI).",
}
"""Description of the seed profile of each feature vector scope (`vector.rpartition("_")[0]`)."""


def seed_name(model: TrainedModel) -> str:
    """e.g. `SVM rois_hog (results)`, `KNN auto_rois_lbp (results)` or
    `RF bottle_color (results)`: the vector name tells the scope apart."""
    vector = (model.metrics_json or {}).get("feature_vector")
    folder, algorithm = split_model_ref(model.model_ref, vector)
    short = ALGORITHM_SHORT.get(algorithm, algorithm.upper())
    return f"{short} {vector} ({folder})" if vector else f"{short} ({folder})"


def current_params(params: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    """`params` restricted to the catalog keys, with the missing keys taken from the training
    values of the model."""
    return {key: params.get(key, training.get(key)) for key in PARAM_KEYS}


def seed_profiles(session: Session) -> list[Profile]:
    """One built-in profile per usable trained model of a known feature vector that has none
    yet (REQ-PR2): every vector is a complete view of the bottle.
    """
    created = []
    with_builtin = set(
        session.scalars(select(Profile.model_ref).where(Profile.is_builtin.is_(True)))
    )
    count = len(list(session.scalars(select(Profile.id))))
    models = session.scalars(select(TrainedModel).order_by(TrainedModel.model_ref))
    for model in models:
        if model.missing or model.load_error or model.model_ref in with_builtin:
            continue
        vector = str((model.metrics_json or {}).get("feature_vector") or "")
        if vector not in FEATURE_VECTORS:
            continue
        profile = Profile(
            name=unique_name(session, seed_name(model)),
            description=SCOPE_DESCRIPTIONS.get(vector.rpartition("_")[0], ""),
            color=PROFILE_COLORS[count % len(PROFILE_COLORS)],
            model_ref=model.model_ref,
            algorithm=model.algorithm,
            params_json=dict(model.training_params_json),
            verdict_json=dict(VERDICT_DEFAULTS),
            background_source="model",
            version=1,
            is_builtin=True,
            feature_length=model.feature_length,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.flush()
        record_version(session, profile)
        created.append(profile)
        count += 1
    return created
