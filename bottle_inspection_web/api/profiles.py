"""Profiles: CRUD, duplicate, export/import, versions/diff, relink, preview, background."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..context import AppContext, get_ctx
from ..db.models import Profile, ProfileVersion, TrainedModel, utcnow
from ..engine.analysis import StepTimer, analyse, geometry_of, prepare_timed
from ..errors import ApiError
from ..params_catalog import (
    EDITABLE_KEYS,
    LOCKED_KEYS,
    PARAM_KEYS,
    VERDICT_DEFAULTS,
    diff_keys,
    ordered_reject_classes,
    to_params,
    validate_params,
    validate_verdict,
)
from ..profiles import (
    PROFILE_COLORS,
    VERSIONED_FIELDS,
    current_params,
    derive_status,
    iso,
    record_version,
    snapshot,
    snapshot_diff,
    unique_name,
)
from .common import parse_json_field, read_upload, source_image
from .models import model_json

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

EXPORT_FORMAT = "bottle_inspection.profile"


class ProfileIn(BaseModel):
    """Body of POST /api/profiles."""

    name: str = Field(min_length=1, max_length=60)
    description: str = ""
    color: str | None = None
    model_ref: str
    params: dict[str, Any] | None = None
    """Flat (dot path) values; missing keys start from the model's training values."""
    verdict: dict[str, Any] | None = None
    background_source: Literal["model"] = "model"


class ProfilePatch(BaseModel):
    """Body of PATCH /api/profiles/{id} (every field optional; params/verdict are merged)."""

    name: str | None = Field(default=None, min_length=1, max_length=60)
    description: str | None = None
    color: str | None = None
    model_ref: str | None = None
    params: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None
    background_source: Literal["model", "captured", "uploaded"] | None = None


class ProfileImport(BaseModel):
    """An exported profile (parameters and model_ref, never the model file)."""

    format: str = EXPORT_FORMAT
    format_version: int = 1
    name: str = Field(min_length=1, max_length=60)
    description: str = ""
    color: str | None = None
    model_ref: str
    algorithm: str | None = None
    params: dict[str, Any]
    verdict: dict[str, Any] | None = None


# ----------------------------------------------------------------- helpers


def get_profile(session: Any, profile_id: int, include_deleted: bool = False) -> Profile:
    """A profile by id, 404 when unknown (or deleted)."""
    profile = session.get(Profile, profile_id)
    if profile is None or (profile.deleted_at is not None and not include_deleted):
        raise ApiError(404, "profile.not_found", "profile not found", {"id": profile_id})
    return profile


def get_model_row(session: Any, ref: str) -> TrainedModel:
    """A usable trained model row, 422 when it does not exist (REQ-PR3)."""
    row = session.scalars(select(TrainedModel).where(TrainedModel.model_ref == ref)).first()
    if row is None or row.missing or row.load_error:
        raise ApiError(
            422, "profile.model_not_found", "trained model not found", {"model_ref": ref}
        )
    return row


def check_name(session: Any, name: str, exclude_id: int | None = None) -> None:
    """Names are unique among all profiles (deleted ones too, as they stay in reports)."""
    query = select(Profile).where(Profile.name == name)
    other = session.scalars(query).first()
    if other is not None and other.id != exclude_id:
        raise ApiError(409, "profile.name_taken", "a profile with this name exists", {"name": name})


def check_color(color: str | None) -> None:
    """Profile colours come from the 8 presets."""
    if color is not None and color.upper() not in PROFILE_COLORS:
        raise ApiError(422, "profile.invalid_color", "invalid colour", {"options": PROFILE_COLORS})


def validated(params: dict[str, Any], verdict: dict[str, Any], model: TrainedModel) -> None:
    """Validates params against the catalog and the model's locked values, and the verdict."""
    errors = validate_params(params, model.training_params_json) + validate_verdict(verdict)
    if errors:
        raise ApiError(
            422,
            "profile.invalid_params",
            "invalid parameters",
            {"errors": [e.to_json() for e in errors]},
        )


def normalize_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    """Verdict with the reject classes in id order."""
    result = {**VERDICT_DEFAULTS, **verdict}
    if isinstance(result.get("reject_classes"), list):
        result["reject_classes"] = ordered_reject_classes(result["reject_classes"])
    return result


def profile_json(
    profile: Profile, model: TrainedModel | None, full: bool = False
) -> dict[str, Any]:
    """JSON form of a profile with its derived status."""
    status = derive_status(profile, model)
    data: dict[str, Any] = {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "color": profile.color,
        "model_ref": profile.model_ref,
        "algorithm": profile.algorithm,
        "background_source": profile.background_source,
        "has_background": bool(profile.background_path),
        "version": profile.version,
        "status": status.status,
        "status_detail": status.to_json(),
        "is_builtin": profile.is_builtin,
        "deleted": profile.deleted_at is not None,
        "created_at": iso(profile.created_at),
        "updated_at": iso(profile.updated_at),
        "model": model_json(model) if model is not None else None,
        "params": profile.params_json,
        "verdict": profile.verdict_json,
    }
    if full:
        data["training_params"] = model.training_params_json if model else None
    return data


def model_row_for(session: Any, profile: Profile) -> TrainedModel | None:
    """The trained model row a profile points to (None when unknown)."""
    return session.scalars(
        select(TrainedModel).where(TrainedModel.model_ref == profile.model_ref)
    ).first()


def apply_versioned_change(session: Any, profile: Profile, before: dict[str, Any]) -> None:
    """Increments the version and records a snapshot when a versioned field changed."""
    after = snapshot(profile)
    if any(before.get(f) != after.get(f) for f in VERSIONED_FIELDS) or (
        before.get("background_path") != after.get("background_path")
    ):
        profile.version += 1
        record_version(session, profile)
    profile.updated_at = utcnow()


# ----------------------------------------------------------------- CRUD


@router.get("")
def list_profiles(
    include_deleted: bool = False, ctx: AppContext = Depends(get_ctx)
) -> list[dict[str, Any]]:
    """Every profile with its derived status."""
    with ctx.db.session() as session:
        query = select(Profile).order_by(Profile.is_builtin.desc(), Profile.id)
        if not include_deleted:
            query = query.where(Profile.deleted_at.is_(None))
        models = {m.model_ref: m for m in session.scalars(select(TrainedModel))}
        return [profile_json(p, models.get(p.model_ref)) for p in session.scalars(query)]


@router.post("", status_code=201)
def create_profile(body: ProfileIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Creates a profile from a trained model (parameters start from its training values)."""
    check_color(body.color)
    with ctx.db.session() as session:
        model = get_model_row(session, body.model_ref)
        check_name(session, body.name)
        params = {**model.training_params_json, **(body.params or {})}
        verdict = normalize_verdict(body.verdict or {})
        validated(params, verdict, model)
        count = len(list(session.scalars(select(Profile.id))))
        profile = Profile(
            name=body.name.strip(),
            description=body.description,
            color=(body.color or PROFILE_COLORS[count % len(PROFILE_COLORS)]).upper(),
            model_ref=model.model_ref,
            algorithm=model.algorithm,
            params_json=params,
            verdict_json=verdict,
            background_source="model",
            version=1,
            is_builtin=False,
            feature_length=model.feature_length,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.flush()
        record_version(session, profile)
        return profile_json(profile, model, full=True)


@router.get("/{profile_id}")
def read_profile(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """One profile, with the training params of its model."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id, include_deleted=True)
        return profile_json(profile, model_row_for(session, profile), full=True)


@router.patch("/{profile_id}")
def patch_profile(
    profile_id: int, body: ProfilePatch, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Edits a profile; a change to model, params, background or verdict makes a new version."""
    check_color(body.color)
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        before = snapshot(profile)
        model = model_row_for(session, profile)
        if body.name is not None and body.name.strip() != profile.name:
            check_name(session, body.name.strip(), exclude_id=profile.id)
            profile.name = body.name.strip()
        if body.description is not None:
            profile.description = body.description
        if body.color is not None:
            profile.color = body.color.upper()
        params = dict(profile.params_json)
        if body.model_ref is not None and body.model_ref != profile.model_ref:
            model = get_model_row(session, body.model_ref)
            profile.model_ref, profile.algorithm = model.model_ref, model.algorithm
            profile.feature_length = model.feature_length
            for key in LOCKED_KEYS:
                params[key] = model.training_params_json.get(key)
        if (
            model is not None
            and not (model.missing or model.load_error)
            and (body.params is not None or body.model_ref is not None)
        ):
            # Keys the catalog does not have are dropped and missing ones start from the
            # training values (otherwise the profile could never be validated again).
            params = current_params(params, model.training_params_json)
        if body.params is not None:
            params.update(body.params)
        verdict = normalize_verdict({**profile.verdict_json, **(body.verdict or {})})
        if body.background_source is not None:
            if body.background_source != "model" and not profile.background_path:
                raise ApiError(
                    422, "profile.background_missing", "capture or upload a background first"
                )
            profile.background_source = body.background_source
        if body.params is not None or body.verdict is not None or body.model_ref is not None:
            if model is None or model.missing or model.load_error:
                raise ApiError(409, "profile.model_missing", "the trained model is missing")
            validated(params, verdict, model)
        profile.params_json, profile.verdict_json = params, verdict
        apply_versioned_change(session, profile, before)
        return profile_json(profile, model, full=True)


@router.delete("/{profile_id}")
def delete_profile(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Soft delete: hidden, but its analyses stay in the reports. Built-ins cannot be deleted."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        if profile.is_builtin:
            raise ApiError(409, "profile.builtin", "built-in profiles cannot be deleted")
        profile.deleted_at = utcnow()
        return {"id": profile.id, "deleted": True}


@router.post("/{profile_id}/duplicate", status_code=201)
def duplicate_profile(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """A copy of the profile (not built-in), named "<name> (copy)"."""
    with ctx.db.session() as session:
        source = get_profile(session, profile_id)
        count = len(list(session.scalars(select(Profile.id))))
        profile = Profile(
            name=unique_name(session, f"{source.name} (copy)"[:60]),
            description=source.description,
            color=PROFILE_COLORS[count % len(PROFILE_COLORS)],
            model_ref=source.model_ref,
            algorithm=source.algorithm,
            params_json=dict(source.params_json),
            verdict_json=dict(source.verdict_json),
            background_source=source.background_source,
            background_path=source.background_path,
            version=1,
            is_builtin=False,
            feature_length=source.feature_length,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.flush()
        record_version(session, profile)
        return profile_json(profile, model_row_for(session, profile), full=True)


@router.post("/{profile_id}/relink")
def relink_profile(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Copies the model's current locked values into an incompatible profile (keeps editables)."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        model = get_model_row(session, profile.model_ref)
        before = snapshot(profile)
        params = current_params(profile.params_json, model.training_params_json)
        for key in LOCKED_KEYS:
            params[key] = model.training_params_json.get(key)
        profile.params_json = params
        profile.feature_length = model.feature_length
        apply_versioned_change(session, profile, before)
        return profile_json(profile, model, full=True)


# ----------------------------------------------------------------- export / import


@router.get("/{profile_id}/export")
def export_profile(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Parameters and model_ref as JSON (never the model file)."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id, include_deleted=True)
        return {
            "format": EXPORT_FORMAT,
            "format_version": 1,
            "name": profile.name,
            "description": profile.description,
            "color": profile.color,
            "model_ref": profile.model_ref,
            "algorithm": profile.algorithm,
            "params": profile.params_json,
            "verdict": profile.verdict_json,
            "exported_at": iso(utcnow()),
        }


@router.post("/import")
def import_profile(
    body: ProfileImport, dry_run: bool = True, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Validates an exported profile; with dry_run=false creates it. Returns a preview diff."""
    if body.format != EXPORT_FORMAT:
        raise ApiError(422, "profile.import_format", "not an exported profile")
    check_color(body.color)
    with ctx.db.session() as session:
        model = get_model_row(session, body.model_ref)
        params = {k: v for k, v in body.params.items() if k in PARAM_KEYS}
        unknown = sorted(set(body.params) - set(PARAM_KEYS))
        if unknown:
            raise ApiError(
                422,
                "profile.invalid_params",
                "unknown parameters",
                {"errors": [{"key": k, "code": "param.unknown", "params": {}} for k in unknown]},
            )
        params = {**model.training_params_json, **params}
        verdict = normalize_verdict(body.verdict or {})
        validated(params, verdict, model)
        diff = [
            {"key": k, "training": model.training_params_json.get(k), "value": params.get(k)}
            for k in diff_keys(params, model.training_params_json, EDITABLE_KEYS)
        ]
        name = unique_name(session, body.name.strip())
        preview = {
            "name": name,
            "model_ref": model.model_ref,
            "algorithm": model.algorithm,
            "diff_from_training": diff,
            "verdict": verdict,
        }
        if dry_run:
            return {"dry_run": True, "preview": preview}
        count = len(list(session.scalars(select(Profile.id))))
        profile = Profile(
            name=name,
            description=body.description,
            color=(body.color or PROFILE_COLORS[count % len(PROFILE_COLORS)]).upper(),
            model_ref=model.model_ref,
            algorithm=model.algorithm,
            params_json=params,
            verdict_json=verdict,
            background_source="model",
            version=1,
            is_builtin=False,
            feature_length=model.feature_length,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.flush()
        record_version(session, profile)
        return {
            "dry_run": False,
            "preview": preview,
            "profile": profile_json(profile, model, full=True),
        }


# ----------------------------------------------------------------- versions


@router.get("/{profile_id}/versions")
def list_versions(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Every version, newest first, with the changes from the previous version."""
    with ctx.db.session() as session:
        get_profile(session, profile_id, include_deleted=True)
        rows = session.scalars(
            select(ProfileVersion)
            .where(ProfileVersion.profile_id == profile_id)
            .order_by(ProfileVersion.version)
        ).all()
        result, previous = [], None
        for row in rows:
            changes = snapshot_diff(previous, row.snapshot_json) if previous else []
            result.append(
                {
                    "version": row.version,
                    "created_at": iso(row.created_at),
                    "changes": changes,
                    "snapshot": row.snapshot_json,
                }
            )
            previous = row.snapshot_json
        return list(reversed(result))


def version_snapshot(session: Any, profile_id: int, version: int) -> dict[str, Any]:
    """Snapshot of one version (404 when unknown)."""
    row = session.scalars(
        select(ProfileVersion).where(
            ProfileVersion.profile_id == profile_id, ProfileVersion.version == version
        )
    ).first()
    if row is None:
        raise ApiError(404, "profile.version_not_found", "version not found", {"version": version})
    return row.snapshot_json


@router.get("/{profile_id}/diff")
def diff_versions(
    profile_id: int,
    from_: int | None = Query(default=None, alias="from"),
    to: int | None = None,
    other_profile: int | None = None,
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Diff between two versions (?from=&to=) or with another profile (?other_profile=)."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id, include_deleted=True)
        current = snapshot(profile)
        if other_profile is not None:
            other = snapshot(get_profile(session, other_profile, include_deleted=True))
            return {
                "from": f"profile:{other_profile}",
                "to": "current",
                "changes": snapshot_diff(other, current),
            }
        a = version_snapshot(session, profile_id, from_) if from_ is not None else current
        b = version_snapshot(session, profile_id, to) if to is not None else current
        return {"from": from_, "to": to, "changes": snapshot_diff(a, b)}


@router.post("/{profile_id}/versions/{version}/restore")
def restore_version(
    profile_id: int, version: int, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Applies an old version's settings as a new version."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        old = version_snapshot(session, profile_id, version)
        model = get_model_row(session, old["model_ref"])
        before = snapshot(profile)
        validated(old["params"], normalize_verdict(old["verdict"]), model)
        profile.model_ref, profile.algorithm = model.model_ref, model.algorithm
        profile.params_json = dict(old["params"])
        profile.verdict_json = normalize_verdict(old["verdict"])
        profile.background_source = old.get("background_source", "model")
        profile.background_path = old.get("background_path")
        profile.feature_length = old.get("feature_length", model.feature_length)
        apply_versioned_change(session, profile, before)
        return profile_json(profile, model, full=True)


# ----------------------------------------------------------------- preview


def png_b64(mask: np.ndarray) -> str:
    """A boolean/uint8 mask as a base64 PNG (white = inside)."""
    data = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
    ok, buffer = cv2.imencode(".png", data)
    return base64.b64encode(buffer.tobytes()).decode() if ok else ""


def jpeg_b64(image: np.ndarray, width: int = 960) -> str:
    """A BGR image resized to `width` as a base64 JPEG."""
    height = int(image.shape[0] * width / image.shape[1])
    ok, buffer = cv2.imencode(
        ".jpg", cv2.resize(image, (width, height)), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    return base64.b64encode(buffer.tobytes()).decode() if ok else ""


@router.post("/{profile_id}/preview")
async def preview(
    profile_id: int,
    file: UploadFile | None = File(default=None),
    image_id: int | None = Form(default=None),
    dataset_ref: str | None = Form(default=None),
    params: str | None = Form(default=None),
    scale: float = Form(default=1.0),
    include: str = Form(default="masks"),
    classify: bool = Form(default=False),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Geometry and masks for unsaved params (live preview); optional dry-run class."""
    image, _decode_ms, truth = await source_image(ctx, file, image_id, dataset_ref)
    overrides = parse_json_field(params, "params") or {}
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        model = model_row_for(session, profile)
        if model is None or model.missing or model.load_error:
            raise ApiError(409, "profile.model_missing", "the trained model is missing")
        flat = {**profile.params_json, **overrides}
        errors = validate_params(flat, model.training_params_json)
        if errors:
            raise ApiError(
                422,
                "profile.invalid_params",
                "invalid parameters",
                {"errors": [e.to_json() for e in errors]},
            )
        session.expunge(profile)
    runtime = ctx.inspections.runtime_for(profile, params_override=flat)
    p = to_params(flat)
    work = image
    scale = float(min(max(scale, 0.2), 1.0))
    if scale < 1.0:
        work = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    def compute() -> dict[str, Any]:
        """Runs the preprocessing steps (thread pool)."""
        prepared, _ = prepare_timed(work, runtime.background, p, StepTimer())
        out: dict[str, Any] = {
            "geometry": geometry_of(prepared, p),
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "modified_keys": diff_keys(flat, model.training_params_json, EDITABLE_KEYS),
            "ground_truth": truth,
        }
        parts = set(include.split(","))
        if "masks" in parts:
            out["masks"] = {
                "foreground": png_b64(prepared.foreground_mask),
                "size": list(p.localization.size),
            }
        if "normalized" in parts:
            out["normalized_image"] = jpeg_b64(prepared.image)
        if classify:
            result = analyse(image, runtime)
            out["classification"] = {
                "predicted_class": result.predicted_class,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "scores": {
                    k: (v if np.isfinite(v) else None) for k, v in result.score.scores.items()
                },
                "score_kind": result.score.score_kind,
                "total_ms": result.total_ms,
            }
        return out

    import asyncio

    return await asyncio.get_running_loop().run_in_executor(ctx.executor, compute)


# ----------------------------------------------------------------- background (REQ-M5)


def save_background(
    ctx: AppContext, profile_id: int, image: np.ndarray, source: str
) -> dict[str, Any]:
    """Stores a background at the locked localization size and links it to the profile."""
    with ctx.db.session() as session:
        profile = get_profile(session, profile_id)
        size = tuple(profile.params_json.get("localization.size", [640, 360]))
        folder = ctx.config.paths.backgrounds
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"profile-{profile.id}-{utcnow():%Y%m%dT%H%M%S%f}.png"
        cv2.imwrite(str(path), cv2.resize(image, (int(size[0]), int(size[1]))))
        path.with_suffix(".size").write_text(f"{image.shape[1]}x{image.shape[0]}")
        before = snapshot(profile)
        profile.background_path = str(path)
        profile.background_source = source
        apply_versioned_change(session, profile, before)
        return profile_json(profile, model_row_for(session, profile), full=True)


@router.post("/{profile_id}/background/upload")
async def upload_background(
    profile_id: int, file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Uses an uploaded image of the empty scene as the profile's background."""
    data, _ext = await read_upload(file, float(ctx.settings.get("max_upload_mb")))
    from .common import decode_or_415

    image, _ms = decode_or_415(data, file.filename or "")
    return save_background(ctx, profile_id, image, "uploaded")


class CaptureIn(BaseModel):
    """Body of POST /api/profiles/{id}/background/capture."""

    camera_id: int
    frames: int | None = None


@router.post("/{profile_id}/background/capture")
async def capture_background(
    profile_id: int, body: CaptureIn, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Per-pixel median of N frames of the empty scene from a server camera."""
    cameras = ctx.extras.get("cameras")
    if cameras is None:
        raise ApiError(503, "camera.unavailable", "cameras are not available")
    frames = int(body.frames or ctx.settings.get("background_capture_frames"))
    import asyncio

    stack = await asyncio.get_running_loop().run_in_executor(
        ctx.executor, cameras.grab_frames, body.camera_id, max(1, min(frames, 300))
    )
    median = np.median(np.stack(stack), axis=0).astype(np.uint8)
    return save_background(ctx, profile_id, median, "captured")


@router.get("/{profile_id}/background")
def background_image(profile_id: int, ctx: AppContext = Depends(get_ctx)) -> Any:
    """The current background image of a profile (model, captured or uploaded) as PNG."""
    from fastapi.responses import Response

    with ctx.db.session() as session:
        profile = get_profile(session, profile_id, include_deleted=True)
        source, path = profile.background_source, profile.background_path
        ref = profile.model_ref
    if source != "model" and path and Path(path).is_file():
        image = cv2.imread(path)
    else:
        try:
            image = ctx.registry.load(ref).background
        except KeyError as error:
            raise ApiError(404, "profile.model_missing", "the trained model is missing") from error
    if image is None:
        raise ApiError(404, "profile.background_missing", "background file missing")
    ok, buffer = cv2.imencode(".png", np.asarray(image))
    return Response(buffer.tobytes() if ok else b"", media_type="image/png")
