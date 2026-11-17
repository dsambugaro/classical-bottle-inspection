"""Trained models (read-only): list, refresh, metadata, CLI figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..classes import CLASS_NAMES
from ..context import AppContext, get_ctx
from ..db.models import TrainedModel
from ..errors import ApiError
from ..models.discovery import split_model_ref
from ..profiles import iso, seed_profiles

router = APIRouter(prefix="/api/models", tags=["models"])


def model_json(row: TrainedModel, full: bool = False) -> dict[str, Any]:
    """JSON form of a trained model; `full` adds params, metrics and versions."""
    feature_vector = (row.metrics_json or {}).get("feature_vector")
    folder, algorithm = split_model_ref(row.model_ref, feature_vector)
    validation = (row.metrics_json or {}).get("validation") or {}
    status = "missing" if row.missing else ("error" if row.load_error else "ok")
    data: dict[str, Any] = {
        "model_ref": row.model_ref,
        "folder": folder,
        "feature_vector": feature_vector,
        "algorithm": row.algorithm or algorithm,
        "display_name": (row.metrics_json or {}).get("display_name", algorithm),
        "status": status,
        "file_mtime": row.file_mtime,
        "trained_at": iso_mtime(row.file_mtime),
        "updated_on": iso(row.updated_on),
        "discovered_at": iso(row.discovered_at),
        "accuracy": validation.get("accuracy"),
        "f1_macro": validation.get("f1_macro"),
        "feature_length": row.feature_length,
        "version_warning": row.version_warning,
        "load_error": row.load_error,
        "missing": row.missing,
    }
    if full:
        metrics = dict(row.metrics_json or {})
        per_class = metrics.get("per_class") or {}
        metrics["per_class"] = {name: per_class.get(name, {}) for name in CLASS_NAMES}
        data.update(
            {
                "training_params": row.training_params_json,
                "best_params": row.best_params_json,
                "metrics": metrics,
                "library_versions": row.library_versions_json,
                "classes": CLASS_NAMES,
                "figures": {
                    name: f"/api/models/{row.model_ref}/figures/{name}"
                    for name in metrics.get("figures", [])
                },
            }
        )
    return data


def iso_mtime(mtime: float) -> str:
    """File modification time as ISO-8601 UTC."""
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(mtime, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@router.get("")
def list_models(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Every discovered model (including missing ones)."""
    return [model_json(row) for row in ctx.registry.rows()]


@router.post("/refresh")
def refresh_models(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Rescans the model folders and seeds built-in profiles for new models."""
    changes = ctx.registry.refresh()
    with ctx.db.session() as session:
        created = [p.name for p in seed_profiles(session)]
    return {"changes": changes, "created_profiles": created, "models": list_models(ctx)}


@router.get("/{ref:path}/figures/{name}")
def model_figure(ref: str, name: str, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    """A figure written by the CLI next to the model (confusion matrices)."""
    row = ctx.registry.row(ref)
    if row is None or name not in (row.metrics_json or {}).get("figures", []):
        raise ApiError(404, "model.figure_not_found", "figure not found")
    path = Path(row.path).parent / name
    if not path.is_file():
        raise ApiError(404, "model.figure_not_found", "figure not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{ref:path}")
def get_model(ref: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Metadata, training params and validation metrics (all 7 classes) of one model."""
    row = ctx.registry.row(ref)
    if row is None:
        raise ApiError(404, "model.not_found", "trained model not found", {"model_ref": ref})
    return model_json(row, full=True)
