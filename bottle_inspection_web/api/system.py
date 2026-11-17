"""Health, classes, catalog, settings, notifications, storage and dataset endpoints."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, update

from bottle_inspection.params import library_versions

from .. import __version__
from ..classes import CLASS_NAMES, class_list, ordered_counts
from ..context import AppContext, get_ctx
from ..datasets import dataset_info
from ..db.models import Analysis, Image, Inspection, Notification, utcnow
from ..errors import ApiError
from ..params_catalog import catalog_json
from ..profiles import iso
from ..settings_store import SettingsError

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Status, versions, dataset, cameras and queue sizes."""
    dataset = ctx.registry.dataset_root()
    cameras = ctx.extras.get("cameras")
    rules = ctx.extras.get("rules")
    return {
        "status": "ok",
        "version": __version__,
        "versions": {"app": __version__, **library_versions(), "platform": platform.platform()},
        "dataset_found": dataset is not None,
        "dataset_path": str(dataset) if dataset else None,
        "models": len([r for r in ctx.registry.rows() if ctx.registry.usable(r)]),
        "active_cameras": cameras.active_count() if cameras else 0,
        "active_sessions": cameras.session_count() if cameras else 0,
        "queues": {"actions": rules.queue_size() if rules else 0, "actions_max": 1000},
        "sse_clients": ctx.events.subscriber_count,
    }


@router.get("/classes")
def classes() -> list[dict[str, Any]]:
    """The 7 classes, in id order."""
    return class_list()


@router.get("/params/catalog")
def params_catalog() -> dict[str, Any]:
    """Parameter catalog (§7)."""
    return catalog_json()


@router.get("/settings")
def get_settings(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Every user-editable setting."""
    return ctx.settings.all()


@router.patch("/settings")
def patch_settings(values: dict[str, Any], ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Updates some settings; model folders changes trigger a rediscovery."""
    try:
        updated = ctx.settings.update(values)
    except SettingsError as error:
        raise ApiError(422, error.code, str(error), {"key": error.key}) from error
    if "model_folders" in values or "dataset_path" in values:
        ctx.registry.refresh()
        from ..profiles import seed_profiles

        with ctx.db.session() as session:
            seed_profiles(session)
    return updated


# ----------------------------------------------------------------- notifications


def notification_json(row: Notification) -> dict[str, Any]:
    """JSON form of a notification."""
    return {
        "id": row.id,
        "level": row.level,
        "title": row.title,
        "message": row.message,
        "code": row.code,
        "params": row.params_json,
        "link": row.link,
        "read_at": iso(row.read_at),
        "created_at": iso(row.created_at),
    }


@router.get("/notifications")
def notifications(
    limit: int = 50, unread_only: bool = False, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Latest notifications and the unread count."""
    with ctx.db.session() as session:
        query = select(Notification).order_by(Notification.id.desc()).limit(min(limit, 500))
        if unread_only:
            query = query.where(Notification.read_at.is_(None))
        rows = session.scalars(query).all()
        unread = session.scalar(
            select(func.count(Notification.id)).where(Notification.read_at.is_(None))
        )
        return {"items": [notification_json(r) for r in rows], "unread": int(unread or 0)}


@router.post("/notifications/{notification_id}/read")
def read_notification(notification_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Marks a notification as read."""
    with ctx.db.session() as session:
        row = session.get(Notification, notification_id)
        if row is None:
            raise ApiError(404, "notification.not_found", "notification not found")
        row.read_at = row.read_at or utcnow()
        return notification_json(row)


@router.post("/notifications/read-all")
def read_all_notifications(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Marks every notification as read."""
    with ctx.db.session() as session:
        session.execute(
            update(Notification).where(Notification.read_at.is_(None)).values(read_at=utcnow())
        )
    return {"ok": True}


# ----------------------------------------------------------------- storage (REQ-D1)


def folder_usage(folder: Path) -> tuple[int, int]:
    """(file count, bytes) of every file under `folder`."""
    count, size = 0, 0
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            try:
                size += (Path(dirpath) / name).stat().st_size
                count += 1
            except OSError:
                pass
    return count, size


@router.get("/storage")
def storage(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Disk usage of images and thumbnails and the row counts."""
    paths = ctx.config.paths
    images, images_bytes = folder_usage(paths.images)
    thumbs, thumbs_bytes = folder_usage(paths.thumbs)
    with ctx.db.session() as session:
        n_analyses = session.scalar(select(func.count(Analysis.id))) or 0
        n_inspections = session.scalar(select(func.count(Inspection.id))) or 0
        n_images = session.scalar(select(func.count(Image.id))) or 0
    db_bytes = sum(p.stat().st_size for p in paths.root.glob("app.db*") if p.is_file())
    return {
        "image_files": images + thumbs,
        "image_bytes": images_bytes + thumbs_bytes,
        "images_rows": int(n_images),
        "inspections": int(n_inspections),
        "analyses": int(n_analyses),
        "db_bytes": db_bytes,
        "data_dir": str(paths.root),
    }


class ConfirmImages(BaseModel):
    """Body of DELETE /api/images."""

    confirm: bool


class ConfirmAnalyses(BaseModel):
    """Body of DELETE /api/analyses."""

    confirm: str


def delete_all_images(ctx: AppContext) -> dict[str, Any]:
    """Removes every stored image and thumbnail; analyses and reports are kept (REQ-D1)."""
    paths = ctx.config.paths
    count, size = 0, 0
    with ctx.db.session() as session:
        session.execute(update(Inspection).values(image_id=None))
        rows = session.scalars(select(Image)).all()
        for row in rows:
            path = Path(row.path)
            if path.is_file() and path.resolve().is_relative_to(paths.images.resolve()):
                size += path.stat().st_size
                path.unlink()
                count += 1
            session.delete(row)
    for folder in (paths.thumbs, paths.images):
        for dirpath, _dirs, files in os.walk(folder, topdown=False):
            for name in files:
                file = Path(dirpath) / name
                size += file.stat().st_size
                file.unlink()
                count += 1
            if Path(dirpath) != folder:
                try:
                    Path(dirpath).rmdir()
                except OSError:
                    pass
    return {"deleted_files": count, "freed_bytes": size}


@router.delete("/images")
def delete_images(body: ConfirmImages, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Deletes every stored image (requires {"confirm": true})."""
    if body.confirm is not True:
        raise ApiError(400, "storage.confirm_required", "confirmation required")
    return delete_all_images(ctx)


@router.delete("/analyses")
def delete_analyses(body: ConfirmAnalyses, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Deletes every analysis and inspection (requires {"confirm": "DELETE"})."""
    if body.confirm != "DELETE":
        raise ApiError(400, "storage.confirm_required", "type DELETE to confirm")
    from ..db.models import ActionExecution

    with ctx.db.session() as session:
        session.execute(update(ActionExecution).values(analysis_id=None))
        n = session.query(Analysis).delete()
        session.query(Inspection).delete()
    return {"deleted_analyses": int(n)}


# ----------------------------------------------------------------- dataset


@router.get("/dataset")
def dataset(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Dataset path check and class counts per split (all 7 classes)."""
    root = ctx.registry.dataset_root()
    info = dataset_info(root)
    return {
        "path": ctx.settings.get("dataset_path"),
        "resolved": str(root) if root else None,
        "found": root is not None and info is not None,
        "classes": CLASS_NAMES,
        "counts": (
            {split: ordered_counts(counts) for split, counts in info.items()} if info else None
        ),
    }
