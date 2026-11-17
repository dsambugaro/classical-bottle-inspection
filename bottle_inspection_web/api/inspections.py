"""Inspections (upload + dataset samples), analyses, images, dataset samples and SSE."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy import asc, desc, func, select

from ..context import AppContext, get_ctx
from ..datasets import find_sample, list_samples
from ..db.models import Analysis, Image, Inspection, Profile, Session, utcnow
from ..errors import ApiError
from ..history import HistoryFilters, history_filters_dependency, history_page
from ..inspections import analysis_json
from ..profiles import iso
from ..reports.filters import AnalysisFilters, filters_dependency
from .common import dataset_root_or_404, decode_or_415, parse_ids, read_upload

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["inspections"])

SYNC_LIMIT: int = 10
"""Up to this many images are analysed synchronously; more become a background session."""

THUMB_WIDTH: int = 240


def session_json(row: Session, done: int | None = None) -> dict[str, Any]:
    """JSON form of a session."""
    return {
        "id": row.id,
        "source": row.source,
        "camera_id": row.camera_id,
        "started_at": iso(row.started_at),
        "ended_at": iso(row.ended_at),
        "profile_ids": row.profile_ids_json,
        "mode": row.mode,
        "fps": row.fps,
        "total": row.total,
        "done": done,
        "note": row.note,
    }


def create_session(ctx: AppContext, source: str, profile_ids: list[int], total: int) -> int:
    """A new session row; returns its id."""
    with ctx.db.session() as session:
        row = Session(source=source, profile_ids_json=profile_ids, total=total, started_at=utcnow())
        session.add(row)
        session.flush()
        return row.id


def end_session(ctx: AppContext, session_id: int) -> None:
    """Marks a session as ended."""
    with ctx.db.session() as session:
        row = session.get(Session, session_id)
        if row is not None and row.ended_at is None:
            row.ended_at = utcnow()


@router.post("/inspections")
async def create_inspections(
    files: list[UploadFile] | None = File(default=None),
    profile_ids: list[str] = Form(...),
    dataset_samples: list[str] | None = Form(default=None),
    image_ids: list[str] | None = Form(default=None),
    ctx: AppContext = Depends(get_ctx),
) -> Any:
    """Analyses uploaded images and/or dataset samples with every selected profile.

    Up to 10 images: synchronous, returns the inspections. More: returns 202 with the
    session; progress and results are streamed via SSE (`analysis.completed`,
    `session.progress`, `session.completed`).
    """
    ids = parse_ids(profile_ids)
    profiles = ctx.inspections.profiles_for(ids)
    for profile in profiles:
        ctx.inspections.runtime_for(profile)  # fails fast on unusable profiles
    uploads = files or []
    refs = [r for r in (dataset_samples or []) if r]
    stored_ids = parse_ids(image_ids)
    total = len(uploads) + len(refs) + len(stored_ids)
    if total == 0:
        raise ApiError(422, "inspection.no_images", "no images given")
    max_files = int(ctx.settings.get("max_batch_files"))
    if total > max_files:
        raise ApiError(413, "upload.too_many_files", "too many files", {"max": max_files})
    source = "dataset" if refs and not uploads and not stored_ids else "upload"
    max_mb = float(ctx.settings.get("max_upload_mb"))

    items: list[tuple[str, bytes, str, str, str | None, int | None]] = []
    """(source, bytes, extension, original name, ground truth, existing image id)."""
    for upload in uploads:
        data, ext = await read_upload(upload, max_mb)
        items.append(("upload", data, ext, upload.filename or "", None, None))
    if refs:
        root = dataset_root_or_404(ctx)
        for ref in refs:
            sample = find_sample(root, ref)
            items.append(
                (
                    "dataset",
                    sample.image_path.read_bytes(),
                    ".jpg",
                    sample.image_path.name,
                    sample.label,
                    None,
                )
            )
    for image_id in stored_ids:
        with ctx.db.session() as db:
            row = db.get(Image, image_id)
            if row is None or not Path(row.path).is_file():
                raise ApiError(404, "image.not_found", "image not found", {"id": image_id})
            path = Path(row.path)
            items.append(
                (
                    row.source,
                    path.read_bytes(),
                    path.suffix,
                    row.original_name,
                    row.ground_truth_class,
                    row.id,
                )
            )

    session_id = create_session(ctx, source, [p.id for p in profiles], total)

    async def run_item(
        item: tuple[str, bytes, str, str, str | None, int | None],
    ) -> dict[str, Any]:
        """Decodes and inspects one image (a stored image is reused, not copied)."""
        item_source, data, ext, name, truth, existing = item
        image, decode_ms = decode_or_415(data, name)
        return await ctx.inspections.inspect(
            image,
            decode_ms,
            profiles,
            source=item_source,
            session_id=session_id,
            image_bytes=data,
            extension=ext,
            original_name=name,
            ground_truth=truth,
            existing_image_id=existing,
        )

    if total <= SYNC_LIMIT:
        try:
            results = [await run_item(item) for item in items]
        finally:
            end_session(ctx, session_id)
        return {"session_id": session_id, "inspections": results}

    async def run_batch() -> None:
        """Sequential processing of a large batch with progress events."""
        done, failed = 0, []
        try:
            for item in items:
                try:
                    await run_item(item)
                except ApiError as error:
                    failed.append({"name": item[3], "code": error.code})
                except Exception:  # noqa: BLE001 - one bad file must not stop the batch
                    log.exception("batch item failed")
                    failed.append({"name": item[3], "code": "inspection.failed"})
                done += 1
                ctx.events.publish(
                    "session.progress",
                    {"session_id": session_id, "done": done, "total": total, "failed": len(failed)},
                )
        finally:
            end_session(ctx, session_id)
            ctx.events.publish(
                "session.completed",
                {"session_id": session_id, "done": done, "total": total, "failed": failed},
            )

    task = asyncio.create_task(run_batch())
    ctx.extras.setdefault("batch_tasks", set()).add(task)
    task.add_done_callback(lambda t: ctx.extras["batch_tasks"].discard(t))
    return JSONResponse(
        {"session_id": session_id, "total": total, "status": "running"}, status_code=202
    )


@router.get("/inspections/{inspection_id}")
def read_inspection(inspection_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """One inspection with all its analyses."""
    return ctx.inspections.inspection_json(inspection_id)


@router.get("/inspections/{inspection_id}/frame")
def inspection_frame(inspection_id: int, ctx: AppContext = Depends(get_ctx)) -> Response:
    """A recent camera frame that was not stored (kept in memory for the live view)."""
    data = ctx.inspections.frame(inspection_id)
    if data is None:
        raise ApiError(404, "image.not_found", "frame no longer available")
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"}
    )


@router.get("/inspections")
def list_inspections(
    limit: int = 50, session: int | None = None, ctx: AppContext = Depends(get_ctx)
) -> list[dict[str, Any]]:
    """Latest inspections (history filmstrip), newest first, with every analysis."""
    with ctx.db.session() as db:
        query = select(Inspection.id).order_by(Inspection.id.desc()).limit(min(limit, 200))
        if session is not None:
            query = query.where(Inspection.session_id == session)
        ids = list(db.scalars(query))
    return [ctx.inspections.inspection_json(i) for i in ids]


@router.get("/history")
def list_history(
    filters: HistoryFilters = Depends(history_filters_dependency),
    before: int | None = None,
    limit: int = 20,
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Inspections newest first (cursor `before` = inspection id), each with all its analyses.

    An inspection is listed when at least one of its analyses matches the filters.
    """
    with ctx.db.session() as session:
        return history_page(session, filters, before, limit)


SORTABLE = {
    "created_at": Analysis.created_at,
    "confidence": Analysis.confidence,
    "total_ms": Analysis.total_ms,
    "predicted_class": Analysis.predicted_class,
    "verdict": Analysis.verdict,
    "profile_id": Analysis.profile_id,
}


@router.get("/analyses")
def list_analyses(
    filters: AnalysisFilters = Depends(filters_dependency),
    limit: int = 50,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Paginated, sortable analyses with the report filters."""
    column = SORTABLE.get(sort, Analysis.created_at)
    direction = asc if order == "asc" else desc
    where = filters.clauses()
    with ctx.db.session() as session:
        total = session.scalar(select(func.count(Analysis.id)).where(*where)) or 0
        rows = session.execute(
            select(Analysis, Profile, Inspection)
            .join(Profile, Profile.id == Analysis.profile_id)
            .join(Inspection, Inspection.id == Analysis.inspection_id)
            .where(*where)
            .order_by(direction(column), direction(Analysis.id))
            .limit(min(max(limit, 1), 500))
            .offset(max(offset, 0))
        ).all()
        image_ids = {i.image_id for _, _, i in rows if i.image_id}
        images = (
            {img.id: img for img in session.scalars(select(Image).where(Image.id.in_(image_ids)))}
            if image_ids
            else {}
        )
        items = [
            analysis_json(a, p, images.get(i.image_id) if i.image_id else None, i)
            for a, p, i in rows
        ]
    return {"total": int(total), "items": items, "limit": limit, "offset": offset}


@router.get("/analyses/{analysis_id}")
def read_analysis(analysis_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """One AnalysisResult."""
    return ctx.inspections.result(analysis_id)


@router.get("/sessions/{session_id}")
def read_session(session_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """A session with the number of inspections done."""
    with ctx.db.session() as session:
        row = session.get(Session, session_id)
        if row is None:
            raise ApiError(404, "session.not_found", "session not found")
        done = session.scalar(
            select(func.count(Inspection.id)).where(Inspection.session_id == session_id)
        )
        return session_json(row, int(done or 0))


# ----------------------------------------------------------------- images


def image_path(ctx: AppContext, image_id: int) -> Path:
    """Path of a stored image (404 when deleted)."""
    with ctx.db.session() as session:
        row = session.get(Image, image_id)
        if row is None:
            raise ApiError(404, "image.not_found", "image not found")
        path = Path(row.path)
    if not path.is_file():
        raise ApiError(404, "image.not_found", "image file not found")
    return path


def thumbnail(source: Path, target: Path, width: int = THUMB_WIDTH) -> Path:
    """Creates (once) a JPEG thumbnail."""
    if not target.exists():
        image = cv2.imread(str(source))
        if image is None:
            raise ApiError(404, "image.not_found", "image cannot be read")
        height = max(1, int(image.shape[0] * width / image.shape[1]))
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(target),
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    return target


@router.get("/images/{image_id}")
def get_image(image_id: int, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    """The raw stored image."""
    return FileResponse(
        image_path(ctx, image_id), headers={"Cache-Control": "private, max-age=86400"}
    )


@router.get("/images/{image_id}/thumb")
def get_thumb(image_id: int, ctx: AppContext = Depends(get_ctx)) -> FileResponse:
    """A small JPEG thumbnail of a stored image."""
    source = image_path(ctx, image_id)
    target = thumbnail(source, ctx.config.paths.thumbs / f"{image_id}.jpg")
    return FileResponse(
        target, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
    )


# ----------------------------------------------------------------- dataset


@router.get("/dataset/samples")
def dataset_samples(
    split: str = "val",
    n: int = 12,
    cls: str | None = Query(default=None, alias="class"),
    random: bool = True,
    offset: int = 0,
    seed: int | None = None,
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Dataset images with ground truth (default: random validation images)."""
    root = dataset_root_or_404(ctx)
    return list_samples(root, split, min(max(n, 1), 300), cls, random, offset, seed)


@router.get("/dataset/images/{split}/{name}")
def dataset_image(
    split: str, name: str, thumb: bool = False, ctx: AppContext = Depends(get_ctx)
) -> FileResponse:
    """A dataset image (only files listed by the dataset are served)."""
    sample = find_sample(dataset_root_or_404(ctx), f"{split}/{name}")
    if thumb:
        target = thumbnail(sample.image_path, ctx.config.paths.thumbs / "dataset" / split / name)
        return FileResponse(
            target, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )
    return FileResponse(sample.image_path, headers={"Cache-Control": "max-age=86400"})


# ----------------------------------------------------------------- SSE


@router.get("/events", response_class=EventSourceResponse)
async def events(request: Request, ctx: AppContext = Depends(get_ctx)) -> Any:
    """Server-sent events (analysis.completed, inspection.completed, models.changed, ...)."""
    yield ServerSentEvent(comment="connected", retry=3000)
    queue = ctx.events.open()
    try:
        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                if await request.is_disconnected():
                    break
                yield ServerSentEvent(comment="keep-alive")
                continue
            yield ServerSentEvent(data=data, event=event)
    finally:
        ctx.events.close(queue)
