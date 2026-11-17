"""Helpers shared by routers: reading uploads and resolving image sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..context import AppContext
from ..datasets import find_sample
from ..db.models import Image
from ..engine.analysis import decode_image
from ..errors import ApiError
from ..inspections import IMAGE_EXTENSIONS


async def read_upload(file: UploadFile, max_mb: float) -> tuple[bytes, str]:
    """Reads an uploaded image with the size limit; returns (bytes, normalized extension)."""
    extension = IMAGE_EXTENSIONS.get(Path(file.filename or "").suffix.lower())
    if extension is None:
        raise ApiError(
            415,
            "upload.unsupported_type",
            "unsupported file type",
            {"name": file.filename, "accepted": sorted(set(IMAGE_EXTENSIONS.values()))},
        )
    limit = int(max_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise ApiError(
            413, "upload.too_large", "file too large", {"name": file.filename, "max_mb": max_mb}
        )
    return data, extension


def parse_ids(value: str | list[str] | None) -> list[int]:
    """Profile ids from "1,2,3", repeated form fields or a JSON list."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    ids: list[int] = []
    for item in items:
        text = item.strip()
        if text.startswith("["):
            ids.extend(int(v) for v in json.loads(text))
        else:
            ids.extend(int(v) for v in text.split(",") if v.strip())
    return ids


def parse_json_field(value: str | None, name: str) -> Any:
    """A JSON-encoded multipart field."""
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except ValueError as error:
        raise ApiError(
            422, "request.invalid_json", f"{name} is not valid JSON", {"field": name}
        ) from error


def dataset_root_or_404(ctx: AppContext) -> Path:
    """The dataset root, or a 404 error when it is not configured/found."""
    root = ctx.registry.dataset_root()
    if root is None:
        raise ApiError(
            404,
            "dataset.not_found",
            "dataset not found",
            {"path": ctx.settings.get("dataset_path")},
        )
    return root


async def source_image(
    ctx: AppContext,
    file: UploadFile | None = None,
    image_id: int | None = None,
    dataset_ref: str | None = None,
) -> tuple[Any, float, str | None]:
    """Decoded BGR image from an upload, a stored image or a dataset sample.

    Returns (image, decode ms, ground truth class or None).
    """
    if file is not None:
        data, _ext = await read_upload(file, float(ctx.settings.get("max_upload_mb")))
        image, ms = decode_or_415(data, file.filename or "")
        return image, ms, None
    if image_id is not None:
        with ctx.db.session() as session:
            row = session.get(Image, image_id)
            if row is None or not Path(row.path).is_file():
                raise ApiError(404, "image.not_found", "image not found")
            path, truth = Path(row.path), row.ground_truth_class
        image, ms = decode_or_415(path.read_bytes(), path.name)
        return image, ms, truth
    if dataset_ref:
        sample = find_sample(dataset_root_or_404(ctx), dataset_ref)
        image, ms = decode_or_415(sample.image_path.read_bytes(), sample.image_path.name)
        return image, ms, sample.label
    raise ApiError(422, "image.source_required", "an image, image_id or dataset sample is required")


def decode_or_415(data: bytes, name: str) -> tuple[Any, float]:
    """Decodes image bytes, raising 415 when they are not an image."""
    try:
        return decode_image(data)
    except ValueError as error:
        raise ApiError(
            415, "upload.not_an_image", "the file is not a valid image", {"name": name}
        ) from error
