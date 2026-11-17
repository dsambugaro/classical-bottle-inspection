"""Debug mode: one image kept in memory and traced step by step, then run by any profile.

A debug session holds the decoded image. Its preprocessing can be traced with the parameters
and background of any profile, and every profile can be run on it as often as wanted without
sending the image again. Nothing is stored: no inspection, analysis or image row is created,
so debugging never shows up in the reports. Selecting another image starts a new session.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..context import AppContext, get_ctx
from ..datasets import find_sample
from ..db.models import Image, Profile
from ..engine.analysis import ProfileRuntime
from ..engine.trace import preprocessing_trace, profile_trace
from ..errors import ApiError
from ..params_catalog import validate_params
from .common import dataset_root_or_404, decode_or_415, read_upload
from .profiles import get_profile, model_row_for

router = APIRouter(prefix="/api/debug", tags=["debug"])

MAX_SESSIONS: int = 4
"""Debug sessions kept in memory (one per open tab is enough); the oldest is dropped."""

MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


@dataclass
class DebugSession:
    """The image being debugged."""

    id: str
    image: np.ndarray
    data: bytes
    media_type: str
    name: str
    ground_truth: str | None

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "id": self.id,
            "name": self.name,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "ground_truth": self.ground_truth,
            "image_url": f"/api/debug/sessions/{self.id}/image",
        }


class DebugStore:
    """In-memory debug sessions, newest last."""

    def __init__(self) -> None:
        """Starts empty."""
        self._sessions: OrderedDict[str, DebugSession] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, session: DebugSession) -> None:
        """Keeps a session, dropping the oldest ones beyond MAX_SESSIONS."""
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def get(self, session_id: str) -> DebugSession:
        """A session by id (404 once it was replaced or dropped)."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ApiError(404, "debug.session_not_found", "debug session not found")
        return session

    def remove(self, session_id: str) -> bool:
        """Forgets a session; True when it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def store_of(ctx: AppContext) -> DebugStore:
    """The debug store of the app (created on first use)."""
    return ctx.extras.setdefault("debug", DebugStore())


class TraceIn(BaseModel):
    """Temporary parameter changes applied on top of the profile (never saved)."""

    overrides: dict[str, Any] = Field(default_factory=dict)


def runtime_with(ctx: AppContext, profile_id: int, overrides: dict[str, Any]) -> ProfileRuntime:
    """The runtime of a profile, with validated temporary overrides when given."""
    with ctx.db.session() as session:
        profile: Profile = get_profile(session, profile_id)
        model = model_row_for(session, profile)
        flat = None
        if overrides:
            flat = {**profile.params_json, **overrides}
            training = model.training_params_json if model is not None else None
            errors = validate_params(flat, training)
            if errors:
                raise ApiError(
                    422,
                    "profile.invalid_params",
                    "invalid parameters",
                    {"errors": [e.to_json() for e in errors]},
                )
        session.expunge(profile)
    return ctx.inspections.runtime_for(profile, params_override=flat)


@router.post("/sessions", status_code=201)
async def create_session(
    file: UploadFile | None = File(default=None),
    image_id: int | None = Form(default=None),
    dataset_ref: str | None = Form(default=None),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Starts a debug session from an upload, a stored image or a dataset sample."""
    truth: str | None = None
    if file is not None:
        data, extension = await read_upload(file, float(ctx.settings.get("max_upload_mb")))
        name = file.filename or "upload"
    elif image_id is not None:
        with ctx.db.session() as db:
            row = db.get(Image, image_id)
            if row is None or not Path(row.path).is_file():
                raise ApiError(404, "image.not_found", "image not found")
            path, truth, name = Path(row.path), row.ground_truth_class, row.original_name
        data, extension = path.read_bytes(), path.suffix.lower()
        name = name or path.name
    elif dataset_ref:
        sample = find_sample(dataset_root_or_404(ctx), dataset_ref)
        data, extension = sample.image_path.read_bytes(), ".jpg"
        truth, name = sample.label, sample.image_path.name
    else:
        raise ApiError(
            422, "image.source_required", "an image, image_id or dataset sample is required"
        )
    image, _decode_ms = decode_or_415(data, name)
    session = DebugSession(
        id=uuid.uuid4().hex,
        image=image,
        data=data,
        media_type=MEDIA_TYPES.get(extension, "application/octet-stream"),
        name=name,
        ground_truth=truth,
    )
    store_of(ctx).add(session)
    return session.to_json()


@router.get("/sessions/{session_id}/image")
def session_image(session_id: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    """The image of a debug session, exactly as it was received."""
    session = store_of(ctx).get(session_id)
    return Response(
        session.data,
        media_type=session.media_type,
        headers={"Cache-Control": "private, max-age=600"},
    )


@router.post("/sessions/{session_id}/preprocessing/{profile_id}")
async def trace_preprocessing(
    session_id: str, profile_id: int, body: TraceIn, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Every preprocessing step of the image, with the parameters and background of a profile."""
    session = store_of(ctx).get(session_id)
    runtime = runtime_with(ctx, profile_id, body.overrides)
    trace = await asyncio.get_running_loop().run_in_executor(
        ctx.executor, preprocessing_trace, session.image, runtime.background, runtime.params
    )
    return {"profile_id": profile_id, **trace}


@router.post("/sessions/{session_id}/profiles/{profile_id}")
async def trace_profile(
    session_id: str, profile_id: int, body: TraceIn, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Feature groups, feature vector and classifier decision of one profile on the image."""
    session = store_of(ctx).get(session_id)
    runtime = runtime_with(ctx, profile_id, body.overrides)
    return await asyncio.get_running_loop().run_in_executor(
        ctx.executor,
        profile_trace,
        session.image,
        runtime,
        session.ground_truth,
        f"/api/debug/sessions/{session.id}/image",
    )


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, bool]:
    """Forgets a debug session and its image."""
    return {"deleted": store_of(ctx).remove(session_id)}
