"""Cameras and camera sessions (REQ-I3..I6): CRUD, test snapshot, MJPEG preview, acquisition
modes (continuous / single / motion) for server cameras, and the browser-camera WebSocket."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from bottle_inspection.preprocessing import foreground_mask

from ..context import AppContext, get_ctx
from ..db.models import Camera, Session, utcnow
from ..engine.analysis import decode_image, foreground_fraction
from ..errors import ApiError
from ..notifications import notify
from ..profiles import iso
from .capture import CameraConfig, CameraError, CaptureFactory, CaptureWorker, open_opencv

log = logging.getLogger(__name__)

Mode = Literal["continuous", "single", "motion"]
MJPEG_FPS: float = 5.0
MJPEG_WIDTH: int = 640
STORE_JPEG_QUALITY: int = 90
MOTION_REARM_RATIO: float = 0.5
"""Motion re-arms when the foreground fraction falls below threshold × this ratio."""


def encode_jpeg(frame: np.ndarray, quality: int = STORE_JPEG_QUALITY) -> bytes:
    """JPEG bytes of a BGR frame."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cannot encode frame")
    return buffer.tobytes()


class MotionTrigger:
    """REQ-I5 `motion`: fires when the foreground fraction rises above the threshold after
    having been low (a bottle arrived); re-arms when it falls back below threshold × 0.5."""

    def __init__(self, threshold: float) -> None:
        """Armed until the first bottle."""
        self.threshold = threshold
        self.armed = True

    def update(self, fraction: float) -> bool:
        """True when this frame should be analysed."""
        if self.armed and fraction > self.threshold:
            self.armed = False
            return True
        if not self.armed and fraction < self.threshold * MOTION_REARM_RATIO:
            self.armed = True
        return False


def motion_fraction(frame: np.ndarray, ctx: AppContext, profile_id: int) -> float:
    """Foreground fraction of the whole mask of a frame (Otsu threshold, closing and opening),
    with the first profile's background and parameters (same functions as the analysis)."""
    profile = ctx.inspections.profiles_for([profile_id])[0]
    runtime = ctx.inspections.runtime_for(profile)
    params = runtime.params
    mask = foreground_mask(
        cv2.resize(frame, params.localization.size), runtime.background, params=params
    )
    return foreground_fraction(mask)


@dataclass
class CameraSessionState:
    """A running camera session."""

    id: int
    source: str
    camera_id: int | None
    profile_ids: list[int]
    mode: Mode
    fps: float
    trigger: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    motion: MotionTrigger | None = None
    analysed: int = 0
    error: str | None = None


class CameraService:
    """Owns capture workers and camera sessions (registered as ctx.extras["cameras"])."""

    def __init__(self, ctx: AppContext, factory: CaptureFactory = open_opencv) -> None:
        """No camera is opened until needed."""
        self.ctx = ctx
        self.factory = factory
        self.workers: dict[int, CaptureWorker] = {}
        self.sessions: dict[int, CameraSessionState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Closes sessions left open by a previous run."""
        with self.ctx.db.session() as session:
            for row in session.scalars(
                select(Session).where(
                    Session.ended_at.is_(None),
                    Session.source.in_(("server_camera", "browser_camera")),
                )
            ):
                row.ended_at = utcnow()

    async def stop(self) -> None:
        """Stops every session and capture thread."""
        for state in list(self.sessions.values()):
            await self.stop_session(state.id)
        for worker in list(self.workers.values()):
            worker.stop()

    def active_count(self) -> int:
        """Cameras with a running capture thread."""
        return sum(1 for w in self.workers.values() if w.alive)

    def session_count(self) -> int:
        """Running camera sessions."""
        return len(self.sessions)

    # ------------------------------------------------------------- cameras

    def config(self, camera_id: int) -> CameraConfig:
        """Configuration of a camera (404 when unknown)."""
        with self.ctx.db.session() as session:
            row = session.get(Camera, camera_id)
            if row is None:
                raise ApiError(404, "camera.not_found", "camera not found")
            return CameraConfig(
                row.id,
                row.name,
                row.kind,
                row.device_index,
                row.url,
                row.width,
                row.height,
                row.fps,
            )

    def acquire(self, camera_id: int) -> CaptureWorker:
        """Starts (or reuses) the capture thread of a camera (blocking)."""
        config = self.config(camera_id)
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None or worker.config != config:
                if worker is not None:
                    worker.stop()
                worker = CaptureWorker(config, self.factory)
                self.workers[camera_id] = worker
            worker.users += 1
        try:
            worker.start()
        except CameraError:
            self.release(camera_id)
            raise
        return worker

    def release(self, camera_id: int) -> None:
        """Stops the capture thread when nobody uses it any more."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None:
                return
            worker.users = max(0, worker.users - 1)
            if worker.users == 0:
                worker.stop()
                self.workers.pop(camera_id, None)

    def snapshot(self, camera_id: int) -> dict[str, Any]:
        """Test button: grabs one frame (blocking)."""
        worker = self.acquire(camera_id)
        try:
            frame, _seq = worker.wait_new(0, timeout=5)
            if frame is None:
                raise CameraError("camera.no_frames", "no frame received")
            height, width = frame.shape[:2]
            small = cv2.resize(frame, (MJPEG_WIDTH, int(height * MJPEG_WIDTH / width)))
            return {
                "ok": True,
                "width": width,
                "height": height,
                "snapshot": base64.b64encode(encode_jpeg(small, 80)).decode(),
            }
        finally:
            self.release(camera_id)

    def grab_frames(self, camera_id: int, count: int) -> list[np.ndarray]:
        """`count` distinct frames, for the background calibration (REQ-M5)."""
        try:
            worker = self.acquire(camera_id)
        except CameraError as error:
            raise ApiError(409, error.code, str(error)) from error
        frames, seq = [], 0
        try:
            while len(frames) < count:
                frame, seq = worker.wait_new(seq, timeout=5)
                if frame is None:
                    raise ApiError(409, "camera.no_frames", "the camera stopped sending frames")
                frames.append(frame)
            return frames
        finally:
            self.release(camera_id)

    async def mjpeg(self, camera_id: int) -> AsyncIterator[bytes]:
        """Low-rate MJPEG preview (multipart/x-mixed-replace)."""
        loop = asyncio.get_running_loop()
        try:
            worker = await loop.run_in_executor(self.ctx.executor, self.acquire, camera_id)
        except CameraError as error:
            raise ApiError(409, error.code, str(error)) from error
        seq = 0
        try:
            while worker.alive:
                frame, new_seq = worker.latest()
                if frame is not None and new_seq != seq:
                    seq = new_seq
                    height, width = frame.shape[:2]
                    small = cv2.resize(frame, (MJPEG_WIDTH, int(height * MJPEG_WIDTH / width)))
                    data = encode_jpeg(small, 70)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                await asyncio.sleep(1.0 / MJPEG_FPS)
        finally:
            self.release(camera_id)

    # ------------------------------------------------------------- sessions

    def clamp_fps(self, fps: float | None) -> float:
        """fps within the configured limits (default from the settings)."""
        settings = self.ctx.settings.all()
        value = float(fps if fps is not None else settings["default_fps"])
        return min(float(settings["max_fps"]), max(float(settings["min_fps"]), value))

    def start_session(
        self,
        source: str,
        camera_id: int | None,
        profile_ids: list[int],
        mode: Mode,
        fps: float | None,
    ) -> CameraSessionState:
        """Creates a camera session; server-camera sessions start their capture loop."""
        profiles = self.ctx.inspections.profiles_for(profile_ids)
        for profile in profiles:
            self.ctx.inspections.runtime_for(profile)
        if source == "server_camera":
            if camera_id is None:
                raise ApiError(422, "camera.required", "choose a camera")
            self.config(camera_id)
            if any(s.camera_id == camera_id for s in self.sessions.values()):
                raise ApiError(409, "camera.busy", "this camera already has a running session")
        rate = self.clamp_fps(fps)
        with self.ctx.db.session() as session:
            row = Session(
                source=source,
                camera_id=camera_id,
                profile_ids_json=[p.id for p in profiles],
                mode=mode,
                fps=rate,
                started_at=utcnow(),
            )
            session.add(row)
            session.flush()
            session_id = row.id
        state = CameraSessionState(
            session_id, source, camera_id, [p.id for p in profiles], mode, rate
        )
        if mode == "motion":
            state.motion = MotionTrigger(float(self.ctx.settings.get("motion_threshold")))
        self.sessions[session_id] = state
        if source == "server_camera":
            state.task = asyncio.create_task(self._run_server_session(state))
        self._status(state, "started")
        return state

    async def stop_session(self, session_id: int) -> None:
        """Stops a session and marks it ended."""
        state = self.sessions.pop(session_id, None)
        if state is not None:
            state.stop.set()
            state.trigger.set()
            if state.task is not None and state.task is not asyncio.current_task():
                try:
                    await asyncio.wait_for(state.task, timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    state.task.cancel()
            self._status(state, "stopped")
        with self.ctx.db.session() as session:
            row = session.get(Session, session_id)
            if row is not None and row.ended_at is None:
                row.ended_at = utcnow()

    def _status(self, state: CameraSessionState, status: str, code: str | None = None) -> None:
        """Publishes a `camera.status` event."""
        self.ctx.events.publish(
            "camera.status",
            {
                "session_id": state.id,
                "camera_id": state.camera_id,
                "source": state.source,
                "status": status,
                "code": code,
                "analysed": state.analysed,
            },
        )

    async def analyse_frame(
        self, state: CameraSessionState, frame: np.ndarray, jpeg: bytes | None, decode_ms: float
    ) -> dict[str, Any] | None:
        """Analyses one frame for a session (None when motion mode skips it)."""
        loop = asyncio.get_running_loop()
        if state.motion is not None:
            fraction = await loop.run_in_executor(
                self.ctx.executor, motion_fraction, frame, self.ctx, state.profile_ids[0]
            )
            if not state.motion.update(fraction):
                return None
        profiles = self.ctx.inspections.profiles_for(state.profile_ids)
        camera_name = None
        if state.camera_id is not None:
            camera_name = self.config(state.camera_id).name
        payload = await self.ctx.inspections.inspect(
            frame,
            decode_ms,
            profiles,
            source=state.source,
            session_id=state.id,
            image_bytes=jpeg if jpeg is not None else encode_jpeg(frame),
            extension=".jpg",
            camera_name=camera_name,
            store="policy",
        )
        state.analysed += 1
        return payload

    async def _run_server_session(self, state: CameraSessionState) -> None:
        """Capture loop of a server-camera session (one analysis in flight)."""
        loop = asyncio.get_running_loop()
        assert state.camera_id is not None
        try:
            worker = await loop.run_in_executor(self.ctx.executor, self.acquire, state.camera_id)
        except CameraError as error:
            await self._fail(state, error.code, str(error))
            return
        seq = 0
        interval = 1.0 / state.fps
        try:
            while not state.stop.is_set():
                started = loop.time()
                if state.mode == "single":
                    await state.trigger.wait()
                    state.trigger.clear()
                    if state.stop.is_set():
                        break
                frame, seq = await loop.run_in_executor(
                    self.ctx.executor, worker.wait_new, seq, 2.0
                )
                if frame is not None:
                    await self.analyse_frame(state, frame, None, 0.0)
                if state.mode != "single":
                    delay = interval - (loop.time() - started)
                    if delay > 0:
                        try:
                            await asyncio.wait_for(state.stop.wait(), timeout=delay)
                        except asyncio.TimeoutError:
                            pass
        except CameraError as error:
            await self._fail(state, error.code, str(error))
        except ApiError as error:
            await self._fail(state, error.code, error.message)
        except Exception as error:  # noqa: BLE001 - a camera loop never crashes the server
            log.exception("camera session failed")
            await self._fail(state, "camera.session_failed", str(error))
        finally:
            self.release(state.camera_id)

    async def _fail(self, state: CameraSessionState, code: str, message: str) -> None:
        """Stops a session after an error and tells the user (NF4)."""
        state.error = code
        self._status(state, "error", code)
        notify(
            self.ctx.db,
            self.ctx.events,
            "error",
            f"Camera session {state.id}: {message}",
            message,
            code="camera.error",
            params={"session_id": state.id, "error": code},
        )
        self.sessions.pop(state.id, None)
        with self.ctx.db.session() as session:
            row = session.get(Session, state.id)
            if row is not None and row.ended_at is None:
                row.ended_at = utcnow()


# ----------------------------------------------------------------- API

router = APIRouter(tags=["cameras"])


def cameras(ctx: AppContext) -> CameraService:
    """The camera service of the app."""
    return ctx.extras["cameras"]


class CameraIn(BaseModel):
    """Body of POST /api/cameras."""

    name: str = Field(min_length=1, max_length=60)
    kind: Literal["usb", "rtsp", "http"]
    device_index: int | None = Field(default=None, ge=0, le=63)
    url: str | None = Field(default=None, max_length=1024)
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    fps: float = Field(default=3.0, ge=0.2, le=60)
    enabled: bool = True


class CameraPatch(BaseModel):
    """Body of PATCH /api/cameras/{id}."""

    name: str | None = Field(default=None, min_length=1, max_length=60)
    kind: Literal["usb", "rtsp", "http"] | None = None
    device_index: int | None = Field(default=None, ge=0, le=63)
    url: str | None = Field(default=None, max_length=1024)
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    fps: float | None = Field(default=None, ge=0.2, le=60)
    enabled: bool | None = None


def camera_json(row: Camera) -> dict[str, Any]:
    """JSON form of a camera."""
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "device_index": row.device_index,
        "url": row.url,
        "width": row.width,
        "height": row.height,
        "fps": row.fps,
        "enabled": row.enabled,
    }


def check_camera(kind: str, device_index: int | None, url: str | None) -> None:
    """USB cameras need an index; RTSP/HTTP cameras need a matching URL."""
    if kind == "usb" and device_index is None:
        raise ApiError(422, "camera.index_required", "a USB camera needs a device index")
    if kind in ("rtsp", "http") and not (url and url.lower().startswith(f"{kind}")):
        raise ApiError(422, "camera.url_required", f"a {kind} camera needs a {kind}:// URL")


@router.get("/api/cameras")
def list_cameras(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Every configured server camera, with its running state."""
    service = cameras(ctx)
    with ctx.db.session() as session:
        rows = session.scalars(select(Camera).order_by(Camera.id)).all()
        items = [camera_json(r) for r in rows]
    for item in items:
        worker = service.workers.get(item["id"])
        item["active"] = bool(worker and worker.alive)
        item["session_id"] = next(
            (s.id for s in service.sessions.values() if s.camera_id == item["id"]), None
        )
    return items


@router.post("/api/cameras", status_code=201)
def create_camera(body: CameraIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Adds a server camera."""
    check_camera(body.kind, body.device_index, body.url)
    with ctx.db.session() as session:
        if session.scalars(select(Camera).where(Camera.name == body.name)).first():
            raise ApiError(409, "camera.name_taken", "a camera with this name exists")
        row = Camera(**body.model_dump())
        session.add(row)
        session.flush()
        return camera_json(row)


@router.patch("/api/cameras/{camera_id}")
def patch_camera(
    camera_id: int, body: CameraPatch, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Edits a camera."""
    with ctx.db.session() as session:
        row = session.get(Camera, camera_id)
        if row is None:
            raise ApiError(404, "camera.not_found", "camera not found")
        for key, value in body.model_dump(exclude_unset=True).items():
            setattr(row, key, value)
        check_camera(row.kind, row.device_index, row.url)
        return camera_json(row)


@router.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Removes a camera (its running session is stopped)."""
    service = cameras(ctx)
    for state in [s for s in service.sessions.values() if s.camera_id == camera_id]:
        await service.stop_session(state.id)
    with ctx.db.session() as session:
        row = session.get(Camera, camera_id)
        if row is None:
            raise ApiError(404, "camera.not_found", "camera not found")
        session.delete(row)
    return {"id": camera_id, "deleted": True}


@router.post("/api/cameras/{camera_id}/test")
async def test_camera(camera_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Grabs a snapshot; returns {ok: false, code, message} when the device can't be opened."""
    service = cameras(ctx)
    service.config(camera_id)
    try:
        return await asyncio.get_running_loop().run_in_executor(
            ctx.executor, service.snapshot, camera_id
        )
    except CameraError as error:
        return {"ok": False, "code": error.code, "message": str(error)}


@router.get("/api/cameras/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: int, ctx: AppContext = Depends(get_ctx)) -> StreamingResponse:
    """Live preview as MJPEG at a low rate."""
    service = cameras(ctx)
    service.config(camera_id)
    return StreamingResponse(
        service.mjpeg(camera_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


class SessionIn(BaseModel):
    """Body of POST /api/sessions."""

    source: Literal["server_camera", "browser_camera"]
    camera_id: int | None = None
    profile_ids: list[int]
    mode: Mode = "continuous"
    fps: float | None = None


def state_json(state: CameraSessionState) -> dict[str, Any]:
    """JSON form of a running session."""
    return {
        "id": state.id,
        "source": state.source,
        "camera_id": state.camera_id,
        "profile_ids": state.profile_ids,
        "mode": state.mode,
        "fps": state.fps,
        "analysed": state.analysed,
        "running": True,
    }


@router.post("/api/sessions", status_code=201)
async def start_session(body: SessionIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Starts a camera session (server camera or browser camera)."""
    state = cameras(ctx).start_session(
        body.source, body.camera_id, body.profile_ids, body.mode, body.fps
    )
    return state_json(state)


@router.get("/api/sessions")
def list_sessions(limit: int = 50, ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Latest sessions (for the reports filter)."""
    with ctx.db.session() as session:
        rows = session.scalars(select(Session).order_by(Session.id.desc()).limit(min(limit, 500)))
        return [
            {
                "id": r.id,
                "source": r.source,
                "camera_id": r.camera_id,
                "mode": r.mode,
                "started_at": iso(r.started_at),
                "ended_at": iso(r.ended_at),
                "profile_ids": r.profile_ids_json,
            }
            for r in rows
        ]


@router.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Stops a camera session."""
    await cameras(ctx).stop_session(session_id)
    return {"id": session_id, "running": False}


@router.post("/api/sessions/{session_id}/trigger")
def trigger_session(session_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Single mode: captures and analyses one frame now."""
    state = cameras(ctx).sessions.get(session_id)
    if state is None:
        raise ApiError(404, "session.not_running", "session is not running")
    state.trigger.set()
    return state_json(state)


@router.websocket("/ws/camera/browser")
async def browser_camera(websocket: WebSocket, session_id: int) -> None:
    """Browser camera (REQ-I3): binary JPEG frames in, one inspection JSON out per frame."""
    ctx: AppContext = websocket.app.state.ctx
    service = cameras(ctx)
    state = service.sessions.get(session_id)
    await websocket.accept()
    if state is None or state.source != "browser_camera":
        await websocket.send_json({"type": "error", "code": "session.not_running"})
        await websocket.close(code=4404)
        return
    try:
        while not state.stop.is_set():
            data = await websocket.receive_bytes()
            try:
                frame, decode_ms = decode_image(data)
            except ValueError:
                await websocket.send_json({"type": "error", "code": "upload.not_an_image"})
                continue
            try:
                payload = await service.analyse_frame(state, frame, data, decode_ms)
            except ApiError as error:
                await websocket.send_json({"type": "error", "code": error.code})
                continue
            if payload is None:
                await websocket.send_json({"type": "skipped", "reason": "motion"})
            else:
                await websocket.send_json({"type": "inspection", "inspection": payload})
    except WebSocketDisconnect:
        pass
    finally:
        await service.stop_session(session_id)


def register(app: FastAPI, ctx: AppContext) -> None:
    """Adds the camera service and its routes to the app."""
    ctx.extras["cameras"] = CameraService(ctx)
    app.include_router(router)
