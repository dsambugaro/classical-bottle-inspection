"""Inspection service: runs the selected profiles on one image and persists the results.

REQ-I9: the image is decoded once; each profile runs in the thread pool; each result is
published as soon as it is ready. REQ-I10: every analysis is persisted, handed to the
analysis hooks (rules) and published as an SSE `analysis.completed` event.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import select

from .classes import CLASS_NAMES
from .db.models import Analysis, Image, Inspection, Profile, utcnow
from .db.session import Database
from .engine.analysis import AnalysisOutput, ProfileRuntime, analyse
from .errors import ApiError
from .events import EventBroker
from .models.registry import ModelRegistry
from .params_catalog import to_params
from .profiles import derive_status, iso
from .settings import AppPaths
from .settings_store import SettingsStore

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS: dict[str, str] = {
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".png": ".png",
    ".bmp": ".bmp",
    ".webp": ".webp",
}
"""Accepted upload extensions (the type is still validated by decoding)."""

FRAME_CACHE_SIZE: int = 60
"""Unstored camera frames kept in memory so the live viewer can show them."""

KAGGLE_IMAGE_SIZE: tuple[int, int] = (1280, 720)
"""Resolution of the dataset images the model backgrounds come from."""


@dataclass
class AnalysisContext:
    """What the analysis hooks (rules) receive after an analysis is persisted."""

    analysis_id: int
    inspection_id: int
    profile_id: int
    result: dict[str, Any]


AnalysisHook = Callable[[AnalysisContext], None]


def image_url(image_id: int | None) -> str | None:
    """URL of a stored image."""
    return f"/api/images/{image_id}" if image_id is not None else None


def analysis_json(
    analysis: Analysis,
    profile: Profile,
    image: Image | None,
    inspection: Inspection | None = None,
) -> dict[str, Any]:
    """The AnalysisResult contract of §10."""
    width = image.width if image else (inspection.width if inspection else None)
    height = image.height if image else (inspection.height if inspection else None)
    return {
        "analysis_id": analysis.id,
        "inspection_id": analysis.inspection_id,
        "profile": {
            "id": profile.id,
            "name": profile.name,
            "version": analysis.profile_version,
            "algorithm": profile.algorithm,
            "model_ref": profile.model_ref,
            "status": "modified" if analysis.modified_params else "ready",
            "color": profile.color,
        },
        "image": (
            {
                "id": image.id,
                "width": image.width,
                "height": image.height,
                "url": image_url(image.id),
                "thumb_url": f"/api/images/{image.id}/thumb",
            }
            if image
            else {"id": None, "width": width, "height": height, "url": None, "thumb_url": None}
        ),
        "predicted_class": analysis.predicted_class,
        "verdict": analysis.verdict,
        "confidence": analysis.confidence,
        "scores": {name: analysis.scores_json.get(name) for name in CLASS_NAMES},
        "score_kind": analysis.score_kind,
        "details": analysis.details_json,
        "geometry": analysis.geometry_json,
        "timings_ms": analysis.timings_json,
        "ground_truth": analysis.ground_truth_class,
        "warnings": list(analysis.warnings_json),
        "source": analysis.source,
        "session_id": analysis.session_id,
        "created_at": iso(analysis.created_at),
    }


def _json_number(value: float) -> float | None:
    """JSON-safe float (infinite distances become null)."""
    return float(value) if np.isfinite(value) else None


class InspectionService:
    """Runs profiles on images and stores inspections, images and analyses."""

    def __init__(
        self,
        db: Database,
        registry: ModelRegistry,
        paths: AppPaths,
        settings: SettingsStore,
        events: EventBroker,
        executor: Executor,
    ) -> None:
        """Wires the service; hooks are added by the rules engine."""
        self.db = db
        self.registry = registry
        self.paths = paths
        self.settings = settings
        self.events = events
        self.executor = executor
        self.hooks: list[AnalysisHook] = []
        self._runtimes: dict[tuple[Any, ...], ProfileRuntime] = {}
        self.frames: OrderedDict[int, bytes] = OrderedDict()
        """Recent frames that were not stored (camera policy), by inspection id."""
        self._frames_lock = threading.Lock()
        self._runtime_lock = threading.Lock()

    # ------------------------------------------------------------------ profiles

    def runtime_for(
        self, profile: Profile, params_override: dict[str, Any] | None = None
    ) -> ProfileRuntime:
        """ProfileRuntime of a profile (cached by profile version and model file mtime)."""
        model_row = self.registry.row(profile.model_ref)
        status = derive_status(profile, model_row)
        if not status.can_analyse or model_row is None:
            raise ApiError(
                409,
                f"profile.{status.status}",
                f"profile '{profile.name}' cannot analyse ({status.status})",
                {"profile_id": profile.id, "status": status.status},
            )
        params_json = params_override if params_override is not None else profile.params_json
        key = (
            profile.id,
            profile.version,
            model_row.file_mtime,
            profile.background_path,
            params_override is not None and repr(sorted(params_override.items())),
        )
        with self._runtime_lock:
            cached = self._runtimes.get(key)
        if cached is not None:
            return cached
        loaded = self.registry.load(profile.model_ref)
        background = loaded.background
        reference_size: tuple[int, int] | None = KAGGLE_IMAGE_SIZE
        if profile.background_source != "model" and profile.background_path:
            background, reference_size = self.load_background(Path(profile.background_path))
        params = to_params(params_json)
        if params_override is not None:
            from .params_catalog import EDITABLE_KEYS, diff_keys

            modified = bool(
                diff_keys(params_override, model_row.training_params_json, EDITABLE_KEYS)
            )
            modified = modified or profile.background_source != "model"
        else:
            modified = status.status == "modified"
        runtime = ProfileRuntime(
            profile_id=profile.id,
            name=profile.name,
            version=profile.version,
            algorithm=profile.algorithm,
            model_ref=profile.model_ref,
            status="modified" if modified else "ready",
            color=profile.color,
            params=params,
            background=background,
            model=loaded,
            verdict=dict(profile.verdict_json),
            modified=modified,
            reference_size=reference_size,
        )
        if params_override is None:
            with self._runtime_lock:
                self._runtimes = {k: v for k, v in self._runtimes.items() if k[0] != profile.id}
                self._runtimes[key] = runtime
        return runtime

    @staticmethod
    def load_background(path: Path) -> tuple[np.ndarray, tuple[int, int] | None]:
        """A captured/uploaded background (stored at the localization size) and its source size."""
        image = cv2.imread(str(path))
        if image is None:
            raise ApiError(409, "profile.background_missing", f"background file missing: {path}")
        size_file = path.with_suffix(".size")
        reference = None
        if size_file.exists():
            w, h = (int(v) for v in size_file.read_text().split("x"))
            reference = (w, h)
        return image, reference

    def profiles_for(self, profile_ids: list[int]) -> list[Profile]:
        """Active profiles in the requested order (404 for unknown or deleted ones)."""
        if not profile_ids:
            raise ApiError(422, "inspection.no_profiles", "select at least one profile")
        limit = int(self.settings.get("max_profiles"))
        if len(profile_ids) > limit:
            raise ApiError(422, "inspection.too_many_profiles", "too many profiles", {"max": limit})
        with self.db.session() as session:
            rows = {
                p.id: p
                for p in session.scalars(
                    select(Profile).where(Profile.id.in_(profile_ids), Profile.deleted_at.is_(None))
                )
            }
        missing = [pid for pid in profile_ids if pid not in rows]
        if missing:
            raise ApiError(404, "profile.not_found", "profile not found", {"ids": missing})
        return [rows[pid] for pid in dict.fromkeys(profile_ids)]

    # ------------------------------------------------------------------ images

    def store_image(
        self,
        data: bytes,
        extension: str,
        source: str,
        width: int,
        height: int,
        original_name: str = "",
        ground_truth: str | None = None,
    ) -> int:
        """Copies the raw image to webdata/images/yyyy/mm/dd/<uuid>.<ext>; returns its id."""
        now = datetime.now()
        folder = self.paths.images / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid.uuid4().hex}{extension}"
        path.write_bytes(data)
        with self.db.session() as session:
            row = Image(
                path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                width=width,
                height=height,
                source=source,
                original_name=original_name[:255],
                ground_truth_class=ground_truth,
                created_at=utcnow(),
            )
            session.add(row)
            session.flush()
            return row.id

    # ------------------------------------------------------------------ analysis

    def _persist(
        self,
        output: AnalysisOutput,
        runtime: ProfileRuntime,
        inspection: Inspection,
    ) -> int:
        """Stores one analysis row; returns its id."""
        scores = {name: _json_number(v) for name, v in output.score.scores.items()}
        model_row = self.registry.row(runtime.model_ref)
        with self.db.session() as session:
            row = Analysis(
                inspection_id=inspection.id,
                profile_id=runtime.profile_id,
                profile_version=runtime.version,
                trained_model_id=model_row.id if model_row else None,
                model_mtime=model_row.file_mtime if model_row else None,
                modified_params=runtime.modified,
                predicted_class=output.predicted_class,
                verdict=output.verdict,
                confidence=output.confidence,
                scores_json=scores,
                score_kind=output.score.score_kind,
                details_json=output.score.details,
                geometry_json=output.geometry,
                timings_json=output.timings_ms,
                total_ms=output.total_ms,
                warnings_json=output.warnings,
                source=inspection.source,
                session_id=inspection.session_id,
                ground_truth_class=inspection.ground_truth_class,
                created_at=utcnow(),
            )
            session.add(row)
            session.flush()
            return row.id

    def _run_one(
        self, image: np.ndarray, decode_ms: float, runtime: ProfileRuntime, inspection: Inspection
    ) -> tuple[int, ProfileRuntime]:
        """Analyses and persists one profile (runs in the thread pool)."""
        output = analyse(image, runtime, decode_ms)
        analysis_id = self._persist(output, runtime, inspection)
        return analysis_id, runtime

    def result(self, analysis_id: int) -> dict[str, Any]:
        """AnalysisResult JSON of a stored analysis."""
        with self.db.session() as session:
            analysis = session.get(Analysis, analysis_id)
            if analysis is None:
                raise ApiError(404, "analysis.not_found", "analysis not found")
            profile = session.get(Profile, analysis.profile_id)
            inspection = session.get(Inspection, analysis.inspection_id)
            image = (
                session.get(Image, inspection.image_id)
                if inspection and inspection.image_id
                else None
            )
            assert profile is not None
            return analysis_json(analysis, profile, image, inspection)

    def _run_hooks(self, context: AnalysisContext) -> None:
        """Calls every analysis hook; a failing hook never breaks the inspection (NF4)."""
        for hook in self.hooks:
            try:
                hook(context)
            except Exception:  # noqa: BLE001
                log.exception("analysis hook failed")

    async def inspect(
        self,
        image: np.ndarray,
        decode_ms: float,
        profiles: list[Profile],
        *,
        source: str,
        session_id: int | None = None,
        image_bytes: bytes | None = None,
        extension: str = ".jpg",
        original_name: str = "",
        ground_truth: str | None = None,
        camera_name: str | None = None,
        store: str = "always",
        on_result: Callable[[dict[str, Any]], None] | None = None,
        existing_image_id: int | None = None,
    ) -> dict[str, Any]:
        """Runs every profile on `image`; returns the inspection with its analyses.

        `store`: "always" (uploads and dataset samples), "policy" (camera frames, the
        storage policy of the settings decides after the verdicts are known) or "never".
        """
        loop = asyncio.get_running_loop()
        runtimes = [self.runtime_for(profile) for profile in profiles]
        height, width = image.shape[:2]
        image_id = existing_image_id
        if image_id is None and store == "always" and image_bytes is not None:
            image_id = await loop.run_in_executor(
                self.executor,
                lambda: self.store_image(
                    image_bytes, extension, source, width, height, original_name, ground_truth
                ),
            )
        with self.db.session() as session:
            inspection = Inspection(
                session_id=session_id,
                image_id=image_id,
                source=source,
                camera_name=camera_name,
                original_name=original_name[:255],
                ground_truth_class=ground_truth,
                width=width,
                height=height,
                created_at=utcnow(),
            )
            session.add(inspection)
            session.flush()
            session.expunge(inspection)

        futures = [
            loop.run_in_executor(self.executor, self._run_one, image, decode_ms, rt, inspection)
            for rt in runtimes
        ]
        results: dict[int, dict[str, Any]] = {}
        for future in asyncio.as_completed(futures):
            analysis_id, runtime = await future
            result = await loop.run_in_executor(self.executor, self.result, analysis_id)
            results[runtime.profile_id] = result
            if on_result is not None:
                on_result(result)
            self.events.publish("analysis.completed", result)
            context = AnalysisContext(analysis_id, inspection.id, runtime.profile_id, result)
            await loop.run_in_executor(self.executor, self._run_hooks, context)

        ordered = [results[rt.profile_id] for rt in runtimes]
        if store == "policy" and image_bytes is not None:
            policy = self.settings.get("storage_policy")
            rejected = any(r["verdict"] != "pass" for r in ordered)
            if policy == "all" or (policy == "rejects_only" and rejected):
                image_id = await loop.run_in_executor(
                    self.executor,
                    lambda: self.store_image(
                        image_bytes, extension, source, width, height, original_name, ground_truth
                    ),
                )
                with self.db.session() as session:
                    row = session.get(Inspection, inspection.id)
                    if row is not None:
                        row.image_id = image_id
                ordered = [
                    await loop.run_in_executor(self.executor, self.result, r["analysis_id"])
                    for r in ordered
                ]
        if image_id is None and image_bytes is not None:
            self.remember_frame(inspection.id, image_bytes)
            for result in ordered:
                result["image"]["url"] = f"/api/inspections/{inspection.id}/frame"
        payload = {
            "inspection_id": inspection.id,
            "session_id": session_id,
            "source": source,
            "original_name": original_name,
            "ground_truth": ground_truth,
            "image": ordered[0]["image"] if ordered else None,
            "created_at": iso(inspection.created_at),
            "analyses": ordered,
        }
        self.events.publish(
            "inspection.completed",
            {
                "inspection_id": inspection.id,
                "session_id": session_id,
                "verdicts": {str(r["profile"]["id"]): r["verdict"] for r in ordered},
            },
        )
        return payload

    def remember_frame(self, inspection_id: int, data: bytes) -> None:
        """Keeps the frame of an unstored inspection in memory (last FRAME_CACHE_SIZE)."""
        with self._frames_lock:
            self.frames[inspection_id] = data
            while len(self.frames) > FRAME_CACHE_SIZE:
                self.frames.popitem(last=False)

    def frame(self, inspection_id: int) -> bytes | None:
        """A cached frame, or None once evicted."""
        with self._frames_lock:
            return self.frames.get(inspection_id)

    def inspection_json(self, inspection_id: int) -> dict[str, Any]:
        """A stored inspection with all its analyses."""
        with self.db.session() as session:
            inspection = session.get(Inspection, inspection_id)
            if inspection is None:
                raise ApiError(404, "inspection.not_found", "inspection not found")
            image = session.get(Image, inspection.image_id) if inspection.image_id else None
            rows = session.scalars(
                select(Analysis)
                .where(Analysis.inspection_id == inspection_id)
                .order_by(Analysis.id)
            ).all()
            analyses = []
            for row in rows:
                profile = session.get(Profile, row.profile_id)
                if profile is not None:
                    analyses.append(analysis_json(row, profile, image, inspection))
            return {
                "inspection_id": inspection.id,
                "session_id": inspection.session_id,
                "source": inspection.source,
                "original_name": inspection.original_name,
                "ground_truth": inspection.ground_truth_class,
                "image": analyses[0]["image"] if analyses else None,
                "created_at": iso(inspection.created_at),
                "analyses": analyses,
            }
