"""Keeps the `trained_models` table in sync with the model folders (REQ-M1..M4)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from ..db.models import TrainedModel, utcnow
from ..db.session import Database
from ..settings import AppPaths
from ..settings_store import SettingsStore
from .discovery import (
    LoadedModel,
    ModelCache,
    ModelInfo,
    describe_model,
    scan_folders,
    split_model_ref,
)

log = logging.getLogger(__name__)


def row_to_info(row: TrainedModel) -> ModelInfo:
    """ModelInfo from a DB row."""
    feature_vector = (row.metrics_json or {}).get("feature_vector")
    folder, algorithm = split_model_ref(row.model_ref, feature_vector)
    return ModelInfo(
        model_ref=row.model_ref,
        folder=folder,
        algorithm=algorithm,
        feature_vector=feature_vector,
        path=Path(row.path),
        file_mtime=row.file_mtime,
        training_params=dict(row.training_params_json),
        best_params=dict(row.best_params_json or {}),
        feature_length=row.feature_length,
        metrics=dict(row.metrics_json or {}),
        library_versions=dict(row.library_versions_json or {}),
        version_warning=row.version_warning,
    )


class ModelRegistry:
    """Discovery + cache of the models trained by the CLI."""

    def __init__(self, db: Database, paths: AppPaths, settings: SettingsStore) -> None:
        """Nothing is scanned until `refresh` is called."""
        self.db = db
        self.paths = paths
        self.settings = settings
        self.cache = ModelCache()
        self._refresh_lock = threading.Lock()
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    def model_folders(self) -> list[tuple[str, Path]]:
        """(label, resolved path) of every configured model folder."""
        return [
            (label.rstrip("/"), self.paths.resolve_code_path(label))
            for label in self.settings.get("model_folders")
        ]

    def dataset_root(self) -> Path | None:
        """Resolved dataset path, when it exists."""
        root = self.paths.resolve_code_path(self.settings.get("dataset_path"))
        return root if root.is_dir() else None

    def refresh(self) -> dict[str, list[str]]:
        """Rescans the model folders; returns the refs added, updated and missing."""
        with self._refresh_lock:
            return self._refresh()

    def _refresh(self) -> dict[str, list[str]]:
        """Implementation of refresh (holds the refresh lock)."""
        found = scan_folders(self.model_folders())
        changes: dict[str, list[str]] = {"added": [], "updated": [], "missing": [], "errors": []}
        seen: set[str] = set()
        with self.db.session() as session:
            rows = {row.model_ref: row for row in session.scalars(select(TrainedModel))}
        for location in found:
            ref, algorithm, path = location.ref, location.algorithm, location.path
            seen.add(ref)
            mtime = path.stat().st_mtime
            row = rows.get(ref)
            if (
                row is not None
                and row.file_mtime == mtime
                and not row.missing
                and not row.load_error
            ):
                continue
            try:
                info, artifact = describe_model(location)
            except Exception as error:  # noqa: BLE001 - a broken file must not stop discovery
                log.exception("cannot load trained model %s", ref)
                self._store_error(ref, algorithm, path, mtime, str(error))
                changes["errors"].append(ref)
                continue
            self._store(info, previous=row)
            self.cache.prime(info, artifact, self.dataset_root())
            changes["added" if row is None else "updated"].append(ref)
        with self.db.session() as session:
            for row in session.scalars(select(TrainedModel)):
                if row.model_ref not in seen and not row.missing:
                    row.missing = True
                    changes["missing"].append(row.model_ref)
        if any(changes.values()):
            for listener in self.listeners:
                listener(changes)
        return changes

    def _store(self, info: ModelInfo, previous: TrainedModel | None) -> None:
        """Inserts or updates the DB row of a discovered model."""
        with self.db.session() as session:
            row = session.scalars(
                select(TrainedModel).where(TrainedModel.model_ref == info.model_ref)
            ).first()
            if row is None:
                row = TrainedModel(model_ref=info.model_ref, discovered_at=utcnow())
                session.add(row)
            elif previous is not None and previous.file_mtime != info.file_mtime:
                row.updated_on = utcnow()
            row.algorithm = info.algorithm
            row.path = str(info.path)
            row.file_mtime = info.file_mtime
            row.training_params_json = info.training_params
            row.best_params_json = info.best_params
            row.feature_length = info.feature_length
            row.metrics_json = info.metrics
            row.library_versions_json = info.library_versions
            row.version_warning = info.version_warning
            row.load_error = None
            row.missing = False

    def _store_error(self, ref: str, algorithm: str, path: Path, mtime: float, error: str) -> None:
        """Keeps a row for a model that failed to load (listed with the error)."""
        with self.db.session() as session:
            row = session.scalars(select(TrainedModel).where(TrainedModel.model_ref == ref)).first()
            if row is None:
                row = TrainedModel(
                    model_ref=ref,
                    algorithm=algorithm,
                    path=str(path),
                    file_mtime=mtime,
                    training_params_json={},
                    feature_length=0,
                )
                session.add(row)
            row.file_mtime = mtime
            row.load_error = error
            row.missing = False

    def rows(self) -> list[TrainedModel]:
        """Every known model row (including missing ones)."""
        with self.db.session() as session:
            return list(session.scalars(select(TrainedModel).order_by(TrainedModel.model_ref)))

    def row(self, ref: str) -> TrainedModel | None:
        """The row of one model reference."""
        with self.db.session() as session:
            return session.scalars(
                select(TrainedModel).where(TrainedModel.model_ref == ref)
            ).first()

    def usable(self, row: TrainedModel | None) -> bool:
        """True when the model exists on disk and loaded without error."""
        return row is not None and not row.missing and not row.load_error

    def load(self, ref: str) -> LoadedModel:
        """The loaded model of a reference (raises KeyError when unusable)."""
        row = self.row(ref)
        if row is None or not self.usable(row):
            raise KeyError(ref)
        return self.cache.get(row_to_info(row), self.dataset_root())
