"""Paths, limits and the defaults of the user-editable settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bottle_inspection.env import data_path_setting

CODE_DIR: Path = Path(__file__).resolve().parent.parent
"""The `code/` folder: model folders and the dataset path are relative to it."""

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8000

SETTINGS_DEFAULTS: dict[str, Any] = {
    "language": "en",
    "storage_policy": "rejects_only",
    "image_retention_days": 30,
    "execution_retention_days": 30,
    "model_folders": ["results"],
    "dataset_path": data_path_setting(),
    "max_upload_mb": 25,
    "max_batch_files": 500,
    "max_profiles": 6,
    "default_fps": 3,
    "min_fps": 1,
    "max_fps": 15,
    "background_capture_frames": 30,
    "motion_threshold": 0.04,
}
"""Values used when the `settings` table has no row for a key."""

STORAGE_POLICIES: tuple[str, ...] = ("all", "rejects_only", "none")
LANGUAGES: tuple[str, ...] = ("en", "pt-BR")


@dataclass(frozen=True)
class AppPaths:
    """Every runtime path, under one data folder (default `code/webdata/`)."""

    root: Path
    code_dir: Path = field(default=CODE_DIR)

    @property
    def db(self) -> Path:
        """The SQLite database file."""
        return self.root / "app.db"

    @property
    def images(self) -> Path:
        """Stored inspection images (yyyy/mm/dd/<uuid>.<ext>)."""
        return self.root / "images"

    @property
    def thumbs(self) -> Path:
        """Cached thumbnails."""
        return self.root / "thumbs"

    @property
    def backgrounds(self) -> Path:
        """Captured or uploaded backgrounds of profiles."""
        return self.root / "backgrounds"

    @property
    def exports(self) -> Path:
        """Files written by the `file` integration target and exports."""
        return self.root / "exports"

    @property
    def logs(self) -> Path:
        """JSON-lines logs, rotated daily."""
        return self.root / "logs"

    @property
    def secrets(self) -> Path:
        """Secrets file (mode 0600), never stored in the DB."""
        return self.root / "secrets.toml"

    @property
    def backups(self) -> Path:
        """Copies of app.db made before schema upgrades."""
        return self.root / "backups"

    def ensure(self) -> None:
        """Creates every folder."""
        for folder in (
            self.root,
            self.images,
            self.thumbs,
            self.backgrounds,
            self.exports,
            self.logs,
            self.backups,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def resolve_code_path(self, value: str | Path) -> Path:
        """Resolves a path relative to `code/` (absolute paths are kept)."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.code_dir / path).resolve()


@dataclass(frozen=True)
class AppConfig:
    """Start-up configuration (command line / environment), not stored in the DB."""

    paths: AppPaths
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    frontend_dist: Path = field(default=CODE_DIR / "frontend" / "dist")
    start_workers: bool = True
    """Background workers (rule actions, retention). Tests may disable them."""

    @staticmethod
    def from_env(data_dir: Path | None = None) -> AppConfig:
        """Configuration from BOTTLE_WEB_DATA or the default `code/webdata/`."""
        root = data_dir or Path(os.environ.get("BOTTLE_WEB_DATA", CODE_DIR / "webdata"))
        return AppConfig(paths=AppPaths(root=Path(root).resolve()))
