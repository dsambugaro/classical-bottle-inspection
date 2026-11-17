"""User-editable settings stored in the `settings` table, with defaults."""

from __future__ import annotations

import threading
from typing import Any

from .db.models import Setting
from .db.session import Database
from .settings import LANGUAGES, SETTINGS_DEFAULTS, STORAGE_POLICIES


class SettingsError(ValueError):
    """Invalid setting value; `code` is translated by the frontend."""

    def __init__(self, key: str, code: str) -> None:
        """Error for one key."""
        super().__init__(f"{key}: {code}")
        self.key = key
        self.code = code


def validate_setting(key: str, value: Any) -> Any:
    """Validates and normalizes one setting value."""
    if key not in SETTINGS_DEFAULTS:
        raise SettingsError(key, "settings.unknown")
    default = SETTINGS_DEFAULTS[key]
    if key == "language" and value not in LANGUAGES:
        raise SettingsError(key, "settings.invalid_option")
    if key == "storage_policy" and value not in STORAGE_POLICIES:
        raise SettingsError(key, "settings.invalid_option")
    if key == "model_folders":
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise SettingsError(key, "settings.invalid_type")
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise SettingsError(key, "settings.invalid_type")
    elif isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise SettingsError(key, "settings.invalid_type")
        if isinstance(default, int) and not float(value).is_integer():
            raise SettingsError(key, "settings.invalid_type")
        return type(default)(value)
    elif isinstance(default, str) and not isinstance(value, str):
        raise SettingsError(key, "settings.invalid_type")
    return value


class SettingsStore:
    """Cached access to the settings table."""

    def __init__(self, db: Database) -> None:
        """Reads lazily from `db`."""
        self.db = db
        self._cache: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def all(self) -> dict[str, Any]:
        """Every setting (defaults merged with stored values)."""
        with self._lock:
            if self._cache is None:
                with self.db.session() as session:
                    stored = {row.key: row.value_json for row in session.query(Setting).all()}
                self._cache = {**SETTINGS_DEFAULTS, **stored}
            return dict(self._cache)

    def get(self, key: str) -> Any:
        """One setting."""
        return self.all()[key]

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validates and stores several settings at once."""
        cleaned = {key: validate_setting(key, value) for key, value in values.items()}
        if (
            "min_fps" in cleaned or "max_fps" in cleaned or "default_fps" in cleaned
        ) and not self._fps_ok({**self.all(), **cleaned}):
            raise SettingsError("default_fps", "settings.fps_order")
        with self.db.session() as session:
            for key, value in cleaned.items():
                session.merge(Setting(key=key, value_json=value))
        with self._lock:
            self._cache = None
        return self.all()

    @staticmethod
    def _fps_ok(values: dict[str, Any]) -> bool:
        """min_fps <= default_fps <= max_fps."""
        return values["min_fps"] <= values["default_fps"] <= values["max_fps"]
