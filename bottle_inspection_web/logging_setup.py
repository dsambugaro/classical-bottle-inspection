"""Structured logs (JSON lines) in webdata/logs/, rotated daily and kept 14 days (NF5)."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESERVED: frozenset[str] = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message"}
"""Attributes of a LogRecord that are not user `extra` fields."""


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, extras, exception."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats a record as JSON."""
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Adds the rotating JSON file handler (and a plain console handler) to the root logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_bottle_web", False):
            root.removeHandler(handler)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "app.jsonl", when="midnight", backupCount=14, utc=True, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    setattr(file_handler, "_bottle_web", True)
    root.addHandler(file_handler)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        setattr(console, "_bottle_web", True)
        root.addHandler(console)
