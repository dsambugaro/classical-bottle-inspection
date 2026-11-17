"""The application context shared by the routers (one per process)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from .db.session import Database
from .events import EventBroker
from .inspections import InspectionService
from .models.registry import ModelRegistry
from .settings import AppConfig
from .settings_store import SettingsStore


@dataclass
class AppContext:
    """Every long-lived service of the app."""

    config: AppConfig
    db: Database
    settings: SettingsStore
    registry: ModelRegistry
    events: EventBroker
    executor: ThreadPoolExecutor
    inspections: InspectionService
    extras: dict[str, Any] = field(default_factory=dict)
    """Services added by later phases (cameras, rules), keyed by name."""


def get_ctx(request: Request) -> AppContext:
    """FastAPI dependency returning the app context."""
    return request.app.state.ctx
