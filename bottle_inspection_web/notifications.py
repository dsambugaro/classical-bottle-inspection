"""Notifications shown in the UI drawer (and pushed via SSE `notification.created`)."""

from __future__ import annotations

import logging
from typing import Any

from .db.models import Notification, utcnow
from .db.session import Database
from .events import EventBroker
from .profiles import iso

log = logging.getLogger(__name__)


def notify(
    db: Database,
    events: EventBroker,
    level: str,
    title: str,
    message: str = "",
    code: str | None = None,
    params: dict[str, Any] | None = None,
    link: str | None = None,
) -> int:
    """Stores a notification and publishes it; returns its id. Never raises (NF4)."""
    try:
        with db.session() as session:
            row = Notification(
                level=level,
                title=title[:255],
                message=message,
                code=code,
                params_json=params or {},
                link=link,
                created_at=utcnow(),
            )
            session.add(row)
            session.flush()
            payload = {
                "id": row.id,
                "level": level,
                "title": row.title,
                "message": message,
                "code": code,
                "params": params or {},
                "link": link,
                "created_at": iso(row.created_at),
            }
        events.publish("notification.created", payload)
        return int(payload["id"])
    except Exception:  # noqa: BLE001 - a notification must never break the caller
        log.exception("cannot store notification")
        return 0
