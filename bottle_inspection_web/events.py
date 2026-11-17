"""In-process event broker feeding the SSE endpoint (GET /api/events)."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

EVENT_TYPES: tuple[str, ...] = (
    "analysis.completed",
    "inspection.completed",
    "session.progress",
    "session.completed",
    "models.changed",
    "rule.fired",
    "action.finished",
    "notification.created",
    "camera.status",
)

QUEUE_SIZE: int = 1000
"""Events buffered per subscriber; a slow client loses the oldest ones."""


class EventBroker:
    """Fan-out of events to every connected SSE client. `publish` is thread-safe."""

    def __init__(self) -> None:
        """No subscribers yet; the loop is bound on `bind`."""
        self._subscribers: set[asyncio.Queue[tuple[str, dict[str, Any]]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Binds the broker to the server event loop."""
        self._loop = loop

    def publish(self, event: str, data: dict[str, Any]) -> None:
        """Queues an event for every subscriber (callable from any thread)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._deliver(event, data)
        else:
            loop.call_soon_threadsafe(self._deliver, event, data)

    def _deliver(self, event: str, data: dict[str, Any]) -> None:
        """Puts the event in every queue (runs on the loop)."""
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait((event, data))

    def open(self) -> asyncio.Queue[tuple[str, dict[str, Any]]]:
        """Registers a subscriber queue (call `close` when the client disconnects)."""
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(QUEUE_SIZE)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def close(self, queue: asyncio.Queue[tuple[str, dict[str, Any]]]) -> None:
        """Unregisters a subscriber queue."""
        with self._lock:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        """Connected SSE clients."""
        return len(self._subscribers)
