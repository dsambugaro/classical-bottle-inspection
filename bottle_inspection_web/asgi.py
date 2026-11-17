"""ASGI entry point for `uvicorn --reload` (make web-dev)."""

from __future__ import annotations

from .app import create_app

app = create_app()
