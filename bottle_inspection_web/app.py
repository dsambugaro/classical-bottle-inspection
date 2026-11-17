"""FastAPI application factory and lifespan (DB, model discovery, workers, frontend)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api import debug as debug_api
from .api import inspections as inspections_api
from .api import models as models_api
from .api import profiles as profiles_api
from .api import system as system_api
from .context import AppContext
from .db.session import Database
from .errors import ApiError
from .engine.analysis import configure_threads
from .events import EventBroker
from .inspections import InspectionService
from .logging_setup import setup_logging
from .models.registry import ModelRegistry
from .profiles import seed_profiles
from .settings import AppConfig
from .settings_store import SettingsStore

log = logging.getLogger(__name__)

ExtensionHook = Any
"""Callable(app, ctx) registering the services and routers of later phases."""


def build_context(config: AppConfig) -> AppContext:
    """Creates the folders, migrates the DB and wires every service."""
    config.paths.ensure()
    db = Database(config.paths.db)
    db.upgrade(config.paths.backups)
    settings = SettingsStore(db)
    events = EventBroker()
    registry = ModelRegistry(db, config.paths, settings)
    executor = ThreadPoolExecutor(
        max_workers=max(2, min(6, os.cpu_count() or 2)), thread_name_prefix="analysis"
    )
    inspections = InspectionService(db, registry, config.paths, settings, events, executor)
    ctx = AppContext(config, db, settings, registry, events, executor, inspections)
    registry.listeners.append(lambda changes: events.publish("models.changed", changes))
    return ctx


def discover(ctx: AppContext) -> None:
    """Model discovery + seed profiles (REQ-M3, REQ-PR2); warns when nothing is trained."""
    changes = ctx.registry.refresh()
    with ctx.db.session() as session:
        created = seed_profiles(session)
    usable = [r for r in ctx.registry.rows() if ctx.registry.usable(r)]
    if not usable:
        log.warning("no trained models found: run `make run` in code/ first")
    log.info(
        "model discovery",
        extra={"changes": changes, "seeded": [p.name for p in created], "models": len(usable)},
    )


def install_error_handlers(app: FastAPI) -> None:
    """Every error uses the {"error": {code, message, details}} format."""

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, error: ApiError) -> JSONResponse:
        """Known API errors."""
        return JSONResponse(error.body(), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        """Pydantic validation errors."""
        details = {
            "errors": [
                {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                for e in error.errors()
            ]
        }
        body = ApiError(422, "request.invalid", "invalid request", details).body()
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
        """404/405 and other HTTP errors."""
        code = {404: "http.not_found", 405: "http.method_not_allowed"}.get(
            error.status_code, "http.error"
        )
        return JSONResponse(
            ApiError(error.status_code, code, str(error.detail)).body(),
            status_code=error.status_code,
        )

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, error: Exception) -> JSONResponse:
        """Unexpected errors are logged and never crash the server (NF4)."""
        log.exception("unhandled error", exc_info=error)
        return JSONResponse(ApiError(500, "server.error", "internal error").body(), status_code=500)


def create_app(
    config: AppConfig | None = None, extensions: list[ExtensionHook] | None = None
) -> FastAPI:
    """The FastAPI app. `extensions` register the services of later phases (cameras, rules...)."""
    config = config or AppConfig.from_env()
    setup_logging(config.paths.logs)
    configure_threads()
    ctx = build_context(config)
    hooks = list(extensions if extensions is not None else default_extensions())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start-up and shutdown of the background services."""
        ctx.events.bind(asyncio.get_running_loop())
        await asyncio.get_running_loop().run_in_executor(ctx.executor, discover, ctx)
        services: list[Any] = list(ctx.extras.values())
        for service in services:
            start: Any = getattr(service, "start", None)
            if start is not None:
                await start()
        try:
            yield
        finally:
            for service in reversed(services):
                stop: Any = getattr(service, "stop", None)
                if stop is not None:
                    await stop()
            ctx.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Bottle inspection",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    install_error_handlers(app)
    for module in (system_api, models_api, profiles_api, inspections_api, debug_api):
        app.include_router(module.router)
    for hook in hooks:
        hook(app, ctx)
    if config.frontend_dist.is_dir() and (config.frontend_dist / "index.html").exists():
        app.frontend("/", directory=config.frontend_dist, fallback="index.html")
    else:
        log.warning("frontend build not found at %s (run `make web-build`)", config.frontend_dist)
    return app


def default_extensions() -> list[ExtensionHook]:
    """Services of the later phases, registered when their modules exist."""
    hooks: list[ExtensionHook] = []
    for name in ("cameras.service", "rules.service", "reports.api"):
        try:
            module = __import__(f"{__package__}.{name}", fromlist=["register"])
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith(__package__ or ""):
                continue
            raise
        hooks.append(module.register)
    return hooks
