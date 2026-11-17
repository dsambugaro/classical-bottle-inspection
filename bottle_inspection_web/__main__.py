"""`python -m bottle_inspection_web [--host 127.0.0.1] [--port 8000] [--data webdata]`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .settings import DEFAULT_HOST, DEFAULT_PORT, AppConfig


def main(argv: list[str] | None = None) -> None:
    """Starts the web app with a single uvicorn worker (no auth: binds to 127.0.0.1)."""
    parser = argparse.ArgumentParser(prog="bottle_inspection_web")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--data", type=Path, default=None, help="data folder (default code/webdata)"
    )
    parser.add_argument("--reload", action="store_true", help="development: reload on changes")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger(__name__).warning(
            "binding to %s: the app has no authentication, anyone on the network can use it",
            args.host,
        )
    if args.data is not None:
        import os

        os.environ["BOTTLE_WEB_DATA"] = str(args.data.resolve())
    if args.reload:
        uvicorn.run(
            "bottle_inspection_web.asgi:app", host=args.host, port=args.port, reload=True, workers=1
        )
        return
    from .app import create_app

    uvicorn.run(create_app(AppConfig.from_env()), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
