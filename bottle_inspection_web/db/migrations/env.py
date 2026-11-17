"""Alembic environment: runs migrations on the engine passed by Database.upgrade."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from bottle_inspection_web.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations() -> None:
    """Online migrations (batch mode, required to alter tables on SQLite)."""
    engine = config.attributes.get("engine") or create_engine(
        str(config.get_main_option("sqlalchemy.url"))
    )
    with engine.begin() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
