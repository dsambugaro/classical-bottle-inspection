"""Engine, sessions and start-up migrations of the SQLite database."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

MIGRATIONS_DIR: Path = Path(__file__).resolve().parent / "migrations"

log = logging.getLogger(__name__)


def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """WAL, foreign keys and a busy timeout on every connection (§13)."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def make_engine(db_path: Path) -> Engine:
    """SQLite engine with the connection pragmas; usable from several threads."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    event.listen(engine, "connect", _set_pragmas)
    return engine


class Database:
    """Owns the engine and the session factory."""

    def __init__(self, db_path: Path) -> None:
        """Creates the engine (the file is created on first use)."""
        self.path = db_path
        self.engine = make_engine(db_path)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A short transaction: commits on success, rolls back on error."""
        session = self.sessions()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def alembic_config(self) -> Config:
        """Alembic configuration pointing at this database and the package migrations."""
        config = Config()
        config.set_main_option("script_location", str(MIGRATIONS_DIR))
        config.set_main_option("sqlalchemy.url", f"sqlite:///{self.path}")
        config.attributes["engine"] = self.engine
        return config

    def current_revision(self) -> str | None:
        """Revision stored in the database (None for a new database)."""
        with self.engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()

    def upgrade(self, backups_dir: Path) -> None:
        """Runs pending migrations; backs up app.db first when it already holds data (NF7)."""
        config = self.alembic_config()
        head = ScriptDirectory.from_config(config).get_current_head()
        current = self.current_revision() if self.path.exists() else None
        if current == head:
            return
        if current is not None and self.path.exists():
            backups_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = backups_dir / f"app-{current}-{stamp}.db"
            with self.engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA wal_checkpoint(FULL)")
            shutil.copy2(self.path, backup)
            log.info("database backup before upgrade", extra={"backup": str(backup)})
        command.upgrade(config, "head")
        log.info("database upgraded", extra={"from": current, "to": head})
