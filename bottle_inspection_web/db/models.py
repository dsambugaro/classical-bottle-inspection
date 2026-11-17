"""SQLAlchemy ORM models (§13). JSON columns hold validated dicts/lists."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Named constraints, so Alembic batch migrations can alter them on SQLite."""


def utcnow() -> datetime:
    """Current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base with the naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def as_utc(value: datetime | None) -> datetime | None:
    """Treats naive datetimes read from SQLite as UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class Profile(Base):
    """A named, versioned recipe: trained model + parameters + verdict settings."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(255))
    algorithm: Mapped[str] = mapped_column(String(64))
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    verdict_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    background_source: Mapped[str] = mapped_column(String(16), default="model")
    background_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    feature_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """Feature vector length of the linked model when the profile was created or re-linked."""
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ProfileVersion(Base):
    """Snapshot of the versioned settings of a profile."""

    __tablename__ = "profile_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_profile_versions_profile_version", "profile_id", "version"),)


class TrainedModel(Base):
    """A model.joblib found in a configured model folder (read-only)."""

    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_ref: Mapped[str] = mapped_column(String(255), unique=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(String(1024))
    file_mtime: Mapped[float] = mapped_column(Float)
    training_params_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    best_params_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    feature_length: Mapped[int] = mapped_column(Integer)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    library_versions_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    params_inferred: Mapped[bool] = mapped_column(Boolean, default=False)
    version_warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    load_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_on: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    """Set when the file changed after the first discovery (make run executed again)."""
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)


class Camera(Base):
    """A camera attached to the server (USB, RTSP or HTTP)."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    kind: Mapped[str] = mapped_column(String(8))
    device_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float] = mapped_column(Float, default=3.0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class Session(Base):
    """A continuous run (batch upload or camera stream)."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    profile_ids_json: Mapped[list[int]] = mapped_column(JSON, default=list)
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    total: Mapped[int] = mapped_column(Integer, default=0)
    """Images expected in a batch (0 for camera sessions)."""


class Image(Base):
    """A stored image file (the raw image; overlays are never burned in)."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16))
    original_name: Mapped[str] = mapped_column(String(255), default="")
    ground_truth_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Inspection(Base):
    """One image processed by one or more profiles."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    image_id: Mapped[int | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(16), default="upload")
    camera_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    original_name: Mapped[str] = mapped_column(String(255), default="")
    ground_truth_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_inspections_session_id", "session_id"),
        Index("ix_inspections_created_at", "created_at"),
    )


class Analysis(Base):
    """One inspection image analysed by one profile."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"))
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"))
    profile_version: Mapped[int] = mapped_column(Integer)
    trained_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id", ondelete="SET NULL"), nullable=True
    )
    model_mtime: Mapped[float | None] = mapped_column(Float, nullable=True)
    modified_params: Mapped[bool] = mapped_column(Boolean, default=False)
    predicted_class: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    scores_json: Mapped[dict[str, float]] = mapped_column(JSON)
    score_kind: Mapped[str] = mapped_column(String(32))
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    geometry_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    timings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    total_ms: Mapped[float] = mapped_column(Float)
    warnings_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(16), default="upload")
    """Denormalized from the inspection for fast report filters."""
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ground_truth_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_profile_created", "profile_id", "created_at"),
        Index("ix_analyses_class_created", "predicted_class", "created_at"),
        Index("ix_analyses_verdict_created", "verdict", "created_at"),
        Index("ix_analyses_inspection_id", "inspection_id"),
    )


class Target(Base):
    """Reusable connection settings of an integration (secrets live outside the DB)."""

    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    type: Mapped[str] = mapped_column(String(16))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    secret_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_test_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_test_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_test_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class Rule(Base):
    """A condition -> action automation."""

    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    severity: Mapped[str] = mapped_column(String(8), default="info")
    scope_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trigger: Mapped[str] = mapped_column(String(32), default="analysis_completed")
    condition_logic: Mapped[str] = mapped_column(String(4), default="all")
    cooldown_s: Mapped[int] = mapped_column(Integer, default=60)
    dedupe_key: Mapped[str] = mapped_column(String(16), default="rule+class")
    run_mode: Mapped[str] = mapped_column(String(8), default="single")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RuleCondition(Base):
    """One condition of a rule."""

    __tablename__ = "rule_conditions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("rules.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    negate: Mapped[bool] = mapped_column(Boolean, default=False)


class RuleAction(Base):
    """One action of a rule."""

    __tablename__ = "rule_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("rules.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[int | None] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    timeout_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)


class ActionExecution(Base):
    """One execution (or suppression window) of a rule action."""

    __tablename__ = "action_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rule_id: Mapped[int | None] = mapped_column(
        ForeignKey("rules.id", ondelete="SET NULL"), nullable=True
    )
    action_id: Mapped[int | None] = mapped_column(
        ForeignKey("rule_actions.id", ondelete="SET NULL"), nullable=True
    )
    target_id: Mapped[int | None] = mapped_column(
        ForeignKey("targets.id", ondelete="SET NULL"), nullable=True
    )
    analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True
    )
    inspection_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_uuid: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(24))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    rendered_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    suppressed_count: Mapped[int] = mapped_column(Integer, default=0)
    action_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dedupe_value: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (Index("ix_action_executions_rule_created", "rule_id", "created_at"),)


class Notification(Base):
    """A message shown in the notifications drawer."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    level: Mapped[str] = mapped_column(String(8))
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """Translatable code (REQ-L4); title/message are a fallback or user content."""
    params_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    link: Mapped[str | None] = mapped_column(String(512), nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Setting(Base):
    """One user-editable setting (language, storage policy, model folders, limits...)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON)
