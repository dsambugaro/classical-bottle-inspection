"""Analysis filters shared by GET /api/analyses and the reports (REQ-RP1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Query
from sqlalchemy import ColumnElement, and_, or_

from ..classes import CLASS_NAMES
from ..db.models import Analysis, Inspection
from ..errors import ApiError

PRESETS: dict[str, timedelta | None] = {
    "last_hour": timedelta(hours=1),
    "today": None,
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}


def parse_time(value: str | None) -> datetime | None:
    """ISO-8601 to naive UTC (how SQLite stores the timestamps)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ApiError(422, "filters.invalid_date", "invalid date", {"value": value}) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def split_list(values: list[str] | None) -> list[str]:
    """Flattens repeated and comma-separated query values."""
    out: list[str] = []
    for value in values or []:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


@dataclass
class AnalysisFilters:
    """Every filter of the reports top bar."""

    start: datetime | None = None
    end: datetime | None = None
    profile_ids: list[int] = field(default_factory=list)
    versions: list[tuple[int, int]] = field(default_factory=list)
    """(profile_id, version) pairs."""
    sources: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    session_id: int | None = None
    verdicts: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    inspection_id: int | None = None
    ground_truth_only: bool = False

    def clauses(self) -> list[ColumnElement[bool]]:
        """SQLAlchemy WHERE clauses on Analysis (and Inspection for camera names)."""
        where: list[ColumnElement[bool]] = []
        if self.start is not None:
            where.append(Analysis.created_at >= self.start)
        if self.end is not None:
            where.append(Analysis.created_at < self.end)
        if self.profile_ids:
            where.append(Analysis.profile_id.in_(self.profile_ids))
        if self.versions:
            where.append(
                or_(
                    *[
                        and_(Analysis.profile_id == pid, Analysis.profile_version == v)
                        for pid, v in self.versions
                    ]
                )
            )
        if self.sources:
            where.append(Analysis.source.in_(self.sources))
        if self.cameras:
            where.append(
                Analysis.inspection_id.in_(
                    Inspection.__table__.select()
                    .with_only_columns(Inspection.id)
                    .where(Inspection.camera_name.in_(self.cameras))
                )
            )
        if self.session_id is not None:
            where.append(Analysis.session_id == self.session_id)
        if self.verdicts:
            where.append(Analysis.verdict.in_(self.verdicts))
        if self.classes:
            where.append(Analysis.predicted_class.in_(self.classes))
        if self.inspection_id is not None:
            where.append(Analysis.inspection_id == self.inspection_id)
        if self.ground_truth_only:
            where.append(Analysis.ground_truth_class.is_not(None))
        return where

    def previous_period(self) -> AnalysisFilters | None:
        """The same filters over the previous period of equal length (for KPI deltas)."""
        if self.start is None:
            return None
        end = self.end or datetime.now(timezone.utc).replace(tzinfo=None)
        length = end - self.start
        return AnalysisFilters(
            start=self.start - length,
            end=self.start,
            profile_ids=self.profile_ids,
            versions=self.versions,
            sources=self.sources,
            cameras=self.cameras,
            session_id=self.session_id,
            verdicts=self.verdicts,
            classes=self.classes,
            ground_truth_only=self.ground_truth_only,
        )

    def to_json(self) -> dict[str, Any]:
        """Echo of the applied filters."""
        return {
            "start": self.start.isoformat() + "Z" if self.start else None,
            "end": self.end.isoformat() + "Z" if self.end else None,
            "profile_ids": self.profile_ids,
            "versions": [f"{p}:{v}" for p, v in self.versions],
            "sources": self.sources,
            "cameras": self.cameras,
            "session_id": self.session_id,
            "verdicts": self.verdicts,
            "classes": self.classes,
        }


def filters_dependency(
    preset: str | None = None,
    start: str | None = None,
    end: str | None = None,
    profile: list[str] | None = Query(default=None),
    version: list[str] | None = Query(default=None),
    source: list[str] | None = Query(default=None),
    camera: list[str] | None = Query(default=None),
    session: int | None = None,
    verdict: list[str] | None = Query(default=None),
    cls: list[str] | None = Query(default=None, alias="class"),
    inspection: int | None = None,
    ground_truth_only: bool = False,
) -> AnalysisFilters:
    """FastAPI dependency building AnalysisFilters from the query string."""
    filters = AnalysisFilters(start=parse_time(start), end=parse_time(end))
    if preset and preset in PRESETS and filters.start is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if preset == "today":
            filters.start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif PRESETS[preset] is not None:
            filters.start = now - PRESETS[preset]  # type: ignore[operator]
    try:
        filters.profile_ids = [int(v) for v in split_list(profile)]
        filters.versions = [
            (int(a), int(b)) for a, _, b in (v.partition(":") for v in split_list(version))
        ]
    except ValueError as error:
        raise ApiError(422, "filters.invalid_profile", "invalid profile filter") from error
    filters.sources = split_list(source)
    filters.cameras = split_list(camera)
    filters.session_id = session
    filters.verdicts = [v for v in split_list(verdict) if v in ("pass", "reject", "review")]
    filters.classes = [c for c in split_list(cls) if c in CLASS_NAMES]
    filters.inspection_id = inspection
    filters.ground_truth_only = ground_truth_only
    return filters
