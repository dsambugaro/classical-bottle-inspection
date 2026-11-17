"""Inspection history: a cursor-paginated list of inspections with all their analyses.

An inspection matches when at least one of its analyses matches the filters; the row still
carries every analysis of the inspection (so the profiles stay side by side).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import ColumnElement, and_, distinct, func, select
from sqlalchemy.orm import Session

from .classes import CLASS_NAMES
from .db.models import Analysis, Image, Inspection, Profile
from .inspections import analysis_json
from .profiles import iso
from .reports.filters import AnalysisFilters, filters_dependency, split_list

MAX_PAGE: int = 100
"""Largest page size a client can ask for."""


@dataclass
class HistoryFilters:
    """The reports filters plus the ones that only make sense per inspection."""

    base: AnalysisFilters = field(default_factory=AnalysisFilters)
    truths: list[str] = field(default_factory=list)
    """Ground-truth classes (any of)."""
    only_wrong: bool = False
    """Ground truth known and different from the predicted class."""
    disagree: bool = False
    """The analyses of the inspection do not all predict the same class."""
    min_confidence: float | None = None
    max_confidence: float | None = None
    name: str | None = None
    """Case-insensitive part of the original file name."""

    def analysis_clauses(self) -> list[ColumnElement[bool]]:
        """Conditions one analysis must satisfy (all together) for its inspection to match."""
        # The camera filter is a sub-select on inspections: applied on the outer query instead.
        where = replace(self.base, cameras=[]).clauses()
        if self.truths:
            where.append(Analysis.ground_truth_class.in_(self.truths))
        if self.only_wrong:
            where.append(
                and_(
                    Analysis.ground_truth_class.is_not(None),
                    Analysis.ground_truth_class != Analysis.predicted_class,
                )
            )
        if self.min_confidence is not None:
            where.append(Analysis.confidence >= self.min_confidence)
        if self.max_confidence is not None:
            where.append(Analysis.confidence <= self.max_confidence)
        return where

    def inspection_clauses(self) -> list[ColumnElement[bool]]:
        """Conditions on the inspection row itself."""
        where: list[ColumnElement[bool]] = []
        if self.base.cameras:
            where.append(Inspection.camera_name.in_(self.base.cameras))
        if self.name:
            where.append(Inspection.original_name.icontains(self.name, autoescape=True))
        if self.disagree:
            classes = (
                select(func.count(distinct(Analysis.predicted_class)))
                .where(Analysis.inspection_id == Inspection.id)
                .scalar_subquery()
            )
            where.append(classes > 1)
        return where


def history_filters_dependency(
    base: AnalysisFilters = Depends(filters_dependency),
    truth: list[str] | None = Query(default=None),
    only_wrong: bool = False,
    disagree: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    name: str | None = None,
) -> HistoryFilters:
    """FastAPI dependency: the report filters plus the history-only ones."""
    return HistoryFilters(
        base=base,
        truths=[c for c in split_list(truth) if c in CLASS_NAMES],
        only_wrong=only_wrong,
        disagree=disagree,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        name=(name or "").strip() or None,
    )


def history_page(
    session: Session, filters: HistoryFilters, before: int | None, limit: int
) -> dict[str, Any]:
    """One page of inspections (newest first, ids below `before`) with all their analyses.

    Three queries per page regardless of its size: matching ids (an EXISTS that stops at the
    first matching analysis of each inspection), the inspections with their images, and every
    analysis with its profile.
    """
    limit = min(max(limit, 1), MAX_PAGE)
    matching = select(Analysis.id).where(
        Analysis.inspection_id == Inspection.id, *filters.analysis_clauses()
    )
    query = (
        select(Inspection.id)
        .where(matching.exists(), *filters.inspection_clauses())
        .order_by(Inspection.id.desc())
        .limit(limit + 1)
    )
    if before is not None:
        query = query.where(Inspection.id < before)
    ids = list(session.scalars(query))
    has_more = len(ids) > limit
    ids = ids[:limit]
    if not ids:
        return {"items": [], "next_cursor": None, "has_more": False, "limit": limit}

    inspections = session.execute(
        select(Inspection, Image)
        .outerjoin(Image, Image.id == Inspection.image_id)
        .where(Inspection.id.in_(ids))
        .order_by(Inspection.id.desc())
    ).all()
    by_inspection: dict[int, list[tuple[Analysis, Profile]]] = {i: [] for i in ids}
    for analysis, profile in session.execute(
        select(Analysis, Profile)
        .join(Profile, Profile.id == Analysis.profile_id)
        .where(Analysis.inspection_id.in_(ids))
        .order_by(Analysis.id)
    ):
        by_inspection[analysis.inspection_id].append((analysis, profile))

    items = []
    for inspection, image in inspections:
        analyses = [analysis_json(a, p, image, inspection) for a, p in by_inspection[inspection.id]]
        items.append(
            {
                "inspection_id": inspection.id,
                "session_id": inspection.session_id,
                "source": inspection.source,
                "camera_name": inspection.camera_name,
                "original_name": inspection.original_name,
                "ground_truth": inspection.ground_truth_class,
                "image": analyses[0]["image"] if analyses else None,
                "created_at": iso(inspection.created_at),
                "analyses": analyses,
            }
        )
    return {
        "items": items,
        "next_cursor": ids[-1] if has_more else None,
        "has_more": has_more,
        "limit": limit,
    }
