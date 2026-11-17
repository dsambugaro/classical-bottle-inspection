"""Report aggregations, done in SQL with the indexes of §13 (REQ-RP6).

Every function takes the same AnalysisFilters. Class lists always contain all 7 classes in
id order (zeros included).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any

from sqlalchemy import Integer, case, cast, distinct, func, select
from sqlalchemy.orm import Session

from bottle_inspection.config import FEATURE_GROUPS

from ..classes import CLASS_NAMES, ordered_counts
from ..db.models import ActionExecution, Analysis, Inspection, Profile
from .filters import AnalysisFilters

STEPS: tuple[str, ...] = (
    "decode",
    "normalize",
    "foreground_mask",
    "localize",
    "crop_regions",
)
FEATURE_STEPS: tuple[str, ...] = tuple(FEATURE_GROUPS)
"""One timing per feature group; a model only computes the groups of its own vector."""
CONFIDENCE_BINS: int = 20


def now_naive() -> datetime:
    """Current UTC time, naive (how SQLite stores timestamps)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 with Z."""
    return value.isoformat(timespec="seconds") + "Z" if value else None


# ----------------------------------------------------------------- KPIs


def kpis(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Counts, rates, timing and throughput for one period."""
    where = filters.clauses()
    row = session.execute(
        select(
            func.count(Analysis.id),
            func.count(distinct(Analysis.inspection_id)),
            func.sum(case((Analysis.verdict == "pass", 1), else_=0)),
            func.sum(case((Analysis.verdict == "reject", 1), else_=0)),
            func.sum(case((Analysis.verdict == "review", 1), else_=0)),
            func.avg(Analysis.total_ms),
            func.min(Analysis.created_at),
            func.max(Analysis.created_at),
        ).where(*where)
    ).one()
    analyses, inspected, passed, rejected, review, mean_ms, first, last = row
    analyses = int(analyses or 0)
    p95 = None
    if analyses:
        offset = max(0, int(round(0.95 * (analyses - 1))))
        p95 = session.scalar(
            select(Analysis.total_ms)
            .where(*where)
            .order_by(Analysis.total_ms)
            .offset(offset)
            .limit(1)
        )
    start = filters.start or first
    end = filters.end or (now_naive() if filters.start else last)
    minutes = (end - start).total_seconds() / 60 if start and end and end > start else None
    firings = (
        select(ActionExecution.rule_id, ActionExecution.analysis_id)
        .where(
            ActionExecution.status.in_(("success", "failed", "running", "queued")),
            ActionExecution.is_test.is_(False),
            *([ActionExecution.created_at >= filters.start] if filters.start is not None else []),
            *([ActionExecution.created_at < filters.end] if filters.end is not None else []),
        )
        .distinct()
        .subquery()
    )
    fired = session.scalar(select(func.count()).select_from(firings))
    rate = lambda n: (float(n or 0) / analyses) if analyses else None  # noqa: E731
    return {
        "analyses": analyses,
        "inspected": int(inspected or 0),
        "pass": int(passed or 0),
        "reject": int(rejected or 0),
        "review": int(review or 0),
        "pass_rate": rate(passed),
        "reject_rate": rate(rejected),
        "review_rate": rate(review),
        "first_pass_yield": rate(passed),
        "mean_ms": float(mean_ms) if mean_ms is not None else None,
        "p95_ms": float(p95) if p95 is not None else None,
        "throughput_per_min": (int(inspected or 0) / minutes) if minutes else None,
        "rules_fired": int(fired or 0),
    }


def summary(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """KPI tiles with the previous equivalent period, and the class distribution."""
    current = kpis(session, filters)
    previous_filters = filters.previous_period()
    previous = kpis(session, previous_filters) if previous_filters else None
    counts = dict(
        session.execute(
            select(Analysis.predicted_class, func.count(Analysis.id))
            .where(*filters.clauses())
            .group_by(Analysis.predicted_class)
        ).all()
    )
    total = sum(counts.values())
    distribution = [
        {
            "class": name,
            "count": int(counts.get(name, 0)),
            "share": (counts.get(name, 0) / total) if total else 0.0,
        }
        for name in CLASS_NAMES
    ]
    return {
        "filters": filters.to_json(),
        "kpis": current,
        "previous": previous,
        "class_distribution": distribution,
    }


# ----------------------------------------------------------------- Pareto


def pareto(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Reject classes sorted by count, with the cumulative share (only rejects)."""
    rows = session.execute(
        select(Analysis.predicted_class, func.count(Analysis.id))
        .where(*filters.clauses(), Analysis.verdict == "reject")
        .group_by(Analysis.predicted_class)
    ).all()
    counts = {name: int(n) for name, n in rows}
    total = sum(counts.values())
    order = sorted(CLASS_NAMES, key=lambda name: (-counts.get(name, 0), CLASS_NAMES.index(name)))
    items, cumulative = [], 0
    for name in order:
        count = counts.get(name, 0)
        cumulative += count
        items.append(
            {
                "class": name,
                "count": count,
                "share": count / total if total else 0.0,
                "cumulative": cumulative / total if total else 0.0,
            }
        )
    return {"total_rejects": total, "items": items}


# ----------------------------------------------------------------- time series


def bucket_for(filters: AnalysisFilters, first: datetime | None, last: datetime | None) -> str:
    """minute / hour / day from the span of the data inside the selected range."""
    if first is None or last is None:
        return "hour"
    start = max(first, filters.start) if filters.start else first
    end = min(last, filters.end) if filters.end else last
    span = end - start
    if span <= timedelta(hours=3):
        return "minute"
    if span <= timedelta(days=3):
        return "hour"
    return "day"


FORMATS: dict[str, str] = {
    "minute": "%Y-%m-%dT%H:%M:00",
    "hour": "%Y-%m-%dT%H:00:00",
    "day": "%Y-%m-%dT00:00:00",
}


def timeseries(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Good vs defective per bucket, and rejects by class per bucket."""
    where = filters.clauses()
    first, last = session.execute(
        select(func.min(Analysis.created_at), func.max(Analysis.created_at)).where(*where)
    ).one()
    bucket = bucket_for(filters, first, last)
    key = func.strftime(FORMATS[bucket], Analysis.created_at).label("bucket")
    rows = session.execute(
        select(key, Analysis.verdict, Analysis.predicted_class, func.count(Analysis.id))
        .where(*where)
        .group_by(key, Analysis.verdict, Analysis.predicted_class)
    ).all()
    buckets: dict[str, dict[str, Any]] = {}
    for stamp, verdict, name, count in rows:
        item = buckets.setdefault(
            stamp, {"t": stamp + "Z", "pass": 0, "reject": 0, "review": 0, "rejects_by_class": {}}
        )
        item[verdict] += int(count)
        if verdict == "reject":
            item["rejects_by_class"][name] = item["rejects_by_class"].get(name, 0) + int(count)
    series = []
    for stamp in sorted(buckets):
        item = buckets[stamp]
        item["rejects_by_class"] = ordered_counts(item["rejects_by_class"])
        series.append(item)
    return {"bucket": bucket, "series": series}


# ----------------------------------------------------------------- profiles


def profiles_report(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Profiles x 7 classes (counts), verdict counts and pairwise verdict agreement."""
    where = filters.clauses()
    rows = session.execute(
        select(
            Analysis.profile_id, Analysis.predicted_class, Analysis.verdict, func.count(Analysis.id)
        )
        .where(*where)
        .group_by(Analysis.profile_id, Analysis.predicted_class, Analysis.verdict)
    ).all()
    names = {
        p.id: (p.name, p.color, p.algorithm)
        for p in session.scalars(select(Profile).where(Profile.id.in_({r[0] for r in rows})))
    }
    table: dict[int, dict[str, Any]] = {}
    for profile_id, name, verdict, count in rows:
        entry = table.setdefault(
            profile_id,
            {"classes": {}, "verdicts": {"pass": 0, "reject": 0, "review": 0}, "total": 0},
        )
        entry["classes"][name] = entry["classes"].get(name, 0) + int(count)
        entry["verdicts"][verdict] += int(count)
        entry["total"] += int(count)
    profiles = [
        {
            "profile_id": pid,
            "name": names.get(pid, (f"#{pid}", "#888888", ""))[0],
            "color": names.get(pid, (f"#{pid}", "#888888", ""))[1],
            "algorithm": names.get(pid, (f"#{pid}", "#888888", ""))[2],
            "total": entry["total"],
            "verdicts": entry["verdicts"],
            "classes": ordered_counts(entry["classes"]),
        }
        for pid, entry in sorted(table.items())
    ]
    agreement = pairwise_agreement(session, filters, [p["profile_id"] for p in profiles])
    return {"classes": CLASS_NAMES, "profiles": profiles, "agreement": agreement}


def pairwise_agreement(
    session: Session, filters: AnalysisFilters, profile_ids: list[int]
) -> list[dict[str, Any]]:
    """Share of inspections analysed by both profiles of a pair with the same verdict / class.

    One scan of the filtered analyses ordered by inspection (a self-join is far slower on
    SQLite at 100k rows).
    """
    if len(profile_ids) < 2:
        return []
    rows = session.execute(
        select(
            Analysis.inspection_id, Analysis.profile_id, Analysis.verdict, Analysis.predicted_class
        )
        .where(*filters.clauses())
        .order_by(Analysis.inspection_id)
    ).tuples()
    pairs = {pair: [0, 0, 0] for pair in combinations(sorted(profile_ids), 2)}
    current: int | None = None
    group: dict[int, tuple[str, str]] = {}

    def flush() -> None:
        """Counts the pairs of one inspection."""
        for first, second in combinations(sorted(group), 2):
            counts = pairs.get((first, second))
            if counts is not None:
                counts[0] += 1
                counts[1] += group[first][0] == group[second][0]
                counts[2] += group[first][1] == group[second][1]

    for inspection_id, profile_id, verdict, predicted in rows:
        if inspection_id != current:
            flush()
            current, group = inspection_id, {}
        group[profile_id] = (verdict, predicted)
    flush()
    return [
        {
            "profiles": list(pair),
            "inspections": both,
            "verdict_agreement": same_verdict / both,
            "class_agreement": same_class / both,
        }
        for pair, (both, same_verdict, same_class) in pairs.items()
        if both
    ]


# ----------------------------------------------------------------- confidence, timings


def confidence_report(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Confidence histogram (20 bins) per profile, with its min_confidence when set."""
    bin_index = cast(Analysis.confidence * CONFIDENCE_BINS, Integer).label("bin")
    rows = session.execute(
        select(Analysis.profile_id, bin_index, func.count(Analysis.id))
        .where(*filters.clauses())
        .group_by(Analysis.profile_id, bin_index)
    ).all()
    profiles = {
        p.id: p
        for p in session.scalars(select(Profile).where(Profile.id.in_({r[0] for r in rows})))
    }
    result: dict[int, dict[str, Any]] = {}
    for pid, index, count in rows:
        profile = profiles.get(pid)
        entry = result.setdefault(
            pid,
            {
                "profile_id": pid,
                "name": profile.name if profile else f"#{pid}",
                "color": profile.color if profile else "#888888",
                "algorithm": profile.algorithm if profile else "",
                "min_confidence": (
                    (profile.verdict_json or {}).get("min_confidence") if profile else None
                ),
                "bins": [0] * CONFIDENCE_BINS,
            },
        )
        entry["bins"][max(0, min(CONFIDENCE_BINS - 1, int(index or 0)))] += int(count)
    return {
        "edges": [i / CONFIDENCE_BINS for i in range(CONFIDENCE_BINS + 1)],
        "profiles": [result[k] for k in sorted(result)],
    }


def timings_report(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Mean of each pipeline step, per profile (from the timings JSON)."""
    columns = [
        func.avg(func.json_extract(Analysis.timings_json, f"$.{step}")).label(step)
        for step in STEPS
    ]
    columns += [
        func.avg(func.json_extract(Analysis.timings_json, f"$.features.{step}")).label(step)
        for step in FEATURE_STEPS
    ]
    columns.append(func.avg(func.json_extract(Analysis.timings_json, "$.predict")).label("predict"))
    rows = session.execute(
        select(Analysis.profile_id, func.count(Analysis.id), func.avg(Analysis.total_ms), *columns)
        .where(*filters.clauses())
        .group_by(Analysis.profile_id)
    ).all()
    profiles = {
        p.id: p
        for p in session.scalars(select(Profile).where(Profile.id.in_({r[0] for r in rows})))
    }
    steps = list(STEPS) + list(FEATURE_STEPS) + ["predict"]
    return {
        "steps": steps,
        "profiles": [
            {
                "profile_id": row[0],
                "name": profiles[row[0]].name if row[0] in profiles else f"#{row[0]}",
                "color": profiles[row[0]].color if row[0] in profiles else "#888888",
                "count": int(row[1]),
                "mean_total_ms": float(row[2] or 0),
                "steps": {step: float(value or 0) for step, value in zip(steps, row[3:])},
            }
            for row in rows
        ],
    }


# ----------------------------------------------------------------- ground truth


def macro_f1(matrix: list[list[int]]) -> float:
    """Macro F1 over the classes that appear (as true or predicted)."""
    scores = []
    for i in range(len(matrix)):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(matrix))) - tp
        fn = sum(matrix[i]) - tp
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def confusion_report(session: Session, filters: AnalysisFilters) -> dict[str, Any]:
    """Per profile: 7x7 confusion matrix, accuracy and macro F1 (dataset samples only)."""
    rows = session.execute(
        select(
            Analysis.profile_id,
            Analysis.ground_truth_class,
            Analysis.predicted_class,
            func.count(Analysis.id),
        )
        .where(*filters.clauses(), Analysis.ground_truth_class.is_not(None))
        .group_by(Analysis.profile_id, Analysis.ground_truth_class, Analysis.predicted_class)
    ).all()
    index = {name: i for i, name in enumerate(CLASS_NAMES)}
    matrices: dict[int, list[list[int]]] = {}
    for pid, truth, predicted, count in rows:
        matrix = matrices.setdefault(pid, [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES])
        if truth in index and predicted in index:
            matrix[index[truth]][index[predicted]] += int(count)
    profiles = {
        p.id: p for p in session.scalars(select(Profile).where(Profile.id.in_(set(matrices))))
    }
    result = []
    for pid in sorted(matrices):
        matrix = matrices[pid]
        total = sum(map(sum, matrix))
        correct = sum(matrix[i][i] for i in range(len(matrix)))
        result.append(
            {
                "profile_id": pid,
                "name": profiles[pid].name if pid in profiles else f"#{pid}",
                "color": profiles[pid].color if pid in profiles else "#888888",
                "samples": total,
                "accuracy": correct / total if total else None,
                "f1_macro": macro_f1(matrix),
                "matrix": matrix,
            }
        )
    return {"classes": CLASS_NAMES, "profiles": result}


def filter_options(session: Session) -> dict[str, Any]:
    """Values offered by the filter bar: profiles with versions, sources, cameras, sessions."""
    versions = session.execute(
        select(Analysis.profile_id, Analysis.profile_version, func.count(Analysis.id)).group_by(
            Analysis.profile_id, Analysis.profile_version
        )
    ).all()
    profile_rows = session.scalars(select(Profile).order_by(Profile.id)).all()
    by_profile: dict[int, list[dict[str, int]]] = {}
    for pid, version, count in versions:
        by_profile.setdefault(pid, []).append({"version": int(version), "count": int(count)})
    cameras = [
        name
        for (name,) in session.execute(
            select(Inspection.camera_name).where(Inspection.camera_name.is_not(None)).distinct()
        )
    ]
    sources = [s for (s,) in session.execute(select(Analysis.source).distinct())]
    sessions = session.execute(
        select(Analysis.session_id, func.count(Analysis.id), func.min(Analysis.created_at))
        .where(Analysis.session_id.is_not(None))
        .group_by(Analysis.session_id)
        .order_by(Analysis.session_id.desc())
        .limit(100)
    ).all()
    return {
        "profiles": [
            {
                "id": p.id,
                "name": p.name,
                "color": p.color,
                "deleted": p.deleted_at is not None,
                "versions": sorted(by_profile.get(p.id, []), key=lambda v: v["version"]),
            }
            for p in profile_rows
        ],
        "sources": sorted(s for s in sources if s),
        "cameras": sorted(c for c in cameras if c),
        "sessions": [
            {"id": sid, "count": int(n), "started_at": _iso(first)} for sid, n, first in sessions
        ],
        "classes": CLASS_NAMES,
    }
