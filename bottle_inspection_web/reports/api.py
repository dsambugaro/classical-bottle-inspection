"""Reports endpoints (§11, §14): summary, time series, Pareto, profiles, confidence, timings,
confusion, filter options and the CSV export."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..classes import CLASS_NAMES
from ..context import AppContext, get_ctx
from ..db.models import Analysis, Image, Inspection, Profile
from . import aggregations as agg
from .filters import AnalysisFilters, filters_dependency

router = APIRouter(prefix="/api/reports", tags=["reports"])

EXPORT_BATCH: int = 2000


@router.get("/options")
def options(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Values for the filter bar."""
    with ctx.db.session() as session:
        return agg.filter_options(session)


@router.get("/summary")
def summary(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """KPI tiles (with the previous period) and the class distribution."""
    with ctx.db.session() as session:
        return agg.summary(session, filters)


@router.get("/timeseries")
def timeseries(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Good vs defective and rejects by class over time."""
    with ctx.db.session() as session:
        return agg.timeseries(session, filters)


@router.get("/pareto")
def pareto(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Defect Pareto."""
    with ctx.db.session() as session:
        return agg.pareto(session, filters)


@router.get("/profiles")
def profiles(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Profiles x classes and pairwise agreement."""
    with ctx.db.session() as session:
        return agg.profiles_report(session, filters)


@router.get("/confidence")
def confidence(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Confidence histograms per profile."""
    with ctx.db.session() as session:
        return agg.confidence_report(session, filters)


@router.get("/timings")
def timings(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Mean time of each step per profile."""
    with ctx.db.session() as session:
        return agg.timings_report(session, filters)


@router.get("/confusion")
def confusion(
    filters: AnalysisFilters = Depends(filters_dependency), ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Accuracy, macro F1 and 7x7 confusion matrices (ground truth only)."""
    with ctx.db.session() as session:
        return agg.confusion_report(session, filters)


# ----------------------------------------------------------------- CSV


def number(value: float | int | None, lang: str, digits: int = 4) -> str:
    """Number with the locale decimal separator (no grouping)."""
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    text = f"{float(value):.{digits}f}"
    return text.replace(".", ",") if lang == "pt-BR" else text


def stamp(value: datetime, lang: str) -> str:
    """UTC timestamp in a spreadsheet-friendly local format."""
    return value.strftime("%d/%m/%Y %H:%M:%S" if lang == "pt-BR" else "%Y-%m-%d %H:%M:%S")


CSV_COLUMNS: list[str] = [
    "analysis_id",
    "created_at_utc",
    "inspection_id",
    "session_id",
    "source",
    "camera",
    "image",
    "profile_id",
    "profile_name",
    "profile_version",
    "algorithm",
    "model_ref",
    "modified_params",
    "predicted_class",
    "verdict",
    "confidence",
    "ground_truth",
    "total_ms",
    "warnings",
    "score_kind",
    *[f"score_{name}" for name in CLASS_NAMES],
]


def csv_rows(ctx: AppContext, filters: AnalysisFilters, lang: str) -> Iterator[str]:
    """CSV text in batches (UTF-8 BOM first, so spreadsheets detect the encoding)."""
    delimiter = ";" if lang == "pt-BR" else ","
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    yield "﻿" + buffer.getvalue()
    last_id = 0
    while True:
        with ctx.db.session() as session:
            rows = session.execute(
                select(Analysis, Profile, Inspection, Image)
                .join(Profile, Profile.id == Analysis.profile_id)
                .join(Inspection, Inspection.id == Analysis.inspection_id)
                .outerjoin(Image, Image.id == Inspection.image_id)
                .where(*filters.clauses(), Analysis.id > last_id)
                .order_by(Analysis.id)
                .limit(EXPORT_BATCH)
            ).all()
            if not rows:
                return
            buffer = io.StringIO()
            writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n")
            for analysis, profile, inspection, image in rows:
                scores = analysis.scores_json or {}
                writer.writerow(
                    [
                        analysis.id,
                        stamp(analysis.created_at, lang),
                        analysis.inspection_id,
                        analysis.session_id or "",
                        analysis.source,
                        inspection.camera_name or "",
                        (image.original_name if image else inspection.original_name) or "",
                        profile.id,
                        profile.name,
                        analysis.profile_version,
                        profile.algorithm,
                        profile.model_ref,
                        int(analysis.modified_params),
                        analysis.predicted_class,
                        analysis.verdict,
                        number(analysis.confidence, lang),
                        analysis.ground_truth_class or "",
                        number(analysis.total_ms, lang, 2),
                        " ".join(analysis.warnings_json or []),
                        analysis.score_kind,
                        *[number(scores.get(name), lang) for name in CLASS_NAMES],
                    ]
                )
                last_id = analysis.id
        yield buffer.getvalue()


@router.get("/export.csv")
def export_csv(
    filters: AnalysisFilters = Depends(filters_dependency),
    lang: Literal["en", "pt-BR"] = "en",
    ctx: AppContext = Depends(get_ctx),
) -> StreamingResponse:
    """One row per filtered analysis; the 7 score columns are always present (REQ-RP5)."""
    name = f"analyses-{datetime.now():%Y%m%d-%H%M%S}.csv"
    return StreamingResponse(
        csv_rows(ctx, filters, lang),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def register(app: FastAPI, ctx: AppContext) -> None:
    """Adds the reports routes."""
    app.include_router(router)
