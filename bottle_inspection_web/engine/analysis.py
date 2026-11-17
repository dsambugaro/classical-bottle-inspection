"""Analysis of one image by one profile, with per-step timings (§10).

Every step calls the functions of `bottle_inspection.preprocessing` / `.features` with the
profile's InspectionParams, in the same order as `preprocessing.prepare_image` (normalize ->
foreground mask -> bottle box -> region crops) and `features.extract_group`; they are called one by
one only so each can be timed. Only the feature groups of the model's own vector are computed. A
test checks that the resulting feature vector is bitwise equal to `extraction.extract_image`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from bottle_inspection.config import WHOLE_BOTTLE
from bottle_inspection.features import extract_group, split_group
from bottle_inspection.params import InspectionParams
from bottle_inspection.preprocessing import (
    PreparedImage,
    crop_regions,
    foreground_mask,
    locate_bottle,
    normalize_lighting,
    reference_patch_mean,
)

from ..models.discovery import LoadedModel
from .scores import ScoreResult, compute_scores

LOW_FOREGROUND_FRACTION: float = 0.04
"""Below this fraction of the mask that is foreground the bottle is probably absent."""

ASPECT_TOLERANCE: float = 0.02
"""Relative aspect-ratio difference tolerated before `resolution_mismatch`."""

_thread_limits: Any = None


def configure_threads() -> None:
    """Limits BLAS to one thread per call in the web process (NF1).

    Each analysis is a single small matrix product; with one BLAS thread per call the
    profiles run in parallel in the thread pool instead of fighting over 12 BLAS threads
    (measured: KNN p95 85 -> 45 ms alone, 105 -> 78 ms with 3 profiles). The CLI is not
    affected: this is only called by the web app.
    """
    global _thread_limits
    if _thread_limits is None:
        from threadpoolctl import threadpool_limits

        _thread_limits = threadpool_limits(limits=1, user_api="blas")


@dataclass
class ProfileRuntime:
    """What the engine needs to run one profile (built from the DB profile + model)."""

    profile_id: int
    name: str
    version: int
    algorithm: str
    model_ref: str
    status: str
    color: str
    params: InspectionParams
    background: np.ndarray
    model: LoadedModel
    verdict: dict[str, Any]
    modified: bool
    reference_size: tuple[int, int] | None = None
    """(width, height) of the images the background corresponds to, when known."""


@dataclass
class AnalysisOutput:
    """Result of one analysis, before it is persisted."""

    predicted_class: str
    verdict: str
    confidence: float
    score: ScoreResult
    geometry: dict[str, Any]
    timings_ms: dict[str, Any]
    total_ms: float
    warnings: list[str]
    features: np.ndarray = field(repr=False)


class StepTimer:
    """Measures consecutive steps with time.perf_counter()."""

    def __init__(self) -> None:
        """Starts the clock."""
        self.start = time.perf_counter()
        self.last = self.start

    def lap(self) -> float:
        """Milliseconds since the previous lap."""
        now = time.perf_counter()
        elapsed = (now - self.last) * 1000.0
        self.last = now
        return round(elapsed, 3)

    def total(self) -> float:
        """Milliseconds since the start."""
        return round((time.perf_counter() - self.start) * 1000.0, 3)


def prepare_timed(
    image: np.ndarray, background: np.ndarray, params: InspectionParams, timer: StepTimer
) -> tuple[PreparedImage, dict[str, float]]:
    """preprocessing.prepare_image, step by step, with timings."""
    timings: dict[str, float] = {}
    if params.normalization.enabled:
        reference = reference_patch_mean(background, params=params)
        image = normalize_lighting(image, reference, params=params)
    timings["normalize"] = timer.lap()
    small = cv2.resize(image, params.localization.size)
    mask = foreground_mask(small, background, params=params)
    timings["foreground_mask"] = timer.lap()
    box = locate_bottle(mask, params=params)
    timings["localize"] = timer.lap()
    rois = crop_regions(image, box, params=params)
    timings["crop_regions"] = timer.lap()
    return PreparedImage(image, mask, box, rois), timings


def features_timed(
    prepared: PreparedImage, params: InspectionParams, timer: StepTimer
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """features.extract_group for each group of the model's vector, with timings."""
    values, timings = {}, {}
    for group in params.features.groups:
        values[group] = extract_group(prepared, group, params=params)
        timings[group] = timer.lap()
    return values, timings


def foreground_fraction(mask: np.ndarray) -> float:
    """Fraction of the whole foreground mask that is bottle (0 when the mask is empty).

    The whole mask is used, not the area of the detected box: the box is measured on the mask
    itself, so a tiny speck would still fill its own box and never look "absent"."""
    return float(mask.mean()) if mask.size else 0.0


def model_regions(params: InspectionParams) -> list[str]:
    """Regions the model's feature groups use, in region order, without the whole bottle
    (it is the bottle box itself)."""
    used = {split_group(group)[0] for group in params.features.groups}
    return [name for name in params.region_bands() if name in used and name != WHOLE_BOTTLE]


def geometry_of(prepared: PreparedImage, params: InspectionParams) -> dict[str, Any]:
    """Relative geometry (0..1) of the bottle, regions and reference patch (REQ-R3).

    `rois` holds the rectangle of each region of the model's feature groups (the ad hoc ROIs
    and the automatic regions); a band (0..1 inside the box) becomes image coordinates here.
    """
    box = prepared.box
    rnd = lambda v: round(float(v), 5)  # noqa: E731
    bands = params.region_bands()
    return {
        "bottle_center_x": rnd(box.center_x),
        "bottle_top_y": rnd(prepared.top_y),
        "bottle_box": [rnd(box.x0), rnd(box.y0), rnd(box.x1), rnd(box.y1)],
        "rois": {
            name: [
                rnd(box.x0),
                rnd(box.y0 + bands[name][0] * box.height),
                rnd(box.x1),
                rnd(box.y0 + bands[name][1] * box.height),
            ]
            for name in model_regions(params)
        },
        "reference_patch": [rnd(v) for v in params.normalization.reference_patch],
        "foreground_fraction": rnd(foreground_fraction(prepared.foreground_mask)),
    }


def decide_verdict(predicted: str, confidence: float, verdict: dict[str, Any]) -> str:
    """pass / reject / review from the class and the optional confidence rule."""
    result = "reject" if predicted in verdict.get("reject_classes", []) else "pass"
    min_confidence = verdict.get("min_confidence")
    if min_confidence is not None and confidence < float(min_confidence):
        return str(verdict.get("low_confidence_verdict", "review"))
    return result


def collect_warnings(
    runtime: ProfileRuntime,
    image_shape: tuple[int, ...],
    geometry: dict[str, Any],
    confidence: float,
) -> list[str]:
    """Warning codes of REQ-R5 (non-blocking)."""
    warnings: list[str] = []
    fraction = geometry["foreground_fraction"]
    if fraction < LOW_FOREGROUND_FRACTION:
        warnings.append("low_foreground")
    height, width = image_shape[:2]
    bg_height, bg_width = runtime.background.shape[:2]
    aspect, bg_aspect = width / height, bg_width / bg_height
    mismatch = abs(aspect - bg_aspect) / bg_aspect > ASPECT_TOLERANCE
    if runtime.reference_size is not None and (width, height) != runtime.reference_size:
        mismatch = True
    if mismatch:
        warnings.append("resolution_mismatch")
    if runtime.modified:
        warnings.append("modified_parameters")
    min_confidence = runtime.verdict.get("min_confidence")
    if min_confidence is not None and confidence < float(min_confidence):
        warnings.append("low_confidence")
    return warnings


def analyse(image: np.ndarray, runtime: ProfileRuntime, decode_ms: float = 0.0) -> AnalysisOutput:
    """Runs the full pipeline of one profile on one decoded BGR image."""
    params = runtime.params
    timer = StepTimer()
    prepared, prep_timings = prepare_timed(image, runtime.background, params, timer)
    feature_set, feature_timings = features_timed(prepared, params, timer)
    x = np.concatenate([feature_set[group] for group in params.features.groups])
    model = runtime.model
    with model.lock:
        predicted = str(model.pipeline.predict(x.reshape(1, -1))[0])
        score = compute_scores(model, x, predicted)
    predict_ms = timer.lap()
    geometry = geometry_of(prepared, params)
    confidence = round(score.confidence, 5)
    total = round(timer.total() + decode_ms, 3)
    timings: dict[str, Any] = {
        "decode": round(decode_ms, 3),
        **prep_timings,
        "features": feature_timings,
        "predict": predict_ms,
        "total": total,
    }
    return AnalysisOutput(
        predicted_class=predicted,
        verdict=decide_verdict(predicted, confidence, runtime.verdict),
        confidence=confidence,
        score=score,
        geometry=geometry,
        timings_ms=timings,
        total_ms=total,
        warnings=collect_warnings(runtime, image.shape, geometry, confidence),
        features=x,
    )


def decode_image(data: bytes) -> tuple[np.ndarray, float]:
    """Decodes image bytes with OpenCV (validates the type). Returns (image, ms)."""
    start = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("not a decodable image")
    return image, round((time.perf_counter() - start) * 1000.0, 3)
