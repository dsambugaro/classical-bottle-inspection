"""Step-by-step trace of the pipeline on one image (debug mode).

`preprocessing_trace` exposes every intermediate result of `preprocessing.prepare_image`
(normalization, background difference, Otsu threshold, mask morphology, column/row profiles,
bottle box, region crops) and `profile_trace` every feature group of a profile's vector plus
the classifier decision. The final values always come from the pipeline functions themselves
(`foreground_mask`, `locate_bottle`, `crop_regions`, `analyse`); the intermediate ones are
recomputed here with the same primitives only to be shown, and a test checks they agree.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from bottle_inspection.config import AUTO_ROIS, NOISE_FLOOR, WHOLE_BOTTLE
from bottle_inspection.features import LBP_METHOD, canonical_gray, extract_group, split_group
from bottle_inspection.hog_view import render_hog
from bottle_inspection.params import InspectionParams
from bottle_inspection.preprocessing import (
    MAX_DIFFERENCE,
    PreparedImage,
    background_difference,
    crop_regions,
    foreground_mask,
    bottle_columns,
    locate_bottle,
    normalize_lighting,
    otsu_threshold,
    reference_patch_mean,
)

from .analysis import AnalysisOutput, ProfileRuntime, analyse

HISTOGRAM_BINS: int = 64
"""Bins of the histogram of background differences (clipped to MAX_DIFFERENCE)."""

PIXEL_PATCH: int = 4
"""Side of the square of pixels whose values are listed (the image as a matrix)."""

VIEW_SCALE: int = 4
"""Integer zoom of the small descriptor images (canonical regions are 48..96 px wide)."""

OUTLIER_Z: float = 3.0
"""|standardized value| above which a feature is far from what the model saw in training."""

LBP_NON_UNIFORM_COLOR: tuple[int, int, int] = (127, 127, 127)
"""BGR colour of the non-uniform LBP code (the last one)."""

GOLDEN_RATIO_CONJUGATE: float = 0.618033988749895
"""Hue step (fraction of the colour wheel) that spreads consecutive codes far apart."""

PREPROCESSING_GROUPS: tuple[str, ...] = (
    "localization",
    "normalization",
    "bottle",
    "rois",
    "auto_rois",
)
"""Parameter groups that decide the preprocessing result (everything before the descriptors)."""


# =============================================================================
# Encoding
# =============================================================================


def _data_uri(extension: str, mime: str, image: np.ndarray, options: list[int]) -> str:
    """An image as a base64 data URI."""
    ok, buffer = cv2.imencode(extension, image, options)
    if not ok:
        raise ValueError("image could not be encoded")
    return f"data:{mime};base64,{base64.b64encode(buffer.tobytes()).decode()}"


def png(image: np.ndarray) -> str:
    """Lossless data URI (masks, maps and small crops). Boolean masks become black/white."""
    data = image.astype(np.uint8) * 255 if image.dtype == bool else image
    return _data_uri(".png", "image/png", data, [])


def jpeg(image: np.ndarray) -> str:
    """JPEG data URI (photographs)."""
    return _data_uri(".jpg", "image/jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def enlarge(image: np.ndarray, scale: int = VIEW_SCALE) -> np.ndarray:
    """Integer zoom without interpolation, so each source pixel stays a sharp square."""
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def to_gray_image(values: np.ndarray) -> np.ndarray:
    """Any non-negative array scaled to 0..255 (its maximum becomes white)."""
    top = float(values.max()) if values.size else 0.0
    return (
        np.zeros(values.shape, np.uint8) if top <= 0 else (values * (255.0 / top)).astype(np.uint8)
    )


def rounded(values: np.ndarray, digits: int = 5) -> list[float]:
    """A vector as a JSON list."""
    return [round(float(v), digits) for v in np.asarray(values).ravel()]


def pixel_patch(image: np.ndarray) -> dict[str, Any]:
    """BGR values of the PIXEL_PATCH x PIXEL_PATCH square at the centre of the image."""
    height, width = image.shape[:2]
    x, y = width // 2, height // 2
    return {"x": x, "y": y, "values": image[y : y + PIXEL_PATCH, x : x + PIXEL_PATCH].tolist()}


def stats(values: np.ndarray) -> dict[str, float]:
    """Summary of a vector."""
    return {
        "min": round(float(values.min()), 5),
        "max": round(float(values.max()), 5),
        "mean": round(float(values.mean()), 5),
        "std": round(float(values.std()), 5),
    }


# =============================================================================
# Preprocessing
# =============================================================================


def preprocessing_key(params: InspectionParams, background: np.ndarray) -> str:
    """Identifies a preprocessing configuration: two profiles with the same key give the very
    same normalized image, mask, bottle box and region crops."""
    groups = {name: params.to_dict()[name] for name in PREPROCESSING_GROUPS}
    digest = hashlib.sha1(repr(sorted(groups.items())).encode())
    digest.update(np.ascontiguousarray(background).tobytes())
    return digest.hexdigest()[:16]


def region_kind(name: str) -> str:
    """adhoc (cap/label/body), auto (learned regions) or bottle (the whole box)."""
    if name == WHOLE_BOTTLE:
        return "bottle"
    return "auto" if name in AUTO_ROIS else "adhoc"


def mask_steps(
    small: np.ndarray, background: np.ndarray, params: InspectionParams
) -> dict[str, Any]:
    """The stages of `preprocessing.foreground_mask`: difference, threshold, closing, opening."""
    localization = params.localization
    difference = background_difference(small, background)
    otsu = otsu_threshold(difference)
    threshold = max(otsu, NOISE_FLOOR)
    raw = (difference > threshold).astype(np.uint8)
    stages: list[np.ndarray] = [raw]
    for operation, size in (
        (cv2.MORPH_CLOSE, localization.close_size),
        (cv2.MORPH_OPEN, localization.open_size),
    ):
        current = stages[-1]
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            current = np.asarray(cv2.morphologyEx(current, operation, kernel))
        stages.append(current)
    return {
        "difference": difference,
        "otsu": otsu,
        "threshold": threshold,
        "raw": stages[0].astype(bool),
        "closed": stages[1].astype(bool),
        "opened": stages[2].astype(bool),
    }


def profile_steps(mask: np.ndarray, params: InspectionParams) -> dict[str, Any]:
    """The column and row profiles `preprocessing.locate_bottle` measures the box on."""
    localization = params.localization
    smoothing = max(localization.profile_smoothing, 1)
    columns_raw = mask.sum(axis=0)
    columns = np.convolve(columns_raw, np.ones(smoothing) / smoothing, mode="same")
    if mask.any():
        x0, x1, centre = bottle_columns(mask, params=params)
    else:
        x0, x1, centre = 0, mask.shape[1], mask.shape[1] / 2
    rows = mask[:, x0:x1].sum(axis=1)
    return {
        "columns_raw": columns_raw.astype(int).tolist(),
        "columns": [round(float(v), 2) for v in columns],
        "column_threshold": round(float(columns.max() * localization.column_fraction), 2),
        "column_peak": int(np.argmax(columns)),
        "rows": rows.astype(int).tolist(),
        "top_threshold": round(float(rows.max() * localization.top_fraction), 2),
        "row_threshold": round(float(rows.max() * localization.row_fraction), 2),
        "x": [int(x0), int(x1)],
        "centre": round(float(centre), 1),
    }


def preprocessing_trace(
    image: np.ndarray, background: np.ndarray, params: InspectionParams
) -> dict[str, Any]:
    """Every intermediate result of the preprocessing of one image, JSON-serializable."""
    localization = params.localization
    height, width = image.shape[:2]
    reference = reference_patch_mean(background, params=params)
    measured = reference_patch_mean(image, params=params)
    enabled = params.normalization.enabled
    normalized = normalize_lighting(image, reference, params=params) if enabled else image

    small = cv2.resize(normalized, localization.size)
    steps = mask_steps(small, background, params)
    mask = foreground_mask(small, background, params=params)
    box = locate_bottle(mask, params=params)
    profiles = profile_steps(mask, params)
    mask_height, mask_width = mask.shape
    difference = steps["difference"]
    clipped = np.clip(difference, 0, MAX_DIFFERENCE).astype(np.uint8)
    histogram = np.bincount(
        clipped.ravel() // ((MAX_DIFFERENCE + 1) // HISTOGRAM_BINS), minlength=HISTOGRAM_BINS
    )

    crops = crop_regions(normalized, box, params=params)
    bands = params.region_bands()
    regions = [
        {
            "name": name,
            "kind": region_kind(name),
            "band": [round(float(v), 5) for v in bands[name]],
            "rect": [
                round(box.x0, 5),
                round(box.y0 + bands[name][0] * box.height, 5),
                round(box.x1, 5),
                round(box.y0 + bands[name][1] * box.height, 5),
            ],
            "shape": list(crop.shape),
            "canonical_size": list(params.region_sizes()[name]),
            "image": png(crop),
        }
        for name, crop in crops.items()
    ]
    return {
        "key": preprocessing_key(params, background),
        "params": {
            key: value
            for key, value in params.to_flat_dict().items()
            if key.split(".")[0] in ("localization", "normalization")
        },
        "original": {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "patch": pixel_patch(image),
        },
        "background": {"image": png(background), "shape": list(background.shape)},
        "normalization": {
            "enabled": enabled,
            "reference_patch": list(params.normalization.reference_patch),
            "patch_mean_background": rounded(reference, 2),
            "patch_mean_image": rounded(measured, 2),
            "gain": rounded(reference / (measured + 1e-6), 4) if enabled else [1.0, 1.0, 1.0],
            "image": jpeg(normalized),
            "patch": pixel_patch(normalized),
        },
        "difference": {
            "size": list(localization.size),
            "small": jpeg(small),
            "heatmap": png(cv2.applyColorMap(clipped, cv2.COLORMAP_MAGMA)),
            "max": int(difference.max()),
            "clip": MAX_DIFFERENCE,
            "histogram": histogram.astype(int).tolist(),
            "otsu": steps["otsu"],
            "noise_floor": NOISE_FLOOR,
            "threshold": steps["threshold"],
        },
        "mask": {
            "raw": png(steps["raw"]),
            "closed": png(steps["closed"]),
            "opened": png(mask),
            "close_size": localization.close_size,
            "open_size": localization.open_size,
            "fraction_raw": round(float(steps["raw"].mean()), 5),
            "fraction_closed": round(float(steps["closed"].mean()), 5),
            "fraction": round(float(mask.mean()), 5),
        },
        "localization": {
            **profiles,
            "size": [mask_width, mask_height],
            "smoothing": localization.profile_smoothing,
            "column_fraction": localization.column_fraction,
            "bottle_width": localization.bottle_width,
            "box_x": [round(box.x0 * mask_width, 1), round(box.x1 * mask_width, 1)],
            "top_fraction": localization.top_fraction,
            "row_fraction": localization.row_fraction,
            "y": [round(box.y0 * mask_height), round(box.y1 * mask_height)],
            "box": [round(v, 5) for v in (box.x0, box.y0, box.x1, box.y1)],
            "box_px": [
                int(box.x0 * width),
                int(box.y0 * height),
                int(box.x1 * width),
                int(box.y1 * height),
            ],
            "empty": not bool(mask.any()),
        },
        "regions": regions,
    }


# =============================================================================
# Features and classification of one profile
# =============================================================================


def lbp_palette(n_codes: int) -> np.ndarray:
    """BGR colour of each LBP code: golden-ratio hues with alternating saturation and value.

    The last code (non-uniform) is gray, so every code gets a colour that differs from the others.
    """
    hsv = np.zeros((n_codes, 1, 3), np.uint8)
    for k in range(n_codes):
        hsv[k, 0] = (
            int((k * GOLDEN_RATIO_CONJUGATE % 1.0) * 180),
            255 if k % 2 == 0 else 150,
            255 if k % 3 == 0 else 200 if k % 3 == 1 else 140,
        )
    palette = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n_codes, 3)
    palette[-1] = LBP_NON_UNIFORM_COLOR
    return palette


def hog_view(gray: np.ndarray, region: str, params: InspectionParams) -> dict[str, Any]:
    """Gradient strength and the HOG drawing (high resolution) of a canonical gray region."""
    cell = params.hog.cells()[region]
    block = params.hog.block
    source = gray.astype(np.float32)
    strength = np.hypot(cv2.Sobel(source, cv2.CV_32F, 1, 0), cv2.Sobel(source, cv2.CV_32F, 0, 1))
    cells = [gray.shape[1] // cell[0], gray.shape[0] // cell[1]]
    return {
        "cell": list(cell),
        "cells": cells,
        "blocks": [cells[0] - block[0] + 1, cells[1] - block[1] + 1],
        "block": list(block),
        "orientations": params.hog.orientations,
        "gradient": png(enlarge(cv2.applyColorMap(to_gray_image(strength), cv2.COLORMAP_INFERNO))),
        "rendering": png(render_hog(gray, params.hog.orientations, cell)),
    }


def lbp_view(gray: np.ndarray, region: str, params: InspectionParams) -> dict[str, Any]:
    """The LBP code of every pixel (one colour per code) with the cell grid drawn on it."""
    lbp = params.lbp
    codes = local_binary_pattern(gray, lbp.points, lbp.radius, method=LBP_METHOD).astype(np.int64)
    n_codes = lbp.points + 2
    palette = lbp_palette(n_codes)
    colored = enlarge(palette[codes])
    cell_width, cell_height = lbp.cells()[region]
    for x in range(0, gray.shape[1] + 1, cell_width):
        cv2.line(
            colored, (x * VIEW_SCALE, 0), (x * VIEW_SCALE, colored.shape[0]), (255, 255, 255), 1
        )
    for y in range(0, gray.shape[0] + 1, cell_height):
        cv2.line(
            colored, (0, y * VIEW_SCALE), (colored.shape[1], y * VIEW_SCALE), (255, 255, 255), 1
        )
    return {
        "cell": [cell_width, cell_height],
        "cells": [gray.shape[1] // cell_width, gray.shape[0] // cell_height],
        "points": lbp.points,
        "radius": lbp.radius,
        "n_codes": n_codes,
        "codes": png(colored),
        "palette": [f"#{r:02x}{g:02x}{b:02x}" for b, g, r in palette.tolist()],
        "histogram": rounded(np.bincount(codes.ravel(), minlength=n_codes) / codes.size, 4),
    }


def color_view(crop: np.ndarray, params: InspectionParams) -> dict[str, Any]:
    """The H, S and V channels of a region (the histograms are the feature values themselves)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = np.stack(
        [hsv[..., 0], np.full_like(hsv[..., 0], 255), np.full_like(hsv[..., 0], 255)], axis=2
    )
    return {
        "bins": list(params.color.hsv_bins),
        "h": png(cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)),
        "s": png(hsv[..., 1]),
        "v": png(hsv[..., 2]),
    }


def group_trace(
    prepared: PreparedImage, group: str, values: np.ndarray, offset: int, params: InspectionParams
) -> dict[str, Any]:
    """One feature group: the region it is computed on, how the descriptor sees it and its values."""
    region, descriptor = split_group(group)
    crop = prepared.rois[region]
    trace: dict[str, Any] = {
        "name": group,
        "region": region,
        "descriptor": descriptor,
        "offset": offset,
        "length": int(values.size),
        "stats": stats(values),
        "values": rounded(values),
        "crop": png(crop),
        "crop_shape": list(crop.shape),
    }
    if descriptor == "color":
        trace["color"] = color_view(crop, params)
        return trace
    gray = canonical_gray(crop, region, params)
    trace["canonical"] = png(enlarge(gray))
    trace["canonical_size"] = [int(gray.shape[1]), int(gray.shape[0])]
    trace[descriptor] = (hog_view if descriptor == "hog" else lbp_view)(gray, region, params)
    return trace


def scaled_trace(runtime: ProfileRuntime, x: np.ndarray) -> dict[str, Any] | None:
    """The feature vector after the model's StandardScaler (what SVM and KNN really compare).

    Values far from zero mean the image differs from the training images in that feature.
    """
    scaler = getattr(runtime.model.pipeline, "named_steps", {}).get("scaler")
    if scaler is None:
        return None
    z = np.asarray(scaler.transform(x.reshape(1, -1)))[0]
    magnitude = np.abs(z)
    return {
        "values": rounded(z, 4),
        "mean_abs": round(float(magnitude.mean()), 4),
        "max_abs": round(float(magnitude.max()), 4),
        "outliers": int((magnitude > OUTLIER_Z).sum()),
        "outlier_z": OUTLIER_Z,
    }


def analysis_json(
    output: AnalysisOutput, runtime: ProfileRuntime, image: np.ndarray, truth: str | None, url: str
) -> dict[str, Any]:
    """The AnalysisResult contract of an analysis that is not stored (ids are 0)."""
    finite = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
    return {
        "analysis_id": 0,
        "inspection_id": 0,
        "profile": {
            "id": runtime.profile_id,
            "name": runtime.name,
            "version": runtime.version,
            "algorithm": runtime.algorithm,
            "model_ref": runtime.model_ref,
            "status": runtime.status,
            "color": runtime.color,
        },
        "image": {
            "id": None,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "url": url,
            "thumb_url": None,
        },
        "predicted_class": output.predicted_class,
        "verdict": output.verdict,
        "confidence": output.confidence,
        "scores": {name: finite(value) for name, value in output.score.scores.items()},
        "score_kind": output.score.score_kind,
        "details": output.score.details,
        "geometry": output.geometry,
        "timings_ms": output.timings_ms,
        "ground_truth": truth,
        "warnings": output.warnings,
        "source": "debug",
        "session_id": None,
        "created_at": None,
    }


def profile_trace(
    image: np.ndarray, runtime: ProfileRuntime, truth: str | None = None, url: str = ""
) -> dict[str, Any]:
    """Every feature group of the profile's vector and the decision of its classifier.

    The vector and the decision are those of `analyse` (the function the inspections use); the
    crops and descriptor images are recomputed from the same preprocessing to be shown.
    """
    params = runtime.params
    output = analyse(image, runtime)
    normalized = image
    if params.normalization.enabled:
        normalized = normalize_lighting(
            image, reference_patch_mean(runtime.background, params=params), params=params
        )
    mask = foreground_mask(
        cv2.resize(normalized, params.localization.size), runtime.background, params=params
    )
    box = locate_bottle(mask, params=params)
    prepared = PreparedImage(normalized, mask, box, crop_regions(normalized, box, params=params))

    groups, offset = [], 0
    for group in params.features.groups:
        length = int(extract_group(prepared, group, params=params).size)
        groups.append(
            group_trace(prepared, group, output.features[offset : offset + length], offset, params)
        )
        offset += length
    return {
        "preprocessing_key": preprocessing_key(params, runtime.background),
        "feature_vector": runtime.model.info.feature_vector,
        "groups": groups,
        "vector": {"length": int(output.features.size), "stats": stats(output.features)},
        "scaled": scaled_trace(runtime, output.features),
        "analysis": analysis_json(output, runtime, image, truth, url),
    }
