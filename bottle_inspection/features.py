"""Feature descriptors: turn a preprocessed bottle into fixed-length vectors.

Three generic descriptors are computed on every region of the detected bottle: the ad
hoc ROIs (cap, label, body), the automatic regions (auto_1..auto_3, once learned),
the whole bottle, a wide fixed area of the image around it and the whole image. Each
(region, descriptor) pair is a feature group named `<region>_<descriptor>`:

==========  ============================================================
descriptor  content
==========  ============================================================
hog         histograms of gradient orientations (shape / edges)
lbp         histograms of uniform local binary patterns per cell (texture)
color       HSV histograms + mean/std of each channel
==========  ============================================================

Sizes with the default configuration (the automatic regions depend on the learned bands):

=======  =====  ====  =====
region   hog    lbp   color
=======  =====  ====  =====
cap       840    312     74
label     360    624     74
body      528    936     74
bottle   2760   3744     74
wide     5040   6240     74
image    9720  11648     74
=======  =====  ====  =====

The groups are never all concatenated: each classifier is trained on one descriptor at
a time, over the ROIs, over the automatic regions or over the whole bottle (see
config.FEATURE_VECTORS). Nothing here depends on the color or shape of a specific product.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from .config import DESCRIPTORS, REGIONS
from .params import InspectionParams, resolve
from .preprocessing import PreparedImage

FeatureSet = dict[str, np.ndarray]
"""Feature groups of one image, keyed by the names in config.FEATURE_GROUPS."""

HSV_CHANNEL_RANGES: tuple[int, int, int] = (180, 256, 256)
"""Upper bound of the H, S and V channels in OpenCV."""

HOG_BLOCK: tuple[int, int] = (2, 2)
"""HOG block size, in cells."""

LBP_METHOD: str = "uniform"
"""Rotation-invariant uniform patterns: `points + 2` different codes."""


def canonical_gray(bgr: np.ndarray, region: str, values: InspectionParams) -> np.ndarray:
    """Grayscale version of a region crop, resized to the canonical size of the region."""
    return cv2.cvtColor(cv2.resize(bgr, values.region_sizes()[region]), cv2.COLOR_BGR2GRAY)


def hsv_color_features(bgr: np.ndarray, *, params: InspectionParams | None = None) -> np.ndarray:
    """Normalized H, S and V histograms plus the mean and std of each channel."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    parts = []
    bins_per_channel = resolve(params).color.hsv_bins
    for channel, (bins, upper) in enumerate(zip(bins_per_channel, HSV_CHANNEL_RANGES)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], [0, upper]).ravel()
        parts.append(histogram / (histogram.sum() + 1e-9))
    pixels = hsv.reshape(-1, 3)
    parts.append(pixels.mean(axis=0) / 255.0)
    parts.append(pixels.std(axis=0) / 255.0)
    return np.concatenate(parts)


def hog_features(
    bgr: np.ndarray, region: str, *, params: InspectionParams | None = None
) -> np.ndarray:
    """HOG of a region resized to its canonical size (gradient-orientation histograms)."""
    values = resolve(params)
    return hog(
        canonical_gray(bgr, region, values),
        orientations=values.hog.orientations,
        pixels_per_cell=values.hog.cells()[region],
        cells_per_block=values.hog.block,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def lbp_features(
    bgr: np.ndarray, region: str, *, params: InspectionParams | None = None
) -> np.ndarray:
    """LBP of a region resized to its canonical size: one code histogram per cell.

    Each pixel gets a code that says how its neighbors compare with it (brighter or
    darker), which describes the local texture regardless of the overall brightness.
    The region is split into cells and the normalized histogram of the codes of each
    cell is kept, so the descriptor also tells where each texture is.
    """
    values = resolve(params)
    lbp = values.lbp
    gray = canonical_gray(bgr, region, values)
    codes = local_binary_pattern(gray, lbp.points, lbp.radius, method=LBP_METHOD).astype(np.int64)
    n_codes = lbp.points + 2
    cell_width, cell_height = lbp.cells()[region]
    rows, cols = gray.shape[0] // cell_height, gray.shape[1] // cell_width
    cells = (
        codes[: rows * cell_height, : cols * cell_width]
        .reshape(rows, cell_height, cols, cell_width)
        .transpose(0, 2, 1, 3)
        .reshape(rows * cols, -1)
    )
    # Offsetting the codes of each cell lets a single bincount build every histogram.
    offsets = np.arange(rows * cols)[:, None] * n_codes
    histograms = np.bincount((cells + offsets).ravel(), minlength=rows * cols * n_codes)
    return histograms / (cell_width * cell_height)


Descriptor = Callable[[np.ndarray, str, InspectionParams], np.ndarray]
"""Computes one descriptor of a region: (BGR crop, region name, params) -> vector."""

DESCRIPTOR_FUNCTIONS: dict[str, Descriptor] = {
    "hog": lambda bgr, region, values: hog_features(bgr, region, params=values),
    "lbp": lambda bgr, region, values: lbp_features(bgr, region, params=values),
    "color": lambda bgr, _region, values: hsv_color_features(bgr, params=values),
}
"""The function of each descriptor in config.DESCRIPTORS."""

assert list(DESCRIPTOR_FUNCTIONS) == DESCRIPTORS


def split_group(group: str) -> tuple[str, str]:
    """Splits a feature group name into (region, descriptor), e.g. "cap_hog" -> ("cap", "hog")."""
    region, _, descriptor = group.rpartition("_")
    if region not in REGIONS or descriptor not in DESCRIPTOR_FUNCTIONS:
        raise ValueError(f"unknown feature group: {group}")
    return region, descriptor


def extract_group(
    prepared: PreparedImage, group: str, *, params: InspectionParams | None = None
) -> np.ndarray:
    """Computes one feature group (one descriptor of one region) of a preprocessed image."""
    region, descriptor = split_group(group)
    return DESCRIPTOR_FUNCTIONS[descriptor](prepared.rois[region], region, resolve(params))


def extract_features(
    prepared: PreparedImage, *, params: InspectionParams | None = None
) -> FeatureSet:
    """Computes every available feature group of one preprocessed image.

    The groups of the automatic regions are included only when `params` holds their bands.
    """
    values = resolve(params)
    return {
        group: extract_group(prepared, group, params=values) for group in values.feature_groups()
    }


def build_matrix(feature_sets: list[FeatureSet], groups: list[str]) -> np.ndarray:
    """Stacks the chosen feature groups of every image into an (n_images, n_features) matrix."""
    return np.stack([np.concatenate([features[g] for g in groups]) for features in feature_sets])
