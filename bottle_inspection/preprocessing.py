"""Preprocessing: background model, lighting normalization, bottle detection and crops.

The camera is fixed, so the per-pixel median of many images is the empty scene
(background). The bottle is whatever differs from it: the position and the height of its
box are measured on the foreground mask; only the width of the box is given.

Every function takes an optional keyword argument `params`. When it is None (the CLI),
the constants of config.py are used; the web interface passes the values of a profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import N_BACKGROUND_IMAGES, NOISE_FLOOR, SEED, WHOLE_BOTTLE, WHOLE_IMAGE, WIDE_AREA
from .params import InspectionParams, resolve

MAX_DIFFERENCE: int = 255
"""The BGR difference to the background is clipped to this value before Otsu's method."""


@dataclass(frozen=True)
class BottleBox:
    """Bounding box of the detected bottle, in relative image coordinates (0..1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_x(self) -> float:
        """Relative horizontal center of the bottle."""
        return (self.x0 + self.x1) / 2

    @property
    def width(self) -> float:
        """Relative width of the bottle."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Relative height of the bottle."""
        return self.y1 - self.y0


@dataclass
class PreparedImage:
    """An image after preprocessing, ready for feature extraction."""

    image: np.ndarray
    """Full-resolution BGR image (lighting-normalized when enabled)."""
    foreground_mask: np.ndarray
    """Boolean bottle mask at LOCALIZATION_SIZE."""
    box: BottleBox
    """Detected bottle box."""
    rois: dict[str, np.ndarray]
    """BGR crop of each region (ad hoc ROIs, automatic regions when learned, whole bottle)."""

    @property
    def center_x(self) -> float:
        """Relative horizontal center of the bottle."""
        return self.box.center_x

    @property
    def top_y(self) -> float:
        """Relative height of the top of the bottle."""
        return self.box.y0


def read_image(path: Path) -> np.ndarray:
    """Reads an image from disk as a BGR array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_background(
    image_paths: list[Path], *, params: InspectionParams | None = None
) -> np.ndarray:
    """Per-pixel median of N_BACKGROUND_IMAGES random images, at LOCALIZATION_SIZE."""
    size = resolve(params).localization.size
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(image_paths), min(N_BACKGROUND_IMAGES, len(image_paths)), replace=False)
    stack = [cv2.resize(read_image(image_paths[i]), size) for i in chosen]
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def reference_patch_mean(
    image: np.ndarray, *, params: InspectionParams | None = None
) -> np.ndarray:
    """Mean BGR color of the REFERENCE_PATCH wall region."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = resolve(params).normalization.reference_patch
    patch = image[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
    return patch.reshape(-1, 3).mean(axis=0)


def normalize_lighting(
    image: np.ndarray, reference_color: np.ndarray, *, params: InspectionParams | None = None
) -> np.ndarray:
    """Multiplies each channel so the reference patch gets `reference_color`.

    Compensates global gain/brightness and white balance changes.
    """
    gain = reference_color / (reference_patch_mean(image, params=params) + 1e-6)
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def background_difference(small_image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """|image - background| summed over B, G and R, for each pixel (0..765)."""
    return np.abs(small_image.astype(np.int16) - background.astype(np.int16)).sum(axis=2)


def otsu_threshold(difference: np.ndarray) -> float:
    """Threshold that best splits the differences into two groups (Otsu's method).

    It is computed for each image, so no fixed threshold has to be chosen.
    """
    clipped = np.clip(difference, 0, MAX_DIFFERENCE).astype(np.uint8)
    threshold, _ = cv2.threshold(clipped, 0, MAX_DIFFERENCE, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def foreground_mask(
    small_image: np.ndarray, background: np.ndarray, *, params: InspectionParams | None = None
) -> np.ndarray:
    """Marks as bottle the pixels that differ from the background.

    The difference is thresholded with Otsu's method (never below NOISE_FLOOR, so an
    empty scene gives an empty mask); a morphological closing fills the holes of the
    transparent bottle and an opening removes thin lines and specks.
    """
    localization = resolve(params).localization
    difference = background_difference(small_image, background)
    mask = (difference > max(otsu_threshold(difference), NOISE_FLOOR)).astype(np.uint8)
    for operation, size in (
        (cv2.MORPH_CLOSE, localization.close_size),
        (cv2.MORPH_OPEN, localization.open_size),
    ):
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, operation, kernel)
    return mask.astype(bool)


def peak_run(profile: np.ndarray, fraction: float) -> tuple[int, int]:
    """[start, end) of the contiguous run around the peak that stays >= fraction * peak."""
    threshold = profile.max() * fraction
    start = end = int(np.argmax(profile))
    while start > 0 and profile[start - 1] >= threshold:
        start -= 1
    while end < len(profile) - 1 and profile[end + 1] >= threshold:
        end += 1
    return start, end + 1


def bottle_columns(
    mask: np.ndarray, *, params: InspectionParams | None = None
) -> tuple[int, int, float]:
    """Finds the bottle in the column profile of the mask: (run start, run end, centre).

    The smoothed count of foreground pixels per column peaks at the bottle, and the run of
    columns around the peak locates it (the smoothing keeps a mask split in two halves in
    one run). The centre is the middle of the first and the last column of the run that
    reach the same fraction of the unsmoothed profile, which the smoothing does not shift.
    """
    localization = resolve(params).localization
    counts = mask.sum(axis=0)
    smoothing = max(localization.profile_smoothing, 1)
    columns = np.convolve(counts, np.ones(smoothing) / smoothing, mode="same")
    x0, x1 = peak_run(columns, localization.column_fraction)
    inside = counts[x0:x1]
    reached = np.flatnonzero(inside >= inside.max() * localization.column_fraction)
    return x0, x1, x0 + (int(reached[0]) + int(reached[-1]) + 1) / 2


def locate_bottle(mask: np.ndarray, *, params: InspectionParams | None = None) -> BottleBox:
    """Measures the bottle box on the foreground mask.

    Columns: the bottle is found in the column profile (bottle_columns) and the box has the
    fixed width BOTTLE_WIDTH, centred on it. Rows: inside the columns of the bottle, the box
    goes from the first row that reaches the top fraction of the densest row to the last
    row that reaches the row fraction (the top is narrower than the body and the bottom
    has the shadow, so each edge has its own fraction). The position and the height are
    measured; only the width is given. An empty mask gives the whole image.
    """
    localization = resolve(params).localization
    height, width = mask.shape
    if not mask.any():
        return BottleBox(0.0, 0.0, 1.0, 1.0)
    x0, x1, centre = bottle_columns(mask, params=params)
    rows = mask[:, x0:x1].sum(axis=1)
    y0 = int(np.flatnonzero(rows >= rows.max() * localization.top_fraction)[0])
    y1 = int(np.flatnonzero(rows >= rows.max() * localization.row_fraction)[-1]) + 1
    half = localization.bottle_width / 2
    left = min(max(centre / width - half, 0.0), 1.0 - 2 * half)
    return BottleBox(left, y0 / height, left + 2 * half, y1 / height)


def crop_band(image: np.ndarray, box: BottleBox, band: tuple[float, float]) -> np.ndarray:
    """Crops the columns of the bottle box between two heights of the box.

    `band` is (y0, y1) inside the box: 0 is the top of the bottle and 1 its bottom.
    """
    height, width = image.shape[:2]
    x0, x1 = int(box.x0 * width), max(int(box.x1 * width), int(box.x0 * width) + 1)
    y0 = int((box.y0 + band[0] * box.height) * height)
    y1 = max(int((box.y0 + band[1] * box.height) * height), y0 + 1)
    return image[y0:y1, x0:x1]


def crop_regions(
    image: np.ndarray, box: BottleBox, *, params: InspectionParams | None = None
) -> dict[str, np.ndarray]:
    """BGR crop of every available region of the detected bottle.

    The ad hoc ROIs and the whole bottle are cropped at full resolution. The automatic
    regions are slices of the bottle resized to BOTTLE_SIZE, which is where their
    boundaries were learned. The wide area is a fixed rectangle of the image (WIDE_AREA_RECT),
    the same in every image; the whole image is the image itself.
    """
    values = resolve(params)
    crops = {name: crop_band(image, box, band) for name, band in values.rois.bands().items()}
    bottle = crop_band(image, box, (0.0, 1.0))
    canonical = cv2.resize(bottle, values.bottle.crop_size)
    rows = canonical.shape[0]
    for name, (y0, y1) in values.auto_bands().items():
        crops[name] = canonical[round(y0 * rows) : round(y1 * rows)]
    crops[WHOLE_BOTTLE] = bottle
    height, width = image.shape[:2]
    x0, y0, x1, y1 = values.wide.rect
    crops[WIDE_AREA] = image[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
    crops[WHOLE_IMAGE] = image
    return crops


def prepare_image(
    image: np.ndarray,
    background: np.ndarray,
    normalize: bool,
    *,
    params: InspectionParams | None = None,
) -> PreparedImage:
    """Runs the whole preprocessing chain on one image.

    Steps: lighting normalization -> foreground mask -> bottle box -> region crops.
    """
    values = resolve(params)
    if normalize:
        image = normalize_lighting(
            image, reference_patch_mean(background, params=values), params=values
        )
    mask = foreground_mask(cv2.resize(image, values.localization.size), background, params=values)
    box = locate_bottle(mask, params=values)
    return PreparedImage(
        image=image,
        foreground_mask=mask,
        box=box,
        rois=crop_regions(image, box, params=values),
    )
