"""High-resolution drawing of a HOG descriptor, for figures.

`skimage.feature.hog(visualize=True)` draws its picture at the resolution of the region
itself (e.g. 64x48 pixels), so the strokes of each cell are a handful of pixels and blur
when the figure is enlarged. `render_hog` draws the same picture on a canvas many times
larger, with anti-aliased strokes: one stroke per orientation bin in every cell, along the
edge direction, as bright as the gradient strength accumulated in that bin.
"""

from __future__ import annotations

import cv2
import numpy as np

TARGET_SIDE: int = 1600
"""The longer side of the drawing has at least this many pixels."""

MIN_SCALE: int = 8
"""Smallest zoom factor (canvas pixels per region pixel)."""

SUBPIXEL_BITS: int = 4
"""Fractional bits of the stroke end points (cv2.line `shift`), for smooth strokes."""


def cell_histograms(gray: np.ndarray, orientations: int, cell: tuple[int, int]) -> np.ndarray:
    """Orientation histogram of every cell, shape (rows, cols, orientations).

    The same values `skimage.feature.hog` draws: gradients by central differences (zero on
    the borders), unsigned orientation in [0, 180) split into `orientations` equal bins, and
    in each cell the gradient magnitudes of a bin added up and divided by the cell area.
    """
    image = gray.astype(np.float64)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    magnitude = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    bins = np.minimum((angle / (180.0 / orientations)).astype(int), orientations - 1)

    cell_width, cell_height = cell
    rows, cols = gray.shape[0] // cell_height, gray.shape[1] // cell_width
    histograms = np.zeros((rows, cols, orientations))
    for row in range(rows):
        for col in range(cols):
            window = (
                slice(row * cell_height, (row + 1) * cell_height),
                slice(col * cell_width, (col + 1) * cell_width),
            )
            histograms[row, col] = np.bincount(
                bins[window].ravel(), weights=magnitude[window].ravel(), minlength=orientations
            )
    return histograms / (cell_width * cell_height)


def render_scale(gray: np.ndarray) -> int:
    """Zoom factor that makes the longer side of the drawing at least TARGET_SIDE pixels."""
    return max(MIN_SCALE, int(np.ceil(TARGET_SIDE / max(gray.shape))))


def render_hog(
    gray: np.ndarray, orientations: int, cell: tuple[int, int], scale: int | None = None
) -> np.ndarray:
    """Draws the HOG of a gray region as a uint8 image `scale` times larger than the region.

    Each cell shows one stroke per orientation bin, centred on the cell and drawn along the
    edge (perpendicular to the gradient direction of the bin). The brightest stroke of the
    whole region is white; weaker ones are drawn first, so strong edges stay on top.
    """
    scale = scale if scale is not None else render_scale(gray)
    histograms = cell_histograms(gray, orientations, cell)
    rows, cols, _ = histograms.shape
    cell_width, cell_height = cell
    canvas = np.zeros((gray.shape[0] * scale, gray.shape[1] * scale), np.uint8)
    peak = float(histograms.max())
    if peak <= 0:
        return canvas

    radius = (min(cell_width, cell_height) / 2 - 0.5) * scale
    thickness = max(1, round(min(cell_width, cell_height) * scale / 28))
    midpoints = np.pi * (np.arange(orientations) + 0.5) / orientations
    fixed = 1 << SUBPIXEL_BITS
    strokes = []
    for row in range(rows):
        for col in range(cols):
            center_x = (col + 0.5) * cell_width * scale
            center_y = (row + 0.5) * cell_height * scale
            for k, angle in enumerate(midpoints):
                value = histograms[row, col, k] / peak
                if value > 0:
                    # The stroke lies along the edge: perpendicular to the gradient angle.
                    dx, dy = radius * np.sin(angle), -radius * np.cos(angle)
                    strokes.append(
                        (value, center_x - dx, center_y - dy, center_x + dx, center_y + dy)
                    )
    for value, x0, y0, x1, y1 in sorted(strokes):
        cv2.line(
            canvas,
            (round(x0 * fixed), round(y0 * fixed)),
            (round(x1 * fixed), round(y1 * fixed)),
            int(round(255 * value)),
            thickness,
            cv2.LINE_AA,
            SUBPIXEL_BITS,
        )
    return canvas
