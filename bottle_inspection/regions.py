"""Automatic regions: splits the bottle into bands without hand-picked positions.

The ad hoc ROIs (config.ROIS) say where the cap, the label and the body are. The
automatic alternative learns the bands from the training images, with no labels:

1. every training bottle is resized to BOTTLE_SIZE and they are averaged, which
   gives the typical appearance of a bottle;
2. each strip of AUTO_ROI_UNIT rows of that mean bottle is described by its colors;
3. the strips are split into AUTO_ROI_COUNT contiguous bands so that the strips of
   a band are as alike as possible (least sum of squared distances to the band mean).

Only the number of bands is given; where they start and end comes from the data.
"""

from __future__ import annotations

import numpy as np

from .config import AUTO_ROI_COUNT, AUTO_ROI_UNIT

MIN_BAND_STRIPS: int = 2
"""Smallest band, in strips: HOG needs at least 2x2 cells to form a block."""

Range = tuple[float, float]
"""(y0, y1) of a band inside the bottle box (0 = top, 1 = bottom)."""


def mean_bottle(crops: list[np.ndarray]) -> np.ndarray:
    """Per-pixel mean of bottle crops of the same size (float32, BGR)."""
    total = np.zeros(crops[0].shape, dtype=np.float64)
    for crop in crops:
        total += crop
    return (total / len(crops)).astype(np.float32)


def strip_descriptors(image: np.ndarray, unit: int = AUTO_ROI_UNIT) -> np.ndarray:
    """One row per horizontal strip of `unit` rows: its mean colors, standardized.

    Each strip is described by the mean of every column and channel over its rows;
    every value is then standardized over the strips, so no channel dominates.
    """
    n_strips = image.shape[0] // unit
    strips = image[: n_strips * unit].reshape(n_strips, unit, -1).mean(axis=1)
    return (strips - strips.mean(axis=0)) / (strips.std(axis=0) + 1e-9)


def contiguous_segments(descriptors: np.ndarray, n_segments: int, min_size: int = 1) -> list[int]:
    """Best split of consecutive items into `n_segments` contiguous groups of at least
    `min_size` items.

    Minimizes the sum of squared distances of each item to the mean of its group
    (the k-means criterion, restricted to contiguous groups) by dynamic programming,
    so the result is the exact optimum. Returns the index where each group after
    the first starts.
    """
    n_items = len(descriptors)
    if n_segments < 1 or n_segments * min_size > n_items:
        raise ValueError(f"cannot split {n_items} items into {n_segments} segments")
    sums = np.vstack([np.zeros(descriptors.shape[1]), np.cumsum(descriptors, axis=0)])
    squares = np.concatenate([[0.0], np.cumsum((descriptors**2).sum(axis=1))])

    def cost(start: int, end: int) -> float:
        """Sum of squared distances to the mean of items start..end-1."""
        total = sums[end] - sums[start]
        return float(squares[end] - squares[start] - total @ total / (end - start))

    best = np.full((n_segments + 1, n_items + 1), np.inf)
    previous = np.zeros((n_segments + 1, n_items + 1), dtype=int)
    best[0, 0] = 0.0
    for segment in range(1, n_segments + 1):
        for end in range(segment * min_size, n_items + 1):
            for start in range((segment - 1) * min_size, end - min_size + 1):
                value = best[segment - 1, start] + cost(start, end)
                if value < best[segment, end]:
                    best[segment, end], previous[segment, end] = value, start
    starts, end = [], n_items
    for segment in range(n_segments, 0, -1):
        end = int(previous[segment, end])
        starts.append(end)
    return starts[::-1][1:]


def learn_auto_bands(
    crops: list[np.ndarray], n_bands: int = AUTO_ROI_COUNT, unit: int = AUTO_ROI_UNIT
) -> tuple[Range, ...]:
    """Learns the bands of the automatic regions from BOTTLE_SIZE crops of training bottles."""
    descriptors = strip_descriptors(mean_bottle(crops), unit)
    n_strips = len(descriptors)
    edges = [0, *contiguous_segments(descriptors, n_bands, MIN_BAND_STRIPS), n_strips]
    return tuple((start / n_strips, end / n_strips) for start, end in zip(edges, edges[1:]))
