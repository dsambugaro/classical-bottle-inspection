"""Figures produced by the pipeline: confusion matrices, ROIs, errors and the
comparison of feature vectors and classifiers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .config import CLASSES, LOCALIZATION_SIZE, SEED  # noqa: E402
from .dataset import Sample  # noqa: E402
from .extraction import BottleView  # noqa: E402
from .preprocessing import read_image  # noqa: E402

ROI_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "cap": (0, 0, 255),
    "label": (0, 200, 0),
    "body": (255, 0, 0),
}
"""BGR color used to outline each ROI."""


VECTOR_RASTER_DPI: int = 300
"""Minimum resolution of the images embedded in a vector (SVG or PDF) figure."""


def save_figure(fig: Figure, path: Path, dpi: int = 130) -> None:
    """Applies a tight layout, saves the figure to `path` and closes it.

    The format follows the suffix of `path`. In a vector format the images drawn with imshow are
    embedded at VECTOR_RASTER_DPI or more."""
    if path.suffix.lower() in (".svg", ".pdf"):
        dpi = max(dpi, VECTOR_RASTER_DPI)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to RGB so that matplotlib shows the right colors."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, normalize: bool, title: str
) -> None:
    """Saves a confusion matrix, optionally normalized so that each row sums to 1."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    if normalize:
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            value = matrix[row, col]
            text = f"{value:.2f}" if normalize else str(int(value))
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title + (" (normalized)" if normalize else ""))
    save_figure(fig, path)


def draw_bands(
    crop: np.ndarray, bands: dict[str, tuple[float, float]], colors: list[tuple[int, int, int]]
) -> np.ndarray:
    """Returns a copy of a bottle crop with one rectangle per band (y0, y1 inside the crop)."""
    image = crop.copy()
    height, width = image.shape[:2]
    for k, (y0, y1) in enumerate(bands.values()):
        top, bottom = int(y0 * height), min(int(y1 * height), height - 1)
        # Shift each rectangle inwards so overlapping bands stay distinguishable.
        cv2.rectangle(image, (1 + k * 2, top), (width - 2 - k * 2, bottom), colors[k], 2)
    return image


def plot_regions_per_class(
    samples: list[Sample],
    bottles: list[BottleView],
    bands: dict[str, tuple[float, float]],
    title: str,
    path: Path,
) -> None:
    """Saves one detected bottle per class with the given bands drawn on top."""
    n = len(CLASSES)
    colors = list(ROI_COLORS_BGR.values())
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 6), squeeze=False)
    for ax, class_name in zip(axes[0], CLASSES):
        index = next((i for i, s in enumerate(samples) if s.label == class_name), None)
        ax.axis("off")
        ax.set_title(class_name, fontsize=8)
        if index is not None:
            ax.imshow(bgr_to_rgb(draw_bands(bottles[index].crop, bands, colors)))
    fig.suptitle(title)
    save_figure(fig, path)


def plot_errors(
    samples: list[Sample],
    bottles: list[BottleView],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    limit: int = 16,
) -> bool:
    """Saves a grid of misclassified bottles; returns False (writing nothing) if there are none."""
    wrong = np.flatnonzero(np.asarray(y_true) != np.asarray(y_pred))[:limit]
    n = len(wrong)
    if n == 0:
        return False

    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), wrong):
        ax.imshow(bgr_to_rgb(bottles[index].crop))
        title = f"{samples[index].image_path.stem}\ntrue: {y_true[index]}\npred: {y_pred[index]}"
        ax.set_title(title, fontsize=6)
    save_figure(fig, path)
    return True


def plot_localization(
    samples: list[Sample],
    bottles: list[BottleView],
    background: np.ndarray,
    path: Path,
    n_examples: int = 6,
) -> None:
    """Saves the background model and example images with the detected bottle box."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(samples), size=n_examples, replace=False)
    width, height = LOCALIZATION_SIZE

    cols = (n_examples + 1) // 2 + 1
    fig, axes = plt.subplots(2, cols, figsize=(16, 6), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes:
        ax.axis("off")

    flat_axes[0].imshow(bgr_to_rgb(background) if background.ndim == 3 else background, cmap="gray")
    flat_axes[0].set_title("Background model (median)")

    for ax, index in zip(flat_axes[1:], chosen):
        image = cv2.resize(read_image(samples[index].image_path), (width, height))
        box = bottles[index].box
        top_left = (int(box.x0 * width), int(box.y0 * height))
        bottom_right = (int(box.x1 * width), int(box.y1 * height))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
        ax.imshow(bgr_to_rgb(image))
        ax.set_title(samples[index].label)
    save_figure(fig, path, dpi=110)


def annotate_bars(ax: Axes, bars: object, fmt: str) -> None:
    """Writes the height of each bar above it using the format string `fmt`."""
    for bar in bars:  # type: ignore[attr-defined]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_model_comparison(comparison: pd.DataFrame, path: Path, feature: str) -> None:
    """Saves quality (accuracy, macro F1) and speed (bottles per second) of each model
    trained on the feature vector `feature`."""
    models = list(comparison["model"])
    x = np.arange(len(models))
    fig, (ax_quality, ax_speed) = plt.subplots(1, 2, figsize=(11, 4))

    width = 0.38
    for offset, column in [(-width / 2, "accuracy"), (width / 2, "f1_macro")]:
        bars = ax_quality.bar(x + offset, comparison[column], width, label=column)
        annotate_bars(ax_quality, bars, "{:.3f}")
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(models, rotation=45, ha="right")
    ax_quality.set_ylim(0, 1.05)
    ax_quality.legend()

    bars = ax_speed.bar(x, comparison["bottles_per_second"])
    annotate_bars(ax_speed, bars, "{:.0f}")
    ax_speed.set_xticks(x)
    ax_speed.set_xticklabels(models, rotation=45, ha="right")
    ax_speed.set_ylabel("Bottles per second")

    fig.suptitle(f"Model comparison on {feature} (Kaggle validation split)")
    save_figure(fig, path)


def plot_feature_matrix(matrices: dict[str, pd.DataFrame], path: Path) -> None:
    """Saves one heatmap per metric: feature vectors in the rows, classifiers in the columns."""
    n_rows = max(len(matrix) for matrix in matrices.values())
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(4.6 * len(matrices), 1.5 + 0.42 * n_rows), squeeze=False
    )
    for ax, (metric, matrix) in zip(axes[0], matrices.items()):
        values = matrix.to_numpy(dtype=float)
        ax.imshow(
            values, cmap="Blues", vmin=min(0.5, float(np.nanmin(values))), vmax=1.0, aspect="auto"
        )
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = values[row, col]
                color = "white" if value > 0.8 else "black"
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
        ax.set_title(metric)
    fig.suptitle("Feature vector x classifier on the Kaggle validation split")
    save_figure(fig, path)
