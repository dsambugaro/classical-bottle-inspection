"""Experiment: sensitivity of the bottle localization to its parameters.

The localization has seven parameters (closing and opening of the mask, smoothing and
fraction of the column profile, the width of the bottle box and the fractions of the row
profile for the top and for the bottom edge). Each one is varied alone,
the others staying at their reference (config.py) values, and the detected box is compared
with the annotated bottle box of every TRAINING image: overlap (IoU), error of the horizontal
centre and of each edge. The annotations are used only for this check; the pipeline never
reads them, and the validation images are not used here.

Outputs, in `<results>/parameter_search/`:

    localization_sensitivity.csv   one row per (parameter, value)
    localization_otsu.json         Otsu threshold of the training images (min, median, max)

Usage (from the `code` folder)::

    .venv/bin/python experiments/localization_sensitivity.py --data ../archive --results results
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bottle_inspection import config  # noqa: E402
from bottle_inspection.env import default_data_dir  # noqa: E402
from bottle_inspection.dataset import load_dataset, read_yolo_boxes, split_indices  # noqa: E402
from bottle_inspection.evaluation import REFERENCE_IMAGE_WIDTH  # noqa: E402
from bottle_inspection.params import InspectionParams  # noqa: E402
from bottle_inspection.preprocessing import (  # noqa: E402
    background_difference,
    build_background,
    foreground_mask,
    locate_bottle,
    normalize_lighting,
    otsu_threshold,
    read_image,
    reference_patch_mean,
)

REFERENCE_IMAGE_HEIGHT: int = 720
"""Original image height, used to report the top and bottom errors in pixels."""

VALUES: dict[str, list[Any]] = {
    "close_size": [1, 5, 9, 15, 21, 31],
    "open_size": [1, 3, 5, 9, 13],
    "profile_smoothing": [1, 5, 11, 21, 31, 41, 51],
    "column_fraction": [0.2, 0.3, 0.4, 0.5, 0.6],
    "bottle_width": [0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16],
    "top_fraction": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "row_fraction": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}
"""Values tested for each field of LocalizationParams (the reference value is among them)."""

Box = tuple[float, float, float, float]
"""(x0, y0, x1, y1) in relative image coordinates."""


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def load_training(data_dir: Path) -> tuple[list[np.ndarray], list[Box], np.ndarray]:
    """Normalized training images at the localization size, their annotated bottle boxes
    and the background."""
    samples = load_dataset(data_dir)
    train = split_indices(samples, "train")
    background = build_background([samples[i].image_path for i in train])
    reference = reference_patch_mean(background)
    size = InspectionParams.defaults().localization.size
    images, boxes = [], []
    for i in train:
        annotated = [
            b
            for b in read_yolo_boxes(samples[i].label_path)
            if b.class_id in config.BOTTLE_CLASS_IDS
        ]
        if not annotated:
            continue
        box = annotated[0]
        boxes.append(
            (
                box.center_x - box.width / 2,
                box.center_y - box.height / 2,
                box.center_x + box.width / 2,
                box.center_y + box.height / 2,
            )
        )
        image = normalize_lighting(read_image(samples[i].image_path), reference)
        images.append(cv2.resize(image, size))
    return images, boxes, background


def measure(
    images: list[np.ndarray], boxes: list[Box], background: np.ndarray, params: InspectionParams
) -> dict[str, float]:
    """Localization quality of one parameter set over the training images."""
    ious, centers, edges = [], [], []
    for image, annotated in zip(images, boxes):
        box = locate_bottle(foreground_mask(image, background, params=params), params=params)
        detected = (box.x0, box.y0, box.x1, box.y1)
        ious.append(iou(detected, annotated))
        centers.append(abs(box.center_x - (annotated[0] + annotated[2]) / 2))
        edges.append(np.subtract(detected, annotated))
    signed = np.array(edges)
    width, height = REFERENCE_IMAGE_WIDTH, REFERENCE_IMAGE_HEIGHT
    return {
        "mean_iou": float(np.mean(ious)),
        "p5_iou": float(np.percentile(ious, 5)),
        "min_iou": float(np.min(ious)),
        "mean_center_error_px": float(np.mean(centers) * width),
        "p95_center_error_px": float(np.percentile(centers, 95) * width),
        # Positive = the box is inside the bottle on that side (it cuts the bottle).
        "mean_left_inset_px": float(signed[:, 0].mean() * width),
        "mean_right_inset_px": float(-signed[:, 2].mean() * width),
        "mean_top_inset_px": float(signed[:, 1].mean() * height),
        "mean_bottom_inset_px": float(-signed[:, 3].mean() * height),
        "max_top_inset_px": float(signed[:, 1].max() * height),
        "top_cut_over_15px": int((signed[:, 1] * height > 15).sum()),
        "top_past_over_15px": int((signed[:, 1] * height < -15).sum()),
        "bottom_past_over_15px": int((signed[:, 3] * height > 15).sum()),
    }


def main() -> None:
    """Varies each localization parameter alone and writes the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_dir(),
        help="dataset root (default: BOTTLE_DATA, see .env.example)",
    )
    parser.add_argument("--results", type=Path, default=Path("results"), help="results folder")
    args = parser.parse_args()
    folder = args.results / "parameter_search"
    folder.mkdir(parents=True, exist_ok=True)

    print("== Loading the training images")
    images, boxes, background = load_training(args.data)
    base = InspectionParams.defaults()
    thresholds = [otsu_threshold(background_difference(image, background)) for image in images]
    otsu = {
        "training_images": len(images),
        "min": float(np.min(thresholds)),
        "median": float(np.median(thresholds)),
        "max": float(np.max(thresholds)),
        "noise_floor": config.NOISE_FLOOR,
    }
    (folder / "localization_otsu.json").write_text(json.dumps(otsu, indent=2))

    rows = []
    for name, values in VALUES.items():
        reference = getattr(base.localization, name)
        for value in values:
            localization = dataclasses.replace(base.localization, **{name: value})
            params = dataclasses.replace(base, localization=localization)
            row = {"parameter": name, "value": value, "is_reference": value == reference}
            rows.append({**row, **measure(images, boxes, background, params)})
            print(f"   {name} = {value}: IoU {rows[-1]['mean_iou']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "localization_sensitivity.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))
    print(f"\nResults in {folder.resolve()}")


if __name__ == "__main__":
    main()
