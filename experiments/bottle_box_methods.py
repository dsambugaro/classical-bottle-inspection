"""Experiment: three ways of taking the bottle box from the same foreground mask.

The mask is always the same (background difference, Otsu threshold, closing and opening).
What changes is how the box is taken from it:

automatic   every edge from the profiles: the columns of the run around the peak of the
            column profile, and the rows that reach a fraction of the densest row
fixed       a box of ad hoc width and height, centred on the run and resting on the
            bottom row found in the mask
mixed       ad hoc width centred on the bottle found in the column profile; top and bottom
            from the row profile (the method of preprocessing.locate_bottle)

Every method is tried with several values of its parameters. The boxes are compared with
the annotated bottle boxes of the TRAINING images (mean, 5th percentile and minimum IoU and
the error of the horizontal centre); the method and the values are chosen there. The
validation images are measured only for the best setting of each method, as a check.

Outputs, in `<results>/parameter_search/`:

    bottle_box_methods.csv        one row per (method, parameter values), training images
    bottle_box_methods_best.csv   best setting of each method, on training and validation

Usage (from the `code` folder)::

    .venv/bin/python experiments/bottle_box_methods.py --data ../archive --results results
"""

from __future__ import annotations

import argparse
import itertools
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
    build_background,
    foreground_mask,
    normalize_lighting,
    peak_run,
    read_image,
    reference_patch_mean,
)

SMOOTHINGS: list[int] = [11, 21, 31, 41, 51]
COLUMN_FRACTIONS: list[float] = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
TOP_FRACTIONS: list[float] = [0.2, 0.3, 0.4, 0.5]
BOTTOM_FRACTIONS: list[float] = [0.4, 0.5, 0.6, 0.7]
WIDTHS: list[float] = [0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16]
HEIGHTS: list[float] = [0.73, 0.75, 0.77, 0.79, 0.81]
"""Values tried: profile smoothing (columns), fractions of the profiles, and the ad hoc
width and height of the box as fractions of the image width and height."""

Image = dict[str, Any]
"""One image: its foreground mask and its annotated bottle box (x0, y0, x1, y1)."""


def load_split(data_dir: Path) -> dict[str, list[Image]]:
    """Foreground mask and annotated bottle box of every training and validation image."""
    samples = load_dataset(data_dir)
    train = split_indices(samples, "train")
    background = build_background([samples[i].image_path for i in train])
    reference = reference_patch_mean(background)
    size = InspectionParams.defaults().localization.size
    images: dict[str, list[Image]] = {"train": [], "val": []}
    for sample in samples:
        boxes = [
            b for b in read_yolo_boxes(sample.label_path) if b.class_id in config.BOTTLE_CLASS_IDS
        ]
        if not boxes:
            continue
        box = boxes[0]
        annotated = np.array(
            [
                box.center_x - box.width / 2,
                box.center_y - box.height / 2,
                box.center_x + box.width / 2,
                box.center_y + box.height / 2,
            ]
        )
        image = cv2.resize(normalize_lighting(read_image(sample.image_path), reference), size)
        images[sample.split].append({"mask": foreground_mask(image, background), "box": annotated})
    return images


def column_run(mask: np.ndarray, smoothing: int, fraction: float) -> tuple[int, int, float]:
    """Run of columns around the peak of the smoothed profile and the centre of the bottle
    in it (as preprocessing.bottle_columns, with the values given)."""
    counts = mask.sum(axis=0)
    columns = np.convolve(counts, np.ones(smoothing) / smoothing, mode="same")
    x0, x1 = peak_run(columns, fraction)
    inside = counts[x0:x1]
    reached = np.flatnonzero(inside >= inside.max() * fraction)
    return x0, x1, x0 + (int(reached[0]) + int(reached[-1]) + 1) / 2


def rows_of(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Foreground count of each row inside the columns of the bottle."""
    return mask[:, x0:x1].sum(axis=1)


def row_edges(rows: np.ndarray, top: float, bottom: float) -> tuple[int, int]:
    """First row that reaches `top` and last row that reaches `bottom` of the densest row."""
    peak = rows.max()
    return (
        int(np.flatnonzero(rows >= peak * top)[0]),
        int(np.flatnonzero(rows >= peak * bottom)[-1]) + 1,
    )


def quality(boxes: np.ndarray, images: list[Image]) -> dict[str, float]:
    """Overlap and centre error of the detected boxes against the annotated ones."""
    annotated = np.array([image["box"] for image in images])
    width = np.clip(
        np.minimum(boxes[:, 2], annotated[:, 2]) - np.maximum(boxes[:, 0], annotated[:, 0]), 0, None
    )
    height = np.clip(
        np.minimum(boxes[:, 3], annotated[:, 3]) - np.maximum(boxes[:, 1], annotated[:, 1]), 0, None
    )
    inter = width * height
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    iou = inter / (area(boxes) + area(annotated) - inter)
    centre = np.abs((boxes[:, 0] + boxes[:, 2]) / 2 - (annotated[:, 0] + annotated[:, 2]) / 2)
    return {
        "mean_iou": float(iou.mean()),
        "p5_iou": float(np.percentile(iou, 5)),
        "min_iou": float(iou.min()),
        "boxes_below_iou_0.7": int((iou < 0.7).sum()),
        "mean_center_error_px": float(centre.mean() * REFERENCE_IMAGE_WIDTH),
        "p95_center_error_px": float(np.percentile(centre, 95) * REFERENCE_IMAGE_WIDTH),
    }


def detect(images: list[Image], method: str, values: dict[str, float]) -> np.ndarray:
    """Boxes (relative x0, y0, x1, y1) of one method with one set of parameter values."""
    boxes = []
    for image in images:
        mask = image["mask"]
        mask_height, mask_width = mask.shape
        x0, x1, centre = column_run(mask, int(values["smoothing"]), values["column_fraction"])
        top, bottom = row_edges(
            rows_of(mask, x0, x1), values["top_fraction"], values["bottom_fraction"]
        )
        left, right = x0 / mask_width, x1 / mask_width
        if method != "automatic":
            left = centre / mask_width - values["width"] / 2
            right = left + values["width"]
        y1 = bottom / mask_height
        y0 = y1 - values["height"] if method == "fixed" else top / mask_height
        boxes.append([left, y0, right, y1])
    return np.array(boxes)


def settings() -> list[tuple[str, dict[str, float]]]:
    """Every (method, parameter values) tried."""
    grid: list[tuple[str, dict[str, float]]] = []
    for smoothing, column, top, bottom in itertools.product(
        SMOOTHINGS, COLUMN_FRACTIONS, TOP_FRACTIONS, BOTTOM_FRACTIONS
    ):
        base = {
            "smoothing": smoothing,
            "column_fraction": column,
            "top_fraction": top,
            "bottom_fraction": bottom,
        }
        grid.append(("automatic", base))
        if column in (0.3, 0.4, 0.5) and top in (0.3, 0.4):
            grid.extend(("mixed", {**base, "width": width}) for width in WIDTHS)
        if column in (0.3, 0.4, 0.5) and top == 0.3:
            grid.extend(
                ("fixed", {**base, "width": width, "height": height})
                for width, height in itertools.product(WIDTHS, HEIGHTS)
            )
    return grid


def main() -> None:
    """Measures every setting on the training images and the best ones on validation."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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

    print("== Building the masks of the training and validation images", flush=True)
    images = load_split(args.data)
    rows = []
    for method, values in settings():
        rows.append(
            {
                "method": method,
                **values,
                **quality(detect(images["train"], method, values), images["train"]),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    table.to_csv(folder / "bottle_box_methods.csv", index=False, float_format="%.4f")

    best = []
    for method in ("automatic", "fixed", "mixed"):
        chosen = table[table["method"] == method].iloc[0]
        values = {
            k: float(chosen[k])
            for k in (
                "smoothing",
                "column_fraction",
                "top_fraction",
                "bottom_fraction",
                "width",
                "height",
            )
            if pd.notna(chosen[k])
        }
        for split in ("train", "val"):
            best.append(
                {
                    "method": method,
                    "split": split,
                    **values,
                    **quality(detect(images[split], method, values), images[split]),
                }
            )
    used = InspectionParams.defaults().localization
    values = {
        "smoothing": used.profile_smoothing,
        "column_fraction": used.column_fraction,
        "top_fraction": used.top_fraction,
        "bottom_fraction": used.row_fraction,
        "width": used.bottle_width,
    }
    for split in ("train", "val"):
        best.append(
            {
                "method": "mixed (config.py)",
                "split": split,
                **values,
                **quality(detect(images[split], "mixed", values), images[split]),
            }
        )
    summary = pd.DataFrame(best)
    summary.to_csv(folder / "bottle_box_methods_best.csv", index=False, float_format="%.4f")
    print(summary.round(4).to_string(index=False))
    print(f"\nResults in {folder.resolve()}")


if __name__ == "__main__":
    main()
