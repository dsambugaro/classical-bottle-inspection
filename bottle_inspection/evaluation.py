"""Metrics computed from predictions: classification, pass/reject criteria,
production alerts and bottle localization error."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .config import (
    ALERT_THRESHOLD,
    ALERT_WINDOW,
    BOTTLE_CLASS_IDS,
    CLASSES,
    INSPECTION_CRITERIA,
    OK_CLASS,
)
from .dataset import Sample, read_yolo_boxes
from .extraction import BottleView

ZERO_DIVISION: Any = 0.0
"""Metric value when a class is never predicted (e.g. a collapsed model under
strong lighting changes). Typed as Any because scikit-learn infers this
parameter as str."""

REFERENCE_IMAGE_WIDTH: int = 1280
"""Original image width, used to report the center error in pixels."""


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro F1 and the F1 of each class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=ZERO_DIVISION)),
        "f1_per_class": {
            name: float(f1_score(y_true == name, y_pred == name, zero_division=ZERO_DIVISION))
            for name in CLASSES
        },
    }


def pass_reject_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of the binary decision "good bottle" versus "reject"."""
    return float(accuracy_score(y_true == OK_CLASS, y_pred == OK_CLASS))


def criterion_metrics(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, reject_classes: list[str]
) -> dict[str, Any]:
    """Binary metrics of one inspection criterion (reject = any class in `reject_classes`)."""
    true_reject = np.isin(y_true, reject_classes)
    pred_reject = np.isin(y_pred, reject_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_reject, pred_reject, average="binary", zero_division=ZERO_DIVISION
    )
    return {
        "criterion": name,
        "actual_rejects": int(true_reject.sum()),
        "accuracy": float(accuracy_score(true_reject, pred_reject)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_passes": int((true_reject & ~pred_reject).sum()),
        "false_rejects": int((~true_reject & pred_reject).sum()),
    }


def inspection_criteria_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Metrics of every inspection criterion plus the overall pass/reject decision."""
    criteria = dict(INSPECTION_CRITERIA)
    criteria["pass_vs_reject"] = [name for name in CLASSES if name != OK_CLASS]
    rows = [criterion_metrics(name, y_true, y_pred, classes) for name, classes in criteria.items()]
    return pd.DataFrame(rows)


def simulate_alerts(samples: list[Sample], y_pred: np.ndarray) -> pd.DataFrame:
    """Simulates the repeated-defect alert of Section 4.1 of the thesis proposal.

    The images are inspected in file order, as if they were a sequence of bottles.
    An alert is raised when the same defect appears at least ALERT_THRESHOLD
    times among the last ALERT_WINDOW inspections; it is re-armed once the defect
    falls below the threshold.
    """
    window: deque[str] = deque(maxlen=ALERT_WINDOW)
    active: set[str] = set()
    alerts = []
    for position, (sample, predicted) in enumerate(zip(samples, y_pred)):
        window.append(predicted)
        defect_counts = Counter(label for label in window if label != OK_CLASS)
        for defect, count in defect_counts.items():
            if count >= ALERT_THRESHOLD and defect not in active:
                active.add(defect)
                alerts.append(
                    {
                        "position": position,
                        "image": sample.image_path.name,
                        "defect": defect,
                        "occurrences_in_window": count,
                    }
                )
        active &= {defect for defect, count in defect_counts.items() if count >= ALERT_THRESHOLD}
    return pd.DataFrame(alerts, columns=["position", "image", "defect", "occurrences_in_window"])


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    intersection = max(width, 0.0) * max(height, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def localization_error(samples: list[Sample], bottles: list[BottleView]) -> dict[str, float]:
    """Compares the detected bottle box with the annotated bottle box of each image.

    The annotations are used only here, to validate the detection: the pipeline never
    reads them to find the bottle.
    """
    center_errors, edge_errors, ious = [], [], []
    for sample, bottle in zip(samples, bottles):
        for box in read_yolo_boxes(sample.label_path):
            if box.class_id not in BOTTLE_CLASS_IDS:
                continue
            annotated = (
                box.center_x - box.width / 2,
                box.center_y - box.height / 2,
                box.center_x + box.width / 2,
                box.center_y + box.height / 2,
            )
            detected = (bottle.box.x0, bottle.box.y0, bottle.box.x1, bottle.box.y1)
            center_errors.append(abs(bottle.center_x - box.center_x))
            edge_errors.append(np.abs(np.subtract(detected, annotated)))
            ious.append(box_iou(detected, annotated))
    centers, edges, iou = np.array(center_errors), np.array(edge_errors), np.array(ious)
    result = {
        "mean_error_relative": float(centers.mean()),
        "p95_error_relative": float(np.percentile(centers, 95)),
        "max_error_relative": float(centers.max()),
        "mean_error_px": float(centers.mean() * REFERENCE_IMAGE_WIDTH),
        "p95_error_px": float(np.percentile(centers, 95) * REFERENCE_IMAGE_WIDTH),
        "mean_iou": float(iou.mean()),
        "p5_iou": float(np.percentile(iou, 5)),
        "min_iou": float(iou.min()),
    }
    for k, edge in enumerate(("left", "top", "right", "bottom")):
        result[f"mean_{edge}_error_relative"] = float(edges[:, k].mean())
        result[f"p95_{edge}_error_relative"] = float(np.percentile(edges[:, k], 95))
    return result
