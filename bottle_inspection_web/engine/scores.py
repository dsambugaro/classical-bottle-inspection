"""Per-class scores and confidence of each algorithm (REQ-R1, REQ-R2).

The values are the real decision values of the trained models, computed with the same
functions as the didactic figures (bottle_inspection.didactic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from sklearn.pipeline import Pipeline

from bottle_inspection.didactic import (
    KNN_NEIGHBORS_SHOWN,
    feature_names,
    forest_decisions,
    knn_decisions,
    svm_decisions,
)

from ..classes import CLASS_NAMES, ordered_scores
from ..models.discovery import LoadedModel

CONFIDENCE_HELP: dict[str, str] = {
    "svm": "(s_win - s_runner_up) / max(1, |s_win|), clipped to [0, 1]",
    "knn": "1 - d_win / d_runner_up (class distances)",
    "random_forest": "maximum averaged tree probability",
}
"""Per-algorithm confidence definitions (also translated in the UI)."""


@dataclass
class ScoreResult:
    """Scores of the 7 classes plus the algorithm-specific details."""

    scores: dict[str, float]
    score_kind: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


class _Identity:
    """Scaler stand-in that returns its input (the rows are already standardized)."""

    @staticmethod
    def transform(X: np.ndarray) -> np.ndarray:
        """Returns X unchanged."""
        return X


class _PrescaledKnn:
    """Pipeline view with an identity scaler, for didactic.knn_decisions."""

    def __init__(self, knn: Any) -> None:
        """Wraps the fitted KNeighborsClassifier."""
        self.named_steps = {"scaler": _Identity(), "knn": knn}


def svm_confidence(scores: dict[str, float], winner: str) -> float:
    """(s_win - s_runner_up) / max(1, |s_win|), clipped to [0, 1]."""
    runner_up = max(value for name, value in scores.items() if name != winner)
    s_win = scores[winner]
    return float(np.clip((s_win - runner_up) / max(1.0, abs(s_win)), 0.0, 1.0))


def knn_confidence(distances: dict[str, float]) -> float:
    """1 - d_win / d_runner_up, where d are the smallest per-class distances."""
    ordered = sorted(distances.values())
    if len(ordered) < 2 or ordered[1] <= 0:
        return 0.0
    return float(np.clip(1.0 - ordered[0] / ordered[1], 0.0, 1.0))


def forest_confidence(probabilities: dict[str, float]) -> float:
    """The maximum probability."""
    return float(max(probabilities.values()))


def compute_scores(model: LoadedModel, x: np.ndarray, predicted: str) -> ScoreResult:
    """Scores for one feature vector `x` (shape (n_features,))."""
    X = x.reshape(1, -1)
    steps = getattr(model.pipeline, "named_steps", {})
    algorithm = model.info.algorithm
    if "svm" in steps and "scaler" in steps:
        decision = svm_decisions(model.pipeline, X)[0]
        scores = ordered_scores(decision.scores)
        return ScoreResult(
            scores=scores,
            score_kind="svm_ovr_score",
            confidence=svm_confidence(scores, predicted),
            details={"svm_votes": {name: int(decision.votes.get(name, 0)) for name in CLASS_NAMES}},
        )
    if "knn" in steps and model.knn_train_X is not None and model.knn_train_y is not None:
        # didactic.knn_decisions standardizes both matrices with the pipeline scaler. The fitted
        # KNN already stores the standardized training rows, so the query is standardized once
        # here and the function gets an identity scaler: same distances, no per-call transform.
        scaled_query = model.knn_scaler.transform(X)
        view = cast(Pipeline, _PrescaledKnn(model.pipeline.named_steps["knn"]))
        decision = knn_decisions(view, model.knn_train_X, model.knn_train_y, scaled_query)[0]
        distances = ordered_scores(decision.class_min_distance, fill=float("inf"))
        neighbors = []
        for row, distance in zip(
            decision.neighbor_rows[:KNN_NEIGHBORS_SHOWN], decision.neighbor_distances
        ):
            item: dict[str, Any] = {
                "class": str(model.knn_train_y[int(row)]),
                "distance": float(distance),
                "train_index": int(row),
            }
            if model.train_names is not None:
                item["image"] = model.train_names[int(row)]
            neighbors.append(item)
        return ScoreResult(
            scores=distances,
            score_kind="knn_class_distance",
            confidence=knn_confidence(distances),
            details={"k": decision.k, "neighbors": neighbors, "lower_is_better": True},
        )
    if "forest" in steps:
        groups = model.feature_groups
        names = feature_names(groups if sum(groups.values()) == X.shape[1] else {"x": X.shape[1]})
        decision = forest_decisions(model.pipeline, X, names)[0]
        probabilities = ordered_scores(decision.probabilities)
        return ScoreResult(
            scores=probabilities,
            score_kind="rf_probability",
            confidence=forest_confidence(probabilities),
            details={"n_trees": decision.n_trees, "first_tree_path": decision.first_tree_path},
        )
    return generic_scores(model, X, predicted, algorithm)


def generic_scores(
    model: LoadedModel, X: np.ndarray, predicted: str, algorithm: str
) -> ScoreResult:
    """Fallback for algorithms added later to the ModelSpec registry."""
    estimator = model.pipeline
    classes = [str(c) for c in getattr(estimator, "classes_", model.classes)]
    if hasattr(estimator, "predict_proba"):
        values = dict(zip(classes, np.asarray(estimator.predict_proba(X))[0].tolist()))
        scores = ordered_scores(values)
        return ScoreResult(
            scores, "probability", forest_confidence(scores), {"algorithm": algorithm}
        )
    if hasattr(estimator, "decision_function"):
        values = dict(zip(classes, np.asarray(estimator.decision_function(X)).ravel().tolist()))
        scores = ordered_scores(values, fill=float("-inf"))
        return ScoreResult(
            scores, "decision", svm_confidence(scores, predicted), {"algorithm": algorithm}
        )
    scores = {name: 1.0 if name == predicted else 0.0 for name in CLASS_NAMES}
    return ScoreResult(scores, "one_hot", 1.0, {"algorithm": algorithm})
