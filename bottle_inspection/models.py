"""Classifiers and their hyperparameter search.

Every classifier is described by a ModelSpec and registered in MODELS, so the
pipeline trains and evaluates all of them in exactly the same way: same
feature vectors, same training images, same grid search, same metrics. Each
classifier is trained once per feature vector (see config.FEATURE_VECTORS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import GRID_SEARCH_SCORING, N_FOLDS, SEED

ParamGrid = dict[str, list[Any]] | list[dict[str, list[Any]]]
"""Hyperparameter grid in scikit-learn format ("step__param": [values]); a list of grids
searches each one in turn (e.g. one grid per SVM kernel)."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to train and describe one classifier."""

    name: str
    """Short identifier, also used as the output folder name."""
    display_name: str
    """Human-readable name for tables and figures."""
    build: Callable[[], Pipeline]
    """Returns a new, untrained pipeline."""
    param_grid: ParamGrid
    """Grid explored by the grid search."""
    describe: Callable[[Pipeline], dict[str, Any]] = field(default=lambda _model: {})
    """Returns model-specific facts of a trained pipeline (e.g. number of support vectors)."""


# =============================================================================
# SVM
# =============================================================================


def build_svm() -> Pipeline:
    """Standardization followed by an SVM; the kernel is chosen by the grid search.

    The SVM separates two classes with the widest possible margin. With the linear
    kernel the boundary is a plane; the polynomial and RBF kernels let it curve. With
    7 classes, scikit-learn trains one SVM per pair of classes (21 in total) and the
    predicted class is the most voted. `class_weight="balanced"` makes errors on rare
    classes cost more.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
        ]
    )


def describe_svm(model: Pipeline) -> dict[str, Any]:
    """Number of support vectors (training images that define the boundaries)."""
    return {"n_support_vectors": int(model.named_steps["svm"].n_support_.sum())}


SVM = ModelSpec(
    name="svm",
    display_name="SVM",
    build=build_svm,
    param_grid=[
        {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1]},
        {"svm__kernel": ["rbf"], "svm__C": [1, 10, 100], "svm__gamma": ["scale", 1e-3, 1e-4]},
        {
            "svm__kernel": ["poly"],
            "svm__degree": [2, 3],
            "svm__C": [1, 10, 100],
            "svm__gamma": ["scale"],
        },
    ],
    describe=describe_svm,
)

# =============================================================================
# KNN
# =============================================================================


def build_knn() -> Pipeline:
    """Standardization followed by a k-nearest neighbors classifier.

    KNN does not learn a boundary: it stores the training images and labels a new
    image with the most common class among its k closest training images
    (Euclidean distance between feature vectors). Standardization is essential,
    otherwise features with larger ranges would dominate the distance.
    """
    return Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])


KNN = ModelSpec(
    name="knn",
    display_name="KNN",
    build=build_knn,
    param_grid={"knn__n_neighbors": [1, 3, 5, 7, 11, 15], "knn__weights": ["uniform", "distance"]},
)

# =============================================================================
# Random Forest
# =============================================================================


def build_random_forest() -> Pipeline:
    """A random forest: many decision trees voting together.

    Each tree is trained on a bootstrap sample of the training images and, at each
    split, considers only a random subset of the features; the predicted class is
    the majority vote of the trees. Trees compare one feature with a threshold at a
    time, so no standardization is needed. `class_weight="balanced"` gives rare
    classes the same total weight as frequent ones.
    """
    forest = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
    return Pipeline([("forest", forest)])


def describe_random_forest(model: Pipeline) -> dict[str, Any]:
    """Number of trees and mean depth of the trained forest."""
    trees = model.named_steps["forest"].estimators_
    return {
        "n_trees": len(trees),
        "mean_tree_depth": float(np.mean([t.get_depth() for t in trees])),
    }


RANDOM_FOREST = ModelSpec(
    name="random_forest",
    display_name="Random Forest",
    build=build_random_forest,
    param_grid={
        "forest__n_estimators": [100, 300],
        "forest__max_depth": [None, 20],
        "forest__max_features": ["sqrt", 0.1],
    },
    describe=describe_random_forest,
)

# =============================================================================
# Registry
# =============================================================================

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (SVM, KNN, RANDOM_FOREST)}
"""Every available classifier, keyed by name."""


def grid_search(spec: ModelSpec, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
    """Chooses the hyperparameters by stratified k-fold cross-validation on (X, y).

    Only the training images must be passed. The best combination is refit on
    all of them and is available as `result.best_estimator_`.
    """
    search = GridSearchCV(
        spec.build(),
        spec.param_grid,
        scoring=GRID_SEARCH_SCORING,
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    return search.fit(X, y)


def feature_importance_by_group(model: Pipeline, group_sizes: dict[str, int]) -> dict[str, float]:
    """Sums the per-feature importances of a tree model by feature group.

    Returns an empty dict for models without `feature_importances_` (SVM, KNN).
    The importances of a random forest add up to 1, so each value is the share
    of the decisions attributed to that group.
    """
    importances = getattr(model[-1], "feature_importances_", None)
    if importances is None:
        return {}
    shares, start = {}, 0
    for group, size in group_sizes.items():
        shares[group] = float(importances[start : start + size].sum())
        start += size
    return shares


def json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Converts numpy scalars in a parameter dict to plain Python types."""
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in params.items()
    }
