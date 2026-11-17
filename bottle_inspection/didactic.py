"""Didactic figures of how each classifier decides (optional, `--didactic-plots`).

Every number drawn here comes from the real trained models, in the full space of
the feature vector they were trained on; nothing is projected or retrained:

- KNN: real distances (standardized features) to the training images;
- Random Forest: real class probabilities (average of the 100 trees) and the
  real path of the image through the first tree;
- SVM: real votes of the 21 one-vs-one SVMs and the real per-class scores.

Outputs:

- classifier_margins.png              one point per validation image and model: by how
                                      much the true class won (> 0 = correct)
- didactic/<validation image>.png     one figure per validation image with the decision
                                      of the three classifiers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .config import CLASSES  # noqa: E402

CLASS_COLORS: dict[str, str] = {
    "bottle_no_defect": "#2b8a3e",
    "bottle_other_brand": "#868e96",
    "defect_low_fill": "#1c7ed6",
    "defect_no_cap": "#e03131",
    "defect_loose_cap": "#f08c00",
    "defect_debris": "#7048e8",
    "defect_label_damage": "#c2255c",
}
"""One fixed color per class, used by every didactic figure."""

KNN_NEIGHBORS_SHOWN: int = 5
"""Closest training images drawn next to the validation image."""

TREE_PATH_STEPS_SHOWN: int = 9
"""Maximum questions of the first-tree path written in the figure."""

MODEL_TITLES: dict[str, str] = {
    "knn": "KNN",
    "random_forest": "RANDOM FOREST",
    "svm": "SVM",
}
"""Column titles, in the column order of the figures."""


# =============================================================================
# Real decision data, computed once for all validation images
# =============================================================================


@dataclass
class KnnDecision:
    """Real KNN distances of one validation image."""

    neighbor_rows: np.ndarray
    """Indices of the closest training images, closest first."""
    neighbor_distances: np.ndarray
    class_min_distance: dict[str, float]
    """Distance to the closest training image of each class (the smallest wins with k=1)."""
    k: int


@dataclass
class ForestDecision:
    """Real Random Forest output for one validation image."""

    probabilities: dict[str, float]
    """Average class probability over all trees; the largest is the prediction."""
    first_tree_path: list[str]
    """Questions answered by the image in the first tree, root to leaf."""
    n_trees: int


@dataclass
class SvmDecision:
    """Real SVM output for one validation image."""

    votes: dict[str, int]
    """Votes received from the 21 one-vs-one SVMs (max 6 per class)."""
    scores: dict[str, float]
    """scikit-learn one-vs-rest scores: votes plus a small confidence term."""


def feature_names(group_sizes: dict[str, int]) -> list[str]:
    """Readable name of every value of the feature vector, e.g. "cap_color[12]"."""
    return [f"{group}[{i}]" for group, size in group_sizes.items() for i in range(size)]


def knn_decisions(
    model: Pipeline, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> list[KnnDecision]:
    """Distances, in the standardized feature space, from each validation image to the training set."""
    scaler, knn = model.named_steps["scaler"], model.named_steps["knn"]
    train_scaled, val_scaled = scaler.transform(X_train), scaler.transform(X_val)
    squared = (
        (val_scaled**2).sum(axis=1)[:, None]
        + (train_scaled**2).sum(axis=1)[None, :]
        - 2 * val_scaled @ train_scaled.T
    )
    distances = np.sqrt(np.clip(squared, 0, None))
    decisions = []
    for row in distances:
        order = np.argsort(row)[:KNN_NEIGHBORS_SHOWN]
        per_class = {name: float(row[y_train == name].min()) for name in CLASSES}
        decisions.append(KnnDecision(order, row[order], per_class, int(knn.n_neighbors)))
    return decisions


def tree_path(estimator: Any, x: np.ndarray, names: list[str], classes: np.ndarray) -> list[str]:
    """Questions answered by one image in one decision tree, root to leaf."""
    tree = estimator.tree_
    node, steps = 0, []
    while tree.children_left[node] != -1:
        feature, threshold = int(tree.feature[node]), float(tree.threshold[node])
        value = float(x[feature])
        goes_left = value <= threshold
        sign = "<=" if goes_left else ">"
        steps.append(f"{names[feature]} = {value:.3f} {sign} {threshold:.3f}")
        node = int(tree.children_left[node] if goes_left else tree.children_right[node])
    leaf_class = str(classes[int(np.argmax(tree.value[node][0]))])
    if len(steps) > TREE_PATH_STEPS_SHOWN:
        hidden = len(steps) - TREE_PATH_STEPS_SHOWN + 1
        steps = steps[: TREE_PATH_STEPS_SHOWN - 1] + [f"... {hidden} more questions ..."]
    return steps + [f"leaf -> {leaf_class}"]


def forest_decisions(model: Pipeline, X_val: np.ndarray, names: list[str]) -> list[ForestDecision]:
    """Class probabilities of the forest and the path through its first tree."""
    forest = model.named_steps["forest"]
    probabilities = forest.predict_proba(X_val)
    classes = [str(name) for name in forest.classes_]
    return [
        ForestDecision(
            probabilities=dict(zip(classes, row.tolist())),
            first_tree_path=tree_path(forest.estimators_[0], x, names, forest.classes_),
            n_trees=len(forest.estimators_),
        )
        for x, row in zip(X_val, probabilities)
    ]


def svm_votes(decision_ovo: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts the votes of the one-vs-one SVMs (pair order: (0,1), (0,2), ..., (n-2,n-1)).

    A positive decision value is a vote for the first class of the pair.
    """
    votes = np.zeros((len(decision_ovo), n_classes), dtype=int)
    pair = 0
    for first in range(n_classes):
        for second in range(first + 1, n_classes):
            votes[:, first] += decision_ovo[:, pair] > 0
            votes[:, second] += decision_ovo[:, pair] <= 0
            pair += 1
    return votes


def svm_decisions(model: Pipeline, X_val: np.ndarray) -> list[SvmDecision]:
    """Votes of the 21 pairwise SVMs and the one-vs-rest scores of each validation image."""
    svc = model.named_steps["svm"]
    scaled = model.named_steps["scaler"].transform(X_val)
    original_shape = svc.decision_function_shape
    svc.decision_function_shape = "ovo"
    decision_ovo = svc.decision_function(scaled)
    svc.decision_function_shape = original_shape
    scores = svc.decision_function(scaled)
    classes = [str(name) for name in svc.classes_]
    votes = svm_votes(decision_ovo, len(classes))
    return [
        SvmDecision(dict(zip(classes, v.tolist())), dict(zip(classes, s.tolist())))
        for v, s in zip(votes, scores)
    ]


# =============================================================================
# Global figure: margin of victory of the true class
# =============================================================================


def true_class_margin(values: dict[str, float], true_label: str, larger_wins: bool = True) -> float:
    """How much the true class beat the best wrong class (> 0 means a correct prediction)."""
    others = [value for name, value in values.items() if name != true_label]
    if larger_wins:
        return values[true_label] - max(others)
    return min(others) - values[true_label]


MARGIN_AXIS_LABELS: dict[str, str] = {
    "knn": "distance to closest wrong class\n- distance to closest true class",
    "random_forest": "probability of true class\n- best wrong class",
    "svm": "score of true class\n- best wrong class",
}
"""Y-axis label of each model in classifier_margins.png."""


def plot_classifier_margins(margins: dict[str, np.ndarray], y_val: np.ndarray, path: Path) -> None:
    """One strip per class and model: every validation image as a point at its real margin."""
    names = [name for name in MODEL_TITLES if name in margins]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        for position, class_name in enumerate(CLASSES):
            values = margins[name][y_val == class_name]
            jitter = rng.uniform(-0.25, 0.25, len(values))
            ax.scatter(position + jitter, values, s=12, color=CLASS_COLORS[class_name], alpha=0.8)
        ax.axhline(0, color="black", lw=1)
        wrong = int((margins[name] <= 0).sum())
        ax.set_title(
            f"{MODEL_TITLES[name]}: {wrong} of {len(y_val)} at or below 0", fontweight="bold"
        )
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(MARGIN_AXIS_LABELS[name], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Margin of victory of the true class on each validation image "
        "(real models; above 0 = correct)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# Per-image figure
# =============================================================================


def class_bars(ax: Axes, values: dict[str, float], highlight: str, xlabel: str, fmt: str) -> None:
    """Horizontal bars for the 7 classes; the winning class gets a black edge."""
    data = [values.get(name, 0.0) for name in CLASSES]
    bars = ax.barh(range(len(CLASSES)), data, color=[CLASS_COLORS[name] for name in CLASSES])
    for name, bar, value in zip(CLASSES, bars, data):
        if name == highlight:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            " " + fmt.format(value),
            va="center",
            fontsize=7,
        )
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.margins(x=0.25)


def show_crop(ax: Axes, crop: np.ndarray, title: str, label: str) -> None:
    """A bottle crop framed with the color of its class."""
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(CLASS_COLORS[label])
        spine.set_linewidth(3)


def draw_knn(
    fig: Figure,
    slots: tuple[Any, Any],
    decision: KnnDecision,
    query_crop: np.ndarray,
    query_label: str,
    neighbor_crops: list[np.ndarray],
    neighbor_labels: list[str],
) -> str:
    """Validation image, its closest training images and the closest distance per class."""
    strip = slots[0].subgridspec(1, len(neighbor_crops) + 1, wspace=0.08)
    show_crop(fig.add_subplot(strip[0]), query_crop, "validation\nimage", query_label)
    for rank, (crop, label, distance) in enumerate(
        zip(neighbor_crops, neighbor_labels, decision.neighbor_distances)
    ):
        voter = " votes" if rank < decision.k else ""
        title = f"#{rank + 1}{voter}\nd={distance:.1f}"
        show_crop(fig.add_subplot(strip[rank + 1]), crop, title, label)
    distances = decision.class_min_distance
    predicted = min(distances, key=lambda name: distances[name])
    class_bars(
        fig.add_subplot(slots[1]),
        distances,
        predicted,
        f"distance to the closest training image of each class\n(k={decision.k}: smallest wins)",
        "{:.1f}",
    )
    return predicted


def draw_forest(fig: Figure, slots: tuple[Any, Any], decision: ForestDecision) -> str:
    """Average class probability of the trees and the path through the first tree."""
    probabilities = decision.probabilities
    predicted = max(probabilities, key=lambda name: probabilities[name])
    class_bars(
        fig.add_subplot(slots[0]),
        probabilities,
        predicted,
        f"average probability over the {decision.n_trees} trees (largest wins)",
        "{:.2f}",
    )
    ax = fig.add_subplot(slots[1])
    ax.axis("off")
    lines = [f"{i + 1:>2}. {step}" for i, step in enumerate(decision.first_tree_path)]
    ax.text(
        0.0,
        1.0,
        "Path of this image in tree 1:\n" + "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=7,
        transform=ax.transAxes,
    )
    return predicted


def draw_svm(fig: Figure, slots: tuple[Any, Any], decision: SvmDecision) -> str:
    """Votes of the 21 one-vs-one SVMs and the final one-vs-rest scores."""
    scores = decision.scores
    predicted = max(scores, key=lambda name: scores[name])
    votes = {name: float(count) for name, count in decision.votes.items()}
    class_bars(
        fig.add_subplot(slots[0]),
        votes,
        predicted,
        "votes from the 21 pairwise SVMs (max 6; most votes wins)",
        "{:.0f}",
    )
    class_bars(
        fig.add_subplot(slots[1]),
        scores,
        predicted,
        "score = votes + confidence (largest wins)",
        "{:.2f}",
    )
    return predicted


@dataclass
class ImageCase:
    """Everything needed to draw the decision figure of one validation image."""

    path: Path
    image_name: str
    true_label: str
    crop: np.ndarray
    neighbor_crops: list[np.ndarray]
    neighbor_labels: list[str]
    knn: KnnDecision | None
    forest: ForestDecision | None
    svm: SvmDecision | None


def plot_image_decision(case: ImageCase) -> None:
    """One figure for one validation image: a column per classifier, two rows each."""
    columns: list[tuple[str, Any]] = [
        (name, decision)
        for name, decision in (("knn", case.knn), ("random_forest", case.forest), ("svm", case.svm))
        if decision is not None
    ]
    fig = plt.figure(figsize=(6.4 * len(columns), 8.4))
    grid = fig.add_gridspec(2, len(columns), height_ratios=(1.15, 1), hspace=0.5, wspace=0.6)
    verdicts = []
    for column, (name, decision) in enumerate(columns):
        slots = (grid[0, column], grid[1, column])
        if isinstance(decision, KnnDecision):
            predicted = draw_knn(
                fig,
                slots,
                decision,
                case.crop,
                case.true_label,
                case.neighbor_crops,
                case.neighbor_labels,
            )
        elif isinstance(decision, ForestDecision):
            predicted = draw_forest(fig, slots, decision)
        else:
            predicted = draw_svm(fig, slots, decision)
        mark = "correct" if predicted == case.true_label else "WRONG"
        verdicts.append(f"{MODEL_TITLES[name]}: {predicted} ({mark})")
        fig.text(
            (column + 0.5) / len(columns),
            0.905,
            MODEL_TITLES[name],
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.suptitle(
        f"{case.image_name}  -  true class: {case.true_label}\n" + "   |   ".join(verdicts),
        fontsize=10,
    )
    fig.savefig(case.path, dpi=110)
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================


def generate_didactic_plots(
    models: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_crops: list[np.ndarray],
    val_crops: list[np.ndarray],
    val_names: list[str],
    group_sizes: dict[str, int],
    output_dir: Path,
) -> None:
    """Writes classifier_margins.png and one decision figure per validation image."""
    print("\n== Didactic plots (real decisions of the trained models)")
    n_val = len(y_val)
    knn: list[KnnDecision | None] = [None] * n_val
    forest: list[ForestDecision | None] = [None] * n_val
    svm: list[SvmDecision | None] = [None] * n_val
    margins: dict[str, np.ndarray] = {}

    if "knn" in models:
        knn_list = knn_decisions(models["knn"], X_train, y_train, X_val)
        knn = list(knn_list)
        margins["knn"] = np.array(
            [true_class_margin(d.class_min_distance, t, False) for d, t in zip(knn_list, y_val)]
        )
    if "random_forest" in models:
        names = feature_names(group_sizes)
        forest_list = forest_decisions(models["random_forest"], X_val, names)
        forest = list(forest_list)
        margins["random_forest"] = np.array(
            [true_class_margin(d.probabilities, t) for d, t in zip(forest_list, y_val)]
        )
    if "svm" in models:
        svm_list = svm_decisions(models["svm"], X_val)
        svm = list(svm_list)
        margins["svm"] = np.array([true_class_margin(d.scores, t) for d, t in zip(svm_list, y_val)])

    if not margins:
        return
    plot_classifier_margins(margins, y_val, output_dir / "classifier_margins.png")

    folder = output_dir / "didactic"
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for i in range(n_val):
        knn_case = knn[i]
        rows = knn_case.neighbor_rows if knn_case is not None else np.array([], dtype=int)
        cases.append(
            ImageCase(
                path=folder / f"{Path(val_names[i]).stem}.png",
                image_name=val_names[i],
                true_label=str(y_val[i]),
                crop=val_crops[i],
                neighbor_crops=[train_crops[row] for row in rows],
                neighbor_labels=[str(y_train[row]) for row in rows],
                knn=knn_case,
                forest=forest[i],
                svm=svm[i],
            )
        )
    print(f"   writing {n_val} figures to {folder}")
    Parallel(n_jobs=-1)(delayed(plot_image_decision)(case) for case in cases)
