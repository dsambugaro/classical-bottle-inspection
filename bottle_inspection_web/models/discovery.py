"""Discovery of the models trained by the CLI (§8.1). Read-only: nothing here trains.

A trained model is `<model folder>/<feature vector>/<algorithm>/model.joblib`. Its reference is
`<folder>/<feature vector>/<algorithm>` (e.g. `results/rois_hog/svm`). A folder that
directly holds `<algorithm>/model.joblib` (the `<feature vector>` folder of a CLI run) also works;
its reference is `<folder>/<algorithm>`. The metrics are read from the folder that holds the
algorithm folder (`summary.json`, `model_comparison.csv`, ...). Only files inside the configured
model folders are loaded (joblib/pickle is unsafe for untrusted files, REQ-M4).
"""

from __future__ import annotations

import json
import logging
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bottle_inspection.params import ARTIFACT_FORMAT_VERSION, InspectionParams

from ..classes import CLASS_NAMES

log = logging.getLogger(__name__)

MODEL_FILE: str = "model.joblib"
LRU_SIZE: int = 8
"""Loaded models kept in memory (NF3)."""


@dataclass
class ModelInfo:
    """Metadata of one trained model, as stored in the `trained_models` table."""

    model_ref: str
    folder: str
    algorithm: str
    feature_vector: str | None
    """Name of the feature vector the model was trained on (config.FEATURE_VECTORS)."""
    path: Path
    file_mtime: float
    training_params: dict[str, Any]
    """Flat dict (dot paths) of the training InspectionParams."""
    best_params: dict[str, Any]
    feature_length: int
    metrics: dict[str, Any]
    library_versions: dict[str, Any]
    version_warning: str | None = None


@dataclass
class LoadedModel:
    """A trained model in memory, ready to predict."""

    info: ModelInfo
    pipeline: Pipeline
    background: np.ndarray
    classes: list[str]
    params: InspectionParams
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Serializes scoring: the SVM scores temporarily change `decision_function_shape`."""
    knn_train_X: np.ndarray | None = None
    """Standardized training matrix stored by the fitted KNN (row i = i-th train sample)."""
    knn_train_y: np.ndarray | None = None
    knn_scaler: Any = None
    """The StandardScaler of the KNN pipeline (applied once to the query vector)."""
    feature_groups: dict[str, int] = field(default_factory=dict)
    """Size of each feature group of the model's vector."""
    train_names: list[str] | None = None
    """File name of each training image (same order as the KNN rows), when the dataset exists."""


def load_artifact(path: Path) -> tuple[dict[str, Any], str | None]:
    """Loads a model.joblib, returning (artifact, version warning text or None)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = joblib.load(path)
    messages = [str(w.message) for w in caught if _is_version(w)]
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"{path} is not a bottle_inspection model artifact")
    return artifact, ("; ".join(messages) or None)


def _is_version(caught: warnings.WarningMessage) -> bool:
    """True for scikit-learn's InconsistentVersionWarning (and similar unpickle warnings)."""
    return caught.category.__name__ == "InconsistentVersionWarning"


def _read_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON file, or None when missing or invalid."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def parse_classification_report(text: str) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, f1 and support from classification_report.txt."""
    per_class: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in CLASS_NAMES:
            precision, recall, f1, support = (float(v) for v in parts[1:])
            per_class[parts[0]] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
    return {name: per_class.get(name, {}) for name in CLASS_NAMES}


def confusion_from_predictions(path: Path) -> list[list[int]] | None:
    """7x7 confusion matrix (rows true, columns predicted, id order) from the CLI predictions."""
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    index = {name: i for i, name in enumerate(CLASS_NAMES)}
    matrix = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    for true, predicted in zip(frame["true"], frame["predicted"]):
        if true in index and predicted in index:
            matrix[index[true]][index[predicted]] += 1
    return matrix


@dataclass(frozen=True)
class ModelLocation:
    """Where one model.joblib was found."""

    label: str
    """Label of the configured model folder."""
    directory: Path
    """Folder that holds `<algorithm>/model.joblib` and the CLI summary files."""
    algorithm: str
    vector_dir: str | None = None
    """Name of the feature vector folder between the model folder and the algorithm, if any."""

    @property
    def ref(self) -> str:
        """Model reference: `<label>[/<feature vector>]/<algorithm>`."""
        parts = [self.label, self.vector_dir, self.algorithm]
        return "/".join(part for part in parts if part)

    @property
    def path(self) -> Path:
        """Path of the model.joblib."""
        return self.directory / self.algorithm / MODEL_FILE


def split_model_ref(model_ref: str, feature_vector: str | None) -> tuple[str, str]:
    """(folder label, algorithm) of a `<folder>[/<feature vector>]/<algorithm>` reference."""
    rest, _, algorithm = model_ref.rpartition("/")
    suffix = f"/{feature_vector}"
    if feature_vector and rest.endswith(suffix):
        rest = rest[: -len(suffix)]
    return rest, algorithm


def read_metrics(folder: Path, algorithm: str) -> dict[str, Any]:
    """Validation results written by the CLI run (summary, reports, comparison)."""
    summary = _read_json(folder / "summary.json") or {}
    model_summary = summary.get("models", {}).get(algorithm, {})
    metrics: dict[str, Any] = {
        "display_name": model_summary.get("display_name", algorithm),
        "validation": model_summary.get("validation"),
        "cv_f1_macro": model_summary.get("cv_f1_macro"),
        "best_params": model_summary.get("best_params"),
        "inspection_criteria": model_summary.get("inspection_criteria"),
        "train_seconds": model_summary.get("train_seconds"),
        "predict_ms_per_image": model_summary.get("predict_ms_per_image"),
        "extract_ms_per_image": summary.get("extract_ms_per_image"),
        "feature_importance_by_group": model_summary.get("feature_importance_by_group"),
        "facts": {
            k: v
            for k, v in model_summary.items()
            if k in ("n_support_vectors", "n_trees", "mean_tree_depth")
        },
        "photometric_normalization": summary.get("photometric_normalization"),
        "n_features": summary.get("n_features"),
        "dataset": summary.get("dataset"),
    }
    report = folder / algorithm / "classification_report.txt"
    if report.exists():
        metrics["per_class"] = parse_classification_report(report.read_text())
    metrics["confusion_matrix"] = confusion_from_predictions(
        folder / algorithm / "validation_predictions.csv"
    )
    try:
        comparison = pd.read_csv(folder / "model_comparison.csv")
        row = comparison[comparison["model"] == algorithm]
        if len(row):
            metrics["comparison"] = json.loads(str(row.iloc[0].to_json()))
    except (OSError, ValueError, KeyError):
        pass
    metrics["figures"] = [
        name
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png")
        if (folder / algorithm / name).exists()
    ]
    return metrics


def describe_model(location: ModelLocation) -> tuple[ModelInfo, dict[str, Any]]:
    """Loads one model.joblib and its CLI outputs. Returns (info, loaded artifact).

    Raises:
        ValueError: when the artifact lacks the expected keys (`feature_vector`, `params`, ...)
            or has a `format_version` other than `ARTIFACT_FORMAT_VERSION`.
    """
    folder, algorithm, path = location.directory, location.algorithm, location.path
    artifact, version_warning = load_artifact(path)
    feature_vector = artifact.get("feature_vector")
    saved_params = artifact.get("params")
    format_version = artifact.get("format_version")
    if (
        not isinstance(feature_vector, str)
        or not isinstance(saved_params, dict)
        or format_version != ARTIFACT_FORMAT_VERSION
    ):
        raise ValueError(
            f"{path} is not a model this app can load (it needs feature_vector, params and "
            f"format_version {ARTIFACT_FORMAT_VERSION}, found format_version {format_version})"
        )
    summary = _read_json(folder / "summary.json")
    params = InspectionParams.from_dict(saved_params)
    feature_length = artifact.get("feature_length")
    if feature_length is None:
        n_features_in = getattr(artifact["model"], "n_features_in_", None)
        feature_length = int(n_features_in) if n_features_in is not None else 0
    metrics = read_metrics(folder, algorithm)
    metrics["feature_vector"] = feature_vector
    metrics["format_version"] = format_version
    info = ModelInfo(
        model_ref=location.ref,
        folder=location.label,
        algorithm=algorithm,
        feature_vector=feature_vector,
        path=path,
        file_mtime=path.stat().st_mtime,
        training_params=params.to_flat_dict(),
        best_params=dict(
            artifact.get("best_params")
            or (summary or {}).get("models", {}).get(algorithm, {}).get("best_params", {})
        ),
        feature_length=int(feature_length),
        metrics=metrics,
        library_versions=dict(artifact.get("library_versions") or {}),
        version_warning=version_warning,
    )
    return info, artifact


def scan_folders(folders: list[tuple[str, Path]]) -> list[ModelLocation]:
    """Every model.joblib in the model folders (nested or direct layout, see the module doc)."""
    found = []
    for label, folder in folders:
        if not folder.is_dir():
            continue
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / MODEL_FILE).is_file():
                found.append(ModelLocation(label, folder, sub.name))
                continue
            for algorithm in sorted(sub.iterdir()):
                if algorithm.is_dir() and (algorithm / MODEL_FILE).is_file():
                    found.append(ModelLocation(label, sub, algorithm.name, sub.name))
    return found


class ModelCache:
    """LRU of loaded models keyed by (model_ref, file mtime)."""

    def __init__(self, size: int = LRU_SIZE) -> None:
        """Empty cache holding at most `size` models."""
        self.size = size
        self._items: OrderedDict[tuple[str, float], LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._loading: dict[tuple[str, float], threading.Lock] = {}

    def prime(self, info: ModelInfo, artifact: dict[str, Any], dataset_root: Path | None) -> None:
        """Stores a model whose artifact was already loaded (by discovery)."""
        loaded = load_model(info, dataset_root, artifact)
        with self._lock:
            self._items[(info.model_ref, info.file_mtime)] = loaded
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def get(self, info: ModelInfo, dataset_root: Path | None) -> LoadedModel:
        """Returns the loaded model, loading it once (thread-safe)."""
        key = (info.model_ref, info.file_mtime)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            key_lock = self._loading.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                if key in self._items:
                    return self._items[key]
            loaded = load_model(info, dataset_root)
            with self._lock:
                self._items[key] = loaded
                self._loading.pop(key, None)
                while len(self._items) > self.size:
                    self._items.popitem(last=False)
            return loaded

    def clear(self) -> None:
        """Drops every loaded model."""
        with self._lock:
            self._items.clear()


def load_model(
    info: ModelInfo, dataset_root: Path | None, artifact: dict[str, Any] | None = None
) -> LoadedModel:
    """Loads the pipeline, background and (for KNN) the training rows of a trained model."""
    if artifact is None:
        artifact = load_artifact(info.path)[0]
    pipeline: Pipeline = artifact["model"]
    loaded = LoadedModel(
        info=info,
        pipeline=pipeline,
        background=np.asarray(artifact["background"]),
        classes=[str(c) for c in artifact.get("classes", CLASS_NAMES)],
        params=InspectionParams.from_dict(info.training_params),
        feature_groups=dict(artifact.get("feature_groups") or {}),
    )
    steps = getattr(pipeline, "named_steps", {})
    if "knn" in steps and "scaler" in steps:
        knn, scaler = steps["knn"], steps["scaler"]
        train_X = np.asarray(knn._fit_X)
        loaded.knn_train_X = train_X
        loaded.knn_scaler = scaler
        loaded.knn_train_y = np.asarray(knn.classes_)[np.asarray(knn._y)]
        loaded.train_names = training_image_names(dataset_root, len(train_X))
    return loaded


def training_image_names(dataset_root: Path | None, expected: int) -> list[str] | None:
    """Names of the training images in CLI order (dataset.load_dataset filtered to train)."""
    if dataset_root is None or not (dataset_root / "images" / "train").is_dir():
        return None
    from bottle_inspection.dataset import load_dataset

    try:
        names = [s.image_path.name for s in load_dataset(dataset_root) if s.split == "train"]
    except (OSError, ValueError) as error:
        log.warning("cannot read the dataset for KNN neighbours: %s", error)
        return None
    return names if len(names) == expected else None
