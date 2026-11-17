"""Fixtures: small synthetic trained models, a temporary data folder and a test client."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bottle_inspection.config import (
    CLASSES,
    FEATURE_VECTORS,
    HOG_ORIENTATIONS,
    HSV_HISTOGRAM_BINS,
    LBP_POINTS,
)
from bottle_inspection.params import InspectionParams, training_metadata

from ..app import create_app
from ..db.session import Database
from ..settings import AppConfig, AppPaths
from ..settings_store import SettingsStore

CODE_DIR = Path(__file__).resolve().parents[2]
HOG_BLOCK_COUNTS = {
    "cap": 35,
    "label": 15,
    "body": 22,
    "auto_1": 25,
    "auto_2": 25,
    "auto_3": 55,
    "bottle": 115,
}
"""Number of HOG blocks (2x2 cells) of each region with the default config and the AUTO_BANDS below."""
HOG_SIZES = {region: blocks * 4 * HOG_ORIENTATIONS for region, blocks in HOG_BLOCK_COUNTS.items()}
"""HOG length of each region with the default config and the AUTO_BANDS below."""
LBP_CELL_COUNTS = {
    "cap": 12,
    "label": 24,
    "body": 36,
    "auto_1": 36,
    "auto_2": 36,
    "auto_3": 72,
    "bottle": 144,
}
"""Number of LBP cells of each region with the default config and the AUTO_BANDS below."""
COLOR_SIZE = sum(HSV_HISTOGRAM_BINS) + 2 * 3
"""Length of each colour group: the H, S and V histograms plus the mean and std of each channel."""
GROUP_SIZES = {
    **{f"{region}_hog": size for region, size in HOG_SIZES.items()},
    **{f"{region}_lbp": cells * (LBP_POINTS + 2) for region, cells in LBP_CELL_COUNTS.items()},
    **{f"{region}_color": COLOR_SIZE for region in HOG_SIZES},
}
"""Real group sizes with the default config and the AUTO_BANDS below."""
AUTO_BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0))
"""Bands of the automatic regions saved in the fixture models (multiples of 16 of 384 rows)."""
FIXTURE_VECTORS: tuple[str, ...] = ("rois_hog", "bottle_lbp", "auto_rois_color")
"""Feature vectors written by `write_model_folder` in the nested layout."""


def vector_sizes(vector: str) -> dict[str, int]:
    """Size of each feature group of a feature vector, in vector order."""
    return {group: GROUP_SIZES[group] for group in FEATURE_VECTORS[vector]}


def synthetic_background() -> np.ndarray:
    """A smooth wall-like background at the localization size (640x360)."""
    gradient = np.linspace(120, 200, 640, dtype=np.float32)[None, :, None]
    return np.clip(np.repeat(np.repeat(gradient, 360, axis=0), 3, axis=2), 0, 255).astype(np.uint8)


def synthetic_image(seed: int = 0) -> np.ndarray:
    """A 1280x720 image: the background plus a dark vertical 'bottle'."""
    rng = np.random.default_rng(seed)
    image = cv2.resize(synthetic_background(), (1280, 720))
    x = int(rng.integers(500, 700))
    cv2.rectangle(image, (x, 80), (x + 180, 650), (40, 60, 90), -1)
    cv2.rectangle(image, (x + 40, 60), (x + 140, 120), (200, 80, 20), -1)
    return image


def write_algorithms(
    folder: Path,
    vector: str,
    algorithms: tuple[str, ...],
) -> None:
    """Tiny trained pipelines plus CLI-like outputs, directly in `folder` (the layout of a
    `<output>/<feature vector>/` folder of the CLI)."""
    sizes = vector_sizes(vector)
    length = sum(sizes.values())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(35, length))
    y = np.array([CLASSES[i % 7] for i in range(35)])
    pipelines = {
        "svm": Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
        "knn": Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))]
        ),
        "random_forest": Pipeline([("forest", RandomForestClassifier(n_estimators=5))]),
    }
    folder.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "feature_vector": vector,
        "feature_groups": sizes,
        "photometric_normalization": True,
        "n_features": length,
        "models": {},
    }
    for name in algorithms:
        pipeline = pipelines[name]
        pipeline.fit(X, y)
        artifact = {"model": pipeline, "background": synthetic_background(), "classes": CLASSES}
        params = InspectionParams.defaults().with_auto_bands(AUTO_BANDS)
        artifact.update(training_metadata(True, {"C": 1}, sizes, params))
        artifact["feature_vector"] = vector
        (folder / name).mkdir(exist_ok=True)
        joblib.dump(artifact, folder / name / "model.joblib")
        lines = ["              precision    recall  f1-score   support", ""]
        lines += [f"{c:>20}     1.0000    1.0000    1.0000        10" for c in CLASSES]
        (folder / name / "classification_report.txt").write_text("\n".join(lines))
        (folder / name / "validation_predictions.csv").write_text(
            "image,true,predicted\n"
            + "".join(f"img{i}.jpg,{c},{c}\n" for i, c in enumerate(CLASSES))
        )
        summary["models"][name] = {
            "display_name": name,
            "validation": {
                "accuracy": 1.0,
                "f1_macro": 1.0,
                "f1_per_class": {c: 1.0 for c in CLASSES},
            },
            "best_params": {"C": 1},
            "cv_f1_macro": 0.99,
        }
    (folder / "summary.json").write_text(json.dumps(summary))


def write_model_folder(folder: Path) -> None:
    """A model folder like the CLI output: `<folder>/<feature vector>/<algorithm>/model.joblib`
    for `rois_hog` (svm, knn, random_forest), `bottle_lbp` (svm) and `auto_rois_color` (svm)."""
    write_algorithms(folder / "rois_hog", "rois_hog", ("svm", "knn", "random_forest"))
    write_algorithms(folder / "bottle_lbp", "bottle_lbp", ("svm",))
    write_algorithms(folder / "auto_rois_color", "auto_rois_color", ("svm",))


def write_broken_model(folder: Path) -> None:
    """`<folder>/rois_hog/svm/model.joblib` holding a dict without the expected keys."""
    (folder / "rois_hog" / "svm").mkdir(parents=True)
    joblib.dump({"model": "not a pipeline"}, folder / "rois_hog" / "svm" / "model.joblib")


@pytest.fixture(scope="session")
def code_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fake `code/` folder with the model folders `fx` (valid models) and `broken` (a
    model.joblib without the expected keys)."""
    root = tmp_path_factory.mktemp("code")
    write_model_folder(root / "fx")
    write_broken_model(root / "broken")
    return root


def make_config(tmp_path: Path, code_dir: Path, folders: list[str] | None = None) -> AppConfig:
    """App config with a fresh data folder and the given model folders."""
    paths = AppPaths(root=tmp_path / "webdata", code_dir=code_dir)
    paths.ensure()
    db = Database(paths.db)
    db.upgrade(paths.backups)
    SettingsStore(db).update(
        {"model_folders": folders or ["fx", "broken"], "dataset_path": "no_dataset"}
    )
    return AppConfig(paths=paths, frontend_dist=tmp_path / "no_dist")


@pytest.fixture()
def client(tmp_path: Path, code_dir: Path) -> Iterator[TestClient]:
    """Test client of an app using the fixture models."""
    app = create_app(make_config(tmp_path, code_dir))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def jpeg() -> bytes:
    """A synthetic bottle image as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", synthetic_image())
    assert ok
    return buffer.tobytes()
