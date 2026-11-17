"""Orchestration: runs every step in order and writes the results.

Every classifier is trained once per feature vector of config.FEATURE_VECTORS (one
descriptor at a time), and the results are compared in a feature x classifier matrix.

Output layout::

    <output>/
        feature_cache.joblib          cached features (reused by later runs)
        summary.json                  every metric of the run
        feature_comparison.csv        one row per (feature vector, classifier)
        feature_matrix_<metric>.csv   feature x classifier matrix (accuracy, f1_macro)
        feature_matrix.png            the two matrices as heatmaps
        background_model.png, bottle_localization.png, rois_per_class.png, auto_rois_per_class.png
        <feature>/                    one folder per feature vector (e.g. rois_hog)
            summary.json, model_comparison.{csv,png}
            classifier_margins.png, didactic/<image>.png (only with --didactic-plots)
            <model>/                  one folder per classifier
                model.joblib, classification_report.txt, confusion_matrix*.png,
                criteria_metrics.csv, validation_predictions.csv, simulated_alerts.csv,
                validation_errors.png (only when there are errors),
                feature_importance.csv (tree models only)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from . import plots
from .config import CLASSES, FEATURE_VECTORS, ROIS, SEED
from .dataset import Sample, class_distribution, count_duplicate_images, load_dataset, split_indices
from .didactic import generate_didactic_plots
from .evaluation import (
    ZERO_DIVISION,
    classification_metrics,
    inspection_criteria_metrics,
    localization_error,
    pass_reject_accuracy,
    simulate_alerts,
)
from .extraction import ExtractedDataset, load_or_extract
from .features import build_matrix
from .models import (
    MODELS,
    ModelSpec,
    feature_importance_by_group,
    grid_search,
    json_safe_params,
)
from .params import training_metadata

ModelKey = tuple[str, str]
"""(feature vector name, classifier name) of one trained model."""

MATRIX_METRICS: list[str] = ["accuracy", "f1_macro"]
"""Metrics written as feature x classifier matrices."""


@dataclass(frozen=True)
class RunOptions:
    """Command line options of one run."""

    data_dir: Path
    output_dir: Path
    model_names: list[str]
    feature_names: list[str]
    recompute: bool
    normalize: bool
    didactic_plots: bool = False


@dataclass
class DatasetSplit:
    """Samples, labels and the original Kaggle train/val indices."""

    samples: list[Sample]
    y: np.ndarray
    train: list[int]
    validation: list[int]

    @property
    def validation_samples(self) -> list[Sample]:
        """Validation samples, in file order."""
        return [self.samples[i] for i in self.validation]


@dataclass
class TrainedModel:
    """A classifier trained on one feature vector, with its validation predictions."""

    feature: str
    """Name of the feature vector (key of config.FEATURE_VECTORS)."""
    spec: ModelSpec
    model: Pipeline
    n_features: int
    best_params: dict[str, Any]
    cv_score: float
    train_seconds: float
    y_pred: np.ndarray
    predict_ms_per_image: float


# =============================================================================
# Steps
# =============================================================================


def prepare_data(options: RunOptions) -> tuple[DatasetSplit, ExtractedDataset]:
    """Loads the dataset with the Kaggle split and extracts (or loads cached) features."""
    print("== Loading dataset")
    samples = load_dataset(options.data_dir)
    train, validation = split_indices(samples, "train"), split_indices(samples, "val")
    print(pd.DataFrame(class_distribution(samples)).to_string())

    cache_path = options.output_dir / "feature_cache.joblib"
    extracted = load_or_extract(samples, train, options.normalize, cache_path, options.recompute)
    y = np.array([sample.label for sample in samples])
    print(f"{len(train)} train / {len(validation)} validation images")
    return DatasetSplit(samples, y, train, validation), extracted


def save_preprocessing_figures(
    data: DatasetSplit, extracted: ExtractedDataset, output_dir: Path
) -> None:
    """Figures that do not depend on the features or the classifier."""
    cv2.imwrite(str(output_dir / "background_model.png"), extracted.background)
    plots.plot_localization(
        data.samples,
        extracted.bottles,
        extracted.background,
        output_dir / "bottle_localization.png",
    )
    validation_bottles = [extracted.bottles[i] for i in data.validation]
    plots.plot_regions_per_class(
        data.validation_samples,
        validation_bottles,
        dict(ROIS),
        "Detected bottle and ad hoc ROIs (red=cap, green=label, blue=body)",
        output_dir / "rois_per_class.png",
    )
    plots.plot_regions_per_class(
        data.validation_samples,
        validation_bottles,
        extracted.params.auto_bands(),
        "Detected bottle and automatic regions (learned from the training images)",
        output_dir / "auto_rois_per_class.png",
    )


def feature_group_sizes(extracted: ExtractedDataset, feature: str) -> dict[str, int]:
    """Number of values of each feature group of a feature vector, in vector order."""
    first_image = extracted.feature_sets[0]
    return {group: len(first_image[group]) for group in FEATURE_VECTORS[feature]}


def train_model(spec: ModelSpec, feature: str, X: np.ndarray, data: DatasetSplit) -> TrainedModel:
    """Grid search on the training images, then prediction of the validation images.

    `X` is the matrix of the feature vector `feature` for every dataset image.
    """
    print(f"\n== Training {spec.display_name} on {feature} ({X.shape[1]} values per image)")
    start = time.perf_counter()
    search = grid_search(spec, X[data.train], data.y[data.train])
    train_seconds = time.perf_counter() - start
    model = search.best_estimator_

    start = time.perf_counter()
    y_pred = model.predict(X[data.validation])
    predict_ms = 1000 * (time.perf_counter() - start) / len(data.validation)

    print(f"Best parameters: {search.best_params_} (CV F1-macro={search.best_score_:.4f})")
    return TrainedModel(
        feature=feature,
        spec=spec,
        model=model,
        n_features=int(X.shape[1]),
        best_params=json_safe_params(search.best_params_),
        cv_score=float(search.best_score_),
        train_seconds=train_seconds,
        y_pred=y_pred,
        predict_ms_per_image=predict_ms,
    )


def save_model_outputs(
    trained: TrainedModel, data: DatasetSplit, extracted: ExtractedDataset, feature_dir: Path
) -> None:
    """Report, confusion matrices, criteria, predictions, alerts and errors of one model."""
    folder = feature_dir / trained.spec.name
    folder.mkdir(parents=True, exist_ok=True)
    y_true = data.y[data.validation]
    samples = data.validation_samples
    bottles = [extracted.bottles[i] for i in data.validation]
    group_sizes = feature_group_sizes(extracted, trained.feature)
    title = f"{trained.spec.display_name} on {trained.feature}"

    joblib.dump(
        {
            "model": trained.model,
            "background": extracted.background,
            "classes": CLASSES,
            "feature_vector": trained.feature,
            **training_metadata(
                normalize=extracted.normalize,
                best_params=trained.best_params,
                feature_groups=group_sizes,
                params=extracted.params,
            ),
        },
        folder / "model.joblib",
        compress=3,
    )
    report = classification_report(
        y_true, trained.y_pred, labels=CLASSES, digits=4, zero_division=ZERO_DIVISION
    )
    (folder / "classification_report.txt").write_text(str(report))
    print(report)

    for normalize in (False, True):
        suffix = "_normalized" if normalize else ""
        plots.plot_confusion_matrix(
            y_true, trained.y_pred, folder / f"confusion_matrix{suffix}.png", normalize, title
        )
    inspection_criteria_metrics(y_true, trained.y_pred).to_csv(
        folder / "criteria_metrics.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(
        {"image": [s.image_path.name for s in samples], "true": y_true, "predicted": trained.y_pred}
    ).to_csv(folder / "validation_predictions.csv", index=False)
    simulate_alerts(samples, trained.y_pred).to_csv(folder / "simulated_alerts.csv", index=False)
    errors_path = folder / "validation_errors.png"
    if not plots.plot_errors(samples, bottles, y_true, trained.y_pred, errors_path):
        errors_path.unlink(missing_ok=True)

    importance = feature_importance_by_group(trained.model, group_sizes)
    if importance:
        pd.Series(importance, name="importance").rename_axis("feature_group").to_csv(
            folder / "feature_importance.csv", float_format="%.4f"
        )


def save_didactic_plots(
    feature: str,
    trained: dict[str, TrainedModel],
    X: np.ndarray,
    data: DatasetSplit,
    extracted: ExtractedDataset,
    feature_dir: Path,
) -> None:
    """Figures of the real decision of each classifier on one feature vector (see didactic.py)."""
    generate_didactic_plots(
        models={name: result.model for name, result in trained.items()},
        X_train=X[data.train],
        y_train=data.y[data.train],
        X_val=X[data.validation],
        y_val=data.y[data.validation],
        train_crops=[extracted.bottles[i].crop for i in data.train],
        val_crops=[extracted.bottles[i].crop for i in data.validation],
        val_names=[sample.image_path.name for sample in data.validation_samples],
        group_sizes=feature_group_sizes(extracted, feature),
        output_dir=feature_dir,
    )


def build_comparison(
    trained: dict[ModelKey, TrainedModel],
    data: DatasetSplit,
    extracted: ExtractedDataset,
) -> pd.DataFrame:
    """One row per (feature vector, classifier) with the metrics used to compare them."""
    y_true = data.y[data.validation]
    rows = []
    for (feature, name), result in trained.items():
        metrics = classification_metrics(y_true, result.y_pred)
        extract_ms = extracted.extract_ms(FEATURE_VECTORS[feature])
        scope, _, descriptor = feature.rpartition("_")
        rows.append(
            {
                "feature": feature,
                "descriptor": descriptor,
                "scope": scope,
                "model": name,
                "display_name": result.spec.display_name,
                "n_features": result.n_features,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "pass_vs_reject_accuracy": pass_reject_accuracy(y_true, result.y_pred),
                "n_errors": int((y_true != result.y_pred).sum()),
                "cv_f1_macro": result.cv_score,
                **{f"f1_{label}": metrics["f1_per_class"][label] for label in CLASSES},
                "train_seconds": result.train_seconds,
                "extract_ms_per_image": extract_ms,
                "predict_ms_per_image": result.predict_ms_per_image,
                "bottles_per_second": 1000 / (extract_ms + result.predict_ms_per_image),
                "best_params": json.dumps(result.best_params),
            }
        )
    return pd.DataFrame(rows)


def comparison_matrix(comparison: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Feature x classifier matrix of one metric (rows and columns in run order)."""
    features = list(dict.fromkeys(comparison["feature"]))
    models = list(dict.fromkeys(comparison["model"]))
    matrix = comparison.pivot(index="feature", columns="model", values=metric)
    return matrix.loc[features, models]


def model_summary(
    result: TrainedModel, y_true: np.ndarray, extracted: ExtractedDataset
) -> dict[str, Any]:
    """Every metric of one trained model, JSON-serializable."""
    return {
        "display_name": result.spec.display_name,
        "best_params": result.best_params,
        "cv_f1_macro": result.cv_score,
        **result.spec.describe(result.model),
        "validation": classification_metrics(y_true, result.y_pred),
        "inspection_criteria": inspection_criteria_metrics(y_true, result.y_pred).to_dict(
            orient="records"
        ),
        "train_seconds": result.train_seconds,
        "predict_ms_per_image": result.predict_ms_per_image,
        "feature_importance_by_group": feature_importance_by_group(
            result.model, feature_group_sizes(extracted, result.feature)
        ),
    }


def build_feature_summary(
    feature: str,
    options: RunOptions,
    data: DatasetSplit,
    extracted: ExtractedDataset,
    trained: dict[ModelKey, TrainedModel],
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Every metric of the classifiers trained on one feature vector."""
    y_true = data.y[data.validation]
    group_sizes = feature_group_sizes(extracted, feature)
    return {
        "feature_vector": feature,
        "feature_groups": group_sizes,
        "dataset": dataset_summary(data),
        "photometric_normalization": options.normalize,
        "n_features": sum(group_sizes.values()),
        "extract_ms_per_image": extracted.extract_ms(FEATURE_VECTORS[feature]),
        "models": {
            name: model_summary(result, y_true, extracted)
            for (model_feature, name), result in trained.items()
            if model_feature == feature
        },
        "comparison": comparison[comparison["feature"] == feature].to_dict(orient="records"),
    }


def dataset_summary(data: DatasetSplit) -> dict[str, Any]:
    """Size, class distribution and train/validation duplicates of the dataset."""
    return {
        "train_images": len(data.train),
        "validation_images": len(data.validation),
        "distribution": class_distribution(data.samples),
        "train_val_identical_images": count_duplicate_images(data.samples),
    }


def build_summary(
    options: RunOptions,
    data: DatasetSplit,
    extracted: ExtractedDataset,
    feature_summaries: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Collects every metric of the run into a JSON-serializable dict."""
    return {
        "dataset": dataset_summary(data),
        "photometric_normalization": options.normalize,
        "prepare_ms_per_image": extracted.prepare_ms_per_image,
        "group_ms_per_image": extracted.group_ms_per_image,
        "bottle_localization": localization_error(data.samples, extracted.bottles),
        "auto_roi_bands": extracted.params.auto_bands(),
        "matrices": {
            metric: comparison_matrix(comparison, metric).to_dict(orient="index")
            for metric in MATRIX_METRICS
        },
        "features": {
            feature: {key: value for key, value in summary.items() if key != "dataset"}
            for feature, summary in feature_summaries.items()
        },
    }


# =============================================================================
# Entry point
# =============================================================================


def run(options: RunOptions) -> None:
    """Trains the selected classifiers on each selected feature vector and compares them."""
    np.random.seed(SEED)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data, extracted = prepare_data(options)
    save_preprocessing_figures(data, extracted, output_dir)

    specs = [MODELS[name] for name in options.model_names]
    trained: dict[ModelKey, TrainedModel] = {}
    for feature in options.feature_names:
        X = build_matrix(extracted.feature_sets, FEATURE_VECTORS[feature])
        feature_dir = output_dir / feature
        for spec in specs:
            result = train_model(spec, feature, X, data)
            save_model_outputs(result, data, extracted, feature_dir)
            trained[(feature, spec.name)] = result
        if options.didactic_plots:
            of_feature = {name: r for (f, name), r in trained.items() if f == feature}
            save_didactic_plots(feature, of_feature, X, data, extracted, feature_dir)

    comparison = build_comparison(trained, data, extracted)
    comparison.to_csv(output_dir / "feature_comparison.csv", index=False, float_format="%.4f")
    matrices = {metric: comparison_matrix(comparison, metric) for metric in MATRIX_METRICS}
    for metric, matrix in matrices.items():
        matrix.to_csv(output_dir / f"feature_matrix_{metric}.csv", float_format="%.4f")
    plots.plot_feature_matrix(matrices, output_dir / "feature_matrix.png")

    feature_summaries = {}
    for feature in options.feature_names:
        rows = comparison[comparison["feature"] == feature]
        rows.to_csv(output_dir / feature / "model_comparison.csv", index=False, float_format="%.4f")
        plots.plot_model_comparison(rows, output_dir / feature / "model_comparison.png", feature)
        summary = build_feature_summary(feature, options, data, extracted, trained, comparison)
        (output_dir / feature / "summary.json").write_text(json.dumps(summary, indent=2))
        feature_summaries[feature] = summary

    summary = build_summary(options, data, extracted, feature_summaries, comparison)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for metric, matrix in matrices.items():
        print(f"\n== {metric} per feature vector and classifier (Kaggle validation split)")
        print(matrix.round(4).to_string())
    print(f"\nResults in {output_dir.resolve()}")
