"""Experiment: sensitivity of each descriptor to its parameters, to choose their values.

This experiment varies the descriptor parameters (HOG orientations and cell size, LBP
neighbourhood and cell size, HSV histogram bins) one factor at a time around the reference
values (those of config.py) and measures each configuration only on the TRAINING
images, with the same procedure the pipeline uses to choose hyperparameters:

1. the 1200 training bottles are located and cropped once (preprocessing is not varied);
2. for every configuration of a descriptor, its feature groups are extracted and the three
   feature vectors of the descriptor are built (ad hoc ROIs, automatic regions, whole
   bottle);
3. every classifier runs its grid search with stratified 5-fold cross-validation on each
   vector; the score of the configuration is the mean cross-validated macro F1 of the nine
   (vector, classifier) pairs.

A value replaces the reference only when its score exceeds the reference score by more than
one standard error of the reference (so differences within the fold-to-fold noise change
nothing). When more than one factor of a descriptor changes, the combination is measured
too, and it is adopted only if it is not worse than the best single change. The validation
images are never read here: they stay untouched for the final evaluation of the pipeline.

Outputs, in `<results>/parameter_search/`:

    descriptor_search.csv        one row per (configuration, vector, classifier)
    descriptor_search_summary.csv  one row per configuration (mean score, size, time)
    descriptor_decisions.json    reference and chosen value of every factor, with the rule
    descriptor_search.jsonl      raw rows, appended as they finish (lets a run resume)

Usage (from the `code` folder)::

    .venv/bin/python experiments/descriptor_search.py --data ../archive --results results
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bottle_inspection import config  # noqa: E402
from bottle_inspection.env import default_data_dir  # noqa: E402
from bottle_inspection.dataset import load_dataset, split_indices  # noqa: E402
from bottle_inspection.extraction import bottle_view  # noqa: E402
from bottle_inspection.features import extract_group  # noqa: E402
from bottle_inspection.models import MODELS, grid_search, json_safe_params  # noqa: E402
from bottle_inspection.params import InspectionParams  # noqa: E402
from bottle_inspection.preprocessing import (  # noqa: E402
    BottleBox,
    PreparedImage,
    build_background,
    crop_regions,
    prepare_image,
    read_image,
)
from bottle_inspection.regions import learn_auto_bands  # noqa: E402

REFERENCE: str = "reference"
"""Name of the configuration with every parameter at its reference (config.py) value."""

COMBINED: str = "combined"
"""Name of the configuration that applies every adopted change of a descriptor at once."""

ROI_REGIONS: tuple[str, ...] = ("cap", "label", "body")
OTHER_REGIONS: tuple[str, ...] = ("label", "body", "auto", "bottle")
"""Regions whose cell size is varied together (the cap is the smallest region)."""

Change = dict[str, Any]
"""Fields of one parameter group replaced by a configuration, e.g. {"orientations": 12}."""


def cells(cap: int, others: int) -> Change:
    """Square cell of `cap` pixels on the cap and of `others` pixels on every other region."""
    change = {"cell_cap": (cap, cap)}
    change.update({f"cell_{region}": (others, others) for region in OTHER_REGIONS})
    return change


FACTORS: dict[str, dict[str, dict[str, Change]]] = {
    "hog": {
        "orientations": {
            "6": {},
            "9": {"orientations": 9},
            "12": {"orientations": 12},
            "18": {"orientations": 18},
        },
        "cell": {
            "8 px": cells(8, 8),
            "8 px (cap), 16 px": {},
            "16 px": cells(16, 16),
        },
    },
    "lbp": {
        "neighbourhood": {
            "P=8, R=1": {"points": 8, "radius": 1},
            "P=8, R=2": {"points": 8, "radius": 2},
            "P=16, R=2": {"points": 16, "radius": 2},
            "P=24, R=3": {},
            "P=24, R=4": {"points": 24, "radius": 4},
            "P=32, R=4": {"points": 32, "radius": 4},
        },
        "cell": {
            "8 px": cells(8, 8),
            "16 px": {},
            "32 px": cells(32, 32),
        },
    },
    "color": {
        "hsv_bins": {
            "9, 4, 4": {"hsv_bins": (9, 4, 4)},
            "18, 8, 8": {"hsv_bins": (18, 8, 8)},
            "36, 16, 16": {},
            "72, 32, 32": {"hsv_bins": (72, 32, 32)},
        },
    },
}
"""{descriptor: {factor: {value label: change}}}; the empty change is the reference value."""


def reference_label(descriptor: str, factor: str) -> str:
    """Label of the reference value of a factor."""
    return next(label for label, change in FACTORS[descriptor][factor].items() if not change)


def with_change(params: InspectionParams, descriptor: str, change: Change) -> InspectionParams:
    """A copy of `params` with fields of the descriptor's parameter group replaced."""
    group = getattr(params, descriptor)
    return dataclasses.replace(params, **{descriptor: dataclasses.replace(group, **change)})


def is_valid(params: InspectionParams, descriptor: str) -> bool:
    """True when every region has room for the cells (and HOG blocks) of the configuration."""
    for region, (width, height) in params.region_sizes().items():
        if descriptor == "hog":
            cell, block = params.hog.cells()[region], params.hog.block
            if width // cell[0] < block[0] or height // cell[1] < block[1]:
                return False
        if descriptor == "lbp":
            cell = params.lbp.cells()[region]
            if width // cell[0] < 1 or height // cell[1] < 1:
                return False
    return True


def prepare_training_bottles(
    data_dir: Path,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, InspectionParams]:
    """Region crops of every training bottle, their labels and the parameters with the
    automatic bands learned from them (the same steps as extraction.extract_dataset)."""
    samples = load_dataset(data_dir)
    train = split_indices(samples, "train")
    background = build_background([samples[i].image_path for i in train])
    base = InspectionParams.defaults()
    prepared = [
        prepare_image(read_image(samples[i].image_path), background, True, params=base)
        for i in train
    ]
    views = [bottle_view(image, base) for image in prepared]
    params = base.with_auto_bands(learn_auto_bands([view.crop for view in views]))
    whole = BottleBox(0.0, 0.0, 1.0, 1.0)
    crops = []
    for image, view in zip(prepared, views):
        # The automatic regions are slices of the BOTTLE_SIZE crop, as in the pipeline.
        regions = {name: image.rois[name] for name in (*ROI_REGIONS, config.WHOLE_BOTTLE)}
        auto = crop_regions(view.crop, whole, params=params)
        regions.update({name: auto[name] for name in config.AUTO_ROIS})
        crops.append(regions)
    labels = np.array([samples[i].label for i in train])
    return crops, labels, params


def image_groups(
    regions: dict[str, np.ndarray], groups: list[str], params: InspectionParams
) -> dict[str, np.ndarray]:
    """The given feature groups of one bottle."""
    prepared = PreparedImage(
        image=np.zeros((1, 1, 3), np.uint8),
        foreground_mask=np.zeros((1, 1), dtype=bool),
        box=BottleBox(0.0, 0.0, 1.0, 1.0),
        rois=regions,
    )
    return {group: extract_group(prepared, group, params=params) for group in groups}


def evaluate(
    descriptor: str,
    params: InspectionParams,
    crops: list[dict[str, np.ndarray]],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Cross-validated score of every (vector, classifier) pair of one configuration."""
    groups = [f"{region}_{descriptor}" for region in config.REGIONS]
    start = time.perf_counter()
    features = Parallel(n_jobs=-1)(delayed(image_groups)(c, groups, params) for c in crops)
    extract_ms = 1000 * (time.perf_counter() - start) / len(crops)
    rows = []
    for scope, regions in config.SCOPE_REGIONS.items():
        X = np.stack(
            [np.concatenate([f[f"{region}_{descriptor}"] for region in regions]) for f in features]
        )
        for name, spec in MODELS.items():
            start = time.perf_counter()
            search = grid_search(spec, X, labels)
            best = search.best_index_
            rows.append(
                {
                    "scope": scope,
                    "model": name,
                    "n_features": int(X.shape[1]),
                    "cv_f1_macro": float(search.best_score_),
                    "cv_f1_std": float(search.cv_results_["std_test_score"][best]),
                    "best_params": json.dumps(json_safe_params(search.best_params_)),
                    "train_seconds": time.perf_counter() - start,
                    "extract_ms_all_regions": extract_ms,
                }
            )
    return rows


class Search:
    """Runs configurations once and remembers their rows (also across interrupted runs)."""

    def __init__(self, folder: Path, crops: Any, labels: np.ndarray, params: InspectionParams):
        """Loads the rows already written to the raw file."""
        self.crops, self.labels, self.params = crops, labels, params
        self.raw = folder / "descriptor_search.jsonl"
        self.rows: list[dict[str, Any]] = []
        if self.raw.exists():
            self.rows = [json.loads(line) for line in self.raw.read_text().splitlines() if line]

    def run(self, descriptor: str, factor: str, value: str, change: Change) -> pd.DataFrame:
        """Rows of one configuration, computing them when they are not stored yet."""
        key = {"descriptor": descriptor, "factor": factor, "value": value}
        done = [row for row in self.rows if all(row[k] == v for k, v in key.items())]
        if not done:
            params = with_change(self.params, descriptor, change)
            if not is_valid(params, descriptor):
                raise ValueError(f"{descriptor} {factor}={value}: cells do not fit a region")
            print(f"== {descriptor}: {factor} = {value}", flush=True)
            done = [{**key, **row} for row in evaluate(descriptor, params, self.crops, self.labels)]
            with self.raw.open("a") as stream:
                for row in done:
                    stream.write(json.dumps(row) + "\n")
            self.rows.extend(done)
            frame = pd.DataFrame(done)
            print(f"   mean CV F1-macro {frame['cv_f1_macro'].mean():.4f}", flush=True)
        return pd.DataFrame(done)


def score(rows: pd.DataFrame) -> float:
    """Score of a configuration: mean cross-validated macro F1 of its nine pairs."""
    return float(rows["cv_f1_macro"].mean())


def standard_error(rows: pd.DataFrame) -> float:
    """Mean standard error of the nine cross-validated scores (fold std / sqrt(folds))."""
    return float((rows["cv_f1_std"] / np.sqrt(config.N_FOLDS)).mean())


def search_descriptor(search: Search, descriptor: str) -> dict[str, Any]:
    """Varies each factor of one descriptor and decides its value."""
    reference = search.run(descriptor, REFERENCE, REFERENCE, {})
    threshold = score(reference) + standard_error(reference)
    decision: dict[str, Any] = {
        "reference_score": score(reference),
        "reference_standard_error": standard_error(reference),
        "factors": {},
    }
    adopted: Change = {}
    best_single = score(reference)
    for factor, values in FACTORS[descriptor].items():
        scores = {}
        for label, change in values.items():
            rows = search.run(descriptor, factor, label, change) if change else reference
            scores[label] = score(rows)
        current = reference_label(descriptor, factor)
        best = max(scores, key=lambda label: scores[label])
        chosen = best if scores[best] > threshold else current
        if chosen != current:
            adopted.update(values[chosen])
            best_single = max(best_single, scores[chosen])
        decision["factors"][factor] = {
            "reference": current,
            "scores": scores,
            "best": best,
            "chosen": chosen,
            "changed": chosen != current,
        }
    changed = [f for f, d in decision["factors"].items() if d["changed"]]
    if len(changed) > 1:
        combined = score(search.run(descriptor, COMBINED, COMBINED, adopted))
        decision["combined_score"] = combined
        if combined < best_single:
            # The changes do not add up: keep only the single change with the best score.
            keep = max(
                changed,
                key=lambda f: decision["factors"][f]["scores"][decision["factors"][f]["chosen"]],
            )
            for factor in changed:
                if factor != keep:
                    item = decision["factors"][factor]
                    item["chosen"], item["changed"] = item["reference"], False
            adopted = dict(FACTORS[descriptor][keep][decision["factors"][keep]["chosen"]])
    decision["adopted_change"] = {
        k: list(v) if isinstance(v, tuple) else v for k, v in adopted.items()
    }
    return decision


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: score, spread, vector sizes and cost."""
    grouped = rows.groupby(["descriptor", "factor", "value"], sort=False)
    summary = grouped.agg(
        cv_f1_macro=("cv_f1_macro", "mean"),
        cv_f1_min=("cv_f1_macro", "min"),
        cv_f1_std=("cv_f1_std", "mean"),
        n_features_min=("n_features", "min"),
        n_features_max=("n_features", "max"),
        train_seconds=("train_seconds", "sum"),
        extract_ms_all_regions=("extract_ms_all_regions", "first"),
    )
    for model in MODELS:
        summary[f"cv_f1_{model}"] = (
            rows[rows["model"] == model]
            .groupby(["descriptor", "factor", "value"], sort=False)["cv_f1_macro"]
            .mean()
        )
    for scope in config.SCOPE_REGIONS:
        summary[f"cv_f1_{scope}"] = (
            rows[rows["scope"] == scope]
            .groupby(["descriptor", "factor", "value"], sort=False)["cv_f1_macro"]
            .mean()
        )
    return summary.reset_index()


def main() -> None:
    """Runs the search for every descriptor and writes the tables and the decisions."""
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
    np.random.seed(config.SEED)

    print("== Locating and cropping the training bottles", flush=True)
    crops, labels, params = prepare_training_bottles(args.data)
    search = Search(folder, crops, labels, params)
    decisions = {
        "selection": "mean 5-fold cross-validated macro F1 over the 9 (vector, classifier) "
        "pairs of the descriptor, training images only",
        "rule": "a value replaces the reference only if its score exceeds the reference "
        "score by more than one standard error of the reference",
        "training_images": len(labels),
        "auto_bands": [list(band) for band in params.auto_rois.bands],
        "descriptors": {d: search_descriptor(search, d) for d in config.DESCRIPTORS},
    }
    rows = pd.DataFrame(search.rows)
    rows.to_csv(folder / "descriptor_search.csv", index=False, float_format="%.5f")
    summarize(rows).to_csv(
        folder / "descriptor_search_summary.csv", index=False, float_format="%.5f"
    )
    (folder / "descriptor_decisions.json").write_text(json.dumps(decisions, indent=2))
    print(json.dumps({d: v["factors"] for d, v in decisions["descriptors"].items()}, indent=1))
    print(f"\nResults in {folder.resolve()}")


if __name__ == "__main__":
    main()
