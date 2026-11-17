"""Experiment: how the automatic regions change with the number of bands and with the
images they are learned from.

Four variants are compared, each with the three descriptors and the three classifiers:

==============  =====  ==========================================
variant         bands  learned from
==============  =====  ==========================================
3_all           3      every training bottle (the pipeline default)
5_all           5      every training bottle
3_good          3      training bottles without defects only
5_good          5      training bottles without defects only
==============  =====  ==========================================

The bottle crops come from the feature cache of a finished run, so the images are not
read again. Nothing of that run is changed; the outputs go to
`<results>/auto_region_variants/`.

Usage (from the `code` folder)::

    .venv/bin/python experiments/auto_region_variants.py --data ../archive --results results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bottle_inspection import config  # noqa: E402
from bottle_inspection.env import default_data_dir  # noqa: E402
from bottle_inspection.dataset import load_dataset, split_indices  # noqa: E402
from bottle_inspection.extraction import ExtractedDataset  # noqa: E402
from bottle_inspection.features import DESCRIPTOR_FUNCTIONS  # noqa: E402
from bottle_inspection.models import MODELS, grid_search  # noqa: E402
from bottle_inspection.params import InspectionParams  # noqa: E402
from bottle_inspection.regions import Range, learn_auto_bands, mean_bottle  # noqa: E402

MAX_BANDS: int = 5
"""Largest number of automatic regions tested."""

VARIANTS: dict[str, tuple[int, bool]] = {
    "3_all": (3, False),
    "5_all": (5, False),
    "3_good": (3, True),
    "5_good": (5, True),
}
"""{variant: (number of bands, learn from the bottles without defects only)}."""

VARIANT_TITLES: dict[str, str] = {
    "3_all": "3 faixas\ntodas as garrafas",
    "5_all": "5 faixas\ntodas as garrafas",
    "3_good": "3 faixas\nsó garrafas boas",
    "5_good": "5 faixas\nsó garrafas boas",
}

BAND_COLORS_RGB: list[tuple[int, int, int]] = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (255, 127, 0),
    (152, 78, 163),
]
"""One color per band, from the top of the bottle to the bottom."""

ROI_COLOR_RGB: tuple[int, int, int] = (255, 255, 255)
"""Color of the ad hoc ROIs drawn for reference."""


def allow_more_auto_regions(count: int) -> None:
    """Registers the names auto_4.. so the package accepts more than AUTO_ROI_COUNT bands.

    The lists of config.py are extended in place, because the other modules hold
    references to them. FEATURE_GROUPS and FEATURE_VECTORS are left as they are: this
    script builds its own vectors.
    """
    for k in range(len(config.AUTO_ROIS), count):
        name = f"auto_{k + 1}"
        config.AUTO_ROIS.append(name)
        config.REGIONS.insert(config.REGIONS.index(config.WHOLE_BOTTLE), name)


def region_features(crops: list[np.ndarray], bands: tuple[Range, ...]) -> dict[str, np.ndarray]:
    """One matrix per descriptor: the descriptor of every band of every crop, concatenated."""
    params = InspectionParams.defaults().with_auto_bands(bands)
    rows = crops[0].shape[0]
    slices = {
        name: (round(y0 * rows), round(y1 * rows)) for name, (y0, y1) in params.auto_bands().items()
    }
    matrices = {}
    for descriptor, function in DESCRIPTOR_FUNCTIONS.items():
        matrices[descriptor] = np.stack(
            [
                np.concatenate(
                    [
                        function(crop[top:bottom], name, params)
                        for name, (top, bottom) in slices.items()
                    ]
                )
                for crop in crops
            ]
        )
    return matrices


def draw_bands(image: np.ndarray, bands: tuple[Range, ...], scale: int = 3) -> np.ndarray:
    """An enlarged RGB copy of a bottle crop with one colored rectangle per band."""
    rgb = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    height, width = rgb.shape[:2]
    for color, (y0, y1) in zip(BAND_COLORS_RGB, bands):
        top, bottom = round(y0 * height), round(y1 * height) - 1
        cv2.rectangle(rgb, (2, top + 2), (width - 3, bottom - 2), color, 3)
    return rgb


def draw_rois(image: np.ndarray, scale: int = 3) -> np.ndarray:
    """An enlarged RGB copy of a bottle crop with the ad hoc ROIs, for reference."""
    rgb = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    height, width = rgb.shape[:2]
    for k, (name, (y0, y1)) in enumerate(config.ROIS.items()):
        top, bottom = round(y0 * height), round(y1 * height) - 1
        cv2.rectangle(rgb, (2 + 6 * k, top + 2), (width - 3 - 6 * k, bottom - 2), ROI_COLOR_RGB, 2)
        cv2.putText(
            rgb, name, (10 + 6 * k, top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR_RGB, 2
        )
    return rgb


def plot_bands(
    means: dict[bool, np.ndarray],
    example: np.ndarray,
    bands: dict[str, tuple[Range, ...]],
    path: Path,
) -> None:
    """Saves the bands of every variant over the mean bottle they were learned from and
    over one bottle without defects, next to the ad hoc ROIs."""
    fig, axes = plt.subplots(2, len(VARIANTS) + 1, figsize=(3.0 * (len(VARIANTS) + 1), 15))
    for row, label in enumerate(["garrafa média usada no aprendizado", "uma garrafa sem defeito"]):
        axes[row, 0].imshow(draw_rois(means[False] if row == 0 else example))
        axes[row, 0].set_title("ROIs manuais\n(referência)", fontsize=11)
        axes[row, 0].set_ylabel(label, fontsize=11)
        for ax, (variant, (_count, good_only)) in zip(axes[row, 1:], VARIANTS.items()):
            image = means[good_only] if row == 0 else example
            ax.imshow(draw_bands(image, bands[variant]))
            ax.set_title(VARIANT_TITLES[variant], fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Regiões automáticas: número de faixas × imagens usadas no aprendizado", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_dir(),
        help="dataset root (default: BOTTLE_DATA, see .env.example)",
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--bands-only", action="store_true", help="learn and draw the bands only")
    args = parser.parse_args()
    output = args.results / "auto_region_variants"
    output.mkdir(exist_ok=True)

    samples = load_dataset(args.data)
    train, validation = split_indices(samples, "train"), split_indices(samples, "val")
    y = np.array([sample.label for sample in samples])
    extracted: ExtractedDataset = joblib.load(args.results / "feature_cache.joblib")
    assert len(extracted.bottles) == len(samples)
    crops = [view.crop for view in extracted.bottles]
    allow_more_auto_regions(MAX_BANDS)

    learn_from = {
        False: train,
        True: [i for i in train if samples[i].label == config.OK_CLASS],
    }
    print(f"training bottles: {len(learn_from[False])} (without defects: {len(learn_from[True])})")
    bands = {
        variant: learn_auto_bands([crops[i] for i in learn_from[good_only]], n_bands=count)
        for variant, (count, good_only) in VARIANTS.items()
    }
    for variant, variant_bands in bands.items():
        print(f"{variant:7s} " + "  ".join(f"{a:.3f}-{b:.3f}" for a, b in variant_bands))
    assert bands["3_all"] == extracted.params.auto_rois.bands, "3_all must match the cached run"
    (output / "bands.json").write_text(
        json.dumps(
            {
                "rois": config.ROIS,
                "learned_from": {
                    "all": len(learn_from[False]),
                    "good": len(learn_from[True]),
                },
                "bands": bands,
            },
            indent=2,
        )
    )
    means = {
        good_only: mean_bottle([crops[i] for i in indices])
        for good_only, indices in learn_from.items()
    }
    example = crops[next(i for i in validation if samples[i].label == config.OK_CLASS)]
    plot_bands(means, example, bands, output / "bands.png")
    if args.bands_only:
        return

    rows = []
    for variant, variant_bands in bands.items():
        start = time.perf_counter()
        matrices = region_features(crops, variant_bands)
        print(f"\n== {variant}: features extracted in {time.perf_counter() - start:.0f} s")
        for descriptor, X in matrices.items():
            for spec in MODELS.values():
                search = grid_search(spec, X[train], y[train])
                y_pred = search.best_estimator_.predict(X[validation])
                row = {
                    "variant": variant,
                    "n_bands": VARIANTS[variant][0],
                    "learned_from": "good" if VARIANTS[variant][1] else "all",
                    "descriptor": descriptor,
                    "model": spec.name,
                    "n_features": X.shape[1],
                    "accuracy": accuracy_score(y[validation], y_pred),
                    "f1_macro": f1_score(y[validation], y_pred, average="macro"),
                    "cv_f1_macro": search.best_score_,
                    "errors": int((y_pred != y[validation]).sum()),
                }
                rows.append(row)
                print(
                    f"   {descriptor:5s} {spec.name:13s} acc={row['accuracy']:.4f} "
                    f"f1={row['f1_macro']:.4f} errors={row['errors']}"
                )
                pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)

    comparison = pd.DataFrame(rows)
    for metric in ("accuracy", "errors"):
        matrix = comparison.pivot_table(
            index=["descriptor", "model"], columns="variant", values=metric, sort=False
        )[list(VARIANTS)]
        matrix.to_csv(output / f"matrix_{metric}.csv")
        print(f"\n{metric}\n{matrix.to_string()}")


if __name__ == "__main__":
    main()
