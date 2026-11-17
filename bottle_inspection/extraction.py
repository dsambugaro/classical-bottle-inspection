"""Dataset-level feature extraction, with an on-disk cache.

Extraction is the slow part of a run, so the result for the whole dataset
is cached in `<output>/feature_cache.joblib` and reused by later runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from .config import FEATURE_GROUPS, WHOLE_BOTTLE
from .dataset import Sample
from .features import FeatureSet, extract_features, extract_group
from .params import InspectionParams, resolve
from .preprocessing import (
    BottleBox,
    PreparedImage,
    build_background,
    crop_regions,
    prepare_image,
    read_image,
)
from .regions import learn_auto_bands

PROGRESS_EVERY: int = 200
"""Print a progress line every this many images."""


@dataclass
class BottleView:
    """What figures and the evaluation need to know about one detected bottle."""

    box: BottleBox
    """Detected bottle box (relative image coordinates)."""
    crop: np.ndarray
    """Whole-bottle BGR crop at BOTTLE_SIZE."""

    @property
    def center_x(self) -> float:
        """Relative horizontal center of the bottle."""
        return self.box.center_x

    @property
    def top_y(self) -> float:
        """Relative height of the top of the bottle."""
        return self.box.y0


@dataclass
class ExtractedDataset:
    """Features of every dataset image plus the artifacts needed to reproduce them."""

    background: np.ndarray
    params: InspectionParams
    """Parameters used, including the automatic region bands learned from the training images."""
    feature_sets: list[FeatureSet]
    bottles: list[BottleView]
    prepare_ms_per_image: float
    """Mean preprocessing time (normalization, bottle detection and crops)."""
    group_ms_per_image: dict[str, float]
    """Mean time to compute each feature group."""
    normalize: bool

    def extract_ms(self, groups: list[str]) -> float:
        """Mean time per image to preprocess it and compute only the given groups."""
        return self.prepare_ms_per_image + sum(self.group_ms_per_image[g] for g in groups)

    @property
    def extract_ms_per_image(self) -> float:
        """Mean time per image to preprocess it and compute every feature group."""
        return self.extract_ms(list(self.group_ms_per_image))


def bottle_view(prepared: PreparedImage, params: InspectionParams | None = None) -> BottleView:
    """The detected bottle of a preprocessed image, with its crop at BOTTLE_SIZE."""
    crop = cv2.resize(prepared.rois[WHOLE_BOTTLE], resolve(params).bottle.crop_size)
    return BottleView(prepared.box, crop)


def extract_image(
    image: np.ndarray,
    background: np.ndarray,
    normalize: bool,
    *,
    params: InspectionParams | None = None,
) -> tuple[FeatureSet, BottleView]:
    """Preprocesses one image and computes its features.

    `params` defaults to the constants of config.py, which have no automatic region
    bands; pass the parameters of the extracted dataset (or of a trained model) to
    also get the groups of the automatic regions.
    """
    prepared = prepare_image(image, background, normalize, params=params)
    return extract_features(prepared, params=params), bottle_view(prepared, params)


def timed_groups(
    prepared: PreparedImage,
    groups: list[str],
    params: InspectionParams,
    seconds: dict[str, float],
) -> FeatureSet:
    """Computes the given feature groups, adding the time of each one to `seconds`."""
    features = {}
    for group in groups:
        start = time.perf_counter()
        features[group] = extract_group(prepared, group, params=params)
        seconds[group] += time.perf_counter() - start
    return features


def extract_dataset(
    samples: list[Sample], train_indices: list[int], normalize: bool
) -> ExtractedDataset:
    """Builds the background, learns the automatic regions and extracts features of all images.

    The background and the automatic regions use training images only. The steps of
    extract_image are timed one by one, so the cost of each feature vector can be
    reported separately.
    """
    print("== Building the background model (training images only)")
    background = build_background([samples[i].image_path for i in train_indices])
    base = InspectionParams.defaults()
    base_groups = base.feature_groups()

    print("== Detecting the bottles and extracting the features of the ROIs and of the bottle")
    feature_sets, bottles = [], []
    prepare_seconds = 0.0
    group_seconds = dict.fromkeys(FEATURE_GROUPS, 0.0)
    for count, sample in enumerate(samples, start=1):
        image = read_image(sample.image_path)
        start = time.perf_counter()
        prepared = prepare_image(image, background, normalize, params=base)
        prepare_seconds += time.perf_counter() - start
        feature_sets.append(timed_groups(prepared, base_groups, base, group_seconds))
        bottles.append(bottle_view(prepared, base))
        if count % PROGRESS_EVERY == 0:
            print(f"   {count}/{len(samples)}")

    print("== Learning the automatic regions (training bottles only)")
    bands = learn_auto_bands([bottles[i].crop for i in train_indices])
    params = base.with_auto_bands(bands)
    print("   bands inside the bottle box: " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in bands))
    auto_groups = [g for g in params.feature_groups() if g not in base_groups]
    whole = BottleBox(0.0, 0.0, 1.0, 1.0)
    for features, view in zip(feature_sets, bottles):
        # The automatic regions are slices of the BOTTLE_SIZE crop, which is already
        # stored, so the images do not have to be read again.
        prepared = PreparedImage(
            image=view.crop,
            foreground_mask=np.zeros((1, 1), dtype=bool),
            box=view.box,
            rois=crop_regions(view.crop, whole, params=params),
        )
        features.update(timed_groups(prepared, auto_groups, params, group_seconds))

    per_image = 1000 / len(samples)
    return ExtractedDataset(
        background=background,
        params=params,
        feature_sets=[{g: features[g] for g in FEATURE_GROUPS} for features in feature_sets],
        bottles=bottles,
        prepare_ms_per_image=prepare_seconds * per_image,
        group_ms_per_image={g: seconds * per_image for g, seconds in group_seconds.items()},
        normalize=normalize,
    )


def cache_is_current(cached: object, n_samples: int, normalize: bool) -> bool:
    """True when a cached extraction was made with the current parameters and groups.

    A cache written before a parameter was added holds objects without that field; reading
    it raises AttributeError, which also means the cache is outdated.
    """
    try:
        return (
            isinstance(cached, ExtractedDataset)
            and len(cached.feature_sets) == n_samples
            and cached.normalize == normalize
            and list(cached.feature_sets[0]) == FEATURE_GROUPS
            and cached.params.with_auto_bands(()) == InspectionParams.defaults()
        )
    except AttributeError:
        return False


def load_or_extract(
    samples: list[Sample],
    train_indices: list[int],
    normalize: bool,
    cache_path: Path,
    recompute: bool,
) -> ExtractedDataset:
    """Returns cached features when valid; otherwise extracts and caches them.

    The cache is ignored when `recompute` is set, when it holds a different number
    of images or other feature groups, or when it
    was built with another normalization setting or other parameters.
    """
    if cache_path.exists() and not recompute:
        cached = joblib.load(cache_path)
        if cache_is_current(cached, len(samples), normalize):
            print(f"== Loading features from {cache_path}")
            return cached
        print("== Feature cache is outdated; extracting again")
    extracted = extract_dataset(samples, train_indices, normalize)
    joblib.dump(extracted, cache_path, compress=3)
    return extracted
