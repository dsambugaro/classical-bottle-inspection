"""Dataset samples with ground truth (REQ-I2), read with the original Kaggle split."""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any

from bottle_inspection.dataset import SPLITS, Sample, class_distribution, load_dataset

from .errors import ApiError

_cache: dict[Path, list[Sample]] = {}
_lock = threading.Lock()


def samples_of(root: Path) -> list[Sample]:
    """Every dataset sample (train and val), cached per dataset root."""
    with _lock:
        if root not in _cache:
            _cache[root] = load_dataset(root)
        return _cache[root]


def dataset_info(root: Path | None) -> dict[str, dict[str, int]] | None:
    """Class counts per split, or None when the dataset cannot be read."""
    if root is None:
        return None
    try:
        return class_distribution(samples_of(root))
    except (OSError, ValueError):
        return None


def sample_json(sample: Sample) -> dict[str, Any]:
    """JSON form of a sample."""
    name = sample.image_path.name
    return {
        "name": name,
        "split": sample.split,
        "ref": f"{sample.split}/{name}",
        "label": sample.label,
        "url": f"/api/dataset/images/{sample.split}/{name}",
        "thumb_url": f"/api/dataset/images/{sample.split}/{name}?thumb=1",
    }


def list_samples(
    root: Path,
    split: str = "val",
    n: int = 12,
    class_name: str | None = None,
    shuffle: bool = True,
    offset: int = 0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Samples of one split, optionally of one class, random or in file order."""
    if split not in SPLITS:
        raise ApiError(422, "dataset.invalid_split", "invalid split", {"splits": list(SPLITS)})
    chosen = [s for s in samples_of(root) if s.split == split]
    if class_name:
        chosen = [s for s in chosen if s.label == class_name]
    total = len(chosen)
    if shuffle:
        chosen = random.Random(seed).sample(chosen, min(n, total))
    else:
        chosen = chosen[offset : offset + n]
    return {"total": total, "items": [sample_json(s) for s in chosen]}


def find_sample(root: Path, ref: str) -> Sample:
    """The sample `<split>/<file name>` (only names listed by the dataset are accepted)."""
    split, _, name = ref.partition("/")
    for sample in samples_of(root):
        if sample.split == split and sample.image_path.name == name:
            return sample
    raise ApiError(404, "dataset.sample_not_found", "dataset sample not found", {"ref": ref})
