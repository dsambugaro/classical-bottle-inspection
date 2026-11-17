"""Loading of the Kaggle dataset, using its original train/val split as is.

Expected layout (YOLO format)::

    <root>/images/{train,val}/*.jpg
    <root>/labels/{train,val}/*.txt   one "class_id cx cy w h" line per box
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import CLASSES, FILE_PREFIX_TO_CLASS, OK_CLASS, YOLO_CLASS_NAMES

SPLITS: tuple[str, str] = ("train", "val")


@dataclass(frozen=True)
class YoloBox:
    """One annotated bounding box, in relative coordinates."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class Sample:
    """One dataset image and its image-level class."""

    image_path: Path
    label_path: Path
    split: str
    label: str


def read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    """Parses a YOLO label file into a list of boxes."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            class_id, *coords = line.split()
            boxes.append(YoloBox(int(class_id), *(float(c) for c in coords)))
    return boxes


def image_label(boxes: list[YoloBox]) -> str:
    """Converts the YOLO boxes of an image into a single image-level class.

    A defective image has two boxes (the bottle and the defect) and gets the
    defect class. Without a defect box, it gets the bottle class.

    Raises:
        ValueError: if the image has more than one defect or bottle-brand box type.
    """
    names = {YOLO_CLASS_NAMES[box.class_id] for box in boxes}
    non_ok = sorted(names - {OK_CLASS})
    if len(non_ok) > 1:
        raise ValueError(f"more than one non-OK class in the same image: {non_ok}")
    return non_ok[0] if non_ok else OK_CLASS


def check_file_prefix(image_path: Path, label: str) -> None:
    """Checks that the file name prefix (e.g. "no_cap_0012_...") matches the label.

    Raises:
        ValueError: if the prefix and the annotated label disagree.
    """
    prefix = image_path.stem.rsplit("_", 3)[0]
    if FILE_PREFIX_TO_CLASS.get(prefix) != label:
        raise ValueError(f"{image_path.name}: prefix '{prefix}' != label '{label}'")


def load_dataset(root: Path) -> list[Sample]:
    """Loads every train and val image of the dataset, in file name order."""
    samples = []
    for split in SPLITS:
        for image_path in sorted((root / "images" / split).glob("*.jpg")):
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            label = image_label(read_yolo_boxes(label_path))
            check_file_prefix(image_path, label)
            samples.append(Sample(image_path, label_path, split, label))
    return samples


def split_indices(samples: list[Sample], split: str) -> list[int]:
    """Returns the indices of the samples that belong to the given split."""
    return [i for i, sample in enumerate(samples) if sample.split == split]


def class_distribution(samples: list[Sample]) -> dict[str, dict[str, int]]:
    """Counts images per split and class: {split: {class: count}}."""
    distribution = {split: {name: 0 for name in CLASSES} for split in SPLITS}
    for sample in samples:
        distribution[sample.split][sample.label] += 1
    return distribution


def count_duplicate_images(samples: list[Sample]) -> int:
    """Counts byte-identical images present in both train and val (MD5 hash)."""
    hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    for sample in samples:
        hashes[sample.split].add(hashlib.md5(sample.image_path.read_bytes()).hexdigest())
    return len(hashes["train"] & hashes["val"])
