"""Command line interface.

Examples::

    python -m bottle_inspection --data ../archive --output results
    python -m bottle_inspection --models svm --features rois_hog bottle_hog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import FEATURE_VECTORS
from .env import default_data_dir
from .models import MODELS
from .pipeline import RunOptions, run


def parse_args(argv: list[str] | None = None) -> RunOptions:
    """Parses the command line into RunOptions."""
    parser = argparse.ArgumentParser(
        prog="bottle_inspection",
        description="Bottle defect classification on the Kaggle Ron 88 dataset with classical classifiers.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data_dir(),
        help="dataset root folder (default: BOTTLE_DATA, see .env.example)",
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="results folder")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="classifiers to train and compare (default: all)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=list(FEATURE_VECTORS),
        default=list(FEATURE_VECTORS),
        metavar="FEATURE",
        help="feature vectors classified separately, <scope>_<descriptor> with scope in "
        "rois/auto_rois/bottle/wide/image and descriptor in hog/lbp/color (default: all)",
    )
    parser.add_argument("--recompute", action="store_true", help="ignore the feature cache")
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="disable the photometric normalization by the reference patch",
    )
    parser.add_argument(
        "--didactic-plots",
        action="store_true",
        help="also draw the real decision of each classifier for every validation image",
    )
    args = parser.parse_args(argv)
    return RunOptions(
        data_dir=args.data,
        output_dir=args.output,
        model_names=args.models,
        feature_names=args.features,
        recompute=args.recompute,
        normalize=not args.no_normalization,
        didactic_plots=args.didactic_plots,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point of `python -m bottle_inspection`."""
    run(parse_args(argv))
