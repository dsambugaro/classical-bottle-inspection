"""Central configuration: every constant of the pipeline lives here.

The bottle is found by background subtraction: its position and its height are measured
on the foreground mask. Two things are chosen by hand: the width of the bottle box
(BOTTLE_WIDTH) and where the regions of interest (ROIs) lie inside the box; the ROIs have
an automatic alternative (regions learned from the training images). The classifiers only see generic descriptors (HOG,
LBP, HSV color histogram).

Image coordinates are relative (0..1) to the image size and ROI coordinates are
relative to the detected bottle box, so they do not depend on the resolution.
"""

from __future__ import annotations

SEED: int = 42

# =============================================================================
# Classes (exactly the 7 classes of the Kaggle data.yaml)
# =============================================================================

YOLO_CLASS_NAMES: list[str] = [
    "bottle_no_defect",
    "bottle_other_brand",
    "defect_low_fill",
    "defect_no_cap",
    "defect_loose_cap",
    "defect_debris",
    "defect_label_damage",
]
"""Class names in YOLO id order (id 0 = bottle_no_defect, ..., id 6 = defect_label_damage).
Id 0 is called "bottle_ron88" in the Kaggle data.yaml; it is renamed here to make clear
that it is the bottle without defects."""

CLASSES: list[str] = list(YOLO_CLASS_NAMES)
"""Classes predicted by the classifiers. An image gets the class of its annotated
defect; without a defect it gets the bottle class."""

OK_CLASS: str = "bottle_no_defect"
"""The only class that passes the inspection (a Ron 88 bottle without defects)."""

BOTTLE_CLASS_IDS: tuple[int, ...] = (0, 1)
"""YOLO ids whose boxes enclose the whole bottle (used to validate localization)."""

FILE_PREFIX_TO_CLASS: dict[str, str] = {
    "good": "bottle_no_defect",
    "wrong_bottle": "bottle_other_brand",
    "underfilled": "defect_low_fill",
    "no_cap": "defect_no_cap",
    "loose_cap": "defect_loose_cap",
    "debris": "defect_debris",
    "damaged_label": "defect_label_damage",
}
"""File name prefix of each class, used only as a consistency check of the labels."""

INSPECTION_CRITERIA: dict[str, list[str]] = {
    "fill_level": ["defect_low_fill"],
    "cap": ["defect_no_cap", "defect_loose_cap"],
    "label": ["defect_label_damage"],
    "contamination": ["defect_debris"],
    "brand": ["bottle_other_brand"],
}
"""Binary pass/reject view of the 7 classes, grouped by inspection type."""

# =============================================================================
# Background model and bottle localization
# =============================================================================

LOCALIZATION_SIZE: tuple[int, int] = (640, 360)
"""(width, height) used only to locate the bottle; smaller is faster."""

N_BACKGROUND_IMAGES: int = 200
"""Number of training images whose per-pixel median forms the background."""

NOISE_FLOOR: int = 20
"""Smallest BGR difference that can count as foreground. Otsu's method always splits the
differences into two groups, so in an empty scene it would split the sensor noise; a
threshold below this floor is raised to it, which leaves the mask of an empty scene empty.
On the dataset the Otsu threshold is never below 44, so the floor changes no image."""

MASK_CLOSE_SIZE: int = 15
"""Diameter (pixels at LOCALIZATION_SIZE) of the morphological closing of the foreground
mask. The bottle is transparent, so its mask is full of holes; closing joins them."""

MASK_OPEN_SIZE: int = 5
"""Diameter of the morphological opening applied after the closing. It removes thin
lines and specks (e.g. a table edge that moved one pixel between captures)."""

PROFILE_SMOOTHING: int = 31
"""Width of the moving average applied to the column profile of the mask. The mask of a
transparent bottle can split into two vertical halves; a narrow average leaves a valley
between them that falls below the column fraction, and the box then covers half a bottle.
With 31 columns the valley is filled: on the training images no box is left with an IoU
below 0.6 against the annotation (5 with 11 columns) and the mean IoU rises from 0.857 to
0.889 (experiments/localization_sensitivity.py). Wider averages (41, 51) raise the mean IoU a
little more by widening every box; 31 was kept as the smallest value that removes the
split boxes."""

COLUMN_PROFILE_FRACTION: float = 0.4
"""The bottle is searched in the columns around the peak of the smoothed column profile
whose foreground count stays above this fraction of the peak. Its centre is the middle of
the first and the last column of that run that reach the same fraction of the unsmoothed
profile."""

BOTTLE_WIDTH: float = 0.145
"""Width of the bottle box, as a fraction of the image width, centred on the bottle found
in the column profile. It is an ad hoc value: the bottles of the dataset have the same
width and stand at the same distance, and the edges of a transparent bottle are the
weakest part of the mask. Taking the width from the mask (every edge from the thresholds)
gave a mean IoU of at most 0.920 with the annotated boxes of the training images; this
fixed width, with the centre and the height still measured on the mask, gives 0.936
(experiments/bottle_box_methods.py). A fixed height was worse than the measured one,
because bottles without cap, with a raised cap or of another brand differ in height."""

TOP_PROFILE_FRACTION: float = 0.3
"""Inside those columns, the top of the bottle is the first row whose foreground count
reaches this fraction of the densest row. The neck and the cap are narrower than the
body, and a transparent neck is mostly holes, so their rows are far from the densest one:
with 0.5 the box started below the cap or the neck in 18 training images (by up to 129
pixels of the 720); 0.3 is the highest value with no top cut above 15 pixels."""

ROW_PROFILE_FRACTION: float = 0.5
"""Inside those columns, the bottom of the bottle is the last row whose foreground count
reaches this fraction of the densest row. The shadow of the bottle on the table lies below
it, so a lower fraction pulls the box down into the shadow; 0.5 gave the highest overlap
with the annotated boxes (experiments/localization_sensitivity.py)."""

# =============================================================================
# Photometric normalization
# =============================================================================

NORMALIZE_LIGHTING: bool = True
"""When True, each image is rescaled so that the reference patch matches the background."""

REFERENCE_PATCH: tuple[float, float, float, float] = (0.80, 0.02, 0.97, 0.25)
"""(x0, y0, x1, y1) of a wall region never covered by a bottle."""

# =============================================================================
# Bottle crop and regions of interest (ROIs)
# =============================================================================

BOTTLE_SIZE: tuple[int, int] = (96, 384)
"""(width, height) the detected bottle box is resized to (region "bottle" of the
descriptors, source of the automatic regions and crop shown in the figures)."""

ROIS: dict[str, tuple[float, float]] = {
    "cap": (0.00, 0.12),
    "label": (0.20, 0.52),
    "body": (0.12, 1.00),
}
"""Ad hoc ROIs: vertical band (y0, y1) of each ROI inside the detected bottle box
(0 = top of the bottle, 1 = bottom). These are the only hand-picked positions."""

ROI_SIZE: dict[str, tuple[int, int]] = {
    "cap": (64, 48),
    "label": (64, 96),
    "body": (48, 192),
}
"""Canonical (width, height) each ROI is resized to before computing HOG and LBP."""

AUTO_ROI_COUNT: int = 3
"""Number of automatic regions (the same number as the ad hoc ROIs, to compare them).
Their positions are not configured: they are learned from the training images."""

AUTO_ROI_UNIT: int = 16
"""Automatic region boundaries are multiples of this many rows of the BOTTLE_SIZE crop
(the HOG/LBP cell size, so that the cells of a region never straddle a boundary)."""

AUTO_ROIS: list[str] = [f"auto_{k + 1}" for k in range(AUTO_ROI_COUNT)]
"""Names of the automatic regions, from the top of the bottle to the bottom."""

WHOLE_BOTTLE: str = "bottle"
"""Name of the region that covers the whole detected bottle box (at BOTTLE_SIZE)."""

WIDE_AREA: str = "wide"
"""Name of the region that covers a large fixed area of the image around the bottle."""

WIDE_AREA_RECT: tuple[float, float, float, float] = (0.20, 0.02, 0.78, 0.97)
"""(x0, y0, x1, y1) of the wide area, relative to the image. It is fixed by hand and does not
use the detected bottle box: it is about as wide as three or four bottles and holds every
bottle of the dataset (the detected boxes lie between x = 0.22 and 0.72 and between
y = 0.05 and 0.92). Test of how the descriptors behave when much of the crop is background."""

WIDE_AREA_SIZE: tuple[int, int] = (256, 240)
"""(width, height) the wide area is resized to; about its aspect ratio at 1280x720 (742x684)."""

WHOLE_IMAGE: str = "image"
"""Name of the region that covers the whole image (no localization, no crop)."""

WHOLE_IMAGE_SIZE: tuple[int, int] = (448, 256)
"""(width, height) the whole image is resized to. Same scale as the wide area (about 0.345 of
1280x720), so the bottle has the same size in pixels in both and only the amount of
background changes."""

REGIONS: list[str] = [*ROIS, *AUTO_ROIS, WHOLE_BOTTLE, WIDE_AREA, WHOLE_IMAGE]
"""Regions a descriptor is computed on: the ad hoc ROIs, the automatic regions, the whole
bottle, the wide fixed area around it and the whole image."""

# =============================================================================
# Feature descriptors
# =============================================================================

DESCRIPTORS: list[str] = ["hog", "lbp", "color"]
"""Descriptors compared by the experiments. Each one is classified on its own."""

HOG_CELL: dict[str, tuple[int, int]] = {
    "cap": (8, 8),
    "label": (16, 16),
    "body": (16, 16),
    "auto": (AUTO_ROI_UNIT, AUTO_ROI_UNIT),
    "bottle": (16, 16),
    "wide": (16, 16),
    "image": (16, 16),
}
"""HOG cell size per region, in pixels ("auto" is used by every automatic region)."""

HOG_ORIENTATIONS: int = 6
"""Number of gradient orientation bins of HOG (0..180 degrees, 30 degrees each). In the
parameter search (experiments/descriptor_search.py) 6 bins had the best score, with a third
fewer values than the usual 9 bins of Dalal and Triggs (within one standard error of it)."""

LBP_POINTS: int = 24
"""Neighbors compared with the central pixel by the LBP operator."""

LBP_RADIUS: int = 3
"""Radius, in pixels, of the circle the LBP neighbors lie on. The pair (24 neighbors,
radius 3) was chosen by the parameter search (experiments/descriptor_search.py): on the
training images it gave the highest cross-validated macro F1 among (8, 1), (8, 2), (16, 2)
and (24, 3)."""

LBP_CELL: dict[str, tuple[int, int]] = {
    "cap": (16, 16),
    "label": (16, 16),
    "body": (16, 16),
    "auto": (AUTO_ROI_UNIT, AUTO_ROI_UNIT),
    "bottle": (16, 16),
    "wide": (16, 16),
    "image": (16, 16),
}
"""Cell size per region, in pixels. One histogram of uniform LBP codes is computed per
cell, which keeps where in the region each texture is."""

HSV_HISTOGRAM_BINS: tuple[int, int, int] = (36, 16, 16)
"""Histogram bins for the H, S and V channels, chosen by the parameter search
(experiments/descriptor_search.py) among (9, 4, 4), (18, 8, 8) and (36, 16, 16)."""

FEATURE_GROUPS: list[str] = [
    f"{region}_{descriptor}" for region in REGIONS for descriptor in DESCRIPTORS
]
"""Every extracted feature group: one descriptor of one region (e.g. "cap_hog")."""

ALL_ROIS: str = "rois"
"""Scope of the vectors that concatenate one descriptor over the three ad hoc ROIs."""

ALL_AUTO_ROIS: str = "auto_rois"
"""Scope of the vectors that concatenate one descriptor over the automatic regions."""

SCOPE_REGIONS: dict[str, list[str]] = {
    ALL_ROIS: list(ROIS),
    ALL_AUTO_ROIS: list(AUTO_ROIS),
    WHOLE_BOTTLE: [WHOLE_BOTTLE],
    WIDE_AREA: [WIDE_AREA],
    WHOLE_IMAGE: [WHOLE_IMAGE],
}
"""Regions concatenated by each scope. A single ROI is never classified on its own."""

FEATURE_VECTORS: dict[str, list[str]] = {
    f"{scope}_{descriptor}": [f"{region}_{descriptor}" for region in regions]
    for descriptor in DESCRIPTORS
    for scope, regions in SCOPE_REGIONS.items()
}
"""Feature vectors classified separately, as {name: feature groups concatenated}.

Each vector holds a single descriptor, so the results show the impact of each one:
`rois_<descriptor>` is the descriptor of the three ad hoc ROIs, `auto_rois_<descriptor>`
of the automatic regions, `bottle_<descriptor>` of the whole bottle (no ROI) and
`wide_<descriptor>` of the wide fixed area around the bottle and `image_<descriptor>` of the
whole image (both without localization)."""

# =============================================================================
# Training
# =============================================================================

N_FOLDS: int = 5
"""Stratified folds of the hyperparameter grid search (training set only)."""

GRID_SEARCH_SCORING: str = "f1_macro"
"""Metric optimized by the grid search; macro F1 weights every class equally."""

# =============================================================================
# Alerts
# =============================================================================

ALERT_WINDOW: int = 10
"""Number of last inspected bottles considered by the alert rule (Section 4.1)."""

ALERT_THRESHOLD: int = 4
"""Occurrences of the same defect within the window that raise an alert."""
