"""Generates the figures of the thesis (TCC 2) with Portuguese labels.

The script only READS the bottle_inspection package, the results folder (default
results/, relative to code/) and the dataset. Figures are written to --output (by
default ../TCC_2/figuras/tcc2, the LaTeX project; use another folder to preview them).
The numbers quoted in the thesis text are written to <output>/thesis_facts.json.

Method drawn here: background subtraction + Otsu -> bottle box from the mask profiles;
three ad hoc ROIs, three automatic regions and the whole bottle; HOG, LBP and HSV
histogram per region, never concatenated with each other; 9 feature vectors x 3
classifiers (27 models).

Usage (from code/):
    make thesis-figures
    .venv/bin/python thesis/make_figures.py [--data ../archive] [--results results]
        [--output ../TCC_2/figuras/tcc2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from skimage.feature import hog, local_binary_pattern  # noqa: E402
from sklearn.datasets import make_circles  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from bottle_inspection.env import default_data_dir  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--data",
    type=Path,
    default=default_data_dir(),
    help="dataset root (default: BOTTLE_DATA, see .env.example)",
)
parser.add_argument(
    "--results",
    type=Path,
    default=Path("results"),
    help="results folder, relative to code/ unless absolute",
)
parser.add_argument(
    "--output",
    type=Path,
    default=CODE.parent / "TCC_2" / "figuras" / "tcc2",
    help="folder where the figures and thesis_facts.json are written",
)
args = parser.parse_args()
ARCHIVE = args.data
RESULTS = args.results if args.results.is_absolute() else CODE / args.results
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
FACTS_PATH = OUT / "thesis_facts.json"

sys.path.insert(0, str(CODE))
from bottle_inspection import config  # noqa: E402
from bottle_inspection.dataset import (  # noqa: E402
    load_dataset,
    read_yolo_boxes,
    split_indices,
)
from bottle_inspection.evaluation import box_iou  # noqa: E402
from bottle_inspection.features import build_matrix  # noqa: E402
from bottle_inspection.hog_view import render_hog  # noqa: E402
from bottle_inspection.preprocessing import (  # noqa: E402
    background_difference,
    foreground_mask,
    bottle_columns,
    locate_bottle,
    normalize_lighting,
    otsu_threshold,
    prepare_image,
    read_image,
    reference_patch_mean,
)
from bottle_inspection.regions import mean_bottle, strip_descriptors  # noqa: E402

sys.path.insert(0, str(CODE / "thesis"))
from print_size import fit_text_to_print  # noqa: E402

EXAMPLE_IMAGE = "loose_cap_0001_20260210_152014.jpg"
CLASSES = list(config.CLASSES)
MODELS = ["svm", "knn", "random_forest"]
MODEL_NAMES = {"svm": "SVM", "knn": "KNN", "random_forest": "Random Forest"}
MODEL_COLORS = {"svm": "#1f77b4", "knn": "#ff7f0e", "random_forest": "#2ca02c"}
VECTORS = list(config.FEATURE_VECTORS)
SCOPES = list(config.SCOPE_REGIONS)
SCOPE_PT = {
    "rois": "ROIs ad hoc",
    "auto_rois": "regiões automáticas",
    "bottle": "garrafa inteira",
    "wide": "área ampla",
    "image": "imagem inteira",
}
DESCRIPTORS = list(config.DESCRIPTORS)
DESCRIPTOR_PT = {"hog": "HOG", "lbp": "LBP", "color": "Cor (HSV)"}
DESCRIPTOR_COLORS = {"hog": "#e8590c", "lbp": "#0c8599", "color": "#9c36b5"}
ROI_PT = {"cap": "tampa", "label": "rótulo", "body": "corpo"}
REGION_COLORS = {
    "cap": "#e03131",
    "label": "#2f9e44",
    "body": "#1c7ed6",
    "auto_1": "#f08c00",
    "auto_2": "#0c8599",
    "auto_3": "#c2255c",
    "bottle": "#7048e8",
    "wide": "#495057",
    "image": "#adb5bd",
}
SHOWN = ["cap", "label", "body", "bottle"]
"""Regions drawn in the per-region descriptor figures (automatic ones have their own)."""

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
)

facts: dict = {}
PRINTED_FONTS: dict[str, float] = {}
"""Smallest font size of each figure on the printed page (points)."""


def comma(value: float, digits: int = 2) -> str:
    """Formats a number with a decimal comma (Portuguese)."""
    return f"{value:.{digits}f}".replace(".", ",")


COMMA_FORMATTER = FuncFormatter(lambda v, _pos: f"{v:g}".replace(".", ","))


def use_comma(ax, axis: str = "both") -> None:
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(COMMA_FORMATTER)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(COMMA_FORMATTER)


RASTER_DPI = 300
"""Resolution of the raster parts of a figure (photos, masks, heat maps) inside the PDF."""


PRESENTATION_STEPS = CODE / "presentation" / "assets" / "steps"
"""Figures of the presentation (HTML slides, SVG)."""


def save(
    fig,
    name: str,
    dpi: int = RASTER_DPI,
    slide: str | None = None,
    width: float = 1.0,
    relayout: bool = False,
) -> None:
    """Saves the figure as a vector PDF (lines, text and markers stay sharp at any zoom).

    Images drawn with imshow are embedded as rasters at `dpi` (never below RASTER_DPI). The
    fonts are embedded as TrueType (pdf.fonttype = 42), as PDF/A asks for. `width` is the
    fraction of \\textwidth the figure takes in LaTeX: every text is enlarged to at least
    MIN_PRINT_PT on paper (with `relayout`, tight_layout runs again afterwards). With `slide`,
    the same figure is also saved as `<slide>.svg` in the presentation figures.
    """
    printed_pt = fit_text_to_print(fig, width, relayout)
    PRINTED_FONTS[name] = round(printed_pt, 1)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, dpi=max(dpi, RASTER_DPI), bbox_inches="tight")
    if slide is not None:
        PRESENTATION_STEPS.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PRESENTATION_STEPS / f"{slide}.svg", dpi=max(dpi, RASTER_DPI), bbox_inches="tight"
        )
        print("  ", PRESENTATION_STEPS / f"{slide}.svg")
    plt.close(fig)
    print("  ", path)


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def hide_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def panel_rows(n: int, figsize, per_row: int = 3, sharey: bool = False) -> list:
    """`n` panels in rows of `per_row`; an incomplete last row is centred.

    Each panel spans two columns of a grid with 2 * per_row columns, so a row with fewer
    panels can start half a panel later. Uses constrained layout.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    n_rows = -(-n // per_row)
    grid = fig.add_gridspec(n_rows, 2 * per_row)
    axes = []
    for k in range(n):
        row, col = divmod(k, per_row)
        in_row = min(per_row, n - row * per_row)
        start = 2 * col + (per_row - in_row)
        share = axes[0] if (sharey and axes) else None
        ax = fig.add_subplot(grid[row, start : start + 2], sharey=share)
        if share is not None and col > 0:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    return axes


def descriptor_of(group: str) -> str:
    return group.rpartition("_")[2]


def region_of(group: str) -> str:
    return group.rpartition("_")[0]


def heatmap(ax, matrix: pd.DataFrame, title: str, vmin: float, fmt="{:.3f}", cmap="Blues") -> None:
    """Annotated heat map of a DataFrame; a white line separates the groups of 3 rows."""
    values = matrix.to_numpy(dtype=float)
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), list(matrix.columns))
    ax.set_yticks(range(matrix.shape[0]), list(matrix.index))
    threshold = vmin + 0.62 * (1 - vmin)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(values[i, j]).replace(".", ","),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if values[i, j] > threshold else "black",
            )
    for edge in range(len(SCOPES), matrix.shape[0], len(SCOPES)):
        ax.axhline(edge - 0.5, color="white", lw=3)
    ax.set_title(title)
    ax.tick_params(length=0)


# =============================================================================
# Data
# =============================================================================
print("== Loading dataset and feature cache from", RESULTS)
samples = load_dataset(ARCHIVE)
train = split_indices(samples, "train")
val = split_indices(samples, "val")
labels = np.array([s.label for s in samples])
name_to_index = {s.image_path.name: i for i, s in enumerate(samples)}
cache = joblib.load(RESULTS / "feature_cache.joblib")
background = cache.background
params = cache.params
summary = json.loads((RESULTS / "summary.json").read_text())
comparison = pd.read_csv(RESULTS / "feature_comparison.csv")


def pivot(column: str) -> pd.DataFrame:
    """Vector x classifier table of one column of feature_comparison.csv."""
    table = comparison.pivot(index="feature", columns="model", values=column)
    return table.loc[VECTORS, MODELS]


facts["results_folder"] = RESULTS.name
facts["n_images"] = len(samples)
facts["n_train"] = len(train)
facts["n_val"] = len(val)
facts["distribution"] = summary["dataset"]["distribution"]
facts["classes"] = CLASSES
facts["photometric_normalization"] = summary["photometric_normalization"]

# =============================================================================
# Theory chapter: illustrative examples (synthetic data)
# =============================================================================
print("== Theory examples")

# ---- KNN -------------------------------------------------------------------
class_a = np.array([[1, 2], [2, 3], [3, 1], [2, 1]], dtype=float)
class_b = np.array([[6, 5], [7, 7], [5, 6], [4, 4]], dtype=float)
query = np.array([4.0, 3.0])
points = np.vstack([class_a, class_b])
point_labels = ["A"] * len(class_a) + ["B"] * len(class_b)
distances = np.sqrt(((points - query) ** 2).sum(axis=1))
order = np.argsort(distances, kind="stable")
facts["knn_example"] = [
    {
        "point": points[i].tolist(),
        "class": point_labels[i],
        "distance": round(float(distances[i]), 3),
    }
    for i in order
]
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.scatter(*class_a.T, s=90, marker="o", color="#1c7ed6", label="Classe A", zorder=3)
ax.scatter(*class_b.T, s=90, marker="s", color="#e8590c", label="Classe B", zorder=3)
ax.scatter(*query, s=220, marker="*", color="black", label="Amostra nova", zorder=4)
r1, r3 = distances[order[0]], distances[order[2]]
ax.add_patch(Circle(query, r1, fill=False, ls="--", color="#555555", lw=1.4))
ax.add_patch(Circle(query, r3, fill=False, ls=":", color="#555555", lw=1.6))
ax.annotate("$k=1$", xy=(query[0] + r1 * 0.72, query[1] + r1 * 0.72), fontsize=12)
ax.annotate("$k=3$", xy=(query[0] + r3 * 0.74, query[1] - r3 * 0.80), fontsize=12)
for x, y in points:
    ax.annotate(
        f"({x:g}, {y:g})", xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=10
    )
ax.annotate("(4, 3)", xy=query, xytext=(8, -14), textcoords="offset points", fontsize=10)
ax.set_xlim(0, 8.5)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.set_xlabel("Atributo $x_1$")
ax.set_ylabel("Atributo $x_2$")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
save(fig, "teoria_knn_exemplo", width=0.6)

# ---- linear SVM (margin) and RBF SVM ----------------------------------------------
rng = np.random.default_rng(7)
pos = rng.normal([2.0, 2.0], 0.55, size=(18, 2))
neg = rng.normal([5.0, 4.6], 0.55, size=(18, 2))
X_lin = np.vstack([pos, neg])
y_lin = np.array([1] * len(pos) + [-1] * len(neg))
svm_lin = SVC(kernel="linear", C=1e6).fit(X_lin, y_lin)
X_circ, y_circ = make_circles(n_samples=160, factor=0.4, noise=0.08, random_state=3)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma=2.0).fit(X_circ, y_circ)
facts["svm_example"] = {
    "linear_support_vectors": int(svm_lin.n_support_.sum()),
    "linear_margin_width": round(float(2 / np.linalg.norm(svm_lin.coef_)), 3),
    "rbf_train_accuracy": round(float(svm_rbf.score(X_circ, y_circ)), 3),
}

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
xx, yy = np.meshgrid(np.linspace(0, 7, 300), np.linspace(0, 7, 300))
zz = svm_lin.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
a.contourf(xx, yy, zz > 0, levels=1, colors=["#fde3cf", "#d0e4f7"], alpha=0.6)
a.contour(
    xx,
    yy,
    zz,
    levels=[-1, 0, 1],
    colors="k",
    linestyles=["--", "-", "--"],
    linewidths=[1.2, 1.8, 1.2],
)
a.scatter(*pos.T, s=45, marker="o", color="#1c7ed6", label="Classe $+1$", zorder=3)
a.scatter(*neg.T, s=45, marker="s", color="#e8590c", label="Classe $-1$", zorder=3)
sv = svm_lin.support_vectors_
a.scatter(
    sv[:, 0],
    sv[:, 1],
    s=200,
    facecolors="none",
    edgecolors="k",
    lw=1.6,
    label="Vetores de suporte",
    zorder=4,
)
a.set_xlabel("Atributo $x_1$")
a.set_ylabel("Atributo $x_2$")
a.set_title("(a) Margem máxima (kernel linear)")
a.legend(loc="upper left", fontsize=9)
a.set_aspect("equal")

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
zz = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
b.contourf(xx, yy, zz > 0, levels=1, colors=["#fde3cf", "#d0e4f7"], alpha=0.6)
b.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.8)
b.scatter(*X_circ[y_circ == 1].T, s=30, marker="o", color="#1c7ed6", label="Classe $+1$")
b.scatter(*X_circ[y_circ == 0].T, s=30, marker="s", color="#e8590c", label="Classe $-1$")
b.set_xlabel("Atributo $x_1$")
b.set_ylabel("Atributo $x_2$")
b.set_title("(b) Fronteira não linear (kernel RBF)")
b.legend(loc="upper left", fontsize=9)
b.set_aspect("equal")
for ax in (a, b):
    use_comma(ax)
save(fig, "teoria_svm_exemplo", width=0.97)

# =============================================================================
# Dataset
# =============================================================================
print("== Dataset")
dist = summary["dataset"]["distribution"]
fig, ax = plt.subplots(figsize=(10, 4.6))
xs = np.arange(len(CLASSES))
tr = [dist["train"][c] for c in CLASSES]
va = [dist["val"][c] for c in CLASSES]
bars_t = ax.bar(xs - 0.2, tr, 0.4, label="Treino (1200 imagens)", color="#1c7ed6")
bars_v = ax.bar(xs + 0.2, va, 0.4, label="Validação (300 imagens)", color="#f08c00")
for bars in (bars_t, bars_v):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            str(int(bar.get_height())),
            ha="center",
            fontsize=10,
        )
ax.set_xticks(xs, CLASSES, rotation=30, ha="right")
ax.set_ylabel("Número de imagens")
ax.set_ylim(0, max(tr) * 1.12)
ax.grid(axis="y", alpha=0.3)
ax.legend()
save(fig, "base_distribuicao_classes", width=0.85)

# One whole original image (scene) and one cropped example per class (validation)
first_val_of = {c: next(i for i in val if samples[i].label == c) for c in CLASSES}
# good bottle with the label front facing the camera (the first ones show its back)
first_val_of["bottle_no_defect"] = name_to_index["good_0051_20260210_123439.jpg"]
scene = read_image(samples[first_val_of["bottle_no_defect"]].image_path)
facts["scene_image"] = samples[first_val_of["bottle_no_defect"]].image_path.name
facts["image_shape"] = list(scene.shape)
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.imshow(rgb(scene))
ax.axis("off")
save(fig, "base_imagem_original", width=0.75)

fig, axes = plt.subplots(1, len(CLASSES), figsize=(2.1 * len(CLASSES), 6.2))
facts["class_examples"] = {}
for ax, c in zip(axes, CLASSES):
    i = first_val_of[c]
    facts["class_examples"][c] = samples[i].image_path.name
    ax.imshow(rgb(cache.bottles[i].crop))
    ax.set_title(c, fontsize=10)
    ax.axis("off")
save(fig, "base_exemplos_classes")

# =============================================================================
# Preprocessing (validation example image)
# =============================================================================
print("== Preprocessing")
index = name_to_index[EXAMPLE_IMAGE]
assert samples[index].split == "val"
image = read_image(samples[index].image_path)
facts["example_image"] = EXAMPLE_IMAGE
facts["example_class"] = samples[index].label
h, w = image.shape[:2]

# ---- background model ----------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.9))
a.imshow(rgb(cv2.resize(image, config.LOCALIZATION_SIZE)))
a.set_title("(a) Imagem da base")
b.imshow(rgb(background))
b.set_title(f"(b) Modelo de fundo (mediana de {config.N_BACKGROUND_IMAGES} imagens)")
for ax in (a, b):
    ax.axis("off")
save(fig, "prep_modelo_fundo")
facts["background_images"] = config.N_BACKGROUND_IMAGES
facts["localization_size"] = list(config.LOCALIZATION_SIZE)

# ---- photometric normalization -------------------------------------------------
ref = reference_patch_mean(background)
cur = reference_patch_mean(image)
gain = ref / (cur + 1e-6)
facts["patch_pixels"] = [
    int(config.REFERENCE_PATCH[0] * w),
    int(config.REFERENCE_PATCH[1] * h),
    int(config.REFERENCE_PATCH[2] * w),
    int(config.REFERENCE_PATCH[3] * h),
]
facts["patch_mean_background_bgr"] = ref.round(1).tolist()
facts["patch_mean_image_bgr"] = cur.round(1).tolist()
facts["gain_bgr"] = gain.round(3).tolist()
normalized = normalize_lighting(image, ref)

px0, py0, px1, py1 = config.REFERENCE_PATCH
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, img_, title in zip(
    axes,
    [image, normalized],
    [
        "(a) Imagem original e região de referência",
        "(b) Após a normalização: cada canal × ganho ("
        + "; ".join(comma(g_, 3) for g_ in gain)
        + ")",
    ],
):
    ax.imshow(rgb(img_))
    ax.add_patch(
        Rectangle(
            (px0 * w, py0 * h),
            (px1 - px0) * w,
            (py1 - py0) * h,
            fill=False,
            ec="white",
            lw=2,
            ls="--",
        )
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
save(fig, "prep_normalizacao")

# ---- difference to the background and Otsu threshold -----------------------------
small = cv2.resize(normalized, config.LOCALIZATION_SIZE)
diff = background_difference(small, background)
threshold = otsu_threshold(diff)
facts["otsu_threshold"] = round(float(threshold), 2)
facts["max_difference_clip"] = 255
fig, axes = plt.subplots(
    1, 3, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1.5, 1.6, 1.1], "wspace": 0.45}
)
axes[0].imshow(rgb(small))
axes[0].set_title("(a) Imagem normalizada (640×360)")
axes[0].axis("off")
im = axes[1].imshow(diff, cmap="magma", vmin=0, vmax=300)
axes[1].set_title("(b) Diferença $D(x,y)$ em relação ao fundo")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.03)
axes[2].hist(np.clip(diff, 0, 255).ravel(), bins=64, color="#868e96", log=True)
axes[2].axvline(threshold, color="#e03131", lw=2, label=f"Limiar de Otsu = {comma(threshold, 0)}")
axes[2].set_xlabel("Diferença (limitada a 255)")
axes[2].set_ylabel("Número de pixels (escala log)")
axes[2].set_title("(c) Histograma de $D$")
axes[2].legend(loc="upper right", fontsize=9)
save(fig, "prep_otsu")

# ---- mask clean-up: closing and opening -----------------------------------------


def ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


raw = diff > threshold
closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(config.MASK_CLOSE_SIZE))
mask = foreground_mask(small, background)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ellipse(config.MASK_OPEN_SIZE)).astype(bool)
assert np.array_equal(mask, opened), "foreground_mask no longer equals Otsu + closing + opening"
facts["mask_fraction_raw"] = round(float(raw.mean()), 4)
facts["mask_fraction_closed"] = round(float(closed.mean()), 4)
facts["mask_fraction"] = round(float(mask.mean()), 4)
facts["mask_close_size"] = config.MASK_CLOSE_SIZE
facts["mask_open_size"] = config.MASK_OPEN_SIZE
mh, mw = mask.shape
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, m_, title in zip(
    axes,
    [raw, closed, mask],
    [
        "(a) $D(x,y) >$ limiar de Otsu",
        f"(b) Fechamento ({config.MASK_CLOSE_SIZE} px): une os furos",
        f"(c) Abertura ({config.MASK_OPEN_SIZE} px): remove linhas finas",
    ],
):
    ax.imshow(m_, cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
save(fig, "prep_mascara")

# ---- bottle box from the column and row profiles ----------------------------------
box = locate_bottle(mask, params=params)
smoothing = config.PROFILE_SMOOTHING
columns = np.convolve(mask.sum(axis=0), np.ones(smoothing) / smoothing, mode="same")
bx0, bx1, centre = bottle_columns(mask, params=params)
box_x0, box_x1 = box.x0 * mw, box.x1 * mw
rows = mask[:, bx0:bx1].sum(axis=1)
by0, by1 = int(box.y0 * mh), int(box.y1 * mh)
assert abs(centre / mw - box.center_x) < 1e-9
facts["box"] = [round(v, 4) for v in (box.x0, box.y0, box.x1, box.y1)]
facts["box_px_640"] = [round(box_x0, 1), by0, round(box_x1, 1), by1]
facts["bottle_run_640"] = [bx0, bx1]
facts["bottle_centre_640"] = round(centre, 1)
facts["bottle_width"] = params.localization.bottle_width
facts["box_px_1280"] = [int(box.x0 * w), int(box.y0 * h), int(box.x1 * w), int(box.y1 * h)]
facts["profile_smoothing"] = smoothing
facts["column_profile_fraction"] = params.localization.column_fraction
facts["row_profile_fraction"] = params.localization.row_fraction
facts["top_profile_fraction"] = params.localization.top_fraction
fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(
    2, 2, width_ratios=[3.2, 1.3], height_ratios=[1, 2.2], wspace=0.06, hspace=0.1
)
top = fig.add_subplot(gs[0, 0])
main = fig.add_subplot(gs[1, 0], sharex=top)
side = fig.add_subplot(gs[1, 1], sharey=main)
top.plot(columns, color="#4a5361")
top.axhline(
    columns.max() * params.localization.column_fraction,
    color="#e03131",
    ls="--",
    label=f"{comma(params.localization.column_fraction * 100, 0)}% do pico",
)
top.axvspan(bx0, bx1, color="#e03131", alpha=0.15)
top.set_ylabel("Pixels por coluna")
top.legend(fontsize=9, loc="upper left")
top.tick_params(labelbottom=False)
top.set_title(
    f"(a) Perfil de colunas: garrafa de $x$ = {bx0} a {bx1}, centro em {centre:.0f}", fontsize=11
)
main.imshow(mask, cmap="gray", aspect="auto")
main.add_patch(
    Rectangle((box_x0, by0), box_x1 - box_x0, by1 - by0, fill=False, ec="#e03131", lw=2.5)
)
main.axvline(centre, color="#e03131", lw=1, ls=":")
main.set_xlabel("$x$ (pixels, largura 640)")
main.set_ylabel("$y$ (pixels, altura 360)")
side.plot(rows, np.arange(mh), color="#4a5361")
side.plot(
    [rows.max() * params.localization.top_fraction] * 2,
    [0, mh / 2],
    color="#e03131",
    ls="--",
    label=f"topo: {comma(params.localization.top_fraction * 100, 0)}% do máximo",
)
side.plot(
    [rows.max() * params.localization.row_fraction] * 2,
    [mh / 2, mh],
    color="#f08c00",
    ls="--",
    label=f"base: {comma(params.localization.row_fraction * 100, 0)}% do máximo",
)
side.axhspan(by0, by1, color="#e03131", alpha=0.15)
side.set_xlabel("Pixels por linha")
side.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
side.tick_params(labelleft=False)
side.set_title(f"(b) Perfil de linhas:\n$y$ de {by0} a {by1}", fontsize=11)
save(fig, "prep_localizacao", width=0.9)

# ---- regions: ad hoc ROIs, automatic regions and the whole bottle -----------------
prepared = prepare_image(image, background, True, params=params)
assert prepared.box == box
regions = prepared.rois
facts["region_shapes_hw"] = {k: list(v.shape[:2]) for k, v in regions.items()}
facts["roi_bands"] = {k: list(v) for k, v in config.ROIS.items()}
bw, bh = (box.x1 - box.x0) * w, (box.y1 - box.y0) * h
fig = plt.figure(figsize=(15, 5.4))
gs = fig.add_gridspec(1, 5, width_ratios=[2.6, 0.8, 0.8, 0.8, 0.8])
ax = fig.add_subplot(gs[0])
ax.imshow(rgb(normalized))
ax.add_patch(
    Rectangle(
        (box.x0 * w - 4, box.y0 * h - 4),
        bw + 8,
        bh + 8,
        fill=False,
        ec=REGION_COLORS["bottle"],
        lw=2.5,
        ls="--",
    )
)
for k, (name, (r0, r1)) in enumerate(config.ROIS.items()):
    ax.add_patch(
        Rectangle(
            (box.x0 * w + k * 3, box.y0 * h + r0 * bh),
            bw - k * 6,
            (r1 - r0) * bh,
            fill=False,
            ec=REGION_COLORS[name],
            lw=2.5,
        )
    )
ax.set_title("(a) Caixa da garrafa (tracejada) e as três ROIs")
ax.axis("off")
for k, name in enumerate(SHOWN):
    a2 = fig.add_subplot(gs[k + 1])
    a2.imshow(rgb(regions[name]))
    hh, ww = regions[name].shape[:2]
    label = ROI_PT.get(name, "garrafa inteira")
    a2.set_title(f"({'bcde'[k]}) {label}\n{ww}×{hh} px", color=REGION_COLORS[name], fontsize=11)
    a2.axis("off")
save(fig, "prep_rois")

# ---- regions without localization: wide area and whole image ----------------------------
wx0, wy0, wx1, wy1 = config.WIDE_AREA_RECT
facts["wide_area"] = {
    "rect": list(config.WIDE_AREA_RECT),
    "crop_wh": [regions[config.WIDE_AREA].shape[1], regions[config.WIDE_AREA].shape[0]],
    "size_wh": list(config.WIDE_AREA_SIZE),
}
facts["whole_image"] = {"size_wh": list(config.WHOLE_IMAGE_SIZE)}
fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.78, 1.07, 1.75])
ax = fig.add_subplot(gs[0])
ax.imshow(rgb(normalized))
ax.add_patch(
    Rectangle(
        (box.x0 * w, box.y0 * h), bw, bh, fill=False, ec=REGION_COLORS["bottle"], lw=2, ls="--"
    )
)
ax.add_patch(
    Rectangle(
        (wx0 * w, wy0 * h),
        (wx1 - wx0) * w,
        (wy1 - wy0) * h,
        fill=False,
        ec=REGION_COLORS[config.WIDE_AREA],
        lw=2.5,
    )
)
ax.set_title(
    "(a) Imagem inteira, área ampla (contínua)\ne caixa da garrafa (tracejada)", fontsize=11
)
ax.axis("off")
for k, name in enumerate([config.WIDE_AREA, config.WHOLE_IMAGE]):
    a2 = fig.add_subplot(gs[k + 1])
    a2.imshow(
        rgb(cv2.resize(regions[name], params.region_sizes()[name], interpolation=cv2.INTER_AREA))
    )
    hh, ww = regions[name].shape[:2]
    cw_, ch_ = params.region_sizes()[name]
    a2.set_title(
        f"({'bc'[k]}) {SCOPE_PT[name]}: {ww}×{hh} px\nreduzida para {cw_}×{ch_} px", fontsize=11
    )
    a2.axis("off")
fig.tight_layout()
save(fig, "prep_sem_localizacao", slide="p02_sem_localizacao", relayout=True)

# ---- automatic regions: how they are learned --------------------------------------
auto_bands = params.auto_bands()
mean_img = mean_bottle([cache.bottles[i].crop for i in train])
strips = strip_descriptors(mean_img)
n_strips = len(strips)
facts["auto_bands"] = {k: [round(v, 4) for v in band] for k, band in auto_bands.items()}
facts["auto_bands_rows_384"] = {
    k: [int(round(band[0] * 384)), int(round(band[1] * 384))] for k, band in auto_bands.items()
}
facts["auto_bands_strips"] = {
    k: [int(round(band[0] * n_strips)), int(round(band[1] * n_strips))]
    for k, band in auto_bands.items()
}
facts["auto_n_strips"] = n_strips
facts["auto_strip_rows"] = config.AUTO_ROI_UNIT
facts["auto_bands_summary_json_equal"] = all(
    np.allclose(auto_bands[k], summary["auto_roi_bands"][k]) for k in auto_bands
)
crop = cache.bottles[index].crop
crop_h, crop_w = crop.shape[:2]


def draw_auto_bands(ax, with_names: bool = True, lw: float = 3) -> None:
    for name, (r0, r1) in auto_bands.items():
        ax.add_patch(
            Rectangle(
                (1.5, r0 * crop_h + 1),
                crop_w - 4,
                (r1 - r0) * crop_h - 3,
                fill=False,
                ec=REGION_COLORS[name],
                lw=lw,
            )
        )
        if with_names:
            ax.text(
                crop_w / 2,
                (r0 + r1) / 2 * crop_h,
                name.replace("auto_", "auto "),
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": REGION_COLORS[name], "edgecolor": "none", "pad": 2},
            )


fig, axes = plt.subplots(
    1, 4, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1, 1.7, 1, 1], "wspace": 0.3}
)
axes[0].imshow(rgb(mean_img.astype(np.uint8)), aspect="auto")
axes[0].set_title(f"(a) Garrafa média\n({len(train)} imagens de treino)", fontsize=11)
for edge in range(1, n_strips):
    axes[0].axhline(edge * config.AUTO_ROI_UNIT - 0.5, color="white", lw=0.5)
axes[1].imshow(
    strips,
    aspect="auto",
    cmap="coolwarm",
    vmin=-2.5,
    vmax=2.5,
    extent=(0, strips.shape[1], n_strips, 0),
)
axes[1].set_title(
    f"(b) Cores de cada faixa de\n{config.AUTO_ROI_UNIT} linhas (padronizadas)", fontsize=11
)
axes[1].set_xlabel("Coluna × canal")
axes[1].set_ylabel("Faixa (0 = topo)")
for band in list(auto_bands.values())[1:]:
    axes[1].axhline(band[0] * n_strips, color="black", lw=2)
axes[2].imshow(rgb(mean_img.astype(np.uint8)), aspect="auto")
axes[2].set_title("(c) Divisão ótima em\n3 grupos contíguos", fontsize=11)
axes[3].imshow(rgb(crop), aspect="auto")
axes[3].set_title("(d) As mesmas regiões\nem outra garrafa", fontsize=11)
for ax in (axes[2], axes[3]):
    draw_auto_bands(ax)
for ax in (axes[0], axes[2], axes[3]):
    hide_ticks(ax)
save(fig, "prep_regioes_automaticas")

# ---- ad hoc ROIs and automatic regions on one bottle per class --------------------
fig, axes = plt.subplots(2, len(CLASSES), figsize=(2.1 * len(CLASSES), 11.4))
for col, c in enumerate(CLASSES):
    crop_c = cache.bottles[first_val_of[c]].crop
    for row in range(2):
        ax = axes[row, col]
        ax.imshow(rgb(crop_c))
        ax.axis("off")
        if row == 0:
            ax.set_title(c, fontsize=9)
    for k, (name, (r0, r1)) in enumerate(config.ROIS.items()):
        inset = 1 + 2 * k
        axes[0, col].add_patch(
            Rectangle(
                (inset, r0 * crop_h),
                crop_w - 2 * inset,
                (r1 - r0) * crop_h,
                fill=False,
                ec=REGION_COLORS[name],
                lw=2,
            )
        )
    for name, (r0, r1) in auto_bands.items():
        axes[1, col].add_patch(
            Rectangle(
                (2, r0 * crop_h + 1),
                crop_w - 5,
                (r1 - r0) * crop_h - 3,
                fill=False,
                ec=REGION_COLORS[name],
                lw=2,
            )
        )
axes[0, 0].text(
    -0.08,
    0.5,
    "ROIs ad hoc",
    transform=axes[0, 0].transAxes,
    rotation=90,
    ha="right",
    va="center",
    fontsize=11,
)
axes[1, 0].text(
    -0.08,
    0.5,
    "Regiões automáticas",
    transform=axes[1, 0].transAxes,
    rotation=90,
    ha="right",
    va="center",
    fontsize=11,
)
handles = [
    Rectangle((0, 0), 1, 1, fc="none", ec=REGION_COLORS[r_], lw=2)
    for r_ in [*config.ROIS, *config.AUTO_ROIS]
]
fig.legend(
    handles,
    [ROI_PT.get(r_, r_.replace("auto_", "auto ")) for r_ in [*config.ROIS, *config.AUTO_ROIS]],
    loc="lower center",
    ncol=6,
    frameon=False,
    fontsize=10,
)
fig.subplots_adjust(bottom=0.06, wspace=0.08, hspace=0.04)
save(fig, "prep_rois_por_classe")

# =============================================================================
# Features
# =============================================================================
print("== Features")
sizes = params.region_sizes()
grays = {
    name: cv2.cvtColor(cv2.resize(regions[name], sizes[name]), cv2.COLOR_BGR2GRAY)
    for name in config.REGIONS
}
facts["region_sizes_wh"] = {k: list(sizes[k]) for k in config.REGIONS}
GRID = {"width_ratios": [64, 64, 48, 48]}

# ---- HOG ----------------------------------------------------------------------------
hog_cells = params.hog.cells()
hog_facts = {}
for name in config.REGIONS:
    cell = hog_cells[name]
    ncx, ncy = grays[name].shape[1] // cell[0], grays[name].shape[0] // cell[1]
    hog_facts[name] = {
        "cell": list(cell),
        "cells": [ncx, ncy],
        "blocks": [ncx - 1, ncy - 1],
        "length": (ncx - 1) * (ncy - 1) * 4 * config.HOG_ORIENTATIONS,
    }
# One figure per region (LaTeX groups them as sub-figures): resized region with the cell grid,
# gradient magnitude and HOG drawing, side by side at the true aspect ratio of the region.
HOG_FIGURE_WIDTH = {"cap": 1.0, "label": 0.8, "body": 0.32, "bottle": 0.64}
"""Fraction of \\textwidth of each region figure in LaTeX (body and bottle at the same height)."""
for name in SHOWN:
    g = grays[name]
    cell = hog_cells[name]
    vec = hog(
        g,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    assert len(vec) == hog_facts[name]["length"]
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1)
    fig_w = HOG_FIGURE_WIDTH[name] * 16 / 2.54
    panel_w = (fig_w - 0.3) / 3
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, panel_w * g.shape[0] / g.shape[1] + 0.55), layout="constrained"
    )
    axes[0].imshow(g, cmap="gray", interpolation="nearest")
    for x in range(0, g.shape[1] + 1, cell[0]):
        axes[0].axvline(x - 0.5, color=REGION_COLORS[name], lw=0.9)
    for y in range(0, g.shape[0] + 1, cell[1]):
        axes[0].axhline(y - 0.5, color=REGION_COLORS[name], lw=0.9)
    axes[0].set_title(f"Células de\n{cell[0]}×{cell[1]} px", fontsize=8.5)
    axes[1].imshow(np.hypot(gx, gy), cmap="inferno")
    axes[1].set_title("Magnitude do\ngradiente", fontsize=8.5)
    axes[2].imshow(
        render_hog(g, config.HOG_ORIENTATIONS, cell), cmap="gray", interpolation="lanczos"
    )
    axes[2].set_title(f"HOG\n({len(vec)} valores)", fontsize=8.5)
    for ax in axes:
        hide_ticks(ax)
    save(fig, f"atr_hog_{name}", width=HOG_FIGURE_WIDTH[name])
facts["hog"] = hog_facts
facts["hog_orientations"] = config.HOG_ORIENTATIONS

# histogram of one cap cell (computed by hand, [-1, 0, 1] masks)
g = grays["cap"].astype(np.float64)
gxm = np.zeros_like(g)
gym = np.zeros_like(g)
gxm[:, 1:-1] = g[:, 2:] - g[:, :-2]
gym[1:-1, :] = g[2:, :] - g[:-2, :]
mag = np.hypot(gxm, gym)
ang = np.rad2deg(np.arctan2(gym, gxm)) % 180
r, cc = 2, 3
cell_mag = mag[r * 8 : (r + 1) * 8, cc * 8 : (cc + 1) * 8]
cell_ang = ang[r * 8 : (r + 1) * 8, cc * 8 : (cc + 1) * 8]
n_bins = config.HOG_ORIENTATIONS
step = 180 // n_bins
hist, _ = np.histogram(cell_ang, bins=n_bins, range=(0, 180), weights=cell_mag)
facts["hog_cell_hist"] = hist.round(1).tolist()
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
a.imshow(grays["cap"], cmap="gray", interpolation="nearest")
a.add_patch(Rectangle((cc * 8 - 0.5, r * 8 - 0.5), 8, 8, fill=False, ec="#e03131", lw=2.5))
a.set_title("(a) Região da tampa e uma célula de 8×8")
hide_ticks(a)
centres = np.arange(n_bins) * step + step / 2
b.bar(centres, hist, width=0.9 * step, color="#e03131")
b.set_xticks(centres, [f"{d0}–{d0 + step}°" for d0 in range(0, 180, step)], rotation=30)
b.set_title("(b) Histograma de orientações da célula")
b.set_xlabel("Orientação do gradiente")
b.set_ylabel("Soma das magnitudes")
use_comma(b, "y")
save(fig, "atr_hog_celula", width=0.85)

# ---- LBP: one pixel ---------------------------------------------------------------------
P, R = config.LBP_POINTS, config.LBP_RADIUS
codes = {
    name: local_binary_pattern(grays[name], P, R, method="uniform").astype(int)
    for name in config.REGIONS
}
g = grays["cap"].astype(np.float64)


def circle_neighbors(gray: np.ndarray, row: int, col: int):
    """Values of the P neighbors on the circle of radius R (bilinear), as skimage does."""
    values, offsets = [], []
    for k in range(P):
        dr = round(-R * np.sin(2 * np.pi * k / P), 5)
        dc = round(R * np.cos(2 * np.pi * k / P), 5)
        rr, cc_ = row + dr, col + dc
        r0, c0 = int(np.floor(rr)), int(np.floor(cc_))
        fr, fc = rr - r0, cc_ - c0
        r1, c1 = min(r0 + 1, gray.shape[0] - 1), min(c0 + 1, gray.shape[1] - 1)
        values.append(
            gray[r0, c0] * (1 - fr) * (1 - fc)
            + gray[r0, c1] * (1 - fr) * fc
            + gray[r1, c0] * fr * (1 - fc)
            + gray[r1, c1] * fr * fc
        )
        offsets.append((dr, dc))
    return np.array(values), offsets


def uniform_code(bits) -> tuple[int, int]:
    """(code, transitions): number of ones if at most 2 circular transitions, else P + 1."""
    transitions = int(sum(bits[k] != bits[(k + 1) % P] for k in range(P)))
    return (int(sum(bits)), transitions) if transitions <= 2 else (P + 1, transitions)


# first pixel on a clear edge of the cap whose hand-made code matches skimage; the code must be
# near P / 2 (a balanced uniform pattern) and the neighborhood must stay inside the image
margin = R + 1
lbp_row = lbp_col = None
for row in range(margin, g.shape[0] - margin):
    for col in range(margin, g.shape[1] - margin):
        values, _ = circle_neighbors(g, row, col)
        bits = (values >= g[row, col]).astype(int)
        code, transitions = uniform_code(bits)
        if (
            code == codes["cap"][row, col]
            and transitions == 2
            and abs(code - P / 2) <= P / 6
            and np.ptp(values) > 40
        ):
            lbp_row, lbp_col = row, col
            break
    if lbp_row is not None:
        break
assert lbp_row is not None, "no example pixel found for the LBP figure"
values, offsets = circle_neighbors(g, lbp_row, lbp_col)
center = g[lbp_row, lbp_col]
bits = (values >= center).astype(int)
code, transitions = uniform_code(bits)
facts["lbp_pixel"] = {
    "row": lbp_row,
    "col": lbp_col,
    "center": int(center),
    "neighbors": [round(float(v), 1) for v in values],
    "bits": bits.tolist(),
    "transitions": transitions,
    "code": code,
}
side = 2 * R + 1
fig = plt.figure(figsize=(9.5, 8.4), layout="constrained")
_grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1.25, 1])
a, b = fig.add_subplot(_grid[0, 0]), fig.add_subplot(_grid[0, 1])
c = fig.add_subplot(_grid[1, :])
a.imshow(grays["cap"], cmap="gray", interpolation="nearest")
a.add_patch(
    Rectangle(
        (lbp_col - R - 0.5, lbp_row - R - 0.5),
        side,
        side,
        fill=False,
        ec=DESCRIPTOR_COLORS["lbp"],
        lw=2.5,
    )
)
a.set_title(f"(a) Tampa em cinza; pixel (x={lbp_col}, y={lbp_row})", fontsize=11)
hide_ticks(a)
patch = grays["cap"][lbp_row - R : lbp_row + R + 1, lbp_col - R : lbp_col + R + 1]
b.imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
for rr in range(side):
    for cc_ in range(side):
        v = int(patch[rr, cc_])
        b.text(
            cc_,
            rr,
            str(v),
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v < 140 else "black",
            fontweight="bold" if (rr, cc_) == (R, R) else "normal",
        )
b.add_patch(Rectangle((R - 0.5, R - 0.5), 1, 1, fill=False, ec=DESCRIPTOR_COLORS["lbp"], lw=2))
b.set_title(f"(b) Vizinhança {side}×{side} (níveis de cinza)", fontsize=11)
hide_ticks(b)
lim = R + 1.5
c.set_xlim(-lim, lim)
c.set_ylim(lim, -lim)
c.set_aspect("equal")
hide_ticks(c)
c.add_patch(Circle((0, 0), R, fill=False, ec="#adb5bd", ls="--"))
c.text(0, 0, f"{int(center)}", ha="center", va="center", fontsize=12, fontweight="bold")
for (dr, dc), v, bit in zip(offsets, values, bits):
    c.add_patch(Circle((dc, dr), 0.33, color=DESCRIPTOR_COLORS["lbp"] if bit else "#dee2e6"))
    c.text(
        dc,
        dr,
        f"{v:.0f}",
        ha="center",
        va="center",
        fontsize=6.5,
        color="white" if bit else "black",
    )
    c.text(
        dc * (R + 0.6) / R,
        dr * (R + 0.6) / R,
        str(bit),
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
c.set_title(
    f"(c) {P} vizinhos a raio {R}: vizinho ≥ centro → 1\n"
    f"padrão {''.join(map(str, bits))}\n"
    f"{transitions} transições, código = {code}",
    fontsize=10,
)
save(fig, "atr_lbp_pixel")

# ---- LBP: code maps and one cell --------------------------------------------------------
# P + 1 uniform codes on a sequential map, the non-uniform code (P + 1) in a contrasting colour
LBP_CMAP = ListedColormap([*plt.cm.viridis(np.linspace(0, 1, P + 1)), "#e03131"])
LBP_NORM = {"cmap": LBP_CMAP, "vmin": -0.5, "vmax": P + 1.5}
lbp_cells = params.lbp.cells()
lbp_facts = {}
for name in config.REGIONS:
    cw, ch = lbp_cells[name]
    ncx, ncy = codes[name].shape[1] // cw, codes[name].shape[0] // ch
    lbp_facts[name] = {"cell": [cw, ch], "cells": [ncx, ncy], "length": ncx * ncy * (P + 2)}
facts["lbp"] = lbp_facts
fig, axes = plt.subplots(1, 4, figsize=(14, 6.6), gridspec_kw={**GRID, "wspace": 0.35})
for ax, name in zip(axes, SHOWN):
    cw, ch = lbp_cells[name]
    im = ax.imshow(
        codes[name],
        **LBP_NORM,
        interpolation="nearest",
        aspect="auto" if name == "bottle" else "equal",
    )
    for x in range(0, codes[name].shape[1] + 1, cw):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    for y in range(0, codes[name].shape[0] + 1, ch):
        ax.axhline(y - 0.5, color="white", lw=0.8)
    n = lbp_facts[name]
    label = ROI_PT.get(name, "garrafa inteira")
    ax.set_title(
        f"{label}\n{n['cells'][0]}×{n['cells'][1]} células\n{n['length']} valores",
        color=REGION_COLORS[name],
        fontsize=10,
    )
    hide_ticks(ax)
for ax in axes:
    ax.set_anchor("N")
cbar = fig.colorbar(im, ax=axes, fraction=0.02, ticks=range(0, P + 2, 5))
cbar.set_label(f"Código LBP (0 a {P}: uniformes; {P + 1}: não uniforme)")
save(fig, "atr_lbp_mapa", width=0.85)

cell_codes = codes["cap"][16:32, 32:48]
cell_hist = np.bincount(cell_codes.ravel(), minlength=P + 2) / cell_codes.size
facts["lbp_cell_hist"] = cell_hist.round(3).tolist()
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
a.imshow(codes["cap"], **LBP_NORM, interpolation="nearest")
a.add_patch(Rectangle((32 - 0.5, 16 - 0.5), 16, 16, fill=False, ec="black", lw=2.5))
a.set_title("(a) Códigos LBP da tampa e uma célula de 16×16")
hide_ticks(a)
b.bar(range(P + 2), cell_hist, color=[LBP_CMAP(i) for i in range(P + 2)])
b.set_xticks(range(0, P + 2, 5))
b.set_title("(b) Histograma de códigos da célula")
b.set_xlabel("Código LBP")
b.set_ylabel(f"Fração dos {cell_codes.size} pixels")
use_comma(b, "y")
save(fig, "atr_lbp_celula", width=0.85)

# ---- HSV colour ------------------------------------------------------------------------------
cap = regions["cap"]
hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
fig, axes = plt.subplots(
    2, 4, figsize=(15, 6.6), gridspec_kw={"height_ratios": [0.6, 1]}, layout="constrained"
)
axes[0, 0].imshow(rgb(cap))
axes[0, 0].set_title("(a) Região da tampa")
for i, (nm, cmap) in enumerate(
    [
        ("(b) Canal H (matiz)", "hsv"),
        ("(c) Canal S (saturação)", "gray"),
        ("(d) Canal V (valor)", "gray"),
    ]
):
    axes[0, i + 1].imshow(hsv[..., i], cmap=cmap)
    axes[0, i + 1].set_title(nm)
for ax in axes[0]:
    ax.axis("off")
color_vec = []
for i, (bins, upper, nm) in enumerate(
    [
        (config.HSV_HISTOGRAM_BINS[0], 180, "(e) Histograma de H"),
        (config.HSV_HISTOGRAM_BINS[1], 256, "(f) Histograma de S"),
        (config.HSV_HISTOGRAM_BINS[2], 256, "(g) Histograma de V"),
    ]
):
    hst = cv2.calcHist([hsv], [i], None, [bins], [0, upper]).ravel()
    hst = hst / hst.sum()
    color_vec += hst.tolist()
    axes[1, i].bar(range(bins), hst, color=["#4dabf7", "#adb5bd", "#495057"][i])
    axes[1, i].set_title(f"{nm} ({bins} faixas)", fontsize=11)
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_xlabel("Faixa")
    use_comma(axes[1, i], "y")
axes[1, 0].set_ylabel("Fração dos pixels")
m = hsv.reshape(-1, 3).mean(0) / 255
s = hsv.reshape(-1, 3).std(0) / 255
axes[1, 3].bar(range(6), list(m) + list(s), color=["#4dabf7", "#adb5bd", "#495057"] * 2)
axes[1, 3].set_xticks(
    range(6),
    ["média H", "média S", "média V", "desvio H", "desvio S", "desvio V"],
    rotation=45,
    fontsize=10,
)
axes[1, 3].set_title("(h) Médias e desvios (÷ 255)", fontsize=11)
use_comma(axes[1, 3], "y")
save(fig, "atr_hsv")
facts["cap_hue_hist"] = [round(v, 3) for v in color_vec[: config.HSV_HISTOGRAM_BINS[0]]]
facts["hsv_histogram_bins"] = list(config.HSV_HISTOGRAM_BINS)

# ---- the 9 feature vectors -----------------------------------------------------------------
feats = cache.feature_sets[index]
group_sizes = {g_: int(len(feats[g_])) for g_ in config.FEATURE_GROUPS}
vector_sizes = {n: sum(group_sizes[g_] for g_ in gs_) for n, gs_ in config.FEATURE_VECTORS.items()}
facts["group_sizes"] = group_sizes
facts["vector_sizes"] = vector_sizes
facts["vector_groups"] = {n: list(gs_) for n, gs_ in config.FEATURE_VECTORS.items()}
facts["n_groups"] = len(group_sizes)
fig, axes = plt.subplots(3, 1, figsize=(8, 7.6), layout="constrained")
for ax, descriptor in zip(axes, DESCRIPTORS):
    for row, scope in enumerate(SCOPES):
        name = f"{scope}_{descriptor}"
        start = 0
        for group in config.FEATURE_VECTORS[name]:
            n = group_sizes[group]
            ax.barh(row, n, left=start, color=REGION_COLORS[region_of(group)], edgecolor="white")
            start += n
        ax.text(start, row, f"  {start}", va="center", fontsize=10)
    ax.set_yticks(range(len(SCOPES)), [f"{scope}_{descriptor}" for scope in SCOPES])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vector_sizes[f"{s_}_{descriptor}"] for s_ in SCOPES) * 1.2)
    ax.set_xlabel("Valores no vetor")
    ax.set_title(f"Vetores de {DESCRIPTOR_PT[descriptor]}", color=DESCRIPTOR_COLORS[descriptor])
    ax.spines[["top", "right"]].set_visible(False)
handles = [Rectangle((0, 0), 1, 1, color=REGION_COLORS[r_]) for r_ in config.REGIONS]
region_names = [
    ROI_PT.get(r_, SCOPE_PT.get(r_, r_.replace("auto_", "auto "))) for r_ in config.REGIONS
]
fig.legend(
    handles,
    region_names,
    loc="outside lower center",
    ncol=5,
    frameon=False,
    title="Região de onde vem cada parte do vetor",
)
save(fig, "atr_vetor")

# ---- matrix X of each descriptor (ad hoc ROIs) -------------------------------------------
order_rows = np.argsort([CLASSES.index(v) for v in labels], kind="stable")
counts = [int((labels == c).sum()) for c in CLASSES]
edges = np.cumsum([0] + counts)
fig, axes = plt.subplots(
    1, 3, figsize=(11, 5.4), gridspec_kw={"width_ratios": [2.2, 1.4, 1]}, layout="constrained"
)
for k, (ax, descriptor) in enumerate(zip(axes, DESCRIPTORS)):
    name = f"rois_{descriptor}"
    X = build_matrix(cache.feature_sets, config.FEATURE_VECTORS[name])
    Z = (X - X[train].mean(0)) / (X[train].std(0) + 1e-12)
    im = ax.imshow(
        Z[order_rows], aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, interpolation="nearest"
    )
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="black", lw=0.6)
    ax.set_title(f"({'abc'[k]}) $X$ de {name}:\n{X.shape[0]} × {X.shape[1]}", fontsize=11)
    ax.set_xlabel("Índice do atributo")
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.set_yticks((edges[:-1] + edges[1:]) / 2, CLASSES if k == 0 else [""] * 7, fontsize=9)
fig.colorbar(im, ax=axes, fraction=0.015, label="Valor padronizado")
save(fig, "atr_matriz")
facts["X_shapes"] = {n: [len(samples), size] for n, size in vector_sizes.items()}

# what the classifiers receive: real numbers of the first vector (rois_hog)
EXAMPLE_VECTOR = VECTORS[0]
models = {n: joblib.load(RESULTS / EXAMPLE_VECTOR / n / "model.joblib")["model"] for n in MODELS}
X = build_matrix(cache.feature_sets, config.FEATURE_VECTORS[EXAMPLE_VECTOR])
row = X[index : index + 1]
scaler = models["svm"].named_steps["scaler"]
show = {
    "cap_hog[0]": 0,
    "label_hog[0]": group_sizes["cap_hog"],
    "body_hog[0]": group_sizes["cap_hog"] + group_sizes["label_hog"],
}
facts["example_vector"] = EXAMPLE_VECTOR
facts["standardization_example"] = [
    {
        "feature": nm,
        "x": round(float(row[0, j]), 4),
        "mean": round(float(scaler.mean_[j]), 4),
        "std": round(float(scaler.scale_[j]), 4),
        "z": round(float((row[0, j] - scaler.mean_[j]) / scaler.scale_[j]), 2),
    }
    for nm, j in show.items()
]
svc = models["svm"].named_steps["svm"]
facts["svm_n_support"] = int(svc.n_support_.sum())
facts["svm_gamma"] = float(svc._gamma)
facts["svm_kernel"] = str(svc.kernel)
facts["svm_C"] = float(svc.C)
facts["svm_n_pairs"] = len(CLASSES) * (len(CLASSES) - 1) // 2
knn = models["knn"].named_steps["knn"]
facts["knn"] = {"k": int(knn.n_neighbors), "weights": str(knn.weights)}
forest = models["random_forest"].named_steps["forest"]
facts["rf"] = {"trees": len(forest.estimators_), "max_features": str(forest.max_features)}
facts["example_decisions"] = {n: str(m.predict(row)[0]) for n, m in models.items()}

# =============================================================================
# Results
# =============================================================================
print("== Results")

# ---- the vector x classifier matrices (main result) ------------------------------------------


def named(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns=MODEL_NAMES)


fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(VECTORS) + 1.8))
heatmap(axes[0], named(pivot("accuracy")), "(a) Acurácia (300 imagens de validação)", 0.9)
heatmap(axes[1], named(pivot("f1_macro")), "(b) F1-macro (média das 7 classes)", 0.9)
axes[1].set_yticklabels([])
fig.tight_layout()
save(fig, "res_comparacao", relayout=True)
facts["accuracy"] = pivot("accuracy").round(4).to_dict(orient="index")
facts["f1_macro"] = pivot("f1_macro").round(4).to_dict(orient="index")
facts["n_errors"] = pivot("n_errors").astype(int).to_dict(orient="index")
facts["cv_f1_macro"] = pivot("cv_f1_macro").round(4).to_dict(orient="index")
assert all(
    abs(summary["matrices"]["accuracy"][v][m] - pivot("accuracy").loc[v, m]) < 1e-4
    for v in VECTORS
    for m in MODELS
)
best_row = comparison.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
facts["best_model"] = {
    "feature": best_row["feature"],
    "model": best_row["model"],
    "accuracy": round(float(best_row["accuracy"]), 4),
    "f1_macro": round(float(best_row["f1_macro"]), 4),
}
facts["accuracy_mean_by_descriptor"] = {
    d_: round(float(comparison[comparison["descriptor"] == d_]["accuracy"].mean()), 4)
    for d_ in DESCRIPTORS
}
facts["accuracy_mean_by_scope"] = {
    s_: round(float(comparison[comparison["scope"] == s_]["accuracy"].mean()), 4) for s_ in SCOPES
}
facts["accuracy_mean_by_model"] = {
    m_: round(float(comparison[comparison["model"] == m_]["accuracy"].mean()), 4) for m_ in MODELS
}
facts["best_params"] = {
    f"{r_.feature}/{r_.model}": json.loads(r_.best_params) for r_ in comparison.itertuples()
}

# ---- ad hoc ROIs x automatic regions x whole bottle -------------------------------------------
# One row per descriptor (full width), so that the 15 bars of each one stay legible on paper.
fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.6), sharey=True, layout="constrained")
low = np.floor(comparison["accuracy"].min() * 50) / 50 - 0.01
for k, (ax, descriptor) in enumerate(zip(axes, DESCRIPTORS)):
    rows_d = comparison[comparison["descriptor"] == descriptor]
    xs = np.arange(len(SCOPES))
    for j, model in enumerate(MODELS):
        values = rows_d[rows_d["model"] == model].set_index("scope").loc[SCOPES, "accuracy"]
        bars = ax.bar(
            xs + (j - 1) * 0.27, values, 0.27, color=MODEL_COLORS[model], label=MODEL_NAMES[model]
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.001,
                comma(v, 3),
                ha="center",
                rotation=90,
                fontsize=8,
            )
    ax.set_xticks(xs, [SCOPE_PT[s_] for s_ in SCOPES])
    ax.set_ylim(low, 1.025)
    ax.set_title(f"({'abc'[k]}) {DESCRIPTOR_PT[descriptor]}", color=DESCRIPTOR_COLORS[descriptor])
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Acurácia na validação")
    use_comma(ax, "y")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
save(fig, "res_regioes")

# ---- F1 of each of the 7 classes for every vector and classifier ----------------------------
f1_cols = [f"f1_{c_}" for c_ in CLASSES]
f1_floor = np.floor(comparison[f1_cols].min().min() * 20) / 20
# Classes in the rows and vectors in the columns; one panel per classifier, stacked, so that
# the 105 values of each panel stay legible on paper. White lines separate the scopes.
fig, axes = plt.subplots(3, 1, figsize=(10, 11.5), sharex=True, layout="constrained")
facts["f1_per_class"] = {}
for k, (ax, model) in enumerate(zip(axes, MODELS)):
    table = comparison[comparison["model"] == model].set_index("feature").loc[VECTORS, f1_cols]
    table.columns = CLASSES
    values = table.to_numpy(dtype=float).T
    ax.imshow(values, cmap="Blues", vmin=f1_floor, vmax=1.0, aspect="auto")
    threshold = f1_floor + 0.62 * (1 - f1_floor)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                comma(values[i, j], 2),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if values[i, j] > threshold else "black",
            )
    for edge in range(len(DESCRIPTORS), len(VECTORS), len(DESCRIPTORS)):
        ax.axvline(edge - 0.5, color="white", lw=3)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xticks(range(len(VECTORS)), VECTORS, rotation=60, ha="right")
    ax.tick_params(length=0)
    ax.set_title(f"({'abc'[k]}) {MODEL_NAMES[model]}")
    facts["f1_per_class"][model] = table.round(4).to_dict(orient="index")
save(fig, "res_f1_por_classe")
facts["f1_floor_color_scale"] = float(f1_floor)

# ---- cost: time per group and throughput ---------------------------------------------------------
group_ms = summary["group_ms_per_image"]
fig, (a, b) = plt.subplots(2, 1, figsize=(9, 9.6), layout="constrained")
a.bar(
    range(len(group_ms)),
    list(group_ms.values()),
    color=[DESCRIPTOR_COLORS[descriptor_of(g_)] for g_ in group_ms],
)
a.set_xticks(range(len(group_ms)), list(group_ms), rotation=60, ha="right", fontsize=9)
a.set_ylabel("Milissegundos por imagem")
a.set_title(
    f"(a) Custo de cada grupo (pré-processamento: {comma(summary['prepare_ms_per_image'], 1)} ms)",
    fontsize=11,
)
a.grid(axis="y", alpha=0.3)
a.legend(
    [Rectangle((0, 0), 1, 1, color=DESCRIPTOR_COLORS[d_]) for d_ in DESCRIPTORS],
    [DESCRIPTOR_PT[d_] for d_ in DESCRIPTORS],
    loc="upper left",
    fontsize=9,
)
use_comma(a, "y")
xs = np.arange(len(VECTORS))
for j, model in enumerate(MODELS):
    values = pivot("bottles_per_second")[model]
    b.bar(xs + (j - 1) * 0.27, values, 0.27, color=MODEL_COLORS[model], label=MODEL_NAMES[model])
b.set_xticks(xs, VECTORS, rotation=60, ha="right", fontsize=9)
b.set_ylabel("Garrafas por segundo")
b.set_title("(b) Vazão (pré-processamento + extração + predição)", fontsize=11)
b.legend(loc="upper right", fontsize=9)
b.grid(axis="y", alpha=0.3)
b.set_ylim(0, pivot("bottles_per_second").to_numpy().max() * 1.25)
save(fig, "res_custo")
facts["prepare_ms_per_image"] = round(summary["prepare_ms_per_image"], 2)
facts["group_ms_per_image"] = {k: round(v, 2) for k, v in group_ms.items()}
facts["bottles_per_second"] = pivot("bottles_per_second").round(2).to_dict("index")
facts["bottles_per_second_range"] = [
    round(float(comparison["bottles_per_second"].min()), 1),
    round(float(comparison["bottles_per_second"].max()), 1),
]
facts["extract_ms_per_image"] = {
    r_.feature: round(float(r_.extract_ms_per_image), 2)
    for r_ in comparison.drop_duplicates("feature").itertuples()
}
facts["predict_ms_per_image"] = pivot("predict_ms_per_image").round(3).to_dict("index")
facts["train_seconds"] = pivot("train_seconds").round(1).to_dict("index")

# ---- bottle detection quality against the annotated boxes -----------------------------------
ious, edge_err, center_err, boxes_gt, owners = [], [], [], [], []
for i, sample in enumerate(samples):
    b_ = cache.bottles[i].box
    detected = (b_.x0, b_.y0, b_.x1, b_.y1)
    for ann in read_yolo_boxes(sample.label_path):
        if ann.class_id not in config.BOTTLE_CLASS_IDS:
            continue
        gt = (
            ann.center_x - ann.width / 2,
            ann.center_y - ann.height / 2,
            ann.center_x + ann.width / 2,
            ann.center_y + ann.height / 2,
        )
        ious.append(box_iou(detected, gt))
        edge_err.append(np.abs(np.subtract(detected, gt)))
        center_err.append(abs(b_.center_x - ann.center_x))
        boxes_gt.append(gt)
        owners.append(i)
ious, edge_err = np.array(ious), np.array(edge_err)
loc = summary["bottle_localization"]
assert abs(float(ious.mean()) - loc["mean_iou"]) < 1e-6
worst = int(np.argmin(ious))
fig = plt.figure(figsize=(9, 8.2), layout="constrained")
_grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
axes = [
    fig.add_subplot(_grid[0, 0]),
    fig.add_subplot(_grid[0, 1]),
    fig.add_subplot(_grid[1, :]),
]
axes[0].hist(ious, bins=np.linspace(0.2, 1, 41), color="#1c7ed6")
axes[0].axvline(ious.mean(), color="#e03131", ls="--", label=f"Média = {comma(ious.mean(), 3)}")
axes[0].axvline(
    np.percentile(ious, 5),
    color="#f08c00",
    ls=":",
    label=f"5% menores < {comma(np.percentile(ious, 5), 3)}",
)
axes[0].set_xlabel("IoU entre a caixa detectada e a anotada")
axes[0].set_ylabel("Número de imagens")
axes[0].set_title(f"(a) IoU em {len(ious)} imagens", fontsize=11)
axes[0].legend(loc="upper left", fontsize=9)
axes[0].set_yscale("log")
use_comma(axes[0], "x")
axes[1].boxplot(edge_err * 100, showfliers=True, flierprops={"markersize": 2})
axes[1].set_xticks(range(1, 5), ["esquerda", "topo", "direita", "base"])
axes[1].set_ylabel("Erro da borda (% do lado da imagem)")
axes[1].set_title("(b) Erro de cada borda", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
use_comma(axes[1], "y")
wi = owners[worst]
worst_image = read_image(samples[wi].image_path)
axes[2].imshow(rgb(worst_image))
bd = cache.bottles[wi].box
for (x0_, y0_, x1_, y1_), col_, lab in [
    (boxes_gt[worst], "#1c7ed6", "anotada"),
    ((bd.x0, bd.y0, bd.x1, bd.y1), "#e03131", "detectada"),
]:
    axes[2].add_patch(
        Rectangle(
            (x0_ * w, y0_ * h),
            (x1_ - x0_) * w,
            (y1_ - y0_) * h,
            fill=False,
            ec=col_,
            lw=2.5,
            label=lab,
        )
    )
axes[2].legend(loc="lower left", fontsize=9)
axes[2].set_title(
    f"(c) Pior caso (IoU = {comma(ious[worst], 2)}), {samples[wi].label}", fontsize=11
)
axes[2].axis("off")
save(fig, "res_localizacao")
facts["localization"] = loc
facts["localization_n_boxes"] = int(len(ious))
facts["localization_iou_percentiles"] = {
    str(p): round(float(np.percentile(ious, p)), 4) for p in (1, 5, 25, 50, 75, 95)
}
facts["localization_fraction_iou_above_0_7"] = round(float((ious > 0.7).mean()), 4)
facts["localization_fraction_iou_above_0_8"] = round(float((ious > 0.8).mean()), 4)
facts["localization_edge_error_mean_percent"] = {
    e_: round(float(edge_err[:, k].mean() * 100), 3)
    for k, e_ in enumerate(("left", "top", "right", "bottom"))
}
facts["localization_worst_case"] = {
    "image": samples[wi].image_path.name,
    "class": samples[wi].label,
    "iou": round(float(ious[worst]), 4),
}

# ---- random-forest importance per region inside each vector ---------------------------------------
importance = {}
for v_ in VECTORS:
    if len(config.SCOPE_REGIONS[v_.rpartition("_")[0]]) == 1:
        continue
    df = pd.read_csv(RESULTS / v_ / "random_forest" / "feature_importance.csv")
    importance[v_] = {r_.feature_group: round(float(r_.importance), 4) for r_ in df.itertuples()}
facts["rf_importance_per_region"] = importance
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for k, (ax, scope) in enumerate(zip(axes, ["rois", "auto_rois"])):
    region_list = config.SCOPE_REGIONS[scope]
    xs = np.arange(len(DESCRIPTORS))
    for j, region in enumerate(region_list):
        vals = [importance[f"{scope}_{d_}"][f"{region}_{d_}"] for d_ in DESCRIPTORS]
        bars = ax.bar(
            xs + (j - 1) * 0.27,
            vals,
            0.27,
            color=REGION_COLORS[region],
            label=ROI_PT.get(region, region.replace("auto_", "auto ")),
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.01, comma(v, 2), ha="center", fontsize=9
            )
    ax.set_xticks(xs, [DESCRIPTOR_PT[d_] for d_ in DESCRIPTORS])
    ax.set_title(f"({'ab'[k]}) Vetores {scope}_*: {SCOPE_PT[scope]}", fontsize=11)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    use_comma(ax, "y")
axes[0].set_ylabel("Importância (soma no grupo)")
fig.tight_layout()
save(fig, "res_importancia_regioes", width=0.9, relayout=True)

# ---- confusion matrices and errors -------------------------------------------------------------------
# Rule for the single-vector figures: the confusion matrices of the three classifiers are
# drawn for (1) the first vector of config.FEATURE_VECTORS (rois_hog: ad hoc ROIs with the
# descriptor of the baseline method) and (2) the vector with the lowest mean validation
# accuracy over the three classifiers (the hardest one). The error figure shows every
# validation image misclassified by at least one classifier on vector (1).


def predictions_of(vector: str, model: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / vector / model / "validation_predictions.csv")


def plot_confusions(vector: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9), layout="constrained")
    for k, (ax, m_) in enumerate(zip(axes, MODELS)):
        pred = predictions_of(vector, m_)
        cm = confusion_matrix(pred["true"], pred["predicted"], labels=CLASSES)
        ax.imshow(cm, cmap="Blues")
        for r_ in range(len(CLASSES)):
            for c_ in range(len(CLASSES)):
                v = cm[r_, c_]
                wrong = v > 0 and r_ != c_
                ax.text(
                    c_,
                    r_,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > cm.max() / 2 else ("#e03131" if wrong else "black"),
                    fontweight="bold" if wrong else "normal",
                )
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)), CLASSES if k == 0 else [""] * len(CLASSES), fontsize=8)
        ax.tick_params(length=2)
        ax.set_xlabel("Classe prevista")
        if k == 0:
            ax.set_ylabel("Classe verdadeira")
        n_err = int((pred["true"] != pred["predicted"]).sum())
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.set_title(f"({'abc'[k]}) {MODEL_NAMES[m_]}: {n_err} erro(s)", fontsize=9)
        facts["confusion_matrices"][vector][m_] = cm.tolist()
    save(fig, name)


facts["confusion_matrices"] = {}
# The second figure takes the hardest of the 15 vectors (with and without localization).
mean_acc = pivot("accuracy").mean(axis=1)
hardest = str(mean_acc.idxmin())
facts["confusion_rule"] = (
    "figure 1: first vector of config.FEATURE_VECTORS; figure 2: vector with the lowest mean "
    "accuracy over the three classifiers among the 15 vectors"
)
facts["confusion_vectors"] = {"first": EXAMPLE_VECTOR, "hardest": hardest}
for vec, fname in [
    (EXAMPLE_VECTOR, "res_matrizes_confusao"),
    (hardest, "res_matrizes_confusao_dificil"),
]:
    facts["confusion_matrices"][vec] = {}
    plot_confusions(vec, fname)

wrong_images: dict[str, dict] = {}
for m_ in MODELS:
    pred = predictions_of(EXAMPLE_VECTOR, m_)
    for r_ in pred[pred["true"] != pred["predicted"]].itertuples():
        entry = wrong_images.setdefault(r_.image, {"true": r_.true, "predicted": {}})
        entry["predicted"][m_] = r_.predicted
facts["validation_errors"] = {EXAMPLE_VECTOR: wrong_images}
shown_errors = sorted(wrong_images)[:8]
fig, axes = plt.subplots(
    1, max(len(shown_errors), 1), figsize=(2.9 * max(len(shown_errors), 1), 7.2), squeeze=False
)
for ax, nm in zip(axes[0], shown_errors):
    ax.imshow(rgb(cache.bottles[name_to_index[nm]].crop))
    info = wrong_images[nm]
    lines = [f"({'abcdefgh'[shown_errors.index(nm)]}) {info['true']}"]
    lines += [
        (
            f"{MODEL_NAMES[m_]}: {info['predicted'][m_]}"
            if m_ in info["predicted"]
            else f"{MODEL_NAMES[m_]}: correto"
        )
        for m_ in MODELS
    ]
    ax.set_title("\n".join(lines), fontsize=8)
    ax.axis("off")
if not shown_errors:
    axes[0, 0].text(0.5, 0.5, "Nenhum erro", ha="center")
    axes[0, 0].axis("off")
save(fig, "res_erros", width=0.22)

# =============================================================================
# Parameter search (experiments/localization_sensitivity.py and descriptor_search.py)
# =============================================================================
SEARCH = RESULTS / "parameter_search"
LOCALIZATION_LABELS = {
    "close_size": "Fechamento (px)",
    "open_size": "Abertura (px)",
    "profile_smoothing": "Suavização do perfil (px)",
    "column_fraction": "Fração do perfil de colunas",
    "bottle_width": "Largura da caixa",
    "top_fraction": "Fração do topo (linhas)",
    "row_fraction": "Fração da base (linhas)",
}
FACTOR_LABELS = {
    ("hog", "orientations"): "HOG: número de direções",
    ("hog", "cell"): "HOG: tamanho da célula",
    ("lbp", "neighbourhood"): "LBP: vizinhança (P, R)",
    ("lbp", "cell"): "LBP: tamanho da célula",
    ("color", "hsv_bins"): "Cor: faixas de H/S/V",
}

if (SEARCH / "localization_sensitivity.csv").exists():
    table = pd.read_csv(SEARCH / "localization_sensitivity.csv")
    # Rows of 3 panels (the last row centred), so that each panel stays legible on paper.
    axes = panel_rows(len(LOCALIZATION_LABELS), figsize=(9.5, 8.4), sharey=True)
    fig = axes[0].figure
    for ax, (name, label) in zip(axes, LOCALIZATION_LABELS.items()):
        rows = table[table["parameter"] == name]
        ax.plot(range(len(rows)), rows["mean_iou"], marker="o", color="#1c7ed6")
        chosen = rows["is_reference"].to_numpy().argmax()
        ax.plot(chosen, rows["mean_iou"].iloc[chosen], marker="o", ms=11, color="#e03131")
        ax.set_xticks(
            range(len(rows)),
            [comma(v, 3).rstrip("0").rstrip(",") if v < 1 else f"{v:.0f}" for v in rows["value"]],
            fontsize=8,
        )
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        use_comma(ax, "y")
    for ax in axes[::3]:
        ax.set_ylabel("IoU médio")
    save(fig, "res_sensibilidade_localizacao")
    facts["localization_sensitivity"] = table.round(4).to_dict(orient="records")
    facts["localization_otsu"] = json.loads((SEARCH / "localization_otsu.json").read_text())

if (SEARCH / "descriptor_decisions.json").exists():
    decisions = json.loads((SEARCH / "descriptor_decisions.json").read_text())
    axes = panel_rows(len(FACTOR_LABELS), figsize=(9.5, 6.4), sharey=True)
    fig = axes[0].figure
    for ax, ((descriptor, factor), label) in zip(axes, FACTOR_LABELS.items()):
        item = decisions["descriptors"][descriptor]
        factor_scores = item["factors"][factor]["scores"]
        values = list(factor_scores)
        colors = [
            "#e03131" if v == item["factors"][factor]["chosen"] else DESCRIPTOR_COLORS[descriptor]
            for v in values
        ]
        bars = ax.bar(range(len(values)), [factor_scores[v] for v in values], color=colors)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                factor_scores[v] - 0.0008,
                comma(factor_scores[v], 4),
                ha="center",
                va="top",
                rotation=90,
                color="white",
                fontsize=8,
            )
        band = item["reference_score"] + item["reference_standard_error"]
        ax.axhline(band, color="gray", ls="--", lw=1)
        ax.set_xticks(
            range(len(values)),
            [
                v.replace(" px", "").replace(", ", ",\n" if factor == "neighbourhood" else "/")
                for v in values
            ],
            fontsize=8,
        )
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    lowest = min(
        min(f["scores"].values())
        for d in decisions["descriptors"].values()
        for f in d["factors"].values()
    )
    axes[0].set_ylim(max(0.0, lowest - 0.01), 1.006)
    for ax in axes[::3]:
        ax.set_ylabel("F1-macro médio (VC)")
        use_comma(ax, "y")
    save(fig, "res_sensibilidade_descritores", slide="p01_descritores")
    facts["descriptor_search"] = decisions
    facts["descriptor_search_summary"] = (
        pd.read_csv(SEARCH / "descriptor_search_summary.csv").round(5).to_dict(orient="records")
    )

# =============================================================================
# Remaining facts
# =============================================================================
facts["auto_roi_bands_summary"] = summary["auto_roi_bands"]
facts["classification_report_n_val"] = int(len(val))
FACTS_PATH.write_text(json.dumps(facts, indent=1, ensure_ascii=False))
print("== Smallest printed font of each figure (pt):")
for name_, pt_ in sorted(PRINTED_FONTS.items(), key=lambda item: item[1]):
    print(f"   {pt_:5.1f}  {name_}")
print("== Facts written to", FACTS_PATH)
