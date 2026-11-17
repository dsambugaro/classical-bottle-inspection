"""Generates one figure per pipeline step for a real dataset image, using the real package code,
plus the figures of the comparison between feature vectors and classifiers.

Run from `code/`:
    .venv/bin/python presentation/src/make_steps.py presentation/assets/steps [results folder]
"""

import json
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import confusion_matrix

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from bottle_inspection import config  # noqa: E402
from bottle_inspection import plots  # noqa: E402
from bottle_inspection.env import default_data_dir  # noqa: E402
from bottle_inspection.dataset import load_dataset, split_indices  # noqa: E402
from bottle_inspection.features import build_matrix, extract_features  # noqa: E402
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

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = CODE / (sys.argv[2] if len(sys.argv) > 2 else "results")
IMAGE_NAME = "loose_cap_0001_20260210_152014.jpg"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.edgecolor": "#9aa1ab",
        "axes.labelcolor": "#2b2f36",
        "xtick.color": "#4a5361",
        "ytick.color": "#4a5361",
    }
)
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
"""Regions drawn in the per-region figures (the automatic ones have their own figure)."""
DESCRIPTOR_COLORS = {"hog": "#e8590c", "lbp": "#0c8599", "color": "#9c36b5"}
MODEL_NAMES = {"svm": "SVM", "knn": "KNN", "random_forest": "Random Forest"}
MODEL_COLORS = {"svm": "#1c7ed6", "knn": "#2f9e44", "random_forest": "#e8590c"}
SCOPES = list(config.SCOPE_REGIONS)
facts: dict = {"image": IMAGE_NAME}


RASTER_DPI = 300
"""Resolution of the raster parts of a figure (photos, masks, heat maps) inside the SVG."""


def save(fig, name, dpi=RASTER_DPI):
    """Saves the figure as SVG: lines, text and markers stay sharp at any zoom. Images drawn
    with imshow are embedded as rasters at `dpi` (never below RASTER_DPI)."""
    fig.savefig(OUT / Path(name).with_suffix(".svg"), dpi=max(dpi, RASTER_DPI), bbox_inches="tight")
    plt.close(fig)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------- data
samples = load_dataset(default_data_dir())
train = split_indices(samples, "train")
val = split_indices(samples, "val")
index = next(i for i in val if samples[i].image_path.name == IMAGE_NAME)
cache = joblib.load(RESULTS / "feature_cache.joblib")
background = cache.background
params = cache.params
image = read_image(samples[index].image_path)
facts["true_class"] = samples[index].label
facts["image_shape"] = list(image.shape)

# ---------------------------------------------------------------- 01 original as matrix
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
a.imshow(rgb(image))
y0, x0 = 120, 640
a.add_patch(Rectangle((x0, y0), 6, 6, fill=False, ec="yellow", lw=2))
a.set_title(f"Imagem original: matriz {image.shape[0]}×{image.shape[1]}×3 (uint8, BGR)")
a.axis("off")
patch = image[y0 : y0 + 4, x0 : x0 + 4]
b.imshow(rgb(patch), interpolation="nearest")
for r in range(4):
    for c in range(4):
        bgr = patch[r, c]
        b.text(
            c,
            r,
            f"B{bgr[0]}\nG{bgr[1]}\nR{bgr[2]}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if bgr.mean() < 140 else "black",
        )
b.set_title(f"Zoom: 4×4 pixels em (x={x0}, y={y0})")
hide_ticks(b)
save(fig, "s01_original.png")
facts["pixel_example"] = image[y0, x0].tolist()

# ---------------------------------------------------------------- 02 background
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.imshow(rgb(background))
ax.set_title("Modelo de fundo: mediana por pixel de 200 imagens de treino (640×360×3)")
ax.axis("off")
save(fig, "s02_background.png")

# ---------------------------------------------------------------- 03 reference patch + gains
ref = reference_patch_mean(background)
cur = reference_patch_mean(image)
gain = ref / (cur + 1e-6)
facts["patch_mean_background_bgr"] = ref.round(1).tolist()
facts["patch_mean_image_bgr"] = cur.round(1).tolist()
facts["gain_bgr"] = gain.round(3).tolist()
h, w = image.shape[:2]
px0, py0, px1, py1 = config.REFERENCE_PATCH
facts["patch_pixels"] = [int(px0 * w), int(py0 * h), int(px1 * w), int(py1 * h)]
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
a.imshow(rgb(image))
a.add_patch(
    Rectangle(
        (px0 * w, py0 * h),
        (px1 - px0) * w,
        (py1 - py0) * h,
        fill=False,
        ec="white",
        lw=2.5,
        ls="--",
    )
)
a.text(px0 * w, py0 * h - 8, "REFERENCE_PATCH", color="white", fontsize=10, fontweight="bold")
a.set_title("Região da parede usada como referência de brilho/cor")
a.axis("off")
xs = np.arange(3)
b.bar(xs - 0.2, ref, 0.4, label="fundo", color="#868e96")
b.bar(xs + 0.2, cur, 0.4, label="esta imagem", color="#1c7ed6")
for i in range(3):
    b.text(i, max(ref[i], cur[i]) + 3, f"ganho\n{gain[i]:.3f}", ha="center", fontsize=9)
b.set_xticks(xs, ["B", "G", "R"])
b.set_ylim(0, max(ref.max(), cur.max()) * 1.3)
b.set_title("Média do patch por canal")
b.legend(fontsize=9, loc="lower right")
save(fig, "s03_patch.png")

# ---------------------------------------------------------------- 04 normalized
normalized = normalize_lighting(image, ref)
facts["pixel_example_normalized"] = normalized[y0, x0].tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
axes[0].imshow(rgb(image))
axes[0].set_title("Antes")
axes[1].imshow(rgb(normalized))
axes[1].set_title(f"Depois: cada canal × ganho (B {gain[0]:.3f}, G {gain[1]:.3f}, R {gain[2]:.3f})")
for ax in axes:
    ax.axis("off")
save(fig, "s04_normalized.png")

# ---------------------------------------------------------------- 05 difference
small = cv2.resize(normalized, config.LOCALIZATION_SIZE)
diff = background_difference(small, background)
threshold = otsu_threshold(diff)
facts["otsu_threshold"] = threshold
# The colour bar is attached to the image (same height) and wspace leaves room for its tick
# labels, so they never run into the histogram's y label.
fig, axes = plt.subplots(
    1, 3, figsize=(16.5, 4), gridspec_kw={"width_ratios": [1.5, 1.6, 1], "wspace": 0.45}
)
axes[0].imshow(rgb(small))
axes[0].set_title("Imagem normalizada reduzida para 640×360")
axes[0].axis("off")
im = axes[1].imshow(diff, cmap="magma", vmin=0, vmax=300)
axes[1].set_title("|imagem − fundo| somado em B+G+R")
axes[1].axis("off")
fig.colorbar(im, cax=make_axes_locatable(axes[1]).append_axes("right", size="3%", pad=0.08))
axes[2].hist(np.clip(diff, 0, 255).ravel(), bins=64, color="#868e96", log=True)
axes[2].axvline(threshold, color="#e03131", lw=2, label=f"limiar de Otsu = {threshold:.0f}")
axes[2].set_xlabel("diferença (limitada a 255)")
axes[2].set_ylabel("nº de pixels (log)")
axes[2].set_title("Histograma das diferenças")
axes[2].legend(fontsize=9)
save(fig, "s05_diff.png")

# ---------------------------------------------------------------- 06 mask
raw = diff > threshold
kernel = lambda size: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))  # noqa: E731
closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel(config.MASK_CLOSE_SIZE))
mask = foreground_mask(small, background)
assert np.array_equal(
    mask, cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel(config.MASK_OPEN_SIZE)).astype(bool)
)
facts["mask_fraction_raw"] = round(float(raw.mean()), 4)
facts["mask_fraction"] = round(float(mask.mean()), 4)
mh, mw = mask.shape
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, m_, title in zip(
    axes,
    [raw, closed, mask],
    [
        f"1) diferença > {threshold:.0f} (Otsu)",
        f"2) fechamento ({config.MASK_CLOSE_SIZE} px): junta os furos",
        f"3) abertura ({config.MASK_OPEN_SIZE} px): remove linhas finas",
    ],
):
    ax.imshow(m_, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
save(fig, "s06_mask.png")

# ---------------------------------------------------------------- 07 locate: bottle box from the mask profiles
box = locate_bottle(mask)
smoothing = config.PROFILE_SMOOTHING
columns = np.convolve(mask.sum(axis=0), np.ones(smoothing) / smoothing, mode="same")
bx0, bx1, centre = bottle_columns(mask)
box_x0, box_x1 = box.x0 * mw, box.x1 * mw
rows = mask[:, bx0:bx1].sum(axis=1)
by0, by1 = int(box.y0 * mh), int(box.y1 * mh)
assert abs(centre / mw - box.center_x) < 1e-9
facts["box"] = [round(v, 4) for v in (box.x0, box.y0, box.x1, box.y1)]
facts["box_px_640"] = [round(box_x0, 1), by0, round(box_x1, 1), by1]
facts["bottle_run_640"] = [bx0, bx1]
facts["bottle_centre_640"] = round(centre, 1)
facts["box_px_1280"] = [int(box.x0 * w), int(box.y0 * h), int(box.x1 * w), int(box.y1 * h)]
fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 2.2], wspace=0.05, hspace=0.08)
top = fig.add_subplot(gs[0, 0])
main = fig.add_subplot(gs[1, 0], sharex=top)
side = fig.add_subplot(gs[1, 1], sharey=main)
top.plot(columns, color="#4a5361")
top.axhline(
    columns.max() * config.COLUMN_PROFILE_FRACTION,
    color="#e03131",
    ls="--",
    label=f"{config.COLUMN_PROFILE_FRACTION:.0%} do pico".replace(".", ","),
)
top.axvspan(bx0, bx1, color="#e03131", alpha=0.15)
top.set_ylabel("brancos por coluna")
top.legend(fontsize=8, loc="upper left")
top.tick_params(labelbottom=False)
top.set_title(
    f"Perfil de colunas → garrafa nas colunas {bx0} a {bx1}, centro em x = {centre:.0f} · perfil de linhas → y de {by0} a {by1}"
)
main.imshow(mask, cmap="gray", aspect="auto")
main.add_patch(
    Rectangle((box_x0, by0), box_x1 - box_x0, by1 - by0, fill=False, ec="#e03131", lw=2.5)
)
main.axvline(centre, color="#e03131", lw=1, ls=":")
main.set_xlabel("x (pixels, largura 640)")
main.set_ylabel("y (pixels, altura 360)")
side.plot(rows, np.arange(mh), color="#4a5361")
side.plot(
    [rows.max() * config.TOP_PROFILE_FRACTION] * 2,
    [0, mh / 2],
    color="#e03131",
    ls="--",
    label=f"topo: {config.TOP_PROFILE_FRACTION:.0%} do máximo",
)
side.plot(
    [rows.max() * config.ROW_PROFILE_FRACTION] * 2,
    [mh / 2, mh],
    color="#f08c00",
    ls="--",
    label=f"base: {config.ROW_PROFILE_FRACTION:.0%} do máximo",
)
side.axhspan(by0, by1, color="#e03131", alpha=0.15)
side.set_xlabel("brancos por linha")
side.legend(fontsize=8, loc="lower right")
side.tick_params(labelleft=False)
save(fig, "s07_locate.png")

# ---------------------------------------------------------------- 08 regions (3 ad hoc ROIs + whole bottle)
prepared = prepare_image(image, background, True, params=params)
assert prepared.box == box
regions = prepared.rois
facts["region_shapes"] = {k: list(v.shape) for k, v in regions.items()}
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
ax.set_title("Caixa detectada (tracejada) e as 3 ROIs ad hoc, relativas à caixa")
ax.axis("off")
for k, name in enumerate(SHOWN):
    a2 = fig.add_subplot(gs[k + 1])
    a2.imshow(rgb(regions[name]))
    hh, ww = regions[name].shape[:2]
    a2.set_title(f"{name}\n{ww}×{hh} px", color=REGION_COLORS[name])
    a2.axis("off")
save(fig, "s08_regions.png")

# ---------------------------------------------------------------- 08b automatic regions (learned from the training bottles)
auto_bands = params.auto_bands()
facts["auto_bands"] = {k: [round(v, 4) for v in band] for k, band in auto_bands.items()}
mean_img = mean_bottle([cache.bottles[i].crop for i in train])
strips = strip_descriptors(mean_img)
n_strips = len(strips)
crop = cache.bottles[index].crop
fig, axes = plt.subplots(
    1, 4, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.6, 1, 1], "wspace": 0.3}
)
axes[0].imshow(rgb(mean_img.astype(np.uint8)), aspect="auto")
axes[0].set_title(f"Garrafa média\n({len(train)} de treino)")
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
axes[1].set_title(f"Cores médias de cada faixa\nde {config.AUTO_ROI_UNIT} linhas (padronizadas)")
axes[1].set_xlabel("coluna × canal")
axes[1].set_ylabel("faixa")
for ax, img_, title in [
    (axes[2], mean_img.astype(np.uint8), "Divisão ótima em\n3 grupos contíguos"),
    (axes[3], crop, "As mesmas regiões\nnesta garrafa"),
]:
    ax.imshow(rgb(img_), aspect="auto")
    ax.set_title(title)
    for name, (r0, r1) in auto_bands.items():
        ax.add_patch(
            Rectangle(
                (1.5, r0 * 384 + 1),
                92,
                (r1 - r0) * 384 - 3,
                fill=False,
                ec=REGION_COLORS[name],
                lw=3,
            )
        )
        ax.text(
            48,
            (r0 + r1) / 2 * 384,
            name,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            bbox={"facecolor": REGION_COLORS[name], "edgecolor": "none", "pad": 2},
        )
for band in list(auto_bands.values())[1:]:
    axes[1].axhline(band[0] * n_strips, color="black", lw=2)
for ax in (axes[0], axes[2], axes[3]):
    hide_ticks(ax)
save(fig, "s08b_auto_regions.png")

# ---------------------------------------------------------------- 09 resized gray
sizes = params.region_sizes()
grays = {
    name: cv2.cvtColor(cv2.resize(regions[name], sizes[name]), cv2.COLOR_BGR2GRAY) for name in sizes
}
facts["region_sizes"] = {k: list(v) for k, v in sizes.items()}
fig, axes = plt.subplots(1, 4, figsize=(12, 6), gridspec_kw={"width_ratios": [64, 64, 48, 48]})
for ax, name in zip(axes, SHOWN):
    g = grays[name]
    ax.imshow(
        g, cmap="gray", interpolation="nearest", aspect="auto" if name == "bottle" else "equal"
    )
    ax.set_title(f"{name}\n{sizes[name][0]}×{sizes[name][1]} em cinza", color=REGION_COLORS[name])
    hide_ticks(ax)
save(fig, "s09_resized.png")

# ---------------------------------------------------------------- 10 HOG: one cell
g = grays["cap"].astype(np.float64)
gxm = np.zeros_like(g)
gym = np.zeros_like(g)
gxm[:, 1:-1] = g[:, 2:] - g[:, :-2]
gym[1:-1, :] = g[2:, :] - g[:-2, :]
mag = np.hypot(gxm, gym)
ang = np.rad2deg(np.arctan2(gym, gxm)) % 180
r, cc = 2, 3
cm, ca = (
    mag[r * 8 : (r + 1) * 8, cc * 8 : (cc + 1) * 8],
    ang[r * 8 : (r + 1) * 8, cc * 8 : (cc + 1) * 8],
)
n_bins = config.HOG_ORIENTATIONS
step = 180 // n_bins
hist, _ = np.histogram(ca, bins=n_bins, range=(0, 180), weights=cm)
facts["hog_cell_hist"] = hist.round(1).tolist()
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
a.imshow(grays["cap"], cmap="gray", interpolation="nearest")
a.add_patch(Rectangle((cc * 8 - 0.5, r * 8 - 0.5), 8, 8, fill=False, ec="#e03131", lw=2.5))
a.set_title("ROI cap (64×48); uma célula 8×8")
hide_ticks(a)
centres = np.arange(n_bins) * step + step / 2
b.bar(centres, hist, width=0.9 * step, color=DESCRIPTOR_COLORS["hog"])
b.set_xticks(centres, [f"{c:g}°" for c in centres])
b.set_title(f"Histograma da célula: {n_bins} direções, ponderadas pela força do gradiente")
b.set_xlabel("direção da borda")
b.set_ylabel("soma das magnitudes")
save(fig, "s10_hog_cell.png")

# ---------------------------------------------------------------- 11 HOG: every region
hog_facts = {}
fig, axes = plt.subplots(2, 4, figsize=(13, 8), gridspec_kw={"width_ratios": [64, 64, 48, 48]})
hog_cells = params.hog.cells()
for name in grays:
    cell = hog_cells[name]
    ncx, ncy = grays[name].shape[1] // cell[0], grays[name].shape[0] // cell[1]
    hog_facts[name] = {
        "cell": list(cell),
        "cells": [ncx, ncy],
        "blocks": [ncx - 1, ncy - 1],
        "length": (ncx - 1) * (ncy - 1) * 4 * config.HOG_ORIENTATIONS,
    }
for k, name in enumerate(SHOWN):
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
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1)
    aspect = "auto" if name == "bottle" else "equal"
    axes[0, k].imshow(np.hypot(gx, gy), cmap="inferno", aspect=aspect)
    axes[0, k].set_title(f"{name}\nforça do gradiente", color=REGION_COLORS[name])
    axes[1, k].imshow(
        render_hog(g, config.HOG_ORIENTATIONS, cell),
        cmap="gray",
        aspect=aspect,
        interpolation="lanczos",
    )
    axes[1, k].set_title(f"HOG ({len(vec)} valores)", color=REGION_COLORS[name])
    for ax in axes[:, k]:
        hide_ticks(ax)
    assert len(vec) == hog_facts[name]["length"]
save(fig, "s11_hog.png")
facts["hog"] = hog_facts

# ---------------------------------------------------------------- 12 LBP: one pixel
P, R = config.LBP_POINTS, config.LBP_RADIUS
codes = {
    name: local_binary_pattern(grays[name], P, R, method="uniform").astype(int) for name in grays
}
g = grays["cap"].astype(np.float64)


def circle_neighbors(gray, row, col):
    """Values of the P neighbors on the circle of radius R (bilinear interpolation), as skimage does."""
    values, offsets = [], []
    for k in range(P):
        dr = -R * np.sin(2 * np.pi * k / P)
        dc = R * np.cos(2 * np.pi * k / P)
        dr, dc = round(dr, 5), round(dc, 5)
        rr, cc_ = row + dr, col + dc
        r0, c0 = int(np.floor(rr)), int(np.floor(cc_))
        fr, fc = rr - r0, cc_ - c0
        r1, c1 = min(r0 + 1, gray.shape[0] - 1), min(c0 + 1, gray.shape[1] - 1)
        value = (
            gray[r0, c0] * (1 - fr) * (1 - fc)
            + gray[r0, c1] * (1 - fr) * fc
            + gray[r1, c0] * fr * (1 - fc)
            + gray[r1, c1] * fr * fc
        )
        values.append(value)
        offsets.append((dr, dc))
    return np.array(values), offsets


def uniform_code(bits):
    """Number of ones when the circular pattern has at most 2 transitions, else P + 1."""
    transitions = int(sum(bits[k] != bits[(k + 1) % P] for k in range(P)))
    return (int(sum(bits)), transitions) if transitions <= 2 else (P + 1, transitions)


# pick a pixel on a clear edge of the cap whose manual code matches skimage; the code must be near
# P / 2 (a balanced uniform pattern) and the neighborhood must stay inside the image
margin = R + 1
lbp_row, lbp_col = None, None
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
assert lbp_row is not None, "no example pixel found"
values, offsets = circle_neighbors(g, lbp_row, lbp_col)
center = g[lbp_row, lbp_col]
bits = (values >= center).astype(int)
code, transitions = uniform_code(bits)
assert code == codes["cap"][lbp_row, lbp_col]
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
fig, (a, b, c) = plt.subplots(
    1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.2, 1, 1.1], "wspace": 0.2}
)
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
a.set_title(f"ROI cap em cinza; pixel (x={lbp_col}, y={lbp_row})")
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
b.set_title(f"Vizinhança {side}×{side} (cinza)")
hide_ticks(b)
lim = R + 1.5
c.set_xlim(-lim, lim)
c.set_ylim(lim, -lim)
c.set_aspect("equal")
hide_ticks(c)
c.add_patch(plt.Circle((0, 0), R, fill=False, ec="#adb5bd", ls="--"))
c.text(0, 0, f"{int(center)}", ha="center", va="center", fontsize=12, fontweight="bold")
for (dr, dc), v, bit in zip(offsets, values, bits):
    c.add_patch(plt.Circle((dc, dr), 0.33, color=DESCRIPTOR_COLORS["lbp"] if bit else "#dee2e6"))
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
    f"{P} vizinhos a raio {R}: vizinho ≥ centro → 1\npadrão {''.join(map(str, bits))}\n"
    f"{transitions} transições → código = nº de uns = {code}",
    fontsize=10,
)
save(fig, "s12_lbp_pixel.png")

# ---------------------------------------------------------------- 13 LBP: code maps and cells
lbp_facts = {}
# P + 1 uniform codes on a sequential map, the non-uniform code (P + 1) in a contrasting colour
LBP_CMAP = ListedColormap([*plt.cm.viridis(np.linspace(0, 1, P + 1)), "#e03131"])
LBP_NORM = dict(cmap=LBP_CMAP, vmin=-0.5, vmax=P + 1.5)
fig, axes = plt.subplots(
    1, 4, figsize=(14, 6.4), gridspec_kw={"width_ratios": [64, 64, 48, 48], "wspace": 0.35}
)
lbp_cells = params.lbp.cells()
for name in grays:
    cw, ch = lbp_cells[name]
    ncx, ncy = codes[name].shape[1] // cw, codes[name].shape[0] // ch
    lbp_facts[name] = {"cell": [cw, ch], "cells": [ncx, ncy], "length": ncx * ncy * (P + 2)}
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
    ncx, ncy = codes[name].shape[1] // cw, codes[name].shape[0] // ch
    ax.set_title(
        f"{name}\n{ncx}×{ncy} células\n{ncx * ncy * (P + 2)} valores", color=REGION_COLORS[name]
    )
    hide_ticks(ax)
cbar = fig.colorbar(im, ax=axes, fraction=0.02, ticks=range(P + 2))
cbar.ax.tick_params(labelsize=7)
cbar.set_label(f"código LBP (0–{P} uniformes, {P + 1} = não uniforme)")
save(fig, "s13_lbp_map.png")
facts["lbp"] = lbp_facts

cell_codes = codes["cap"][16:32, 32:48]
cell_hist = np.bincount(cell_codes.ravel(), minlength=P + 2) / cell_codes.size
facts["lbp_cell_hist"] = cell_hist.round(3).tolist()
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
a.imshow(codes["cap"], **LBP_NORM, interpolation="nearest")
a.add_patch(Rectangle((32 - 0.5, 16 - 0.5), 16, 16, fill=False, ec="black", lw=2.5))
a.set_title("Códigos LBP da ROI cap; uma célula 16×16")
hide_ticks(a)
b.bar(range(P + 2), cell_hist, color=[LBP_CMAP(i) for i in range(P + 2)])
b.set_xticks(range(P + 2))
b.tick_params(axis="x", labelsize=7)
b.set_title(f"Histograma da célula: fração dos {cell_codes.size} pixels com cada código")
b.set_xlabel("código LBP")
b.set_ylabel("fração")
save(fig, "s13b_lbp_cell.png")

# ---------------------------------------------------------------- 14 HSV colour
cap = regions["cap"]
hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
axes[0, 0].imshow(rgb(cap))
axes[0, 0].set_title("ROI cap (BGR)")
for i, (nm, cmap) in enumerate(
    [("H (matiz 0–180)", "hsv"), ("S (saturação)", "gray"), ("V (brilho)", "gray")]
):
    axes[0, i + 1].imshow(hsv[..., i], cmap=cmap)
    axes[0, i + 1].set_title(nm)
for ax in axes[0]:
    ax.axis("off")
h_hist = None
for i, (bins, upper, nm) in enumerate(
    [
        (config.HSV_HISTOGRAM_BINS[0], 180, "H"),
        (config.HSV_HISTOGRAM_BINS[1], 256, "S"),
        (config.HSV_HISTOGRAM_BINS[2], 256, "V"),
    ]
):
    hst = cv2.calcHist([hsv], [i], None, [bins], [0, upper]).ravel()
    hst = hst / hst.sum()
    if i == 0:
        h_hist = hst
    axes[1, i].bar(range(bins), hst, color=["#4dabf7", "#adb5bd", "#495057"][i])
    axes[1, i].set_title(f"{nm}: {bins} faixas")
    axes[1, i].set_ylim(0, 1)
m = hsv.reshape(-1, 3).mean(0) / 255
s = hsv.reshape(-1, 3).std(0) / 255
axes[1, 3].bar(range(6), list(m) + list(s), color=["#4dabf7", "#adb5bd", "#495057"] * 2)
axes[1, 3].set_xticks(
    range(6),
    ["média H", "média S", "média V", "desvio H", "desvio S", "desvio V"],
    rotation=45,
    fontsize=8,
)
axes[1, 3].set_title("média e desvio ÷ 255")
save(fig, "s14_hsv.png")
facts["cap_hue_hist"] = [round(float(v), 3) for v in h_hist]

# ---------------------------------------------------------------- 15 the 12 feature groups
feats = extract_features(prepared, params=params)
cached = cache.feature_sets[index]
facts["groups_equal_cache"] = bool(
    all(np.array_equal(feats[g_], cached[g_]) for g_ in config.FEATURE_GROUPS)
)
facts["group_sizes"] = {g_: int(len(feats[g_])) for g_ in config.FEATURE_GROUPS}
fig, axes = plt.subplots(
    len(config.DESCRIPTORS), len(config.REGIONS), figsize=(2.6 * len(config.REGIONS), 6.4)
)
for i, descriptor in enumerate(config.DESCRIPTORS):
    for j, region in enumerate(config.REGIONS):
        ax = axes[i, j]
        v = feats[f"{region}_{descriptor}"]
        ax.plot(v, color=DESCRIPTOR_COLORS[descriptor], lw=0.6)
        ax.set_title(f"{region}_{descriptor} · {len(v)}", fontsize=10, color=REGION_COLORS[region])
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, len(v))
fig.suptitle(
    f"Os {len(config.FEATURE_GROUPS)} grupos de atributos desta imagem: 3 descritores × {len(config.REGIONS)} regiões",
    fontweight="bold",
)
fig.tight_layout()
save(fig, "s15_groups.png")

# ---------------------------------------------------------------- 16 the 15 feature vectors
vector_sizes = {
    name: sum(facts["group_sizes"][g_] for g_ in groups)
    for name, groups in config.FEATURE_VECTORS.items()
}
facts["vector_sizes"] = vector_sizes
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, descriptor in zip(axes, config.DESCRIPTORS):
    for row, scope in enumerate(SCOPES):
        name = f"{scope}_{descriptor}"
        start = 0
        for group in config.FEATURE_VECTORS[name]:
            n = facts["group_sizes"][group]
            ax.barh(
                row, n, left=start, color=REGION_COLORS[group.rpartition("_")[0]], edgecolor="white"
            )
            start += n
        ax.text(start, row, f"  {start}", va="center", fontsize=10)
    ax.set_yticks(range(len(SCOPES)), [f"{scope}_{descriptor}" for scope in SCOPES])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vector_sizes[f"{s_}_{descriptor}"] for s_ in SCOPES) * 1.25)
    ax.set_xlabel("valores no vetor")
    ax.set_title(f"vetores de {descriptor.upper()}", color=DESCRIPTOR_COLORS[descriptor])
    ax.spines[["top", "right"]].set_visible(False)
handles = [Rectangle((0, 0), 1, 1, color=REGION_COLORS[r_]) for r_ in config.REGIONS]
fig.legend(
    handles,
    config.REGIONS,
    loc="lower center",
    ncol=len(config.REGIONS),
    frameon=False,
    title="região de onde vem cada parte",
)
fig.tight_layout(rect=(0, 0.14, 1, 1))
save(fig, "s16_vectors.png")

# ---------------------------------------------------------------- 17 one matrix X per vector
y = np.array([s_.label for s_ in samples])
order = np.argsort([config.CLASSES.index(v) for v in y], kind="stable")
counts = [int((y == c_).sum()) for c_ in config.CLASSES]
edges = np.cumsum([0] + counts)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [2.2, 1.4, 1]})
for k, (ax, name) in enumerate(
    zip(axes, ["rois_hog", "rois_lbp", "rois_color"])
):  # one per descriptor
    X = build_matrix(cache.feature_sets, config.FEATURE_VECTORS[name])
    Z = (X - X[train].mean(0)) / (X[train].std(0) + 1e-12)
    im = ax.imshow(
        Z[order], aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, interpolation="nearest"
    )
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="black", lw=0.6)
    ax.set_title(f"X de {name}: {X.shape[0]} × {X.shape[1]} (padronizada)")
    ax.set_xlabel("índice do atributo")
    ax.set_yticks((edges[:-1] + edges[1:]) / 2, config.CLASSES if k == 0 else [""] * 7, fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.015)
save(fig, "s17_matrices.png")
facts["X_shapes"] = {name: [len(samples), size] for name, size in vector_sizes.items()}

# ---------------------------------------------------------------- what the classifiers receive (rois_hog, real numbers)
EXAMPLE_VECTOR = "rois_hog"
X = build_matrix(cache.feature_sets, config.FEATURE_VECTORS[EXAMPLE_VECTOR])
row = X[index : index + 1]
models = {
    n: joblib.load(RESULTS / EXAMPLE_VECTOR / n / "model.joblib")["model"] for n in MODEL_NAMES
}
scaler = models["svm"].named_steps["scaler"]
# First values of each group and the last one, with indices from the real group sizes.
sizes = {g: len(cache.feature_sets[0][g]) for g in config.FEATURE_VECTORS[EXAMPLE_VECTOR]}
start = dict(zip(sizes, np.cumsum([0, *sizes.values()])[:-1]))
show = [0, 1, start["label_hog"], start["body_hog"], X.shape[1] - 1]
assert X.shape[1] == sum(sizes.values())
names = [
    "cap_hog[0]",
    "cap_hog[1]",
    "label_hog[0]",
    "body_hog[0]",
    f"body_hog[{sizes['body_hog'] - 1}]",
]
facts["example_vector"] = EXAMPLE_VECTOR
facts["inputs"] = [
    {
        "feature": nm,
        "raw": round(float(row[0, j]), 4),
        "mean": round(float(scaler.mean_[j]), 4),
        "std": round(float(scaler.scale_[j]), 4),
        "scaled": round(float((row[0, j] - scaler.mean_[j]) / scaler.scale_[j]), 2),
    }
    for j, nm in zip(show, names)
]
svc = models["svm"].named_steps["svm"]
facts["svm_support_vectors"] = [int(v) for v in svc.support_vectors_.shape]
facts["svm_gamma"] = float(svc._gamma)
facts["svm_kernel"] = str(svc.kernel)
facts["svm_best"] = {"C": float(svc.C)}
knn = models["knn"].named_steps["knn"]
z = models["knn"].named_steps["scaler"].transform(row)
d = np.sqrt(((knn._fit_X - z) ** 2).sum(1))
facts["knn"] = {
    "k": int(knn.n_neighbors),
    "weights": str(knn.weights),
    "fit_shape": list(knn._fit_X.shape),
    "nearest_distance": round(float(d.min()), 2),
    "nearest_class": str(y[train][d.argmin()]),
}
forest = models["random_forest"].named_steps["forest"]
proba = forest.predict_proba(row)[0]
facts["rf"] = {
    "trees": len(forest.estimators_),
    "max_features": str(forest.max_features),
    "proba_max": round(float(proba.max()), 3),
    "proba_class": str(forest.classes_[proba.argmax()]),
}

# prediction of this image by every one of the 45 models
comparison = pd.read_csv(RESULTS / "feature_comparison.csv")
feature_order = list(dict.fromkeys(comparison["feature"]))
predictions = {}
for name in feature_order:
    Xv = build_matrix([cache.feature_sets[index]], config.FEATURE_VECTORS[name])
    predictions[name] = {
        n: str(joblib.load(RESULTS / name / n / "model.joblib")["model"].predict(Xv)[0])
        for n in MODEL_NAMES
    }
facts["predictions"] = predictions
facts["predictions_correct"] = int(
    sum(p == facts["true_class"] for v in predictions.values() for p in v.values())
)

# ================================================================ results


def pivot(column):
    return comparison.pivot(index="feature", columns="model", values=column).loc[
        feature_order, list(MODEL_NAMES)
    ]


def heatmap(ax, matrix, title, vmin, fmt="{:.3f}", cmap="Blues", xlabels=None):
    values = matrix.to_numpy(dtype=float)
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), xlabels or [MODEL_NAMES[m_] for m_ in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
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


# r01: the comparison matrix
fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(feature_order) + 1.4))
heatmap(axes[0], pivot("accuracy"), "Acurácia (300 imagens de validação)", 0.9)
heatmap(axes[1], pivot("f1_macro"), "F1-macro (média das 7 classes)", 0.9)
axes[1].set_yticklabels([])
fig.tight_layout()
save(fig, "r01_matrix.png")
facts["accuracy"] = pivot("accuracy").round(4).to_dict(orient="index")

# r02: the five scopes. The bars start at the bottom of the axis (not at 0), so they need no
# clipping, which some SVG viewers ignore.
low = min(0.9, float(np.floor(comparison["accuracy"].min() * 50) / 50) - 0.01)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
for ax, descriptor in zip(axes, config.DESCRIPTORS):
    rows = comparison[comparison["descriptor"] == descriptor]
    xs = np.arange(len(SCOPES))
    for k, model in enumerate(MODEL_NAMES):
        values = rows[rows["model"] == model].set_index("scope").loc[SCOPES, "accuracy"]
        bars = ax.bar(
            xs + (k - 1) * 0.27,
            values - low,
            0.27,
            bottom=low,
            color=MODEL_COLORS[model],
            label=MODEL_NAMES[model],
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.001,
                f"{v:.3f}".replace(".", ","),
                ha="center",
                fontsize=7,
                rotation=90,
                va="bottom",
            )
    ax.set_xticks(xs, SCOPES)
    ax.set_ylim(low, 1.02)
    ax.set_title(descriptor.upper(), color=DESCRIPTOR_COLORS[descriptor])
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("acurácia")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.suptitle(
    "ROIs ad hoc (rois) × regiões automáticas (auto_rois) × garrafa inteira (bottle)"
    " × área ampla (wide) × imagem inteira (image)",
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "r02_scopes.png")

# r03: F1 of each of the 7 classes (SVM)
svm_rows = comparison[comparison["model"] == "svm"].set_index("feature").loc[feature_order]
per_class = svm_rows[[f"f1_{c_}" for c_ in config.CLASSES]]
per_class.columns = config.CLASSES
fig, ax = plt.subplots(figsize=(13, 0.42 * len(feature_order) + 1.8))
heatmap(
    ax,
    per_class,
    "F1 de cada uma das 7 classes — SVM em cada vetor",
    0.9,
    fmt="{:.2f}",
    xlabels=config.CLASSES,
)
ax.set_xticklabels(config.CLASSES, rotation=20, ha="right")
fig.tight_layout()
save(fig, "r03_per_class.png")
facts["f1_per_class_svm"] = per_class.round(3).to_dict(orient="index")


# c_<vector>: confusion matrix of every classifier on every vector
def read_predictions(vector, model):
    return pd.read_csv(RESULTS / vector / model / "validation_predictions.csv")


confusions = {}
for vector in feature_order:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    confusions[vector] = {}
    for k, (ax, model) in enumerate(zip(axes, MODEL_NAMES)):
        rows = read_predictions(vector, model)
        matrix = confusion_matrix(rows["true"], rows["predicted"], labels=config.CLASSES)
        errors = int(matrix.sum() - np.trace(matrix))
        confusions[vector][model] = [
            {"true": config.CLASSES[i], "predicted": config.CLASSES[j], "count": int(matrix[i, j])}
            for i in range(len(config.CLASSES))
            for j in range(len(config.CLASSES))
            if i != j and matrix[i, j]
        ]
        shown = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(config.CLASSES)):
            for j in range(len(config.CLASSES)):
                if matrix[i, j]:
                    wrong = i != j
                    ax.text(
                        j,
                        i,
                        str(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold" if wrong else "normal",
                        color="#c92a2a" if wrong else ("white" if shown[i, j] > 0.6 else "black"),
                    )
        ax.set_xticks(
            range(len(config.CLASSES)), config.CLASSES, rotation=40, ha="right", fontsize=9
        )
        ax.set_yticks(
            range(len(config.CLASSES)),
            config.CLASSES if k == 0 else [""] * len(config.CLASSES),
            fontsize=9,
        )
        ax.set_xlabel("classe prevista")
        if k == 0:
            ax.set_ylabel("classe real")
        ax.set_title(
            f"{MODEL_NAMES[model]} · {errors} {'erro' if errors == 1 else 'erros'} em {int(matrix.sum())}",
            color=MODEL_COLORS[model],
        )
        ax.tick_params(length=0)
    fig.suptitle(
        f"Matrizes de confusão de {vector} (validação; linhas = classe real, colunas = classe prevista)",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, f"c_{vector}.png")
facts["confusions"] = confusions

# r04: vector size and time
summary = json.loads((RESULTS / "summary.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8))
group_ms = summary["group_ms_per_image"]
a.bar(
    range(len(group_ms)),
    list(group_ms.values()),
    color=[DESCRIPTOR_COLORS[g_.rpartition("_")[2]] for g_ in group_ms],
)
a.set_xticks(range(len(group_ms)), list(group_ms), rotation=45, ha="right")
a.set_ylabel("ms por imagem")
a.set_title(
    f"Custo de cada grupo (o pré-processamento custa {summary['prepare_ms_per_image']:.0f} ms)"
)
a.spines[["top", "right"]].set_visible(False)
svm_speed = comparison[comparison["model"] == "svm"].set_index("feature").loc[feature_order]
b.bar(
    range(len(feature_order)),
    svm_speed["bottles_per_second"],
    color=[DESCRIPTOR_COLORS[f_.rpartition("_")[2]] for f_ in feature_order],
)
b.set_xticks(range(len(feature_order)), feature_order, rotation=45, ha="right")
b.set_ylabel("garrafas por segundo")
b.set_title("Vazão com o SVM (extração + predição)")
b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "r04_time.png")
facts["prepare_ms"] = round(summary["prepare_ms_per_image"], 1)
facts["localization"] = summary["bottle_localization"]
facts["group_ms"] = {k: round(v, 2) for k, v in group_ms.items()}
facts["bottles_per_second_range"] = [
    round(float(comparison["bottles_per_second"].min()), 1),
    round(float(comparison["bottles_per_second"].max()), 1),
]
facts["n_errors"] = (
    comparison.pivot(index="feature", columns="model", values="n_errors")
    .loc[feature_order, list(MODEL_NAMES)]
    .to_dict(orient="index")
)
facts["best_params"] = {
    f"{r_.feature}/{r_.model}": json.loads(r_.best_params) for r_ in comparison.itertuples()
}

# ---------------------------------------------------------------- pipeline figures (same code as
# pipeline.save_preprocessing_figures), written next to steps/ as SVG
val_samples = [samples[i] for i in val]
val_bottles = [cache.bottles[i] for i in val]
plots.plot_localization(samples, cache.bottles, background, OUT.parent / "bottle_localization.svg")
plots.plot_regions_per_class(
    val_samples,
    val_bottles,
    dict(config.ROIS),
    "Detected bottle and ad hoc ROIs (red=cap, green=label, blue=body)",
    OUT.parent / "rois_per_class.svg",
)
plots.plot_regions_per_class(
    val_samples,
    val_bottles,
    params.auto_bands(),
    "Detected bottle and automatic regions (learned from the training images)",
    OUT.parent / "auto_rois_per_class.svg",
)

(OUT / "facts.json").write_text(json.dumps(facts, indent=1))
print(
    json.dumps(
        {
            k: v
            for k, v in facts.items()
            if k not in ("f1_per_class_svm", "best_params", "predictions")
        },
        indent=1,
    )
)
