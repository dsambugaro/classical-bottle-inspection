"""Draws the didactic figures of the theory chapter ("Conceitos") with Portuguese labels.

All figures use synthetic toy data with fixed random seeds (reproducible). The only real
image is the example bottle of figure 1, found recursively under --data/images. The script
does not need the results folder. Figures are written as vector PDFs (file names start with
`teoria_`) to --output (default ../TCC_2/figuras/tcc2). The key numbers each figure shows
are printed and saved to --facts (default thesis/preview/theory_facts.json).

Usage (from code/):
    make theory-figures
    .venv/bin/python thesis/make_theory_figures.py [--data ../archive] [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, hsv_to_rgb  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)
from matplotlib.ticker import FuncFormatter  # noqa: E402
from skimage.draw import disk as draw_disk  # noqa: E402
from skimage.draw import line as draw_line  # noqa: E402
from skimage.draw import polygon as draw_polygon  # noqa: E402
from skimage.feature import local_binary_pattern  # noqa: E402
from skimage.filters import gaussian, threshold_otsu  # noqa: E402
from skimage.morphology import closing, dilation, disk, opening  # noqa: E402
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "thesis"))
from bottle_inspection.env import default_data_dir  # noqa: E402
from print_size import fit_text_to_print  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--data",
    type=Path,
    default=default_data_dir(),
    help="dataset root (default: BOTTLE_DATA, see .env.example)",
)
parser.add_argument(
    "--output",
    type=Path,
    default=CODE.parent / "TCC_2" / "figuras" / "tcc2",
    help="folder where the PDFs are written",
)
parser.add_argument(
    "--facts",
    type=Path,
    default=CODE / "thesis" / "preview" / "theory_facts.json",
    help="JSON with the numbers shown in the figures",
)
parser.add_argument(
    "--preview",
    type=Path,
    default=CODE / "thesis" / "preview" / "theory",
    help="folder with 100 dpi PNG previews (empty string disables)",
)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
GREY, LIGHT = "#555555", "#e6e6e6"
PALE_BLUE, PALE_ORANGE = "#cfe2f3", "#fde3cf"

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
written: list[Path] = []
RASTER_DPI = 300


def comma(value: float, digits: int = 2) -> str:
    """Formats a number with a decimal comma (Portuguese)."""
    return f"{value:.{digits}f}".replace(".", ",")


COMMA_FORMATTER = FuncFormatter(lambda v, _pos: f"{v:g}".replace(".", ","))


def use_comma(ax, axis: str = "both") -> None:
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(COMMA_FORMATTER)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(COMMA_FORMATTER)


def save(fig, name: str, width: float = 1.0, relayout: bool = False) -> None:
    """Vector PDF (rasters at 300 dpi) plus a 100 dpi PNG preview.

    `width` is the fraction of \\textwidth the figure takes in LaTeX: every text is enlarged
    to at least MIN_PRINT_PT on paper (with `relayout`, tight_layout runs again afterwards).
    """
    fit_text_to_print(fig, width, relayout)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, dpi=RASTER_DPI, bbox_inches="tight")
    written.append(path)
    if str(args.preview):
        args.preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  ", path)


def hide_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def panel(ax, text: str) -> None:
    ax.set_title(text, fontsize=12)


def numbers_in_cells(ax, values, fmt="{:.0f}", dark_below=128, fontsize=8) -> None:
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j,
            i,
            fmt.format(v).replace(".", ","),
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white" if v < dark_below else "black",
        )


def pixel_grid(ax, n_rows: int, n_cols: int, color="white", lw=0.6) -> None:
    for k in range(n_rows + 1):
        ax.axhline(k - 0.5, color=color, lw=lw)
    for k in range(n_cols + 1):
        ax.axvline(k - 0.5, color=color, lw=lw)


def find_example() -> Path:
    found = sorted((args.data / "images").rglob("loose_cap_0001_20260210_152014.jpg"))
    if not found:
        raise SystemExit("example image not found under " + str(args.data / "images"))
    return found[0]


# =============================================================================
# 1. Digital image
# =============================================================================
print("== 1 teoria_imagem_digital")
img_bgr = cv2.imread(str(find_example()))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
X0, Y0, S = 613, 70, 8
patch_gray = img_gray[Y0 : Y0 + S, X0 : X0 + S]
patch_rgb = img_rgb[Y0 : Y0 + S, X0 : X0 + S]
pr, pc = 3, 4
facts["imagem_digital"] = {
    "imagem": "loose_cap_0001_20260210_152014.jpg",
    "tamanho_pixels_altura_largura": list(img_gray.shape),
    "regiao_canto_superior_esquerdo_xy": [X0, Y0],
    "regiao_tamanho": S,
    "cinza_min": int(patch_gray.min()),
    "cinza_max": int(patch_gray.max()),
    "pixel_destacado_linha_coluna": [pr, pc],
    "pixel_destacado_RGB": [int(v) for v in patch_rgb[pr, pc]],
}
fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.3, 2.1], wspace=0.12)
ax = fig.add_subplot(gs[0])
cx0, cx1, cy0, cy1 = 520, 820, 30, 330
ax.imshow(img_rgb[cy0:cy1, cx0:cx1], extent=(cx0 - 0.5, cx1 - 0.5, cy1 - 0.5, cy0 - 0.5))
ax.add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), S, S, fill=False, ec=RED, lw=1.6))
ax.annotate(
    "",
    xy=(X0 - 1, Y0 + S / 2),
    xytext=(X0 - 45, Y0 + 45),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
)
ax.set_xlabel(f"quadrado vermelho: região {S}×{S}", color=RED, fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
panel(ax, "(a) Imagem colorida")
ax = fig.add_subplot(gs[1])
ax.imshow(patch_gray, cmap="gray", vmin=0, vmax=255)
numbers_in_cells(ax, patch_gray, fontsize=10)
pixel_grid(ax, S, S, color="#888888")
for sp in ax.spines.values():
    sp.set_edgecolor(RED)
    sp.set_linewidth(2)
hide_ticks(ax)
panel(ax, "(b) Região em cinza (0 a 255)")
sub = gs[2].subgridspec(2, 3, height_ratios=[1, 0.22], wspace=0.08, hspace=0.05)
for k, (name, cmap) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
    ax = fig.add_subplot(sub[0, k])
    ax.imshow(patch_rgb[:, :, k], cmap=cmap, vmin=0, vmax=255)
    pixel_grid(ax, S, S, color="#bbbbbb", lw=0.4)
    ax.add_patch(Rectangle((pc - 0.5, pr - 0.5), 1, 1, fill=False, ec="black", lw=2))
    hide_ticks(ax)
    ax.set_title(("(c) " if k == 0 else "") + f"Canal {name}")
ax = fig.add_subplot(sub[1, :])
ax.axis("off")
r, g, b = (int(v) for v in patch_rgb[pr, pc])
ax.text(
    0.5,
    0.5,
    f"pixel destacado: (R, G, B) = ({r}, {g}, {b})",
    ha="center",
    va="center",
    fontsize=12,
)
save(fig, "teoria_imagem_digital")

# =============================================================================
# 2. Inspection stages (block diagram)
# =============================================================================
print("== 2 teoria_inspecao_etapas")


def box(ax, x, y, w, h, title, sub=None, fc="#dbe9f6", ec=BLUE, dashed=False, tfs=12, sfs=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            fc=fc,
            ec=ec,
            lw=1.6,
            ls="--" if dashed else "-",
        )
    )
    if sub:
        ax.text(
            x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=tfs, weight="bold"
        )
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color=GREY)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, weight="bold")


def arrow(ax, p0, p1, color="black", lw=1.8, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=16,
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.2)
ax.axis("off")
W, H = 4.4, 1.7
xs3 = [0.2, 5.8, 11.4]
row1, row2 = 6.9, 3.9
stages = [
    ("1. Aquisição da imagem", "câmera, iluminação, disparo"),
    ("2. Pré-processamento", "normalização e segmentação"),
    ("3. Extração de atributos", "HOG, LBP, histograma HSV"),
    ("4. Classificação", "modelo treinado (SVM, KNN, RF)"),
    ("5. Decisão", "aprovar ou rejeitar"),
    ("6. Atuação e registro", "rejeição na linha, alertas, relatórios"),
]
palette = ["#dbe9f6", "#dbe9f6", "#dbe9f6", "#fde3cf", "#fde3cf", "#d8efd8"]
edges = [BLUE, BLUE, BLUE, ORANGE, ORANGE, GREEN]
pos = []
for k, (t, s) in enumerate(stages):
    x = xs3[k % 3]
    y = row1 if k < 3 else row2
    box(ax, x, y, W, H, t, s, fc=palette[k], ec=edges[k], tfs=11.5, sfs=9.5)
    pos.append((x, y))
for k in (0, 1):
    arrow(ax, (xs3[k] + W + 0.05, row1 + H / 2), (xs3[k + 1] - 0.05, row1 + H / 2))
for k in (0, 1):
    arrow(ax, (xs3[k] + W + 0.05, row2 + H / 2), (xs3[k + 1] - 0.05, row2 + H / 2))
# elbow from box 3 down to box 4
ymid = (row1 + row2 + H) / 2
ax.plot([xs3[2] + W / 2, xs3[2] + W / 2], [row1 - 0.03, ymid], color="black", lw=1.8)
ax.plot([xs3[2] + W / 2, xs3[0] + W / 2], [ymid, ymid], color="black", lw=1.8)
arrow(ax, (xs3[0] + W / 2, ymid), (xs3[0] + W / 2, row2 + H + 0.05))
# feedback
box(
    ax,
    0.2,
    0.4,
    6.6,
    1.5,
    "Linha de produção",
    "etapas anteriores (envase, tampagem, rotulagem)",
    fc="#f2f2f2",
    ec=GREY,
    dashed=True,
    sfs=9.5,
)
ax.plot([xs3[2] + W / 2, xs3[2] + W / 2], [row2 - 0.03, 1.15], color=GREY, lw=1.1)
arrow(ax, (xs3[2] + W / 2, 1.15), (6.9, 1.15), color=GREY, lw=1.1)
ax.text(
    9.9,
    1.5,
    "retroalimentação: alertas e registros",
    ha="center",
    va="bottom",
    fontsize=10,
    color=GREY,
    style="italic",
)
save(fig, "teoria_inspecao_etapas")


# =============================================================================
# 3 and 4. Background subtraction and Otsu
# =============================================================================
def make_scene(seed: int = 11):
    rng = np.random.default_rng(seed)
    h, w = 120, 160
    grad = 80 + 50 * np.linspace(0, 1, w)[None, :] + 10 * np.linspace(0, 1, h)[:, None]
    background = grad + rng.normal(0, 5, (h, w))
    frame = grad + rng.normal(0, 5, (h, w))
    yy_, xx_ = np.mgrid[0:h, 0:w]
    shadow = 32 * np.exp(-(((yy_ - 106) / 5.0) ** 2 + ((xx_ - 108) / 12.0) ** 2))
    frame = frame - shadow
    obj = np.zeros((h, w), bool)
    obj[30:110, 62:98] = True
    obj[14:30, 74:86] = True
    obj[8:15, 72:88] = True
    shade = 175 + 40 * np.sin(np.linspace(0.15, 0.85, w) * np.pi)[None, :]
    alpha = gaussian(obj.astype(float), sigma=1.2)
    frame = alpha * (shade + rng.normal(0, 5, (h, w))) + (1 - alpha) * frame
    background = np.clip(background, 0, 255)
    frame = np.clip(frame, 0, 255)
    diff = np.rint(np.abs(frame - background)).astype(np.uint8)
    return background, frame, diff


bg_img, frame_img, diff_img = make_scene()

print("== 3 teoria_subtracao_fundo")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, im, t in zip(
    axes,
    (bg_img, frame_img, diff_img),
    ("(a) Modelo de fundo", "(b) Quadro com a garrafa", "(c) Diferença absoluta"),
):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    hide_ticks(ax)
    panel(ax, t)
facts["subtracao_fundo"] = {
    "tamanho_pixels": list(diff_img.shape),
    "diferenca_media_fundo": round(float(diff_img[:6, :].mean()), 2),
    "diferenca_media_objeto": round(float(diff_img[40:100, 66:94].mean()), 2),
}
save(fig, "teoria_subtracao_fundo")

print("== 4 teoria_otsu")
hist = np.bincount(diff_img.ravel(), minlength=256)
prob = hist / hist.sum()
omega = np.cumsum(prob)
mu = np.cumsum(prob * np.arange(256))
with np.errstate(divide="ignore", invalid="ignore"):
    sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
sigma_b = np.nan_to_num(sigma_b)
t_manual = int(np.argmax(sigma_b))
t_otsu = int(threshold_otsu(diff_img))
mask = diff_img > t_otsu
facts["otsu"] = {
    "limiar_skimage": t_otsu,
    "limiar_maximo_variancia_entre_classes": t_manual,
    "variancia_entre_classes_maxima": round(float(sigma_b[t_otsu]), 3),
    "fracao_pixels_objeto": round(float(mask.mean()), 4),
}
fig, (a, b, c) = plt.subplots(
    1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1.3, 0.9]}
)
levels = np.arange(256)
a.bar(levels[: t_otsu + 1], hist[: t_otsu + 1], width=1, color=BLUE, label="classe fundo")
a.bar(levels[t_otsu + 1 :], hist[t_otsu + 1 :], width=1, color=ORANGE, label="classe objeto")
a.axvline(t_otsu + 0.5, color=RED, lw=1.8, ls="--", label=f"limiar $t={t_otsu}$")
a.set_yscale("log")
a.set_ylim(0.8, hist.max() * 80)
a.set_xlim(0, 255)
a.set_xlabel("Diferença absoluta (nível de cinza)")
a.set_ylabel("Número de pixels")
a.legend(loc="upper right", fontsize=9)
panel(a, "(a) Histograma da diferença")
b.plot(levels, sigma_b, color=GREY, lw=2)
b.plot([t_otsu], [sigma_b[t_otsu]], "o", color=RED, ms=9, zorder=3, label=f"máximo em $t={t_otsu}$")
b.axvline(t_otsu, color=RED, lw=1.2, ls="--")
b.set_xlim(0, 255)
b.set_xlabel("Limiar $t$")
b.set_ylabel(r"$\sigma_B^2(t)$", labelpad=2)
b.legend(loc="upper right", fontsize=9)
use_comma(b, "y")
panel(b, r"(b) Variância entre classes $\sigma_B^2$")
c.imshow(mask, cmap="gray")
hide_ticks(c)
panel(c, "(c) Máscara binária")
save(fig, "teoria_otsu")

# =============================================================================
# 5. Morphology
# =============================================================================
print("== 5 teoria_morfologia")
rngm = np.random.default_rng(5)
sil = np.zeros((100, 100), bool)
rr, cc = draw_polygon([92, 92, 48, 38, 38, 48], [30, 70, 70, 58, 42, 30], sil.shape)
sil[rr, cc] = True
sil[22:38, 42:58] = True
sil[10:22, 40:60] = True
holes = [
    (60, 40, 2.5),
    (70, 58, 2),
    (80, 45, 2.5),
    (55, 62, 1.5),
    (30, 50, 1.5),
    (85, 63, 2),
    (16, 50, 1.5),
]
mask5 = sil.copy()
for r0, c0, rad in holes:
    rr, cc = draw_disk((r0, c0), rad + 0.2, shape=mask5.shape)
    mask5[rr, cc] = False
far = ~dilation(sil, disk(9))
noise = np.zeros_like(sil)
cand = np.argwhere(far)
for idx in rngm.choice(len(cand), 40, replace=False):
    noise[tuple(cand[idx])] = True
for p0, p1 in [((20, 5), (20, 20)), ((52, 90), (72, 90)), ((95, 5), (80, 12))]:
    rr, cc = draw_line(*p0, *p1)
    noise[rr, cc] = True
noisy = mask5 | (noise & far)
se = disk(3)
closed = closing(noisy, se)
cleaned = opening(closed, se)
facts["morfologia"] = {
    "elemento_estruturante": "disco de raio 3 (7x7)",
    "pixels_mascara_com_furos_e_ruido": int(noisy.sum()),
    "pixels_apos_fechamento": int(closed.sum()),
    "pixels_apos_fechamento_e_abertura": int(cleaned.sum()),
    "pixels_silhueta_ideal": int(sil.sum()),
    "diferenca_para_silhueta_ideal": int((cleaned != sil).sum()),
}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})
for ax, im, t in zip(
    axes[:3],
    (noisy, closed, cleaned),
    ("(a) Máscara com defeitos", "(b) Após fechamento", "(c) Fechamento e abertura"),
):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    hide_ticks(ax)
    panel(ax, t)
ax = axes[3]
ax.imshow(se, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
pixel_grid(ax, 7, 7, color="#999999", lw=0.8)
hide_ticks(ax)
panel(ax, "(d) Elemento\nestruturante")
ax.set_xlabel("disco de raio 3\n(7×7 pixels)", fontsize=11)
save(fig, "teoria_morfologia")

# =============================================================================
# 6. Stratified 5-fold cross-validation
# =============================================================================
print("== 6 teoria_validacao_cruzada")
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
ax.add_patch(
    FancyBboxPatch(
        (1.0, 8.5),
        14.0,
        1.1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        fc="#f2f2f2",
        ec=GREY,
        lw=1.5,
        ls="--",
    )
)
ax.text(
    8.0,
    9.05,
    "Conjunto de validação do projeto (300 imagens): mantido à parte, não usado na busca",
    ha="center",
    va="center",
    fontsize=11.5,
)
ax.text(
    7.35,
    7.95,
    "Treino: 1200 imagens em 5 partes estratificadas (240 cada)",
    ha="center",
    va="center",
    fontsize=11.5,
)
bx0, pw, bh = 2.6, 1.9, 0.82
for k in range(5):
    y = 6.5 - k * 1.15
    ax.text(bx0 - 0.15, y + bh / 2, f"Rodada {k + 1}", ha="right", va="center", fontsize=11.5)
    for j in range(5):
        val = j == k
        ax.add_patch(
            Rectangle(
                (bx0 + j * pw, y),
                pw,
                bh,
                fc=ORANGE if val else "#aec7e8",
                ec="white",
                lw=2,
            )
        )
        ax.text(
            bx0 + j * pw + pw / 2,
            y + bh / 2,
            "validação" if val else "treino",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold" if val else "normal",
        )
    ax.text(
        bx0 + 5 * pw + 0.3, y + bh / 2, rf"$F_1^{{({k + 1})}}$", ha="left", va="center", fontsize=13
    )
ax.text(bx0 + 5 * pw + 0.3, 7.95, "$F_1$ da rodada", ha="left", va="center", fontsize=11.5)
ax.plot([1.0, 15.0], [0.75, 0.75], color=GREY, lw=1)
ax.text(
    8.0,
    0.25,
    r"média das 5 rodadas: $\overline{F_1}=\frac{1}{5}\sum_{k=1}^{5}F_1^{(k)}$, e seu erro padrão",
    ha="center",
    va="center",
    fontsize=12.5,
)
facts["validacao_cruzada"] = {
    "dobras": 5,
    "imagens_treino": 1200,
    "imagens_por_dobra": 240,
    "imagens_validacao_projeto": 300,
}
save(fig, "teoria_validacao_cruzada")

# =============================================================================
# 7. Grid search
# =============================================================================
print("== 7 teoria_busca_grade")
C_VALUES = [0.01, 0.1, 1, 10, 100]
G_LABELS = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
GRID = np.array(
    [
        [0.412, 0.538, 0.742, 0.861, 0.903],
        [0.538, 0.742, 0.903, 0.968, 0.962],
        [0.703, 0.884, 0.947, 0.955, 0.949],
        [0.455, 0.612, 0.790, 0.822, 0.822],
    ]
)
bi, bj = np.unravel_index(np.argmax(GRID), GRID.shape)
facts["busca_grade"] = {
    "C": C_VALUES,
    "gamma": [1e-4, 1e-3, 1e-2, 1e-1],
    "melhor_C": C_VALUES[bj],
    "melhor_gamma": [1e-4, 1e-3, 1e-2, 1e-1][bi],
    "melhor_F1_medio": float(GRID[bi, bj]),
    "n_combinacoes": int(GRID.size),
    "n_ajustes_com_5_dobras": int(GRID.size * 5),
}
fig, ax = plt.subplots(figsize=(8.2, 4.6))
im = ax.imshow(GRID, cmap="Blues", vmin=0.4, vmax=1.0, origin="lower", aspect="auto")
for (i, j), v in np.ndenumerate(GRID):
    ax.text(
        j,
        i,
        comma(v, 3),
        ha="center",
        va="center",
        fontsize=11,
        color="white" if v > 0.78 else "black",
    )
ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=RED, lw=3.5))
ax.text(
    bj, bi + 0.33, "melhor", ha="center", va="center", fontsize=9.5, color="white", weight="bold"
)
ax.set_xticks(range(5), [f"{c:g}".replace(".", ",") for c in C_VALUES])
ax.set_yticks(range(4), G_LABELS)
ax.set_xlabel("$C$")
ax.set_ylabel(r"$\gamma$")
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("$F_1$ médio na validação cruzada")
cb.ax.yaxis.set_major_formatter(COMMA_FORMATTER)
save(fig, "teoria_busca_grade", width=0.75)

# =============================================================================
# 8. One-standard-error rule
# =============================================================================
print("== 8 teoria_um_erro_padrao")
labels8 = ["6", "9", "12", "18"]
mean8 = np.array([0.962, 0.965, 0.964, 0.958])
se8 = np.array([0.004, 0.004, 0.0035, 0.005])
best = int(np.argmax(mean8))
limit = mean8[best] - se8[best]
chosen = int(np.argmax(mean8 >= limit))
facts["um_erro_padrao"] = {
    "candidatos": labels8,
    "F1_medio": mean8.tolist(),
    "erro_padrao": se8.tolist(),
    "melhor": labels8[best],
    "limite_melhor_menos_1EP": round(float(limit), 4),
    "escolhido": labels8[chosen],
}
fig, ax = plt.subplots(figsize=(7.4, 4.4))
xs8 = np.arange(4)
ax.errorbar(
    xs8,
    mean8,
    yerr=se8,
    fmt="o",
    color=BLUE,
    ms=8,
    capsize=5,
    lw=1.8,
    label="$F_1$ médio $\\pm$ 1 erro padrão",
)
ax.axhline(limit, color=RED, ls="--", lw=1.6, label="melhor $-$ 1 erro padrão")
ax.axhspan(limit, 0.975, color=RED, alpha=0.06)
ax.plot(
    [best], [mean8[best]], "*", color=ORANGE, ms=17, mec="black", zorder=4, label="melhor média"
)
ax.plot(
    [chosen],
    [mean8[chosen]],
    "o",
    ms=20,
    mfc="none",
    mec=GREEN,
    mew=2.8,
    zorder=5,
    label="escolhido (mais simples na faixa)",
)
ax.set_xticks(xs8, labels8)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0.950, 0.973)
ax.set_xlabel("Número de orientações (complexidade crescente $\\rightarrow$)")
ax.set_ylabel("$F_1$ médio na validação cruzada")
ax.grid(alpha=0.3)
use_comma(ax, "y")
ax.text(
    3.45,
    0.9507,
    "valores ilustrativos",
    ha="right",
    va="bottom",
    fontsize=9.5,
    style="italic",
    color=GREY,
)
from matplotlib.lines import Line2D  # noqa: E402

ax.legend(
    handles=[
        Line2D(
            [], [], marker="o", color=BLUE, ms=7, lw=1.8, label="$F_1$ médio $\\pm$ 1 erro padrão"
        ),
        Line2D([], [], color=RED, ls="--", lw=1.6, label="melhor $-$ 1 erro padrão"),
        Line2D([], [], marker="*", color=ORANGE, mec="black", ms=12, lw=0, label="melhor média"),
        Line2D(
            [],
            [],
            marker="o",
            mfc="none",
            mec=GREEN,
            mew=2.2,
            ms=11,
            lw=0,
            label="escolhido (mais simples na faixa)",
        ),
    ],
    loc="lower left",
    fontsize=9,
)
save(fig, "teoria_um_erro_padrao", width=0.7)

# =============================================================================
# 9. HOG steps
# =============================================================================
print("== 9 teoria_hog_passos")
yy9, xx9 = np.mgrid[0:32, 0:32]
img9 = 40 + 160 * ((yy9 - 15.5) ** 2 + (xx9 - 15.5) ** 2 <= 10**2)
img9 = gaussian(img9.astype(float), sigma=0.8)
gx = np.zeros_like(img9)
gy = np.zeros_like(img9)
gx[:, 1:-1] = img9[:, 2:] - img9[:, :-2]
gy[1:-1, :] = img9[2:, :] - img9[:-2, :]
mag = np.hypot(gx, gy)
ang = np.rad2deg(np.arctan2(gy, gx)) % 180
CELL = (1, 1)
cr, cc_ = CELL
sl = (slice(cr * 8, cr * 8 + 8), slice(cc_ * 8, cc_ * 8 + 8))
bins = np.minimum((ang[sl] // 30).astype(int), 5)
hog_cell = np.bincount(bins.ravel(), weights=mag[sl].ravel(), minlength=6)
facts["hog_passos"] = {
    "imagem": "32x32, celulas 8x8",
    "celula_destacada_linha_coluna": [cr, cc_],
    "histograma_6_bins_30graus": [round(float(v), 1) for v in hog_cell],
    "bin_dominante": f"{int(np.argmax(hog_cell)) * 30}-{int(np.argmax(hog_cell)) * 30 + 30} graus",
    "magnitude_maxima": round(float(mag.max()), 1),
}
fig, axes = plt.subplots(2, 2, figsize=(9, 8.2), layout="constrained")
(a, b), (c, d) = axes
a.imshow(img9, cmap="gray", vmin=0, vmax=255)
for k in range(0, 33, 8):
    a.axhline(k - 0.5, color=ORANGE, lw=0.8)
    a.axvline(k - 0.5, color=ORANGE, lw=0.8)
a.add_patch(Rectangle((cc_ * 8 - 0.5, cr * 8 - 0.5), 8, 8, fill=False, ec=RED, lw=2.5))
panel(a, "(a) Imagem e células 8×8")
b.imshow(mag, cmap="magma")
panel(b, "(b) Magnitude do gradiente")
ori = np.ma.masked_where(mag < 0.1 * mag.max(), ang)
cmap_o = plt.get_cmap("hsv").copy()
cmap_o.set_bad("#dddddd")
c.imshow(ori, cmap=cmap_o, vmin=0, vmax=180)
sub = np.s_[1:31:2, 1:31:2]
Ys, Xs = np.mgrid[0:32, 0:32]
m_ = mag[sub] > 0.15 * mag.max()
th = np.deg2rad(ang[sub])
c.quiver(
    Xs[sub][m_],
    Ys[sub][m_],
    np.cos(th)[m_],
    np.sin(th)[m_],
    angles="xy",
    scale_units="xy",
    scale=0.45,
    pivot="mid",
    headlength=0,
    headaxislength=0,
    width=0.007,
    color="black",
)
c.add_patch(Rectangle((cc_ * 8 - 0.5, cr * 8 - 0.5), 8, 8, fill=False, ec=RED, lw=2.5))
panel(c, "(c) Orientação (0° a 180°)")
for ax in (a, b, c):
    hide_ticks(ax)
sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 180))
cb = fig.colorbar(sm, ax=c, fraction=0.046, pad=0.03, ticks=[0, 60, 120, 180])
cb.ax.set_yticklabels(["0°", "60°", "120°", "180°"])
d.bar(range(6), hog_cell, color=BLUE, edgecolor="black", width=0.8)
d.set_xticks(
    range(6), [f"{30 * k}–{30 * k + 30}" for k in range(6)], fontsize=9.5, rotation=40, ha="right"
)
d.set_xlabel("Intervalo de orientação (°)")
d.set_ylabel("Soma das magnitudes")
d.set_ylim(0, hog_cell.max() * 1.1)
panel(d, "(d) Histograma da célula destacada")
use_comma(d, "y")
save(fig, "teoria_hog_passos", width=0.8)

# =============================================================================
# 10. LBP step by step
# =============================================================================
print("== 10 teoria_lbp_passos")
centre = 50
nb = [62, 70, 41, 38, 55, 90, 47, 51]  # clockwise from top-left
ring = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
bits = [1 if v >= centre else 0 for v in nb]
weights = [2**k for k in range(8)]
code = sum(b_ * w_ for b_, w_ in zip(bits, weights))


def transitions(bs):
    return sum(bs[i] != bs[(i + 1) % len(bs)] for i in range(len(bs)))


ex_uni = [0, 0, 0, 1, 1, 1, 1, 0]
ex_non = [0, 1, 0, 1, 0, 0, 1, 0]
facts["lbp_passos"] = {
    "centro": centre,
    "vizinhos_horario_a_partir_do_canto_superior_esquerdo": nb,
    "bits": "".join(map(str, bits)),
    "pesos": weights,
    "codigo_decimal": code,
    "transicoes_do_codigo": transitions(bits),
    "exemplo_uniforme": {"padrao": "".join(map(str, ex_uni)), "transicoes": transitions(ex_uni)},
    "exemplo_nao_uniforme": {
        "padrao": "".join(map(str, ex_non)),
        "transicoes": transitions(ex_non),
    },
}


def grid3(ax, cells, centre_text, fills, fontsize=14):
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    for (r_, c_), text, fc in zip(ring, cells, fills):
        ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, fc=fc, ec="black", lw=1.2))
        ax.text(c_, r_, text, ha="center", va="center", fontsize=fontsize)
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fc="#ffe9a8", ec="black", lw=1.2))
    ax.text(1, 1, centre_text, ha="center", va="center", fontsize=fontsize, weight="bold")
    ax.axis("off")


fig = plt.figure(figsize=(12.5, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 0.7], hspace=0.2, wspace=0.12)
ax = fig.add_subplot(gs[0, 0])
grid3(ax, [str(v) for v in nb], str(centre), ["#eef3f8"] * 8)
panel(ax, "(a) Vizinhança 3×3")
ax = fig.add_subplot(gs[0, 1])
grid3(ax, [str(b_) for b_ in bits], "", [PALE_ORANGE if b_ else "#eeeeee" for b_ in bits])
ax.text(1, 1, f"$\\geq {centre}$?", ha="center", va="center", fontsize=11)
panel(ax, "(b) Limiarização (vizinho ≥ centro)")
ax = fig.add_subplot(gs[0, 2])
grid3(ax, [f"{w_}" for w_ in weights], "", [PALE_BLUE] * 8)
ax.text(1, 1, "pesos\n$2^k$", ha="center", va="center", fontsize=11)
panel(ax, "(c) Pesos (sentido horário) e código")
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 20)
ax.set_ylim(0, 4.2)
ax.axis("off")
# code row
ax.text(0.0, 3.5, "Código LBP do exemplo:", ha="left", va="center", fontsize=11.5, weight="bold")
terms = " + ".join(f"{w_}" for b_, w_ in zip(bits, weights) if b_)
ax.text(
    0.0,
    2.6,
    f"bits = {''.join(map(str, bits))}   →   {terms} = {code}",
    ha="left",
    va="center",
    fontsize=12.5,
)
ax.text(
    0.0,
    1.8,
    f"código LBP = {code}; {transitions(bits)} transições (padrão não uniforme)",
    ha="left",
    va="center",
    fontsize=11.5,
    color=GREY,
)
ax.text(
    11.0,
    3.5,
    "(d) Padrões uniformes (≤ 2 transições, leitura circular):",
    ha="left",
    va="center",
    fontsize=11.5,
    weight="bold",
)
for row, (bs, label, colr) in enumerate(
    [(ex_uni, "uniforme", GREEN), (ex_non, "não uniforme", RED)]
):
    y = 2.55 - row * 1.2
    x0 = 11.0
    for k, b_ in enumerate(bs):
        ax.add_patch(
            Rectangle(
                (x0 + k * 0.5, y - 0.22),
                0.5,
                0.44,
                fc=PALE_ORANGE if b_ else "#eeeeee",
                ec="black",
                lw=1,
            )
        )
        ax.text(x0 + k * 0.5 + 0.25, y, str(b_), ha="center", va="center", fontsize=11)
        if k < 7 and bs[k] != bs[k + 1]:
            ax.plot([x0 + (k + 1) * 0.5] * 2, [y - 0.34, y + 0.34], color=RED, lw=2.2)
    if bs[-1] != bs[0]:
        ax.plot([x0, x0 + 4.0], [y - 0.36] * 2, color=RED, lw=1.2, ls=":")
        ax.plot([x0 + 4.0, x0 + 4.0], [y - 0.36, y - 0.25], color=RED, lw=1.2)
        ax.plot([x0, x0], [y - 0.36, y - 0.25], color=RED, lw=1.2)
    ax.text(
        x0 + 4.3,
        y,
        f"{''.join(map(str, bs))}: {transitions(bs)} transições, {label}",
        ha="left",
        va="center",
        fontsize=11,
        color=colr,
    )
save(fig, "teoria_lbp_passos")

# =============================================================================
# 11. LBP circular neighbourhoods
# =============================================================================
print("== 11 teoria_lbp_vizinhanca")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (P, R) in zip(axes, [(8, 1), (16, 2), (24, 3)]):
    ext = R + 1.5
    n = int(np.ceil(ext))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc="#f2f2f2", ec="#bbbbbb", lw=0.8))
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fc="#ffe9a8", ec="black", lw=1.4))
    ax.add_patch(Circle((0, 0), R, fill=False, ec=GREY, ls="--", lw=1.3))
    th = 2 * np.pi * np.arange(P) / P
    ax.plot(R * np.cos(th), -R * np.sin(th), "o", color=BLUE, ms=7, mec="black", zorder=3)
    ax.plot([0], [0], "s", color="black", ms=6, zorder=3)
    ax.plot([0, R], [0, 0], color=RED, lw=1.5)
    ax.text(R / 2 + 0.1, 0.12, "$R$", color=RED, ha="center", va="top", fontsize=12)
    ax.set_xlim(-n - 0.5, n + 0.5)
    ax.set_ylim(n + 0.5, -n - 0.5)
    ax.set_aspect("equal")
    hide_ticks(ax)
    panel(ax, f"({'abc'[(8, 16, 24).index(P)]}) $P={P}$, $R={R}$")
facts["lbp_vizinhanca"] = {
    "vizinhancas_P_R": [[8, 1], [16, 2], [24, 3]],
    "codigos_distintos_sem_uniformidade": [2**8, 2**16, 2**24],
}
save(fig, "teoria_lbp_vizinhanca")

# =============================================================================
# 12. LBP textures
# =============================================================================
print("== 12 teoria_lbp_texturas")
rng12 = np.random.default_rng(3)
yy, xx = np.mgrid[0:64, 0:64]
smooth = 60 + 2.2 * xx + 0.8 * yy + rng12.normal(0, 0.25, (64, 64))
rough = gaussian(rng12.normal(128, 60, (64, 64)), sigma=0.6)
textures = [("smooth", np.clip(smooth, 0, 255)), ("rough", np.clip(rough, 0, 255))]
hists12 = {}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
for row, (name, tex) in enumerate(textures):
    lbp = local_binary_pattern(tex.astype(np.uint8), 8, 1, method="uniform")
    h12 = np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size
    hists12[name] = h12
    axes[row, 0].imshow(tex, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].imshow(lbp, cmap="viridis", vmin=0, vmax=9)
    axes[row, 2].bar(range(10), h12, color=BLUE if row == 0 else ORANGE, edgecolor="black")
    axes[row, 2].set_xticks(range(10), [str(k) if k < 9 else "9" for k in range(10)])
    axes[row, 2].set_ylim(0, 0.85)
    axes[row, 2].set_xlabel("Código LBP (9 = não uniforme)")
    axes[row, 2].set_ylabel("Frequência relativa")
    use_comma(axes[row, 2], "y")
    hide_ticks(axes[row, 0])
    hide_ticks(axes[row, 1])
    lab = "abcdef"[row * 3 : row * 3 + 3]
    tname = "suave" if row == 0 else "rugosa"
    axes[row, 0].set_title(f"({lab[0]}) Textura {tname}")
    axes[row, 1].set_title(f"({lab[1]}) Mapa LBP")
    axes[row, 2].set_title(f"({lab[2]}) Histograma")
fig.subplots_adjust(hspace=0.35, wspace=0.25)
facts["lbp_texturas"] = {
    k: {
        "histograma_10_codigos": [round(float(v), 3) for v in h],
        "fracao_nao_uniformes": round(float(h[9]), 3),
    }
    for k, h in hists12.items()
}
save(fig, "teoria_lbp_texturas")

# =============================================================================
# 13. HSV
# =============================================================================
print("== 13 teoria_hsv")
yy, xx = np.mgrid[-1:1:500j, -1:1:500j]
rad = np.hypot(xx, yy)
hue = (np.arctan2(-yy, xx) / (2 * np.pi)) % 1.0
wheel = hsv_to_rgb(np.dstack([hue, np.clip(rad, 0, 1), np.ones_like(rad)]))
wheel[rad > 1] = 1.0
rng13 = np.random.default_rng(13)
img13 = np.ones((60, 60, 3))
img13[14:24, 22:38] = hsv_to_rgb(np.array([0.0, 0.85, 0.9]))
img13[24:58, 20:40] = hsv_to_rgb(np.array([0.55, 0.12, 0.98]))
img13[34:46, 20:40] = hsv_to_rgb(np.array([0.62, 0.9, 0.8]))
img13[48:52, 20:40] = hsv_to_rgb(np.array([0.33, 0.8, 0.7]))
img13 = np.clip(img13 + rng13.normal(0, 0.015, img13.shape), 0, 1)
hsv13 = cv2.cvtColor((img13 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
h_deg = hsv13[:, :, 0].astype(float) * 2
sel = hsv13[:, :, 1] > 40
hist13, _ = np.histogram(h_deg[sel], bins=36, range=(0, 360))
frac13 = hist13 / hist13.sum()
top = np.argsort(frac13)[::-1][:3]
facts["hsv"] = {
    "bins": 36,
    "largura_bin_graus": 10,
    "criterio": "pixels com saturacao > 40/255",
    "bins_mais_cheios": [
        {"intervalo_graus": [int(k * 10), int(k * 10 + 10)], "fracao": round(float(frac13[k]), 3)}
        for k in top
    ],
}
fig = plt.figure(figsize=(14, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.8, 1.8], wspace=0.2)
a = fig.add_subplot(gs[0])
a.imshow(wheel, extent=(-1, 1, -1, 1))
for deg, name in [
    (0, "vermelho"),
    (60, "amarelo"),
    (120, "verde"),
    (180, "ciano"),
    (240, "azul"),
    (300, "magenta"),
]:
    t = np.deg2rad(deg)
    ha = "left" if np.cos(t) > 0.1 else ("right" if np.cos(t) < -0.1 else "center")
    va = "bottom" if np.sin(t) > 0.1 else ("top" if np.sin(t) < -0.1 else "center")
    a.text(1.07 * np.cos(t), 1.07 * np.sin(t), f"{deg}°\n{name}", ha=ha, va=va, fontsize=10.5)
a.annotate(
    "",
    xy=(0.62, -0.62),
    xytext=(0.05, -0.05),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.3),
)
a.text(0.1, -0.7, "$S$ cresce", fontsize=10, ha="left", va="top")
a.text(0, -1.62, "centro: branco ($S=0$)", fontsize=10.5, ha="center", va="top")
a.set_xlim(-1.7, 1.7)
a.set_ylim(-1.8, 1.55)
a.axis("off")
panel(a, "(a) Matiz (ângulo) e saturação")
b = fig.add_subplot(gs[1])
b.imshow(img13)
hide_ticks(b)
panel(b, "(b) Imagem sintética")
c = fig.add_subplot(gs[2])
centres = np.arange(36) * 10 + 5
c.bar(
    centres,
    frac13,
    width=10,
    color=[hsv_to_rgb([cc / 360, 1, 1]) for cc in centres],
    edgecolor="black",
    lw=0.6,
)
c.set_xlim(0, 360)
c.set_xticks(range(0, 361, 60), [f"{k}°" for k in range(0, 361, 60)])
c.set_xlabel("Matiz (intervalos de 10°)")
c.set_ylabel("Fração dos pixels coloridos")
use_comma(c, "y")
panel(c, "(c) Histograma de matiz (36 intervalos)")
save(fig, "teoria_hsv")

# =============================================================================
# 14. KNN: k = 1, 7, 31
# =============================================================================
print("== 14 teoria_knn_k")
X14, y14 = make_moons(n_samples=200, noise=0.3, random_state=0)
gx14, gy14 = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2.1, 300))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
facts["knn_k"] = {}
for ax, k, lab in zip(axes, (1, 7, 31), "abc"):
    knn = KNeighborsClassifier(k).fit(X14, y14)
    zz = knn.predict(np.c_[gx14.ravel(), gy14.ravel()]).reshape(gx14.shape)
    ax.contourf(gx14, gy14, zz, levels=[-0.5, 0.5, 1.5], colors=[PALE_BLUE, PALE_ORANGE])
    ax.scatter(*X14[y14 == 0].T, s=22, color=BLUE, edgecolor="black", lw=0.4, label="Classe A")
    ax.scatter(
        *X14[y14 == 1].T,
        s=22,
        color=ORANGE,
        edgecolor="black",
        lw=0.4,
        marker="s",
        label="Classe B",
    )
    ax.set_xlabel("Atributo $x_1$")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.6, 2.1)
    use_comma(ax)
    panel(ax, f"({lab}) $k={k}$")
    facts["knn_k"][f"k={k}"] = {"acuracia_treino": round(float(knn.score(X14, y14)), 3)}
axes[0].set_ylabel("Atributo $x_2$")
axes[0].legend(loc="upper left", fontsize=9)
save(fig, "teoria_knn_k")

# =============================================================================
# 15. SVM margin and the C parameter
# =============================================================================
print("== 15 teoria_svm_margem")
rng15 = np.random.default_rng(7)
pos = rng15.normal([1.5, 1.5], 0.5, (15, 2))
neg = rng15.normal([4.5, 4.0], 0.5, (15, 2))
Xs15 = np.vstack([pos, neg])
ys15 = np.array([1] * 15 + [-1] * 15)
ov_pos = rng15.normal([1.8, 1.8], 1.0, (40, 2))
ov_neg = rng15.normal([3.6, 3.6], 1.0, (40, 2))
Xo = np.vstack([ov_pos, ov_neg])
yo = np.array([1] * 40 + [-1] * 40)
gx15, gy15 = np.meshgrid(np.linspace(-0.5, 6.5, 300), np.linspace(-0.5, 6.5, 300))


def draw_svm(ax, model, X, y, c1="Classe $+1$", c2="Classe $-1$"):
    zz = model.decision_function(np.c_[gx15.ravel(), gy15.ravel()]).reshape(gx15.shape)
    ax.contourf(gx15, gy15, zz > 0, levels=1, colors=[PALE_ORANGE, PALE_BLUE], alpha=0.6)
    ax.contour(
        gx15,
        gy15,
        zz,
        levels=[-1, 0, 1],
        colors="k",
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 1.8, 1.2],
    )
    ax.scatter(*X[y == 1].T, s=34, color=BLUE, edgecolor="black", lw=0.4, label=c1, zorder=3)
    ax.scatter(
        *X[y == -1].T, s=34, color=ORANGE, edgecolor="black", lw=0.4, marker="s", label=c2, zorder=3
    )
    sv = model.support_vectors_
    ax.scatter(
        sv[:, 0],
        sv[:, 1],
        s=140,
        facecolors="none",
        edgecolors="black",
        lw=1.4,
        label="Vetores de suporte",
        zorder=4,
    )
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Atributo $x_1$")
    use_comma(ax)


fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), layout="constrained")
hard = SVC(kernel="linear", C=1e6).fit(Xs15, ys15)
draw_svm(axes[0], hard, Xs15, ys15)
w_ = hard.coef_[0]
margin_hard = 2 / np.linalg.norm(w_)
wn = w_ / np.linalg.norm(w_)
x_mid = 0.5 * (Xs15[ys15 == 1].mean(0) + Xs15[ys15 == -1].mean(0))
perp = np.array([-wn[1], wn[0]])
anchor = x_mid + 1.6 * perp
p0 = anchor - (hard.decision_function([anchor])[0] + 1) * w_ / (w_ @ w_)
p1 = p0 + 2 * w_ / (w_ @ w_)
axes[0].annotate(
    "", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8), zorder=6
)
axes[0].text(
    *(0.5 * (p0 + p1) + 0.45 * perp),
    "margem\n$2/\\|\\mathbf{w}\\|$",
    color=RED,
    fontsize=11,
    ha="left",
    va="center",
)
axes[0].set_ylabel("Atributo $x_2$")
# Legend below the panels, so that it does not cover the points
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
panel(axes[0], "(a) Margem máxima")
facts["svm_margem"] = {
    "margem_dados_separaveis": round(float(margin_hard), 3),
    "vetores_suporte_separavel": int(hard.n_support_.sum()),
}
for ax, C_, lab in zip(axes[1:], (0.01, 100), "bc"):
    m_ = SVC(kernel="linear", C=C_).fit(Xo, yo)
    draw_svm(ax, m_, Xo, yo)
    errs = int((m_.predict(Xo) != yo).sum())
    ctext = "0{,}01" if C_ < 1 else "100"
    panel(ax, f"({lab}) Margem suave, $C={ctext}$")
    ax.set_yticklabels([])
    facts["svm_margem"][f"C={C_}"] = {
        "largura_margem": round(float(2 / np.linalg.norm(m_.coef_)), 3),
        "vetores_suporte": int(m_.n_support_.sum()),
        "erros_treino": errs,
    }
    ax.text(
        0.03,
        0.03,
        f"margem = {comma(2 / np.linalg.norm(m_.coef_))}\n{errs} erros de treino",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(fc="white", ec=GREY, alpha=0.9),
    )
save(fig, "teoria_svm_margem")

# =============================================================================
# 16. Tree and Random Forest partitions
# =============================================================================
print("== 16 teoria_arvore_particao")
X16, y16 = make_blobs(
    n_samples=180, centers=[(-2, -1), (2, -1.5), (0, 3)], cluster_std=1.45, random_state=4
)
gx16, gy16 = np.meshgrid(np.linspace(-6, 6, 400), np.linspace(-5, 7, 400))
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X16, y16)
forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=0).fit(X16, y16)
facts["arvore_floresta"] = {
    "arvore_profundidade": 3,
    "arvore_acuracia_treino": round(float(tree.score(X16, y16)), 3),
    "arvore_folhas": int(tree.get_n_leaves()),
    "floresta_arvores": 100,
    "floresta_acuracia_treino": round(float(forest.score(X16, y16)), 3),
}
pale3 = ListedColormap([PALE_BLUE, PALE_ORANGE, "#d5ecd5"])
dark3 = [BLUE, ORANGE, GREEN]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
for ax, model, t in zip(
    axes,
    (tree, forest),
    ("(a) Árvore de decisão (profundidade 3)", "(b) Floresta aleatória (100 árvores)"),
):
    zz = model.predict(np.c_[gx16.ravel(), gy16.ravel()]).reshape(gx16.shape)
    ax.contourf(
        gx16, gy16, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=[PALE_BLUE, PALE_ORANGE, "#d5ecd5"]
    )
    for k, (mk, nm) in enumerate(zip("os^", ("Classe A", "Classe B", "Classe C"))):
        ax.scatter(
            *X16[y16 == k].T, s=28, color=dark3[k], marker=mk, edgecolor="black", lw=0.4, label=nm
        )
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 7)
    ax.set_xlabel("Atributo $x_1$")
    use_comma(ax)
    panel(ax, t)
axes[0].set_ylabel("Atributo $x_2$")
axes[0].legend(loc="upper left", fontsize=9)
save(fig, "teoria_arvore_particao")

# =============================================================================
# 17. Confusion matrix
# =============================================================================
print("== 17 teoria_matriz_confusao")
CM = np.array([[45, 3, 2], [4, 38, 8], [1, 5, 44]])
names = ["Classe A", "Classe B", "Classe C"]
tp = np.diag(CM)
prec = tp / CM.sum(0)
rec = tp / CM.sum(1)
f1 = 2 * prec * rec / (prec + rec)
facts["matriz_confusao"] = {
    "matriz_linhas_real_colunas_prevista": CM.tolist(),
    "acuracia": round(float(tp.sum() / CM.sum()), 4),
    "precisao": {n: round(float(v), 4) for n, v in zip(names, prec)},
    "revocacao": {n: round(float(v), 4) for n, v in zip(names, rec)},
    "F1": {n: round(float(v), 4) for n, v in zip(names, f1)},
    "macro_precisao": round(float(prec.mean()), 4),
    "macro_revocacao": round(float(rec.mean()), 4),
    "macro_F1": round(float(f1.mean()), 4),
}
fig = plt.figure(figsize=(12.5, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.12)
ax = fig.add_subplot(gs[0])
ax.imshow(CM, cmap="Blues", vmin=0, vmax=60)
for (i, j), v in np.ndenumerate(CM):
    ax.text(
        j,
        i,
        str(v),
        ha="center",
        va="center",
        fontsize=15,
        color="white" if v > 30 else "black",
        weight="bold" if i == j else "normal",
    )
ax.set_xticks(range(3), names)
ax.set_yticks(range(3), names)
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe real")
ax.tick_params(length=0)
panel(ax, "(a) Matriz de confusão")
ax = fig.add_subplot(gs[1])
ax.axis("off")
rows = [[n, comma(p, 3), comma(r_, 3), comma(f, 3)] for n, p, r_, f in zip(names, prec, rec, f1)]
rows.append(["Média macro", comma(prec.mean(), 3), comma(rec.mean(), 3), comma(f1.mean(), 3)])
tab = ax.table(
    cellText=rows,
    colLabels=["", "Precisão", "Revocação", "$F_1$"],
    loc="center",
    cellLoc="center",
    bbox=[0.0, 0.2, 1.0, 0.65],
)
tab.auto_set_font_size(False)
tab.set_fontsize(12)
for (r_, c_), cell in tab.get_celld().items():
    cell.set_edgecolor("#999999")
    if r_ == 0:
        cell.set_facecolor("#dbe9f6")
        cell.set_text_props(weight="bold")
    if r_ == 4:
        cell.set_facecolor("#f2f2f2")
        cell.set_text_props(weight="bold")
    if c_ == 0 and r_ > 0:
        cell.set_text_props(ha="left")
ax.text(
    0.5,
    0.06,
    "Precisão $=\\mathrm{VP}/(\\mathrm{VP}+\\mathrm{FP})$;  revocação $=\\mathrm{VP}/(\\mathrm{VP}+\\mathrm{FN})$;\n$F_1=2PR/(P+R)$",
    ha="center",
    va="center",
    fontsize=11,
    color=GREY,
)
panel(ax, "(b) Métricas por classe")
save(fig, "teoria_matriz_confusao")

# =============================================================================
# 18. IoU
# =============================================================================
print("== 18 teoria_iou")


def rect_iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter, a[2] * a[3] + b[2] * b[3] - inter


def draw_iou(ax, a, b, label=True, fs=11):
    """B must be up-right (greater x and y) of A and overlap it."""
    inter, uni = rect_iou(a, b)
    ax.add_patch(Rectangle(a[:2], a[2], a[3], fc=BLUE, alpha=0.18, ec="none"))
    ax.add_patch(Rectangle(b[:2], b[2], b[3], fc=ORANGE, alpha=0.18, ec="none"))
    ax.add_patch(Rectangle(a[:2], a[2], a[3], fill=False, ec=BLUE, lw=2))
    ax.add_patch(Rectangle(b[:2], b[2], b[3], fill=False, ec=ORANGE, lw=2))
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    ax.add_patch(
        Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, hatch="///", ec=GREEN, lw=0)
    )
    ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=GREEN, lw=1.5))
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    if abs(by0 - ay0) < 1e-9:
        verts = [(ax0, ay0), (bx1, ay0), (bx1, by1), (ax0, ay1)]
    else:
        verts = [
            (ax0, ay0),
            (ax1, ay0),
            (ax1, by0),
            (bx1, by0),
            (bx1, by1),
            (bx0, by1),
            (bx0, ay1),
            (ax0, ay1),
        ]
    ax.add_patch(Polygon(verts, closed=True, fill=False, ec="black", lw=1.6, ls=(0, (4, 3))))
    ax.set_xlim(min(ax0, bx0) - 0.8, max(ax1, bx1) + 0.8)
    ax.set_ylim(min(ay0, by0) - 0.8, max(ay1, by1) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return inter / uni


fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.12)
ax = fig.add_subplot(gs[0, :2])
A18, B18 = (0, 0, 6, 4), (2, 1, 6, 4)
iou_main = draw_iou(ax, A18, B18)
inter_m, uni_m = rect_iou(A18, B18)
ax.text(0.2, -0.45, "caixa anotada", color=BLUE, fontsize=11, va="top")
ax.text(7.8, 5.45, "caixa prevista", color=ORANGE, fontsize=11, ha="right", va="bottom")
ax.set_ylim(-1.0, 6.1)
leg = [
    Patch(fc="white", ec=GREEN, hatch="///", label="interseção"),
    Patch(fc="white", ec="black", ls="--", label="união (contorno)"),
]
ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, -0.06), fontsize=9.5, frameon=False)
panel(ax, "(a) Caixas anotada e prevista")
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.text(
    0.0,
    0.7,
    r"$\mathrm{IoU}=\dfrac{\mathrm{área\ da\ interseção}}{\mathrm{área\ da\ união}}$",
    fontsize=14,
    va="center",
)
ax.text(
    0.0, 0.38, f"= {int(inter_m)}/{int(uni_m)} = {comma(inter_m / uni_m)}", fontsize=14, va="center"
)
ax.text(
    0.0,
    0.15,
    f"interseção {int(inter_m)}, união {int(uni_m)}",
    fontsize=10.5,
    va="center",
    color=GREY,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
facts["iou"] = {
    "principal": {
        "area_intersecao": int(inter_m),
        "area_uniao": int(uni_m),
        "iou": round(inter_m / uni_m, 4),
    },
    "casos": {},
}


def shifted(target):
    lo, hi = 0.0, 10.0
    for _ in range(60):
        s = (lo + hi) / 2
        i_, u_ = rect_iou((0, 0, 10, 10), (s, s / 2, 10, 10))
        lo, hi = (s, hi) if i_ / u_ > target else (lo, s)
    return (lo + hi) / 2


for col, (target, lab) in enumerate(zip((0.3, 0.6, 0.9), "bcd")):
    ax = fig.add_subplot(gs[1, col])
    s = shifted(target)
    val = draw_iou(ax, (0, 0, 10, 10), (s, s / 2, 10, 10))
    panel(ax, f"({lab}) IoU = {comma(val)}")
    facts["iou"]["casos"][lab] = round(float(val), 4)
save(fig, "teoria_iou")

# =============================================================================
# 19. p-chart
# =============================================================================
print("== 19 teoria_carta_controle")
N19, P0, K19, SHIFT_AT = 200, 0.03, 30, 24
sigma_p = np.sqrt(P0 * (1 - P0) / N19)
UCL = P0 + 3 * sigma_p
LCL = max(0.0, P0 - 3 * sigma_p)
for seed in range(200):
    r19 = np.random.default_rng(seed)
    p_true = np.where(np.arange(1, K19 + 1) >= SHIFT_AT, 0.075, P0)
    pi = r19.binomial(N19, p_true) / N19
    above = np.where(pi > UCL)[0] + 1
    if len(above) in (1, 2) and above.min() >= SHIFT_AT:
        break
facts["carta_controle"] = {
    "n_por_lote": N19,
    "lotes": K19,
    "p_barra": P0,
    "LSC": round(float(UCL), 4),
    "LIC": round(float(LCL), 4),
    "LIC_calculado_antes_do_max": round(float(P0 - 3 * sigma_p), 4),
    "semente": seed,
    "lotes_acima_do_LSC": [int(v) for v in above],
    "p_nos_lotes_acima": [round(float(pi[v - 1]), 4) for v in above],
}
fig, ax = plt.subplots(figsize=(10, 4.4))
xs19 = np.arange(1, K19 + 1)
ax.plot(xs19, pi, "-", color=GREY, lw=1.2, zorder=2)
ax.plot(xs19, pi, "o", color=BLUE, ms=6, zorder=3, label="$p_i$ (fração rejeitada)")
ax.plot(above, pi[above - 1], "o", color=RED, ms=9, zorder=4)
ax.axhline(P0, color=GREEN, lw=1.8, label=f"$\\bar p$ = {comma(P0)}")
ax.axhline(UCL, color=RED, lw=1.6, ls="--", label=f"LSC = {comma(UCL, 3)}")
ax.axhline(LCL, color=ORANGE, lw=1.6, ls="--", label=f"LIC = {comma(LCL, 3)}")
ax.axvspan(SHIFT_AT - 0.5, K19 + 0.5, color=RED, alpha=0.06)
ax.text(
    SHIFT_AT + 0.2,
    0.0,
    "deslocamento do processo",
    fontsize=9.5,
    style="italic",
    color=GREY,
    va="bottom",
)
for v in above:
    ax.annotate(
        "alerta",
        xy=(v, pi[v - 1]),
        xytext=(v - 3.0, pi[v - 1] + 0.012),
        color=RED,
        fontsize=12,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )
ax.set_xlim(0.3, K19 + 0.7)
ax.set_ylim(-0.003, max(pi.max(), UCL) + 0.03)
ax.set_xlabel("Amostra")
ax.set_ylabel("Fração rejeitada")
ax.grid(alpha=0.3)
use_comma(ax, "y")
ax.legend(loc="upper left", fontsize=9.5, ncol=2)
save(fig, "teoria_carta_controle")

# =============================================================================
# 20. Sliding alert window
# =============================================================================
print("== 20 teoria_janela_alerta")
OK, NOCAP, LOOSE, LABEL = "aprovada", "sem tampa", "tampa solta", "rótulo danificado"
cols20 = {OK: GREEN, NOCAP: RED, LOOSE: PURPLE, LABEL: ORANGE}
seq = [OK] * 20
for k in (2, 5, 12, 15, 17, 19):
    seq[k] = NOCAP
for k in (7, 10):
    seq[k] = LABEL
seq[14] = LOOSE
WIN, THR = 10, 4
alert_at = next(i for i in range(WIN - 1, 20) if seq[i - WIN + 1 : i + 1].count(NOCAP) >= THR)
facts["janela_alerta"] = {
    "garrafas": 20,
    "janela": WIN,
    "limiar": THR,
    "alerta_na_garrafa": alert_at + 1,
    "contagem_sem_tampa_na_janela": seq[alert_at - WIN + 1 : alert_at + 1].count(NOCAP),
}
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(-0.5, 20.3)
ax.set_ylim(-2.3, 3.0)
ax.axis("off")
for i, st in enumerate(seq):
    ax.add_patch(
        FancyBboxPatch(
            (i + 0.1, 0.0),
            0.62,
            1.15,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            fc=cols20[st],
            ec="black",
            lw=1,
        )
    )
    ax.add_patch(Rectangle((i + 0.27, 1.15), 0.3, 0.3, fc=cols20[st], ec="black", lw=1))
    if st != NOCAP:
        ax.add_patch(Rectangle((i + 0.22, 1.45), 0.4, 0.2, fc="#555555", ec="black", lw=1))
    ax.text(i + 0.41, -0.3, str(i + 1), ha="center", va="top", fontsize=9, color=GREY)
x0w, x1w = alert_at - WIN + 1 + 0.02, alert_at + 0.8
yb = -0.85
ax.plot([x0w, x0w, x1w, x1w], [yb + 0.2, yb, yb, yb + 0.2], color="black", lw=1.8)
ax.text(
    (x0w + x1w) / 2,
    yb - 0.15,
    f"janela deslizante: últimas {WIN} garrafas",
    ha="center",
    va="top",
    fontsize=11.5,
)
ax.annotate(
    f"alerta: {THR} de {WIN} garrafas\ncom o mesmo defeito (sem tampa)",
    xy=(alert_at + 0.41, 1.95),
    xytext=(alert_at - 3.2, 2.45),
    ha="center",
    va="center",
    color=RED,
    fontsize=12,
    weight="bold",
    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8),
)
handles = [Patch(fc=c_, ec="black", label=n) for n, c_ in cols20.items()]
ax.legend(
    handles=handles,
    loc="lower left",
    ncol=4,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.0, -0.04),
)
ax.text(0.0, 2.55, "ordem de inspeção $\\rightarrow$", fontsize=10.5, color=GREY, va="center")
save(fig, "teoria_janela_alerta")

# =============================================================================
# 21. Integration architecture
# =============================================================================
print("== 21 teoria_integracao")
fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_xlim(0, 17)
ax.set_ylim(0, 10)
ax.axis("off")
ax.add_patch(
    FancyBboxPatch(
        (0.2, 1.0),
        4.6,
        8.0,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        fc="#f7f7f7",
        ec=GREY,
        lw=1.6,
        ls="--",
    )
)
ax.text(2.5, 8.55, "Estação de inspeção", ha="center", va="center", fontsize=12.5, weight="bold")
box(ax, 0.7, 5.6, 3.6, 1.7, "Câmera", "e iluminação", fc="#dbe9f6", ec=BLUE)
box(
    ax,
    0.7,
    2.0,
    3.6,
    2.6,
    "Computador",
    "com o modelo\n(classificação e decisão)",
    fc="#dbe9f6",
    ec=BLUE,
)
arrow(ax, (2.5, 5.55), (2.5, 4.65))
ys = [8.0, 5.0, 2.0]
heads = ["(a) Controle da linha", "(b) Monitoramento", "(c) Integração com outros sistemas"]
for y, h in zip(ys, heads):
    ax.text(5.3, y + 1.15, h, ha="left", va="center", fontsize=11.5, weight="bold")
# connectors from the computer box
ax.plot([4.3, 5.1], [3.3, 3.3], color="black", lw=1.8)
ax.plot([5.1, 5.1], [min(ys), max(ys)], color="black", lw=1.8)
for y in ys:
    ax.plot([5.1, 5.5], [y, y], color="black", lw=1.8)
# row a
arrow(ax, (5.5, ys[0]), (8.4, ys[0]))
ax.text(6.95, ys[0] + 0.22, "sinal digital", ha="center", va="bottom", fontsize=10)
box(ax, 8.5, ys[0] - 0.7, 2.6, 1.4, "CLP", "controlador", fc="#fde3cf", ec=ORANGE)
arrow(ax, (11.15, ys[0]), (12.3, ys[0]))
box(ax, 12.4, ys[0] - 0.7, 4.3, 1.4, "Atuador de rejeição", "na esteira", fc="#d8efd8", ec=GREEN)
ax.text(
    9.8,
    ys[0] - 0.95,
    "OPC UA / Modbus TCP / E/S digital",
    ha="center",
    va="top",
    fontsize=10,
    style="italic",
    color=GREY,
)
# row b
arrow(ax, (5.5, ys[1]), (8.4, ys[1]))
ax.text(6.95, ys[1] + 0.22, "publica", ha="center", va="bottom", fontsize=10)
box(ax, 8.5, ys[1] - 0.7, 2.6, 1.4, "Broker MQTT", None, fc="#fde3cf", ec=ORANGE)
arrow(ax, (11.15, ys[1]), (12.3, ys[1]))
ax.text(11.72, ys[1] + 0.22, "assina", ha="center", va="bottom", fontsize=10)
box(
    ax,
    12.4,
    ys[1] - 0.7,
    4.3,
    1.4,
    "Painéis e supervisório",
    "publish/subscribe",
    fc="#d8efd8",
    ec=GREEN,
)
# row c
arrow(ax, (5.5, ys[2]), (12.3, ys[2]))
ax.text(8.9, ys[2] + 0.22, "webhook HTTP (POST)", ha="center", va="bottom", fontsize=10)
box(
    ax,
    12.4,
    ys[2] - 0.7,
    4.3,
    1.4,
    "Outros sistemas",
    "e-mail, ordens de manutenção",
    fc="#d8efd8",
    ec=GREEN,
)
save(fig, "teoria_integracao")

# =============================================================================
print("== Summary")
args.facts.parent.mkdir(parents=True, exist_ok=True)
args.facts.write_text(json.dumps(facts, indent=2, ensure_ascii=False))
print(json.dumps(facts, indent=2, ensure_ascii=False))
print(f"\n{len(written)} files written:")
for p in written:
    print("  ", p)
print("facts:", args.facts)
