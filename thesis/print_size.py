"""Keeps the text of the thesis figures readable on the printed page.

The figures are drawn in inches larger than the page and LaTeX scales them down to the
width given in \\includegraphics. A 10 pt label in a 15 inch wide figure printed 16 cm
wide ends up below 5 pt. `fit_text_to_print` measures the final (tight) width of the
figure, works out that scale and enlarges every text smaller than MIN_PRINT_PT on paper.
"""

from __future__ import annotations

from matplotlib.text import Text

TEXTWIDTH_IN = 16 / 2.54
"""Width of the text block of the thesis (A4, 3 cm + 2 cm margins)."""

MIN_PRINT_PT = 8.0
"""Smallest font size accepted on the printed page."""


def printed_scale(fig, width: float) -> float:
    """Printed size / drawn size of the figure when included at `width` x \\textwidth."""
    renderer = fig.canvas.get_renderer()
    return width * TEXTWIDTH_IN / fig.get_tightbbox(renderer).width


def fit_text_to_print(fig, width: float = 1.0, relayout: bool = False) -> float:
    """Enlarges every text of `fig` that would print below MIN_PRINT_PT.

    `width` is the fraction of \\textwidth used in LaTeX. With `relayout`, tight_layout is
    run again after the fonts change (only for figures laid out with tight_layout). The
    width of the figure grows with the fonts, so the check is repeated until it holds.
    Returns the smallest printed font size, in points.
    """
    for _ in range(4):
        floor = MIN_PRINT_PT / printed_scale(fig, width)
        small = [
            t
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < floor - 0.01
        ]
        if not small:
            break
        for t in small:
            t.set_fontsize(floor * 1.02)
        if relayout:
            fig.tight_layout()
    sizes = [
        t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
    ]
    return min(sizes) * printed_scale(fig, width) if sizes else float("nan")
