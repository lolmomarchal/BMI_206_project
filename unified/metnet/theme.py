"""One visual theme for every figure, taken from the slide deck.

Deck palette: title blue #3F619E, accent teal #00665F, ink #172932, section-slide blue #E3EAF3.
Titles/labels use Poppins (the deck font); DejaVu Sans is the per-glyph fallback (Greek, math).
"""
from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager as fm

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BLUE, TEAL, INK, MUTED = "#3F619E", "#00665F", "#172932", "#405761"
SOFT_BLUE, PALE_BLUE, GRID = "#899EC3", "#E3EAF3", "#D5DCE8"
RED, AMBER, PURPLE, SKY = "#C8453B", "#E08A1E", "#8E5EA2", "#5BA3D9"

# degree types (same marker/colour language as the paper's Fig. 2 and the deck's figures)
KIND_STYLE = {
    "total": dict(color=INK, marker="o", label="Total degree"),
    "in":    dict(color=RED, marker="s", label="In-degree"),
    "out":   dict(color=BLUE, marker="^", label="Out-degree"),
}
KINDS = list(KIND_STYLE)

# categorical palette for methods / filtering tiers (validated for colour-blind separation by hue + lightness)
CAT = [BLUE, TEAL, AMBER, RED, PURPLE, SKY, INK, "#9AA5B1"]


def apply():
    import dataclasses
    for f in (ASSETS / "fonts").glob("*.ttf"):
        fm.fontManager.addfont(str(f))
    # the three files register as "Poppins", "Poppins Medium", "Poppins SemiBold": merge into one family with weights
    fm.fontManager.ttflist = [dataclasses.replace(e, name="Poppins") if "Poppins-" in e.fname else e
                              for e in fm.fontManager.ttflist]
    mpl.rcParams.update({
        "font.family": ["Poppins", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "medium", "axes.titlecolor": BLUE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax, text, x=0.04, y=0.06, **kw):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, fontweight="medium", color=INK, **kw)
