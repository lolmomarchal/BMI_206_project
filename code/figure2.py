# fig 2 for our project from the network stuff paper, degree dists for the 5 reaction graphs
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

path = "../data/12859_2019_2897_MOESM2_ESM.xlsx"
names = ["E. coli", "B. subtilis", "G. metallireducens", "K. pneumoniae", "S. cerevisiae"]

wb = openpyxl.load_workbook(path, read_only=True)  # this takes a bit to load

mats = []
for ws in wb.worksheets:
    rows = list(ws.iter_rows(values_only=True))
    # drop header row + first col, those are just rxn names
    mats.append(np.array([r[1:] for r in rows[1:]], dtype=float))


def get_pk(deg):
    # fraction of nodes w/ each degree, ignoring 0
    n = len(deg)
    deg = deg[deg > 0].astype(int)
    k, cnt = np.unique(deg, return_counts=True)
    return k, cnt / n


# (what, color, marker)
kinds = [("total", "black", "o"), ("in", "red", "s"), ("out", "blue", "^")]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i in range(5):
    A = mats[i]
    ax = axes[i]

    indeg = A.sum(axis=0)   # col sums
    outdeg = A.sum(axis=1)  # row sums
    degs = {"total": indeg + outdeg, "in": indeg, "out": outdeg}

    gam = {}
    for what, col, mk in kinds:
        k, p = get_pk(degs[what])
        ax.scatter(k, p, c=col, marker=mk, s=14, zorder=3)

        # straight line fit on the log-log
        m, b = np.polyfit(np.log10(k), np.log10(p), 1)
        gam[what] = m
        x = np.array([k.min(), k.max()])
        ax.plot(x, 10 ** (b + m * np.log10(x)), color=col, lw=1.3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** -0.2, 10 ** 2.5)
    ax.set_ylim(1e-4, 1)

    ax.text(0.05, 0.05, "abcde"[i], transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.text(0.17, 0.05, names[i], transform=ax.transAxes, fontsize=14,
            fontweight="bold", style="italic")

    for j, (what, col, _) in enumerate(kinds):
        #HOW TO GET NUMBERS TO LINE UP??? THEY ROUND SOMETIMES BUT OTHER TIMES THEY JUST TRUNCATE???? Should we round or truncate? does it matter?
        g = np.trunc(gam[what] * 100) / 100
        ax.text(0.97, 0.93 - 0.09 * j, "$\\gamma$ (%s) = %.2f" % (what, g),
                transform=ax.transAxes, color=col, ha="right", va="top",
                fontsize=10, fontweight="bold")

    if i in (0, 3):
        ax.set_ylabel("Pr($k$)", fontsize=13, fontweight="bold")
    if i in (2, 3, 4):
        ax.set_xlabel("degree ($k$)", fontsize=13, fontweight="bold")

# last spot is empty so legend goes there
axes[5].axis("off")
handles = [
    Line2D([], [], color="black", marker="o", ls="", label="Total degree"),
    Line2D([], [], color="red", marker="s", ls="", label="In-degree"),
    Line2D([], [], color="blue", marker="^", ls="", label="Out-degree"),
    Line2D([], [], color="black", label="Regression of total degrees"),
    Line2D([], [], color="red", label="Regression of in-degrees"),
    Line2D([], [], color="blue", label="Regression of out-degrees"),
]
axes[5].legend(handles=handles, loc="center", frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig("figure2.png", dpi=200)
