"""Side-by-side comparisons: original Fig. 2 (extracted from the paper PDF) vs our two recreations."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mi

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "../code")
INK, MUTED = "#222222", "#6B7280"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

orig = mi.imread(os.path.join(HERE, "original_figure2_from_paper.png"))
JOBS = [
    ("1_original_vs_supplement_recreation.png", os.path.join(CODE, "figure2.png"),
     "Recreation from the paper's supplementary matrices",
     "Degrees read from the adjacency matrices in Additional file 2",
     "γ: 14 of 15 values identical to the paper.  The one difference is E. coli in-degree (−1.31 here, −1.32 printed)."),
    ("2_original_vs_raw_data_recreation.png", os.path.join(CODE, "figure2_rebuilt.png"),
     "Recreation from the raw BiGG models",
     "Graphs rebuilt from the original metabolic models (currency, transport and boundary rules inferred)",
     "γ: 11 of 15 identical. Differences: E. coli in −1.31 (−1.32), K. pneumoniae in −1.28 (−1.31), out −1.48 (−1.47), S. cerevisiae in −1.09 (−1.08).\n"
     "Node counts differ slightly (E. coli 1257 vs 1251, K. pneumoniae 1146 vs 1137, S. cerevisiae 890 vs 881)."),
]
for fname, path, title, sub, cap in JOBS:
    rec = mi.imread(path)
    fig = plt.figure(figsize=(20, 8.6), facecolor="white")
    for k, (im, t, s) in enumerate([(orig, "Original", "Kim et al. 2019, Fig. 2 (extracted from the PDF)"), (rec, title, sub)]):
        ax = fig.add_axes([0.02 + k * 0.495, 0.13, 0.465, 0.74]); ax.imshow(im); ax.axis("off")
        fig.text(0.02 + k * 0.495, 0.945, t, fontsize=20, fontweight="bold", color=INK, va="center")
        fig.text(0.02 + k * 0.495, 0.90, s, fontsize=12, color=MUTED, va="center")
    fig.text(0.02, 0.055, cap, fontsize=12.5, color=INK, va="center", linespacing=1.5)
    fig.savefig(os.path.join(HERE, fname), dpi=170, facecolor="white"); plt.close(fig); print("wrote", fname)
