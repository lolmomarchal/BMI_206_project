"""Figure functions. Every figure uses metnet.theme (the slide-deck look)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

from . import theme as T, data, graph, regression as R

T.apply()
trunc2 = lambda x: np.trunc(x * 100) / 100
LOG_AX = dict(xscale="log", yscale="log")


def _style_axes(ax, xlim=(10 ** -0.2, 10 ** 2.6), ylim=(1e-4, 1.5)):
    ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim)
    ax.grid(True, which="major", alpha=0.5)
    ax.set_axisbelow(True)


def scatter_fit(ax, deg, kind, kmin=1, fit=True, s=14, alpha=1.0, color=None, lw=1.3, z=3):
    """Pr(k) points and the paper's OLS line for one degree type. Returns gamma (fit on k >= kmin)."""
    k, p = graph.degree_distribution(deg)
    st = T.KIND_STYLE[kind]
    c = color or st["color"]
    ax.scatter(k, p, s=s, color=c, marker=st["marker"], alpha=alpha, zorder=z, linewidths=0)
    g = np.nan
    if fit:
        g, b = R.ols_fit(deg, kmin)
        xs = np.array([max(k.min(), kmin), k.max()])
        ax.plot(xs, 10 ** (b + g * np.log10(xs)), color=c, lw=lw, zorder=z - 1)
    return g


def fig2_axes(axes5, legend_ax, degs, truncate=True):
    """The paper's Fig. 2 layout: five organism panels + legend."""
    for i, (ax, key) in enumerate(zip(axes5, data.KEYS)):
        _style_axes(ax)
        gam = {}
        for kind in T.KINDS:
            gam[kind] = scatter_fit(ax, degs[key][kind], kind)
        for j, kind in enumerate(T.KINDS):
            g = trunc2(gam[kind]) if truncate else gam[kind]
            ax.text(0.97, 0.95 - 0.095 * j, rf"$\gamma$ ({kind}) = {g:.2f}".replace("-", "−"),
                    transform=ax.transAxes, ha="right", va="top", color="black",
                    fontsize=8.5)
        ax.text(0.04, 0.05, "abcde"[i], transform=ax.transAxes, fontsize=11)
        ax.text(0.14, 0.05, data.NAME[key], transform=ax.transAxes, fontsize=10, style="italic")
        if i in (0, 3):
            ax.set_ylabel("Pr(k)")
        if i in (2, 3, 4):
            ax.set_xlabel("degree (k)")
    T.square(*axes5)
    legend_ax.axis("off")
    h = [Line2D([], [], color=T.KIND_STYLE[k]["color"], marker=T.KIND_STYLE[k]["marker"], ls="", ms=5,
                label=T.KIND_STYLE[k]["label"]) for k in T.KINDS]
    h += [Line2D([], [], color=T.KIND_STYLE[k]["color"], lw=1.3, label=f"Regression of {k} degrees") for k in T.KINDS]
    legend_ax.legend(handles=h, loc="center", fontsize=9)


def compare_with_original(degs, path, show_original=True):
    """Original Fig. 2 (extracted from the paper PDF) next to our recreation."""
    orig = mpimg.imread(T.ASSETS / "original_figure2_from_paper.png")
    if show_original:
        fig = plt.figure(figsize=(16.5, 6.4))
        sf = fig.subfigures(1, 2, wspace=0.02)
        ax = sf[0].subplots()
        ax.imshow(orig); ax.axis("off")
        right = sf[1]
    else:
        fig = plt.figure(figsize=(8.5, 6.4))
        right = fig
    axs = right.subplots(2, 3, gridspec_kw=dict(wspace=0.4, hspace=0.3)).ravel()
    fig2_axes(axs[:5], axs[5], degs)
    fig.savefig(path)
    return fig


# ------------------------------------------------------------------ filtering sensitivity
TIER_COLOR = {
    "Unfiltered": "#9AA5B1",
    "Currency filtered": T.AMBER,
    "Compartment filtered": T.SKY,
    "Fully filtered (simple)": T.TEAL,
    "Fully filtered (reconstruction)": T.BLUE,
}


def tier_figure(key, tier_degs, path=None):
    """One organism: total / in / out degree distributions under every filtering tier."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2), sharey=True)
    for ax, kind in zip(axes, T.KINDS):
        _style_axes(ax, xlim=(0.8, 2e3), ylim=(1e-4, 1.5))
        for name, D in tier_degs.items():
            c = TIER_COLOR[name]
            hero = name.endswith("(reconstruction)")
            g = scatter_fit(ax, D[kind], kind, s=16 if hero else 11, alpha=0.9 if hero else 0.45,
                            color=c, lw=2.4 if hero else 1.3, z=5 if hero else 3)
            if name.endswith("(simple)"):      # nearly identical to the reconstruction: draw it dashed on top
                k_, _p = graph.degree_distribution(D[kind]); g_, b_ = R.ols_fit(D[kind])
                xs_ = np.array([k_.min(), k_.max()])
                ax.plot(xs_, 10 ** (b_ + g_ * np.log10(xs_)), color="white", lw=1.0, ls=(0, (4, 3)), zorder=7)
                ax.plot(xs_, 10 ** (b_ + g_ * np.log10(xs_)), color=c, lw=1.0, ls=(3, (4, 3)), zorder=7)
            ax.plot([], [], color=c, lw=2, label=rf"{name}  ($\gamma$ = {g:.2f})".replace("-", "−"))
        ax.set_title(T.KIND_STYLE[kind]["label"], loc="left")
        ax.set_xlabel("degree (k)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, handlelength=1.4)
        T.square(ax)
    axes[0].set_ylabel("Pr(k)")
    fig.suptitle(f"{data.NAME[key]} ({data.MODEL[key]})", x=0.01, ha="left", fontsize=14, y=1.02)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def two_panel(degs_left, degs_right, titles, kmins=(1, 1), path=None, xlim=(0.8, 2e3), suptitle=None):
    """E. coli comparison used by Test 1 (currency excluded/included) and Test 2 (k>=1 / k>=10)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4), sharey=True)
    out = []
    for ax, D, ttl, km in zip(axes, (degs_left, degs_right), titles, kmins):
        _style_axes(ax, xlim=xlim, ylim=(5e-4, 1.0))
        gam, handles = {}, []
        for kind in T.KINDS:
            gam[kind] = scatter_fit(ax, D[kind], kind, kmin=km)
            st = T.KIND_STYLE[kind]
            handles.append(Line2D([], [], color=st["color"], marker=st["marker"], ls="", ms=5,
                                  label=rf"{kind.capitalize()}: $\gamma$ = {gam[kind]:.2f}".replace("-", "−")))
        if km > 1:
            ax.axvline(km, color=T.MUTED, ls=":", lw=1)
            ax.text(km * 1.08, 0.78, f"fit k ≥ {km}", fontsize=8)
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("degree k")
        ax.legend(handles=handles, loc="upper right", fontsize=8.5)
        T.square(ax)
        out.append(gam)
    axes[0].set_ylabel("Pr(k)")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, y=1.03)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig, out


def currency_schematic(path=None):
    """Slide: what adding currency metabolites does (A -> ATP -> B creates the arc A -> B)."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    for x, s, c in [(1.2, "A", T.TEAL), (5, "ATP", T.AMBER), (8.8, "B", T.TEAL)]:
        ax.text(x, 1.6, s, ha="center", va="center", fontsize=26 if s != "ATP" else 24)
    for x0, x1 in [(1.9, 4.0), (6.0, 8.1)]:
        ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6), arrowprops=dict(arrowstyle="-|>", color=T.MUTED, lw=1.6))
    ax.text(1.2, 2.6, "Reaction A\nproduces ATP", ha="center", va="center", fontsize=9)
    ax.text(8.8, 2.6, "Reaction B\nconsumes ATP", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(8.4, 0.55), xytext=(1.6, 0.55), arrowprops=dict(arrowstyle="-|>", color=T.BLUE, lw=1.6, ls="--",
                connectionstyle="arc3,rad=0"))
    ax.text(5, 0.12, "if ATP may create links, the graph gains the arc A → B", ha="center", fontsize=9)
    if path:
        fig.savefig(path)
    return fig


# ------------------------------------------------------------------ regression sensitivity
KIND_COL = {k: T.KIND_STYLE[k]["color"] for k in T.KINDS}


def gamma_by_method(table, path=None, delta=False):
    """table: DataFrame with columns organism(key), kind, method, gamma, paper. One facet per organism."""
    methods = list(R.ESTIMATORS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharey=True)
    axes = axes.ravel()
    for ax, key in zip(axes, data.KEYS):
        sub = table[table.organism == key]
        ax.grid(True, axis="x", alpha=0.5); ax.set_axisbelow(True)
        for kind in T.KINDS:
            s = sub[sub.kind == kind]
            y = [methods.index(m) for m in s.method]
            x = s.gamma - (s.paper if delta else 0)
            ax.scatter(x, y, color=KIND_COL[kind], s=30, zorder=3, label=kind, linewidths=0)
            if not delta:
                ax.axvline(data.paper_gamma(key, kind), color=KIND_COL[kind], ls="--", lw=1, alpha=0.8, zorder=1)
        if delta:
            ax.axvline(0, color=T.INK, ls="--", lw=1)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8.5)
        T.square(ax)
        ax.invert_yaxis()
        ax.set_title(data.NAME[key], loc="left")
        ax.set_xlabel("γ − published" if delta else "γ")
    axes[5].axis("off")
    h = [Line2D([], [], color=KIND_COL[k], marker="o", ls="", ms=6, label=k) for k in T.KINDS]
    if not delta:
        h.append(Line2D([], [], color=T.MUTED, ls="--", lw=1, label="published (Fig. 2)"))
    axes[5].legend(handles=h, title="degree", loc="center left", fontsize=10, title_fontsize=10)
    fig.tight_layout(w_pad=0.2)
    if path:
        fig.savefig(path)
    return fig


MODEL_COLOR = {
    "log-log OLS": T.BLUE, "log-log WLS (weight = count)": T.TEAL,
    "log-log OLS (k seen in >= 2 nodes)": T.AMBER, "log-linear OLS (exponential)": T.RED,
    "linear OLS": T.PURPLE, "MLE (kmin = 1)": T.SKY, "MLE (kmin by KS)": "#7A5C3E",
}


def fits_by_model(key, degs, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, kind in zip(axes, T.KINDS):
        _style_axes(ax, xlim=(0.8, 400), ylim=(1e-4, 1.5))
        k, p = graph.degree_distribution(degs[kind])
        st = T.KIND_STYLE[kind]
        ax.scatter(k, p, s=16, color=st["color"], marker=st["marker"], zorder=4, linewidths=0)
        for name, (x, y) in R.fitted_curves(degs[kind]).items():
            ok = y > 0
            ax.plot(x[ok], y[ok], color=MODEL_COLOR[name], lw=1.6, label=name, zorder=3)
        ax.set_title(st["label"], loc="left")
        ax.set_xlabel("degree (k)")
        T.square(ax)
    axes[0].set_ylabel("Pr(k)")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8.5, title="model", title_fontsize=9)
    fig.tight_layout()
    fig.suptitle(data.NAME[key], x=axes[0].get_position().x0, ha="left", va="bottom", fontsize=14,
                 y=axes[0].get_position().y1 + 0.08)
    
    if path:
        fig.savefig(path)
    return fig


def simulation_figure(sim, example_deg, true_alpha, path=None):
    methods = list(R.ESTIMATORS)
    col = {m: c for m, c in zip(methods, T.CAT)}
    fig = plt.figure(figsize=(16, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])
    ax = fig.add_subplot(gs[0])
    _style_axes(ax, xlim=(0.8, 1e4), ylim=(5e-4, 1.0))
    k, p = graph.degree_distribution(example_deg)
    ax.scatter(k, p, s=14, color=T.INK, zorder=3, linewidths=0, label="simulated Pr(k)")
    g, b = R.ols_fit(example_deg)
    xs = np.array([k.min(), k.max()])
    ax.plot(xs, 10 ** (b + g * np.log10(xs)), color=T.BLUE, lw=2, label=rf"paper-style OLS ($\gamma$ = {g:.2f})".replace("-", "−"))
    ks = np.array([1, 1e4])
    ax.plot(ks, (ks ** -true_alpha) / np.sum(np.arange(1, 10001.0) ** -true_alpha), color=T.TEAL, lw=2, ls="--",
            label=rf"truth ($\gamma$ = −{true_alpha:.2f})".replace("\\u2212", "−"))
    ax.axhline(1 / len(example_deg), color=T.RED, ls=":", lw=1)
    ax.set_title("One simulated network", loc="left"); ax.set_xlabel("degree (k)"); ax.set_ylabel("Pr(k)")
    ax.legend(loc="upper right", fontsize=8)
    for a, kmax in zip((fig.add_subplot(gs[1]), fig.add_subplot(gs[2])), sorted(sim.kmax.unique())):
        a.grid(True, alpha=0.5); a.set_axisbelow(True)
        s = sim[sim.kmax == kmax]
        for m in methods:
            r = s[s.method == m].sort_values("true")
            a.plot(r.true, r.bias, "o-", color=col[m], lw=1.6, ms=4.5, label=m, ls="-" if R.ESTIMATORS[m][1] else "--")
        a.axhline(0, color=T.INK, lw=0.9)
        a.set_title(f"True power law on k = 1…{kmax:,}", loc="left")
        a.set_xlabel("true exponent |γ|")
    fig.axes[1].set_ylabel("bias in |γ|  (estimate − truth)")
    T.square(*fig.axes)
    fig.axes[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title="method (dashed = extra)", title_fontsize=8.5)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
