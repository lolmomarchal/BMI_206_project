"""Builds unified_analysis.ipynb (run with: python build_notebook.py && jupyter nbconvert --execute ...)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Identification of critical connectors in the directed reaction-centric graphs of microbial metabolic networks
**Hassan Bukhari, Haruna Tomono, Lorenzo Olmo Marchal, Sohan Addagudi** · BMI 206 group project

This notebook is the single, unified pipeline behind the presentation. It reproduces Fig. 2 of Kim et al. (2019, *BMC Bioinformatics* 20:328)
— the degree distributions of five reaction-centric graphs — and then tests how sensitive the fitted exponent γ is to (1) how metabolites are
included/excluded and (2) how the power law is fitted.

Everything runs through one implementation (`metnet/`): **one graph builder** whose filtering choices are explicit switches, and **one estimator module** for every
way of fitting γ. Previously each of us had our own version of these steps, which could shift the sensitivity results slightly; here every
analysis differs from the baseline only in the single switch being tested.

| Slide section | Notebook section | Unified from |
|---|---|---|
| Reproduction from supplemental file | 1 | Sohan, Haruna, Lorenzo (supplement-matrix versions) |
| Reproduction from raw data / issues | 2 | Sohan's rule set (best match to the paper), checked against Haruna's arc audit and Lorenzo's filter |
| Modifying metabolite filtering, Tests 1–2 | 3 | Lorenzo's filtering tiers + Hassan's Test 1/2, all on the same builder |
| Changing method of power-law fitting | 4 | Haruna's six estimators + Sohan/`further_group_analysis` estimators and simulation, one module |

Conventions used throughout: γ is the slope of log Pr(k) vs log k (negative, as printed in the paper); Pr(k) = (# reactions with degree k) / (# reactions in the graph);
in-degree = arcs arriving, out-degree = arcs leaving, total = in + out.
""")

code(r"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from metnet import data, graph, regression as R, simulation as S, theme as T, plots as P

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
RES = Path("results"); RES.mkdir(exist_ok=True)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
trunc2 = lambda x: np.trunc(np.asarray(x) * 100) / 100

sup = data.load_supplement()                                   # authors' adjacency matrices (Additional file 2)
models = {k: data.read_model(data.MODEL[k]) for k in data.KEYS}  # original 2019-era BiGG models
sup_deg = {k: data.degrees_from_matrix(sup[k][1]) for k in data.KEYS}
print("loaded:", ", ".join(f"{data.NAME[k]} ({data.MODEL[k]})" for k in data.KEYS))
""")

# ---------------------------------------------------------------- 1
md(r"""
## 1 · Reproduction from the supplemental file

The paper's Additional file 2 gives the five adjacency matrices. We read degrees straight from them (in = column sum, out = row sum) and fit γ with an
ordinary least-squares line through log₁₀ Pr(k) vs log₁₀ k, one point per observed degree.

**Result: it works.** The paper does not say how γ was fitted; plain OLS reproduces its printed values — **14 of 15 identical** — provided the value is *truncated* to two decimals
rather than rounded (rounding matches only 9 of 15). The one miss is *E. coli* in-degree (−1.316, printed −1.32).
""")
code(r"""
fig = P.compare_with_original(
    sup_deg, "Recreation from the paper's supplementary matrices",
    "Degrees read from the adjacency matrices in Additional file 2",
    "γ: 14 of 15 values identical to the paper (truncated to 2 decimals). The one difference is E. coli in-degree (−1.31 here, −1.32 printed).",
    FIG / "01_fig2_from_supplement.png")
plt.show()
""")
code(r"""
rows = []
for k in data.KEYS:
    for kind in data.KIND_ORDER:
        g = R.ols(sup_deg[k][kind]); p = data.paper_gamma(k, kind)
        rows.append(dict(organism=data.NAME[k], degree=kind, ours=round(g, 4), truncated=float(trunc2(g)), rounded=round(g, 2),
                         paper=p, match_trunc=bool(trunc2(g) == p), match_round=bool(round(g, 2) == p)))
tab = pd.DataFrame(rows); tab.to_csv(RES / "01_supplement_gamma_vs_paper.csv", index=False)
print(f"matches the printed γ — truncated: {tab.match_trunc.sum()}/15   rounded: {tab.match_round.sum()}/15")
display(tab[~tab.match_trunc])
""")

# ---------------------------------------------------------------- 2
md(r"""
## 2 · Reproduction from raw data

Rebuilding the graphs from the BiGG models the paper cites. The paper only describes the conversion in words (currency metabolites removed, transport and exchange reactions
removed, arc B → C if a product of B is consumed by C), so the rules have to be inferred. This is the one graph builder (`metnet.graph.build_graph`) used for everything that follows.

### 2.1 Issues when it comes to full reproduction
**(i) Which BiGG release.** Based on the publication date we use the *older* BiGG models. They match the paper's Table 1 exactly; the newer release uses different IDs/reaction names and
differs slightly in content.
""")
code(r"""
table1 = {"ecoli": (2583, 1805), "bsub": (1250, 990), "gmet": (1285, 1109), "kpneu": (2262, 1658), "scer": (1577, 1226)}  # paper Table 1: reactions, metabolites
rows = []
for k in data.KEYS:
    m = data.MODEL[k]
    ro, mo = data.count_sbml(data.OLD_BIGG / f"{m}.xml.gz")
    rn, mn = data.count_sbml(data.DATA / f"{m}.biggr.sbml.gz")
    ids = set(sup[k][0])
    new_ids = {i.removeprefix("R_").removeprefix(f"__{m}__") for i in
               __import__("metnet.data", fromlist=["x"]).read_model(m, data.DATA / f"{m}.biggr.sbml.gz", with_subsystems=False).rxns}
    rows.append({"model": f"{data.NAME[k]} ({m})", "Table 1 (rxn, met)": table1[k], "old BiGG (rxn, met)": (ro, mo), "new BiGG (rxn, met)": (rn, mn),
                 "old = Table 1": (ro, mo) == table1[k],
                 "supplement rxn IDs found: old": f"{len(ids & set(models[k].rxns))}/{len(ids)}", "new": f"{len(ids & new_ids)}/{len(ids)}"})
display(pd.DataFrame(rows).set_index("model"))
""")
md(r"""
**(ii) The currency-metabolite list is unclear** (the paper names only ATP, NAD and H₂O, and does not say whether a numeric cutoff was used). We recovered it from the supplement itself:
for every metabolite, build the arcs it would create between the supplement's own reactions and ask what fraction of them the supplement actually contains. Currency metabolites
create many arcs of which almost none are in the supplement; everything else is nearly all in. The split is clean — no cutoff is needed, the two groups do not overlap.
""")
code(r"""
rows = []
for k in data.KEYS:
    M = models[k]; ids, A0 = sup[k]
    idx = {i: j for j, i in enumerate(ids)}
    found = [i for i in ids if i in M.rxns]
    S0 = {(a, b) for a, b in zip(*np.nonzero(A0))}
    prod, cons = {}, {}
    for i in found:
        r = M.rxns[i]
        for x in r.subs:
            cons.setdefault(x, set()).add(idx[i]);  r.reversible and prod.setdefault(x, set()).add(idx[i])
        for x in r.prods:
            prod.setdefault(x, set()).add(idx[i]);  r.reversible and cons.setdefault(x, set()).add(idx[i])
    for x in prod.keys() & cons.keys():
        arcs = {(a, b) for a in prod[x] for b in cons[x] if a != b}
        if arcs:
            rows.append(dict(organism=k, met=x, base=graph.base(x), n_arcs=len(arcs), frac_in_supp=len(arcs & S0) / len(arcs)))
audit = pd.DataFrame(rows)
audit["currency"] = (audit.n_arcs >= 20) & (audit.frac_in_supp <= 0.05)
found_cur = sorted(audit[audit.currency & (audit.organism != "scer") | (audit.currency & audit.met.str.endswith("_c"))].base.unique())
print("metabolites removed by the supplement (union over organisms):", found_cur)
print("== our CURRENCY set:", sorted(graph.CURRENCY), "| identical:", set(found_cur) == set(graph.CURRENCY))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.grid(True, alpha=.5); ax.set_axisbelow(True)
for flag, c, lab in [(False, T.SOFT_BLUE, "kept by the paper"), (True, T.RED, "removed (currency)")]:
    s = audit[audit.currency == flag]; ax.scatter(s.n_arcs, s.frac_in_supp, s=14, color=c, alpha=.7, label=lab, linewidths=0)
ax.set(xscale="log", xlabel="arcs the metabolite would create", ylabel="fraction of those arcs in the supplement")
ax.set_title("Currency metabolites separate cleanly", loc="left"); ax.legend(loc="center right")
fig.savefig(FIG / "02_currency_audit.png"); plt.show()
""")
md(r"""
**(iii) Regression method is not stated** — but OLS on log–log Pr(k) reproduces the published γ from the supplement (section 1), so the same fit is used for every raw-data result.

### 2.2 Which reactions are nodes?
The supplement has no exchange/boundary reactions and almost no periplasm- or extracellular-only reactions. The reaction filter (`reactions="reconstruction"`) is: drop exchange/demand/sink/biomass; drop reactions
whose BiGG subsystem is transport/exchange; drop reactions touching the extracellular space; drop periplasm-only reactions; in yeast keep only cytosol/mitochondria/ER; in *E. coli* also drop `…pp` ids;
drop reactions with no arcs. The first rules are directly supported by the supplement's node names; the last few are a best fit (documented in `metnet/graph.py`).

**Check:** given the supplement's own node set, does the rebuilt arc list match the supplement's?
""")
code(r"""
rows = []
for k in data.KEYS:
    M = models[k]; ids, A0 = sup[k]; found = [i for i in ids if i in M.rxns]
    _, A = graph.build_graph(M, graph.RECONSTRUCTION, nodes=found)
    idx = {i: j for j, i in enumerate(ids)}; sel = [idx[i] for i in found]
    S0 = A0[np.ix_(sel, sel)].astype(bool); A = A.astype(bool)
    rows.append(dict(organism=data.NAME[k], supplement_nodes=len(ids), nodes_found_in_model=len(found), supplement_arcs=int(A0.sum()),
                     arcs_shared=int((A & S0).sum()), only_ours=int((A & ~S0).sum()), only_theirs=int((~A & S0).sum())))
chk = pd.DataFrame(rows).set_index("organism"); chk.to_csv(RES / "02_arc_check_given_supplement_nodes.csv"); display(chk)
""")
md(r"""
### 2.3 Figure 2 rebuilt from the raw BiGG models
""")
code(r"""
rebuilt, raw_deg = {}, {}
for k in data.KEYS:
    ids, A = graph.build_graph(models[k], graph.RECONSTRUCTION)
    rebuilt[k] = (ids, A); raw_deg[k] = graph.degrees(A)
fig = P.compare_with_original(
    raw_deg, "Recreation from the raw BiGG models",
    "Graphs rebuilt from the original metabolic models (currency, transport and boundary rules inferred)",
    "", FIG / "03_fig2_from_raw_data.png")
plt.show()
rows = []
for k in data.KEYS:
    n, a = len(rebuilt[k][0]), int(rebuilt[k][1].sum()); pn, pa = data.PAPER_COUNTS[k]
    row = dict(organism=data.NAME[k], nodes=n, nodes_paper=pn, arcs=a, arcs_paper=pa)
    for kind in data.KIND_ORDER:
        row[f"γ {kind}"] = float(trunc2(R.ols(raw_deg[k][kind]))); row[f"γ {kind} (paper)"] = data.paper_gamma(k, kind)
    rows.append(row)
raw_tab = pd.DataFrame(rows).set_index("organism"); raw_tab.to_csv(RES / "03_raw_rebuild_vs_paper.csv")
display(raw_tab)
same = sum(raw_tab[f"γ {kd}"].iloc[i] == raw_tab[f"γ {kd} (paper)"].iloc[i] for i in range(5) for kd in data.KIND_ORDER)
print(f"γ identical to the paper: {same}/15 (the rest differ by 0.01-0.03)")
""")
md(r"""
Node/arc counts are within ~0.5 % for four organisms; *K. pneumoniae* is furthest off (the paper's list uses old reaction IDs that no longer exist in the model, so it could not be
checked reaction by reaction). The remaining differences are individual reactions the paper keeps or drops that we cannot distinguish from the model alone.
""")

# ---------------------------------------------------------------- 3
md(r"""
## 3 · Sensitivity 1 — modifying metabolite filtering

### 3.1 Paper vs different filtering steps
Two switches in the same graph builder, plus the paper-matching node rule:

| Tier | Currency metabolites | Reactions (nodes) |
|---|---|---|
| Unfiltered | all allowed to make links | every reaction in the model |
| Currency filtered | removed | every reaction |
| Compartment filtered | allowed | no boundary reactions; cytosol only (+ mitochondria in yeast) |
| Fully filtered (simple) | removed | as above |
| **Fully filtered (reconstruction)** — ours | removed | the paper-matching rules of §2.2 |

Reactions with no arc are not nodes, in every tier. All fits use the same log–log OLS.
""")
code(r"""
tier_deg, tier_rows = {}, []
for k in data.KEYS:
    tier_deg[k] = {}
    for rules in graph.TIERS:
        ids, A = graph.build_graph(models[k], rules)
        D = graph.degrees(A); tier_deg[k][rules.name] = D
        tier_rows.append(dict(organism=data.NAME[k], tier=rules.name, nodes=len(ids), arcs=int(A.sum()),
                              **{f"γ {kd}": R.ols(D[kd]) for kd in data.KIND_ORDER}))
tiers = pd.DataFrame(tier_rows); tiers.to_csv(RES / "04_filtering_tiers.csv", index=False)
""")
code(r"""
for k in data.KEYS:
    P.tier_figure(k, tier_deg[k], FIG / f"04_filtering_tiers_{k}.png")
    plt.show()
""")
code(r"""
display(tiers.round(2).set_index(["organism", "tier"]))
""")
md(r"""
### 3.2 Test 1 · What does adding currency metabolites do?
""")
code(r"""
P.currency_schematic(FIG / "05_currency_schematic.png"); plt.show()
""")
md(r"""
**Compare two versions of *E. coli*.** Same reactions, same directions, same fit — only the currency switch changes. (1) Exclude the currency list and build links from the remaining
metabolites; (2) allow the currency list to create links; (3) compare the distributions and γ. Our currency list is a documented approximation of the paper's filtering (§2.1).
""")
code(r"""
ids_e, A_e = rebuilt["ecoli"]                                   # fixed node set = the reconstruction
_, A_exc = graph.build_graph(models["ecoli"], graph.RECONSTRUCTION, nodes=ids_e)
_, A_inc = graph.build_graph(models["ecoli"], graph.Rules("currency allowed", currency=False), nodes=ids_e)
D_exc, D_inc = graph.degrees(A_exc), graph.degrees(A_inc)
assert (A_exc == A_e).all() and (A_inc >= A_exc).all()
fig, (g_exc, g_inc) = P.two_panel(D_exc, D_inc, ("Currency excluded", "Currency included"), path=FIG / "06_test1_currency.png")
plt.show()
t1 = pd.DataFrame({
    "Currency excluded": {"Reactions": len(ids_e), "Directed links": int(A_exc.sum()), "Median total degree": np.median(D_exc["total"]),
                          "Total-degree γ": g_exc["total"], "In-degree γ": g_exc["in"], "Out-degree γ": g_exc["out"]},
    "Currency included": {"Reactions": len(ids_e), "Directed links": int(A_inc.sum()), "Median total degree": np.median(D_inc["total"]),
                          "Total-degree γ": g_inc["total"], "In-degree γ": g_inc["in"], "Out-degree γ": g_inc["out"]}})
t1.to_csv(RES / "05_test1_currency_ecoli.csv"); display(t1.round(2))
display(Markdown(f"**Directed links increase from {int(A_exc.sum()):,} to {int(A_inc.sum()):,}.** Common metabolites create many more connections between reactions, "
                 f"and the fitted result depends strongly on currency filtering (total γ {g_exc['total']:.2f} → {g_inc['total']:.2f})."))
""")
md(r"""
### 3.3 Test 2 · Fit different parts of the same distribution
Same currency-excluded *E. coli* graph; fit (1) every observed positive degree (k ≥ 1) and (2) only degrees k ≥ 10 (an exploratory cutoff). Both panels show all points.
""")
code(r"""
D = raw_deg["ecoli"]
fig, (g1, g10) = P.two_panel(D, D, ("E. coli fit uses k ≥ 1", "E. coli fit uses k ≥ 10"), kmins=(1, 10), path=FIG / "07_test2_fit_range.png")
plt.show()
t2 = pd.DataFrame({"k ≥ 1": g1, "k ≥ 10": g10}).rename(index=lambda s: f"{s} γ"); t2.to_csv(RES / "06_test2_fit_range_ecoli.csv"); display(t2.round(2))
display(Markdown(f"Total γ {g1['total']:.2f} → {g10['total']:.2f}; **in γ {g1['in']:.2f} → {g10['in']:.2f}**; out γ {g1['out']:.2f} → {g10['out']:.2f}. Fewer degree values enter the k ≥ 10 fit."))
""")
code(r"""
tt = tiers[tiers.organism == "E. coli"].set_index("tier")
display(Markdown(f'''
### What do the sensitivity tests tell us?
- **Currency filtering changes the network.** Including the currency list adds links ({int(A_exc.sum()):,} → {int(A_inc.sum()):,} in *E. coli*) and produces much flatter fitted slopes
  (total γ {g_exc['total']:.2f} → {g_inc['total']:.2f}). Across filtering tiers (*E. coli*, total γ): unfiltered {tt.loc['Unfiltered','γ total']:.2f}; currency filter alone {tt.loc['Currency filtered','γ total']:.2f};
  compartment/boundary filter alone {tt.loc['Compartment filtered','γ total']:.2f}; both (reconstruction) {tt.loc['Fully filtered (reconstruction)','γ total']:.2f}. Only the combination recovers the paper's slope.
- **The fitting range changes the estimate.** Total γ moves least ({g1['total']:.2f} → {g10['total']:.2f}); in- and out-degree γ move more ({g1['in']:.2f} → {g10['in']:.2f}, {g1['out']:.2f} → {g10['out']:.2f}).
- **Our comparisons show dependence on analysis choices. Assessing power-law fit requires additional analysis** → section 4.
'''))
""")

# ---------------------------------------------------------------- 4
md(r"""
## 4 · Sensitivity 2 — changing the method of power-law fitting

The paper never says how γ was fitted. Here the *same* published graphs (the supplement's degree sequences) are fitted by every estimator we tried, all in one module (`metnet.regression`):

| Method | What it does |
|---|---|
| log-log OLS | **the paper's method** — unweighted OLS of log Pr(k) on log k, one point per observed k |
| log-log WLS (weight = count) | same, each k weighted by the number of nodes that have it |
| log-log OLS (k seen in ≥ 2 nodes) | same, dropping degrees seen in a single node (the flat 1/N floor) |
| log-log OLS on cumulative | OLS on P(K ≥ k); γ = slope − 1 |
| MLE (kmin = 1) | discrete power-law maximum likelihood over all degrees |
| MLE (kmin by KS) | maximum likelihood above the kmin that minimises the KS distance (Clauset et al.) |
| log-log OLS (log-binned) *(extra)* | OLS on log-binned densities |
| MLE bounded (= Poisson GLM) *(extra)* | power-law MLE on the observed support; identical to a Poisson regression over all degree bins |

The semi-log (`log-linear`, an exponential) and `linear` regressions have no γ and appear only as curves in 4.2.
""")
code(r"""
rows = []
for k in data.KEYS:
    for kind in data.KIND_ORDER:
        for m, g in R.gamma_table(sup_deg[k][kind]).items():
            rows.append(dict(organism=k, kind=kind, method=m, gamma=g, paper=data.paper_gamma(k, kind)))
gt = pd.DataFrame(rows); gt.to_csv(RES / "07_gamma_by_method.csv", index=False)
wide = gt.pivot_table(index=["organism", "kind"], columns="method", values="gamma")[list(R.ESTIMATORS)]
wide.insert(0, "paper", gt.drop_duplicates(["organism", "kind"]).set_index(["organism", "kind"]).paper)
display(wide.rename(index=data.NAME, level=0).round(2))
""")
md(r"""
### 4.1 Using different methods of fitting the power law changes γ
Dashed lines are the published values. Every method that is not the paper's OLS moves γ — by up to ~1 (cumulative OLS, KS-kmin MLE) — and the in/out/total ordering is not stable across methods.
""")
code(r"""
P.gamma_by_method(gt, FIG / "08_gamma_by_method.png"); plt.show()
""")
code(r"""
P.gamma_by_method(gt, FIG / "09_gamma_minus_published.png", delta=True); plt.show()
""")
md(r"""
### 4.2 Power-law fits look fine on the data, but the method of fitting changes the result
*E. coli*, all seven curves from the deck's comparison. The log-linear and linear fits are not power laws and bend away from the points on log–log axes.
""")
code(r"""
P.fits_by_model("ecoli", sup_deg["ecoli"], FIG / "10_ecoli_fits_by_model.png"); plt.show()
""")
md(r"""
### 4.3 What method was best? Exponent simulation
We simulate networks with a **known** exponent (n = 1000 nodes; degrees from a power law on 1…k_max; 300 replicates per setting), ask every method to estimate it, and measure the bias.

- The paper-style OLS is **biased low** (slope too shallow) — by about 0.5–1.0 here — even when the data are exactly a power law; the other methods are much closer.
- Why: at the high-connection end most degrees are empty because very few reactions have that many connections. Every degree seen once sits at the same floor, Pr = 1/n, which flattens the tail and drags the OLS line toward zero slope (left panel).
- MLE (kmin = 1) and the bounded MLE/Poisson fit are essentially unbiased (|bias| ≲ 0.05).
""")
code(r"""
sim = S.run(n=1000, reps=300, alphas=(1.5, 2.0, 2.5, 3.0), kmaxes=(200, 10000), seed=1)
sim.to_csv(RES / "08_simulation_bias.csv", index=False)
P.simulation_figure(sim, S.example(n=1000, alpha=2.0, kmax=10000), 2.0, FIG / "11_simulation_bias.png"); plt.show()
bias = sim.pivot_table(index="method", columns=["kmax", "true"], values="bias").loc[list(R.ESTIMATORS)]
display(bias.round(2))
ols_b = sim[sim.method == "log-log OLS"].bias
display(Markdown(f"Paper-style OLS bias ranges from **{ols_b.min():.2f} to {ols_b.max():.2f}** across settings; the bounded MLE (= Poisson) ranges "
                 f"{sim[sim.method.str.startswith('MLE bounded')].bias.min():+.2f} to {sim[sim.method.str.startswith('MLE bounded')].bias.max():+.2f}; MLE (kmin = 1) "
                 f"{sim[sim.method=='MLE (kmin = 1)'].bias.min():+.2f} to {sim[sim.method=='MLE (kmin = 1)'].bias.max():+.2f}."))
""")

nb.cells = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "unified_analysis.ipynb")
print("wrote notebook with", len(C), "cells")
