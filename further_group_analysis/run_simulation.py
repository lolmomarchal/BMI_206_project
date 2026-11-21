"""Simulation: (A) estimator bias/RMSE when truth is a (bounded) power law;
(B) can non-power-law data (fitted lognormal / PL+cutoff) reproduce the paper-style OLS gamma?"""
import numpy as np, csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import BMI206_groupproject.BMI_206_project.further_group_analysis.degfit as F
from BMI206_groupproject.BMI_206_project.further_group_analysis.load_data import load, NAMES

rng = np.random.default_rng(1)
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
N, R = 1000, 300
EST = {
    "OLS log-log (paper)": F.ols_loglog,
    "OLS singletons dropped": lambda d: F.ols_loglog(d, drop_singletons=True),
    "OLS log-binned": F.ols_logbin,
    "OLS on CCDF": F.ols_ccdf,
    "MLE zeta": lambda d: F.mle_pl(d, 1)[0],
    "MLE bounded (=Poisson GLM)": lambda d: F.mle_pl(d, 1, int(d.max()))[0],
}

# ---------------- A
res = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, kmax in zip(axes, (200, 10000)):
    ks = np.arange(1, kmax + 1)
    for ai, a in enumerate((1.5, 2.0, 2.5, 3.0)):
        p = ks ** -a; cdf = np.cumsum(p) / p.sum()
        out = {e: [] for e in EST}
        for _ in range(R):
            d = ks[np.searchsorted(cdf, rng.random(N))]
            for e, f in EST.items():
                out[e].append(f(d))
        for e, v in out.items():
            v = np.array(v); res.append(dict(kmax=kmax, true=a, est=e, bias=v.mean() - a, sd=v.std(), rmse=np.sqrt(np.mean((v - a) ** 2))))
    for j, e in enumerate(EST):
        for a in (1.5, 2.0, 2.5, 3.0):
            pass
        xs = [r["true"] for r in res if r["kmax"] == kmax and r["est"] == e]
        ys = [r["bias"] for r in res if r["kmax"] == kmax and r["est"] == e]
        ax.plot(xs, ys, "o-", color=OI[j], label=e)
    ax.axhline(0, color="k", lw=.8); ax.set_title(f"truth: power law on k = 1..{kmax}, n = {N} nodes")
    ax.set_xlabel("true γ"); ax.grid(alpha=.3)
axes[0].set_ylabel("mean bias of estimate (γ̂ − γ)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/07_simulation_bias.png", dpi=170); plt.close(fig)
with open("results/simulation_bias.csv", "w", newline="") as f:
    w = csv.DictWriter(f, res[0].keys()); w.writeheader(); w.writerows(res)

# ---------------- B: data generated from non-power-law fits of the real degree sequences
D = load(); rows = []
fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
for i, nm in enumerate(NAMES):
    d = D[nm]["total"]; n = len(d); n_pos = (d > 0).sum()
    t = F.model_table(d, 1); grid = np.arange(1, F.KGRID)
    mu, s = t["_params"]["lognorm"]; ta, tl = t["_params"]["trunc"]
    models = {"lognormal": -np.log(grid) - (np.log(grid) - mu) ** 2 / (2 * s ** 2),
              "PL+cutoff": -ta * np.log(grid) - tl * grid}
    obs_ols = F.ols_loglog(d); obs_mle = F.mle_pl(d, 1)[0]
    for r_i, (mn, lw) in enumerate(models.items()):
        p = np.exp(lw - lw.max()); cdf = np.cumsum(p) / p.sum()
        o, m = [], []
        for _ in range(R):
            x = grid[np.searchsorted(cdf, rng.random(n_pos))]
            x = np.concatenate([x, np.zeros(n - n_pos, int)])  # keep the real count of isolated/zero-degree nodes
            o.append(F.ols_loglog(x)); m.append(F.mle_pl(x, 1)[0])
        ax = axes[r_i, i]; ax.hist(o, bins=25, color=OI[0], alpha=.8, label="OLS γ̂"); ax.hist(m, bins=25, color=OI[3], alpha=.8, label="MLE α̂")
        ax.axvline(obs_ols, color=OI[0], ls="--"); ax.axvline(obs_mle, color=OI[3], ls="--")
        ax.set_title(f"{nm}\nsimulated from fitted {mn}", fontsize=9, style="italic")
        rows.append(dict(organism=nm, null=mn, obs_ols=obs_ols, sim_ols_mean=np.mean(o), sim_ols_sd=np.std(o), obs_mle=obs_mle,
                         sim_mle_mean=np.mean(m), sim_mle_sd=np.std(m)))
axes[0, 0].legend(fontsize=8)
fig.suptitle("Total degree: estimator distribution on data simulated from NON-power-law fits (dashed = observed)")
fig.tight_layout(); fig.savefig("figures/08_nonpowerlaw_null.png", dpi=170); plt.close(fig)
with open("results/nonpowerlaw_null.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
