"""Every estimator of the degree exponent used in the project, behind one interface.

Convention: gamma is the slope of Pr(k) ~ k^gamma, i.e. NEGATIVE (as printed in the paper).
All functions take an integer degree array over all nodes (zeros allowed, ignored where k >= 1 is needed).
"""
import numpy as np
from scipy import optimize, special

from .graph import degree_distribution


# ---------------------------------------------------------------- least squares
def _wls(x, y, w=None):
    A = np.vstack([x, np.ones_like(x)]).T
    sw = np.sqrt(np.ones_like(x) if w is None else w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return beta  # slope, intercept


def _pk(deg):
    n = len(deg)
    d = deg[deg > 0].astype(int)
    k, c = np.unique(d, return_counts=True)
    return k, c, c / n


def ols(deg, kmin=1):
    """The paper's method: unweighted OLS of log10 Pr(k) on log10 k, one point per observed k >= kmin."""
    k, c, p = _pk(deg)
    m = k >= kmin
    return _wls(np.log10(k[m]), np.log10(p[m]))[0]


def ols_fit(deg, kmin=1):
    k, c, p = _pk(deg)
    m = k >= kmin
    s, b = _wls(np.log10(k[m]), np.log10(p[m]))
    return s, b


def wls_count(deg):
    """Weight each k by the number of nodes that have it."""
    k, c, p = _pk(deg)
    return _wls(np.log10(k), np.log10(p), c.astype(float))[0]


def ols_no_singletons(deg):
    """Drop degrees seen in only one node (the flat 1/N floor)."""
    k, c, p = _pk(deg)
    m = c >= 2
    return _wls(np.log10(k[m]), np.log10(p[m]))[0]


def ols_ccdf(deg):
    """OLS on the cumulative distribution: P(K >= k) ~ k^(gamma+1)  =>  gamma = slope - 1."""
    d = deg[deg > 0].astype(int)
    k, c = np.unique(d, return_counts=True)
    ccdf = c[::-1].cumsum()[::-1] / len(deg)
    return _wls(np.log10(k), np.log10(ccdf))[0] - 1


def ols_logbin(deg, ratio=2.0):
    """OLS on log-binned densities."""
    d = deg[deg > 0].astype(int)
    n = len(deg)
    edges = [1]
    while edges[-1] <= d.max():
        nxt = int(np.ceil(edges[-1] * ratio))
        edges.append(nxt if nxt > edges[-1] else edges[-1] + 1)
    edges = np.array(edges)
    cnt = np.array([np.sum((d >= lo) & (d < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
    width = np.diff(edges)
    ctr = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=1))
    m = cnt > 0
    return _wls(np.log10(ctr[m]), np.log10(cnt[m] / width[m] / n))[0]


def loglin(deg):
    """Semi-log regression log10 Pr(k) ~ k (an exponential, not a power law). Returns (slope, intercept)."""
    k, c, p = _pk(deg)
    return _wls(k.astype(float), np.log10(p))


def linear(deg):
    k, c, p = _pk(deg)
    return _wls(k.astype(float), p)


# ---------------------------------------------------------------- maximum likelihood
def mle_alpha(deg, kmin=1, kmax=None):
    """Discrete power-law MLE for the tail k >= kmin. Unbounded (zeta) or bounded to kmin..kmax
    (the bounded MLE equals a Poisson regression over all degree bins). Returns positive alpha."""
    d = deg[deg >= kmin].astype(int)
    slk, n = np.sum(np.log(d)), len(d)
    if kmax is None:
        nll = lambda a: a * slk + n * np.log(special.zeta(a, kmin))
        lo = 1.0001
    else:
        lk = np.log(np.arange(kmin, kmax + 1))
        nll = lambda a: a * slk + n * special.logsumexp(-a * lk)
        lo = -3
    return optimize.minimize_scalar(nll, bounds=(lo, 8), method="bounded", options={"xatol": 1e-7}).x


def _ks(d, kmin, alpha):
    ks = np.arange(kmin, d.max() + 1)
    emp = np.cumsum(np.bincount(d, minlength=d.max() + 1)[kmin:]) / len(d)
    mod = 1 - special.zeta(alpha, ks + 1) / special.zeta(alpha, kmin)
    return np.max(np.abs(emp - mod))


def clauset(deg, min_tail=30):
    """kmin that minimises the KS distance (Clauset, Shalizi & Newman 2009). Returns (kmin, alpha)."""
    d = deg[deg > 0].astype(int)
    best = None
    for kmin in np.unique(d):
        t = d[d >= kmin]
        if len(t) < min_tail:
            break
        a = mle_alpha(deg, int(kmin))
        D = _ks(t, int(kmin), a)
        if best is None or D < best[0]:
            best = (D, int(kmin), a)
    return best[1], best[2]


def mle_kmin1(deg):
    return -mle_alpha(deg, 1)


def mle_ks(deg):
    return -clauset(deg)[1]


def mle_bounded(deg):
    return -mle_alpha(deg, 1, int(deg.max()))


# ---------------------------------------------------------------- the estimator registry
# (label, function, in the slide deck?)
ESTIMATORS = {
    "log-log OLS": (ols, True),
    "log-log WLS (weight = count)": (wls_count, True),
    "log-log OLS (k seen in >= 2 nodes)": (ols_no_singletons, True),
    "log-log OLS on cumulative": (ols_ccdf, True),
    "MLE (kmin = 1)": (mle_kmin1, True),
    "MLE (kmin by KS)": (mle_ks, True),
    "log-log OLS (log-binned)": (ols_logbin, False),
    "MLE bounded (= Poisson GLM)": (mle_bounded, False),
}


def gamma_table(deg, which=None):
    return {name: f(deg) for name, (f, _) in ESTIMATORS.items() if which is None or name in which}


def fitted_curves(deg):
    """Fitted Pr(k) curves for the deck's 7-model comparison. Returns {label: (k_grid, pk_grid)}.
    The semi-log and linear fits are not power laws and have no gamma, only a curve."""
    n = len(deg)
    k, c, p = _pk(deg)
    grid = np.arange(1, k.max() + 1)
    lg = np.log10(grid)
    out = {}
    s, b = ols_fit(deg); out["log-log OLS"] = (grid, 10 ** (b + s * lg))
    s, b = _wls(np.log10(k), np.log10(p), c.astype(float)); out["log-log WLS (weight = count)"] = (grid, 10 ** (b + s * lg))
    m = c >= 2; s, b = _wls(np.log10(k[m]), np.log10(p[m])); out["log-log OLS (k seen in >= 2 nodes)"] = (grid, 10 ** (b + s * lg))
    s, b = loglin(deg); out["log-linear OLS (exponential)"] = (grid, 10 ** (b + s * grid))
    s, b = linear(deg); out["linear OLS"] = (grid, b + s * grid)
    a = mle_alpha(deg, 1)
    out["MLE (kmin = 1)"] = (grid, (len(deg[deg > 0]) / n) * grid ** -a / special.zeta(a, 1))
    kmin, a = clauset(deg)
    g = grid[grid >= kmin]
    out["MLE (kmin by KS)"] = (g, ((deg >= kmin).sum() / n) * g ** -a / special.zeta(a, kmin))
    return {kk: (x, y) for kk, (x, y) in out.items()}


def cutoff_gamma(deg, kmin):
    """Paper OLS restricted to k >= kmin (Test 2)."""
    return ols(deg, kmin)


# ---------------------------------------------------------------- simulation
def sample_power_law(rng, n, alpha, kmax):
    ks = np.arange(1, kmax + 1)
    p = ks ** -float(alpha)
    cdf = np.cumsum(p) / p.sum()
    return ks[np.searchsorted(cdf, rng.random(n))]
