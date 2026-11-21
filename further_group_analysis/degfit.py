"""Estimators for the degree-distribution exponent gamma (P(k) ~ k^-gamma).

All functions take an integer degree array (zeros allowed; they are dropped where the
estimator conditions on k>=1) and return plain floats / dicts.
"""
import numpy as np
from scipy import optimize, special, stats

KGRID = 5000  # upper limit for normalising unbounded non power-law models


# ---------------------------------------------------------------- regression-type
def pk(deg):
    """Paper's definition: Pr(k) = #nodes with degree k / #all nodes (k>0 only)."""
    n = len(deg)
    d = deg[deg > 0].astype(int)
    k, c = np.unique(d, return_counts=True)
    return k, c, c / n


def _ols(x, y, w=None):
    A = np.vstack([x, np.ones_like(x)]).T
    if w is None:
        w = np.ones_like(x)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    yhat = A @ beta
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return beta[0], beta[1], 1 - ss_res / ss_tot


def ols_loglog(deg, kmin=1, drop_singletons=False, weighted=False):
    """gamma from log10 Pr(k) ~ log10 k (the paper's method; weighted -> w = count)."""
    k, c, p = pk(deg)
    m = k >= kmin
    if drop_singletons:
        m &= c > 1
    if m.sum() < 3:
        return np.nan
    s, b, r2 = _ols(np.log10(k[m]), np.log10(p[m]), c[m].astype(float) if weighted else None)
    return -s


def ols_loglog_r2(deg):
    k, c, p = pk(deg)
    return _ols(np.log10(k), np.log10(p))[2]


def ols_ccdf(deg, kmin=1):
    """log-log regression on the complementary CDF: P(K>=k) ~ k^-(alpha-1)."""
    d = deg[deg >= kmin].astype(int)
    k, c = np.unique(d, return_counts=True)
    ccdf = c[::-1].cumsum()[::-1] / len(d)
    m = k >= kmin
    s, b, r2 = _ols(np.log10(k[m]), np.log10(ccdf[m]))
    return 1 - s


def ols_logbin(deg, ratio=2.0):
    """log-binned density regression (bins [r^j, r^(j+1)) in integers, density = count/width)."""
    d = deg[deg > 0].astype(int)
    n = len(deg)
    edges = [1]
    while edges[-1] <= d.max():
        edges.append(int(np.ceil(edges[-1] * ratio)) if int(np.ceil(edges[-1] * ratio)) > edges[-1] else edges[-1] + 1)
    edges = np.array(edges)
    cnt = np.array([np.sum((d >= lo) & (d < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
    width = np.diff(edges)
    ctr = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=1))
    m = cnt > 0
    s, b, r2 = _ols(np.log10(ctr[m]), np.log10(cnt[m] / width[m] / n))
    return -s


def loglin_ols(deg):
    """semi-log ('log-linear') regression: ln Pr(k) = a - b*k.  returns (b, R2)."""
    k, c, p = pk(deg)
    s, b, r2 = _ols(k.astype(float), np.log(p))
    return -s, r2


def _counts_full(deg, kmin=1, kmax=None):
    d = deg[deg >= kmin].astype(int)
    kmax = kmax or d.max()
    ks = np.arange(kmin, kmax + 1)
    c = np.bincount(d, minlength=kmax + 1)[kmin:kmax + 1]
    return ks, c


def poisson_reg_pl(deg, kmin=1):
    """Poisson GLM: count_k ~ Pois(exp(a - gamma*ln k)), over ALL k in [kmin, kmax]
    (empty bins included - log-OLS silently drops them).  returns gamma."""
    ks, c = _counts_full(deg, kmin)
    lk = np.log(ks)

    def nll(th):
        eta = th[0] - th[1] * lk
        return np.sum(np.exp(eta) - c * eta)

    r = optimize.minimize(nll, [np.log(c.sum()), 1.5], method="BFGS")
    return r.x[1]


def poisson_reg_exp(deg, kmin=1):
    """Poisson GLM log-linear in k: count_k ~ Pois(exp(a - b*k)). returns b."""
    ks, c = _counts_full(deg, kmin)

    def nll(th):
        eta = th[0] - th[1] * ks
        return np.sum(np.exp(eta) - c * eta)

    r = optimize.minimize(nll, [np.log(c.sum()), 0.1], method="BFGS")
    return r.x[1]


# ---------------------------------------------------------------- MLE: pmfs
def _tail(deg, kmin):
    return deg[deg >= kmin].astype(int)


def mle_pl(deg, kmin=1, kmax=None):
    """Discrete power-law MLE.  Unbounded support (zeta) or, if kmax is given, bounded
    support kmin..kmax.  returns (alpha, loglik, n)."""
    d = _tail(deg, kmin)
    slk = np.sum(np.log(d))
    n = len(d)
    if kmax is None:
        def nll(a):
            return a * slk + n * np.log(special.zeta(a, kmin))
    else:
        ks = np.arange(kmin, kmax + 1)
        lk = np.log(ks)

        def nll(a):
            return a * slk + n * special.logsumexp(-a * lk)

    r = optimize.minimize_scalar(nll, bounds=(1.0001 if kmax is None else -3, 8), method="bounded",
                                 options={"xatol": 1e-7})
    return r.x, -r.fun, n


def pl_pointwise_ll(deg, kmin, alpha, kmax=None):
    d = _tail(deg, kmin)
    if kmax is None:
        return -alpha * np.log(d) - np.log(special.zeta(alpha, kmin))
    ks = np.arange(kmin, kmax + 1)
    return -alpha * np.log(d) - special.logsumexp(-alpha * np.log(ks))


def _grid_fit(deg, kmin, logw_fn, x0, bounds=None):
    """generic MLE for pmf ∝ exp(logw(k; theta)) on k>=kmin, normalised on a grid."""
    d = _tail(deg, kmin)
    kmax = max(KGRID, 5 * d.max())
    ks = np.arange(kmin, kmax + 1).astype(float)

    def nll(th):
        lw = logw_fn(ks, th)
        return -(np.sum(logw_fn(d.astype(float), th)) - len(d) * special.logsumexp(lw))

    r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    th = r.x
    pw = logw_fn(d.astype(float), th) - special.logsumexp(logw_fn(ks, th))
    return th, -r.fun, pw


def mle_exp(deg, kmin=1):
    """discrete exponential (geometric) ∝ exp(-lam k)."""
    th, ll, pw = _grid_fit(deg, kmin, lambda k, t: -np.exp(t[0]) * k, [np.log(0.1)])
    return np.exp(th[0]), ll, pw


def mle_trunc(deg, kmin=1):
    """power law with exponential cutoff ∝ k^-a exp(-lam k). returns ((a, lam), ll, pw)."""
    best = None
    for a0 in (1.0, 2.0):
        th, ll, pw = _grid_fit(deg, kmin, lambda k, t: -t[0] * np.log(k) - np.exp(t[1]) * k, [a0, np.log(0.02)])
        if best is None or ll > best[1]:
            best = (th, ll, pw)
    th, ll, pw = best
    return (th[0], np.exp(th[1])), ll, pw


def mle_lognormal(deg, kmin=1):
    """discrete lognormal ∝ (1/k) exp(-(ln k - mu)^2 / 2 s^2). returns ((mu, s), ll, pw)."""
    d = np.log(_tail(deg, kmin))
    th, ll, pw = _grid_fit(deg, kmin,
                           lambda k, t: -np.log(k) - (np.log(k) - t[0]) ** 2 / (2 * np.exp(2 * t[1])),
                           [d.mean(), np.log(d.std() + 0.1)])
    return (th[0], np.exp(th[1])), ll, pw


def vuong(pw1, pw2):
    """Vuong closeness test statistic; >0 favours model 1.  returns (z, two-sided p)."""
    r = pw1 - pw2
    z = np.sqrt(len(r)) * r.mean() / (r.std() + 1e-12)
    return z, 2 * stats.norm.sf(abs(z))


# ---------------------------------------------------------------- Clauset kmin, KS, GOF
def ks_pl(d, kmin, alpha):
    ks = np.arange(kmin, d.max() + 1)
    emp = np.cumsum(np.bincount(d, minlength=d.max() + 1)[kmin:]) / len(d)
    mod = 1 - special.zeta(alpha, ks + 1) / special.zeta(alpha, kmin)
    return np.max(np.abs(emp - mod))


def clauset(deg, min_tail=30):
    """choose kmin minimising the KS distance (Clauset, Shalizi, Newman 2009). returns dict."""
    d = deg[deg > 0].astype(int)
    out = None
    for kmin in np.unique(d):
        t = d[d >= kmin]
        if len(t) < min_tail:
            break
        a, ll, n = mle_pl(deg, kmin)
        D = ks_pl(t, kmin, a)
        if out is None or D < out["D"]:
            out = dict(kmin=int(kmin), alpha=a, D=D, ntail=len(t), ll=ll)
    return out


def gof_pvalue(deg, kmin, B=200, rng=None):
    """parametric-bootstrap p-value for the power-law fit above kmin (tail only, kmin fixed)."""
    rng = rng or np.random.default_rng(0)
    t = deg[deg >= kmin].astype(int)
    a, _, n = mle_pl(deg, kmin)
    D0 = ks_pl(t, kmin, a)
    gk = np.arange(kmin, 200000)
    cdf = np.cumsum(gk ** -a)
    cdf /= cdf[-1]
    cnt = 0
    for _ in range(B):
        s = gk[np.searchsorted(cdf, rng.random(n))]
        a_s, _, _ = mle_pl(s, kmin)
        cnt += ks_pl(s, kmin, a_s) >= D0
    return (cnt + 1) / (B + 1), D0


def model_table(deg, kmin=1):
    """log-likelihood, AIC, BIC of the candidate models on k>=kmin (same data for all)."""
    d = _tail(deg, kmin)
    n = len(d)
    a, ll_pl, _ = mle_pl(deg, kmin)
    pw_pl = pl_pointwise_ll(deg, kmin, a)
    (ta, tl), ll_tr, pw_tr = mle_trunc(deg, kmin)
    lam, ll_ex, pw_ex = mle_exp(deg, kmin)
    (mu, s), ll_ln, pw_ln = mle_lognormal(deg, kmin)
    rows = {
        "power law": (ll_pl, 1, pw_pl),
        "exponential": (ll_ex, 1, pw_ex),
        "lognormal": (ll_ln, 2, pw_ln),
        "PL + cutoff": (ll_tr, 2, pw_tr),
    }
    out = {}
    for name, (ll, kp, pw) in rows.items():
        out[name] = dict(ll=ll, k=kp, aic=2 * kp - 2 * ll, bic=kp * np.log(n) - 2 * ll, pw=pw)
    out["_params"] = dict(alpha=a, trunc=(ta, tl), lam=lam, lognorm=(mu, s))
    out["_n"] = n
    return out
