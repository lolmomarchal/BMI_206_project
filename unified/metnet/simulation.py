"""Known-truth simulation: can each estimator recover an exponent we planted?

Networks of n nodes with degrees drawn from a bounded power law on k = 1..kmax. The truth is
reported as the magnitude |gamma| = alpha; bias = (estimate - truth) in |gamma|, so a negative
bias means the fitted slope is too shallow (the paper-style OLS direction).
"""
import numpy as np
import pandas as pd
from . import regression as R


def run(n=1000, reps=300, alphas=(1.5, 2.0, 2.5, 3.0), kmaxes=(200, 10000), seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for kmax in kmaxes:
        for a in alphas:
            est = {m: [] for m in R.ESTIMATORS}
            for _ in range(reps):
                d = R.sample_power_law(rng, n, a, kmax)
                for m, (f, _) in R.ESTIMATORS.items():
                    est[m].append(-f(d))          # magnitude
            for m, v in est.items():
                v = np.array(v)
                rows.append(dict(kmax=kmax, true=a, method=m, mean_est=v.mean(), bias=v.mean() - a,
                                 sd=v.std(), rmse=np.sqrt(np.mean((v - a) ** 2))))
    return pd.DataFrame(rows)


def example(n=1000, alpha=2.0, kmax=10000, seed=7):
    """One simulated degree sequence, to show the flat 1/n floor of singleton degrees."""
    rng = np.random.default_rng(seed)
    return R.sample_power_law(rng, n, alpha, kmax)
