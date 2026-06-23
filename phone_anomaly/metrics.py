"""Evaluation metrics with statistical rigour.

- :func:`bootstrap_auroc` — AUROC with a 95% bootstrap confidence interval.
- :func:`delong_test` — DeLong test for whether two AUROCs differ significantly.
"""

import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score


def bootstrap_auroc(y, s, n_boot=1000, seed=42):
    """Return ``(mean_auroc, ci_low, ci_high)`` via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    return float(np.mean(aucs)), float(lo), float(hi)


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds, m):
    """Fast DeLong covariance estimate. ``preds`` is (k, n), positives first."""
    k, n = preds.shape
    neg = n - m
    tx = np.empty((k, m))
    ty = np.empty((k, neg))
    tz = np.empty((k, n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r, :])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / neg
    v01 = (tz[:, :m] - tx) / neg
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / neg


def delong_test(y, s1, s2):
    """Compare two scorers on the same labels.

    Returns ``((auc1, auc2), z, p_value)``. ``p < 0.05`` means the AUROC gap is
    statistically significant.
    """
    y = np.asarray(y)
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    order = np.argsort(-y)  # positives (label 1) first
    ys = y[order]
    m = int(ys.sum())
    preds = np.vstack([s1[order], s2[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var = max(var, 1e-12)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - st.norm.cdf(abs(z)))
    return (float(aucs[0]), float(aucs[1])), float(z), float(p)
