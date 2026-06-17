#!/usr/bin/env python3
"""Distribution-free statistics for the Optobot analyses.

Everything here avoids normality assumptions:

- :func:`permutation_test` — two-sided permutation test on a difference of means
  (or medians) between two independent samples.
- :func:`bootstrap_ci` — percentile bootstrap confidence interval for a 1-D sample.
- :func:`bootstrap_ci_matrix` — per-column bootstrap CI resampling over rows
  (units, e.g. flies); used for mean +/- CI bands of time-series / PSTH curves.
- :func:`bh_adjust` — Benjamini-Hochberg FDR correction over a set of p-values.

Resampling is seeded (``numpy`` ``default_rng``) so figures are reproducible.
"""

from __future__ import annotations

import numpy as np


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def _stat_fn(statistic):
    if statistic == "mean":
        return np.mean
    if statistic == "median":
        return np.median
    raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")


def permutation_test(a, b, n_perm=10000, statistic="mean", seed=0):
    """Two-sided permutation test for a difference between two samples.

    Returns ``(observed_difference, p_value)`` where the difference is
    ``stat(a) - stat(b)``. ``p`` uses the standard ``(#>=obs + 1)/(n_perm + 1)``
    estimator (never exactly 0). NaNs are dropped; returns ``(nan, nan)`` if either
    sample is empty.
    """
    a, b = _clean(a), _clean(b)
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    stat = _stat_fn(statistic)
    obs = stat(a) - stat(b)
    pooled = np.concatenate([a, b])
    na = a.size
    rng = np.random.default_rng(seed)
    # Vectorised: argsort of random keys gives n_perm independent permutations.
    order = rng.random((n_perm, pooled.size)).argsort(axis=1)
    perm = pooled[order]
    diffs = stat(perm[:, :na], axis=1) - stat(perm[:, na:], axis=1)
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p)


def paired_permutation_test(x, y, n_perm=10000, statistic="mean", seed=0):
    """Two-sided paired permutation test (sign-flip) on per-unit differences.

    ``x`` and ``y`` are paired (same length); tests whether ``stat(x - y)`` differs
    from 0 by randomly flipping the sign of each pair's difference.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan, np.nan
    stat = _stat_fn(statistic)
    obs = stat(d)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = stat(signs * d, axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p)


def bootstrap_ci(values, n_boot=10000, ci=95.0, statistic="mean", seed=0):
    """Percentile bootstrap CI for a 1-D sample. Returns ``(point, lo, hi)``."""
    v = _clean(values)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    stat = _stat_fn(statistic)
    point = float(stat(v))
    if v.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = stat(v[idx], axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return point, float(lo), float(hi)


def bootstrap_ci_matrix(matrix, n_boot=1000, ci=95.0, seed=0):
    """Per-column mean + bootstrap CI, resampling over rows (units/flies).

    Args:
        matrix: 2-D array ``(n_units, n_points)``; NaNs allowed (nan-mean used).
    Returns:
        ``(point, lo, hi)`` each shape ``(n_points,)``. With <=1 unit the CI
        collapses to the point estimate.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2-D (n_units, n_points)")
    n = M.shape[0]
    with np.errstate(invalid="ignore"):
        point = np.nanmean(M, axis=0) if n else np.full(M.shape[1], np.nan)
    if n <= 1:
        return point, point.copy(), point.copy()
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, M.shape[1]))
    with np.errstate(invalid="ignore"):
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = np.nanmean(M[idx], axis=0)
        lo = np.nanpercentile(boot, (100 - ci) / 2, axis=0)
        hi = np.nanpercentile(boot, 100 - (100 - ci) / 2, axis=0)
    return point, lo, hi


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values; NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    m = int(mask.sum())
    if m == 0:
        return out
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out[order] = np.minimum(adj, 1.0)
    return out
