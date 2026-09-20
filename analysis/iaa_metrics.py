#!/usr/bin/env python3
"""Two-rater agreement statistics shared by the annotation-agreement scripts.

Everything here assumes two raters, one ordinal label each, no missing values.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def krippendorff_alpha_ordinal(a, b, levels):
    """Ordinal Krippendorff's alpha for two raters with no missing values."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    size = len(levels)
    index = {level: i for i, level in enumerate(levels)}
    coincidence = np.zeros((size, size))
    for x, y in zip(a, b):
        coincidence[index[x], index[y]] += 1
        coincidence[index[y], index[x]] += 1

    counts = coincidence.sum(axis=1)
    total = counts.sum()
    # ordinal metric: squared sum of the marginals spanned, ends half-counted
    delta = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                lo, hi = min(i, j), max(i, j)
                delta[i, j] = (counts[lo:hi + 1].sum() - (counts[lo] + counts[hi]) / 2) ** 2

    observed = (coincidence * delta).sum() / total
    expected = (np.outer(counts, counts) * delta).sum() / (total * (total - 1))
    if expected == 0:  # every label identical -> no disagreement to explain
        return float("nan")
    return 1.0 - observed / expected


def gwet_ac1(a, b):
    """Gwet's AC1. Unlike kappa it stays interpretable when one category
    dominates, so it is the honest companion figure on skewed label sets.
    Chance agreement is estimated over the categories actually used."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    used = np.unique(np.concatenate([a, b]))
    if len(used) < 2:
        return float("nan")
    observed = np.mean(a == b)
    pi = np.array([((a == k).sum() + (b == k).sum()) / (2 * len(a)) for k in used])
    expected = (pi * (1 - pi)).sum() / (len(used) - 1)
    return (observed - expected) / (1 - expected)


def _safe_corr(fn, a, b):
    """Correlations are undefined when a rater used a single label."""
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return fn(a, b)[0]


def agreement_metrics(a, b, name, levels, n_boot=0, seed=0):
    """The core two-rater row. `levels` is the full declared scale."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    metrics = {
        "pair": name,
        "n": len(a),
        "exact": np.mean(a == b),
        "within1": np.mean(np.abs(a - b) <= 1),
        "MAE": np.mean(np.abs(a - b)),
        "kappa": cohen_kappa_score(a, b, labels=levels),
        "kappa_lin": cohen_kappa_score(a, b, labels=levels, weights="linear"),
        "kappa_quad": cohen_kappa_score(a, b, labels=levels, weights="quadratic"),
        "alpha_ord": krippendorff_alpha_ordinal(a, b, levels),
        "AC1": gwet_ac1(a, b),
        "spearman": _safe_corr(spearmanr, a, b),
        "pearson": _safe_corr(pearsonr, a, b),
    }
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = [krippendorff_alpha_ordinal(a[i], b[i], levels)
                 for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
        draws = [d for d in draws if np.isfinite(d)]
        if draws:
            metrics["alpha_lo"], metrics["alpha_hi"] = np.percentile(draws, [2.5, 97.5])
    return metrics


def show(rows, title):
    print(f"\n=== {title} ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def confusion(a, b, row_name, col_name, levels, note=""):
    print(f"\n{row_name} (rows) x {col_name} (cols){note}")
    matrix = pd.DataFrame(confusion_matrix(a, b, labels=levels),
                          index=pd.Index(levels, name=row_name),
                          columns=pd.Index(levels, name=col_name))
    print(matrix.to_string())


def breakdown(df, a_col, b_col, keys, levels, title):
    rows = []
    for key in keys:
        if key not in df.columns:
            continue
        for value, group in df.groupby(key):
            rows.append(agreement_metrics(group[a_col], group[b_col], f"{key}={value}", levels))
    show(rows, title)
