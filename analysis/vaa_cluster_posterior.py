#!/usr/bin/env python3
"""Cluster membership probabilities for the direct and indirect tracks, next to the classifier's.

The two VAA tracks score a stance against each k=4 cluster's mean position only: per
(statement, language) the cluster with the highest agreement 1 - |stance - mean| / 2 wins
the cell. That yields an argmax share, but no probability, so the VAA side and the
cluster-track classifier (which outputs a softmax over the same four clusters) can only be
compared on argmax shares. This script gives each cluster a Gaussian around its member
parties and reads a posterior off it, so both sides can be put side by side as mean
probabilities too. Opt-in: nothing else in the pipeline reads it, and it writes only under
tables/cluster_posterior/.

The Gaussians. The clusters are party_kmeans' k=4 fit, not refitted: a refit would move the
labels away from the ones the classifier was trained on. For each cluster and statement, the
mean and variance are taken over the ~164 MEP-holding member parties' raw answers, weighted
by MEPs, so the mean is exactly the cluster position of build_cluster_positions.py (checked
against the positions file). Prior: uniform, matching the logit-adjusted classifiers. Party
answers are five-point Likert, so a cluster whose members all agree has zero spread; every
sd is floored at --floor (several values = a sensitivity run).

Scoring schemes (`scheme`):
  distance   today's scoring: agreement with the cluster mean, ties split. No probability.
  tied       one sd per statement shared by all clusters (the pooled within-cluster sd).
             In one dimension its argmax is the nearest mean, so this differs from
             `distance` only in adding probabilities -- the argmax share must match.
  cluster    each cluster its own sd per statement: a cluster split on a statement is a
             wide Gaussian there and wins that cell by less.

Units (`unit`):
  cell       one (run or language vector, statement) -- what the argmax figures count and
             the analogue of one classified text. P(cluster | stance on that statement).
  vector     the whole 30-statement answer vector, statements independent (diagonal
             covariance): P(cluster | run). The product of 30 likelihoods is sharp, so
             mean_max_prob is reported to show how far from the classifier's it is.

Levels: `run` scores each paraphrase on its own (the classifier also sees one text at a
time); `language` scores the paraphrase-averaged stance, which is what the figures use.
Framings are pooled with the negated one re-oriented, as in vaa_null_model.

Nulls: the uniform and the 0-centered normal null of vaa_null_model (sigma --sigma), drawn
and averaged exactly as vaa_null_model_argmax does, scored under every scheme, so each
VAA number has a null in its own scheme. The classifier's null is classifier_null_model's,
read through NullBaseline when its files exist.

Comparison with the classifier (the open-ended / speeches classifier, framings pooled):
  model distance   per model, half the L1 distance between the VAA side's per-cluster
                   vector and the classifier's, raw and minus each side's null.
  text agreement   indirect track only: the stance and the classifier read the same text,
                   so they are joined text by text (framing, language, paraphrase,
                   statement): argmax agreement, chance agreement from the marginals,
                   Cohen's kappa, and the mean total-variation distance between the
                   posterior and the classifier's softmax.

Outputs (under --out-dir, default data/<dataset>_results/tables/cluster_posterior):
  cluster_gaussians.csv     cluster, statement_idx, mean, sd (unfloored), sd_tied, parties
  shares.csv                track, scheme, floor, unit, level, side, model, cluster,
                            argmax_share, mean_prob, mean_max_prob
  methods.csv               model, method (direct / indirect / classifier / combined),
                            stat (argmax / gaussian_argmax / mean_prob), cluster, value,
                            null, minus_null -- the methods side by side, and their mean
  classifier.csv            model, cluster, argmax_share, mean_prob, mean_max_prob (+ nulls)
  model_distance.csv        the model-level distances above
  text_agreement.csv        the per-text agreement above
  summary.md                the tables a reader needs; cluster_posterior.png the figure

Needs the cluster basis already run (slurm_analyze_clusters.sh): the classified CSVs must
carry the four cluster labels. If they carry EP groups the classifier comparison is skipped.

  python analysis/vaa_cluster_posterior.py
  python analysis/vaa_cluster_posterior.py --floor 0.1 0.25 0.5 --draws 2000
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.core.labels import use_short_party_labels
use_short_party_labels()   # cluster nicknames in every label (utils.PARTY_SHORT)

from analysis import party_kmeans as pk
from analysis.analyze_all import MODEL_DIRS
from analysis.core import model_display_name
from analysis.core.positions import CLUSTER_POSITIONS_PATH, load_party_positions
from analysis.plotting import plot_classified_parties as pcp
from analysis.vaa_null_model import (
    ALL_MODELS, DEFAULT_SIGMA, averaged_draws, load_runs, make_generators, win_credit,
)
from analysis.vaa_null_model_clusters import K, party_answers
from preprocessing.build_cluster_splits import cluster_parties
from utils import PARTY_DISPLAY_ORDER, PARTY_SHORT, configure_stdout

configure_stdout()

TRACKS = {"likert": "direct", "speeches": "indirect"}
SCHEMES = ["distance", "tied", "cluster"]
UNITS = ["cell", "vector"]
LEVELS = ["run", "language"]
NULL_SIDES = {"uniform": "uniform", "normal": "gaussian"}   # vaa_null_model side -> null kind
DEFAULT_FLOORS = [0.1, 0.25, 0.5]
HEADLINE_FLOOR = 0.25   # half a Likert step on the [-1, 1] stance scale


# --------------------------------------------------------------------------- #
# the Gaussians
# --------------------------------------------------------------------------- #

def cluster_gaussians(cluster_seed, statement_count):
    """(mu, sd, sd_tied, names, parties): (K, T) MEP-weighted mean and sd per cluster and
    statement over the members that answered it, and the pooled within-cluster sd per
    statement (T,) -- the same weights, deviations taken from each party's own cluster."""
    parties = cluster_parties(cluster_seed, K)
    answers = party_answers(parties, statement_count)
    labels = parties["cluster_name"].to_numpy()
    weights = parties["meps"].to_numpy(dtype=float)
    names = [n for n in PARTY_DISPLAY_ORDER if n in pk.CLUSTER_NAMES[K]] or pk.CLUSTER_NAMES[K]

    mu = np.full((len(names), statement_count), np.nan)
    sd = np.full_like(mu, np.nan)
    pooled_ss = np.zeros(statement_count)
    pooled_w = np.zeros(statement_count)
    counts = np.zeros_like(mu)
    for ci, name in enumerate(names):
        member = labels == name
        X = answers[member]
        W = np.where(np.isnan(X), 0.0, weights[member, None])
        total = W.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (np.nan_to_num(X) * W).sum(axis=0) / total
            ss = (W * (np.nan_to_num(X) - m) ** 2).sum(axis=0)
            mu[ci] = np.where(total > 0, m, np.nan)
            sd[ci] = np.where(total > 0, np.sqrt(ss / total), np.nan)
        pooled_ss += np.nan_to_num(ss)
        pooled_w += total
        counts[ci] = (~np.isnan(X)).sum(axis=0)
    sd_tied = np.sqrt(pooled_ss / np.maximum(pooled_w, 1e-12))
    return mu, sd, sd_tied, names, counts


def check_against_positions_file(mu, names):
    """The Gaussian means must be the cluster positions the pipeline scores against."""
    path = Path(CLUSTER_POSITIONS_PATH)
    if not path.exists():
        print(f"  (no {path}; skipping the mean check)")
        return
    df = load_party_positions(str(path), "cluster")
    P = (df.pivot_table(index="ep_group", columns="statement_idx", values="normalized_answer")
           .reindex(index=names, columns=range(mu.shape[1])).to_numpy(dtype=float))
    gap = np.nanmax(np.abs(P - mu))
    print(f"  max |Gaussian mean - {path.name}| = {gap:.4f}")
    if gap > 0.01:
        raise SystemExit("The Gaussian means do not match the cluster positions file: rerun "
                         "build_cluster_positions.py, or the clusters have drifted.")


def floored(sd, sd_tied, scheme, floor):
    """(K, T) sd for a scheme, floored."""
    if scheme == "tied":
        return np.broadcast_to(np.maximum(sd_tied, floor), sd.shape)
    return np.maximum(sd, floor)


# --------------------------------------------------------------------------- #
# scoring: each returns (rows, K) scores and whether they are probabilities
# --------------------------------------------------------------------------- #

def cell_log_likelihood(S, mu, s):
    """(N, K, T) Gaussian log-likelihood of each stance; NaN where stance or mean missing."""
    z = (S[:, None, :] - mu[None]) / s[None]
    return -0.5 * z ** 2 - np.log(s)[None]


def normalise(ll, axis=1):
    """Softmax over the cluster axis, treating NaN as -inf; all-NaN slices stay NaN."""
    finite = np.where(np.isnan(ll), -np.inf, ll)
    top = finite.max(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        e = np.exp(finite - top)
        p = e / e.sum(axis=axis, keepdims=True)
    return np.where(np.isfinite(top), p, np.nan)


def scores(S, mu, sd, sd_tied, scheme, unit, floor):
    """(rows, K) scores. Cell unit: one row per (vector, statement) with a stance.
    Distance scores are mean agreements, the others posterior probabilities."""
    valid_cols = ~np.isnan(mu).any(axis=0)   # a statement some cluster has no mean on
    S, mu, sd = S[:, valid_cols], mu[:, valid_cols], sd[:, valid_cols]
    sd_tied = sd_tied[valid_cols]
    if scheme == "distance":
        A = 1 - np.abs(S[:, None, :] - mu[None]) / 2           # (N, K, T)
        if unit == "cell":
            return np.moveaxis(A, 1, 2).reshape(-1, mu.shape[0])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")
            return np.nanmean(A, axis=2)
    ll = cell_log_likelihood(S, mu, floored(sd, sd_tied, scheme, floor))
    if unit == "cell":
        return normalise(np.moveaxis(ll, 1, 2).reshape(-1, mu.shape[0]))
    total = np.where(np.isnan(ll), 0.0, ll).sum(axis=2)
    total[np.isnan(S).all(axis=1)] = np.nan
    return normalise(total)


def summarise(score, is_prob):
    """(argmax_share, mean_prob, mean_max_prob), each (K,), over rows with any score."""
    keep = ~np.isnan(score).all(axis=1)
    score = score[keep]
    if not len(score):
        nan = np.full(score.shape[1], np.nan)
        return nan, nan, nan
    share = win_credit(score).mean(axis=0)
    if not is_prob:
        nan = np.full(score.shape[1], np.nan)
        return share, nan, np.full(score.shape[1], np.nan)
    mean_prob = np.nanmean(score, axis=0)
    mean_max = np.full(score.shape[1], np.nanmax(score, axis=1).mean())
    return share, mean_prob, mean_max


# --------------------------------------------------------------------------- #
# real runs, nulls
# --------------------------------------------------------------------------- #

def level_vectors(runs):
    """{level: {model: (N, T)}} with ALL_MODELS pooled; framings stay separate rows."""
    out = {}
    by_language = runs.groupby(level=["model", "framing", "language"]).mean()
    for level, frame in (("run", runs), ("language", by_language)):
        vectors = {ALL_MODELS: frame.to_numpy(dtype=float)}
        vectors.update({m: f.to_numpy(dtype=float) for m, f in frame.groupby(level="model")})
        out[level] = vectors
    return out


def null_vectors(runs, sigma, draws, rng):
    """{level: {side: (draws, T)}}, averaged over paraphrases at the language level."""
    paraphrases = int(runs.groupby(level=["model", "framing", "language"]).size().median())
    generators = make_generators(runs, sigma, rng)
    return {level: {side: averaged_draws(generators[side], draws,
                                         1 if level == "run" else paraphrases)
                    for side in NULL_SIDES}
            for level in LEVELS}


def score_track(track, runs, gauss, floors, sigma, draws, rng):
    mu, sd, sd_tied, names, _ = gauss
    real = level_vectors(runs)
    null = null_vectors(runs, sigma, draws, rng)
    records = []
    for scheme in SCHEMES:
        for floor in ([np.nan] if scheme == "distance" else floors):
            for unit in UNITS:
                for level in LEVELS:
                    sides = [("real", m, S) for m, S in real[level].items()]
                    sides += [(side, ALL_MODELS, S) for side, S in null[level].items()]
                    for side, model, S in sides:
                        share, prob, mx = summarise(
                            scores(S, mu, sd, sd_tied, scheme, unit, floor),
                            scheme != "distance")
                        for ci, name in enumerate(names):
                            records.append(dict(
                                track=track, scheme=scheme, floor=floor, unit=unit,
                                level=level, side=side, model=model, cluster=name,
                                argmax_share=share[ci], mean_prob=prob[ci],
                                mean_max_prob=mx[ci]))
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #

def classifier_wide(model_dir, names):
    """One row per classified open-ended text: framing, language, paraphrase,
    statement_idx and one probability column per cluster. None when the CSVs are absent
    or carry another label set."""
    by_slug = {pcp.slugify_party_label(n): n for n in names}
    frames = []
    for (source, framing), path in pcp.discover_classified_csvs(model_dir).items():
        if source != "speeches":
            continue
        long = pcp.load_long_predictions(path).dropna(subset=["probability"])
        if long.empty:
            continue
        long["party"] = long["party"].map(lambda p: p if p in names else by_slug.get(p, p))
        unknown = set(long["party"]) - set(names)
        if unknown:
            print(f"  {model_dir.name}: {path.name} carries {sorted(unknown)[:4]}..., not the "
                  "k=4 clusters -- rerun the cluster-basis classification; skipping it.")
            return None
        wide = (long.pivot_table(index=["row_idx", "language", "variant_idx"],
                                 columns="party", values="probability")
                    .reindex(columns=names).reset_index()
                    .rename(columns={"row_idx": "statement_idx", "variant_idx": "paraphrase"}))
        frames.append(wide.assign(framing=framing))
    return pd.concat(frames, ignore_index=True) if frames else None


def classifier_summary(wide, names):
    """Per framing then averaged (plot_argmax_shares' 'pooled'): argmax share, mean
    probability, mean max probability."""
    per_framing = []
    for _, frame in wide.groupby("framing"):
        P = frame[names].to_numpy(dtype=float)
        share, prob, mx = summarise(P, True)
        per_framing.append(np.vstack([share, prob, mx]))
    return np.mean(per_framing, axis=0)


def load_classifier_null(null_dir, sigma):
    """{kind: (null share Series, null mean-probability Series)} for speeches_clf pooled,
    from classifier_null_model's files; empty for any kind whose files are missing."""
    from analysis.null_baseline import NullBaseline
    out = {}
    for kind in NULL_SIDES.values():
        try:
            baseline = NullBaseline.load(null_dir, "cluster", sigma, kind)
            out[kind] = (baseline.share("speeches_clf", "pooled"),
                         baseline.mean_probability("speeches_clf", "pooled"))
        except SystemExit as error:
            print(f"  no classifier {kind} null: {str(error).splitlines()[0][:160]}")
    return out


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #

def half_l1(a, b):
    return 0.5 * float(np.nansum(np.abs(np.asarray(a) - np.asarray(b))))


def model_distances(shares, clf, clf_null, names):
    """Per (track, scheme, floor, unit, level, model, stat, null): half-L1 distance between
    the VAA side's per-cluster vector and the classifier's. stat "argmax" compares argmax
    shares, "prob" mean probabilities (VAA posterior schemes only)."""
    records = []
    real = shares[shares["side"] == "real"]
    keys = ["track", "scheme", "floor", "unit", "level"]
    nulls = shares[shares["side"] != "real"]
    for key, frame in real.groupby(keys, dropna=False):
        null_frame = nulls.merge(pd.DataFrame([dict(zip(keys, key))]), on=keys)
        for model, mframe in frame.groupby("model"):
            if model not in clf.index.get_level_values("model"):
                continue
            vaa = mframe.set_index("cluster").reindex(names)
            c = clf.loc[model].reindex(names)
            for stat, column, clf_column in (("argmax", "argmax_share", "argmax_share"),
                                             ("prob", "mean_prob", "mean_prob")):
                if vaa[column].isna().all():
                    continue
                base = dict(zip(keys, key), model=model, stat=stat)
                records.append({**base, "null": "none",
                                "distance": half_l1(vaa[column], c[clf_column]),
                                "same_top": vaa[column].idxmax() == c[clf_column].idxmax()})
                for side, kind in NULL_SIDES.items():
                    if kind not in clf_null:
                        continue
                    vnull = (null_frame[null_frame["side"] == side]
                             .set_index("cluster")[column].reindex(names))
                    cnull = clf_null[kind][0 if stat == "argmax" else 1].reindex(names)
                    a, b = vaa[column] - vnull, c[clf_column] - cnull
                    records.append({**base, "null": kind, "distance": half_l1(a, b),
                                    "same_top": a.idxmax() == b.idxmax()})
    return pd.DataFrame.from_records(records)


def text_agreement(runs, wide_by_model, gauss, floors):
    """Indirect track, run level: the stance-side scores of each open-ended text joined to
    the classifier's softmax over the same text."""
    mu, sd, sd_tied, names, _ = gauss
    records = []
    for model, wide in wide_by_model.items():
        if model not in runs.index.get_level_values("model"):
            continue
        frame = runs.xs(model, level="model")
        S = frame.to_numpy(dtype=float)
        # scores() returns cell rows in (vector, statement) order; key them the same way.
        T = S.shape[1]
        keys = (frame.index.to_frame(index=False)
                     .iloc[np.repeat(np.arange(len(frame)), T)].reset_index(drop=True))
        keys["statement_idx"] = np.tile(frame.columns.to_numpy(dtype=int), len(frame))
        joined_keys = ["framing", "language", "paraphrase", "statement_idx"]
        for scheme in SCHEMES:
            for floor in ([np.nan] if scheme == "distance" else floors):
                # scores() drops statements some cluster has no mean on; there are none in
                # practice, and the reshape below relies on it.
                if np.isnan(mu).any():
                    raise SystemExit("a cluster has no mean on some statement")
                sc = scores(S, mu, sd, sd_tied, scheme, "cell", floor)
                vaa = pd.concat([keys, pd.DataFrame(sc, columns=names)], axis=1)
                vaa = vaa.dropna(subset=names, how="all")
                both = vaa.merge(wide, on=joined_keys, suffixes=("_vaa", "_clf"))
                if both.empty:
                    continue
                V = both[[f"{n}_vaa" for n in names]].to_numpy(dtype=float)
                C = both[[f"{n}_clf" for n in names]].to_numpy(dtype=float)
                v_credit, c_top = win_credit(V), C.argmax(axis=1)
                agree = v_credit[np.arange(len(C)), c_top].mean()
                chance = float((v_credit.mean(axis=0) * np.eye(len(names))[c_top].mean(axis=0)).sum())
                records.append(dict(
                    model=model, scheme=scheme, floor=floor, texts=len(both),
                    agreement=agree, chance=chance,
                    kappa=(agree - chance) / (1 - chance) if chance < 1 else np.nan,
                    mean_tv=(0.5 * np.abs(V - C).sum(axis=1)).mean()
                    if scheme != "distance" else np.nan))
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def pct(x):
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def scheme_label(scheme, floor):
    return scheme if scheme == "distance" else f"{scheme} (floor {floor:g})"


def select(shares, **kw):
    out = shares
    for k, v in kw.items():
        out = out[out[k].isna()] if isinstance(v, float) and np.isnan(v) else out[out[k] == v]
    return out


def pooled_table(shares, clf, clf_null, names, level, unit, floors):
    """All models pooled: per cluster, raw and minus-null, every scheme, both tracks, and
    the classifier."""
    rows = []
    for track in TRACKS:
        for scheme in SCHEMES:
            for floor in ([np.nan] if scheme == "distance" else floors):
                sub = select(shares, track=track, scheme=scheme, unit=unit, level=level,
                             floor=floor)
                real = select(sub, side="real", model=ALL_MODELS).set_index("cluster").reindex(names)
                normal = select(sub, side="normal").set_index("cluster").reindex(names)
                for stat in ("argmax_share", "mean_prob"):
                    if real[stat].isna().all():
                        continue
                    rows.append([TRACKS[track], scheme_label(scheme, floor),
                                 "argmax" if stat == "argmax_share" else "mean prob",
                                 *(f"{pct(real[stat][n])} ({100 * (real[stat][n] - normal[stat][n]):+.1f})"
                                   for n in names),
                                 pct(real["mean_max_prob"].iloc[0])])
    if clf is not None:
        pooled = clf.groupby("cluster").mean(numeric_only=True).reindex(names)
        null = clf_null.get("gaussian")
        for stat, idx in (("argmax_share", 0), ("mean_prob", 1)):
            cells = []
            for n in names:
                v = pooled[stat][n]
                cells.append(f"{pct(v)} ({100 * (v - null[idx].get(n, np.nan)):+.1f})"
                             if null is not None else pct(v))
            rows.append(["classifier", "open-ended", "argmax" if idx == 0 else "mean prob",
                         *cells, pct(pooled["mean_max_prob"].iloc[0])])
    return md_table(["track", "scheme", "stat", *names, "mean max prob"], rows)


def distance_table(dist, level, unit, floors):
    rows = []
    for track in TRACKS:
        for scheme in SCHEMES:
            for floor in ([np.nan] if scheme == "distance" else floors):
                sub = select(dist, track=track, scheme=scheme, level=level, unit=unit,
                             floor=floor)
                cells = []
                for stat in ("argmax", "prob"):
                    for null in ("none", "gaussian"):
                        s = select(sub, stat=stat, null=null)
                        cells.append("" if s.empty else
                                     f"{100 * s['distance'].mean():.1f} "
                                     f"({int(s['same_top'].sum())}/{len(s)})")
                rows.append([TRACKS[track], scheme_label(scheme, floor), *cells])
    return md_table(["track", "scheme", "argmax raw", "argmax − null",
                     "prob raw", "prob − null"], rows)


def agreement_table(agree, floors):
    rows = []
    for scheme in SCHEMES:
        for floor in ([np.nan] if scheme == "distance" else floors):
            sub = select(agree, scheme=scheme, floor=floor)
            if sub.empty:
                continue
            w = sub["texts"]
            rows.append([scheme_label(scheme, floor), int(w.sum()),
                         pct(np.average(sub["agreement"], weights=w)),
                         pct(np.average(sub["chance"], weights=w)),
                         f"{np.average(sub['kappa'], weights=w):.3f}",
                         pct(np.average(sub["mean_tv"], weights=w))
                         if sub["mean_tv"].notna().all() else ""])
    return md_table(["scheme", "texts", "argmax agreement %", "chance %", "kappa",
                     "mean TV %"], rows)


def spread_table(gauss):
    mu, sd, sd_tied, names, counts = gauss
    rows = [[n, int(np.median(counts[ci])), f"{np.nanmin(sd[ci]):.2f}",
             f"{np.nanmedian(sd[ci]):.2f}", f"{np.nanmax(sd[ci]):.2f}",
             *(int((sd[ci] < f).sum()) for f in (0.1, 0.25, 0.5))]
            for ci, n in enumerate(names)]
    rows.append(["tied (pooled)", "", f"{sd_tied.min():.2f}", f"{np.median(sd_tied):.2f}",
                 f"{sd_tied.max():.2f}", *(int((sd_tied < f).sum()) for f in (0.1, 0.25, 0.5))])
    return md_table(["cluster", "parties (median)", "sd min", "sd median", "sd max",
                     "statements < .1", "< .25", "< .5"], rows)


# --------------------------------------------------------------------------- #
# the methods side by side, and the combined method
# --------------------------------------------------------------------------- #

METHODS = ["direct", "indirect", "classifier", "combined"]
STATS = {
    "argmax": "argmax share",                  # what the pipeline's figures report
    "gaussian_argmax": "Gaussian argmax share",
    "mean_prob": "mean probability",
}
# Where each VAA statistic comes from in shares.csv: (scheme, column).
VAA_STAT_SOURCE = {
    "argmax": ("distance", "argmax_share"),
    "gaussian_argmax": ("cluster", "argmax_share"),
    "mean_prob": ("cluster", "mean_prob"),
}
# The classifier has no Gaussian scoring: its argmax is its argmax under either name.
CLASSIFIER_STAT_SOURCE = {"argmax": ("argmax_share", 0), "gaussian_argmax": ("argmax_share", 0),
                          "mean_prob": ("mean_prob", 1)}


def method_frame(shares, clf, clf_null, names, floor):
    """model, method, stat, cluster, value, null, minus_null: every method under every
    statistic, language level, cell unit. "combined" is the mean over the other methods,
    plot_argmax_shares' "average of methods", over the models every method has; the
    ALL_MODELS rows are the plain mean over models. Nulls: the normal (gaussian) one."""
    sub = shares[(shares["level"] == "language") & (shares["unit"] == "cell")]
    records = []
    for track, method in TRACKS.items():
        for stat, (scheme, column) in VAA_STAT_SOURCE.items():
            s = select(sub, track=track, scheme=scheme,
                       floor=np.nan if scheme == "distance" else floor)
            null = select(s, side="normal").set_index("cluster")[column]
            real = select(s, side="real")
            for model, cluster, value in zip(real["model"], real["cluster"], real[column]):
                if model != ALL_MODELS:
                    records.append(dict(model=model, method=method, stat=stat,
                                        cluster=cluster, value=value, null=null.get(cluster)))
    if clf is not None:
        cnull = clf_null.get("gaussian")
        for (model, cluster), row in clf.iterrows():
            for stat, (column, idx) in CLASSIFIER_STAT_SOURCE.items():
                records.append(dict(model=model, method="classifier", stat=stat,
                                    cluster=cluster, value=row[column],
                                    null=cnull[idx].get(cluster) if cnull is not None else np.nan))
    frame = pd.DataFrame.from_records(records)
    present = frame.groupby("model")["method"].nunique()
    full = present[present == present.max()].index
    combined = (frame[frame["model"].isin(full)]
                .groupby(["model", "stat", "cluster"], as_index=False)[["value", "null"]].mean()
                .assign(method="combined"))
    frame = pd.concat([frame, combined], ignore_index=True)
    pooled = (frame.groupby(["method", "stat", "cluster"], as_index=False)[["value", "null"]]
                   .mean().assign(model=ALL_MODELS))
    frame = pd.concat([frame, pooled], ignore_index=True)
    frame["minus_null"] = frame["value"] - frame["null"]
    return frame


def top_minus_null(frame, model, method, stat, names):
    """The cluster with the largest share above its null, as its short name."""
    s = frame[(frame["model"] == model) & (frame["method"] == method) & (frame["stat"] == stat)]
    s = s.set_index("cluster")["minus_null"].reindex(names)
    return "" if s.isna().all() else short_name(s.idxmax())


def short_name(name):
    return PARTY_SHORT.get(name, name)


def methods_pooled_table(frame, names):
    rows = []
    for method in METHODS:
        for stat, label in STATS.items():
            if method == "classifier" and stat == "gaussian_argmax":
                continue   # identical to its argmax
            s = select(frame, model=ALL_MODELS, method=method, stat=stat).set_index("cluster")
            if s.empty:
                continue
            s = s.reindex(names)
            rows.append([method, label,
                         *(f"{pct(s['value'][n])} ({100 * s['minus_null'][n]:+.1f})" for n in names),
                         top_minus_null(frame, ALL_MODELS, method, stat, names)])
    return md_table(["method", "statistic", *names, "top − null"], rows)


def methods_per_model_table(frame, names):
    """Per model and method: the top cluster above the null under argmax / Gaussian argmax /
    mean probability; the last rows count, per method, the models where the mean
    probability names the same top cluster as the pipeline's argmax."""
    models = [m for m in frame["model"].unique() if m != ALL_MODELS]
    rows = []
    for model in models:
        cells = []
        for method in METHODS:
            tops = [top_minus_null(frame, model, method, stat, names) for stat in STATS]
            cells.append(" / ".join(tops) if any(tops) else "")
        rows.append([model_display_name(model), *cells])
    same = []
    for method in METHODS:
        agree = [top_minus_null(frame, m, method, "argmax", names)
                 == top_minus_null(frame, m, method, "mean_prob", names) for m in models
                 if top_minus_null(frame, m, method, "argmax", names)]
        same.append(f"{sum(agree)}/{len(agree)}")
    rows.append(["**prob top = argmax top**", *same])
    return md_table(["model", *(f"{m}: argmax / Gauss. argmax / prob" for m in METHODS)], rows)


def methods_against_combined(frame, names):
    """Half-L1 distance (pp, minus null) of each method to the combined method, per
    statistic, mean over models: how much each method pulls the combination."""
    models = [m for m in frame["model"].unique() if m != ALL_MODELS]
    rows = []
    for method in METHODS[:-1]:
        cells = []
        for stat in STATS:
            d = []
            for model in models:
                a = select(frame, model=model, method=method, stat=stat).set_index("cluster")
                b = select(frame, model=model, method="combined", stat=stat).set_index("cluster")
                if not a.empty and not b.empty:
                    d.append(half_l1(a["minus_null"].reindex(names), b["minus_null"].reindex(names)))
            cells.append(f"{100 * np.mean(d):.1f}" if d else "")
        rows.append([method, *cells])
    return md_table(["method", *STATS.values()], rows)


def plot(frame, names, floor, out_path):
    """All models (mean), minus the normal null: per method the pipeline's argmax share,
    the Gaussian argmax share and the mean probability."""
    methods = [m for m in METHODS if (frame["method"] == m).any()]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(names))
    styles = {"argmax": ("#4C72B0", ""), "gaussian_argmax": ("#55A868", ""),
              "mean_prob": ("#55A868", "///")}
    width = 0.84 / len(STATS)
    for ax, method in zip(axes, methods):
        for si, (stat, label) in enumerate(STATS.items()):
            if method == "classifier" and stat == "gaussian_argmax":
                continue
            s = select(frame, model=ALL_MODELS, method=method, stat=stat).set_index("cluster")
            color, hatch = styles[stat]
            ax.bar(x - 0.42 + width * (si + 0.5),
                   100 * s["minus_null"].reindex(names).to_numpy(), width=width,
                   color=color if not hatch else "white", edgecolor=color, hatch=hatch,
                   linewidth=1.0, label=label)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(n) for n in names], fontsize=9)
        ax.set_title(method + (" (no Gaussian scoring)" if method == "classifier" else ""),
                     fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("value − normal null (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Mean over models, paraphrase-averaged stances; Gaussian sd floor {floor:g}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--models", nargs="+", default=MODEL_DIRS)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--out-dir", default=None,
                        help="default: <results_dir>/tables/cluster_posterior")
    parser.add_argument("--null_dir", default=None,
                        help="classifier_null_model's files; default <results_dir>/tables/null_model")
    parser.add_argument("--floor", nargs="+", type=float, default=DEFAULT_FLOORS,
                        help="sd floors in stance units (default %(default)s); the headline "
                             "tables use the one closest to 0.25.")
    parser.add_argument("--cluster-seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    args = parser.parse_args()

    results_root = Path(args.results_dir or Path("data") / f"{args.dataset}_results")
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "tables" / "cluster_posterior"
    out_dir.mkdir(parents=True, exist_ok=True)
    null_dir = Path(args.null_dir) if args.null_dir else results_root / "tables" / "null_model"
    model_dirs = [results_root / m for m in args.models]
    rng = np.random.default_rng(args.seed)
    floors = sorted(set(args.floor))
    headline = min(floors, key=lambda f: abs(f - HEADLINE_FLOOR))

    runs = {track: load_runs(model_dirs, track) for track in TRACKS}
    statement_count = runs["likert"].shape[1]

    print(f"Clustering MEP-holding EU&I parties (k={K}, seed {args.cluster_seed})")
    gauss = cluster_gaussians(args.cluster_seed, statement_count)
    mu, sd, sd_tied, names, counts = gauss
    check_against_positions_file(mu, names)
    pd.DataFrame([dict(cluster=n, statement_idx=t, mean=mu[ci, t], sd=sd[ci, t],
                       sd_tied=sd_tied[t], parties=int(counts[ci, t]))
                  for ci, n in enumerate(names) for t in range(statement_count)]
                 ).to_csv(out_dir / "cluster_gaussians.csv", index=False)

    shares = pd.concat([score_track(track, runs[track], gauss, floors, args.sigma,
                                    args.draws, rng) for track in TRACKS], ignore_index=True)
    shares.to_csv(out_dir / "shares.csv", index=False)

    print("Loading the open-ended classifier outputs")
    wide_by_model, clf_records = {}, []
    for model_dir in model_dirs:
        wide = classifier_wide(model_dir, names)
        if wide is None:
            continue
        wide_by_model[model_dir.name] = wide
        stats = classifier_summary(wide, names)
        clf_records += [dict(model=model_dir.name, cluster=n, argmax_share=stats[0][ci],
                             mean_prob=stats[1][ci], mean_max_prob=stats[2][ci])
                        for ci, n in enumerate(names)]
    clf = (pd.DataFrame(clf_records).set_index(["model", "cluster"])
           if clf_records else None)
    clf_null = load_classifier_null(null_dir, args.sigma) if clf is not None else {}

    parts = [
        "# Cluster membership probabilities: VAA tracks vs the classifier", "",
        f"k={K} clusters, MEP-weighted Gaussians per cluster and statement, uniform prior. "
        f"sd floors {', '.join(f'{f:g}' for f in floors)} (headline {headline:g}). Nulls: "
        f"{args.draws} draws, normal sigma {args.sigma:g}. Framings pooled.", "",
        "## Cluster spread per statement (unfloored sd, stance units)", "",
        spread_table(gauss), "",
    ]
    methods = method_frame(shares, clf, clf_null, names, headline)
    methods.to_csv(out_dir / "methods.csv", index=False)
    parts += [
        f"## The methods and their combination, argmax vs probability (floor {headline:g})", "",
        "Mean over models, paraphrase-averaged stances (the figures' unit). Percent; in "
        "parentheses minus the normal (gaussian) null, pp. `argmax share` is what the "
        "pipeline's figures report (distance scoring for the two VAA tracks); `combined` "
        "is the mean over direct, indirect and the open-ended classifier, as the figures' "
        "average of methods. The classifier has no Gaussian scoring, so its Gaussian argmax "
        "is its argmax.", "",
        methods_pooled_table(methods, names), "",
        "### Top cluster above the null, per model", "",
        methods_per_model_table(methods, names), "",
        "### Distance of each method to the combined one (half L1 minus null, pp, mean "
        "over models)", "",
        methods_against_combined(methods, names), "",
    ]
    for level in LEVELS:
        parts += [f"## All models pooled, {level} level, cell unit", "",
                  "Percent; in parentheses minus the normal (gaussian) null, pp. The "
                  "classifier row is the open-ended classifier minus its own null.", "",
                  pooled_table(shares.reset_index(drop=True),
                               clf.reset_index() if clf is not None else None,
                               clf_null, names, level, "cell", floors), ""]
    parts += ["## All models pooled, run level, vector unit (P(cluster | whole run))", "",
              pooled_table(shares, clf.reset_index() if clf is not None else None,
                           clf_null, names, "run", "vector", floors), ""]

    if clf is not None:
        clf_out = clf.reset_index()
        for kind, (share, prob) in clf_null.items():
            clf_out[f"null_share_{kind}"] = clf_out["cluster"].map(share)
            clf_out[f"null_mean_prob_{kind}"] = clf_out["cluster"].map(prob)
        clf_out.to_csv(out_dir / "classifier.csv", index=False)
        dist = model_distances(shares, clf, clf_null, names)
        dist.to_csv(out_dir / "model_distance.csv", index=False)
        agree = text_agreement(runs["speeches"], wide_by_model, gauss, floors)
        agree.to_csv(out_dir / "text_agreement.csv", index=False)
        for level in LEVELS:
            parts += [f"## Distance to the classifier per model, {level} level, cell unit", "",
                      "Mean over models of half the L1 distance (pp) between the VAA "
                      "side's per-cluster vector and the classifier's; in parentheses the "
                      "models whose top cluster agrees. 'prob' compares mean probabilities.",
                      "", distance_table(dist, level, "cell", floors), ""]
        parts += ["## Distance to the classifier per model, run level, vector unit", "",
                  distance_table(dist, "run", "vector", floors), "",
                  "## Text by text, indirect track (run level, cell unit)", "",
                  "The stance read off an open-ended answer and the classifier's softmax "
                  "over the same answer. Chance = agreement expected from the two sides' "
                  "marginal argmax shares.", "",
                  agreement_table(agree, floors), ""]
        per_model = select(agree, scheme="cluster", floor=headline)
        parts += [f"### Per model, Gaussian (floor {headline:g})", "",
                  md_table(["model", "texts", "agreement %", "kappa", "mean TV %"],
                           [[model_display_name(r.model), r.texts, pct(r.agreement),
                             f"{r.kappa:.3f}", pct(r.mean_tv)]
                            for r in per_model.itertuples()]), ""]
    else:
        parts += ["No classified CSVs with the k=4 labels were found; classifier "
                  "comparison skipped.", ""]

    plot(methods, names, headline, out_dir / "cluster_posterior.png")
    summary = "\n".join(parts)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote {out_dir}/: cluster_gaussians.csv, shares.csv, summary.md, "
          "cluster_posterior.png" + (", classifier.csv, model_distance.csv, "
                                     "text_agreement.csv" if clf is not None else ""))


if __name__ == "__main__":
    main()
