"""How the direct and indirect tracks score a stance against the groups it is compared to.

Two scorings, one per positions basis family:

  posterior  (cluster basis, the default there) each k=4 cluster is a Gaussian per
             statement -- the MEP-weighted mean and sd over its member parties' answers,
             vaa_cluster_posterior.cluster_gaussians, the sd floored at POSTERIOR_FLOOR --
             and a stance point scores P(cluster | stance), uniform prior. One point is
             one (statement, language) cell for the argmax figures and the agreement
             panels, one (statement, language, paraphrase) for the rank tables: the same
             units the distance score was computed on. The probabilities sum to one over
             the clusters, like the classifier's softmax, so the two sides share a scale.
  distance   (every other basis) 1 - |position - stance| / 2, averaged over the parties
             standing for a group. The only scoring defined there: the Gaussians need the
             cluster members' answers.

VAA_SCORING=distance (or analyze_all.py --vaa-scoring distance) puts the cluster basis back
on the distance score; VAA_SCORING=posterior on a non-cluster basis is an error.

Every place that scores a VAA stance goes through here -- plot_argmax_shares' cells, the
VAA null (vaa_null_model_argmax), the agreement matrices behind plot_classified_parties'
and plot_topical_parties' panels, rank_consistency_tables' per-statement scores and its
closed-form null (null_baseline.vaa_statement_null) -- so a figure and the null it is
drawn minus are always scored the same way. The VAA null files carry the scoring in their
name (scoring_suffix), so a posterior run can never read a distance null or the reverse.

evaluate_euandi.py's vaa*.csv files and the per-model figures drawn from them stay on the
distance score.
"""
import functools
import os

import numpy as np
import pandas as pd

SCORINGS = ("posterior", "distance")
POSTERIOR_FLOOR = 0.25   # half a Likert step on the [-1, 1] stance scale
CLUSTER_SEED = 0         # party_kmeans' seed: the clusters the classifier was trained on
# What one VAA score is, for axis labels and captions.
SCORE_NOUN = {"posterior": "cluster probability", "distance": "agreement"}


def vaa_scoring(positions):
    """The scoring a run on `positions` uses: VAA_SCORING if set, else posterior for the
    cluster basis and distance for the others."""
    requested = os.environ.get("VAA_SCORING", "").strip()
    if requested and requested not in SCORINGS:
        raise SystemExit(f"VAA_SCORING={requested!r}; expected one of {SCORINGS}.")
    if requested == "posterior" and positions != "cluster":
        raise SystemExit(f"VAA_SCORING=posterior needs --positions cluster (got {positions!r}): "
                         "the Gaussians are fitted on the k=4 clusters' member parties.")
    return requested or ("posterior" if positions == "cluster" else "distance")


def scoring_suffix(scoring):
    """"" for the distance score, "_posterior" otherwise -- the tag of the VAA null files."""
    return "_posterior" if scoring == "posterior" else ""


def scoring_words(scoring):
    """One clause for captions: how a VAA stance became a per-group score."""
    if scoring == "posterior":
        return ("each stance read as cluster membership probabilities under one Gaussian per "
                "cluster and statement (member parties' MEP-weighted mean and SD, "
                f"SD floored at {POSTERIOR_FLOOR:g})")
    return "each stance scored by its agreement 1 - |position - stance| / 2 with the group"


@functools.lru_cache(maxsize=None)
def cluster_gaussians():
    """(names, mu, sd): the clusters in display order and (K, T) mean and floored sd per
    statement. Fitted once per process; the means are checked against the cluster
    positions file the distance score and the classifier labels use."""
    # Imported here: vaa_cluster_posterior pulls in the plotting modules that import this.
    from analysis.core.positions import CLUSTER_POSITIONS_PATH, load_party_positions
    from analysis.vaa_cluster_posterior import (
        check_against_positions_file, cluster_gaussians as fit,
    )
    statements = load_party_positions(CLUSTER_POSITIONS_PATH, "cluster")["statement_idx"]
    mu, sd, _, names, _ = fit(CLUSTER_SEED, int(statements.max()) + 1)
    check_against_positions_file(mu, names)
    return tuple(names), mu, np.maximum(sd, POSTERIOR_FLOOR)


def _softmax(ll, axis):
    """Softmax over `axis`, NaN treated as -inf; an all-NaN slice stays NaN."""
    finite = np.where(np.isnan(ll), -np.inf, ll)
    top = finite.max(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        e = np.exp(finite - top)
        p = e / e.sum(axis=axis, keepdims=True)
    return np.where(np.isfinite(top), p, np.nan)


def posterior_cells(S, groups):
    """(N, G, T) P(cluster | stance) for stance vectors S (N, T), clusters in the order of
    `groups`; NaN where the stance is missing."""
    names, mu, sd = cluster_gaussians()
    missing = [group for group in groups if group not in names]
    if missing:
        raise SystemExit(f"No Gaussian for {missing}: the posterior scoring needs the k=4 "
                         f"cluster basis (clusters {list(names)}).")
    S = np.asarray(S, dtype=float)
    T = min(S.shape[1], mu.shape[1])
    z = (S[:, None, :T] - mu[None, :, :T]) / sd[None, :, :T]
    ll = -0.5 * z ** 2 - np.log(sd[None, :, :T])
    ll = np.where(np.isnan(S[:, None, :T]), np.nan, ll)
    p = _softmax(ll, axis=1)
    out = np.full((S.shape[0], len(names), S.shape[1]), np.nan)
    out[:, :, :T] = p
    return out[:, [names.index(group) for group in groups], :]


def level_posteriors(levels, groups):
    """(L, G, T): the posterior of a stance at each of `levels` on every statement -- what
    a closed-form null averages over its stance distribution."""
    _, mu, _ = cluster_gaussians()
    S = np.repeat(np.asarray(levels, dtype=float)[:, None], mu.shape[1], axis=1)
    return posterior_cells(S, groups)


def score_long(frame, cell_keys, scoring, stance="stance", position="normalized_answer"):
    """The score of every row of a long frame -- one row per (stance point, party or group)
    with columns ep_group, statement_idx, `position` and `stance`. `cell_keys` identify one
    stance point (e.g. statement_idx and language); the posterior is normalised over the
    clusters within it, so every point must carry one row per cluster."""
    if scoring == "distance":
        return 1 - (frame[position] - frame[stance]).abs() / 2
    names, mu, sd = cluster_gaussians()
    index = {name: i for i, name in enumerate(names)}
    unknown = sorted(set(frame["ep_group"].dropna()) - set(index))
    if unknown:
        raise SystemExit(f"No Gaussian for {unknown}: the posterior scoring needs the k=4 "
                         f"cluster basis (clusters {list(names)}).")
    ci = frame["ep_group"].map(index).to_numpy(dtype=int)
    j = frame["statement_idx"].to_numpy(dtype=int)
    s = sd[ci, j]
    ll = pd.Series(-0.5 * ((frame[stance].to_numpy(dtype=float) - mu[ci, j]) / s) ** 2
                   - np.log(s), index=frame.index)
    keys = [frame[key] for key in cell_keys]
    per_point = frame[stance].notna().groupby(keys).transform("sum")
    if ((per_point != len(names)) & frame[stance].notna()).any():
        raise SystemExit("posterior scoring: a stance point does not carry exactly one row "
                         f"per cluster ({len(names)}); check the positions frame.")
    e = np.exp(ll - ll.groupby(keys).transform("max"))
    return e / e.groupby(keys).transform("sum")
