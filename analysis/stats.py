"""Pure statistics behind the rank-consistency tables.

Every function here is a plain array transform: no file access, no LaTeX, no argparse,
nothing that knows what a "model" or a "framing" is. They were embedded in the middle of
`rank_consistency_tables.py`, a 2190-line module that also did data loading, table
assembly, LaTeX rendering and the CLI, which made them effectively untestable.

The vectorised shapes are deliberate. Rankings arrive as (..., raters, objects) with the
leading axes free, so the same call computes the observed statistic and all N bootstrap
replicates at once -- the bootstrap runs in chunks of draws, not one draw at a time.
"""
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, rankdata

# Percentile intervals are reported at this level throughout.
CONFIDENCE = 95


def safe_divide(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(numerator, denominator,
                     out=np.full(np.broadcast(numerator, denominator).shape, np.nan),
                     where=denominator != 0)


def kendalls_w(rankings):
    """Tie-corrected coefficient of concordance for rankings (..., raters, objects).

    W = 12 S / (m^2 (n^3 - n) - m T), with T the usual sum of (t^3 - t) over the tie
    groups of every rater. Writing each element's tie-group size as t, that sum is
    sum(t^2) - n per rater, which vectorises over the bootstrap draws."""
    rankings = np.asarray(rankings, float)
    raters, objects = rankings.shape[-2:]
    rank_sums = rankings.sum(axis=-2)
    deviation = ((rank_sums - rank_sums.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    tie_sizes = (rankings[..., :, None] == rankings[..., None, :]).sum(axis=-1)
    ties = (tie_sizes ** 2).sum(axis=(-1, -2)) - raters * objects
    return safe_divide(12.0 * deviation,
                       raters ** 2 * (objects ** 3 - objects) - raters * ties)


def kendall_pvalue(w, raters, objects):
    """Friedman chi-square test of W against no concordance. For completeness only:
    over this many runs it rejects for any W worth reporting."""
    return float(chi2.sf(raters * (objects - 1) * w, objects - 1))


def icc31(ratings):
    """ICC(3,1) -- two-way MIXED effects, CONSISTENCY, single measurement -- for
    ratings (..., targets, raters). The targets are the EP groups.

    Deliberately not ICC(2,1) (absolute agreement): that form has a rater-main-
    effect term in the denominator, so it penalises two raters whose overall
    level or spread differ, not just whether their relative pattern across
    targets agrees. Here the "raters" are measurement methods on genuinely
    different raw scales -- VAA agreement lives in a narrow band near 0.5-0.7,
    classifier probability spans [0, 1] concentrated near 0/1 -- so even after
    reducing both to a 1..6 rank profile, a method built on a compressed scale
    tends to produce a compressed (low between-group variance) rank profile
    purely from that scale, not from disagreeing with the other raters about
    which group is closest. Absolute agreement would count that scale artefact
    as disagreement; consistency does not, because it has no rater-variance
    term to be sensitive to it -- exactly Shrout & Fleiss's (1979) own guidance
    for choosing consistency over agreement when raters' absolute levels are
    not intended to be comparable, only their relative ordering."""
    ratings = np.asarray(ratings, float)
    targets, raters = ratings.shape[-2:]
    grand = ratings.mean(axis=(-2, -1), keepdims=True)
    target_means = ratings.mean(axis=-1, keepdims=True)

    ss_targets = raters * ((target_means - grand) ** 2).sum(axis=(-2, -1))
    rater_means = ratings.mean(axis=-2, keepdims=True)
    ss_raters = targets * ((rater_means - grand) ** 2).sum(axis=(-2, -1))
    ss_error = ((ratings - grand) ** 2).sum(axis=(-2, -1)) - ss_targets - ss_raters

    mean_square_targets = ss_targets / (targets - 1)
    mean_square_error = ss_error / ((targets - 1) * (raters - 1))
    return safe_divide(mean_square_targets - mean_square_error,
                       mean_square_targets + (raters - 1) * mean_square_error)


def top1_consistency(profiles, tolerance=1e-9):
    """(share of raters agreeing on the rank-1 group, index of that group) for mean
    rank profiles (..., raters, objects) -- the lower the mean rank, the closer.

    A rater can tie two groups at the top; picking one with argmin would make the
    share depend on the order the groups happen to sit in, so a tie is split evenly
    between the groups sharing it. The tolerance only absorbs floating-point noise:
    two genuinely different mean ranks differ by at least 1/(2 x runs)."""
    leaders = profiles <= profiles.min(axis=-1, keepdims=True) + tolerance
    counts = (leaders / leaders.sum(axis=-1, keepdims=True)).sum(axis=-2)
    return counts.max(axis=-1) / profiles.shape[-2], counts.argmax(axis=-1)


# Two mean scores that are mathematically equal can still differ by an ULP
# depending on the order their weighted sums were accumulated in -- a matrix
# multiply against a weight row versus a pandas groupby, or one bootstrap chunk
# size versus another. Ranking is a step function, so a 1e-16 difference turns a
# genuine tie (ranks 2.5, 2.5) into a spurious split (3, 2) and silently moves
# every statistic computed downstream of it. Observed live: EP groups GUE/NGL
# and S&D both scored 0.709051724137931 in Slovak under the negated framing,
# and the two paths disagreed on r_xx by 0.003 purely from that. Rounding
# before ranking makes real ties survive as ties; 12 decimals sits ~4 orders of
# magnitude below the smallest meaningful difference between two agreement
# scores in [0, 1], and ~4 above double-precision noise.
RANK_DECIMALS = 12


def ranks_descending(values):
    """Rank along the last axis, 1 = highest, ties averaged and tie-stable."""
    return rankdata(-np.round(values, RANK_DECIMALS), axis=-1)


def ranks_ascending(values):
    """Rank along the last axis, 1 = lowest, ties averaged and tie-stable."""
    return rankdata(np.round(values, RANK_DECIMALS), axis=-1)


def pearson_matrix(vectors):
    """Pairwise Pearson correlation between raters' vectors, (..., raters,
    raters), over the last axis -- the Gram matrix of the centred, unit-norm
    rows. Used both for Spearman rho (on re-ranked data, see spearman_matrix)
    and directly on data that must NOT be re-ranked, such as the concatenated
    per-language rank blocks the language-stratified rho is built from: ranking
    that concatenation as one vector would destroy the per-language grouping
    (see language_pair_rho)."""
    centred = vectors - vectors.mean(axis=-1, keepdims=True)
    norms = np.sqrt((centred ** 2).sum(axis=-1, keepdims=True))
    unit = safe_divide(centred, norms)
    return unit @ np.swapaxes(unit, -1, -2)


def spearman_matrix(profiles):
    """Pairwise Spearman rho between raters, (..., raters, raters), over the
    objects. Spearman is Pearson on the ranks, and the profiles are already mean
    ranks, so only the re-ranking (which collapses their magnitudes back onto
    1..n) is needed before pearson_matrix."""
    return pearson_matrix(ranks_ascending(profiles))


def pair_indices(raters):
    """Upper-triangle (i, j) pairs, the order the rho columns and the CSV use."""
    return [(i, j) for i in range(raters) for j in range(i + 1, raters)]


def as_distributions(profiles):
    """Rater profiles rescaled to sum to one over the objects.

    The classifier profiles are already distributions -- its six party
    probabilities sum to one per statement, so their mean does too -- and this is a
    no-op for them. A VAA agreement profile is not a distribution: it lives in a
    narrow band well above zero, so normalising it gives a near-uniform vector and
    its divergences come out an order of magnitude smaller. Comparable down a
    method, not across methods."""
    return safe_divide(profiles, profiles.sum(axis=-1, keepdims=True))


def jensen_shannon_matrix(distributions):
    """Pairwise Jensen-Shannon divergence in bits, (..., raters, raters).

    JSD(p, q) = H((p + q) / 2) - (H(p) + H(q)) / 2, which is 0 for identical
    distributions and 1 bit for disjoint support. xlogy keeps 0 log 0 at zero."""
    left = distributions[..., :, None, :]
    right = distributions[..., None, :, :]
    mixture = (left + right) / 2
    return (entropy_bits(mixture) - (entropy_bits(left) + entropy_bits(right)) / 2)


def entropy_bits(distributions):
    return -xlogy(distributions, distributions).sum(axis=-1) / np.log(2)


def offdiagonal_mean(matrix):
    """Mean over the distinct pairs of a symmetric (..., raters, raters) matrix."""
    raters = matrix.shape[-1]
    rows, columns = np.triu_indices(raters, k=1)
    return np.nanmean(matrix[..., rows, columns], axis=-1)


def safe_nanargmax(values):
    """np.nanargmax over the last axis, returning index 0 for a slice that is
    entirely NaN instead of raising. That happens when a bootstrap statement
    resample happens to miss every valid statement for one (language, method,
    model) job -- a real possibility once enough models have partial per-
    language coverage. The result is discarded for every such draw regardless
    (divergent/typical are categorical, kept only from the single observed,
    full-coverage pass, where every rater has data and this never triggers)."""
    all_nan = np.all(np.isnan(values), axis=-1, keepdims=True)
    return np.nanargmax(np.where(all_nan, -np.inf, values), axis=-1)


def safe_nanargmin(values):
    all_nan = np.all(np.isnan(values), axis=-1, keepdims=True)
    return np.nanargmin(np.where(all_nan, np.inf, values), axis=-1)


def row_means(matrix):
    """Each rater's mean value against the others, (..., raters); the diagonal is
    zero for a divergence and one for a correlation, so it is excluded."""
    raters = matrix.shape[-1]
    off = ~np.eye(raters, dtype=bool)
    return np.nanmean(np.where(off, matrix, np.nan), axis=-1)


def standardize_raters(profiles):
    """Z-score each rater's profile across the objects (mean 0, unit SD per
    (draw, rater)), profiles (..., raters, objects).

    ICC(3,1)'s consistency form removes a rater's additive offset, but not its
    SCALE: a rater built on a compressed raw scale produces a compressed
    (low-variance) rank profile purely from that compression, and a plain ICC
    counts that as disagreement rather than measurement noise. Verified
    directly: two raters related by a perfect, if rescaling, linear transform
    give icc31 = 0.13, not the ~1.0 a scale-free agreement statistic should
    show for a perfect relationship. Z-scoring first removes that scale
    artefact along with the location one, so only the raters' relative
    ordering drives the statistic -- the same effect Spearman rho already gets
    for free from its own re-ranking step, extended here to ICC."""
    mean = profiles.mean(axis=-1, keepdims=True)
    std = profiles.std(axis=-1, keepdims=True)
    return safe_divide(profiles - mean, std)


def icc_consistency(rank_profiles):
    """ICC(3,1) consistency, computed on each rater's z-scored profile. See
    icc31 for why consistency over absolute agreement, and standardize_raters
    for why the z-score step is also needed -- consistency alone still leaves
    ICC sensitive to raters having different variances, which is exactly the
    situation here (VAA agreement's narrow band versus classifier
    probability's near-0/1 spread)."""
    return icc31(np.swapaxes(standardize_raters(rank_profiles), -1, -2))


def concordance_metrics(rank_profiles, _score_profiles):
    """W, ICC and argmax agreement for mean rank profiles (draws, raters, objects)."""
    share, winner = top1_consistency(rank_profiles)
    return {
        "w": kendalls_w(ranks_ascending(rank_profiles)),
        "icc": icc_consistency(rank_profiles),
        "top1": share,
        "winner": winner,
    }


def pair_metrics(rank_profiles, score_profiles):
    """Every pairwise Spearman rho, plus W, ICC(3,1) consistency (z-scored),
    argmax (top-1) consistency and each rater's between-party SD (raw score
    spread across the EP groups, before ranking) as the matrix table's summary
    figures.

    Between-party SD is the restriction-of-range check: a model whose semantic
    refusals get scored as neutral pulls every group's score toward the same
    middling value, which mechanically compresses this SD and, downstream,
    every consistency coefficient computed on the resulting rank profile -- a
    low ICC/rho next to a low SD is a measurement artefact, not necessarily a
    finding about genuine disagreement between raters."""
    correlations = spearman_matrix(rank_profiles)
    rows, columns = np.triu_indices(rank_profiles.shape[-2], k=1)
    share, _ = top1_consistency(rank_profiles)
    return {
        "rho": correlations[..., rows, columns],
        "w": kendalls_w(ranks_ascending(rank_profiles)),
        "icc": icc_consistency(rank_profiles),
        "top1": share,
        "between_party_sd": score_profiles.std(axis=-1),
    }


def divergence_metrics(rank_profiles, score_profiles):
    """Mean pairwise JSD between the raters' distributions, W over their orderings,
    and which rater sits furthest from / closest to the rest."""
    divergences = jensen_shannon_matrix(as_distributions(score_profiles))
    per_rater = row_means(divergences)
    return {
        "jsd": offdiagonal_mean(divergences),
        "w": kendalls_w(ranks_ascending(rank_profiles)),
        "rater_jsd": per_rater,
        "divergent": safe_nanargmax(per_rater),
        "typical": safe_nanargmin(per_rater),
    }


# Metrics that name a level rather than measure one: reported for the observed data
# only, since a percentile interval over category labels means nothing.
KINDS = {
    "concordance": dict(metrics=concordance_metrics, categorical=["winner"]),
    "pairs": dict(metrics=pair_metrics, categorical=[]),
    "divergence": dict(metrics=divergence_metrics,
                       categorical=["divergent", "typical", "rater_jsd"]),
}


def percentile_interval(values, axis=0):
    tail = (100 - CONFIDENCE) / 2
    return (np.nanpercentile(values, tail, axis=axis),
            np.nanpercentile(values, 100 - tail, axis=axis))


def spearman_brown(half_correlation):
    """Full-length reliability implied by a correlation between two half-tests:
    r_full = 2 r_half / (1 + r_half).

    The split here is by framing -- base against negated. Each framing supplies
    half the runs behind the pooled profile the internal matrix correlates, so
    the raw base-negated correlation is a HALF-test statistic and needs stepping
    up before it can serve as the reliability of the full (both-framings)
    measure that rho_lang is computed from.

    Undefined for r_half <= 0. Spearman-Brown assumes the two halves are
    parallel measures of one construct; a non-positive correlation between them
    is evidence that they are not, so no reliability can be recovered and the
    formula would return values outside [-1, 1] anyway (r_half = -.39 gives
    -1.28). NaN there, so callers surface it as "not interpretable" rather than
    printing a number that looks usable."""
    half = np.asarray(half_correlation, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        full = 2.0 * half / (1.0 + half)
    result = np.where(np.isfinite(half) & (half > 0), full, np.nan)
    return float(result) if np.ndim(half_correlation) == 0 else result
