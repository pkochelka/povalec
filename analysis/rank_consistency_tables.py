#!/usr/bin/env python3
"""LaTeX tables of how consistent the final EP-group ranking is across a factor.

Every method scores the EP groups on its own scale -- VAA agreement lives in a
narrow band inside [0, 1], classifier probability is dominated by the class prior
-- so only the *ordering* is comparable. Each run (model x method x framing x
language x paraphrase) therefore contributes one ranking of the EP groups, rank 1
being the group it placed closest to the model, exactly as in
plot_ep_group_rank_boxplots.

A "rater" is one level of the factor under study: one method, one language, one
prompt variant. Its ranking is its mean rank per EP group over every run at that
level, and the agreement between raters is summarised three ways:

  Kendall's W   tie-corrected coefficient of concordance over the m raters and the
                n EP groups. 1 = every rater produced the same ordering, 0 = the
                orderings are as unrelated as random permutations.
  ICC(3,1)      two-way mixed effects, CONSISTENCY (not absolute agreement),
                single measurement, on each rater's z-scored mean-rank profile.
                Consistency rather than agreement, and z-scored rather than raw,
                because the raters (methods) are not on a comparable raw scale --
                VAA agreement sits in a narrow band, classifier probability spans
                [0, 1] near 0/1 -- and an agreement-form ICC on unstandardised
                profiles would partly measure that scale/compression difference
                as if it were disagreement about which group is closest. Because
                a ranking fixes every rater's marginal distribution, ICC(3,1) here
                is a monotone companion of W rather than independent evidence --
                it is reported because it is the number readers of the
                reliability literature expect to see.
  Top-1 share   share of raters whose rank-1 group is the modal rank-1 group. The
                argmax consistency: how often the headline claim -- "the models sit
                closest to this group" -- survives the factor.

Neither the raters nor the EP groups are sampled: both are the whole population.
The statements are, so the intervals come from a cluster bootstrap over the 30
euandi statements, the same unit vaa_agreement_ci resamples. Every method, row and
table is scored on the same draws, so the tables are mutually paired.

Tables (--table; 'all' is internal, negation, crosslang, variants, bymodel):

  internal   raters are the measurement methods, same rows as "methods" below, but
             instead of one W over all of them this reports every pairwise
             Spearman rho between two methods' EP-group rankings -- where an
             overall drop in concordance actually comes from.
  negation   invariance to negation per method: the correlation between a
             method's base- and negated-framing rankings. A validity property,
             not a reliability -- a near-zero value means the two framings
             elicit different but internally coherent positions. Its
             Spearman-Brown step-up is what "internal" divides by.
  crosslang  raters are the questionnaire languages, one row per (method, model).
             Reports mean pairwise Jensen-Shannon divergence between languages'
             score profiles (a distributional measure W cannot give: do two
             languages put the same *mass* on each group, not just the same
             order) alongside W as an order-only companion.
  variants   raters are the prompt variants, one row per (method, model).
             Paraphrase indices are per-track wordings, so every row fixes a
             method: v3 of the Likert prompt and v3 of the prose prompt are
             unrelated prompts.
  methods    raters are the measurement methods, one row per model plus a pooled
             row. The single-number version of "internal": does the choice of
             instrument change the ranking at all?
  languages  raters are the questionnaire languages, one row per (method, model).
             The single-number version of "crosslang".

Run from the repo root -- writes to data/<dataset>_results/tables/ by default,
next to the results it summarises rather than into analysis/ where the scripts
live (pass --no-output for stdout only, or --output/--csv/--pairs-csv to choose):
    ./venv/Scripts/python.exe analysis/rank_consistency_tables.py --table all
"""

import argparse
import collections
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core import (
    CELL_KEYS,
    LANGS,
    METHODS,
    PARTY_PROB_COLUMN,
    RESPONSES_CSV,
    REASONS_CLASSIFIED_CSV,
    SLUG_TO_LABEL,
    SPEECHES_CLASSIFIED_CSV,
    SPEECHES_SCORED_CSV,
    VARIANT,
    find_csvs,
    judge_stances,
    likert_stances,
    party_sort_key,
    positions_frame,
    slug_labels,
    speech_stances,
)
from analysis.evaluate_euandi import DEFAULT_POSITIONS, POSITION_CHOICES
from analysis.tables import (
    BYMODEL_FACTORS,
    DEFAULT_METHODS,
    DEFAULT_TABLES,
    FACTOR_NOUN,
    MATRIX_FRAMINGS,
    MATRIX_LABEL,
    METHOD_LABEL,
    POOLED,
    VARIANT_CAPTION,
    VARIANT_FACTORS,
    bymodel_cell,
    bymodel_factor_spec,
    bymodel_row_label,
    check_unique_short_names,
    matrix_title,
    pairs_frame,
    reliability_r_xx,
    reliability_spec,
    render_bymodel_latex,
    render_bymodel_markdown,
    render_latex,
    render_markdown,
    render_matrix_latex,
    render_matrix_markdown,
    render_reliability_latex,
    render_reliability_markdown,
    short_model_name,
    table_body,
    tidy_frame,
)
from analysis.stats import (
    CONFIDENCE,
    KINDS,
    concordance_metrics,
    pearson_matrix,
    ranks_descending,
    safe_divide,
)

# The cross-encoder-scored reason texts are the one route analysis.core.METHODS does
# not carry (it scores that track with the LLM judge instead). Same column layout as the
# scored speeches, so the same stance loader reads it.
REASONS_SCORED_CSV = re.compile(rf"^(?!speeches_)(?P<langs>{LANGS}){VARIANT}_scored\.csv$")

QUESTIONNAIRE_PATH = "data/euandi_2024_data/euandi_2024_questionnaire.jsonl"
# The EU&I questionnaire tags each statement with signed loadings on seven axes.
# Six are policy topics; "Left-Right" is a summary ideological axis that runs
# across them (it shares 9 of its 10 statements with Economy alone), so it is not
# a topic and is excluded -- kept as raters, it and Economy would agree almost by
# construction and understate exactly the topic-skew this column is meant to
# detect.
TOPIC_AXES = ["Ukraine", "Ecology", "Immigration", "Values", "Economy", "Europe"]
IDEOLOGY_AXIS = "Left-Right"
# A topic with only a statement or two gives a 6-group rank profile built from
# almost nothing; it would depress the concordance for sample-size reasons and be
# misread as genuine topic-dependence.
MIN_STATEMENTS_PER_TOPIC = 3

MIN_RATERS = 2
BOOTSTRAP_CHUNK = 100


# --------------------------------------------------------------------------- #
# per-statement scores
#
# plot_ep_group_rank_boxplots averages the statements away immediately; the
# bootstrap needs them, so these keep the statement axis and average it later.
# --------------------------------------------------------------------------- #

def vaa_statement_scores(party_df, stances):
    """Agreement per (EP group, language, paraphrase, statement), averaged over the
    parties standing for the group -- so an abstaining party drops out of that
    statement instead of dragging the group's whole mean."""
    if stances.empty:
        return pd.DataFrame()
    merged = party_df.merge(stances, on="statement_idx")
    merged["score"] = 1 - (merged["normalized_answer"] - merged["stance"]).abs() / 2
    return (merged.groupby(["ep_group", "language", "paraphrase", "statement_idx"],
                           as_index=False)["score"].mean())


def classifier_statement_scores(path):
    """Party probability per (EP group, language, paraphrase, statement); the CSV
    has one row per statement, in questionnaire order."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    labels = slug_labels(raw)
    frames = []
    for column in raw.columns:
        match = PARTY_PROB_COLUMN.match(column)
        if not match:
            continue
        probability = pd.to_numeric(raw[column], errors="coerce")
        if probability.isna().all():
            continue
        slug = match.group("slug")
        frames.append(pd.DataFrame({
            "ep_group": labels.get(slug, SLUG_TO_LABEL.get(slug, slug)),
            "language": match.group("language"),
            "paraphrase": int(match.group("paraphrase")),
            "statement_idx": np.arange(len(raw)),
            "score": probability.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vaa_method(pattern, stance_loader):
    def load(model_dir, party_df):
        frames = []
        for framing, path in find_csvs(model_dir, pattern).items():
            scores = vaa_statement_scores(party_df, stance_loader(path, framing))
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


def judge_method(track):
    def load(model_dir, party_df):
        frames = []
        for framing, stances in judge_stances(model_dir, track).items():
            scores = vaa_statement_scores(party_df, stances)
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


def classifier_method(pattern):
    def load(model_dir, _party_df):
        frames = []
        for framing, path in find_csvs(model_dir, pattern).items():
            scores = classifier_statement_scores(path)
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


ALL_METHODS = {
    "vaa-likert": dict(load=vaa_method(RESPONSES_CSV, likert_stances)),
    "vaa-reasons": dict(load=vaa_method(REASONS_SCORED_CSV, speech_stances)),
    "vaa-speeches": dict(load=vaa_method(SPEECHES_SCORED_CSV, speech_stances)),
    "clf-reasons": dict(load=classifier_method(REASONS_CLASSIFIED_CSV)),
    "clf-speeches": dict(load=classifier_method(SPEECHES_CLASSIFIED_CSV)),
    "vaa-likert-judge": dict(load=judge_method("reasons")),
    "vaa-speeches-judge": dict(load=judge_method("speeches")),
}
assert set(METHOD_LABEL) == set(ALL_METHODS)
# The shared registry defines the six methods the box plots draw; this script adds
# vaa-reasons and reads every one of them per statement rather than per run.
assert set(METHODS) <= set(ALL_METHODS)


def load_scores(model_dirs, methods, party_df):
    frames = []
    for model_dir in model_dirs:
        print(f"\nLoading: {model_dir.name}")
        for method in methods:
            found = ALL_METHODS[method]["load"](model_dir, party_df)
            if not found:
                print(f"  [{method}] no input files, skipping.")
                continue
            scores = pd.concat(found, ignore_index=True)
            frames.append(scores.assign(model=model_dir.name, method=method))
            print(f"  [{method}] {len(scores)} scores over "
                  f"{scores['language'].nunique()} languages, "
                  f"{scores['paraphrase'].nunique()} paraphrases, "
                  f"{scores['framing'].nunique()} framings, "
                  f"{scores['statement_idx'].nunique()} statements")
    if not frames:
        raise SystemExit("No scores could be loaded for any model.")
    return pd.concat(frames, ignore_index=True)


MIN_METHOD_COVERAGE = 0.5


def keep_complete_models(scores, methods, allow_partial, min_coverage=MIN_METHOD_COVERAGE):
    """A model missing a method would enter the language and prompt tables with
    fewer runs than the others and silently reweight their mean ranks.

    A method can also exist as a file but be mostly empty -- an in-progress
    classification run, say -- which "does the method exist" alone would not
    catch. Left in, a rater backed by only a handful of runs will have some
    bootstrap statement-resamples land entirely on missing data, poisoning that
    whole draw with NaN (safe_divide's 0/0) and, once enough raters are pooled
    together, NaN-ing every draw's W/ICC/rho for that scope. Compared against
    the median row count other models have for the same method, not an absolute
    threshold, since methods legitimately differ in how many rows they produce
    (e.g. clf-speeches has fewer paraphrases than clf-reasons)."""
    coverage = scores.groupby("model")["method"].nunique()
    missing = set(coverage[coverage < len(methods)].index)

    rows = scores.groupby(["model", "method"]).size()
    expected = rows.groupby("method").transform("median")
    sparse = rows[rows < min_coverage * expected]
    if not sparse.empty:
        detail = ", ".join(f"{model}/{method} ({count}/{int(expected_count)} rows)"
                           for (model, method), count, expected_count
                           in zip(sparse.index, sparse, expected.loc[sparse.index]))
        print(f"\nSparse method coverage (<{min_coverage:.0%} of the median row "
              f"count for that method): {detail}")
    incomplete = sorted(missing | {model for model, _ in sparse.index})
    if not incomplete:
        return scores
    if allow_partial:
        print(f"Models missing or sparse in at least one method, kept "
              f"(--allow-partial-models): {incomplete}")
        return scores
    print(f"Models missing or sparse in at least one method, excluded: {incomplete}")
    kept = scores[~scores["model"].isin(incomplete)]
    if kept.empty:
        raise SystemExit("Every model is missing at least one of the requested methods.")
    return kept


def common_parties(scores):
    """EP groups every method scores. A group only some methods carry would be
    ranked on a shorter scale by the others, which biases every metric."""
    per_method = scores.groupby("method")["ep_group"].apply(lambda column: set(column.unique()))
    common = set.intersection(*per_method) if len(per_method) else set()
    dropped = set().union(*per_method) - common
    if dropped:
        print(f"\nGroups missing from at least one method, excluded: {sorted(dropped)}")
    if len(common) < 2:
        raise SystemExit("Fewer than two EP groups are scored by every method.")
    return sorted(common, key=party_sort_key)


# --------------------------------------------------------------------------- #
# (run x EP group x statement) tensor
# --------------------------------------------------------------------------- #

def score_tensor(scores, parties):
    """(runs x EP groups x statements) scores, plus the run metadata frame.

    Missing (group, statement) pairs stay NaN -- a party that answered "no opinion"
    has no position to agree with -- and are excluded from the statement means
    rather than counted."""
    scores = scores[scores["ep_group"].isin(parties)]
    statements = np.sort(scores["statement_idx"].unique())
    cell_codes, cell_index = pd.MultiIndex.from_frame(scores[CELL_KEYS]).factorize()
    group_codes = pd.Categorical(scores["ep_group"], categories=parties).codes
    statement_codes = np.searchsorted(statements, scores["statement_idx"].to_numpy())

    flat = (cell_codes * len(parties) + group_codes) * len(statements) + statement_codes
    if np.unique(flat).size != flat.size:
        raise SystemExit("Duplicate (run, EP group, statement) scores; check the inputs.")

    tensor = np.full((len(cell_index), len(parties), len(statements)), np.nan)
    tensor.reshape(-1)[flat] = scores["score"].to_numpy(dtype=float)

    cells = cell_index.set_names(CELL_KEYS).to_frame(index=False)
    cells["prompt"] = cells["framing"] + "/v" + cells["paraphrase"].astype(int).astype(str)
    # Odd/even parity over the prompt paraphrases: the two halves of the
    # split-half reliability the internal matrix disattenuates by. Parity rather
    # than a first-half/second-half cut so neither half is systematically the
    # earlier-written wordings.
    cells["half"] = np.where(cells["paraphrase"].astype(int) % 2 == 0, "even", "odd")

    complete = ~np.isnan(tensor).all(axis=2).any(axis=1)
    if not complete.all():
        print(f"Dropped {int((~complete).sum())} runs that do not score every EP group.")
    return tensor[complete], cells[complete].reset_index(drop=True), statements


def statement_means(filled, valid, weights):
    """Mean over statements for weights (draws, statements), as (draws, runs, groups).
    Bootstrap resampling of the statements is exactly a multinomial reweighting."""
    shape = filled.shape
    numerator = filled.reshape(-1, shape[2]) @ weights.T
    denominator = valid.reshape(-1, shape[2]) @ weights.T
    means = safe_divide(numerator, denominator).reshape(shape[0], shape[1], -1)
    return np.ascontiguousarray(means.transpose(2, 0, 1))


def run_ranks(means):
    """Rank 1 = the group the run scored highest, ties averaged, as in add_ranks."""
    return ranks_descending(means)


# --------------------------------------------------------------------------- #
# table rows
# --------------------------------------------------------------------------- #

def level_order(cells, factor, methods, allowed=None):
    """`allowed`, if given, restricts the raters to a subset -- used when methods
    are pooled and only some of the factor's levels are common to every pooled
    method (see common_paraphrases)."""
    if factor == "method":
        levels = [method for method in methods if method in set(cells["method"])]
    elif factor == "paraphrase":
        levels = sorted(cells["paraphrase"].unique())
    else:
        levels = sorted(cells[factor].astype(str).unique())
    return levels if allowed is None else [level for level in levels if level in allowed]


def load_statement_topics(path, statements):
    """{topic: boolean mask over `statements`} -- a PARTITION of the
    questionnaire, one topic per statement.

    The questionnaire's axis loadings overlap (13 of 30 statements load two axes,
    two load three), but overlapping raters would share statements and so agree
    partly by construction. Each statement is therefore assigned to the rarest
    axis it loads, i.e. the most specific one: "The EU should be enlarged to
    include Ukraine" loads Ukraine+Europe and counts as Ukraine, since Europe is
    the broader bucket. Ties break alphabetically for determinism.

    The questionnaire is 1-based and the rest of the pipeline 0-based, so the
    index is shifted here (verified: questionnaire statement_idx 1 carries the
    same text as the party positions' statement_idx 0)."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    loadings = {int(row["statement_idx"]) - 1: [axis for axis in TOPIC_AXES
                                                if row.get(axis, 0)]
                for row in rows}
    sizes = collections.Counter(axis for axes in loadings.values() for axis in axes)
    assigned = {index: min(axes, key=lambda axis: (sizes[axis], axis))
                for index, axes in loadings.items() if axes}

    missing = [index for index in statements if index not in assigned]
    if missing:
        print(f"  {len(missing)} statement(s) carry no topic axis and are excluded "
              f"from the topic column: {missing}")
    masks, dropped = {}, []
    for topic in TOPIC_AXES:
        mask = np.array([assigned.get(index) == topic for index in statements])
        if mask.sum() >= MIN_STATEMENTS_PER_TOPIC:
            masks[topic] = mask
        elif mask.any():
            dropped.append(f"{topic} ({int(mask.sum())})")
    if dropped:
        print(f"  topics below {MIN_STATEMENTS_PER_TOPIC} statements, excluded: "
              f"{', '.join(dropped)}")
    print("  topics: " + ", ".join(f"{topic} ({int(mask.sum())})"
                                   for topic, mask in masks.items()))
    return masks


def topic_rank_arrays(tensor, topic_masks):
    """{topic: (runs, groups) mean ranks using only that topic's statements}.

    Computed once for every run, since which runs a given table row uses is a
    per-job question but the per-topic ranking is not. A (run, group) with no
    valid statement inside the topic -- a party that abstained on all of them --
    comes out NaN rather than being ranked from nothing."""
    filled, valid = np.nan_to_num(tensor), np.isfinite(tensor).astype(float)
    arrays = {}
    for topic, mask in topic_masks.items():
        means = statement_means(filled, valid, mask.astype(float)[None, :])[0]
        arrays[topic] = np.where(np.isfinite(means), run_ranks(means), np.nan)
    return arrays


def topic_profiles(job, topic_ranks):
    """(1, topics, groups) mean rank profiles for the runs this job covers.

    Runs carrying a NaN for any group inside a topic are dropped from that
    topic's mean: their within-run ranking was built over a shorter scale, so
    averaging it in would bias the profile rather than merely add noise."""
    profiles = []
    for ranks in topic_ranks.values():
        selected = ranks[job["columns"]]
        usable = selected[np.isfinite(selected).all(axis=-1)]
        profiles.append(usable.mean(axis=0) if len(usable)
                        else np.full(selected.shape[-1], np.nan))
    return np.stack(profiles)[None]


def build_topic_jobs(cells, spec, methods):
    """One job per model scope, every run in scope, raters = topics.

    Topics slice the STATEMENT axis, not the run axis, so unlike every other
    factor each rater draws on all of the scope's runs and they differ only in
    which statements feed them. That makes build_jobs's rater machinery
    inapplicable, but the row scoping is identical, so row_scopes is reused."""
    topics = list(spec["topics"])
    jobs = []
    for scope, mask in row_scopes(cells, spec, methods, "full"):
        columns = np.flatnonzero(mask)
        if columns.size and len(topics) >= MIN_RATERS:
            jobs.append({"table": spec["name"], "kind": "concordance", "scope": scope,
                         "levels": topics, "runs": [int(columns.size)] * len(topics),
                         "columns": columns})
    return jobs


def common_paraphrases(cells, methods):
    """Paraphrase indices present for every one of `methods`.

    Pooling methods into one job (as the bymodel table's "Prompt" column does)
    is only a fair concordance computation if every rater draws on the same
    method composition. The prose track has 5 paraphrases (v0-v4) against the
    Likert/reasons track's 8 (v0-v7): without this restriction, rater "v0" would
    average over all 4 methods while rater "v5" averaged over only 2, silently
    comparing different things under the same "paraphrase" label.

    Using all 8/5 would therefore mean abandoning method pooling and averaging
    per-method coefficients instead -- which was measured and rejected. The
    paraphrase-count effect alone is negligible: computing each method's
    concordance over its own paraphrases, restricting to v0-v4 versus using all
    of them moves ICC/W/argmax by a mean of only .008/.010/.012. Switching to
    per-method averaging, by contrast, drags pooled argmax consistency from .98
    to .78 -- that swing is method disagreement (the finding the internal table
    already reports) leaking into a column that is supposed to isolate prompt
    sensitivity. Restricting costs ~.01 and keeps the column measuring one
    thing."""
    per_method = [set(cells.loc[cells["method"] == method, "paraphrase"].unique())
                 for method in methods]
    return set.intersection(*per_method) if per_method else set()


def scope_levels(cells, dimension, poolable, methods):
    """The values a scope dimension takes, POOLED first where it is allowed."""
    if dimension == "method":
        levels = [method for method in methods if method in set(cells["method"])]
    else:
        levels = sorted(cells[dimension].astype(str).unique())
    return ([POOLED] if poolable else []) + [str(level) for level in levels]


def row_scopes(cells, spec, methods, detail):
    """[(scope tuple, run mask)] over the table's scope dimensions, coarse first.

    A scope is one value per dimension, POOLED meaning "do not restrict". Ordering
    the product with the first dimension outermost puts the fully pooled row first,
    then the rows restricted to one method, then a block per model."""
    axes = [scope_levels(cells, dimension, poolable, methods)
            for dimension, poolable in spec["dimensions"]]
    columns = [cells[dimension].astype(str).to_numpy() for dimension, _ in spec["dimensions"]]
    scopes = []
    for scope in itertools.product(*axes):
        if detail == "summary" and sum(value != POOLED for value in scope) > 1:
            continue
        mask = np.ones(len(cells), dtype=bool)
        for column, value in zip(columns, scope):
            if value != POOLED:
                mask &= column == value
        if mask.any():
            scopes.append((scope, mask))
    return scopes


def build_jobs(cells, spec, methods, detail, allowed_levels=None):
    """One job per table row: the raters it averages over, the runs they cover and
    the (raters x runs) matrix that turns per-run ranks into mean rank profiles.

    Only the runs a row actually uses are kept -- a row restricted to one model and
    one method touches a few hundred of the several thousand runs, and carrying the
    zeros through the bootstrap would dominate the cost. `allowed_levels` restricts
    which raters are built at all (see common_paraphrases)."""
    factor = spec["factor"]
    column = cells[factor].astype(str).to_numpy()
    jobs = []
    for scope, mask in row_scopes(cells, spec, methods, detail):
        raters = []
        for level in level_order(cells[mask], factor, methods, allowed_levels):
            selection = np.flatnonzero(mask & (column == str(level)))
            if selection.size:
                raters.append((level, selection))
        if len(raters) < MIN_RATERS:
            print(f"  [{' / '.join(scope)}] fewer than {MIN_RATERS} "
                  f"{FACTOR_NOUN[factor]}s, skipping row.")
            continue
        runs = [len(selection) for _, selection in raters]
        bounds = np.cumsum([0, *runs])
        aggregation = np.zeros((len(raters), bounds[-1]))
        for index, count in enumerate(runs):
            aggregation[index, bounds[index]:bounds[index + 1]] = 1.0 / count
        jobs.append({
            "table": spec["name"],
            "kind": spec["kind"],
            "scope": scope,
            "levels": [level for level, _ in raters],
            "runs": runs,
            "columns": np.concatenate([selection for _, selection in raters]),
            "aggregation": aggregation,
        })
    return jobs


def profiles_for(job, ranks):
    """Mean rank per (rater, EP group) for ranks (draws, runs, groups)."""
    draws, _, groups = ranks.shape
    selected = ranks[:, job["columns"], :].transpose(1, 0, 2)
    flat = np.ascontiguousarray(selected.reshape(len(job["columns"]), -1))
    stacked = job["aggregation"] @ flat
    return stacked.reshape(-1, draws, groups).transpose(1, 0, 2)


def build_block_weights(cells, job, block_column):
    """(blocks, (raters*blocks, columns) weights): like job["aggregation"], but
    averaging each rater's runs within each `block_column` value separately
    instead of collapsing that axis away. Two uses, both attached to a job
    after build_jobs: block_column="language" for the internal table's
    language-stratified rho, block_column="paraphrase" for a method's own
    base-vs-negated negation invariance (see block_pair_rho)."""
    sub = cells.iloc[job["columns"]]
    blocks = sorted(sub[block_column].unique())
    block_codes = pd.Categorical(sub[block_column], categories=blocks).codes
    bounds = np.cumsum([0, *job["runs"]])
    rater_codes = np.zeros(len(sub), dtype=int)
    for rater_index in range(len(job["levels"])):
        rater_codes[bounds[rater_index]:bounds[rater_index + 1]] = rater_index
    combined = rater_codes * len(blocks) + block_codes
    weights = np.zeros((len(job["levels"]) * len(blocks), len(sub)))
    weights[combined, np.arange(len(sub))] = 1.0
    weights = safe_divide(weights, weights.sum(axis=1, keepdims=True))
    return blocks, weights


def block_profiles_for(job, values, weights_key, blocks_key):
    """Mean value per (rater, block, EP group) for values (draws, runs, groups)
    -- profiles_for's block-preserving counterpart, used only where
    job[weights_key] has been attached by build_block_weights."""
    draws, _, groups = values.shape
    selected = values[:, job["columns"], :].transpose(1, 0, 2)
    flat = np.ascontiguousarray(selected.reshape(len(job["columns"]), -1))
    stacked = job[weights_key] @ flat
    raters, blocks = len(job["levels"]), len(job[blocks_key])
    return stacked.reshape(raters, blocks, draws, groups).transpose(2, 0, 1, 3)


def block_pair_rho(job, means, weights_key, blocks_key):
    """Pairwise correlation between raters' EP-group rankings computed
    block-by-block and pooled, rather than after averaging every block into one
    mean rank first (which pair_metrics's plain rho does).

    With only 6 EP groups, a rank-based rho is unstable -- one group swapping
    rank moves rho by a large step, because there is so little to average over.
    Keeping the blocks (languages, or paraphrases) as replicates widens that:
    the groups are ranked WITHIN each block (so a rater's absolute level in one
    block cannot masquerade as a group-ordering difference), then the length-6
    rank blocks are concatenated per rater and correlated as Pearson on that
    already-ranked data -- re-ranking the concatenation as one long vector
    would mix the blocks and destroy exactly the structure this isolates."""
    block_means = block_profiles_for(job, means, weights_key, blocks_key)
    block_ranks = ranks_descending(block_means)
    draws, raters, blocks, groups = block_ranks.shape
    flattened = block_ranks.reshape(draws, raters, blocks * groups)
    correlations = pearson_matrix(flattened)
    rows, columns = np.triu_indices(raters, k=1)
    return correlations[..., rows, columns]


def augment_with_block_rho(job, values, means):
    """Adds "rho_lang" -- the language-blocked pairwise correlation -- to a
    pairs-kind job that build_jobs has attached language weights to; a no-op
    for every other job.

    Both the internal matrix and the negation table block by language, so
    their numbers are on the same footing: the disattenuated ratio divides one
    into the other (rho_lang / sqrt(r_xx r_yy)), which is only meaningful if
    both are computed over the same replicate structure and the same number of
    points."""
    if "language_weights" not in job:
        return values
    return {**values,
            "rho_lang": block_pair_rho(job, means, "language_weights", "languages")}


def observed_metrics(job, rank_profiles, score_profiles):
    """Every metric the job's kind produces, categorical labels included."""
    return KINDS[job["kind"]]["metrics"](rank_profiles, score_profiles)


def job_metrics(job, rank_profiles, score_profiles):
    """The bootstrap-only subset: categorical labels (an argmax winner, which
    language is most divergent) have no percentile interval worth computing."""
    kind = KINDS[job["kind"]]
    values = kind["metrics"](rank_profiles, score_profiles)
    return {key: value for key, value in values.items() if key not in kind["categorical"]}


CATEGORICAL_KEYS = {"winner", "divergent", "typical"}


def squeeze_observed(values):
    """Drop the size-1 draws axis a single-draw call to a metrics function still
    carries, leaving scalars for categorical/0-d metrics and plain arrays (one
    entry per pair or per rater) for the rest."""
    squeezed = {}
    for key, value in values.items():
        array = np.asarray(value)[0]
        squeezed[key] = int(array) if key in CATEGORICAL_KEYS else (
            float(array) if array.ndim == 0 else array)
    return squeezed


def bootstrap(jobs, tensor, statements, draws, seed, chunk=BOOTSTRAP_CHUNK):
    """Percentile intervals for every job, all scored on the same statement draws."""
    replicates = [{} for _ in jobs]
    if not draws:
        return [{} for _ in jobs]
    rng = np.random.default_rng(seed)
    valid = np.isfinite(tensor).astype(float)
    filled = np.nan_to_num(tensor)
    uniform = np.full(len(statements), 1.0 / len(statements))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        weights = rng.multinomial(len(statements), uniform, size=size).astype(float)
        means = statement_means(filled, valid, weights)
        ranks = run_ranks(means)
        for job, store in zip(jobs, replicates):
            values = job_metrics(job, profiles_for(job, ranks), profiles_for(job, means))
            values = augment_with_block_rho(job, values, means)
            for key, value in values.items():
                store.setdefault(key, []).append(value)
        print(f"  bootstrap {min(start + size, draws)}/{draws}", end="\r")
    print(" " * 40, end="\r")
    return [{key: np.concatenate(value) for key, value in store.items()}
            for store in replicates]


# --------------------------------------------------------------------------- #

def table_specs(args, methods):
    method_names = "; ".join(METHOD_LABEL[method] for method in methods)
    shared = (
        "A run is one (model $\\times$ method $\\times$ framing $\\times$ language "
        "$\\times$ paraphrase) tuple and ranks the EP groups $1\\ldots n$ by how "
        "close it places them to the model; a rater's ranking is its mean rank per "
        "group over its runs. $m$ is the number of raters. Each row restricts the "
        f"runs to one scope, `{POOLED}' meaning the column is not restricted. "
        f"Brackets are {CONFIDENCE}\\% percentile intervals from a cluster "
        f"bootstrap over the euandi statements ({args.bootstrap} draws, shared "
        "across all rows and tables)."
    )
    concordance_note = (
        " Under independent random rankings $W$ has expectation $1/m$, so rows "
        "with few raters start from a higher floor."
    )
    return {
        "methods": dict(
            name="methods", factor="method", dimensions=[("model", True)],
            kind="concordance", label="tab:rank-consistency-methods",
            caption=("Consistency of the EP-group ranking across the measurement "
                     f"methods ({method_names}). Top-1 share is the fraction of "
                     "raters whose closest group is the modal one. " + shared
                     + concordance_note)),
        "languages": dict(
            name="languages", factor="language",
            dimensions=[("model", True), ("method", True)],
            kind="concordance", label="tab:rank-consistency-languages",
            caption=("Consistency of the EP-group ranking across the questionnaire "
                     "languages, per model and per method. Top-1 share is the "
                     "fraction of raters whose closest group is the modal one. "
                     + shared + concordance_note)),
        "variants": dict(
            # Pooling methods here would confound the factor with the track: the
            # prose prompt has five paraphrases against the Likert prompt's
            # eight, so a pooled rater "base/v5" would cover fewer methods than
            # "base/v0". Every row therefore fixes a method.
            name="variants", factor=args.variant_factor,
            dimensions=[("model", True), ("method", False)],
            kind="concordance", label="tab:rank-consistency-variants",
            caption=("Consistency of the EP-group ranking across prompt variants "
                     f"({VARIANT_CAPTION[args.variant_factor]}), per model and per "
                     "method. Every row fixes a method: paraphrase indices are "
                     "per-track wordings, so v3 of the Likert prompt and v3 of the "
                     "prose prompt are unrelated prompts and the prose "
                     "track has five paraphrases against the Likert track's eight. "
                     "Top-1 share is the fraction of raters whose closest group is "
                     "the modal one. " + shared + concordance_note)),
        "internal": dict(
            # Raters are the methods (same scoping as "methods"), but instead of
            # one W over all of them this reports every pairwise Spearman rho, so
            # a drop in overall concordance can be pinned on the pair driving it.
            # Rendered as one small matrix per model (see render_matrix_latex),
            # not as rows of a single wide table.
            name="internal", factor="method",
            dimensions=[("framing", False), ("model", True)],
            kind="pairs", label="tab:rank-consistency-internal",
            caption=("Spearman correlation between each pair of measurement "
                     "methods' EP-group rankings. Within every language the 6 "
                     "groups are ranked $1\\ldots6$, and the 21 blocks are "
                     "stacked into one 126-value vector per method; the cell "
                     "is the correlation between two such vectors. 1 means the "
                     "two methods order the groups identically in every "
                     "language, 0 that they are unrelated, negative that they "
                     "order them oppositely. The \\textbf{upper triangle is "
                     "the base framing, the lower the negated} one, each "
                     "computed from its own runs -- the two are not pooled, "
                     "because under negation some methods produce close to the "
                     "reverse ranking (Table~\\ref{tab:negation-invariance}). "
                     f"`*' marks a correlation whose {CONFIDENCE}\\% bootstrap "
                     "CI over the euandi statements excludes zero.")),
        "negation": dict(
            # Raters = framing (base, negated), one method at a time; never
            # pooled, since invariance is inherently per-method. Built
            # automatically whenever "internal" is requested (see main()), so
            # its values are available for the disattenuation there even if
            # this table itself was not asked for.
            #
            # Reported RAW here, deliberately. This is a validity property --
            # does the method give the same ordering when the statement is
            # negated -- not an error-variance one, so it is not itself a
            # reliability and is not labelled as one. main() applies
            # Spearman-Brown separately to turn the same split into the
            # full-length reliability the internal matrix divides by.
            name="negation", label="tab:negation-invariance",
            caption=("\\textbf{Invariance to negation of each measurement "
                     "method} (superscript is the between-party SD of that "
                     "method's scores across the EP groups).")),
        "crosslang": dict(
            # method-major, method never pooled: JSD compares raw score PROFILES,
            # not ranks, and unlike a rank (always 1..n, so pooling raters across
            # methods is scale-free) the raw scores are not on a common scale --
            # VAA agreement sits in a narrow 0.6-0.75 band, classifier probability
            # spans [0, 1] concentrated near 0/1. Averaging those together before
            # normalising to a distribution would blend the scales into a
            # meaningless profile, so every row fixes one method, same reasoning
            # as the "variants" table.
            name="crosslang", factor="language",
            dimensions=[("method", False), ("model", True)],
            kind="divergence", label="tab:rank-consistency-crosslang",
            caption=("Cross-language agreement of the EP-group ranking, per "
                     f"method ({method_names}) and per model; every row fixes a "
                     "method, since pooling raw scores across methods on "
                     "different scales before normalising would not be "
                     "meaningful. Each language's score profile over the EP "
                     "groups is rescaled to sum to one and compared by "
                     "Jensen-Shannon divergence (bits; 0 = identical "
                     "distributions, 1 = disjoint support). Kendall's $W$ over "
                     "the same languages is included as an order-only "
                     "companion. `Most divergent' / `most typical' name the "
                     "language with the highest / lowest mean JSD to the "
                     "others, with that value. " + shared)),
        "bymodel": dict(
            # Not a single-factor spec like the others: built specially in
            # main() as three "language"/"paraphrase"/"framing" job lists
            # sharing the same (pooled + one row per model) row scoping, then
            # rendered as one grid with a column per factor instead of a
            # column per metric. Methods are always pooled -- safe here since
            # concordance metrics are rank-based, unlike crosslang's JSD.
            name="bymodel", label="tab:rank-consistency-bymodel",
            caption=("Per-model consistency of the EP-group ranking across "
                     "four ``annotator'' sets, pooled over all four "
                     "measurement methods. `Lang' treats the 21 questionnaire "
                     "languages as annotators; `Prompt' the five prompt "
                     "variants common to every method; `Negation' the two "
                     "framings, base vs.\\ negated; `Topic' the policy topics "
                     "of the EU\\&I questionnaire, each statement assigned to "
                     "the most specific axis it loads so the topics partition "
                     "the 30 statements (the Left-Right axis is excluded as an "
                     "ideological summary running across topics rather than a "
                     "topic, and topics under three statements are dropped). "
                     "Unlike the other three, `Topic' varies which STATEMENTS "
                     "the ranking is built from rather than which runs, so it "
                     "asks whether the placement survives restricting the "
                     "questionnaire to one policy area. Each cell reports "
                     "ICC(3,1) / Kendall's $W$ / argmax consistency (the "
                     "fraction of annotators agreeing on the model's closest "
                     "EP group), computed on the mean rank per group over the "
                     "model's runs for that annotator set; all four are "
                     "fractions on $[0, 1]$ with the leading zero omitted.")),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--table", default="all",
                        help="Comma-separated subset of: internal, negation, crosslang, "
                             "variants, bymodel, methods, languages; or 'all' (internal, "
                             "negation, crosslang, variants, bymodel).")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help=f"Comma-separated subset of: {', '.join(ALL_METHODS)}")
    parser.add_argument("--model", default=None,
                        help="Comma-separated model dirs; default uses every model.")
    parser.add_argument("--detail", default="full", choices=["full", "summary"],
                        help="full: a row for every model x method scope as well as "
                             "the pooled ones. summary: only rows that restrict at "
                             "most one scope dimension.")
    parser.add_argument("--variant-factor", default="prompt", choices=VARIANT_FACTORS,
                        help="What counts as a prompt variant in the third table: "
                             "prompt (framing x paraphrase), paraphrase, or framing.")
    parser.add_argument("--questionnaire", default=QUESTIONNAIRE_PATH,
                        help="Statement topic loadings, for the bymodel table's Topic "
                             "column (see TOPIC_AXES).")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Whose euandi answers stand for an EP group, as in "
                             "evaluate_euandi.")
    parser.add_argument("--no-collapse-ecr-id", action="store_true",
                        help="Keep ECR and ID apart. The existing VAA/classifier outputs "
                             "are collapsed, so this only fits a fresh, uncollapsed run.")
    parser.add_argument("--allow-partial-models", action="store_true",
                        help="Keep models missing a requested method, or whose method "
                             "coverage is too sparse (see --min-method-coverage).")
    parser.add_argument("--min-method-coverage", default=MIN_METHOD_COVERAGE, type=float,
                        help="A model/method combination with fewer than this fraction "
                             "of the median row count other models have for that method "
                             "is treated as missing (an in-progress classification run, "
                             "say). Default 0.5.")
    parser.add_argument("--bootstrap", default=2000, type=int,
                        help="Bootstrap draws over statements; 0 reports point estimates.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pvalues", action="store_true",
                        help="Add the Friedman chi-square p-value for W.")
    parser.add_argument("--format", default="latex", choices=["latex", "markdown"])
    parser.add_argument("--output", default=None,
                        help="Where to write the tables. Default: "
                             "data/<dataset>_results/tables/rank_consistency.tex "
                             "(analysis/ holds the scripts, not their output).")
    parser.add_argument("--csv", default=None,
                        help="Where to write the long CSV. Default: "
                             "data/<dataset>_results/tables/rank_consistency.csv")
    parser.add_argument("--pairs-csv", default=None,
                        help="Where to write the internal table's full pairwise rho, as a "
                             "long CSV (the table itself only shows a point estimate and a "
                             "marker). Default: "
                             "data/<dataset>_results/tables/rank_consistency_pairs.csv")
    parser.add_argument("--no-output", action="store_true",
                        help="Print to stdout only; do not write any file.")
    return parser.parse_args()


NON_MODEL_DIRS = {"plots", "tables"}


def resolve_model_dirs(results_dir, selected):
    model_dirs = sorted(path for path in results_dir.iterdir()
                        if path.is_dir() and path.name not in NON_MODEL_DIRS)
    if selected:
        wanted = {name.strip() for name in selected.split(",")}
        model_dirs = [path for path in model_dirs if path.name in wanted]
    if not model_dirs:
        raise SystemExit("No model directories found.")
    return model_dirs


@dataclass
class JobSet:
    """Every row the run will compute, plus the index needed to find them again.

    `jobs` is one flat list because the bootstrap resamples all rows on shared draws;
    the slices say which stretch of it each table owns. Topic rows sit outside it --
    they are never bootstrapped -- which is why they are a separate field rather than
    another slice.

    This was 83 lines inline in `main`, building five collections at once with the
    reader expected to keep all five in their head."""

    jobs: list
    table_slices: dict
    bymodel_slices: dict
    reliability_index: dict
    internal_index: dict
    topic_jobs: list
    topic_observed: list


def build_all_jobs(cells, specs, wanted, methods, tensor, statements, args):
    """Assemble the rows for every requested table, in one flat bootstrap-able list."""
    topic_jobs, topic_observed = [], []
    jobs, table_slices, bymodel_slices = [], {}, {}
    reliability_index, internal_index = {}, {}
    if "internal" in wanted or "negation" in wanted:
        # internal and reliability are built together whenever either is
        # requested, even if the user only asked for one: the internal
        # table's disattenuated ratio needs reliability's r_xx, and the
        # reliability table's between-party SD (see pair_metrics) comes from
        # internal's own per-method score profile. Only rendered if the name
        # was actually requested -- see the "continue" branches below.
        print(f"\n[internal] raters = methods, rows = framing x model scopes "
              f"({args.detail})")
        internal_jobs = build_jobs(cells, specs["internal"], methods, args.detail)
        if not internal_jobs:
            raise SystemExit("No usable rows for table 'internal'.")
        for job in internal_jobs:
            job["languages"], job["language_weights"] = build_block_weights(
                cells, job, "language")
        table_slices["internal"] = slice(len(jobs), len(jobs) + len(internal_jobs))
        for offset, job in enumerate(internal_jobs):
            internal_index[job["scope"]] = len(jobs) + offset
        jobs += internal_jobs

        print("\n[negation] raters = framings (base vs negated), "
              "rows = method x model scopes")
        reliability_jobs = build_jobs(cells, reliability_spec(), methods, "full")
        if not reliability_jobs:
            raise SystemExit("No usable rows for the reliability computation.")
        for job in reliability_jobs:
            # Blocked by language, matching the internal matrix's rho_lang, so
            # r_xx and the cross-method rho it is divided into are computed
            # over the same replicate structure (21 languages x 6 groups).
            job["languages"], job["language_weights"] = build_block_weights(
                cells, job, "language")
        table_slices["negation"] = slice(len(jobs), len(jobs) + len(reliability_jobs))
        for offset, job in enumerate(reliability_jobs):
            reliability_index[job["scope"]] = len(jobs) + offset
        jobs += reliability_jobs

    for name in wanted:
        if name in ("internal", "negation"):
            continue  # already built above
        if name == "bymodel":
            # One job list per annotator-set factor, not one overall: see
            # bymodel_factor_spec. All of them share the row scoping (pooled +
            # one row per model), so they line up index-for-index at render
            # time without needing to be looked up by scope.
            for factor, label in BYMODEL_FACTORS:
                print(f"\n[bymodel:{factor}] raters = {FACTOR_NOUN[factor]}s "
                      f"({label}), rows = model scopes")
                if factor == "topic":
                    # Kept out of `jobs`: topic raters slice statements rather
                    # than runs, so the shared statement bootstrap and
                    # profiles_for do not apply to them (see topic_profiles).
                    topic_masks = load_statement_topics(args.questionnaire, statements)
                    if len(topic_masks) < MIN_RATERS:
                        raise SystemExit("Fewer than two usable topics.")
                    topic_ranks = topic_rank_arrays(tensor, topic_masks)
                    spec_topic = {**bymodel_factor_spec("topic"), "topics": topic_masks}
                    topic_jobs = build_topic_jobs(cells, spec_topic, methods)
                    topic_observed = [squeeze_observed(concordance_metrics(
                        topic_profiles(job, topic_ranks), None)) for job in topic_jobs]
                    continue
                allowed = None
                if factor == "paraphrase":
                    allowed = common_paraphrases(cells, methods)
                    print(f"  restricting to paraphrase indices common to every "
                          f"method: {sorted(int(index) for index in allowed)}")
                factor_jobs = build_jobs(cells, bymodel_factor_spec(factor), methods, "full",
                                        allowed_levels=allowed)
                if not factor_jobs:
                    raise SystemExit(f"No usable rows for table 'bymodel' factor '{factor}'.")
                bymodel_slices[factor] = slice(len(jobs), len(jobs) + len(factor_jobs))
                jobs += factor_jobs
            continue
        scopes = " x ".join(dimension for dimension, _ in specs[name]["dimensions"])
        print(f"\n[{name}] raters = {FACTOR_NOUN[specs[name]['factor']]}s, "
              f"rows = {scopes} scopes ({args.detail})")
        table_jobs = build_jobs(cells, specs[name], methods, args.detail)
        if not table_jobs:
            raise SystemExit(f"No usable rows for table '{name}'.")
        table_slices[name] = slice(len(jobs), len(jobs) + len(table_jobs))
        jobs += table_jobs
    return JobSet(jobs, table_slices, bymodel_slices, reliability_index, internal_index,
                  topic_jobs, topic_observed)


@dataclass
class OutputPaths:
    """Where the three artefacts go, resolved once.

    `main` used to write the defaults back into the parsed `args` -- setting
    `args.output = args.csv = args.pairs_csv = None` for --no-output, and `or`-defaults
    otherwise -- so the Namespace meant something different after the call than the CLI
    said. Resolving defaults belongs to a config object, not to the parsed arguments."""

    latex: str | None
    tidy_csv: str | None
    pairs_csv: str | None

    @classmethod
    def resolve(cls, args, tables_dir):
        if args.no_output:
            return cls(None, None, None)
        return cls(
            latex=args.output or str(tables_dir / "rank_consistency.tex"),
            tidy_csv=args.csv or str(tables_dir / "rank_consistency.csv"),
            pairs_csv=args.pairs_csv or str(tables_dir / "rank_consistency_pairs.csv"),
        )


def main():
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Available: {list(ALL_METHODS)}")

    specs = table_specs(args, methods)
    wanted = DEFAULT_TABLES if args.table == "all" else [t.strip() for t in args.table.split(",")]
    unknown = [table for table in wanted if table not in specs]
    if unknown:
        raise SystemExit(f"Unknown table(s): {unknown}. Available: {list(specs)}")

    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir} (run from the repo root)")

    outputs = OutputPaths.resolve(args, results_dir / "tables")

    party_df = positions_frame(args.positions, not args.no_collapse_ecr_id)
    scores = load_scores(resolve_model_dirs(results_dir, args.model), methods, party_df)
    scores = keep_complete_models(scores, methods, args.allow_partial_models,
                                 args.min_method_coverage)
    parties = common_parties(scores)
    tensor, cells, statements = score_tensor(scores, parties)
    print(f"\n{len(cells)} runs, {len(parties)} EP groups ({', '.join(parties)}), "
          f"{len(statements)} statements.")
    if "bymodel" in wanted:
        check_unique_short_names(cells["model"].unique())

    job_set = build_all_jobs(cells, specs, wanted, methods, tensor, statements, args)
    jobs = job_set.jobs
    table_slices, bymodel_slices = job_set.table_slices, job_set.bymodel_slices
    reliability_index, internal_index = job_set.reliability_index, job_set.internal_index
    topic_jobs, topic_observed = job_set.topic_jobs, job_set.topic_observed
    observed_means = statement_means(np.nan_to_num(tensor), np.isfinite(tensor).astype(float),
                                     np.ones((1, len(statements))))
    observed_ranks = run_ranks(observed_means)
    observed = [squeeze_observed(augment_with_block_rho(
                    job, observed_metrics(job, profiles_for(job, observed_ranks),
                                         profiles_for(job, observed_means)),
                    observed_means))
                for job in jobs]
    print(f"\nBootstrapping {args.bootstrap} statement draws over {len(jobs)} rows...")
    replicates = bootstrap(jobs, tensor, statements, args.bootstrap, args.seed)

    provenance = [
        "Generated by analysis/rank_consistency_tables.py -- do not edit by hand.",
        f"methods={','.join(methods)}  positions={args.positions}  "
        f"models={','.join(sorted(cells['model'].unique()))}",
        f"{len(cells)} runs, {len(parties)} EP groups, {len(statements)} statements, "
        f"bootstrap={args.bootstrap} seed={args.seed}",
    ]

    # Between-party SD per (method, model scope) for the negation table's
    # superscript -- the restriction-of-range check. Taken from the negation
    # job's own score profiles, whose raters are the two framings, so the two
    # values are averaged into one figure for the method.
    sd_lookup = {}
    for (method, model_scope), position in reliability_index.items():
        values = observed[position]
        if "between_party_sd" in values:
            sd_lookup[(method, model_scope)] = float(np.nanmean(values["between_party_sd"]))
    rendered, tidy, pairs = [], [], []
    for name in wanted:
        if name == "bymodel":
            spec = specs["bymodel"]
            print(f"\n[{name}]")
            # Topic jobs live outside `jobs` (they were never bootstrapped), so
            # every factor is fetched through one accessor that knows both homes.
            def factor_slice(factor):
                if factor == "topic":
                    return topic_jobs, topic_observed, [{}] * len(topic_jobs)
                span = bymodel_slices[factor]
                return jobs[span], observed[span], replicates[span]

            by_factor = [factor_slice(factor) for factor, _ in BYMODEL_FACTORS]
            jobs_by_factor = [entry[0] for entry in by_factor]
            observed_by_factor = [entry[1] for entry in by_factor]
            # The grid is pivoted by position, so a factor whose scope list
            # differs (a row dropped for too few raters, say) would silently
            # shift every cell in its column onto the wrong model.
            scopes_per_factor = [[job["scope"] for job in factor_jobs]
                                 for factor_jobs in jobs_by_factor]
            if len({tuple(scopes) for scopes in scopes_per_factor}) != 1:
                raise SystemExit(
                    "bymodel factors cover different model scopes: "
                    + "; ".join(f"{label}={len(scopes)}"
                               for (_, label), scopes in zip(BYMODEL_FACTORS,
                                                             scopes_per_factor)))
            rendered.append(render_bymodel_latex(spec, jobs_by_factor, observed_by_factor,
                                                 provenance)
                            if args.format == "latex"
                            else render_bymodel_markdown(spec, jobs_by_factor,
                                                         observed_by_factor))
            for index, job in enumerate(jobs_by_factor[0]):
                pieces = []
                for column, (_, label) in enumerate(BYMODEL_FACTORS):
                    pieces.append(f"{label}={bymodel_cell(observed_by_factor[column][index])}")
                print(f"  {bymodel_row_label(job):16s} " + "  ".join(pieces))
            for (factor, _), (factor_jobs, factor_observed, factor_replicates) in zip(
                    BYMODEL_FACTORS, by_factor):
                tidy.append(tidy_frame(bymodel_factor_spec(factor), factor_jobs,
                                       factor_observed, factor_replicates, parties))
            continue
        if name == "negation":
            spec = {**reliability_spec(), **specs["negation"]}
            print(f"\n[{name}]")
            span = table_slices["negation"]
            rendered.append(render_reliability_latex(spec, methods, jobs[span], observed[span],
                                                      provenance, sd_lookup)
                            if args.format == "latex"
                            else render_reliability_markdown(methods, jobs[span], observed[span],
                                                              sd_lookup))
            for job, values in zip(jobs[span], observed[span]):
                method, model_scope = job["scope"]
                label = "All models" if model_scope == POOLED else short_model_name(model_scope)
                sd = sd_lookup.get((method, model_scope))
                sd_text = f"{sd:.3f}" if sd is not None else "n/a"
                print(f"  {label:16s} {MATRIX_LABEL[method]:28s} "
                      f"r_neg={reliability_r_xx(values):.3f}  between_party_sd={sd_text}")
            reliability_tidy = tidy_frame(spec, jobs[span], observed[span], replicates[span],
                                          parties)
            reliability_tidy["between_party_sd"] = [
                sd_lookup.get((row.method, row.model)) for row in reliability_tidy.itertuples()
            ]
            tidy.append(reliability_tidy)
            continue
        span = table_slices[name]
        spec = specs[name]
        print(f"\n[{name}]")
        if spec["kind"] == "pairs":
            # One small matrix per MODEL, carrying both framings (base in the
            # upper triangle, negated in the lower), rather than one per job:
            # the jobs are per (framing, model) but a reader wants the two
            # framings side by side, and the symmetric half is free space.
            model_scopes = []
            for job in jobs[span]:
                if job["scope"][1] not in model_scopes:
                    model_scopes.append(job["scope"][1])
            blocks = []
            for model_scope in model_scopes:
                by_framing, complete = {}, True
                for framing in MATRIX_FRAMINGS:
                    position = internal_index.get((framing, model_scope))
                    if position is None:
                        complete = False
                        break
                    by_framing[framing] = (jobs[position], observed[position],
                                           replicates[position])
                if not complete:
                    print(f"  {matrix_title(model_scope):24s} missing a framing, skipped.")
                    continue
                blocks.append(
                    render_matrix_latex(spec, methods, model_scope, by_framing,
                                        args.bootstrap, provenance)
                    if args.format == "latex"
                    else render_matrix_markdown(methods, model_scope, by_framing,
                                                args.bootstrap))
                summary = "  ".join(
                    f"{framing}: W={by_framing[framing][1]['w']:.2f} "
                    f"ICC={by_framing[framing][1]['icc']:.2f} "
                    f"top-1={by_framing[framing][1]['top1']:.0%}"
                    for framing in MATRIX_FRAMINGS)
                print(f"  {matrix_title(model_scope):24s} {summary}")
            rendered.append("\n\n".join(blocks))
        else:
            header, rows, prose_columns = table_body(spec, jobs[span], observed[span],
                                                      replicates[span], parties, args)
            rendered.append(render_latex(spec, header, rows, prose_columns, provenance)
                            if args.format == "latex"
                            else render_markdown(spec, header, rows))
        tidy.append(tidy_frame(spec, jobs[span], observed[span], replicates[span], parties))
        if spec["kind"] == "pairs":
            pairs.append(pairs_frame(spec, jobs[span], observed[span], replicates[span]))

    document = "\n\n".join(rendered)
    print()
    print(document)
    if outputs.latex:
        Path(outputs.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(outputs.latex).write_text(document + "\n", encoding="utf-8")
        print(f"\nWrote {outputs.latex}")
    if outputs.tidy_csv:
        Path(outputs.tidy_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tidy, ignore_index=True).to_csv(outputs.tidy_csv, index=False)
        print(f"Wrote {outputs.tidy_csv}")
    if outputs.pairs_csv:
        if not pairs:
            print("--pairs-csv given but no 'internal'-kind table was generated; skipping.")
        else:
            Path(outputs.pairs_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.concat(pairs, ignore_index=True).to_csv(outputs.pairs_csv, index=False)
            print(f"Wrote {outputs.pairs_csv}")


if __name__ == "__main__":
    main()
