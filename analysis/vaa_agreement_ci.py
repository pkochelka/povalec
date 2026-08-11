#!/usr/bin/env python3
"""Bootstrapped 95% confidence intervals for each model's mean euandi (VAA) agreement.

The vaa*.csv files hold agreement already averaged over statements, so they cannot
say whether two models differ by more than statement-sampling noise. This recomputes
agreement per (EP group, language, statement) from the raw LLM answers and the euandi
party positions, then runs a cluster bootstrap over statements -- the unit that is
actually sampled. Languages, parties and prompt variants are fixed by design and are
averaged within each draw, so a drawn statement carries all of its cells.

Every model is scored on the same bootstrap draws (common random numbers), which makes
the difference table paired: comparing two independent CIs understates the evidence,
because the shared statement noise cancels in the difference.
"""
import argparse
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import ALL_LANGS_STR, configure_stdout
from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS, POSITION_CHOICES,
    likert_stance_frame, load_party_positions, positions_path, speech_stance_frame,
)

configure_stdout()

DEFAULT_MODELS = [
    "deepseek-v4-pro", "gemma-4-31b", "glm-5.2", "gpt-oss-120b",
    "kimi-k2.7", "mistral-medium-3.5", "qwen3.5-122b",
]
VARIANT_LABEL = {"": "base", "_negated": "negated"}
SOURCE_LABEL = {"likert": "likert", "speeches": "open-ended"}
CONFIDENCE = 95


def scope_label(source, variant):
    return f"{SOURCE_LABEL[source]} ({VARIANT_LABEL[variant]})"


def stance_frame_for(results_dir, model, source, variant, languages):
    """The model's per-statement stance per language, or None if it was never run."""
    if source == "likert":
        path = os.path.join(results_dir, model, f"{languages}{variant}.csv")
        if not os.path.exists(path):
            return None, []
        return likert_stance_frame(path, negated=variant == "_negated")
    path = os.path.join(results_dir, model, f"speeches_{languages}{variant}_scored.csv")
    if not os.path.exists(path):
        return None, []
    return speech_stance_frame(path, variant)


def agreement_matrices(stance_df, languages, party_df, groups, statement_index):
    """Per (EP group, statement) sum and count of agreement values -- the sufficient
    statistics for a cluster bootstrap of the mean. Kept per group because parties
    abstain on some statements, so the groups carry different observation counts and
    a fixed per-party weight would not reproduce the mean-of-group-means that
    evaluate_euandi writes into vaa*.csv."""
    stance_columns = [f"{lang}_stance" for lang in languages]
    merged = party_df.merge(stance_df[["statement_idx", *stance_columns]], on="statement_idx")
    long = merged.melt(
        id_vars=["ep_group", "statement_idx", "normalized_answer"],
        value_vars=stance_columns, var_name="language", value_name="llm_stance",
    )
    long["agreement"] = 1 - (long["normalized_answer"] - long["llm_stance"]).abs() / 2
    aggregated = long.groupby(["ep_group", "statement_idx"])["agreement"].agg(["sum", "count"])
    return tuple(
        aggregated[column].unstack("statement_idx")
        .reindex(index=groups, columns=statement_index).fillna(0.0).to_numpy(dtype=float)
        for column in ("sum", "count")
    )


def ratio_or_nan(sums, counts):
    return np.divide(sums, counts, out=np.full(np.shape(sums), np.nan, dtype=float),
                     where=counts > 0)


def bootstrap_group_means(matrices, draws):
    """Each EP group's mean agreement, observed (groups) and per draw (groups x draws)."""
    sums, counts = matrices
    observed = ratio_or_nan(sums.sum(axis=1), counts.sum(axis=1))
    replicates = ratio_or_nan(sums[:, draws].sum(axis=2), counts[:, draws].sum(axis=2))
    return observed, replicates


def bootstrap_means(matrices, draws):
    """Mean over EP groups of each group's mean agreement, observed and per draw."""
    observed, replicates = bootstrap_group_means(matrices, draws)
    return float(np.nanmean(observed)), np.nanmean(replicates, axis=0)


def percentile_interval(values, axis=None):
    tail = (100 - CONFIDENCE) / 2
    return (np.nanpercentile(values, tail, axis=axis),
            np.nanpercentile(values, 100 - tail, axis=axis))


def format_cell(observed, replicates):
    low, high = percentile_interval(replicates)
    return f"{observed:.3f} [{low:.3f}, {high:.3f}]"


def format_difference(observed, replicates):
    low, high = percentile_interval(replicates)
    marker = "" if low <= 0 <= high else " \\*"
    return f"{observed:+.3f} [{low:+.3f}, {high:+.3f}]{marker}"


def markdown_table(header, rows):
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = [
        "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(header, widths)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    lines += [
        "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def collect_replicates(args, scopes, party_df, groups, statement_index, draws):
    """{(model, scope): (observed mean, bootstrap replicates)} over every model x scope."""
    results = {}
    for model in args.models:
        for source, variant in scopes:
            stance_df, languages = stance_frame_for(
                args.results_dir, model, source, variant, args.languages)
            if stance_df is None:
                print(f"  [{model}/{scope_label(source, variant)}] no input file, skipping.")
                continue
            matrices = agreement_matrices(
                stance_df, languages, party_df, groups, statement_index)
            if not matrices[1].any():
                print(f"  [{model}/{scope_label(source, variant)}] no usable agreements.")
                continue
            results[(model, (source, variant))] = bootstrap_means(matrices, draws)
    return results


def agreement_table(models, scopes, results):
    header = ["model", *(scope_label(*scope) for scope in scopes)]
    rows = []
    for model in models:
        cells = []
        for scope in scopes:
            entry = results.get((model, scope))
            cells.append(format_cell(*entry) if entry else "--")
        rows.append([model, *cells])
    return markdown_table(header, rows)


def difference_table(models, scopes, results, reference):
    header = ["model", *(scope_label(*scope) for scope in scopes)]
    rows = []
    for model in models:
        if model == reference:
            continue
        cells = []
        for scope in scopes:
            entry, reference_entry = results.get((model, scope)), results.get((reference, scope))
            if entry is None or reference_entry is None:
                cells.append("--")
                continue
            cells.append(format_difference(
                entry[0] - reference_entry[0], entry[1] - reference_entry[1]))
        rows.append([model, *cells])
    return markdown_table(header, rows)


def rank_models(models, scopes, results):
    """Models ordered by mean agreement pooled over the scopes they have."""
    def pooled(model):
        values = [results[(model, scope)][0] for scope in scopes if (model, scope) in results]
        return float(np.mean(values)) if values else float("-inf")
    return sorted(models, key=pooled, reverse=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model dirs.")
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--sources", default="likert,speeches")
    parser.add_argument("--variants", default=",_negated",
                        help="Comma-separated filename infixes; '' is the base framing.")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Which euandi answers stand for an EP group (see evaluate_euandi).")
    parser.add_argument("--collapse-ecr-id", action="store_true",
                        help="Merge ECR and ID; affects the per-group weighting only.")
    parser.add_argument("--bootstrap", default=10000, type=int, help="Bootstrap draws.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default=None,
                        help="Write the markdown here as well as to stdout.")
    args = parser.parse_args()
    args.models = args.models.split(",")
    args.sources = args.sources.split(",")
    args.variants = [v for v in args.variants.split(",")]
    args.results_dir = os.path.join(PROJECT_ROOT, "data", f"{args.dataset}_results")
    return args


def main():
    args = parse_args()
    scopes = [(source, variant) for source in args.sources for variant in args.variants]

    party_df = load_party_positions(positions_path(args.positions), args.positions)
    if args.collapse_ecr_id:
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    party_df = party_df.dropna(subset=["ep_group"])
    statement_index = np.sort(party_df["statement_idx"].unique())
    groups = sorted(party_df["ep_group"].unique())

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(statement_index), size=(args.bootstrap, len(statement_index)))
    print(f"{len(statement_index)} statements, {party_df['short_name'].nunique()} parties in "
          f"{len(groups)} EP groups ({args.positions}), "
          f"{args.bootstrap} bootstrap draws, seed {args.seed}.")

    results = collect_replicates(args, scopes, party_df, groups, statement_index, draws)
    if not results:
        raise SystemExit("No model/scope combination produced agreements.")

    models = rank_models(args.models, scopes, results)
    reference = models[0]
    sections = [
        f"### Mean euandi agreement per model, {CONFIDENCE}% bootstrap CI",
        "",
        agreement_table(models, scopes, results),
        "",
        f"Cluster bootstrap over {len(statement_index)} statements, {args.bootstrap} draws "
        f"(seed {args.seed}); positions basis `{args.positions}`; "
        f"{len(args.languages.split(','))} languages averaged within each draw. "
        "Models are ordered by mean agreement pooled over the scopes above.",
        "",
        f"### Difference from `{reference}` (paired on the same draws)",
        "",
        difference_table(models, scopes, results, reference),
        "",
        "Negative means lower agreement than the reference. `*` marks intervals "
        "excluding zero, i.e. a difference larger than statement-sampling noise.",
    ]
    markdown = "\n".join(sections)

    print()
    print(markdown)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
