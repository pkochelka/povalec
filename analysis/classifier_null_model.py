#!/usr/bin/env python3
"""The 0-centered normal null of vaa_null_model.py, carried over to the party classifier.

The VAA argmax shares have a null: random Likert answers drawn from a normal distribution
centered on 0 (vaa_null_model.py's "normal" null, sigma one Likert step), scored against the
group positions like a model's answers. It shows how much of a group's share is only its
position near the center of the answer space. The classifier rows of the argmax table
(classified reasons, classified open-ended answers) score text, not answer vectors, so
that null cannot be fed to them -- but its *stance distribution* can.

Every classified text carries a stance on its statement: a reason is written for a Likert
answer, and an open-ended answer has the cross-encoder stance the indirect-Likert method
scores. So the classifier's response to stance can be read off per statement:

    p_js(c)   share of texts on statement j at stance level s that the classifier puts
              in group c (levels: the five Likert points, -1 .. +1 in steps of 0.5)

and the null share of group c is what the classifier would report if the stances on every
statement were drawn from the normal null instead of from the models:

    null(c) = sum_j w_j sum_s pi(s) p_js(c)

with w_j the statement's share of the texts and pi(s) the null's probability of level s
(sigma = 0.5: about 38% neutral, 24% each mild level, 7% each strong one). Swapping pi(s)
for the texts' own stance mix gives back the real share exactly; that is checked on every
run. Topic, language, genre and model mix are held as they are -- only stance is
randomised, which is what the VAA null randomises too. p_js is pooled over all models, so
the null is one vector per method and framing, model-independent like the VAA null.

Units follow the argmax table: the classifier counts every text on its own, so pi(s) is
the single-answer null (the VAA null averages eight draws because its cells are
paraphrase-averaged). A thin (statement, level) cell is shrunk towards that level pooled
over all statements, p_js = (n_jsc + m p_s(c)) / (n_js + m); the summary reports how much
of the null's weight rests on cells thinner than m. The per-language nulls shrink towards
the whole run's p_js instead (see slice_target): one language has too few texts per cell
to stand on its own.

Outputs (under --out-dir, next to vaa_null_model_argmax.py's; any basis but ep-group adds
a `_<positions>` suffix, as there: classifier_null_cluster_shares.csv):
  classifier_null_shares.csv   method, variant, scope, slice, ep_group, null_share,
                               real_share, plus the coverage and thin-cell diagnostics.
                               scope "all" (slice "all statements") is the whole run;
                               scope "topic" / "language" rerun the formula on one topic's
                               statements / one language's texts, for the breakdowns
                               plot_argmax_shares draws minus their null
  classifier_null_statements.csv
                               method, variant, language ("all" or a code), statement_idx,
                               ep_group, null_mean_probability: the per-statement
                               mean-probability null rank_consistency_tables subtracts
  classifier_null_summary.md

Usage, from the repository root:
  python analysis/classifier_null_model.py --dataset euandi_2024 --positions cluster
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core import (
    REASONS_CLASSIFIED_CSV,
    SPEECHES_CLASSIFIED_CSV,
    SPEECHES_SCORED_CSV,
    find_csvs,
    party_sort_key,
    short_party,
)
from analysis.plotting.plot_classified_parties import slugify_party_label
from analysis.core.positions import POSITION_CHOICES
from analysis.vaa_null_model import DEFAULT_SIGMA
from analysis.plotting.plot_topical_parties import statement_rows_per_axis
from analysis.vaa_null_model_argmax import WHOLE, output_suffix
from utils import VARIANT_PATTERN, configure_stdout, likert_to_stance

configure_stdout()

# Method keys and labels as plot_argmax_shares / argmax_share_methods.csv spell them.
METHODS = {"reasons_clf": "classified reasons", "speeches_clf": "classified open-ended"}
LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
DEFAULT_SHRINKAGE = 20
POOLED = "pooled"
ALL_SCOPE = "all"

PREDICTED = re.compile(rf"^predicted_party_(?P<language>[a-z]{{2}}){VARIANT_PATTERN}_v(?P<paraphrase>\d+)$")
CHOICE = re.compile(rf"^choice_(?P<language>[a-z]{{2}}){VARIANT_PATTERN}_v(?P<paraphrase>\d+)$")
STANCE = re.compile(rf"^stance_(?P<language>[a-z]{{2}}){VARIANT_PATTERN}_v(?P<paraphrase>\d+)$")
ANSWER = re.compile(rf"^answer_(?P<language>[a-z]{{2}}){VARIANT_PATTERN}_v(?P<paraphrase>\d+)$")
PROBABILITY = re.compile(
    rf"^party_prob_(?P<slug>.+?)_(?P<language>[a-z]{{2}}){VARIANT_PATTERN}_v(?P<paraphrase>\d+)$")
# What the classifier's output is summarised as: the argmax share (a one-hot per text) or
# the mean probability per group (plot_classified_parties' panels). The decomposition is
# the same for both -- only the per-text vector that is averaged differs.
MEASURES = ("share", "mean_probability")


def null_level_probabilities(sigma):
    """pi(s) of the normal null: z ~ N(0, sigma) rounded to the Likert grid, which on the
    stance scale is z rounded to the nearest 0.5 and clipped to [-1, 1] -- the same
    transform vaa_null_model.make_generators applies, in closed form."""
    def cdf(x):
        return 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))
    edges = [-math.inf, -0.75, -0.25, 0.25, 0.75, math.inf]
    return np.array([cdf(high) - cdf(low) for low, high in zip(edges[:-1], edges[1:])])


def level_of(stance):
    """Nearest of the five stance levels; NaN stays NaN."""
    return np.clip(np.rint(np.asarray(stance, dtype=float) * 2.0) / 2.0, -1.0, 1.0)


def columns_by_key(columns, pattern):
    return {(m.group("language"), int(m.group("paraphrase"))): column
            for column in columns if (m := pattern.match(column))}


def probability_columns(columns):
    """{(language, paraphrase): {party slug: party_prob_ column}}."""
    found = {}
    for column in columns:
        if (m := PROBABILITY.match(column)):
            key = (m.group("language"), int(m.group("paraphrase")))
            found.setdefault(key, {})[m.group("slug")] = column
    return found


def long_texts(frame, predicted, stance_values):
    """One row per text: statement, language, paraphrase, predicted group, stance level,
    and the classifier's probability per group as `prob_<party slug>` columns.
    `stance_values` maps (language, paraphrase) to that column's stances."""
    probabilities = probability_columns(frame.columns)
    records = []
    for key, column in predicted.items():
        if key not in stance_values:
            continue
        records.append(pd.DataFrame({
            "statement_idx": np.arange(len(frame)),
            "language": key[0],
            "paraphrase": key[1],
            "predicted": frame[column].where(frame[column].map(type) == str),
            "level": level_of(stance_values[key]),
            **{f"prob_{slug}": pd.to_numeric(frame[prob_column], errors="coerce")
               for slug, prob_column in probabilities.get(key, {}).items()},
        }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def reasons_texts(path):
    """Classified reasons with the Likert answer each was written for. The answer and the
    reason sit in the same row of the same file, so no join is needed."""
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    choices = columns_by_key(frame.columns, CHOICE)
    stances = {}
    for key, column in choices.items():
        likert = pd.to_numeric(frame[column], errors="coerce")
        likert = likert.where(likert.between(1, 5))      # 0 / junk are not an answer
        stances[key] = likert_to_stance(likert.to_numpy(dtype=float))
    return long_texts(frame, columns_by_key(frame.columns, PREDICTED), stances)


def speeches_texts(classified_path, scored_path):
    """Classified open-ended answers with their cross-encoder stance. The two files are
    written from the same speeches CSV and joined by row position, so the answer texts
    are compared first -- a silent misalignment would pair one text's stance with
    another's group, which is the whole computation."""
    classified = pd.read_csv(classified_path, sep=";", encoding="utf-8-sig")
    scored = pd.read_csv(scored_path, sep=";", encoding="utf-8-sig")
    left = columns_by_key(classified.columns, ANSWER)
    right = columns_by_key(scored.columns, ANSWER)
    for key in set(left) & set(right):
        # fillna first: under pandas' string dtype a missing answer stays NA through
        # astype(str), and NA never equals NA -- a refusal in both files is not a mismatch.
        if not np.array_equal(classified[left[key]].fillna("").astype(str).to_numpy(),
                              scored[right[key]].fillna("").astype(str).to_numpy()):
            raise SystemExit(f"{classified_path.name} and {scored_path.name} are not "
                             f"row-aligned ({key}); cannot pair stance with prediction.")
    stances = {key: pd.to_numeric(scored[column], errors="coerce").to_numpy(dtype=float)
               for key, column in columns_by_key(scored.columns, STANCE).items()}
    return long_texts(classified, columns_by_key(classified.columns, PREDICTED), stances)


def load_texts(results_dir):
    """{(method, variant): texts of every model}, only texts the classifier labelled."""
    frames = {}
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        reasons = find_csvs(model_dir, REASONS_CLASSIFIED_CSV)
        speeches = find_csvs(model_dir, SPEECHES_CLASSIFIED_CSV)
        scored = find_csvs(model_dir, SPEECHES_SCORED_CSV)
        if not reasons and not speeches:
            continue
        for variant, path in reasons.items():
            frames.setdefault(("reasons_clf", variant), []).append(
                reasons_texts(path).assign(model=model_dir.name))
        for variant, path in speeches.items():
            if variant not in scored:
                print(f"  {model_dir.name}: no scored file for {variant} speeches, skipping.")
                continue
            frames.setdefault(("speeches_clf", variant), []).append(
                speeches_texts(path, scored[variant]).assign(model=model_dir.name))
        print(f"  {model_dir.name}: loaded")
    return {key: pd.concat(parts, ignore_index=True).dropna(subset=["predicted"])
            for key, parts in frames.items()}


def text_values(texts, groups, measure):
    """(texts kept, texts x groups matrix) the measure averages: a one-hot of the predicted
    group for the argmax share, the classifier's probabilities for the mean probability.
    A text without a full probability vector has no mean probability to contribute."""
    if measure == "share":
        values = (texts["predicted"].to_numpy()[:, None] == np.array(groups)[None]).astype(float)
        return texts, values
    columns = [f"prob_{slugify_party_label(group)}" for group in groups]
    missing = [column for column in columns if column not in texts]
    if missing:
        raise SystemExit(f"No classifier probabilities for {missing} -- the classified CSVs "
                         "carry predicted labels only.")
    texts = texts.dropna(subset=columns)
    return texts, texts[columns].to_numpy(dtype=float)


def null_share(texts, pi, groups, shrinkage, target=None, measure="share"):
    """(null, real value on the same texts, thin-cell weight) per group, plus the shrunk
    cells and the shrinkage targets used, both {statement_idx: levels x groups}. The value
    is the argmax share or the mean probability, as `measure` says.

    Only texts with a stance level take part, in both, so the two are computed on one set
    of texts and their difference is the stance effect alone.

    `target` is what a thin cell is shrunk towards; by default each level's value pooled
    over these texts' statements."""
    texts, values = text_values(texts.dropna(subset=["level"]), groups, measure)
    level_index = {level: i for i, level in enumerate(LEVELS)}
    statements = np.sort(texts["statement_idx"].unique())
    rows = texts["statement_idx"].map({j: i for i, j in enumerate(statements)}).to_numpy()
    cols = texts["level"].map(level_index).to_numpy()
    sums = np.zeros((len(statements), len(LEVELS), len(groups)))     # sum over texts
    np.add.at(sums, (rows, cols), values)
    n_js = np.zeros((len(statements), len(LEVELS)))                  # texts per cell
    np.add.at(n_js, (rows, cols), 1.0)

    n_j = n_js.sum(axis=1)
    w = n_j / n_j.sum()                                              # statement weights
    if target is None:
        p_s = sums.sum(axis=0) / np.maximum(n_js.sum(axis=0)[:, None], 1.0)
        target_cells = np.broadcast_to(p_s[None], sums.shape)
    else:
        target_cells = np.stack([target[j] for j in statements])
    shrunk = (sums + shrinkage * target_cells) / (n_js[..., None] + shrinkage)
    null = np.einsum("j,s,jsc->c", w, pi, shrunk)

    # The real value through the same decomposition, with the texts' own stance mix in
    # place of pi and the raw cell means: it has to equal the plain mean over the texts.
    raw = sums / np.maximum(n_js[..., None], 1.0)
    rebuilt = np.einsum("j,js,jsc->c", w, n_js / n_j[:, None], raw)
    real = values.mean(axis=0)
    if not np.allclose(rebuilt, real):
        raise SystemExit(f"decomposition does not reproduce the real {measure}: "
                         f"{rebuilt} vs {real}")
    thin = float(np.einsum("j,s,js->", w, pi, (n_js < shrinkage).astype(float)))
    return null, real, thin, dict(zip(statements, shrunk)), dict(zip(statements, target_cells))


def slices(texts, axis_rows):
    """(scope, slice, texts) for every breakdown plot_argmax_shares draws: all statements,
    each topic's statements, each language. A slice's null is the formula run on that
    slice's texts alone, so its statement weights and its classifier response p_js are
    the slice's own -- the same subset its real share is counted on."""
    for axis, rows in axis_rows.items():
        subset = texts[texts["statement_idx"].isin(rows)]
        if len(subset):
            yield "topic", axis, subset
    for language, subset in texts.groupby("language"):
        yield "language", language, subset


def slice_target(scope, cells, targets):
    """What a slice's thin cells are shrunk towards. A topic keeps the whole run's
    targets, so its cells -- the same texts -- come out exactly as in the whole-run null.
    A language has a twentieth of the texts per cell, so it is shrunk towards the same
    (statement, level) cell over all languages rather than towards its own level pooled
    over statements: where a language is thin it keeps the statement's classifier
    response, and departs from it only as far as its own texts say."""
    return targets if scope == "topic" else cells


def slice_values(texts, pi, groups, shrinkage, axis_rows, measure):
    """[(scope, slice, texts, (null, real, thin), cells)] for the whole run and every
    slice; `cells` are the slice's shrunk {statement_idx: levels x groups}."""
    *whole, cells, targets = null_share(texts, pi, groups, shrinkage, measure=measure)
    parts = [(ALL_SCOPE, WHOLE, texts, whole, cells)]
    for scope, slice_name, subset in slices(texts, axis_rows):
        *values, slice_cells, _ = null_share(subset, pi, groups, shrinkage,
                                             slice_target(scope, cells, targets),
                                             measure=measure)
        parts.append((scope, slice_name, subset, values, slice_cells))
    return parts


def statement_records(method, variant, parts, pi, groups):
    """The mean-probability null per statement, for the whole run (language "all") and
    each language: sum_s pi(s) q_js(c), one text's expected probability on statement j
    under the null. rank_consistency_tables scores every text on its own statement, so
    this is the null it subtracts."""
    records = []
    for scope, slice_name, _, _, cells in parts:
        if scope not in (ALL_SCOPE, "language"):
            continue
        language = "all" if scope == ALL_SCOPE else slice_name
        for statement, cell in cells.items():
            for group, value in zip(groups, pi @ cell):
                records.append({"method": METHODS[method], "variant": variant,
                                "language": language, "statement_idx": int(statement),
                                "ep_group": group, "null_mean_probability": value})
    return records


def compute(texts_by_key, sigma, shrinkage, axis_rows):
    pi = null_level_probabilities(sigma)
    groups = sorted({g for texts in texts_by_key.values() for g in texts["predicted"].unique()},
                    key=party_sort_key)
    records, statement_rows = [], []
    for (method, variant), texts in sorted(texts_by_key.items()):
        shares = slice_values(texts, pi, groups, shrinkage, axis_rows, "share")
        probabilities = slice_values(texts, pi, groups, shrinkage, axis_rows, "mean_probability")
        statement_rows += statement_records(method, variant, probabilities, pi, groups)
        for (scope, slice_name, subset, (null, real, thin), _), \
                (*_, (null_p, real_p, _), _) in zip(shares, probabilities):
            coverage = float(subset["level"].notna().mean())
            level_mix = subset["level"].value_counts(normalize=True).reindex(LEVELS, fill_value=0)
            if scope == ALL_SCOPE:
                print(f"  {method}/{variant}: {len(subset):,} texts, {coverage:.1%} with a "
                      f"stance, null weight on thin cells {thin:.1%}")
            for i, group in enumerate(groups):
                records.append({"method": METHODS[method], "variant": variant,
                                "scope": scope, "slice": slice_name, "ep_group": group,
                                "null_share": null[i], "real_share": real[i],
                                "null_mean_probability": null_p[i],
                                "real_mean_probability": real_p[i],
                                "texts": len(subset), "stance_coverage": coverage,
                                "thin_cell_weight": thin,
                                "neutral_share_of_texts": float(level_mix.loc[0.0])})
    table = pd.DataFrame(records)
    # The argmax table and figures report the framings pooled as a plain mean
    # (plot_argmax_shares' pooled_over_variants); the null is pooled the same way.
    value_columns = ["null_share", "real_share", "null_mean_probability",
                     "real_mean_probability"]
    pooled = (table.groupby(["method", "scope", "slice", "ep_group"], as_index=False)
                   [value_columns].mean().assign(variant=POOLED))
    return (pd.concat([table, pooled], ignore_index=True), pd.DataFrame(statement_rows),
            pi, groups)


def render_summary(table, pi, groups, sigma, shrinkage):
    sliced = table[table["scope"] != ALL_SCOPE].dropna(subset=["thin_cell_weight"])
    table = table[table["scope"] == ALL_SCOPE]
    lines =["# Classifier null model (0-centered normal stance null)", "",
             f"Stance levels -1 .. +1; null probabilities (sigma = {sigma:.2f}): "
             + ", ".join(f"{level:+.1f}: {p:.1%}" for level, p in zip(LEVELS, pi))
             + f". Thin cells (< {shrinkage} texts) are shrunk towards the level pooled over "
             "statements.", ""]
    for method in METHODS.values():
        part = table[(table["method"] == method) & (table["variant"] == POOLED)].set_index("ep_group")
        if part.empty:
            continue
        lines += [f"## {method} (framings pooled)", "",
                  "| group | real | null | real - null |", "|---|---:|---:|---:|"]
        for group in groups:
            if group in part.index:
                real, null = part.at[group, "real_share"], part.at[group, "null_share"]
                lines.append(f"| {short_party(group)} | {real:.1%} | {null:.1%} | "
                             f"{100 * (real - null):+.1f} pp |")
        lines.append("")
    diagnostics = table[table["variant"] != POOLED].drop_duplicates(["method", "variant"])
    lines += ["## Coverage", "", "| method | framing | texts | with stance | neutral texts | "
              "null weight on thin cells |", "|---|---|---:|---:|---:|---:|"]
    for _, row in diagnostics.iterrows():
        lines.append(f"| {row['method']} | {row['variant']} | {int(row['texts']):,} | "
                     f"{row['stance_coverage']:.1%} | {row['neutral_share_of_texts']:.1%} | "
                     f"{row['thin_cell_weight']:.1%} |")
    if not sliced.empty:
        lines += ["", "Per-topic and per-language nulls (what plot_argmax_shares subtracts from "
                  "those breakdowns) are in the CSV under scope `topic` / `language`. A topic "
                  "uses the whole run's cells; a language's thin cells are shrunk towards "
                  "the same (statement, level) cell over all languages. Null weight on thin "
                  "cells, worst slice per scope:", ""]
        for scope, part in sliced.groupby("scope"):
            worst = part.loc[part["thin_cell_weight"].idxmax()]
            lines.append(f"- {scope}: {worst['thin_cell_weight']:.1%} "
                         f"({worst['method']}, {worst['variant']}, {worst['slice']})")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Read this results directory instead of the --dataset one.")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Null SD in stance units (default: the VAA null's, %(default)s).")
    parser.add_argument("--shrinkage", type=int, default=DEFAULT_SHRINKAGE,
                        help="Pseudo-count pulling a thin (statement, level) cell towards "
                             "the level pooled over statements.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <results_dir>/tables/null_model")
    parser.add_argument("--positions", default="ep-group", choices=POSITION_CHOICES,
                        help="Only names the outputs: the classifier's groups follow the "
                             "basis the pipeline runs on, so an EP-group and a cluster run "
                             "must not overwrite each other.")
    return parser.parse_args()


def output_stem(positions, sigma=DEFAULT_SIGMA):
    """classifier_null for the EP-group basis, classifier_null_<basis> otherwise, plus a
    _sigma<value> tag for a non-default sigma -- the naming vaa_null_model_argmax uses.
    The argmax table finds the file through this."""
    return "classifier_null" + output_suffix(positions, sigma)


def main():
    args = parse_args()
    results_dir = args.results_dir or Path("data") / f"{args.dataset}_results"
    out_dir = args.out_dir or results_dir / "tables" / "null_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading classified texts from {results_dir}")
    texts_by_key = load_texts(results_dir)
    if not texts_by_key:
        raise SystemExit("No classified reasons or open-ended answers found.")
    statements = 1 + max(int(texts["statement_idx"].max()) for texts in texts_by_key.values())
    axis_rows = {axis: rows for axis, rows in
                 statement_rows_per_axis(args.dataset, statements).items() if len(rows)}
    table, statement_table, pi, groups = compute(texts_by_key, args.sigma, args.shrinkage,
                                                 axis_rows)
    stem = output_stem(args.positions, args.sigma)
    table.to_csv(out_dir / f"{stem}_shares.csv", index=False)
    statement_table.to_csv(out_dir / f"{stem}_statements.csv", index=False)
    summary = render_summary(table, pi, groups, args.sigma, args.shrinkage)
    (out_dir / f"{stem}_summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Wrote {out_dir / f'{stem}_shares.csv'}, {stem}_summary.md")


if __name__ == "__main__":
    main()
