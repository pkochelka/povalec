#!/usr/bin/env python3
"""The VAA null model in the terms of the paper's argmax-share figures.

The argmax figures (plot_argmax_shares.py) do not pick one winning group per answer
vector. They take the paraphrase-averaged stance per (statement, language), ask for each
statement separately which group agrees most with it (ties split), pool the base and
negated framings, and report the share of (statement, language) cells each group wins --
per method, and averaged over the methods. This script puts the null baselines of
vaa_null_model.py on exactly that footing:

  real       per model and pooled over models: the direct-Likert VAA share, the indirect
             (cross-encoder) VAA share, and their mean -- the two VAA components of the
             figure's "average of methods" bar. (The two classifier methods have no VAA
             null, so the figure's four-method average is not reproduced here.)
  nulls      uniform, normal (mean 0) and LLM-shuffle vectors, averaged over as many
             draws as a language has paraphrases, then scored cell by cell the same way.
             The LLM-shuffle pools are per track, so its share is reported per track and
             averaged like the real one.

The real shares here equal plot_argmax_shares' `argmax_share_methods.csv` (direct Likert /
indirect Likert, variant "pooled") to floating-point precision.

Outputs (under --out-dir, inside the gitignored data/ tree):
  null_model_argmax_shares.csv   tidy: method, side, model, topic, ep_group, argmax_share
  null_model_argmax_summary.md   per-model tables per method, plus a per-topic table
  null_model_argmax_shares.png   the figure: groups on the x axis, one bar per model,
                                 the nulls as hatched bars, VAA-average method

Usage, from the repository root with the package installed:
  python analysis/vaa_null_model_argmax.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.analyze_all import MODEL_DIRS
from analysis.core import DEFAULT_POSITIONS, model_display_name
from analysis.plotting.plot_topical_parties import statement_rows_per_axis
from analysis.vaa_null_model import (
    ALL_MODELS,
    DEFAULT_SIGMA,
    NULLS,
    SIDE_COLORS,
    SIDE_LABELS,
    averaged_draws,
    cell_scores,
    load_runs,
    make_generators,
    markdown_table,
    pct,
    positions_tensor,
    summarise_cells,
)
from utils import configure_stdout

configure_stdout()

SOURCES = ["likert", "speeches"]
METHOD_LABEL = {"likert": "direct Likert", "speeches": "indirect Likert", "average": "VAA average"}
WHOLE = "all statements"


def language_vectors(runs):
    """{model: (N, statements)} paraphrase-averaged vectors, ALL_MODELS pooled -- the
    stance the argmax figures score."""
    by_language = runs.groupby(level=["model", "framing", "language"]).mean()
    vectors = {ALL_MODELS: by_language.to_numpy(dtype=float)}
    vectors.update({model: frame.to_numpy(dtype=float)
                    for model, frame in by_language.groupby(level="model")})
    return vectors


def cell_shares(S, P, party_group, groups, rows):
    """{group: argmax share} over the (vector, statement) cells of `rows`."""
    summary = summarise_cells(cell_scores(S[:, rows], P[:, rows], party_group, groups), groups)
    return {group: share for group, (share, _, _) in summary.items()}


def score_source(source, runs, P, party_group, groups, topic_rows, args, rng):
    vectors = language_vectors(runs)
    paraphrases = int(runs.groupby(level=["model", "framing", "language"]).size().median())
    generators = make_generators(runs, args.sigma, rng)
    nulls = {name: averaged_draws(draw, args.draws, paraphrases)
             for name, draw in generators.items()}
    print(f"  {source}: {len(runs)} runs, {paraphrases} paraphrases per language, "
          f"{args.draws} null draws x {paraphrases}")

    records = []
    for topic, rows in [(WHOLE, np.arange(P.shape[1])), *topic_rows.items()]:
        for model, S in vectors.items():
            for group, share in cell_shares(S, P, party_group, groups, rows).items():
                records.append(dict(method=source, side="real", model=model, topic=topic,
                                    ep_group=group, argmax_share=share))
        for name, S in nulls.items():
            for group, share in cell_shares(S, P, party_group, groups, rows).items():
                records.append(dict(method=source, side=name, model=ALL_MODELS, topic=topic,
                                    ep_group=group, argmax_share=share))
    return pd.DataFrame.from_records(records)


def with_average(table):
    """Add method="average": the mean of the two tracks' shares for every (side, model,
    topic, group) both tracks have."""
    both = table.groupby(["side", "model", "topic", "ep_group"])["argmax_share"]
    average = both.agg(["mean", "size"]).reset_index()
    average = average[average["size"] == len(SOURCES)].drop(columns="size")
    average = average.rename(columns={"mean": "argmax_share"}).assign(method="average")
    return pd.concat([table, average], ignore_index=True)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def row_label(side, model):
    if side != "real":
        return SIDE_LABELS[side]
    return "all models (pooled)" if model == ALL_MODELS else model_display_name(model)


def shares_table(table, method, topic, groups, models):
    sub = table[(table["method"] == method) & (table["topic"] == topic)]
    sub = sub.set_index(["side", "model", "ep_group"])["argmax_share"]
    rows = []
    for side, model in [*(("real", m) for m in [ALL_MODELS, *models]),
                        *((null, ALL_MODELS) for null in NULLS)]:
        rows.append([row_label(side, model),
                     *(pct(sub.get((side, model, g), np.nan)) for g in groups)])
    return markdown_table(["", *groups], rows)


def render_summary(table, groups, models, topic_rows, args):
    counts = {topic: len(rows) for topic, rows in topic_rows.items()}
    lines = [
        "# VAA null model, argmax-share footing", "",
        f"Argmax share per (statement, language) cell on the paraphrase-averaged stance, ties "
        f"split, base and negated framings pooled -- the quantity of the paper's argmax "
        f"figures. Positions `{args.positions}`, ECR+ID "
        f"{'separate' if args.no_collapse_ecr_id else 'collapsed'}. Nulls: {args.draws} "
        f"vectors each, every vector the mean of as many draws as a language has "
        f"paraphrases; normal null sigma = {args.sigma:.3f}. The LLM-shuffle null resamples "
        "each statement from that track's own pooled answers.", "",
    ]
    for method in ["average", *SOURCES]:
        lines += [f"## {METHOD_LABEL[method]}, all statements", "",
                  shares_table(table, method, WHOLE, groups, models), ""]
    lines += ["## VAA average per topic (statements per topic in parentheses)", ""]
    for topic in topic_rows:
        lines += [f"### {topic} ({counts[topic]})", "",
                  shares_table(table, "average", topic, groups, models), ""]
    return "\n".join(lines)


def plot_shares(table, groups, models, out_path, method="average"):
    sub = table[(table["method"] == method) & (table["topic"] == WHOLE)]
    sub = sub.set_index(["side", "model", "ep_group"])["argmax_share"]
    bars = [("real", m) for m in models] + [(null, ALL_MODELS) for null in NULLS]
    model_colors = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(18, 6.5))
    x = np.arange(len(groups))
    width = 0.86 / len(bars)
    for bi, (side, model) in enumerate(bars):
        values = 100 * np.array([sub.get((side, model, g), np.nan) for g in groups])
        offsets = x - 0.43 + width * (bi + 0.5)
        if side == "real":
            ax.bar(offsets, values, width=width, color=model_colors[model],
                   edgecolor="white", linewidth=0.5)
        else:
            ax.bar(offsets, values, width=width, color="white", edgecolor=SIDE_COLORS[side],
                   hatch="////", linewidth=1.2)
    pooled = 100 * np.array([sub.get(("real", ALL_MODELS, g), np.nan) for g in groups])
    for xi, value in zip(x, pooled):
        ax.hlines(value, xi - 0.43, xi + 0.43, color="#333333", linewidth=1.2,
                  linestyle=(0, (3, 2)))
        ax.text(xi + 0.44, value, f"{value:.0f}", va="center", ha="left", fontsize=8,
                color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Argmax share (%)", fontsize=10)
    ax.set_title(f"Per-statement argmax share, {METHOD_LABEL[method]}: models vs nulls "
                 "(dashed line = all models pooled)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [Patch(color=model_colors[m], label=model_display_name(m)) for m in models]
    handles += [Patch(facecolor="white", edgecolor=SIDE_COLORS[n], hatch="////",
                      label=SIDE_LABELS[n]) for n in NULLS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2024"])
    parser.add_argument("--models", nargs="+", default=MODEL_DIRS)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=["ep-group", "national"])
    parser.add_argument("--no-collapse-ecr-id", action="store_true")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--out-dir", default=None,
                        help="default: data/<dataset>_results/tables/null_model")
    args = parser.parse_args()

    results_root = Path("data") / f"{args.dataset}_results"
    model_dirs = [results_root / m for m in args.models]
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "tables" / "null_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tables, models, P = [], None, None
    for source in SOURCES:
        print(f"Loading real {source} runs")
        runs = load_runs(model_dirs, source)
        if P is None:
            P, party_group, groups = positions_tensor(
                args.positions, not args.no_collapse_ecr_id, runs.shape[1])
            topic_rows = {axis: rows for axis, rows in
                          statement_rows_per_axis(args.dataset, runs.shape[1]).items()
                          if len(rows)}
            models = [m for m in args.models if m in runs.index.get_level_values("model")]
        tables.append(score_source(source, runs, P, party_group, groups, topic_rows, args, rng))
    table = with_average(pd.concat(tables, ignore_index=True))

    table.to_csv(out_dir / "null_model_argmax_shares.csv", index=False)
    summary = render_summary(table, groups, models, topic_rows, args)
    (out_dir / "null_model_argmax_summary.md").write_text(summary, encoding="utf-8")
    plot_shares(table, groups, models, out_dir / "null_model_argmax_shares.png")
    print()
    print(summary.split("\n## VAA average per topic")[0])
    print(f"\nWrote {out_dir / 'null_model_argmax_shares.csv'}, null_model_argmax_summary.md, "
          "null_model_argmax_shares.png")


if __name__ == "__main__":
    main()
