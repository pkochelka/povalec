#!/usr/bin/env python3
"""The VAA null model of vaa_null_model.py, broken down by EU&I policy topic.

The whole-questionnaire null asks whether S&D wins the argmax more often than a
signal-free answer vector would. Per topic the same question is sharper, because the
geometry the reviewer worries about differs from axis to axis: on Ukraine the groups
nearly coincide, on Immigration and Ecology they fan out. So for each topical axis of
the questionnaire (Ukraine, Ecology, Immigration, Values, Economy, Europe, Left-Right)
this script restricts both the real runs and the null draws to the statements loading on
that axis and rescores them -- the same three nulls (uniform, normal with mean 0,
LLM-shuffle), the same four aggregation levels (run, language, cell, model), and the same
scorer, all imported from vaa_null_model. A statement can load on several axes, so the
topics overlap and their shares do not partition the whole-questionnaire result.

The `cell` level per topic is exactly the per-topic argmax share the paper's topic figures
report (plot_argmax_shares.topic_shares); the other levels are the whole-vector argmax
over the topic's statements only.

Outputs (under --out-dir, inside the gitignored data/ tree):
  null_model_topics_<source>_win_shares.csv   tidy long table with a `topic` column
  null_model_topics_<source>_summary.md       per-topic tables plus a focus-group overview
  null_model_topics_<source>_win_shares.png   topics x levels grid of win shares

Usage, from the repository root with the package installed:
  python analysis/vaa_null_model_topics.py
  python analysis/vaa_null_model_topics.py --source speeches
  python analysis/vaa_null_model_topics.py --clusters mean      # k=4 clusters, see below

`--clusters mean|parties` scores against the k=4 party clusters instead of the EP groups,
with the cluster positions of vaa_null_model_clusters.py (`--pooling` there; `--weight
seats` needs `mean`). Those outputs are named null_model_topics_clusters_<pooling>_
<weight>_<source>_*.
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
    LEVELS,
    LEVEL_TITLES,
    NULLS,
    SIDES,
    SIDE_COLORS,
    SIDE_LABELS,
    load_runs,
    make_generators,
    markdown_table,
    null_levels,
    pct,
    positions_tensor,
    real_levels,
    score_everything,
)
from utils import AXES, configure_stdout

configure_stdout()

WHOLE = "all statements"


def slice_levels(levels, rows):
    """The same {level: {key: matrix}} restricted to the statements in `rows`."""
    return {level: {key: S[:, rows] for key, S in by_key.items()}
            for level, by_key in levels.items()}


def score_topics(real, null, P, party_group, groups, source, topic_rows):
    """One tidy table over every topic, with the whole questionnaire as a topic of its own
    so the per-topic rows can be read against it."""
    frames = []
    for topic, rows in [(WHOLE, np.arange(P.shape[1])), *topic_rows.items()]:
        table = score_everything(slice_levels(real, rows), slice_levels(null, rows),
                                 P[:, rows], party_group, groups, source)
        frames.append(table.assign(topic=topic, statements=len(rows)))
        print(f"  {topic}: {len(rows)} statements")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def topic_label(topic, counts):
    return topic if topic == WHOLE else f"{topic} ({counts[topic]})"


def render_summary(table, groups, models, topic_rows, args, sigma):
    counts = {topic: len(rows) for topic, rows in topic_rows.items()}
    topics = [WHOLE, *topic_rows]
    pooled = table[table["model"] == ALL_MODELS]
    focus = args.focus
    lines = [
        f"# VAA null model by topic -- {args.source} track", "",
        f"Positions: `{args.positions}`"
        + ("" if args.clusters else
           f", ECR+ID {'separate' if args.no_collapse_ecr_id else 'collapsed'}")
        + f". Null draws: {args.draws} "
        f"per null and level, seed {args.seed}, normal null sigma = {sigma:.3f}. Statements "
        "per topic in parentheses; a statement can load on several topics.", "",
        "Win share = share of vectors (or cells) the group has the highest agreement on over "
        "the topic's statements only, ties split. Levels: "
        + "; ".join(f"`{l}` = {LEVEL_TITLES[l]}" for l in LEVELS) + ".", "",
        f"## {focus} win share per topic: real runs against the nulls", "",
        "Cells: real / uniform / normal / LLM-shuffle.", "",
    ]
    rows = []
    for topic in topics:
        cells = []
        for level in LEVELS:
            sub = pooled[(pooled["topic"] == topic) & (pooled["level"] == level)
                         & (pooled["ep_group"] == focus)].set_index("side")["win_share"]
            cells.append(" / ".join(pct(sub.get(s)) for s in ["real", *NULLS]))
        rows.append([topic_label(topic, counts), *cells])
    lines += [markdown_table(["topic", *LEVELS], rows), ""]

    lines += [f"## {focus} per-statement argmax share (`cell`) per model and topic", "",
              "Real runs; the last three rows are the nulls.", ""]
    header = ["model", *(topic_label(t, counts) for t in topics)]
    rows = []
    for model in [ALL_MODELS, *models]:
        sub = table[(table["side"] == "real") & (table["model"] == model)
                    & (table["level"] == "cell") & (table["ep_group"] == focus)]
        sub = sub.set_index("topic")["win_share"]
        label = "all models (pooled)" if model == ALL_MODELS else model_display_name(model)
        rows.append([label, *(pct(sub.get(t)) for t in topics)])
    for null in NULLS:
        sub = pooled[(pooled["side"] == null) & (pooled["level"] == "cell")
                     & (pooled["ep_group"] == focus)].set_index("topic")["win_share"]
        rows.append([SIDE_LABELS[null], *(pct(sub.get(t)) for t in topics)])
    lines += [markdown_table(header, rows), ""]

    header = ["group", *(SIDE_LABELS[s] for s in SIDES)]
    for topic in topic_rows:
        lines += [f"## {topic_label(topic, counts)}", ""]
        for level in LEVELS:
            sub = pooled[(pooled["topic"] == topic) & (pooled["level"] == level)]
            sub = sub.set_index(["side", "ep_group"])
            rows = [[g, *(f"{pct(sub.loc[(s, g), 'win_share'])} "
                          f"({sub.loc[(s, g), 'mean_agreement']:.2f})" for s in SIDES)]
                    for g in groups]
            lines += [f"Level `{level}`, win share (mean agreement):", "",
                      markdown_table(header, rows), ""]
    return "\n".join(lines)


def plot_topics(table, groups, topic_rows, out_path, source):
    counts = {topic: len(rows) for topic, rows in topic_rows.items()}
    topics = [WHOLE, *topic_rows]
    pooled = table[table["model"] == ALL_MODELS]
    fig, axes = plt.subplots(len(topics), len(LEVELS),
                             figsize=(4.0 * len(LEVELS), 2.4 * len(topics)),
                             sharex=True, sharey=True)
    x = np.arange(len(groups))
    width = 0.8 / len(SIDES)
    for ti, topic in enumerate(topics):
        for li, level in enumerate(LEVELS):
            ax = axes[ti, li]
            sub = pooled[(pooled["topic"] == topic) & (pooled["level"] == level)]
            sub = sub.set_index(["side", "ep_group"])["win_share"]
            for si, side in enumerate(SIDES):
                values = 100 * np.array([sub.get((side, g), np.nan) for g in groups])
                offsets = x - 0.4 + width * (si + 0.5)
                ax.bar(offsets, values, width=width * 0.9, color=SIDE_COLORS[side],
                       edgecolor="white", linewidth=0.8)
                if side == "real":
                    for xo, v in zip(offsets, values):
                        if np.isfinite(v):
                            ax.text(xo, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                                    fontsize=6.5, color="#333333")
            if ti == 0:
                ax.set_title(LEVEL_TITLES[level], fontsize=9)
            if li == 0:
                ax.set_ylabel(f"{topic_label(topic, counts)}\nwin share (%)", fontsize=8.5)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#bbbbbb")
            ax.spines["bottom"].set_color("#bbbbbb")
            ax.tick_params(axis="y", labelsize=8, colors="#555555")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8.5)
    axes[0, 0].set_ylim(0, 108)
    fig.legend(handles=[Patch(color=SIDE_COLORS[s], label=SIDE_LABELS[s]) for s in SIDES],
               loc="lower center", ncol=len(SIDES), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"VAA argmax win share per topic: nulls vs real runs ({source} track)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2024"])
    parser.add_argument("--source", default="likert", choices=["likert", "speeches"])
    parser.add_argument("--models", nargs="+", default=MODEL_DIRS)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=["ep-group", "national"])
    parser.add_argument("--no-collapse-ecr-id", action="store_true")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Stance std of the normal null, in stance units where one Likert "
                             "step is 0.5.")
    parser.add_argument("--focus", default=None,
                        help="Group for the overview tables (default: S&D, or its cluster).")
    parser.add_argument("--clusters", default=None, choices=["mean", "parties"],
                        help="Score against the k=4 clusters with this pooling instead of "
                             "the EP groups (see vaa_null_model_clusters.py).")
    parser.add_argument("--weight", default="equal", choices=["equal", "seats"],
                        help="Party weights in the cluster mean (--clusters mean only).")
    parser.add_argument("--out-dir", default=None,
                        help="default: data/<dataset>_results/tables/null_model")
    args = parser.parse_args()
    if args.clusters == "parties" and args.weight == "seats":
        parser.error("--weight seats needs --clusters mean (party rows are averaged unweighted)")
    if args.weight == "seats" and not args.clusters:
        parser.error("--weight only applies with --clusters")

    results_root = Path("data") / f"{args.dataset}_results"
    model_dirs = [results_root / m for m in args.models]
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "tables" / "null_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading real {args.source} runs")
    runs = load_runs(model_dirs, args.source)
    models = [m for m in args.models if m in runs.index.get_level_values("model")]
    if args.clusters:
        from analysis import vaa_null_model_clusters as vc
        parties = vc.cluster_parties(0, vc.K)
        P, party_group, groups = vc.cluster_tensor(
            parties, vc.party_answers(parties, runs.shape[1]), args.clusters, args.weight)
        groups = list(groups)
        args.focus = args.focus or vc.DEFAULT_FOCUS
        # render_summary prints these two in its "Positions:" line.
        args.positions = (f"k={vc.K} clusters, {args.weight}-weighted national-party mean"
                          if args.clusters == "mean" else f"k={vc.K} clusters, member parties as rows")
        args.no_collapse_ecr_id = False
        stem = f"null_model_topics_clusters_{args.clusters}_{args.weight}_{args.source}"
    else:
        P, party_group, groups = positions_tensor(args.positions, not args.no_collapse_ecr_id,
                                                  runs.shape[1])
        args.focus = args.focus or "S&D"
        stem = f"null_model_topics_{args.source}"
    topic_rows = {axis: rows for axis, rows in
                  statement_rows_per_axis(args.dataset, runs.shape[1]).items() if len(rows)}
    sigma = args.sigma
    paraphrases = int(runs.groupby(level=["model", "framing", "language"]).size().median())
    runs_per_model = int(runs.groupby(level="model").size().median())
    print(f"  sigma = {sigma:.3f}, {paraphrases} paraphrases per language, "
          f"{runs_per_model} runs per model, topics: "
          + ", ".join(f"{t} ({len(r)})" for t, r in topic_rows.items()))

    # The draws are made once over all 30 statements and sliced per topic, so every topic
    # sees the same draws, and the LLM-shuffle pools stay per statement.
    print(f"Scoring {args.draws} draws per null and level")
    null = null_levels(make_generators(runs, sigma, rng), args.draws, paraphrases, runs_per_model)
    real = real_levels(runs)
    table = score_topics(real, null, P, party_group, groups, args.source, topic_rows)

    table.to_csv(out_dir / f"{stem}_win_shares.csv", index=False)
    summary = render_summary(table, groups, models, topic_rows, args, sigma)
    (out_dir / f"{stem}_summary.md").write_text(summary, encoding="utf-8")
    plot_topics(table, groups, topic_rows, out_dir / f"{stem}_win_shares.png", args.source)
    print()
    print(summary.split("\n## Ukraine")[0] if "\n## Ukraine" in summary else summary)
    print(f"\nWrote {out_dir / (stem + '_win_shares.csv')}, {stem}_summary.md, "
          f"{stem}_win_shares.png")


if __name__ == "__main__":
    main()
