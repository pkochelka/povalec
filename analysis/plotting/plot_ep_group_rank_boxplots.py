"""Box plots of EP-group rank across methods, languages, prompts and models.

Every method scores the EP groups on its own scale -- VAA agreement lives in
[0, 1] and is compressed into a narrow band, classifier probability is dominated
by the class prior -- so the raw scores are not comparable across methods. What
*is* comparable is the ordering: inside one run the six groups get ranks 1..6,
rank 1 being the group the run placed closest to the model.

A run (a "cell") is one (model, method, framing, language, paraphrase) tuple.
Each box therefore answers: if I vary this one factor and hold nothing else
fixed, how much does the group's rank move? A tight box means the finding
survives that factor; a box spanning the whole scale means the finding is an
artefact of it.

Methods
    vaa-likert          VAA agreement from the model's own Likert choice
    vaa-likert-judge    VAA agreement from the LLM-judged reason texts
    vaa-speeches        VAA agreement from the cross-encoder-scored open answers
    vaa-speeches-judge  VAA agreement from the LLM-judged open-ended answers
    clf-reasons         mmBERT party-classifier mean probability on reasons
    clf-speeches        mmBERT party-classifier mean probability on open answers

Run from the repo root:
    python analysis/plotting/plot_ep_group_rank_boxplots.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.core import (
    CELL_KEYS,
    DEFAULT_POSITIONS,
    FRAMING_ORDER,
    METHODS,
    POSITION_CHOICES,
    count,
    load_scores,
    party_sort_key,
    positions_frame,
)
from utils import FALLBACK_PARTY_COLOR, PARTY_COLORS

# Figure-only settings below. Reading the results tree -- the filename grammar, the
# stance loaders, the six scoring methods -- lives in analysis.core; this module is
# the ranking and the figures built on top of it.

FACTORS = ["method", "track", "scoring", "framing", "paraphrase", "language", "model"]
FACTOR_TITLE = {
    "method": "method",
    "track": "answer track",
    "scoring": "scoring pipeline",
    "framing": "prompt framing",
    "paraphrase": "prompt paraphrase",
    "language": "language",
    "model": "model",
}
# Paraphrase v3 of the Likert prompt and v3 of the open-ended prompt are unrelated
# prompts, so that figure is only readable as "how much does swapping the wording
# move the rank", never as a comparison between two tracks at the same index. The
# figures carry no titles, so the caveat is printed for the caption instead.
FACTOR_NOTE = {
    "paraphrase": "paraphrase indices are per-track wordings, not aligned across tracks",
}

# Fixed order, never cycled; validated for CVD separation and lightness band.
FACTOR_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                 "#CC79A7", "#56B4E9", "#5D3A9B", "#A6761D"]
FACET_THRESHOLD = 8

GRID_STYLE = dict(axis="y", linestyle=":", alpha=0.35, zorder=0)
MEDIAN_STYLE = dict(color="black", linewidth=1.4)
MEAN_STYLE = dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                  markeredgewidth=0.8, markersize=4)
FLIER_STYLE = dict(marker=".", markersize=2, markerfacecolor="#666666",
                   markeredgecolor="none", alpha=0.35)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def add_ranks(scores):
    """Rank 1 = highest score inside the cell. Cells that are missing a group are
    dropped: their ranks would sit on a shorter scale and silently bias the box."""
    groups_per_method = scores.groupby("method")["ep_group"].apply(lambda s: set(s.unique()))
    common = set.intersection(*groups_per_method) if len(groups_per_method) else set()
    dropped = set().union(*groups_per_method) - common
    if dropped:
        print(f"\nGroups missing from at least one method, excluded: {sorted(dropped)}")
    scores = scores[scores["ep_group"].isin(common)].copy()

    complete = scores.groupby(CELL_KEYS)["ep_group"].transform("nunique") == len(common)
    if (~complete).any():
        print(f"Dropped {int((~complete).sum() / max(len(common), 1))} incomplete cells.")
    scores = scores[complete].copy()

    scores["rank"] = scores.groupby(CELL_KEYS)["score"].rank(ascending=False, method="average")
    print(f"\n{len(scores)} ranked rows over "
          f"{scores.groupby(CELL_KEYS).ngroups} cells, {len(common)} EP groups.")
    return scores, sorted(common, key=party_sort_key)


def level_order(ranks, factor):
    levels = ranks[factor].dropna().unique().tolist()
    if factor == "framing":
        return [f for f in FRAMING_ORDER if f in levels]
    if factor == "method":
        return [m for m in METHODS if m in levels]
    if factor == "paraphrase":
        return sorted(levels)
    return sorted(levels, key=str)


def level_label(factor, level):
    return f"v{level}" if factor == "paraphrase" else str(level)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def style_boxes(parts, facecolors):
    for box, color in zip(parts["boxes"], facecolors):
        box.set_facecolor(color)
        box.set_edgecolor("black")
        box.set_linewidth(0.7)
        box.set_alpha(0.85)
    for key in ("whiskers", "caps"):
        for artist in parts[key]:
            artist.set_color("#444444")
            artist.set_linewidth(0.9)


def setup_rank_axis(ax, n_groups, ylabel=True):
    ax.set_ylim(n_groups + 0.6, 0.4)  # inverted: rank 1 (closest) on top
    ax.set_yticks(range(1, n_groups + 1))
    ax.set_yticklabels([f"{r}" for r in range(1, n_groups + 1)], fontsize=8)
    if ylabel:
        ax.set_ylabel(f"rank  (1 = closest  ·  {n_groups} = furthest)", fontsize=9)
    ax.grid(**GRID_STYLE)
    ax.set_axisbelow(True)


def draw_group_boxes(ax, ranks, parties, positions, width, facecolors):
    datasets, kept_positions, kept_colors = [], [], []
    for party, position, color in zip(parties, positions, facecolors):
        values = ranks.loc[ranks["ep_group"] == party, "rank"].to_numpy()
        if values.size:
            datasets.append(values)
            kept_positions.append(position)
            kept_colors.append(color)
    if not datasets:
        return
    parts = ax.boxplot(datasets, positions=kept_positions, widths=width,
                       patch_artist=True, showmeans=True, meanprops=MEAN_STYLE,
                       medianprops=MEDIAN_STYLE, flierprops=FLIER_STYLE)
    style_boxes(parts, kept_colors)


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def subtitle(ranks):
    return "  ·  ".join([
        count(ranks.groupby(CELL_KEYS).ngroups, "run"),
        count(ranks["model"].nunique(), "model"),
        count(ranks["method"].nunique(), "method"),
        count(ranks["language"].nunique(), "language"),
        count(ranks["framing"].nunique(), "framing"),
    ])


def plot_pooled(ranks, parties, output_path):
    """The headline: one box per group over every run there is. The share of runs
    that put a group first is printed next to it -- a low median with a low top-1
    share is a group that is consistently second, not an occasional winner."""
    x = np.arange(len(parties))
    fig, ax = plt.subplots(figsize=(max(7.5, len(parties) * 1.3), 5.4))
    draw_group_boxes(ax, ranks, parties, x, 0.6, [color_for(p) for p in parties])

    top1 = ranks[ranks["rank"] == 1]["ep_group"].value_counts()
    n_cells = ranks.groupby(CELL_KEYS).ngroups
    for position, party in zip(x, parties):
        share = 100.0 * top1.get(party, 0) / max(n_cells, 1)
        ax.text(position, 0.48, f"{share:.0f}% top-1", ha="center", va="top", fontsize=7.5,
                color="#333333")

    setup_rank_axis(ax, len(parties))
    ax.set_xticks(x, parties, rotation=15, ha="right", fontsize=9)
    save_figure(fig, output_path)


def plot_grouped(ranks, parties, factor, output_path):
    levels = level_order(ranks, factor)
    if len(levels) < 2:
        return
    x = np.arange(len(parties))
    group_width = 0.82
    box_width = group_width / len(levels)
    colors = [FACTOR_COLORS[i % len(FACTOR_COLORS)] for i in range(len(levels))]

    fig, ax = plt.subplots(figsize=(max(9, len(parties) * len(levels) * 0.45), 6.4))
    for index, (level, color) in enumerate(zip(levels, colors)):
        offsets = x - group_width / 2 + box_width * (index + 0.5)
        draw_group_boxes(ax, ranks[ranks[factor] == level], parties, offsets,
                         box_width * 0.78, [color] * len(parties))

    setup_rank_axis(ax, len(parties))
    ax.set_xticks(x, parties, rotation=15, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(parties) - 0.4)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.6,
                             label=level_label(factor, l))
                       for l, c in zip(levels, colors)],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              title=FACTOR_TITLE[factor], title_fontsize=8.5, frameon=False)
    save_figure(fig, output_path)


def plot_facets(ranks, parties, factor, output_path):
    levels = level_order(ranks, factor)
    if len(levels) < 2:
        return
    n_columns = min(4, len(levels))
    n_rows = int(np.ceil(len(levels) / n_columns))
    x = np.arange(len(parties))
    facecolors = [color_for(p) for p in parties]

    fig, axes = plt.subplots(n_rows, n_columns, sharey=True,
                             figsize=(n_columns * 3.4, n_rows * 2.7), squeeze=False)
    for index, level in enumerate(levels):
        ax = axes[index // n_columns][index % n_columns]
        draw_group_boxes(ax, ranks[ranks[factor] == level], parties, x, 0.62, facecolors)
        setup_rank_axis(ax, len(parties), ylabel=False)
        ax.set_xticks(x, parties, rotation=60, ha="right", fontsize=7)
        ax.set_title(level_label(factor, level), fontsize=9)
    for index in range(len(levels), n_rows * n_columns):
        axes[index // n_columns][index % n_columns].axis("off")

    fig.supylabel(f"rank  (1 = closest  ·  {len(parties)} = furthest)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_path)


def color_for(party):
    return PARTY_COLORS.get(party, FALLBACK_PARTY_COLOR)


# --------------------------------------------------------------------------- #
# summary table
# --------------------------------------------------------------------------- #

def summarise(ranks, parties):
    """Per (factor, level, group): where the rank sits and how far it travels.
    rank_iqr is the consistency number -- 0 means the factor never moved it."""
    rows = []
    for factor in ["overall", *FACTORS]:
        scoped = ranks.assign(level="all") if factor == "overall" else ranks.rename(columns={factor: "level"})
        for (level, party), group in scoped.groupby(["level", "ep_group"]):
            values = group["rank"].to_numpy()
            rows.append({
                "factor": factor,
                "level": level,
                "ep_group": party,
                "n_runs": len(values),
                "rank_mean": values.mean(),
                "rank_median": np.median(values),
                "rank_iqr": np.percentile(values, 75) - np.percentile(values, 25),
                "rank_min": values.min(),
                "rank_max": values.max(),
                "top1_share": float((values == 1).mean()),
            })
    summary = pd.DataFrame(rows)
    summary["ep_group"] = pd.Categorical(summary["ep_group"], parties, ordered=True)
    return summary.sort_values(["factor", "level", "ep_group"])


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None,
                        help="Comma-separated model dirs; default uses every model.")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help=f"Comma-separated subset of: {', '.join(METHODS)}")
    parser.add_argument("--factors", default=",".join(FACTORS),
                        help=f"Which factors to draw a figure for: {', '.join(FACTORS)}")
    parser.add_argument("--layout", default="auto", choices=["auto", "grouped", "facets"],
                        help=f"auto: one grouped figure up to {FACET_THRESHOLD} levels, "
                             "small multiples above that.")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Whose euandi answers stand for an EP group, as in evaluate_euandi.")
    parser.add_argument("--no-collapse-ecr-id", action="store_true",
                        help="Keep ECR and ID apart. The existing VAA/classifier outputs are "
                             "collapsed, so this only makes sense for a fresh, uncollapsed run.")
    parser.add_argument("--out-dir", default=None,
                        help="Default: data/<dataset>_results/plots/ep_group_ranks")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Available: {list(METHODS)}")
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    unknown = [f for f in factors if f not in FACTORS]
    if unknown:
        raise SystemExit(f"Unknown factor(s): {unknown}. Available: {FACTORS}")

    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir} (run from the repo root)")
    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
    if args.model:
        wanted = {m.strip() for m in args.model.split(",")}
        model_dirs = [d for d in model_dirs if d.name in wanted]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    party_df = positions_frame(args.positions, not args.no_collapse_ecr_id)
    scores = load_scores(model_dirs, methods, party_df)
    ranks, parties = add_ranks(scores)

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "plots" / "ep_group_ranks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFigures (for the captions: {subtitle(ranks)}):")
    plot_pooled(ranks, parties, out_dir / "rank_pooled.png")
    for factor in factors:
        levels = level_order(ranks, factor)
        grouped = (args.layout == "grouped"
                   or (args.layout == "auto" and len(levels) <= FACET_THRESHOLD))
        draw = plot_grouped if grouped else plot_facets
        draw(ranks, parties, factor, out_dir / f"rank_by_{factor}.png")
        if (note := FACTOR_NOTE.get(factor)):
            print(f"    note for rank_by_{factor}: {note}")

    ranks.to_csv(out_dir / "ep_group_ranks_long.csv", index=False)
    print(f"  Saved {out_dir / 'ep_group_ranks_long.csv'}")
    summary = summarise(ranks, parties)
    summary.to_csv(out_dir / "ep_group_rank_summary.csv", index=False)
    print(f"  Saved {out_dir / 'ep_group_rank_summary.csv'}")

    print("\nPooled rank per EP group (rank_iqr 0 = every run agreed):")
    overall = summary[summary["factor"] == "overall"].set_index("ep_group")
    print(overall[["n_runs", "rank_mean", "rank_median", "rank_iqr", "top1_share"]].round(2))
    print("\nDone.")


if __name__ == "__main__":
    main()
