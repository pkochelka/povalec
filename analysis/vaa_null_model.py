#!/usr/bin/env python3
"""Geometric null model for the "models converge on S&D" VAA finding.

A reviewer's objection: S&D sits near the center of the EP-group positions on most axes,
so under a "closest group wins" rule a central group wins whenever an answer vector lands
anywhere near the middle. "Models converge on S&D" might then be nothing more than "models
are not extreme, and S&D is the group nearest the center".

This script tests how much of the headline result that geometry alone explains. It
generates answer vectors that carry no political signal, scores them with exactly the VAA
agreement the real runs are scored with (1 - |position - stance| / 2, averaged over the
statements a group's party answered, ECR and ID pooled), and reports how often each group
comes out on top. Three nulls:

  uniform      every statement iid uniform over the five Likert points.
  normal       stance ~ N(0, sigma), rounded to the nearest Likert point; sigma defaults to
               0.5 (one Likert step), so the null is noisy but has no lean. --sigma changes
               it; the pooled std of the real answers is printed for comparison.
  llm-shuffle  every statement drawn from the pooled histogram of what the models actually
               answered to THAT statement (all models, framings, languages, paraphrases),
               independently per statement. It keeps each statement's average answer level
               and destroys the coherence across statements, so it separates "a coherent
               worldview" from "the right average answer per statement".

Each null and the real runs are compared at four aggregation levels, because the geometry
argument bites harder the more averaging is done:

  run        one 30-vector per (model, framing, language, paraphrase); the null draws one
             vector per draw.
  language   mean over the paraphrases per (model, framing, language) -- the vectors the
             production vaa_*.csv files score; the null averages the same number of draws.
  model      mean over every run of a model (plus the grand mean over all models); the null
             averages as many draws as a model has runs.
  cell       per-statement argmax share with ties split, over the paraphrase-averaged
             (language-level) vectors -- exactly the argmax share the paper's figures
             report (plot_argmax_shares.vaa_argmax_share, framings pooled).

How to read the output
  If S&D wins, say, 20% of null vectors but 70%+ of the real runs, the S&D result is signal
  above the geometric baseline. If S&D already wins most null vectors -- in particular most
  llm-shuffle vectors -- then a large part of the finding is structural: the models lean
  center-left on average and S&D's centrality amplifies the argmax, and the paper should
  report agreement margins over the runner-up rather than the argmax alone. The `model`
  level of the uniform and normal nulls shows directly which group is nearest the center of
  the position space, which is the reviewer's stated concern.

Outputs (under --out-dir, inside the gitignored data/ tree):
  null_model_<source>_win_shares.csv   tidy long table, one row per (side, model, level, group)
  null_model_<source>_summary.md       the tables printed to the console
  null_model_<source>_win_shares.png   win share per group, one panel per level

Usage, from the repository root with the package installed:
  python analysis/vaa_null_model.py                       # direct Likert track, 10 000 draws
  python analysis/vaa_null_model.py --source speeches     # indirect (cross-encoder) track
  python analysis/vaa_null_model.py --self-check --draws 200
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
from matplotlib.patches import Patch

from analysis.analyze_all import MODEL_DIRS
from analysis.core import (
    DEFAULT_POSITIONS,
    RESPONSES_CSV,
    SPEECHES_SCORED_CSV,
    find_csvs,
    likert_stances,
    load_party_positions,
    model_display_name,
    positions_frame,
    positions_path,
    speech_stances,
    vaa_scores,
)
from utils import PARTY_DISPLAY_ORDER, configure_stdout, likert_to_stance, stance_to_likert

configure_stdout()

LIKERT_POINTS = np.arange(1, 6)
LIKERT_STANCES = likert_to_stance(LIKERT_POINTS).astype(float)   # 1, .5, 0, -.5, -1

NULLS = ["uniform", "normal", "llm-shuffle"]
SIDES = [*NULLS, "real"]
LEVELS = ["run", "language", "cell", "model"]
LEVEL_TITLES = {
    "run": "individual runs",
    "language": "mean over paraphrases (one vector per language)",
    "cell": "per-statement argmax share (paraphrase-averaged)",
    "model": "mean over all runs of a model",
}
ALL_MODELS = "all"
DEFAULT_SIGMA = 0.5   # one Likert step, in stance units

# One fixed hue per side, so a bar is the same side in every panel.
SIDE_COLORS = {
    "uniform": "#eb6834",
    "normal": "#1baf7a",
    "llm-shuffle": "#eda100",
    "real": "#2a78d6",
}
SIDE_LABELS = {
    "uniform": "uniform null",
    "normal": "normal null (mean 0)",
    "llm-shuffle": "LLM-shuffle null",
    "real": "real LLM runs",
}

# How many averaged vectors are generated at once: the model level averages runs_per_model
# draws per vector, so a chunk of 200 vectors is 200 x 336 x 30 floats, which is small.
CHUNK = 200


# --------------------------------------------------------------------------- #
# real runs
# --------------------------------------------------------------------------- #

def load_runs(model_dirs, source):
    """One row per run (model, framing, language, paraphrase), one column per statement,
    stance in [-1, 1] with the negated framing already re-oriented. NaN where the indirect
    track has no stance (refusal); the direct track has none, since a missing choice is
    scored as neutral by the loader, exactly as evaluate_euandi does."""
    pattern, loader = ((RESPONSES_CSV, likert_stances) if source == "likert"
                       else (SPEECHES_SCORED_CSV, speech_stances))
    frames = []
    for model_dir in model_dirs:
        found = find_csvs(model_dir, pattern)
        if not found:
            print(f"  {model_dir.name}: no {source} files, skipping.")
            continue
        for framing, path in found.items():
            long = loader(path, framing)
            if not long.empty:
                frames.append(long.assign(model=model_dir.name, framing=framing))
    if not frames:
        raise SystemExit(f"No {source} runs found under {model_dirs[0].parent}.")
    long = pd.concat(frames, ignore_index=True)
    wide = (long.set_index(["model", "framing", "language", "paraphrase", "statement_idx"])
                ["stance"].unstack("statement_idx").sort_index())
    wide = wide.dropna(how="all")
    print(f"  {len(wide)} runs x {wide.shape[1]} statements from "
          f"{wide.index.get_level_values('model').nunique()} models")
    return wide


# --------------------------------------------------------------------------- #
# positions and scoring
# --------------------------------------------------------------------------- #

def positions_tensor(positions, collapse_ecr_id, statement_count):
    """(P, party_group, groups): P[party, statement] with NaN for an abstention, the group
    each party row stands for, and the groups in display order. Parties stay as rows so a
    collapsed ECR+ID is the pooled mean over both parties' rows, as agreement_by_ep_group
    computes it."""
    party_df = load_party_positions(positions_path(positions), positions)
    if collapse_ecr_id:
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    party_df = party_df.dropna(subset=["ep_group"])
    P = (party_df.pivot(index="short_name", columns="statement_idx", values="normalized_answer")
                 .reindex(columns=range(statement_count)).astype(float))
    party_group = party_df.drop_duplicates("short_name").set_index("short_name")["ep_group"]
    party_group = party_group.reindex(P.index).to_numpy()
    groups = [g for g in PARTY_DISPLAY_ORDER if g in set(party_group)]
    return P.to_numpy(), party_group, groups


def statement_agreement(S, P):
    """A[n, party, statement] = 1 - |position - stance| / 2; NaN where either is missing."""
    return 1 - np.abs(P[None, :, :] - S[:, None, :]) / 2


def group_scores(S, P, party_group, groups):
    """Mean agreement per group, (N, G): the mean over every (party, statement) row of the
    group with a position and a stance -- the production VAA score."""
    A = statement_agreement(S, P)
    N = A.shape[0]
    out = np.full((N, len(groups)), np.nan)
    with np.errstate(invalid="ignore"):
        for gi, group in enumerate(groups):
            rows = np.flatnonzero(party_group == group)
            out[:, gi] = np.nanmean(A[:, rows, :].reshape(N, -1), axis=1)
    return out


def cell_scores(S, P, party_group, groups):
    """Mean agreement per group and statement, (N, G, statements) -- the cells the
    argmax-share figures are built from."""
    A = statement_agreement(S, P)
    out = np.full((A.shape[0], len(groups), A.shape[2]), np.nan)
    # A cell where every party of the group abstained is an empty mean, and NaN is right.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        for gi, group in enumerate(groups):
            rows = np.flatnonzero(party_group == group)
            out[:, gi, :] = np.nanmean(A[:, rows, :], axis=1)
    return out


def win_credit(scores):
    """Credit per row summing to 1 over the group axis (axis 1): 1 to the top group, split
    equally between exact ties. Group positions are coarse, so ties are routine. A row with
    no score at all gets no credit."""
    finite = np.where(np.isnan(scores), -np.inf, scores)
    top = finite.max(axis=1, keepdims=True)
    is_top = (finite == top) & np.isfinite(top)
    winners = is_top.sum(axis=1, keepdims=True)
    return np.where(winners > 0, is_top / np.maximum(winners, 1), 0.0)


def runner_up_margin(scores):
    """Top score minus second-best score per row, (N,)."""
    ordered = np.sort(np.where(np.isnan(scores), -np.inf, scores), axis=1)
    scorable = np.isfinite(ordered[:, -2])
    margin = np.full(len(ordered), np.nan)
    margin[scorable] = ordered[scorable, -1] - ordered[scorable, -2]
    return margin


def summarise(scores, groups):
    """{group: (win_share, mean_agreement, mean_margin_when_winner)} over the rows."""
    credit = win_credit(scores)
    margin = runner_up_margin(scores)
    records = {}
    for gi, group in enumerate(groups):
        won = credit[:, gi] > 0
        records[group] = (
            float(credit[:, gi].mean()),
            float(np.nanmean(scores[:, gi])),
            float(np.average(margin[won], weights=credit[won, gi])) if won.any() else np.nan,
        )
    return records


def summarise_cells(cells, groups):
    """Same three numbers from (N, G, statements) cells: the argmax share is the mean
    credit over every (vector, statement) cell."""
    N, G, T = cells.shape
    flat = np.moveaxis(cells, 1, 2).reshape(N * T, G)
    return summarise(flat, groups)


# --------------------------------------------------------------------------- #
# null generators
# --------------------------------------------------------------------------- #

def make_generators(runs, sigma, rng):
    """{name: draw(n) -> (n, statements) stance matrix}."""
    statements = runs.shape[1]
    values = runs.to_numpy(dtype=float)
    pools = [column[~np.isnan(column)] for column in values.T]

    def uniform(n):
        return rng.choice(LIKERT_STANCES, size=(n, statements))

    def normal(n):
        z = rng.normal(0.0, sigma, size=(n, statements))
        likert = np.clip(np.rint(stance_to_likert(z)), 1, 5)
        return likert_to_stance(likert)

    def llm_shuffle(n):
        out = np.empty((n, statements))
        for j, pool in enumerate(pools):
            out[:, j] = rng.choice(pool, size=n)
        return out

    return {"uniform": uniform, "normal": normal, "llm-shuffle": llm_shuffle}


def averaged_draws(draw, n, k):
    """n vectors, each the mean of k draws -- the null analogue of averaging k runs."""
    if k == 1:
        return draw(n)
    out = np.empty((n, draw(1).shape[1]))
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        out[start:stop] = draw((stop - start) * k).reshape(stop - start, k, -1).mean(axis=1)
    return out


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #

def real_levels(runs):
    """{level: {model: (N, statements) matrix}} for the real side, with ALL_MODELS pooled.
    Averages skip NaN, as pandas' mean does in the production scorers."""
    by_model = {ALL_MODELS: runs}
    by_model.update({model: frame for model, frame in runs.groupby(level="model")})
    levels = {}
    levels["run"] = {m: f.to_numpy(dtype=float) for m, f in by_model.items()}
    levels["language"] = {
        m: f.groupby(level=["model", "framing", "language"]).mean().to_numpy(dtype=float)
        for m, f in by_model.items()
    }
    # The paper's argmax share is taken per (statement, language) on the paraphrase-averaged
    # stance (plot_argmax_shares.cell_agreement), so the cells come off the language level.
    levels["cell"] = levels["language"]
    model_means = runs.groupby(level="model").mean()
    levels["model"] = {ALL_MODELS: np.vstack([model_means.to_numpy(dtype=float),
                                              runs.mean().to_numpy(dtype=float)[None, :]])}
    levels["model"].update({m: model_means.loc[[m]].to_numpy(dtype=float)
                            for m in model_means.index})
    return levels


def null_levels(generators, draws, paraphrases, runs_per_model):
    """{level: {null: (draws, statements) matrix}}, matching real_levels' averaging."""
    k_for_level = {"run": 1, "language": paraphrases, "model": runs_per_model}
    levels = {}
    for level in LEVELS:
        if level == "cell":
            levels[level] = levels["language"]
            continue
        levels[level] = {name: averaged_draws(draw, draws, k_for_level[level])
                         for name, draw in generators.items()}
    return levels


def score_everything(real, null, P, party_group, groups, source):
    records = []

    def add(side, model, level, S):
        if level == "cell":
            summary = summarise_cells(cell_scores(S, P, party_group, groups), groups)
        else:
            summary = summarise(group_scores(S, P, party_group, groups), groups)
        for group, (share, agreement, margin) in summary.items():
            records.append(dict(source=source, side=side, model=model, level=level,
                                ep_group=group, n_vectors=len(S), win_share=share,
                                mean_agreement=agreement, mean_margin_when_winner=margin))

    for level in LEVELS:
        for name, S in null[level].items():
            add(name, ALL_MODELS, level, S)
        for model, S in real[level].items():
            add("real", model, level, S)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def markdown_table(header, rows):
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]

    def line(row):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"

    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(row) for row in rows)])


def pct(value):
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def render_summary(table, groups, models, args, sigma, runs, paraphrases, runs_per_model):
    pooled = table[table["model"] == ALL_MODELS]
    collapse = "separate" if args.no_collapse_ecr_id else "collapsed"
    lines = [
        f"# VAA null model -- {args.source} track", "",
        f"Positions: `{args.positions}`, ECR+ID {collapse}. Real runs: {len(runs)} from "
        f"{len(models)} models ({runs_per_model} per model, {paraphrases} paraphrases per "
        f"language). Null draws: {args.draws} per null and level, seed {args.seed}. Normal "
        f"null sigma = {sigma:.3f} (pooled std of the real stances: "
        f"{float(np.nanstd(runs.to_numpy(dtype=float))):.3f}).",
        "",
        "Win share = share of vectors (or cells) the group has the highest agreement on, ties "
        "split. Margin = winner's agreement minus the runner-up's, averaged over the vectors "
        "the group won. Levels: " + "; ".join(f"`{l}` = {LEVEL_TITLES[l]}" for l in LEVELS) + ".",
        "",
    ]
    header = ["group", *(SIDE_LABELS[s] for s in SIDES)]
    for level in LEVELS:
        sub = pooled[pooled["level"] == level].set_index(["side", "ep_group"])
        n = {side: int(sub.loc[side]["n_vectors"].iloc[0]) for side in SIDES}
        lines += [f"## Level `{level}`: {LEVEL_TITLES[level]}", "",
                  "Vectors: " + ", ".join(f"{SIDE_LABELS[s]} {n[s]}" for s in SIDES) + ".", "",
                  "Win share:", ""]
        rows = [[g, *(pct(sub.loc[(s, g), "win_share"]) for s in SIDES)] for g in groups]
        lines += [markdown_table(header, rows), ""]
        lines += ["Mean agreement / mean margin over the runner-up when the group wins:", ""]
        rows = [[g, *(f"{sub.loc[(s, g), 'mean_agreement']:.3f} / "
                      f"{sub.loc[(s, g), 'mean_margin_when_winner']:.3f}" for s in SIDES)]
                for g in groups]
        lines += [markdown_table(header, rows), ""]

    focus = args.focus
    lines += [f"## {focus} win share per model (real runs) against the nulls", ""]
    rows = []
    for null in NULLS:
        sub = pooled[(pooled["side"] == null) & (pooled["ep_group"] == focus)]
        sub = sub.set_index("level")["win_share"]
        rows.append([SIDE_LABELS[null], *(pct(sub.get(l)) for l in LEVELS)])
    for model in [ALL_MODELS, *models]:
        sub = table[(table["side"] == "real") & (table["model"] == model)
                    & (table["ep_group"] == focus)].set_index("level")["win_share"]
        label = "all models (pooled)" if model == ALL_MODELS else model_display_name(model)
        rows.append([label, *(pct(sub.get(l)) for l in LEVELS)])
    lines += [markdown_table(["model", *LEVELS], rows), "",
              f"At the `model` level a real model contributes one vector, so its {focus} share "
              "is 0% or 100% (ties split); the pooled row also counts the grand-mean vector.",
              ""]
    return "\n".join(lines)


def plot_win_shares(table, groups, out_path, source):
    pooled = table[table["model"] == ALL_MODELS]
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(4.2 * len(LEVELS), 4.2), sharey=True)
    x = np.arange(len(groups))
    width = 0.8 / len(SIDES)
    for ax, level in zip(axes, LEVELS):
        sub = pooled[pooled["level"] == level].set_index(["side", "ep_group"])["win_share"]
        for si, side in enumerate(SIDES):
            values = 100 * np.array([sub.get((side, g), np.nan) for g in groups])
            offsets = x - 0.4 + width * (si + 0.5)
            ax.bar(offsets, values, width=width * 0.9, color=SIDE_COLORS[side],
                   edgecolor="white", linewidth=1)
            if side == "real":
                for xo, v in zip(offsets, values):
                    if np.isfinite(v):
                        ax.text(xo, v + 1, f"{v:.0f}", ha="center", va="bottom",
                                fontsize=7.5, color="#333333")
        ax.set_title(LEVEL_TITLES[level], fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8.5)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(axis="y", labelsize=8.5, colors="#555555")
    axes[0].set_ylabel("Win share (%)", fontsize=9)
    axes[0].set_ylim(0, 105)
    fig.legend(handles=[Patch(color=SIDE_COLORS[s], label=SIDE_LABELS[s]) for s in SIDES],
               loc="lower center", ncol=len(SIDES), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Which EP group wins the VAA argmax: nulls vs real runs ({source} track)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# self-check against the production scorers
# --------------------------------------------------------------------------- #

def self_check(runs, P, party_group, groups, args, model_dirs):
    """The vectorised scorer must reproduce analysis.core.results.vaa_scores exactly."""
    party_df = positions_frame(args.positions, not args.no_collapse_ecr_id)
    model, framing = runs.index[0][:2]
    model_dir = next(d for d in model_dirs if d.name == model)
    pattern, loader = ((RESPONSES_CSV, likert_stances) if args.source == "likert"
                       else (SPEECHES_SCORED_CSV, speech_stances))
    reference = vaa_scores(party_df, loader(find_csvs(model_dir, pattern)[framing], framing))
    reference = reference.set_index(["language", "paraphrase", "ep_group"])["score"]
    sample = runs.loc[(model, framing)]
    ours = group_scores(sample.to_numpy(dtype=float), P, party_group, groups)
    expected = np.array([[reference.get((lang, para, g), np.nan) for g in groups]
                         for lang, para in sample.index])
    if not np.allclose(ours, expected, equal_nan=True, atol=1e-12):
        raise SystemExit("self-check FAILED: group scores differ from vaa_scores.")
    print(f"  self-check OK: {len(sample)} runs of {model}/{framing} match vaa_scores.")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2024"])
    parser.add_argument("--source", default="likert", choices=["likert", "speeches"],
                        help="likert: the direct track's Likert choices; speeches: the "
                             "indirect track's cross-encoder stances.")
    parser.add_argument("--models", nargs="+", default=MODEL_DIRS,
                        help="result directories under data/<dataset>_results/")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=["ep-group", "national"])
    parser.add_argument("--no-collapse-ecr-id", action="store_true",
                        help="Keep ECR and ID as separate groups (production collapses them).")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Stance std of the normal null, in stance units where one Likert "
                             "step is 0.5.")
    parser.add_argument("--focus", default="S&D", help="Group for the per-model table.")
    parser.add_argument("--out-dir", default=None,
                        help="default: data/<dataset>_results/tables/null_model")
    parser.add_argument("--self-check", action="store_true",
                        help="Verify the scorer against analysis.core.results.vaa_scores.")
    args = parser.parse_args()

    results_root = Path("data") / f"{args.dataset}_results"
    model_dirs = [results_root / m for m in args.models]
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "tables" / "null_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading real {args.source} runs")
    runs = load_runs(model_dirs, args.source)
    models = [m for m in args.models if m in runs.index.get_level_values("model")]
    P, party_group, groups = positions_tensor(args.positions, not args.no_collapse_ecr_id,
                                              runs.shape[1])
    print(f"  groups: {', '.join(groups)}")
    if args.self_check:
        self_check(runs, P, party_group, groups, args, model_dirs)

    sigma = args.sigma
    real_std = float(np.nanstd(runs.to_numpy(dtype=float)))
    paraphrases = int(runs.groupby(level=["model", "framing", "language"]).size().median())
    runs_per_model = int(runs.groupby(level="model").size().median())
    print(f"  sigma = {sigma:.3f} (pooled std of the real stances: {real_std:.3f}), "
          f"{paraphrases} paraphrases per language, {runs_per_model} runs per model")

    print(f"Scoring {args.draws} draws per null and level")
    generators = make_generators(runs, sigma, rng)
    null = null_levels(generators, args.draws, paraphrases, runs_per_model)
    real = real_levels(runs)
    table = score_everything(real, null, P, party_group, groups, args.source)

    stem = f"null_model_{args.source}"
    table.to_csv(out_dir / f"{stem}_win_shares.csv", index=False)
    summary = render_summary(table, groups, models, args, sigma, runs, paraphrases, runs_per_model)
    (out_dir / f"{stem}_summary.md").write_text(summary, encoding="utf-8")
    plot_win_shares(table, groups, out_dir / f"{stem}_win_shares.png", args.source)
    print()
    print(summary)
    print(f"\nWrote {out_dir / (stem + '_win_shares.csv')}, {stem}_summary.md, "
          f"{stem}_win_shares.png")


if __name__ == "__main__":
    main()
