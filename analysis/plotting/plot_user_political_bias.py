"""Political-bias plots for the EU&I 2024 *users*, drawn the same way as the
per-run LLM plots in plot_political_bias.py so the two can be read side by side.

One user's 30 Likert answers play the role of one LLM run: each answer becomes a
stance in [-1, 1], the stance is projected onto the questionnaire's seven signed
dimensions, and the per-dimension mean is that respondent's position. Outputs are
the Left-Right x Europe compass and the per-dimension violins.

Deliberately left out, so the comparison stays clean:
  * statement weights (w01..w36) -- the LLM has no importance weighting;
  * the opt-in blocks (demographics, thermometer, matching scores) -- unused here,
    and filtering on them would restrict the sample to a self-selected subset;
  * countries whose EU&I locale is not one of our 21 generation languages, i.e.
    the duplicate-language colleges (Austria, Belgium, Cyprus, Luxembourg) and
    Croatia, which has no counterpart language in the runs at all.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.evaluate_euandi import DEFAULT_POSITIONS, POSITION_CHOICES
from analysis.plotting.plot_political_bias import (
    AXIS_FONTSIZE,
    FONTSIZE,
    LEGEND_MARKERSIZE,
    MAIN_LEGEND_LOC,
    draw_party_anchors,
    language_colors,
    load_questionnaire,
    party_anchors,
    setup_compass,
)
from utils import ALL_LANGS

USER_DATASET_PATH = "data/euandi_2024_data/EU&I2024 user dataset.dta"

# Same 30-of-36 statement selection as build_group_positions.STATEMENT_COLUMNS,
# zero-padded to the user dataset's variable names. Position i is questionnaire
# statement_idx i (0-based), which is also row i of the response CSVs.
STATEMENT_COLUMNS = [
    "s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10",
    "s12", "s13", "s15", "s16", "s17", "s18", "s19", "s20", "s22", "s23",
    "s24", "s25", "s27", "s28", "s29", "s30", "s31", "s34", "s35", "s36",
]

# EU&I country -> the language code the statements were served in, restricted to
# the codes we generate answers for. Ireland and Malta both ran in English, so
# they share the "en" run. Countries absent from this map are dropped.
COUNTRY_TO_LANG = {
    "Bulgaria": "bg", "Czechia": "cz", "Denmark": "dk", "Estonia": "ee",
    "Finland": "fi", "France": "fr", "Germany": "de", "Greece": "gr",
    "Hungary": "hu", "Ireland": "en", "Italy": "it", "Latvia": "lv",
    "Lithuania": "lt", "Malta": "en", "Netherlands": "nl", "Poland": "pl",
    "Portugal": "pt", "Romania": "ro", "Slovakia": "sk", "Slovenia": "si",
    "Spain": "es", "Sweden": "se",
}

# 0 = completely disagree ... 100 = completely agree, "." = no opinion.
POSITION_MIDPOINT = 50.0
POSITION_HALF_RANGE = 50.0

# EU flag colours: users carry the blue, the EP-group anchors the yellow.
EU_BLUE = "#003399"
EU_YELLOW = "#FFCC00"
USER_COLOR = EU_BLUE
# The grand mean stays green: it has to read against the blue density it sits in.
USER_MEAN_COLOR = "#2ca02c"
MEAN_COLOR = "#111111"
# Sequential = one hue, light -> dark.
USER_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "eu_blue", ["#ffffff", EU_BLUE])


def load_user_stances(path: str, min_answered: int) -> pd.DataFrame:
    """One row per user: language + a stance in [-1, 1] per questionnaire index."""
    df = pd.read_stata(path, columns=["country", *STATEMENT_COLUMNS],
                       convert_categoricals=True)
    df["language"] = df["country"].astype(str).map(COUNTRY_TO_LANG)

    dropped = (
        df.loc[df["language"].isna(), "country"].astype(str).value_counts().to_dict()
    )
    print(f"Users in the file: {len(df):,}")
    print(f"Dropped countries (no run language): {dropped}")
    df = df[df["language"].notna()].copy()

    stances = (df[STATEMENT_COLUMNS].to_numpy(dtype=float) - POSITION_MIDPOINT) / POSITION_HALF_RANGE
    answered = np.isfinite(stances).sum(axis=1)
    keep = answered >= min_answered
    print(f"Users kept: {keep.sum():,} of {len(df):,} "
          f"(>= {min_answered} of {len(STATEMENT_COLUMNS)} statements answered)")

    out = pd.DataFrame(stances[keep], columns=range(len(STATEMENT_COLUMNS)))
    out.insert(0, "language", df.loc[keep, "language"].to_numpy())
    return out


def user_positions(stances: pd.DataFrame, questionnaire: pd.DataFrame,
                   dims: list[str]) -> pd.DataFrame:
    """Per-user mean stance on each signed dimension, NaN where nothing loads."""
    matrix = stances[list(range(len(STATEMENT_COLUMNS)))].to_numpy(dtype=float)
    positions = pd.DataFrame({"language": stances["language"].to_numpy()})
    for dim in dims:
        signs = questionnaire[dim].to_numpy(dtype=float)
        loading = signs != 0
        with warnings.catch_warnings():  # users who skipped a whole dimension
            warnings.simplefilter("ignore", RuntimeWarning)
            positions[dim] = np.nanmean(matrix[:, loading] * signs[loading], axis=1)
    return positions


def style_inset(ax, inset, means, pad=0.06):
    """Frame the inset on the country means and mark the region it magnifies."""
    low, high = means.min(axis=0) - pad, means.max(axis=0) + pad
    inset.set_xlim(low[0], high[0])
    inset.set_ylim(low[1], high[1])
    inset.axhline(0, color="gray", lw=0.6)
    inset.axvline(0, color="gray", lw=0.6)
    inset.tick_params(labelsize=6)
    inset.set_facecolor("white")
    inset.grid(True, linestyle=":", alpha=0.3)
    ax.indicate_inset_zoom(inset, edgecolor="black", alpha=0.6, lw=0.8)


def plot_user_compass(positions, x_dim, y_dim, one_sided, out_path,
                      by_language=False, bins=140, anchors=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_compass(ax, x_dim, y_dim, one_sided)

    points = positions[[x_dim, y_dim]].dropna().to_numpy()
    grand = points.mean(axis=0)
    # 800k respondents: a density image, not 800k overlapping dots.
    counts, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=[[-1, 1], [-1, 1]])
    mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(counts.T, 0),
                         cmap=USER_CMAP, norm=matplotlib.colors.LogNorm(),
                         shading="flat", zorder=0)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("users per cell (log)", fontsize=AXIS_FONTSIZE)
    bar.ax.tick_params(labelsize=FONTSIZE)

    handles = []
    if by_language:
        colors = language_colors(positions["language"].unique())
        means = {lang: group[[x_dim, y_dim]].dropna().to_numpy().mean(axis=0)
                 for lang, group in positions.groupby("language")}
        # The country means sit inside a few hundredths of each other, so they
        # get an inset of their own; on the full compass they are dots only. It
        # sits right of the legend, in the one band of the panel no EP-group
        # anchor falls into.
        inset = ax.inset_axes([0.44, 0.04, 0.34, 0.34])
        for lang, mean in means.items():
            for target, size in ((ax, 60), (inset, 90)):
                target.scatter(*mean, s=size, marker="o", facecolor=colors[lang],
                               edgecolor="white", linewidths=1.2, zorder=5)
            # Identity is never colour-alone: every mean carries its code.
            inset.annotate(lang, mean, textcoords="offset points", xytext=(7, 4),
                           fontsize=8, color="black", zorder=6,
                           path_effects=[matplotlib.patheffects.withStroke(
                               linewidth=2.5, foreground="white")])
        style_inset(ax, inset, np.array(list(means.values())))
        handles.append(Line2D([], [], color="gray", marker="o", ls="",
                              markersize=LEGEND_MARKERSIZE, label="country (inset)"))

    # The star is reserved for the party anchors, so the grand mean is a big dot.
    # It sits in the darkest part of the ramp, hence the white ring.
    ax.scatter(*grand, s=260, marker="o", facecolor=USER_MEAN_COLOR,
               edgecolor="white", linewidths=2.0, zorder=7)
    handles.append(Line2D([], [], color=USER_MEAN_COLOR, marker="o", ls="",
                          markeredgecolor="white", markersize=LEGEND_MARKERSIZE,
                          label="all users"))
    party_handle = draw_party_anchors(ax, anchors, x_dim, y_dim, color=EU_YELLOW)
    if party_handle is not None:
        handles.append(party_handle)

    # Labels are kept short on purpose: at 15pt a wider box reaches the GUE/NGL
    # anchor from the lower-left corner the legend now sits in.
    print(f"  Grand mean ({x_dim}, {y_dim}): {grand[0]:+.3f}, {grand[1]:+.3f}")
    ax.legend(handles=handles, loc=MAIN_LEGEND_LOC, fontsize=FONTSIZE, framealpha=0.9,
              title_fontsize=FONTSIZE, title="mean position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_user_violins(positions, dims, one_sided, out_path, rng, sample=60_000):
    x = np.arange(len(dims))
    datasets = []
    for dim in dims:
        values = positions[dim].dropna().to_numpy()
        if len(values) > sample:  # the KDE is unchanged, the runtime is not
            values = rng.choice(values, sample, replace=False)
        datasets.append(values)

    fig, ax = plt.subplots(figsize=(max(10, len(dims) * 1.5), 6))
    ax.axhline(0, color="gray", lw=0.8)
    parts = ax.violinplot(datasets, positions=x, widths=0.8,
                          showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(USER_COLOR)
        body.set_alpha(0.6)
    parts["cmeans"].set_color("black")

    ax.set_xticks(x)
    ax.set_xticklabels([d + ("\n(one-sided!)" if d in one_sided else "") for d in dims],
                       fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("position  (← disagree   ·   agree →)", fontsize=AXIS_FONTSIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(handles=[Line2D([], [], color=USER_COLOR, marker="s", ls="", label="users")],
              loc="upper right", fontsize=FONTSIZE, title_fontsize=FONTSIZE,
              title="respondents (black bar = mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--users", default=USER_DATASET_PATH)
    parser.add_argument("--x-dim", default="Left-Right")
    parser.add_argument("--y-dim", default="Europe")
    parser.add_argument("--min-answered", type=int, default=15,
                        help="Drop users who gave an opinion on fewer statements.")
    parser.add_argument("--parties", default=DEFAULT_POSITIONS,
                        choices=[*POSITION_CHOICES, "none"],
                        help="Which euandi answers the EP-group anchors come from; "
                             "'none' draws no anchors.")
    parser.add_argument("--violin-sample", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    questionnaire, dims, one_sided = load_questionnaire(args.dataset)
    for d in (args.x_dim, args.y_dim):
        if d not in dims:
            raise SystemExit(f"Dimension {d!r} not in questionnaire. Available: {dims}")
    if len(questionnaire) != len(STATEMENT_COLUMNS):
        raise SystemExit(f"Questionnaire has {len(questionnaire)} statements but "
                         f"{len(STATEMENT_COLUMNS)} user columns are mapped.")
    if one_sided:
        print(f"One-sided dimensions (agreement bias confounded): {sorted(one_sided)}")

    stances = load_user_stances(args.users, args.min_answered)
    unmapped = sorted(set(stances["language"]) - set(ALL_LANGS))
    if unmapped:
        raise SystemExit(f"Languages not in the run set: {unmapped}")
    print(f"Languages: {sorted(set(stances['language']))}")

    positions = user_positions(stances, questionnaire, dims)
    out_dir = Path("data") / f"{args.dataset}_results" / "plots" / "users"
    out_dir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(out_dir / "user_positions.csv", index=False)

    print("\nMean user position per dimension:")
    print(positions[dims].mean().round(3).to_string())

    anchors = None if args.parties == "none" else party_anchors(questionnaire, dims, args.parties)
    plot_user_compass(positions, args.x_dim, args.y_dim, one_sided,
                      out_dir / "political_compass_users.png", anchors=anchors)
    plot_user_compass(positions, args.x_dim, args.y_dim, one_sided,
                      out_dir / "political_compass_users_by_language.png",
                      by_language=True, anchors=anchors)
    plot_user_violins(positions, dims, one_sided,
                      out_dir / "dimension_violins_users.png",
                      np.random.default_rng(args.seed), args.violin_sample)
    print("\nDone.")


if __name__ == "__main__":
    main()
