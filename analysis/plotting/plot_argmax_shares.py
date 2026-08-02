#!/usr/bin/env python3
"""Each model's argmax share per EP group under all four evaluation methods.

The four ways this project reads a political position out of a model disagree about
which group a model looks like, and the disagreement is easy to lose across separate
figures. This puts them side by side, one figure per model:

  direct Likert      -- the model's Likert answers scored against the euandi positions
                        ({languages}{variant}.csv)
  indirect Likert    -- the same scoring, but the stance is read out of the model's
                        open-ended answers (speeches_{languages}{variant}_scored.csv)
  classified reasons -- the party classifier run over the reasons it gave for its
                        Likert answers ({languages}{variant}_classified.csv)
  classified speeches-- the party classifier run over the open-ended answers
                        (speeches_{languages}{variant}_classified.csv)

"Argmax share" is the share of units won by a group. For the classifier that is a
predicted party per (text, language, prompt variant), which is what plot_classified_parties
already counts. The two euandi methods have no argmax of their own, so the analogue is
built here: per (statement, language) the group with the highest mean agreement wins the
cell, and cells with a tie split their credit equally, so the shares still sum to 1.
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS_DIR.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.plotting import plot_classified_parties as pcp
from analysis.plotting.plot_topical_parties import statement_rows_per_axis
from analysis.evaluate_cronbach import AXES
from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS, POSITION_CHOICES, load_party_positions, positions_path,
)
from analysis.vaa_agreement_ci import stance_frame_for
from utils.constants import ALL_LANGS

# (method key, legend label, how the numbers are produced)
METHODS = [
    ("likert_vaa", "direct Likert", "vaa"),
    ("speeches_vaa", "indirect Likert", "vaa"),
    ("reasons_clf", "classified reasons", "classifier"),
    ("speeches_clf", "classified open-ended", "classifier"),
]
METHOD_ORDER = [key for key, _, _ in METHODS]
METHOD_LABEL = {key: label for key, label, _ in METHODS}
METHOD_COLORS = {
    "likert_vaa":   "#4C72B0",
    "speeches_vaa": "#8FB2DC",
    "reasons_clf":  "#C44E52",
    "speeches_clf": "#E39A9C",
}
# In the cross-model figures colour is spent on the model, so the stacked methods are
# told apart by texture instead.
METHOD_HATCHES = {
    "likert_vaa":   "",
    "speeches_vaa": "///",
    "reasons_clf":  "...",
    "speeches_clf": "xxx",
}
# The classifier CSVs are named by the text that was classified; the euandi stance
# files are named by how the answer was elicited.
VAA_METHOD_STANCE_SOURCE = {"likert_vaa": "likert", "speeches_vaa": "speeches"}
CLASSIFIER_METHOD_SOURCE = {"reasons_clf": "reasons", "speeches_clf": "speeches"}

# One colour per topical axis, in the questionnaire's own axis order.
TOPIC_COLORS = {
    axis: plt.get_cmap("tab10")(index % 10) for index, axis in enumerate(AXES)
}


def ordered_languages_present(languages_present):
    """The questionnaire's own language order, as utils.constants lists it, with anything
    unlisted appended -- the same shape as pcp.ordered_parties_present."""
    ordered = [language for language in ALL_LANGS if language in languages_present]
    extras = sorted(language for language in languages_present if language not in ALL_LANGS)
    return ordered + extras


def language_colors(languages):
    """One colour per language. tab20, which plot_split_bars falls back to, repeats at 21;
    nipy_spectral is what plot_classified_parties already spends on languages.

    Sampled short of both ends: the last language would otherwise come out near-white and
    disappear against the panel, and the first near-black."""
    cmap = plt.get_cmap("nipy_spectral")
    positions = np.linspace(0.06, 0.92, max(len(languages), 2))
    return {language: cmap(positions[index]) for index, language in enumerate(languages)}


ARGMAX_SHARE_LABEL = "Argmax share"
POOLED_LABEL = "pooled"
AVERAGE_LABEL = "average of methods"

# Sized for figures that are read at a glance in a folder of PNGs rather than zoomed
# into. At this size the legends no longer fit over the bars, so they sit under the
# axes instead -- see the `legend_below` calls.
TICK_FONTSIZE = 27
AXIS_LABEL_FONTSIZE = 32
TITLE_FONTSIZE = 32
LEGEND_FONTSIZE = 28
LEGEND_TITLE_FONTSIZE = 31
VALUE_FONTSIZE = 20


class Layout(NamedTuple):
    """How wide a split-bar figure is drawn, and how large its type is drawn on it.

    A figure that is browsed as a PNG is read at whatever size it was saved, so the sizes
    above are all it needs. A figure that goes into a LaTeX column is scaled to that
    column whatever it measures, so the only thing that survives the reduction is the
    ratio between the type and the figure -- which is what this record sets."""
    bar_span: float           # inches of figure width allotted per bar
    # Width : height of the panel, not of the saved file. Under `fit_legends` the legends
    # hang below the panel and the saved file grows by however much they take, which at
    # this type size is several inches -- so the saved figure comes out markedly taller
    # than this ratio alone would say.
    aspect: float | None      # None keeps the fixed panel height instead
    font_scale: float         # multiplies every FONTSIZE constant above
    # Applied to the legends on top of font_scale. A legend row costs height that the
    # panel then does not get, and the entries are mostly short codes, so the legends can
    # run a little smaller than the axis labels without becoming the limiting text.
    legend_scale: float
    tick_rotation: int
    # Whether the legends are placed by measuring the figure rather than by the fixed
    # fractions below. The fractions are tuned for the default type and go wrong at 2.5x,
    # but they are also what every existing figure was drawn with, so they stay the
    # default and only the figures that need the measurement opt in.
    fit_legends: bool


# A panel height that does not follow the width, so a figure with more bars gets wider
# rather than larger. Every figure but the per-language ones is drawn this way.
FIXED_PANEL_HEIGHT = 18.0
DEFAULT_LAYOUT = Layout(bar_span=0.34, aspect=None, font_scale=1.0, legend_scale=1.0,
                        tick_rotation=20, fit_legends=False)
# For the per-language figures, which are the widest this script draws: 6 EP groups by 24
# languages is 144 bars, and at the default span that is a 43-inch strip whose 27pt labels
# come out near 2pt once it is reduced to a ~3.3-inch column. A tighter span brings the
# width to ~28 inches, so the reduction is ~8x rather than ~13x, and 2.5x type then lands
# the labels near 8pt on the page. Two legends set that large cost some 11 inches of
# height whatever the panel does, so the panel is drawn well wider than it is tall and the
# figure still saves at close to 1:1 -- room to read the bars in, rather than the strip
# the default proportions give.
COLUMN_LAYOUT = Layout(bar_span=0.22, aspect=1.8, font_scale=2.5, legend_scale=0.8,
                       tick_rotation=45, fit_legends=True)

# The two models the thesis prints the per-language figure for, in panel order. Any other
# pair is one --panel-models away; these are only the default.
DEFAULT_PANEL_MODELS = ("grok-4.5", "gemma-4-31b")


def classifier_long_frames(model_dir):
    """{(method, variant): long predictions} from the *_classified.csv files. The frames
    are kept rather than reduced straight to shares so the per-topic figures can filter
    them by statement."""
    frames = {}
    for (source, variant_label), csv_path in pcp.discover_classified_csvs(model_dir).items():
        method = next(
            (key for key, csv_source in CLASSIFIER_METHOD_SOURCE.items() if csv_source == source),
            None,
        )
        if method is None:
            continue
        long_df = pcp.load_long_predictions(csv_path)
        if long_df.empty:
            print(f"  [{method}/{variant_label}] no predictions in {csv_path.name}")
            continue
        frames[(method, variant_label)] = long_df
    return frames


def language_of_stance_column(column):
    """"bg_negated_stance" -> "bg".

    The stance columns are named after the file's framing as well as its language, so a
    negated Likert run calls the column bg_negated_stance while the classifier frames call
    the same language bg. The framing is already the frame's own key, so it comes off here
    -- the same normalisation plot_classified_parties does to the vaa*.csv language column."""
    stem = column.removesuffix("_stance")
    for suffix in pcp.VARIANT_SUFFIX_TO_LABEL:
        if suffix and stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def cell_agreement(stance_df, languages, party_df):
    """Mean agreement per (statement, language, EP group) -- one row per cell a group
    can win. Kept as a mean over the group's parties so a group with more parties in
    the positions file does not count more, matching evaluate_euandi."""
    stance_columns = [f"{language}_stance" for language in languages]
    merged = party_df.merge(stance_df[["statement_idx", *stance_columns]], on="statement_idx")
    long = merged.melt(
        id_vars=["ep_group", "statement_idx", "normalized_answer"],
        value_vars=stance_columns, var_name="language", value_name="llm_stance",
    )
    long["language"] = long["language"].map(language_of_stance_column)
    long["agreement"] = 1 - (long["normalized_answer"] - long["llm_stance"]).abs() / 2
    return (
        long.dropna(subset=["agreement"])
        .groupby(["statement_idx", "language", "ep_group"], as_index=False)["agreement"]
        .mean()
    )


def vaa_argmax_share(cells):
    """Share of (statement, language) cells each group agrees with the model most on.

    Group positions are coarse, so exact ties are routine; a tied cell splits its credit
    between the tied groups rather than going to whichever one sorts first."""
    if cells.empty:
        return pd.Series(dtype=float)
    cell_keys = ["statement_idx", "language"]
    highest = cells.groupby(cell_keys)["agreement"].transform("max")
    is_top = (cells["agreement"] == highest).astype(float)
    winners_per_cell = is_top.groupby([cells[key] for key in cell_keys]).transform("sum")
    credit = is_top / winners_per_cell
    cell_count = cells.groupby(cell_keys).ngroups
    return credit.groupby(cells["ep_group"]).sum() / cell_count


def vaa_cell_frames(model_dir, party_df):
    """{(method, variant): per (statement, language, group) agreement} from the raw
    euandi stance files, kept unreduced for the same reason as the classifier frames."""
    frames = {}
    for (classifier_source, variant_label), path in pcp.discover_vaa_csvs(model_dir).items():
        method = "speeches_vaa" if classifier_source == "speeches" else "likert_vaa"
        stance_df, languages = stance_frame_for(
            str(model_dir.parent), model_dir.name,
            VAA_METHOD_STANCE_SOURCE[method],
            pcp.VARIANT_LABEL_TO_SUFFIX[variant_label],
            pcp.languages_in_vaa_filename(path),
        )
        if stance_df is None:
            print(f"  [{method}/{variant_label}] no stance file, skipping.")
            continue
        cells = cell_agreement(stance_df, languages, party_df)
        if not cells.empty:
            frames[(method, variant_label)] = cells
    return frames


def shares_from_frames(long_frames, cell_frames, rows=None, language=None):
    """{(method, variant): argmax share per party}, optionally over one topic's
    statements or one language only. `rows` are 0-based statement positions, which is what
    the classifier frames call row_idx and the euandi frames call statement_idx.

    Narrowing to a language needs nothing else: predicted_party_share counts whatever rows
    it is handed, and vaa_argmax_share divides by the cell count of the frame it is given,
    so a one-language slice is normalised against that language's own cells."""
    shares = {}
    for key, long_df in long_frames.items():
        if rows is not None:
            long_df = long_df[long_df["row_idx"].isin(rows)]
        if language is not None:
            long_df = long_df[long_df["language"] == language]
        share = pcp.predicted_party_share(long_df)
        if not share.empty:
            shares[key] = share
    for key, cells in cell_frames.items():
        if rows is not None:
            cells = cells[cells["statement_idx"].isin(rows)]
        if language is not None:
            cells = cells[cells["language"] == language]
        share = vaa_argmax_share(cells)
        if not share.empty:
            shares[key] = share
    return shares


def pooled_over_variants(shares_by_method_variant):
    """The same shares under a 'pooled' variant, averaged over the variants present.
    Every variant covers the same units, so a plain mean is the pooled share."""
    per_method = {}
    for (method, _), share in shares_by_method_variant.items():
        per_method.setdefault(method, []).append(share)
    return {
        (method, POOLED_LABEL): pd.concat(series_list).groupby(level=0).mean()
        for method, series_list in per_method.items()
    }


def variants_present(shares_by_method_variant):
    labels = {variant for _, variant in shares_by_method_variant}
    ordered = [v for v in pcp.VARIANT_LABEL_ORDER if v in labels]
    return ordered + ([POOLED_LABEL] if POOLED_LABEL in labels else [])


def legend_below(ax, handles, title, ncol, y_offset=-0.28, font_scale=1.0):
    """Legends at this type size cover the bars, so they go under the axes.

    They expand to the axes width rather than shrink-wrapping their labels, so the
    method and model legends stack as two boxes of the same width.

    `y_offset` is a fraction of the axes height, so it has to grow with the type: a
    legend row set 2.5x larger takes 2.5x as much of the axes to clear."""
    return ax.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(0.0, y_offset * font_scale, 1.0, 0.001),
        mode="expand", fontsize=LEGEND_FONTSIZE * font_scale, title=title,
        title_fontsize=LEGEND_TITLE_FONTSIZE * font_scale, framealpha=0.9, ncol=ncol,
    )


def stack_legends_below(ax, blocks, legend_font_scale, gap=0.04):
    """Stack legends under the axes, each one placed below whatever is already drawn.

    `legend_below` positions by a fraction of the axes height chosen for the default type
    size. Scale the type up and those fractions put the legends through the tick labels
    and through each other, because a legend's height is set by its font in points while
    the fraction is not. Here the tick labels are drawn first and measured, each legend is
    placed below the last thing measured, and the arithmetic then holds at any type size.

    These legends are also centred under the panel rather than expanded to its width: an
    expanded legend gives every column the same slice of the width whatever is in it, and
    at this type size a patch plus a label overruns the slice and prints over the next
    column. Centred, the columns are sized to their contents instead.

    The measurement is against where the axes finally sits, so the layout is settled
    before measuring. The legends then hang off the figure rather than the axes, and are
    positioned in figure coordinates: tight_layout does not lay out figure-level legends,
    so the panel keeps the whole canvas and the legends stay where they were measured,
    while `bbox_inches="tight"` still grows the saved image to take them in. A legend the
    layout does own instead squeezes the panel it hangs off -- at this type size, to under
    half the canvas -- and one excluded from the layout is dropped from the saved bounding
    box and clipped away entirely."""
    figure = ax.figure
    figure.tight_layout()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()

    def bottom_in_figure(artist):
        return float(to_figure.transform((0.0, artist.get_window_extent(renderer).y0))[1])

    # The legends span the panel, so they take their left edge and width from it.
    panel = ax.get_position()
    lowest = min(bottom_in_figure(label) for label in ax.get_xticklabels())
    for handles, title, ncol in blocks:
        legend = figure.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(panel.x0, lowest - gap, panel.width, 0.001),
            bbox_transform=figure.transFigure,
            fontsize=LEGEND_FONTSIZE * legend_font_scale, title=title,
            title_fontsize=LEGEND_TITLE_FONTSIZE * legend_font_scale, framealpha=0.9,
            ncol=ncol,
            handlelength=1.4, handletextpad=0.6, columnspacing=1.4, borderpad=0.6,
        )
        figure.canvas.draw()
        lowest = bottom_in_figure(legend)


def percentages_per_method(share_per_method, methods, parties):
    return {
        method: 100.0 * share_per_method[method].reindex(parties).fillna(0.0).to_numpy()
        for method in methods
    }


def draw_grouped_panel(ax, percentages, methods, parties):
    """One bar per (party, method): the methods side by side, as they were measured."""
    party_positions = np.arange(len(parties))
    group_width = 0.84
    bar_width = group_width / len(methods)

    for method_index, method in enumerate(methods):
        offsets = party_positions - group_width / 2 + bar_width * (method_index + 0.5)
        ax.bar(offsets, percentages[method], width=bar_width,
               color=METHOD_COLORS[method], edgecolor="black", linewidth=0.4,
               label=METHOD_LABEL[method])
        for offset, percentage in zip(offsets, percentages[method]):
            ax.text(offset, percentage + 0.4, f"{percentage:.0f}", ha="center", va="bottom",
                    fontsize=VALUE_FONTSIZE, rotation=90)
    ax.set_ylabel(f"{ARGMAX_SHARE_LABEL} (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("per method", fontsize=TITLE_FONTSIZE)


def draw_average_panel(ax, percentages, methods, parties):
    """One bar per party, split into what each method contributes to the four-method
    average. Every method's shares sum to 1 across parties, so the methods are
    commensurable and a segment is just that method's share divided by their number --
    the stack totals the average, and its make-up shows which method carries it."""
    party_positions = np.arange(len(parties))
    bottoms = np.zeros(len(parties))
    for method in methods:
        contribution = percentages[method] / len(methods)
        ax.bar(party_positions, contribution, width=0.62, bottom=bottoms,
               color=METHOD_COLORS[method], edgecolor="black", linewidth=0.4,
               label=METHOD_LABEL[method])
        bottoms += contribution
    for position, total in zip(party_positions, bottoms):
        ax.text(position, total + 0.4, f"{total:.0f}", ha="center", va="bottom",
                fontsize=VALUE_FONTSIZE + 2)
    # Kept short: at this type size a full sentence of a label crowds the panel.
    ax.set_ylabel("Mean over methods (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"average of {len(methods)} methods", fontsize=TITLE_FONTSIZE)


def plot_methods_argmax_share(share_per_method, output_path, ylim=None):
    """The methods side by side, and their average split into method contributions."""
    methods = [method for method in METHOD_ORDER if method in share_per_method]
    if not methods:
        return
    parties = pcp.ordered_parties_present(
        set().union(*(share_per_method[method].index for method in methods)))
    if not parties:
        return

    percentages = percentages_per_method(share_per_method, methods, parties)
    tallest = max(float(values.max()) for values in percentages.values())
    panel_ylim = ylim or (0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM))

    fig, axes = plt.subplots(1, 2, figsize=(max(22, len(parties) * 3.8), 10.5),
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.22})
    draw_grouped_panel(axes[0], percentages, methods, parties)
    draw_average_panel(axes[1], percentages, methods, parties)

    for ax in axes:
        ax.set_xticks(np.arange(len(parties)))
        ax.set_xticklabels(parties, rotation=30, ha="right", fontsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.set_xlim(-0.5, len(parties) - 0.5)
        # Both panels are argmax shares in percent, so they share a scale: the average
        # can be read straight off against the methods it averages.
        ax.set_ylim(*panel_ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
               fontsize=LEGEND_FONTSIZE, title="method",
               title_fontsize=LEGEND_TITLE_FONTSIZE, framealpha=0.9,
               ncol=min(len(methods), 4))
    pcp.save_figure(fig, output_path, bbox_inches="tight")


def shares_for_variant(shares_by_method_variant, variant_label):
    return {
        method: share
        for (method, csv_variant), share in shares_by_method_variant.items()
        if csv_variant == variant_label
    }


def share_limits(shares_by_model):
    """(0, tallest bar + headroom) over every figure the run draws, so the models can be
    compared as they are browsed."""
    tallest = 0.0
    for shares_by_method_variant in shares_by_model.values():
        for share in shares_by_method_variant.values():
            values = 100.0 * share.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                tallest = max(tallest, float(finite.max()))
    return (0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM)) if tallest else None


def average_limits(shares_by_model):
    """(0, tallest stack + headroom) over the averaged figures only.

    A stacked bar is the mean of the methods, so it never reaches as high as the
    per-method bars; sharing their scale left the top half of every averaged figure
    empty. These figures stand alone, so they get their own limit -- still one limit
    across the whole run, so the models stay comparable with each other."""
    tallest = 0.0
    for shares_by_method_variant in shares_by_model.values():
        for variant_label in variants_present(shares_by_method_variant):
            average = average_over_methods(
                shares_for_variant(shares_by_method_variant, variant_label))
            values = 100.0 * average.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                tallest = max(tallest, float(finite.max()))
    return (0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM)) if tallest else None


def average_over_methods(share_per_method):
    """The four methods' mean share per party -- the total of the stacked bars."""
    if not share_per_method:
        return pd.Series(dtype=float)
    return pd.concat(share_per_method.values()).groupby(level=0).mean()


def summary_records(model_name, shares_by_method_variant):
    records = [
        {
            "model": model_name,
            "method": METHOD_LABEL[method],
            "variant": variant_label,
            "ep_group": party,
            "argmax_share": float(share_value),
        }
        for (method, variant_label), share in sorted(shares_by_method_variant.items())
        for party, share_value in share.items()
    ]
    for variant_label in variants_present(shares_by_method_variant):
        average = average_over_methods(shares_for_variant(shares_by_method_variant, variant_label))
        records += [
            {
                "model": model_name,
                "method": AVERAGE_LABEL,
                "variant": variant_label,
                "ep_group": party,
                "argmax_share": float(share_value),
            }
            for party, share_value in average.items()
        ]
    return records


def collect_model_shares(model_dir, party_df, dataset):
    """(shares per method x variant, per topic x method, per language x method, statements
    per topic) for one model."""
    print(f"\nLoading: {model_dir.name}")
    long_frames = classifier_long_frames(model_dir)
    cell_frames = vaa_cell_frames(model_dir, party_df)
    shares = shares_from_frames(long_frames, cell_frames)
    if not shares:
        print("  nothing to plot for this model.")
        return {}, {}, {}, {}
    shares.update(pooled_over_variants(shares))
    methods_found = sorted({METHOD_LABEL[method] for method, _ in shares})
    print(f"  methods: {', '.join(methods_found)}")

    axis_rows = statement_rows_per_axis(dataset, statement_count(long_frames, cell_frames))
    per_language = language_shares(long_frames, cell_frames)
    print(f"  languages: {len(per_language)}")
    # The counts come off the very rows the per-topic shares were computed on, so a label
    # can never claim more evidence than the bar above it rests on.
    topic_counts = {axis: len(rows) for axis, rows in axis_rows.items()}
    return (shares, topic_shares(long_frames, cell_frames, axis_rows), per_language,
            topic_counts)


def topic_labels_with_counts(axes_present, topic_counts):
    """"Ukraine" -> "Ukraine (7)". Two statements behind a bar is not the same evidence as
    twelve, so the count travels with the topic on the figure itself."""
    return {
        axis: f"{axis} ({topic_counts[axis]})" if axis in topic_counts else axis
        for axis in axes_present
    }


def plot_model(model_dir, shares_by_method_variant, per_axis, per_language, topic_counts,
               ylim, topic_ylim, language_ylim):
    plots_dir = pcp.plots_dir_for(model_dir)
    for variant_label in variants_present(shares_by_method_variant):
        plot_methods_argmax_share(
            shares_for_variant(shares_by_method_variant, variant_label),
            plots_dir / f"argmax_share_methods_{variant_label}.png",
            ylim,
        )
    # Topics and languages only for the pooled framing: a slice already cuts the units
    # down, and splitting those by framing as well leaves too little behind each bar.
    axes_present = [axis for axis in AXES if axis in per_axis]
    plot_slice_both_ways(
        per_axis, plots_dir, topic_ylim, stem="topics", transposed_stem="topics_by_topic",
        slice_order=axes_present, slice_colors=TOPIC_COLORS, slice_legend_title="topic",
        slice_labels=topic_labels_with_counts(axes_present, topic_counts),
    )
    languages = ordered_languages_present(per_language)
    plot_slice_both_ways(
        per_language, plots_dir, language_ylim,
        stem="languages", transposed_stem="languages_by_lang",
        slice_order=languages, slice_colors=language_colors(languages),
        # Wide rows rather than tall ones: at COLUMN_LAYOUT's type size every legend row
        # is height taken off the panel. Seven fills the rows the same as eight does for
        # 21 languages, and keeps the legend inside the panel's own width.
        slice_legend_title="language", slice_ncol=min(len(languages), 7),
        # 24 languages make these the widest figures here, and they are the ones that go
        # into the thesis a column wide, so they are the ones sized for the reduction.
        layout=COLUMN_LAYOUT,
    )


def plot_slice_both_ways(per_slice, plots_dir, ylim, *, stem, transposed_stem,
                         slice_order, slice_colors, slice_legend_title, slice_ncol=None,
                         slice_labels=None, layout=DEFAULT_LAYOUT):
    """A breakdown drawn both ways round: the slices inside each EP group, and the EP
    groups inside each slice.

    Which grouping reads better depends on the question -- whether one group's share
    travels across the slices, or where one slice leans -- and the two figures are the
    same numbers regrouped, so neither can mislead about the other.

    `slice_labels` renames the slices for display only. A slice is the legend in the first
    figure and the x axis in the second, so the same mapping is handed to whichever of the
    two the slices land on -- both figures then carry the same labels.

    `layout` goes to both, so a figure and its transpose stay a matched pair."""
    if not per_slice:
        return
    plot_split_bars(
        per_slice, plots_dir / f"argmax_share_{stem}_pooled.png", ylim,
        series_order=slice_order, series_colors=slice_colors,
        series_legend_title=slice_legend_title, series_ncol=slice_ncol,
        series_labels=slice_labels, layout=layout,
    )
    per_party = transposed_shares(per_slice)
    plot_split_bars(
        per_party, plots_dir / f"argmax_share_{transposed_stem}_pooled.png", ylim,
        series_order=pcp.ordered_parties_present(per_party),
        series_colors={party: pcp.color_for_party(party) for party in per_party},
        series_legend_title="EP group", category_order=slice_order,
        category_labels=slice_labels, layout=layout,
    )


def stacked_language_panels(languages_by_model, panel_model_dirs, dataset_plots_dir, ylim):
    """The per-language figures again, but with one model per panel and one set of legends.

    Two models drawn as two figures are read one after the other, and the legends -- 24
    languages and four methods, set for a LaTeX column -- are then repeated for nothing.
    Stacked, the same colour in the same x position is the same language in both models,
    so the panels can be compared line by line and the legends are stated once.

    Colours and order come from the languages of all the panels together, so a language
    only one model answered in still keeps its slot in the other panel."""
    panels = [(model_dir.name, languages_by_model[model_dir])
              for model_dir in panel_model_dirs if languages_by_model.get(model_dir)]
    if len(panels) < 2:
        print(f"  only {len(panels)} model(s) for the stacked per-language panels, skipping.")
        return
    languages = ordered_languages_present(
        {language for _, per_language in panels for language in per_language})
    colors = language_colors(languages)
    print(f"  stacked panels: {', '.join(title for title, _ in panels)}")

    plot_stacked_split_bars(
        panels, dataset_plots_dir / "argmax_share_languages_pooled_panels.png", ylim,
        series_order=languages, series_colors=colors, series_legend_title="language",
        series_ncol=min(len(languages), 7), layout=COLUMN_LAYOUT,
    )
    plot_stacked_split_bars(
        [(title, transposed_shares(per_language)) for title, per_language in panels],
        dataset_plots_dir / "argmax_share_languages_by_lang_pooled_panels.png", ylim,
        series_order=pcp.ordered_parties_present(
            {party for _, per_language in panels
             for party in transposed_shares(per_language)}),
        series_colors={party: pcp.color_for_party(party)
                       for _, per_language in panels
                       for party in transposed_shares(per_language)},
        series_legend_title="EP group", category_order=languages, layout=COLUMN_LAYOUT,
    )


def topic_shares(long_frames, cell_frames, axis_rows):
    """{axis: {method: argmax share per party}}, pooled over prompt variants.

    A statement can load several axes, so the topics overlap and the bars of one party
    do not partition its overall share."""
    per_axis = {}
    for axis, rows in axis_rows.items():
        shares = shares_from_frames(long_frames, cell_frames, rows)
        if not shares:
            continue
        pooled = pooled_over_variants(shares)
        for_axis = {method: share for (method, _), share in pooled.items()}
        if for_axis:
            per_axis[axis] = for_axis
    return per_axis


def languages_in_frames(long_frames, cell_frames):
    present = set()
    for frame in (*long_frames.values(), *cell_frames.values()):
        present.update(frame["language"].dropna().unique())
    return ordered_languages_present(present)


def language_shares(long_frames, cell_frames):
    """{language: {method: argmax share per party}}, pooled over prompt variants.

    Unlike the topics, the languages partition the units: every unit was answered in
    exactly one language, so one language's shares sum to 1 across parties and the
    languages of a party do add up to its overall share."""
    per_language = {}
    for language in languages_in_frames(long_frames, cell_frames):
        shares = shares_from_frames(long_frames, cell_frames, language=language)
        if not shares:
            continue
        pooled = pooled_over_variants(shares)
        for_language = {method: share for (method, _), share in pooled.items()}
        if for_language:
            per_language[language] = for_language
    return per_language


def transposed_shares(share_per_series_and_method):
    """Swap the series with the party the shares are indexed by.

    plot_split_bars puts the index of the shares on the x axis and the outer key in the
    legend, so {language: {method: share per party}} draws languages inside each party;
    handing it the transpose draws the parties inside each language instead. Same numbers,
    regrouped -- nothing is recomputed, so the two figures cannot disagree."""
    by_party = {}
    for series_name, per_method in share_per_series_and_method.items():
        for method, share in per_method.items():
            for party, value in share.items():
                by_party.setdefault(party, {}).setdefault(method, {})[series_name] = float(value)
    return {
        party: {method: pd.Series(values) for method, values in per_method.items()}
        for party, per_method in by_party.items()
    }


def stacked_series_limits(series_by_model):
    """(0, tallest stack + headroom) over every split-bar figure of one family in the run.

    Takes {model: {series: {method: share}}}, so it serves the per-topic and the
    per-language figures alike."""
    tallest = 0.0
    for per_axis in series_by_model.values():
        for share_per_method in per_axis.values():
            values = 100.0 * average_over_methods(share_per_method).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                tallest = max(tallest, float(finite.max()))
    return (0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM)) if tallest else None


def statement_count(long_frames, cell_frames):
    """How many statements the questionnaire was administered with, read off whichever
    frame is available: the classifier calls the 0-based position row_idx, euandi calls
    it statement_idx."""
    positions = [int(df["row_idx"].max()) for df in long_frames.values()]
    positions += [int(df["statement_idx"].max()) for df in cell_frames.values()]
    return 1 + max(positions) if positions else 0


def methods_present(shares_by_model):
    found = {method for shares in shares_by_model.values() for method, _ in shares}
    return [method for method in METHOD_ORDER if method in found]


def cross_model_scopes(shares_by_model):
    """(method, variant, filename suffix, scope label) for the cross-model figure and
    each of its slices. `method is None` means the average over methods, mirroring how
    plot_classified_parties pools its sources."""
    methods = methods_present(shares_by_model)
    variants = [
        variant for variant in pcp.VARIANT_LABEL_ORDER
        if any(variant == csv_variant for shares in shares_by_model.values()
               for _, csv_variant in shares)
    ]

    scopes = [(None, POOLED_LABEL, "", "average of methods, variants pooled")]
    scopes += [(method, POOLED_LABEL, f"_{method}", f"{METHOD_LABEL[method]}, variants pooled")
               for method in methods]
    if len(variants) > 1:
        scopes += [(None, variant, f"_{variant}", f"average of methods, {variant}")
                   for variant in variants]
        scopes += [(method, variant, f"_{method}_{variant}",
                    f"{METHOD_LABEL[method]}, {variant}")
                   for method in methods for variant in variants]
    return scopes


def scoped_share_per_model_and_method(shares_by_model, variant_label, methods=None):
    """{model: {method: share per party}} for one framing -- the input to the cross-model
    figures, whose bars keep the four-way split. `methods` narrows it to a subset."""
    per_model = {}
    for model_dir, shares in shares_by_model.items():
        for_variant = shares_for_variant(shares, variant_label)
        if methods is not None:
            for_variant = {method: for_variant[method]
                           for method in methods if method in for_variant}
        if for_variant:
            per_model[model_dir.name] = for_variant
    return per_model


def split_bar_dimensions(share_dicts, series_order, category_order):
    """(series, methods, categories) covering every panel a figure draws.

    Taken over all the panels at once rather than per panel, so a stacked figure draws the
    same series in the same order on every panel: a language missing from one model would
    otherwise shift that panel's bars out of line with the panel above it, and the shared
    legend would then name the wrong colours."""
    present_series = set().union(*(set(shares) for shares in share_dicts))
    series = [name for name in (series_order or sorted(present_series))
              if name in present_series]
    methods = [
        method for method in METHOD_ORDER
        if any(method in per_method for shares in share_dicts
               for per_method in shares.values())
    ]
    categories_present = {
        category
        for shares in share_dicts
        for per_method in shares.values()
        for share in per_method.values()
        for category in share.index
    }
    categories = ([c for c in category_order if c in categories_present]
                  if category_order is not None
                  else pcp.ordered_parties_present(categories_present))
    return series, methods, categories


def draw_split_bar_panel(ax, share_per_series_and_method, series, methods, categories,
                         color_for, layout, ylim, category_labels=None,
                         show_xticklabels=True):
    """One panel of grouped, method-stacked bars. Returns the tallest stack drawn, so a
    caller that did not fix `ylim` can size the panel to what it got."""
    category_positions = np.arange(len(categories))
    group_width = 0.86
    bar_width = group_width / len(series)
    hairline = 0.3 * layout.font_scale

    tallest = 0.0
    for series_index, name in enumerate(series):
        per_method = share_per_series_and_method.get(name, {})
        offsets = category_positions - group_width / 2 + bar_width * (series_index + 0.5)
        bottoms = np.zeros(len(categories))
        for method in methods:
            share = per_method.get(method)
            if share is None:
                continue
            contribution = 100.0 * share.reindex(categories).fillna(0.0).to_numpy() / len(methods)
            ax.bar(offsets, contribution, width=bar_width, bottom=bottoms,
                   color=color_for(series_index, name), edgecolor="black", linewidth=hairline,
                   hatch=METHOD_HATCHES[method])
            bottoms += contribution
        tallest = max(tallest, float(bottoms.max()))

    ax.set_xticks(category_positions)
    if show_xticklabels:
        ax.set_xticklabels([display_name(category, category_labels) for category in categories],
                           rotation=layout.tick_rotation, ha="right",
                           fontsize=TICK_FONTSIZE * layout.font_scale)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE * layout.font_scale)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    for boundary in category_positions[:-1] + 0.5:
        ax.axvline(boundary, color="black", linewidth=0.5 * layout.font_scale, alpha=0.15)
    ax.set_ylabel(stacked_axis_label(methods),
                  fontsize=AXIS_LABEL_FONTSIZE * layout.font_scale)
    ax.set_ylim(*(ylim or (0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM))))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    return tallest


def split_bar_legend_blocks(series, methods, color_for, layout, figure_width,
                            series_legend_title, series_ncol, series_labels):
    """The (handles, title, ncol) blocks both split-bar figures put under their panels.

    The legends stretch to the axes width, so a narrow figure has to wrap the long method
    labels instead of running them across one row. The width that buys a row of four is in
    inches of label, so it goes up with the type: "classified open-ended" set at 2.5x needs
    roughly 2.5x the room to run four across."""
    method_ncol = min(len(methods), 4 if figure_width >= 21 * layout.font_scale else 2)
    method_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.5 * layout.font_scale,
              hatch=METHOD_HATCHES[method], label=METHOD_LABEL[method])
        for method in methods
    ]
    method_title = "method (stacked)" if len(methods) > 1 else "method"
    series_handles = [
        Patch(facecolor=color_for(index, name), edgecolor="black",
              linewidth=0.5 * layout.font_scale, label=display_name(name, series_labels))
        for index, name in enumerate(series)
    ]
    return [
        (method_handles, method_title, method_ncol),
        (series_handles, series_legend_title, series_ncol or min(len(series), 4)),
    ]


def series_color_lookup(series, series_colors):
    if series_colors:
        return lambda index, name: series_colors[name]
    default_cmap = plt.get_cmap("tab20", max(len(series), 2))
    return lambda index, name: default_cmap(index)


def plot_split_bars(share_per_series_and_method, output_path, ylim=None,
                    series_order=None, series_colors=None, series_legend_title="model",
                    category_order=None, series_ncol=None,
                    series_labels=None, category_labels=None, layout=DEFAULT_LAYOUT):
    """One bar per (category, series) as plot_classified_parties draws them, but each bar
    stacked into what the four methods contribute to its average.

    The series is whatever the figure compares -- models, topical axes, languages. Colour
    is spent on it, so the methods are told apart by hatch. Bar height is the four-method
    average, exactly as in the per-model figures.

    The category on the x axis is whatever the shares are indexed by, EP groups unless
    `category_order` says otherwise -- that is the only hook the per-language figure needs
    to draw the same numbers the other way round.

    `series_labels` and `category_labels` are display names only: the ordering, the colour
    lookup and the share indexing all keep using the keys themselves, so a label cannot
    move a bar.

    `layout` sets the width per bar and the type size relative to it -- the default draws
    a figure to be read at the size it is saved, COLUMN_LAYOUT one to be reduced into a
    LaTeX column."""
    series, methods, categories = split_bar_dimensions(
        [share_per_series_and_method], series_order, category_order)
    if not series or not methods or not categories:
        return

    color_for = series_color_lookup(series, series_colors)
    # Proportioned for a full-page-width figure, so all the bars stay legible.
    figure_width = max(18, len(categories) * len(series) * layout.bar_span)
    figure_height = (FIXED_PANEL_HEIGHT if layout.aspect is None
                     else figure_width / layout.aspect)
    # Hatches tell the four methods apart, and their line width is absolute rather than
    # relative to the figure -- at 2.5x type they would reduce to nothing, so they and the
    # bar outlines are thickened with the rest. The width is read when a bar is created,
    # not when the figure is saved, so the whole panel is drawn inside the context.
    plt.rcParams["hatch.linewidth"] = 1.0 * layout.font_scale
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    draw_split_bar_panel(ax, share_per_series_and_method, series, methods, categories,
                         color_for, layout, ylim, category_labels)

    blocks = split_bar_legend_blocks(series, methods, color_for, layout, figure_width,
                                     series_legend_title, series_ncol, series_labels)
    if layout.fit_legends:
        stack_legends_below(ax, blocks, layout.font_scale * layout.legend_scale)
    else:
        (method_handles, method_title, method_ncol), series_block = blocks
        method_rows = -(-len(methods) // method_ncol)
        ax.add_artist(legend_below(ax, method_handles, method_title, ncol=method_ncol,
                                   y_offset=-0.10, font_scale=layout.font_scale))
        series_handles, title, series_ncol = series_block
        legend_below(ax, series_handles, title, ncol=series_ncol,
                     y_offset=-0.10 - 0.06 * (method_rows + 1),
                     font_scale=layout.font_scale)
    pcp.save_figure(fig, output_path, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = matplotlib.rcParamsDefault["hatch.linewidth"]


def plot_stacked_split_bars(panels, output_path, ylim=None,
                            series_order=None, series_colors=None,
                            series_legend_title="model", category_order=None,
                            series_ncol=None, series_labels=None, category_labels=None,
                            layout=DEFAULT_LAYOUT):
    """The same figure as plot_split_bars, one panel per model stacked vertically under a
    single pair of legends.

    `panels` is [(panel title, shares per series and method)], drawn top to bottom. The
    colour coding is what makes the panels comparable and it does not change between them,
    so repeating the legend under each one only costs height that the panels could have
    had -- the legends are drawn once, under the bottom panel.

    The panels share the x axis for the same reason: the categories are identical, so only
    the bottom panel spells them out."""
    panels = [(title, shares) for title, shares in panels if shares]
    if len(panels) < 2:
        return
    series, methods, categories = split_bar_dimensions(
        [shares for _, shares in panels], series_order, category_order)
    if not series or not methods or not categories:
        return

    color_for = series_color_lookup(series, series_colors)
    figure_width = max(18, len(categories) * len(series) * layout.bar_span)
    panel_height = (FIXED_PANEL_HEIGHT if layout.aspect is None
                    else figure_width / layout.aspect)
    plt.rcParams["hatch.linewidth"] = 1.0 * layout.font_scale
    fig, axes = plt.subplots(len(panels), 1, figsize=(figure_width, panel_height * len(panels)),
                             sharex=True)
    tallest = 0.0
    for index, ((title, shares), ax) in enumerate(zip(panels, axes)):
        tallest = max(tallest, draw_split_bar_panel(
            ax, shares, series, methods, categories, color_for, layout, ylim,
            category_labels, show_xticklabels=index == len(panels) - 1))
        ax.set_title(title, fontsize=TITLE_FONTSIZE * layout.font_scale)
    if ylim is None:
        # One scale across the panels whatever the caller passed: panels that do not share
        # a y axis cannot be read against each other, which is the whole point of stacking.
        for ax in axes:
            ax.set_ylim(0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM))

    blocks = split_bar_legend_blocks(series, methods, color_for, layout, figure_width,
                                     series_legend_title, series_ncol, series_labels)
    # The gap is a fraction of the figure height, and the figure is as tall as the panels
    # it stacks -- left alone it would open a panel-sized hole above the legends.
    stack_legends_below(axes[-1], blocks, layout.font_scale * layout.legend_scale,
                        gap=0.04 / len(panels))
    pcp.save_figure(fig, output_path, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = matplotlib.rcParamsDefault["hatch.linewidth"]


def display_name(key, labels):
    return key if labels is None else labels.get(key, key)


def stacked_axis_label(methods):
    # Kept short: at this type size a longer label runs past the top of the panel.
    return f"{ARGMAX_SHARE_LABEL} (%)" if len(methods) == 1 else "Mean share (%)"


def plot_cross_model_argmax_shares(shares_by_model, dataset_plots_dir, ylim, average_ylim):
    """One grouped-bar figure per scope, one bar per model, as
    plot_classified_parties draws its cross-model distributions."""
    for method, variant_label, suffix, scope_label in cross_model_scopes(shares_by_model):
        # An averaged scope stacks all four methods inside each model's bar; a
        # single-method scope goes through the same renderer with one segment, so the
        # whole family keeps one look.
        per_model_and_method = scoped_share_per_model_and_method(
            shares_by_model, variant_label, methods=None if method is None else [method])
        if len(per_model_and_method) < 2:
            print(f"  only {len(per_model_and_method)} model(s) for '{scope_label}', skipping.")
            continue
        plot_split_bars(
            per_model_and_method,
            dataset_plots_dir / f"argmax_all_models_party_distribution{suffix}.png",
            average_ylim if method is None else ylim)


def party_positions_frame(model_dirs, positions):
    """The euandi positions the two VAA methods are scored against, collapsed to
    ECR+ID when the model's vaa*.csv files say evaluate_euandi.py was run that way --
    they are the only record of it, and the classifier labels are collapsed too."""
    party_df = load_party_positions(positions_path(positions), positions)
    every_vaa_csv = [
        path for model_dir in model_dirs
        for path in pcp.discover_vaa_csvs(model_dir).values()
    ]
    if every_vaa_csv and pcp.vaa_groups_are_collapsed(every_vaa_csv):
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    return party_df.dropna(subset=["ep_group"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    # Several names, so the stacked panels can be redrawn without reprocessing every model.
    parser.add_argument("--model", nargs="+", default=None)
    parser.add_argument("--panel-models", nargs="+", default=list(DEFAULT_PANEL_MODELS),
                        help="Models to stack, one per panel, in the combined per-language "
                             "figure that shares a single set of legends. Given in panel "
                             "order, top to bottom.")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Which euandi answers stand for an EP group in the two VAA "
                             "methods; must match the basis evaluate_euandi.py was run with.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
    if args.model:
        model_dirs = [d for d in model_dirs if d.name in set(args.model)]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    party_df = party_positions_frame(model_dirs, args.positions)
    shares_by_model, topics_by_model, languages_by_model, topic_counts_by_model = {}, {}, {}, {}
    for model_dir in model_dirs:
        shares, per_axis, per_language, topic_counts = collect_model_shares(
            model_dir, party_df, args.dataset)
        if shares:
            shares_by_model[model_dir] = shares
            topics_by_model[model_dir] = per_axis
            languages_by_model[model_dir] = per_language
            topic_counts_by_model[model_dir] = topic_counts
    if not shares_by_model:
        raise SystemExit("No model produced argmax shares.")

    ylim = share_limits(shares_by_model)
    average_ylim = average_limits(shares_by_model)
    topic_ylim = stacked_series_limits(topics_by_model)
    # One limit for both per-language figures: they draw the same stacks, regrouped.
    language_ylim = stacked_series_limits(languages_by_model)
    print(f"\nShared y-axis: per method 0.0% .. {ylim[1]:.1f}%, "
          f"stacked averages 0.0% .. {average_ylim[1]:.1f}%, "
          f"per topic 0.0% .. {topic_ylim[1]:.1f}%, "
          f"per language 0.0% .. {language_ylim[1]:.1f}%")

    for model_dir, shares in shares_by_model.items():
        print(f"\nProcessing: {model_dir.name}")
        plot_model(model_dir, shares, topics_by_model[model_dir],
                   languages_by_model[model_dir], topic_counts_by_model[model_dir],
                   ylim, topic_ylim, language_ylim)

    if len(shares_by_model) > 1:
        print("\nCross-model argmax shares:")
        plot_cross_model_argmax_shares(
            shares_by_model, results_dir / "plots", ylim, average_ylim)

    # Keyed by name so the panels come out in the order they were asked for rather than
    # in the order the directories were listed.
    by_name = {model_dir.name: model_dir for model_dir in languages_by_model}
    panel_dirs = [by_name[name] for name in args.panel_models if name in by_name]
    if panel_dirs:
        print("\nStacked per-language panels:")
        stacked_language_panels(languages_by_model, panel_dirs, results_dir / "plots",
                                language_ylim)

    records = [
        record
        for model_dir, shares in shares_by_model.items()
        for record in summary_records(model_dir.name, shares)
    ]
    summary_path = results_dir / "plots" / "argmax_share_methods.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(summary_path, index=False)
    print(f"\n  Wrote {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
