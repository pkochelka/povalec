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
import itertools
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
    # Floor for the computed figure width. Width is half of the type-to-figure ratio that
    # survives the reduction, so a figure with few bars is better off narrow than padded
    # out to a width its bars do not need. Defaults to the 18 inches every existing figure
    # was drawn at, so only a layout that opts out changes.
    min_width: float = 18.0
    # Applied to the X tick labels on top of font_scale. They are the one text set at an
    # angle and anchored at its right end, so the leftmost label hangs off the side of the
    # axes -- and bbox_inches="tight" then widens the canvas to keep it, which shrinks the
    # panels rather than the label. Trimming just this text fixes that without shrinking
    # the y-axis numbers, which have no such problem.
    xtick_scale: float = 1.0
    # The same for the Y tick labels and the Y axis label together. Both sit in the left
    # margin, and the margin is taken out of the panel's width, so oversized y text costs
    # bar width across every category at once. Kept as one knob because the two are read
    # as a unit and shrinking only one of them looks like a mistake.
    yaxis_scale: float = 1.0
    # Whether the two legends are set beside each other rather than one above the other.
    # Stacked, neither of them fills the panel's width, and the pair costs the figure two
    # bands of height; beside each other they spend width that was there anyway and cost
    # one band. Only has an effect under `fit_legends`, which is what measures them, and
    # only taken when the pair actually fits -- see `place_legends_side_by_side`.
    side_by_side_legends: bool = False
    # How far below the panel's tick labels the legends hang, as a fraction of the figure's
    # height. Only read under `fit_legends`, which is what measures the distance; the
    # default is what every figure was drawn with before the panel figures were tightened.
    legend_gap: float = 0.04


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
# The same figure with one model per panel, which is the one the thesis prints and so the
# one whose height is worth spending effort on: two panels of it and the legends are some
# 40 inches, and every inch of that is page height at whatever width it goes in at. The
# legends are set a little smaller than the per-model figure's and hung closer under the
# panels -- the two together take about a fifth off the band, and the band is the one part
# of the figure that carries no data. Kept apart from COLUMN_LAYOUT rather than folded into
# it so the thirteen per-model figures, which are browsed rather than printed as a set, keep
# the legends they were drawn with.
LANGUAGE_PANEL_LAYOUT = COLUMN_LAYOUT._replace(legend_scale=0.68, legend_gap=0.012,
                                               side_by_side_legends=True)
# For the stacked per-topic panels, which are the opposite problem to the per-language
# ones: 6 EP groups by 7 topics is only 42 bars, so the figure does not need the 18-inch
# floor and is better off without it -- at ~10 inches the reduction to a ~3.3-inch column
# is ~3x rather than ~5.5x, and modest 1.15x type then lands the labels near 10pt, the
# size of the body text around them. `aspect` is what shortens the figure: the panel
# follows the width at 1:2.4 instead of taking the fixed 18-inch height, so two panels
# and their shared legends come to ~11 inches rather than the ~40 the default gives.
# `aspect` is the width : height of the panel's ALLOTMENT, not of the axes it ends up
# with -- the title, the rotated tick labels and the inter-panel spacing all come out of
# it. What sets the floor is the y-axis label: "Mean share (%)" at 37pt is ~2.6 inches
# of rotated text, so a panel allotted much less than 4.5 inches has its two panels'
# labels run into each other, which is exactly what 2.4 did before the legends were
# brought down to two short rows.
TOPIC_PANEL_LAYOUT = Layout(bar_span=0.24, aspect=2.2, font_scale=1.15, legend_scale=0.6,
                            tick_rotation=30, fit_legends=True, min_width=10.0,
                            xtick_scale=0.8, yaxis_scale=0.8, side_by_side_legends=True,
                            legend_gap=0.012)
# For the cross-model figures: 13 models by 6 EP groups is ~27 inches of width, and the
# fixed 18-inch height makes that a near-square panel in which the tallest bar still uses
# well under half the height. Following the width at 1:2.6 gives a ~10-inch panel that the
# bars fill, with the bar width -- and so the type-to-figure ratio -- left alone.
# `fit_legends` comes with it: the untouched fractions in `legend_below` are shares of the
# axes height, so on a panel this much shorter they place the legends through the tick
# labels; measured placement holds at any panel height.
CROSS_MODEL_LAYOUT = DEFAULT_LAYOUT._replace(aspect=2.6, fit_legends=True,
                                             side_by_side_legends=True,
                                             legend_scale=0.85, legend_gap=0.012)
# For the per-topic grid. A panel is the same 42 bars as TOPIC_PANEL_LAYOUT's, but the
# grid puts two of them across the page instead of one, so each is reduced into ~half the
# text width -- ~3.3x rather than the stacked figure's ~3x. That is close enough to keep
# the type-to-figure ratio, and the grid gets some of it back: x tick labels are drawn on
# one panel per column and the y axis label on one per row, so a panel spends less of its
# height and width on furniture and can be drawn a little flatter.
# `fit_legends` is off because the grid does not hang its legends below the panels at all
# -- they go in the cell the eleven panels leave over. `side_by_side_legends` goes with it:
# the spare cell is one panel wide and a whole row tall, so height is the one thing the
# legends there are not short of, and side by side they would only have half the width.
# `legend_scale` comes back up with them: the stacked figure's legends were made smaller to
# save page height, and the cell here costs the figure no height to begin with.
TOPIC_GRID_LAYOUT = TOPIC_PANEL_LAYOUT._replace(aspect=2.5, fit_legends=False,
                                                side_by_side_legends=False,
                                                legend_scale=0.7, legend_gap=0.04)

# The two models the thesis prints the per-language figure for, in panel order. Any other
# pair is one --panel-models away; these are only the default.
DEFAULT_PANEL_MODELS = ("grok-4.5", "gemma-4-31b")
# The per-topic figure prints a different pair, so it gets its own flag rather than
# sharing --panel-models.
DEFAULT_TOPIC_PANEL_MODELS = ("granite-4.1-8b", "gemini3.5-flash")
# Rows x columns of the per-topic grid, and the models it covers: everything except the
# pair above, which already has the stacked figure of its own. Eleven models fill this
# shape leaving one cell over, and that cell is where their shared legends go -- see
# grid_topic_panels.
TOPIC_GRID_SHAPE = (6, 2)
DEFAULT_TOPIC_GRID_EXCLUDED = DEFAULT_TOPIC_PANEL_MODELS
# Panel titles for the stacked figures, from the one map every figure in the project
# names its models by.
model_display_name = pcp.model_display_name


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


def legend_style(title, ncol, legend_font_scale):
    """The look every legend hung off a split-bar figure is drawn with.

    Kept in one place because the three placements below -- stacked under the panel, side
    by side under it, and in a grid's spare cell -- differ only in where the box goes, and
    a legend that looked different in one of them would read as meaning something else."""
    return dict(
        loc="upper center",
        fontsize=LEGEND_FONTSIZE * legend_font_scale, title=title,
        title_fontsize=LEGEND_TITLE_FONTSIZE * legend_font_scale, framealpha=0.9,
        ncol=ncol, handlelength=1.4, handletextpad=0.6, columnspacing=1.4, borderpad=0.6,
    )


def efficient_ncols(count):
    """The column counts worth considering for `count` entries: the narrowest one per
    distinct number of rows. Five entries in three columns and in four are both two rows
    tall, so only the three-column form is ever wanted."""
    narrowest_per_row_count = {}
    for ncol in range(1, count + 1):
        narrowest_per_row_count.setdefault(-(-count // ncol), ncol)
    return sorted(narrowest_per_row_count.values())


def measure_legend_size(figure, renderer, handles, title, ncol, legend_font_scale):
    """(width, height) of the legend in pixels, without leaving it on the figure.

    A legend shrink-wraps its contents, so neither measurement is something the number of
    columns predicts: matplotlib sizes each column to its own widest entry, and the entries
    here are anything from "es" to "classified open-ended". So it is drawn, measured and
    removed rather than estimated."""
    legend = figure.legend(
        handles=handles, bbox_to_anchor=(0.0, 0.0, 1.0, 0.001),
        bbox_transform=figure.transFigure,
        **legend_style(title, ncol, legend_font_scale),
    )
    extent = legend.get_window_extent(renderer)
    legend.remove()
    return float(extent.width), float(extent.height)


def side_by_side_shape(figure, renderer, blocks, legend_font_scale, room, stacked_height):
    """(columns per block, width per block) in pixels for legends set in one row, or None
    if a row is no use here.

    Each block is re-wrapped rather than kept at the number of columns it asked for: those
    were chosen for a legend with the whole width to run across, and beside each other the
    blocks have a share of it each. The shape picked is the shortest that fits `room` --
    the row is as tall as the taller legend in it, and height is the only thing a row saves
    -- and the narrowest among equally short ones, so the pair stays compact rather than
    sprawling to the edges.

    None when nothing fits, and equally when the best row is no shorter than the
    `stacked_height` it would replace. The second is the case a wide legend puts the figure
    in: 24 languages that ran across seven columns have to come down to three or four to
    sit beside anything, and eight rows of them are taller than the two boxes they were
    meant to save."""
    options = [efficient_ncols(len(handles)) for handles, _, _ in blocks]
    sizes = {
        (index, ncol): measure_legend_size(figure, renderer, handles, title, ncol,
                                           legend_font_scale)
        for index, ((handles, title, _), ncols) in enumerate(zip(blocks, options))
        for ncol in ncols
    }
    best = None
    for combination in itertools.product(*options):
        chosen = [sizes[(index, ncol)] for index, ncol in enumerate(combination)]
        width = sum(size[0] for size in chosen)
        height = max(size[1] for size in chosen)
        if width > room or height >= stacked_height:
            continue
        if best is None or (height, width) < best[0]:
            best = ((height, width), list(combination), [size[0] for size in chosen])
    return (best[1], best[2]) if best else None


def place_legends_side_by_side(figure, renderer, panel, top, blocks, legend_font_scale,
                               gap=0.02, margin=0.005):
    """Set the legends in one row under the panel, centred on it. True if they fit.

    `gap` and `margin` are fractions of the figure's width: the space kept between the
    legends, and the space kept outside them. The row is measured against the FIGURE rather
    than the panel it hangs under -- the panel is inset by its y axis label and its margins,
    and a row is worth the width that buys even though it then overhangs the panel a little
    at each end. Past the figure's own edge is where it stops: `bbox_inches="tight"` would
    widen the whole canvas to keep an overhanging legend, and every inch added that way is
    an inch the type has to grow again to survive the reduction to a column.

    False leaves the figure untouched, so the caller can fall back to stacking -- see
    `side_by_side_shape` for when a row is refused."""
    figure_pixels = float(figure.bbox.width)
    gap_pixels = gap * figure_pixels
    stacked_height = sum(
        measure_legend_size(figure, renderer, handles, title, ncol, legend_font_scale)[1]
        for handles, title, ncol in blocks
    )
    shape = side_by_side_shape(
        figure, renderer, blocks, legend_font_scale,
        figure_pixels * (1 - 2 * margin) - gap_pixels * (len(blocks) - 1), stacked_height)
    if shape is None:
        return False
    ncols, widths = shape

    total = sum(widths) + gap_pixels * (len(blocks) - 1)
    # Centred on the panel where there is room for that, and pushed back inside the figure
    # where there is not -- an off-centre row reads as a mistake, a clipped one as damage.
    centred = figure_pixels * (panel.x0 + panel.width / 2) - total / 2
    left = min(max(centred, margin * figure_pixels),
               figure_pixels * (1 - margin) - total)
    for (handles, title, _), ncol, width in zip(blocks, ncols, widths):
        figure.legend(
            handles=handles,
            bbox_to_anchor=(left / figure_pixels, top, width / figure_pixels, 0.001),
            bbox_transform=figure.transFigure,
            **legend_style(title, ncol, legend_font_scale),
        )
        left += width + gap_pixels
    return True


def stack_legends_below(ax, blocks, legend_font_scale, gap=0.04, side_by_side=False):
    """Put the legends under the axes, each one placed below whatever is already drawn.

    `side_by_side` tries them in one row first, and only stacks them if that row cannot be
    made to fit the panel -- see `place_legends_side_by_side`.

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
    if side_by_side and place_legends_side_by_side(figure, renderer, panel, lowest - gap,
                                                   blocks, legend_font_scale):
        return
    for handles, title, ncol in blocks:
        legend = figure.legend(
            handles=handles,
            bbox_to_anchor=(panel.x0, lowest - gap, panel.width, 0.001),
            bbox_transform=figure.transFigure,
            **legend_style(title, ncol, legend_font_scale),
        )
        figure.canvas.draw()
        lowest = bottom_in_figure(legend)


def lowest_in_figure(artists, renderer, to_figure):
    """The bottom of the lowest of `artists`, in figure coordinates."""
    return min(float(to_figure.transform((0.0, artist.get_window_extent(renderer).y0))[1])
               for artist in artists)


def visible_xticklabels(ax):
    return [label for label in ax.get_xticklabels()
            if label.get_visible() and label.get_text()]


def legends_in_cell(ax, blocks, legend_font_scale, overhang_from=None, gap=0.006):
    """Stack the shared legends inside a spare grid cell instead of under the panels.

    A grid that does not divide by its number of panels leaves one cell empty, and that
    cell is already the width of a panel and the height of a row -- more room than the
    legends need, and room that costs the figure nothing because it is there either way.
    Legends hung below the grid instead would add their own band of height to a figure that
    is already six rows tall, and on the page that band is taken out of the panels.

    `overhang_from` is the panel directly above the cell. Its x tick labels are drawn
    outside the row's own allotment (see plot_grid_split_bars) and so hang down into this
    cell, and the legends start below wherever they end rather than at the top of the cell.

    Placed by measurement, in figure coordinates, for the reasons `stack_legends_below`
    sets out: the fractions are shares of an axes height chosen for the default type size
    and go wrong once it is scaled, while a figure-level legend is left where it was put by
    the layout and still taken into the saved bounding box."""
    figure = ax.figure
    ax.axis("off")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()

    cell = ax.get_position()
    top = cell.y1
    if overhang_from is not None and (labels := visible_xticklabels(overhang_from)):
        top = min(top, lowest_in_figure(labels, renderer, to_figure) - gap)
    for handles, title, ncol in blocks:
        legend = figure.legend(
            handles=handles,
            bbox_to_anchor=(cell.x0, top, cell.width, 0.001),
            bbox_transform=figure.transFigure,
            **legend_style(title, ncol, legend_font_scale),
        )
        figure.canvas.draw()
        top = lowest_in_figure([legend], renderer, to_figure) - gap


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
    only one model answered in still keeps its slot in the other panel.

    `ylim` is expected to be the limit of THESE models rather than of the whole run, so
    the panels are not stretched to a height neither of them reaches; see main()."""
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
        series_ncol=min(len(languages), 7), layout=LANGUAGE_PANEL_LAYOUT,
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
        series_legend_title="EP group", category_order=languages,
        layout=LANGUAGE_PANEL_LAYOUT,
    )


def stacked_topic_panels(topics_by_model, topic_counts_by_model, panel_model_dirs,
                         dataset_plots_dir, ylim):
    """The per-topic figure with one model per panel and one set of legends.

    Only the transposed view (EP groups coloured, topics along the x axis): that is the
    figure the thesis prints, and the two legends it repeats -- four methods and six EP
    groups -- cost more of the page than either panel's bars do.

    `ylim` is expected to be the limit of THESE models rather than of the whole run, so
    the panels are not stretched to a height neither of them reaches; see main()."""
    panels, axes_present, merged_counts, parties = topic_panel_inputs(
        topics_by_model, topic_counts_by_model, panel_model_dirs)
    if len(panels) < 2:
        print(f"  only {len(panels)} model(s) for the stacked per-topic panels, skipping.")
        return
    print(f"  stacked panels: {', '.join(title for title, _ in panels)}")

    plot_stacked_split_bars(
        panels, dataset_plots_dir / "argmax_share_topics_by_topic_pooled_panels.png", ylim,
        series_order=parties,
        series_colors={party: pcp.color_for_party(party) for party in parties},
        series_legend_title="EP group", category_order=axes_present,
        category_labels=topic_labels_with_counts(axes_present, merged_counts),
        # Three across, not six: a legend wider than the panels makes bbox_inches="tight"
        # widen the whole figure to fit it, and every inch of width added that way is an
        # inch the type has to be scaled up again to survive the reduction. Two short rows
        # under the panels cost less than that.
        series_ncol=3,
        layout=TOPIC_PANEL_LAYOUT,
        # Shorter than the default "Mean share (%)" so the panels can be: the rotated label
        # is what stops two stacked panels being drawn any shorter than it is tall, and the
        # caption carries what the share is a mean over.
        ylabel="Share (%)",
    )


def topic_panel_inputs(topics_by_model, topic_counts_by_model, model_dirs):
    """(panels, axes present, topic counts, EP groups) shared by both per-topic
    multi-model figures.

    The axes and the groups are taken over all the models together rather than per model,
    so a topic or a group only some of them produced still holds its slot everywhere and
    the panels stay column-for-column comparable -- which is the only reason to put them
    on one page."""
    panels = [(model_display_name(model_dir), transposed_shares(topics_by_model[model_dir]))
              for model_dir in model_dirs if topics_by_model.get(model_dir)]
    axes_present = [axis for axis in AXES
                    if any(axis in topics_by_model[model_dir] for model_dir in model_dirs
                           if topics_by_model.get(model_dir))]
    # The counts come off the same questionnaire either way, so merging only avoids
    # depending on which model happens to be listed first.
    merged_counts = {}
    for model_dir in model_dirs:
        merged_counts.update(topic_counts_by_model.get(model_dir, {}))
    parties = pcp.ordered_parties_present({party for _, shares in panels for party in shares})
    return panels, axes_present, merged_counts, parties


def grid_topic_panels(topics_by_model, topic_counts_by_model, grid_model_dirs,
                      dataset_plots_dir, ylim, shape=TOPIC_GRID_SHAPE):
    """The per-topic figure for every remaining model, as one grid.

    One panel per model, filled row-major in the order the models are given. Eleven models
    in a 6x2 grid leave one cell over, and the legends go in it: the coding is the same in
    all eleven panels, so it is drawn once, in space the grid was giving up anyway.

    Every panel is on the ylim of all the models together, so any bar can be read against
    any other."""
    panels, axes_present, merged_counts, parties = topic_panel_inputs(
        topics_by_model, topic_counts_by_model, grid_model_dirs)
    if not panels:
        print("  no models for the per-topic grid, skipping.")
        return
    print(f"  grid panels: {', '.join(title for title, _ in panels)}")
    plot_grid_split_bars(
        panels,
        dataset_plots_dir / "argmax_share_topics_by_topic_pooled_grid.png",
        ylim, shape=shape,
        series_order=parties,
        series_colors={party: pcp.color_for_party(party) for party in parties},
        series_legend_title="EP group", category_order=axes_present,
        category_labels=topic_labels_with_counts(axes_present, merged_counts),
        series_ncol=3,
        layout=TOPIC_GRID_LAYOUT,
        ylabel="Share (%)",
        legends_in_spare_cell=True,
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
                         show_xticklabels=True, ylabel=None, show_ylabel=True):
    """One panel of grouped, method-stacked bars. Returns the tallest stack drawn, so a
    caller that did not fix `ylim` can size the panel to what it got.

    `ylabel` overrides the default wording. Rotated text is the one thing on a panel whose
    height does not follow the panel's, so on a short panel the label is what sets the
    floor: at 37pt "Mean share (%)" is ~2.6 inches of it, and two panels shorter than that
    run their labels into each other.

    `show_ylabel` drops it entirely, which is what a grid's right-hand column wants: the
    panels share one scale, so the label is the same text twice across the page, and the
    width it costs comes out of the bars."""
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
                           fontsize=TICK_FONTSIZE * layout.font_scale * layout.xtick_scale)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y",
                   labelsize=TICK_FONTSIZE * layout.font_scale * layout.yaxis_scale)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    for boundary in category_positions[:-1] + 0.5:
        ax.axvline(boundary, color="black", linewidth=0.5 * layout.font_scale, alpha=0.15)
    if show_ylabel:
        ax.set_ylabel(ylabel or stacked_axis_label(methods),
                      fontsize=AXIS_LABEL_FONTSIZE * layout.font_scale * layout.yaxis_scale)
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
    figure_width = max(layout.min_width, len(categories) * len(series) * layout.bar_span)
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
        stack_legends_below(ax, blocks, layout.font_scale * layout.legend_scale,
                            gap=layout.legend_gap,
                            side_by_side=layout.side_by_side_legends)
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
                            layout=DEFAULT_LAYOUT, ylabel=None):
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
    figure_width = max(layout.min_width, len(categories) * len(series) * layout.bar_span)
    panel_height = (FIXED_PANEL_HEIGHT if layout.aspect is None
                    else figure_width / layout.aspect)
    plt.rcParams["hatch.linewidth"] = 1.0 * layout.font_scale
    fig, axes = plt.subplots(len(panels), 1, figsize=(figure_width, panel_height * len(panels)),
                             sharex=True)
    tallest = 0.0
    for index, ((title, shares), ax) in enumerate(zip(panels, axes)):
        tallest = max(tallest, draw_split_bar_panel(
            ax, shares, series, methods, categories, color_for, layout, ylim,
            category_labels, show_xticklabels=index == len(panels) - 1, ylabel=ylabel))
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
                        gap=layout.legend_gap / len(panels),
                        side_by_side=layout.side_by_side_legends)
    pcp.save_figure(fig, output_path, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = matplotlib.rcParamsDefault["hatch.linewidth"]


def last_panel_in_each_column(panel_count, ncols):
    """Indices of the bottom-most drawn panel of each column, row-major.

    Not simply the last row: a grid whose final cell holds the legends has its last column
    ending a row early, and that column's x tick labels have to move up with it or the
    topics go unnamed under half the figure."""
    return {max(index for index in range(panel_count) if index % ncols == column)
            for column in range(min(ncols, panel_count))}


def layout_rows_evenly(fig):
    """Lay the grid out with the x tick labels taken out of the measurement, so every row
    is spaced the same, and let the labels hang below their panel afterwards.

    `tight_layout` gives a grid ONE row spacing and sets it from the worst row. Only two
    panels here carry x tick labels -- the bottom of each column -- so measured as they
    are, one label's worth of height is opened between every pair of rows in the figure,
    and the panels pay for it eleven times over for a label drawn twice. Rows spaced
    unevenly instead, which is what a constrained layout gives, is worse to read: the eye
    takes the wider gap for a division between the models above and below it, and there
    isn't one.

    Hidden during the measurement, the labels cost no row anything and the rows come out
    evenly spaced. The two sets of labels then hang past the bottom of their own panel:
    the left column's into the margin, where `bbox_inches="tight"` grows the canvas to
    take them, and the right column's into the spare cell below it, which holds the
    legends and has the room -- see `legends_in_cell`."""
    hidden = [label for ax in fig.axes for label in visible_xticklabels(ax)]
    for label in hidden:
        label.set_visible(False)
    fig.tight_layout()
    for label in hidden:
        label.set_visible(True)


def widen_gutter_for_overhang(fig, flat_axes, ncols, pad=0.006):
    """Open the gap between the columns until the x tick labels fit in it.

    The labels are rotated and anchored at their right end, so a label reaches to the LEFT
    of the tick it belongs to -- for "Ukraine (2)" at this size, past the left edge of its
    own panel. In a single-column figure that overhang lands in the margin and
    `bbox_inches="tight"` simply keeps it. In a grid it lands on the panel in the next
    column along, which draws over it: the first topic under the right-hand column comes
    out with its first letters missing.

    So the gutter is measured against the overhang that has to fit in it and widened if it
    does not, which costs the panels a little width rather than costing a label. Nothing is
    done when it already fits, which is the usual case -- only the columns after the first
    can overhang onto anything."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()

    def leftmost(artist):
        return float(to_figure.transform((artist.get_window_extent(renderer).x0, 0.0))[0])

    overhang = 0.0
    for index, ax in enumerate(flat_axes):
        if index % ncols == 0:
            continue
        labels = visible_xticklabels(ax)
        if labels:
            overhang = max(overhang, ax.get_position().x0 - min(map(leftmost, labels)))
    if overhang <= 0:
        return

    columns = [ax.get_position() for ax in flat_axes[:ncols]]
    span = columns[-1].x1 - columns[0].x0
    gutter = min(right.x0 - left.x1 for left, right in zip(columns, columns[1:]))
    needed = overhang + pad
    if needed <= gutter:
        return
    # subplots_adjust keeps the margins tight_layout worked out and only re-divides what
    # is between them, so widening the gutter narrows the panels and moves nothing else.
    panel_width = (span - (ncols - 1) * needed) / ncols
    fig.subplots_adjust(wspace=needed / panel_width)


def plot_grid_split_bars(panels, output_path, ylim=None, *, shape, series_order=None,
                         series_colors=None, series_legend_title="model",
                         category_order=None, series_ncol=None, series_labels=None,
                         category_labels=None, layout=DEFAULT_LAYOUT, ylabel=None,
                         legends_in_spare_cell=False):
    """The same panels as `plot_stacked_split_bars`, laid out as a grid rather than a
    single column.

    A column of eleven panels is a figure some ten times taller than it is wide, which no
    page takes; `shape` (rows, columns) folds them into a block that does. The panels are
    filled row-major, so they read in the order they are given.

    The grid is what lets the repeated furniture go: every panel is the same categories on
    the same scale, so the x tick labels are drawn once per column and the y axis label
    once per row, and only the panels differ across the figure.

    `legends_in_spare_cell` puts the legends in the cell left over when the panels do not
    fill the grid. A figure without a spare cell then gets none at all -- which is the
    point when a set of panels is split across two figures printed together: the coding is
    identical, so it is stated once, in the one empty cell the split leaves behind."""
    panels = [(title, shares) for title, shares in panels if shares]
    if not panels:
        return
    nrows, ncols = shape
    if len(panels) > nrows * ncols:
        raise SystemExit(f"{len(panels)} panels do not fit a {nrows}x{ncols} grid "
                         f"({output_path.name}).")
    series, methods, categories = split_bar_dimensions(
        [shares for _, shares in panels], series_order, category_order)
    if not series or not methods or not categories:
        return

    color_for = series_color_lookup(series, series_colors)
    panel_width = max(layout.min_width, len(categories) * len(series) * layout.bar_span)
    panel_height = (FIXED_PANEL_HEIGHT if layout.aspect is None
                    else panel_width / layout.aspect)
    plt.rcParams["hatch.linewidth"] = 1.0 * layout.font_scale
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_width * ncols, panel_height * nrows))
    # Not sharex: with a shared axis the tick formatter is shared too, so whichever panel
    # set its labels last would speak for every column at once -- and here the columns do
    # not all end on the same row. Each panel fixes its own xlim and ticks instead.
    flat_axes = np.atleast_1d(axes).ravel()

    label_columns = last_panel_in_each_column(len(panels), ncols)
    tallest = 0.0
    for index, ((title, shares), ax) in enumerate(zip(panels, flat_axes)):
        tallest = max(tallest, draw_split_bar_panel(
            ax, shares, series, methods, categories, color_for, layout, ylim,
            category_labels, show_xticklabels=index in label_columns, ylabel=ylabel,
            show_ylabel=index % ncols == 0))
        ax.set_title(title, fontsize=TITLE_FONTSIZE * layout.font_scale)
    if ylim is None:
        # One scale across the grid whatever the caller passed, for the same reason the
        # stacked figure shares one: panels on different scales cannot be read against
        # each other, which is the whole point of putting them on one page.
        for ax in flat_axes[:len(panels)]:
            ax.set_ylim(0.0, min(100.0, tallest + pcp.PERCENT_HEADROOM))

    spare_axes = list(flat_axes[len(panels):])
    for ax in spare_axes:
        ax.axis("off")
    layout_rows_evenly(fig)
    widen_gutter_for_overhang(fig, flat_axes, ncols)

    if legends_in_spare_cell and spare_axes:
        blocks = split_bar_legend_blocks(series, methods, color_for, layout, panel_width,
                                         series_legend_title, series_ncol, series_labels)
        # The panel above the spare cell is the one whose x tick labels hang into it.
        above_index = len(panels) - ncols
        legends_in_cell(spare_axes[0], blocks, layout.font_scale * layout.legend_scale,
                        overhang_from=flat_axes[above_index] if above_index >= 0 else None)
    pcp.save_figure(fig, output_path, tight_layout=False, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = matplotlib.rcParamsDefault["hatch.linewidth"]


def display_name(key, labels):
    return key if labels is None else labels.get(key, key)


def stacked_axis_label(methods):
    # Kept short: at this type size a longer label runs past the top of the panel.
    return f"{ARGMAX_SHARE_LABEL} (%)" if len(methods) == 1 else "Mean share (%)"


def plot_cross_model_argmax_shares(shares_by_model, dataset_plots_dir, ylim):
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
            # The averaged scopes are sized to their own tallest bar. One limit across
            # them was set by whichever framing leans hardest on a group (the base one, at
            # ~42%), which left the top sixth of the pooled figure -- the one the thesis
            # prints -- empty. A single-method scope keeps the run-wide `ylim`, which is
            # what makes it comparable with the per-model figures.
            None if method is None else ylim,
            layout=CROSS_MODEL_LAYOUT)


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
    parser.add_argument("--topic-panel-models", nargs="+",
                        default=list(DEFAULT_TOPIC_PANEL_MODELS),
                        help="The same for the combined per-topic figure, which the thesis "
                             "prints for a different pair of models. Given in panel order, "
                             "top to bottom.")
    parser.add_argument("--topic-grid-models", nargs="+", default=None,
                        help="Models for the two per-topic grid figures, in panel order. "
                             "Defaults to every model processed except --topic-panel-models, "
                             "which already has a stacked figure of its own.")
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
    topic_ylim = stacked_series_limits(topics_by_model)
    # One limit for both per-language figures: they draw the same stacks, regrouped.
    language_ylim = stacked_series_limits(languages_by_model)
    print(f"\nShared y-axis: per method 0.0% .. {ylim[1]:.1f}%, "
          f"per topic 0.0% .. {topic_ylim[1]:.1f}%, "
          f"per language 0.0% .. {language_ylim[1]:.1f}%")

    for model_dir, shares in shares_by_model.items():
        print(f"\nProcessing: {model_dir.name}")
        plot_model(model_dir, shares, topics_by_model[model_dir],
                   languages_by_model[model_dir], topic_counts_by_model[model_dir],
                   ylim, topic_ylim, language_ylim)

    if len(shares_by_model) > 1:
        print("\nCross-model argmax shares:")
        plot_cross_model_argmax_shares(shares_by_model, results_dir / "plots", ylim)

    # Keyed by name so the panels come out in the order they were asked for rather than
    # in the order the directories were listed.
    by_name = {model_dir.name: model_dir for model_dir in shares_by_model}
    panel_dirs = [by_name[name] for name in args.panel_models if name in by_name]
    if panel_dirs:
        # As for the topic panels below: the run-wide limit is set by whichever of the
        # thirteen models leans hardest on one group, and these two reach ~36% of it.
        panel_language_ylim = stacked_series_limits(
            {model_dir: languages_by_model[model_dir] for model_dir in panel_dirs
             if model_dir in languages_by_model})
        print(f"\nStacked per-language panels (y-axis 0.0% .. "
              f"{panel_language_ylim[1]:.1f}% over these models alone):"
              if panel_language_ylim else "\nStacked per-language panels:")
        stacked_language_panels(languages_by_model, panel_dirs, results_dir / "plots",
                                panel_language_ylim)

    topic_panel_dirs = [by_name[name] for name in args.topic_panel_models if name in by_name]
    if topic_panel_dirs:
        # The limit of these two models rather than of the whole run: the global one is set
        # by whichever of the thirteen leans hardest on one group, which leaves this pair's
        # panels a third empty.
        panel_topic_ylim = stacked_series_limits(
            {model_dir: topics_by_model[model_dir] for model_dir in topic_panel_dirs
             if model_dir in topics_by_model})
        print(f"\nStacked per-topic panels (y-axis 0.0% .. "
              f"{panel_topic_ylim[1]:.1f}% over these models alone):"
              if panel_topic_ylim else "\nStacked per-topic panels:")
        stacked_topic_panels(topics_by_model, topic_counts_by_model, topic_panel_dirs,
                             results_dir / "plots", panel_topic_ylim)

    # Everything the stacked pair does not already print, unless asked for by name.
    grid_names = (args.topic_grid_models if args.topic_grid_models is not None
                  else [name for name in by_name
                        if name not in set(args.topic_panel_models)])
    grid_dirs = [by_name[name] for name in grid_names if name in by_name]
    if grid_dirs:
        # The grid's panels are read against each other and against nothing else, so it
        # gets its own limit for the same reason the stacked pair does.
        grid_topic_ylim = stacked_series_limits(
            {model_dir: topics_by_model[model_dir] for model_dir in grid_dirs
             if model_dir in topics_by_model})
        print(f"\nPer-topic grid over {len(grid_dirs)} model(s) (y-axis 0.0% .. "
              f"{grid_topic_ylim[1]:.1f}% over these models alone):"
              if grid_topic_ylim else f"\nPer-topic grid over {len(grid_dirs)} model(s):")
        grid_topic_panels(topics_by_model, topic_counts_by_model, grid_dirs,
                          results_dir / "plots", grid_topic_ylim)

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
