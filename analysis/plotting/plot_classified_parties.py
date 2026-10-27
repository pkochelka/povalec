import argparse
import re
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.core.labels import use_short_party_labels
use_short_party_labels()   # cluster nicknames in every label (utils.PARTY_SHORT)
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr

from analysis.core import is_output_dir, model_display_name
from analysis.vaa_null_model import DEFAULT_SIGMA, SUBTRACTED_NULLS
from analysis.vaa_scoring import SCORE_NOUN, vaa_scoring
from utils import FALLBACK_PARTY_COLOR, PARTY_COLORS, PARTY_DISPLAY_ORDER, VARIANT_LABELS, VARIANT_PATTERN

from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS, POSITION_CHOICES, load_party_positions, positions_path,
)
from analysis.vaa_agreement_ci import (
    agreement_matrices, bootstrap_group_means, stance_frame_for,
)

VARIANT_SUFFIX_TO_LABEL = VARIANT_LABELS
VARIANT_LABEL_TO_SUFFIX = {label: suffix for suffix, label in VARIANT_SUFFIX_TO_LABEL.items()}
VARIANT_LABEL_ORDER = ["base", "negated", "question"]
from analysis.core.paths import CLASSIFIED_TRACKS  # noqa: E402

SOURCE_LABELS = [s for s in ["speeches", "reasons"] if s in CLASSIFIED_TRACKS]
SOURCE_DISPLAY = {"speeches": "open-ended", "reasons": "reasons"}
# Top-to-bottom order of the stacked per-source panels, which is the reverse of
# SOURCE_LABELS: the printed figure leads with the reasons track.
SOURCE_PANEL_ORDER = [s for s in ["reasons", "speeches"] if s in CLASSIFIED_TRACKS]
VAA_SOURCE_FOR_CLASSIFIER_SOURCE = {"speeches": "vaa_speeches", "reasons": "vaa_likert"}
# vaa_agreement_ci.py names the raw answer files by how they were elicited, not by
# what the classifier later reads out of them.
STANCE_SOURCE_FOR_CLASSIFIER_SOURCE = {"speeches": "speeches", "reasons": "likert"}
# The two VAA tracks under the source key their replicates carry, in panel order, named
# the way plot_argmax_shares names the methods.
VAA_TRACK_DISPLAY = {"reasons": "Direct Likert", "speeches": "Indirect Likert"}

SPEECHES_CLASSIFIED_PATTERN = re.compile(
    r"^speeches_[a-z]{2}(?:,[a-z]{2})*" + VARIANT_PATTERN + r"_classified\.csv$"
)
REASONS_CLASSIFIED_PATTERN = re.compile(
    r"^(?!speeches_)[a-z]{2}(?:,[a-z]{2})*" + VARIANT_PATTERN + r"_classified\.csv$"
)
VAA_LIKERT_PATTERN = re.compile(
    r"^vaa" + VARIANT_PATTERN + r"_(?P<languages>[a-z]{2}(?:,[a-z]{2})*)\.csv$"
)
VAA_SPEECHES_PATTERN = re.compile(
    r"^vaa_speeches" + VARIANT_PATTERN + r"_(?P<languages>[a-z]{2}(?:,[a-z]{2})*)\.csv$"
)

PREDICTED_PARTY_COLUMN_PATTERN = re.compile(
    r"^predicted_party_(?P<language>[a-z]{2})" + VARIANT_PATTERN + r"_v(?P<variant_idx>\d+)$"
)
PARTY_PROBABILITY_COLUMN_PATTERN = re.compile(
    r"^party_prob_(?P<party_slug>.+?)_(?P<language>[a-z]{2})" + VARIANT_PATTERN + r"_v(?P<variant_idx>\d+)$"
)


MEAN_PROBABILITY_LABEL = "Mean probability"
MEAN_AGREEMENT_LABEL = "Mean VAA agreement"
# The cross-model VAA panels' title and axis label; main() switches them to the cluster
# probability when the VAA stances are scored as Gaussian posteriors (vaa_scoring).
VAA_PANEL_TITLE = "VAA agreement"
VAA_MEAN_LABEL = MEAN_AGREEMENT_LABEL
VAA_DELTA_LABEL = "Δ agreement"
# The same for the classifier panels of the combined figure: the classifier's labels are
# the positions basis' groups, so on the cluster basis its probability is a cluster's.
CLASSIFIER_PANEL_TITLE = "Classifier probability"
LEGEND_UPPER_RIGHT = "upper right"

BOX_MEDIAN_STYLE = {"color": "black", "linewidth": 1.3}

# Sized for a figure that is reduced into a LaTeX column rather than browsed as a
# PNG: the reduction scales the type with everything else, so the only thing that
# survives it is the RATIO of type to figure -- the same reasoning the Layout
# record in plot_argmax_shares.py sets out. These figures are drawn 9-14 inches
# wide and land in a ~3.3-inch column, so a 2.5-3x reduction; the old 7-10pt
# labels came out at 3pt on the page, which is what this block fixes. At roughly
# 20pt on a 9-inch figure they land near 7pt printed, and the constants are
# deliberately shared so no one figure drifts out of step with the others.
TICK_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 22
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 17
LEGEND_TITLE_FONTSIZE = 18
# The per-bar value annotations sit between neighbouring bars, so they are the one
# text that cannot simply grow -- kept a step down and dropped entirely from the
# figures whose bars are too narrow to hold them.
VALUE_FONTSIZE = 14
ANNOTATION_FONTSIZE = 17
HEATMAP_VALUE_FONTSIZE = 13
FOOTNOTE_FONTSIZE = 11

# At this type size a many-entry legend no longer fits over the bars without
# covering them (the model legend is the worst case: up to 16 entries), so those
# legends hang under the axes instead and the figure is saved with a tight bbox so
# the extra rows are not cropped. Legends of two or three entries still fit in the
# corner and stay there.
LEGEND_BELOW = dict(loc="upper center", fontsize=LEGEND_FONTSIZE,
                    title_fontsize=LEGEND_TITLE_FONTSIZE, framealpha=0.9)

# The combined VAA + classifier figure is the one the paper prints a single column wide
# (\linewidth of a two-column page, ~3 inches), and at the 13 inches the other figures
# here are drawn at, its 20pt type landed under 5pt -- half the caption below it. The
# type cannot grow to make up for it (the four level party labels already fill their
# slots), so the figure is drawn narrower instead: at 7.5 inches the reduction is 2.5x
# and the same 20pt prints at 8pt, next to a 10pt caption. The width grows with the
# party count so a basis with more groups than the four clusters keeps the same slot
# per label, at the price of printing smaller.
COMBINED_FIGURE_WIDTH = 7.5
COMBINED_WIDTH_PER_PARTY = 1.85
COMBINED_PANEL_HEIGHT = 3.7
# Titles and party labels are set at the tick size: it is the one size that fits the
# party labels, and a figure this small reads better in few sizes than many.
COMBINED_FONTSIZE = TICK_FONTSIZE
# The y axis is the exception, label and tick numbers a step down: the panel titles
# already name what the bars are, so the label only has to carry the unit, and at the
# smaller size it fits the panel's height on ONE line -- the width a second line and
# wider numbers would take goes to the bars.
COMBINED_YAXIS_FONTSIZE = 16
# The model names sit at the y axis' size rather than the titles': a dozen of them at
# the tick size made the legend the heaviest block of type in the figure.
COMBINED_LEGEND_FONTSIZE = COMBINED_YAXIS_FONTSIZE
# The bars of a group fill more of its slot than GROUP_WIDTH gives the wider figures: with
# a dozen models to a group a bar prints well under a millimetre wide, and the gap
# between groups still reads at a bar and a bit.
COMBINED_GROUP_WIDTH = 0.92
# How far a panel title or y label may be stepped down to stay inside its panel (see
# shrink_texts_to_fit); only the split figure's longer titles get anywhere near it.
COMBINED_SMALLEST_FONTSIZE = 14
# The legend hangs from the bottom panel's axes, so a fraction of ONE panel's height:
# enough to clear the level party labels under it.
COMBINED_LEGEND_OFFSET = -0.14
# Level party labels: the figure has one x axis for all its panels and the groups are few
# and short, so they fit their slots without the tilt the other bar figures need.
COMBINED_TICK_ROTATION = 0

# Marks and gaps trimmed well in from matplotlib's defaults, which spend some 4.8 em per
# entry on the handle and the column gap alone. The entries are colour swatches next to a
# model name, so the mark carries no shape worth 2 em; what the trim buys is a second
# column at this type size, and rows close enough that the legend stays shorter than a panel.
COMBINED_LEGEND_STYLE = dict(loc="upper center", framealpha=0.9, handlelength=1.2,
                             handletextpad=0.4, columnspacing=1.0, borderpad=0.4,
                             labelspacing=0.3)

# The width the constants above are calibrated against. One figure here -- the
# per-model box panel -- sizes itself by (parties x models) and runs to 26 inches
# with a full model roster, so the same 20pt lands at half the printed size the
# other figures get. type_scale keeps the type-to-figure ratio fixed instead.
NOMINAL_FIGURE_WIDTH = 13.0


def type_scale(figure_width):
    """Multiplier keeping type proportional to a figure wider than the nominal one.
    Never shrinks type: a figure narrower than nominal is already legible."""
    return max(1.0, figure_width / NOMINAL_FIGURE_WIDTH)


def slugify_party_label(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


# Stamped on every saved figure while set; plot_topical_parties.py uses it to mark
# which statements a per-topic figure was drawn from, since the helpers below take
# no title argument.
FIGURE_FOOTNOTE = None


SAVE_ATTEMPTS = 5


def save_figure(fig, output_path, tight_layout=True, **savefig_kwargs):
    """`tight_layout=False` for a figure laid out some other way -- calling it on a
    constrained-layout figure replaces the engine and undoes what it did."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if FIGURE_FOOTNOTE:
        fig.text(0.99, 0.005, FIGURE_FOOTNOTE, ha="right", va="bottom",
                 fontsize=FOOTNOTE_FONTSIZE, color="#666666")
    if tight_layout:
        fig.tight_layout()
    # An image open in a viewer is briefly locked on Windows while the viewer reloads it,
    # and the write fails with EINVAL/EACCES; a short wait is enough, and one open preview
    # should not abort a run that has a hundred more figures to draw.
    for attempt in range(SAVE_ATTEMPTS):
        try:
            fig.savefig(output_path, dpi=300, **savefig_kwargs)
            break
        except OSError:
            if attempt == SAVE_ATTEMPTS - 1:
                raise
            time.sleep(2.0)
    plt.close(fig)
    print(f"  Saved {output_path}")


def color_for_party(party):
    return PARTY_COLORS.get(party, FALLBACK_PARTY_COLOR)


def ordered_parties_present(parties_present):
    ordered = [party for party in PARTY_DISPLAY_ORDER if party in parties_present]
    extras = sorted(party for party in parties_present if party not in PARTY_DISPLAY_ORDER)
    return ordered + extras


def discover_classified_csvs(model_dir):
    classified_by_source_and_variant = {}
    for path in model_dir.glob("*_classified.csv"):
        if (match := SPEECHES_CLASSIFIED_PATTERN.match(path.name)):
            source = "speeches"
        elif (match := REASONS_CLASSIFIED_PATTERN.match(path.name)):
            source = "reasons"
        else:
            continue
        if source not in CLASSIFIED_TRACKS:
            continue
        variant_label = VARIANT_SUFFIX_TO_LABEL[match.group("variant")]
        classified_by_source_and_variant[(source, variant_label)] = path
    return classified_by_source_and_variant


LONG_PREDICTION_COLUMNS = ["row_idx", "language", "variant_idx", "party", "probability", "predicted_party"]


def index_classified_columns(df):
    probability_columns_by_key = {}
    predicted_columns_by_key = {}
    parties_by_slug = {}

    for column in df.columns:
        if (match := PARTY_PROBABILITY_COLUMN_PATTERN.match(column)):
            language = match.group("language")
            variant_idx = int(match.group("variant_idx"))
            party_slug = match.group("party_slug")
            probability_columns_by_key.setdefault((language, variant_idx), {})[party_slug] = column
            parties_by_slug.setdefault(party_slug, party_slug)
        elif (match := PREDICTED_PARTY_COLUMN_PATTERN.match(column)):
            language = match.group("language")
            variant_idx = int(match.group("variant_idx"))
            predicted_columns_by_key[(language, variant_idx)] = column

    for party_column in predicted_columns_by_key.values():
        for label in df[party_column].dropna().unique():
            parties_by_slug[slugify_party_label(label)] = label

    return probability_columns_by_key, predicted_columns_by_key, parties_by_slug


def probability_records_for_text(df, row_idx, language, variant_idx, probability_columns,
                                 parties_by_slug, predicted_party_label):
    records = []
    for party_slug, party_column in probability_columns.items():
        probability = df.at[row_idx, party_column]
        if pd.isna(probability):
            continue
        records.append({
            "row_idx": row_idx,
            "language": language,
            "variant_idx": variant_idx,
            "party": parties_by_slug[party_slug],
            "probability": float(probability),
            "predicted_party": predicted_party_label,
        })
    return records


def predicted_only_record(row_idx, language, variant_idx, predicted_party_label):
    return {
        "row_idx": row_idx,
        "language": language,
        "variant_idx": variant_idx,
        "party": predicted_party_label,
        "probability": np.nan,
        "predicted_party": predicted_party_label,
    }


def predicted_party_or_none(df, row_idx, predicted_column):
    if predicted_column is None:
        return None
    value = df.at[row_idx, predicted_column]
    return value if isinstance(value, str) else None


def load_long_predictions(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    probability_columns_by_key, predicted_columns_by_key, parties_by_slug = index_classified_columns(df)

    if not probability_columns_by_key and not predicted_columns_by_key:
        return pd.DataFrame(columns=LONG_PREDICTION_COLUMNS)

    records = []
    for (language, variant_idx) in sorted(set(probability_columns_by_key) | set(predicted_columns_by_key)):
        predicted_column = predicted_columns_by_key.get((language, variant_idx))
        probability_columns = probability_columns_by_key.get((language, variant_idx), {})
        for row_idx in df.index:
            predicted_party_label = predicted_party_or_none(df, row_idx, predicted_column)
            if probability_columns:
                records.extend(probability_records_for_text(
                    df, row_idx, language, variant_idx, probability_columns,
                    parties_by_slug, predicted_party_label,
                ))
            elif predicted_party_label is not None:
                records.append(predicted_only_record(
                    row_idx, language, variant_idx, predicted_party_label,
                ))
    return pd.DataFrame.from_records(records, columns=LONG_PREDICTION_COLUMNS)


def mean_probability_per_party(long_df):
    if long_df.empty:
        return pd.Series(dtype=float)
    return long_df.groupby("party")["probability"].mean()


def mean_probability_per_language_and_party(long_df):
    if long_df.empty:
        return pd.DataFrame()
    return (
        long_df.groupby(["language", "party"])["probability"]
        .mean()
        .unstack("party")
    )


def predicted_party_share(long_df):
    if long_df.empty:
        return pd.Series(dtype=float)
    one_prediction_per_text = long_df.drop_duplicates(subset=["row_idx", "language", "variant_idx"])
    counts = one_prediction_per_text["predicted_party"].dropna().value_counts(normalize=True)
    return counts


def plot_party_distribution_bar(mean_probability, predicted_share, output_path):
    parties_present = set(mean_probability.index) | set(predicted_share.index)
    parties = ordered_parties_present(parties_present)
    if not parties:
        return

    bar_positions = np.arange(len(parties))
    bar_width = 0.4
    probabilities = mean_probability.reindex(parties).fillna(0.0).to_numpy()
    shares = predicted_share.reindex(parties).fillna(0.0).to_numpy()
    bar_colors = [color_for_party(party) for party in parties]

    fig, ax = plt.subplots(figsize=(max(9, len(parties) * 1.3), 6.5))
    ax.bar(bar_positions - bar_width / 2, probabilities, width=bar_width,
           color=bar_colors, edgecolor="black", linewidth=0.5, label=MEAN_PROBABILITY_LABEL)
    ax.bar(bar_positions + bar_width / 2, shares, width=bar_width,
           color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.55,
           hatch="//", label="Argmax share")

    for position, probability in zip(bar_positions - bar_width / 2, probabilities):
        ax.text(position, probability + 0.005, f"{probability:.2f}", ha="center",
                va="bottom", fontsize=VALUE_FONTSIZE, rotation=90)
    for position, share in zip(bar_positions + bar_width / 2, shares):
        ax.text(position, share + 0.005, f"{share:.2f}", ha="center",
                va="bottom", fontsize=VALUE_FONTSIZE, rotation=90)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    # Shortened from "Mean class probability / predicted-party share": at this type
    # size the old wording is taller than the axis it labels.
    ax.set_ylabel("Probability / argmax share", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    save_figure(fig, output_path)


def plot_party_distribution_across_variants(mean_probability_by_variant, output_path):
    parties_present = set().union(*(series.index for series in mean_probability_by_variant.values()))
    parties = ordered_parties_present(parties_present)
    variants_present = [v for v in VARIANT_LABEL_ORDER if v in mean_probability_by_variant]
    if not parties or not variants_present:
        return

    bar_positions = np.arange(len(parties))
    group_width = 0.8
    bar_width = group_width / len(variants_present)
    variant_cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(9, len(parties) * 1.4), 6.5))
    for variant_index, variant_label in enumerate(variants_present):
        probabilities = mean_probability_by_variant[variant_label].reindex(parties).fillna(0.0).to_numpy()
        offsets = bar_positions - group_width / 2 + bar_width * (variant_index + 0.5)
        ax.bar(offsets, probabilities, width=bar_width,
               color=variant_cmap(variant_index), edgecolor="black", linewidth=0.4,
               label=variant_label)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(MEAN_PROBABILITY_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=LEGEND_FONTSIZE, framealpha=0.9,
              title="variant", title_fontsize=LEGEND_TITLE_FONTSIZE)
    save_figure(fig, output_path)


def plot_language_party_heatmap(language_party_probability, output_path):
    if language_party_probability.empty:
        return
    parties = ordered_parties_present(set(language_party_probability.columns))
    languages = sorted(language_party_probability.index)
    matrix = language_party_probability.reindex(index=languages, columns=parties).to_numpy(dtype=float)

    # A cell has to hold "0.00" at the new type size, so the grid is given a fixed
    # inch budget per cell rather than the old, tighter one.
    fig, ax = plt.subplots(figsize=(max(9, len(parties) * 1.5),
                                    max(7, len(languages) * 0.62)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.3, np.nanmax(matrix)))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(MEAN_PROBABILITY_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(range(len(parties)), parties, rotation=30, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(len(languages)), languages, fontsize=TICK_FONTSIZE)
    for row_index in range(len(languages)):
        for column_index in range(len(parties)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=HEATMAP_VALUE_FONTSIZE,
                        color="white" if value > 0.45 else "black")
    save_figure(fig, output_path)


def plot_all_languages_party_distribution(mean_probability_per_language, output_path):
    languages = sorted(mean_probability_per_language)
    parties_present = set().union(*(series.index for series in mean_probability_per_language.values()))
    parties = ordered_parties_present(parties_present)
    if not parties or not languages:
        return

    bar_positions = np.arange(len(parties))
    group_width = 0.8
    bar_width = group_width / len(languages)
    language_cmap = plt.get_cmap("nipy_spectral", len(languages))

    fig, ax = plt.subplots(figsize=(max(13, len(parties) * 2.1), 7.5))
    for language_index, language in enumerate(languages):
        probabilities = mean_probability_per_language[language].reindex(parties).fillna(0.0).to_numpy()
        offsets = bar_positions - group_width / 2 + bar_width * (language_index + 0.5)
        ax.bar(offsets, probabilities, width=bar_width,
               color=language_cmap(language_index), edgecolor="black", linewidth=0.3,
               label=language)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(MEAN_PROBABILITY_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Fewer columns than before: the entries are two-letter codes but each now
    # carries a legend handle sized to match 17pt text, so ten across no longer fit.
    ax.legend(bbox_to_anchor=(0.5, -0.26), ncol=min(len(languages), 8),
              title="language", **LEGEND_BELOW)
    save_figure(fig, output_path, bbox_inches="tight")


def plot_speeches_vs_reasons(mean_probability_by_source, output_path):
    sources_present = [s for s in SOURCE_LABELS if s in mean_probability_by_source]
    if len(sources_present) < 2:
        return
    parties_present = set().union(*(series.index for series in mean_probability_by_source.values()))
    parties = ordered_parties_present(parties_present)
    if not parties:
        return

    bar_positions = np.arange(len(parties))
    bar_width = 0.4
    bar_colors = [color_for_party(party) for party in parties]

    fig, ax = plt.subplots(figsize=(max(9, len(parties) * 1.3), 6.5))
    for source_index, source in enumerate(sources_present):
        probabilities = mean_probability_by_source[source].reindex(parties).fillna(0.0).to_numpy()
        offsets = bar_positions - bar_width / 2 + bar_width * source_index
        ax.bar(offsets, probabilities, width=bar_width,
               color=bar_colors, edgecolor="black", linewidth=0.4,
               alpha=1.0 if source == "speeches" else 0.6,
               hatch="" if source == "speeches" else "//",
               label=source)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(MEAN_PROBABILITY_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=LEGEND_FONTSIZE, framealpha=0.9,
              title="source", title_fontsize=LEGEND_TITLE_FONTSIZE)
    save_figure(fig, output_path)


def plot_models_party_distribution(mean_probability_per_model, output_path):
    if not mean_probability_per_model:
        return
    parties_present = set().union(*(series.index for series in mean_probability_per_model.values()))
    parties = ordered_parties_present(parties_present)
    models = sorted(mean_probability_per_model)

    matrix = np.array([
        mean_probability_per_model[model].reindex(parties).fillna(0.0).to_numpy()
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(parties) * 1.6), max(7, len(models) * 0.7)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.4, matrix.max()))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(MEAN_PROBABILITY_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(range(len(parties)), parties, rotation=30, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(len(models)), models, fontsize=TICK_FONTSIZE)
    for row_index in range(len(models)):
        for column_index in range(len(parties)):
            value = matrix[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}",
                    ha="center", va="center", fontsize=HEATMAP_VALUE_FONTSIZE,
                    color="white" if value > 0.45 else "black")
    save_figure(fig, output_path)


def scoped_mean_per_model(values_by_model_source_variant, source=None, variant_label=None):
    """Per-party mean per model over the (source, variant) cells matching the
    scope; None means "every value of that key"."""
    series_per_model = {}
    for (model, csv_source, csv_variant), series in values_by_model_source_variant.items():
        if source is not None and csv_source != source:
            continue
        if variant_label is not None and csv_variant != variant_label:
            continue
        if not series.empty:
            series_per_model.setdefault(model, []).append(series)
    return {
        model: pd.concat(series_list).groupby(level=0).mean()
        for model, series_list in series_per_model.items()
    }


def cross_model_scopes(values_by_model_source_variant):
    """(source, variant, filename suffix, scope label) for the pooled cross-model
    plot and each of its slices: per framing, per source, and every source x
    framing cell. Slices that would just duplicate the pooled plot are dropped."""
    keys = set(values_by_model_source_variant)
    # Every source the values carry, not just SOURCE_LABELS: the VAA agreement replicates
    # keep the direct Likert under "reasons" even when the reasons texts are not
    # classified, and its slices must exist if the pooled plot averages it in. Classified
    # values only ever carry the classified tracks, so they are unaffected.
    sources = [s for s in ["speeches", "reasons"] if any(key[1] == s for key in keys)]
    variants = [v for v in VARIANT_LABEL_ORDER if any(key[2] == v for key in keys)]

    scopes = [(None, None, "", "pooled over sources & variants")]
    if len(variants) > 1:
        scopes += [(None, v, f"_{v}", f"{v}, both sources") for v in variants]
    if len(sources) > 1:
        scopes += [(s, None, f"_{s}", f"{SOURCE_DISPLAY[s]}, all variants") for s in sources]
    if len(sources) * len(variants) > 1:
        scopes += [(s, v, f"_{s}_{v}", f"{SOURCE_DISPLAY[s]}, {v}")
                   for s in sources for v in variants]
    return scopes


PERCENT_HEADROOM = 3.0


def bar_extents(values_by_model_source_variant, replicates_by_model_source_variant=None):
    """(lowest, highest) percent over the bars of every scope plot_cross_model_party_bars
    draws and, when replicates are given, over the ends of their interval whiskers; None
    when there is nothing to draw."""
    lowest, highest = np.inf, -np.inf
    for source, variant_label, _, _ in cross_model_scopes(values_by_model_source_variant):
        drawn = [series.to_numpy(dtype=float) for series in scoped_mean_per_model(
            values_by_model_source_variant, source, variant_label).values()]
        if replicates_by_model_source_variant:
            drawn += [interval.to_numpy(dtype=float).ravel()
                      for interval in replicate_intervals(scoped_replicates_per_model(
                          replicates_by_model_source_variant, source, variant_label)).values()]
        for values in drawn:
            finite = 100.0 * values[np.isfinite(values)]
            if finite.size:
                lowest, highest = min(lowest, float(finite.min())), max(highest, float(finite.max()))
    return (lowest, highest) if np.isfinite(lowest) else None


def percent_bar_limits(values_by_model_source_variant, replicates_by_model_source_variant=None):
    """(0, tallest bar or whisker + headroom) over every scope plot_cross_model_party_bars
    draws, so the figures of one run share a y-axis and can be compared as they are
    browsed."""
    extents = bar_extents(values_by_model_source_variant, replicates_by_model_source_variant)
    if extents is None or extents[1] <= 0:
        return None
    return 0.0, min(100.0, extents[1] + PERCENT_HEADROOM)


def whisker_extent(values):
    """The whisker ends matplotlib would draw for `values` (the same 1.5 x IQR rule),
    so a shared y-range can be worked out before the first box is drawn."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    stats = boxplot_stats(finite)[0]
    return float(stats["whislo"]), float(stats["whishi"])


def percent_box_limits(replicates_by_model_source_variant, clamp=True):
    """One padded y-range covering the whiskers of every scope a box figure draws
    (plot_vaa_merged_blocs; the cross-model VAA figures here are bars now). Not zero-based: the models sit within a few points of each other, so a bar from
    zero would spend the whole axis on agreement no model is anywhere near. `clamp=False`
    for differences from the null, which may go below zero."""
    lowest, highest = np.inf, -np.inf
    for source, variant_label, _, _ in cross_model_scopes(replicates_by_model_source_variant):
        scoped = scoped_replicates_per_model(
            replicates_by_model_source_variant, source, variant_label)
        for replicates in scoped.values():
            for row in 100.0 * replicates.to_numpy(dtype=float):
                if (extent := whisker_extent(row)) is not None:
                    lowest, highest = min(lowest, extent[0]), max(highest, extent[1])
    if not np.isfinite(lowest):
        return None
    return padded_range(lowest, highest, clamp)


def replicate_means(replicates_by_key):
    """{key: mean score per group} from {key: bootstrap replicates per group} -- the bar
    the VAA panels draw. The mean over the statement-bootstrap draws is the observed mean
    up to resampling noise (a few hundredths of a point at 10,000 draws)."""
    return {key: replicates.mean(axis=1) for key, replicates in replicates_by_key.items()}


# The whiskers on the cross-model bars: the central 95% of the statement bootstrap.
INTERVAL_PERCENTILES = (2.5, 97.5)


def replicate_intervals(replicates_by_key):
    """{key: low and high per group} -- the percentile interval of each key's bootstrap
    replicates, what a bar's whisker spans. A group with no replicates gets NaN."""
    intervals = {}
    for key, replicates in replicates_by_key.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            low, high = np.nanpercentile(replicates.to_numpy(dtype=float),
                                         INTERVAL_PERCENTILES, axis=1)
        intervals[key] = pd.DataFrame({"low": low, "high": high}, index=replicates.index)
    return intervals


def padded_range(lowest, highest, clamp=True):
    """Room around the drawn values, clamped to the 0-100% a share can actually take --
    unless `clamp` is off, for a difference from the null, which has no such bounds."""
    margin = max(0.06 * (highest - lowest), 0.5)
    if not clamp:
        return lowest - margin, highest + margin
    return max(0.0, lowest - margin), min(100.0, highest + margin)


def signed_bar_limits(values_by_model_source_variant, replicates_by_model_source_variant=None):
    """percent_bar_limits for differences from the null: zero kept inside the axis and
    headroom on whichever side has bars (or whiskers)."""
    lowest, highest = bar_extents(
        values_by_model_source_variant, replicates_by_model_source_variant) or (0.0, 0.0)
    return min(lowest, 0.0) - PERCENT_HEADROOM, max(highest, 0.0) + PERCENT_HEADROOM


def tight_limits(drawn, signed):
    """A y-range fitted to what one figure draws: `drawn` is a list of {model: Series or
    DataFrame of fractions} (bar heights, whisker ends). Only a sliver of padding, so a
    figure printed a column wide spends its height on the bars -- the run-wide limits
    above trade that for a scale shared between figures. Zero stays in view: it is the
    baseline of a share and the reference line of a difference from the null."""
    values = np.concatenate([100.0 * np.asarray(frame, dtype=float).ravel()
                             for frames in drawn for frame in frames.values()] or [np.zeros(1)])
    finite = values[np.isfinite(values)]
    lowest = min(0.0, float(finite.min())) if finite.size else 0.0
    highest = max(0.0, float(finite.max())) if finite.size else 0.0
    margin = max(0.04 * (highest - lowest), 0.3)
    if signed:
        return lowest - margin, highest + margin
    return 0.0, min(100.0, highest + margin)


def widest_limits(limits):
    """The union of several (low, high) ranges; None entries are ignored."""
    present = [limit for limit in limits if limit is not None]
    if not present:
        return None
    return min(low for low, _ in present), max(high for _, high in present)


MODEL_LEGEND_NCOL = 4
GROUP_WIDTH = 0.86

# The result directories are named for the API model id (see answer_generation/
# generate_all.py); a figure is read as prose, so it names the model the way its maker
# writes it -- except gpt-oss, whose lowercase house styling reads as a typo beside a
# dozen capitalised names. Anything not listed falls through to the directory name, so
# a new model dir still plots.
def draw_models_party_bars(ax, value_per_model, parties, models, model_cmap,
                           value_label, ylim, label_x=True, scale=1.0, unit="%",
                           intervals_per_model=None, tick_rotation=20,
                           group_width=GROUP_WIDTH):
    """One grouped-bar panel: a bar per (party, model), colour-indexed by the
    model's position in `models`. `intervals_per_model` (replicate_intervals: low and
    high per party) adds a whisker to every bar it covers. `tick_rotation=0` sets the
    party labels level and centred, for a figure with few enough parties to fit them.
    `group_width` is the share of a party's slot its bars fill.

    `models` is passed in rather than derived from `value_per_model` so a panel
    that is missing a model still colours the ones it has the same as its
    neighbouring panel -- the whole point of a figure that shares one legend."""
    bar_positions = np.arange(len(parties))
    bar_width = group_width / len(models)
    highest_percentage = 0.0
    for model_index, model in enumerate(models):
        if model not in value_per_model:
            continue
        percentages = 100.0 * value_per_model[model].reindex(parties).fillna(0.0).to_numpy()
        highest_percentage = max(highest_percentage, float(percentages.max()))
        offsets = bar_positions - group_width / 2 + bar_width * (model_index + 0.5)
        ax.bar(offsets, percentages, width=bar_width,
               color=model_cmap(model_index), edgecolor="black", linewidth=0.3,
               label=model)
        if intervals_per_model and model in intervals_per_model:
            interval = 100.0 * intervals_per_model[model].reindex(parties).to_numpy(dtype=float)
            # Measured from the bar; a bar a hair outside its own interval (the observed
            # mean against bootstrap percentiles) gets a zero-length side, not an error.
            below = np.clip(percentages - interval[:, 0], 0.0, None)
            above = np.clip(interval[:, 1] - percentages, 0.0, None)
            shown = np.isfinite(below) & np.isfinite(above)
            if shown.any():
                ax.errorbar(offsets[shown], percentages[shown],
                            yerr=[below[shown], above[shown]], fmt="none", ecolor="black",
                            elinewidth=0.9 * scale, capsize=1.8 * scale,
                            capthick=0.9 * scale, zorder=4)
                highest_percentage = max(highest_percentage, float(interval[shown, 1].max()))

    ax.set_xticks(bar_positions)
    # The upper panel of a stacked figure hands its x-labels to the lower one;
    # ticks stay so the party boundaries still read across both panels.
    ax.set_xticklabels(parties if label_x else [""] * len(parties),
                       rotation=tick_rotation, ha="right" if tick_rotation else "center",
                       fontsize=TICK_FONTSIZE * scale)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE * scale)
    ax.set_xlim(-0.5, len(parties) - 0.5)
    ax.set_ylabel(f"{value_label} ({unit})", fontsize=AXIS_LABEL_FONTSIZE * scale)
    ax.set_ylim(*(ylim or (0.0, min(100.0, highest_percentage + PERCENT_HEADROOM))))
    mark_zero_if_signed(ax, unit)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def mark_zero_if_signed(ax, unit):
    """A difference from the null ("pp") is read against zero, so the line is drawn."""
    if unit == "pp":
        ax.axhline(0.0, color="black", linewidth=1.0, zorder=3)


def plot_models_party_bars(value_per_model, value_label, output_path, ylim=None,
                           intervals_per_model=None):
    if not value_per_model:
        return
    parties_present = set().union(*(series.index for series in value_per_model.values()))
    parties = ordered_parties_present(parties_present)
    models = sorted(value_per_model)
    if not parties or not models:
        return

    # Wider than the old 9 inches: this is the figure the thesis prints (mean
    # probability per model, one panel per source), and at 20pt type the bars need
    # the room the legend used to take over them.
    fig, ax = plt.subplots(figsize=(max(13, len(parties) * 2.0), 7.5))
    draw_models_party_bars(ax, value_per_model, parties, models,
                           plt.get_cmap("tab20", max(len(models), 2)), value_label, ylim,
                           intervals_per_model=intervals_per_model)
    # Moved out from under "upper right": a 16-model legend at 17pt covers the
    # tallest bars, which on this figure are the finding. The offset clears the
    # rotated party labels, which at 20pt reach well below the axes.
    ax.legend(bbox_to_anchor=(0.5, -0.26), ncol=min(len(models), MODEL_LEGEND_NCOL),
              title="model", **LEGEND_BELOW)
    save_figure(fig, output_path, bbox_inches="tight")


def plot_models_party_bars_stacked(value_per_model_per_panel, value_label, output_path,
                                   ylim=None):
    """The per-source figures stacked into one, sharing a single legend.

    Two copies of a 13-entry model legend is the most expensive thing on the page
    when the two panels are placed side by side in LaTeX -- it is the same legend
    twice, and at the type size these figures now use it costs more vertical space
    than either panel's bars. Panels therefore share one x-axis and one legend
    below, and the y-range is shared too so a bar in the upper panel can be
    compared against the lower one by eye rather than by reading both axes.

    `value_per_model_per_panel` is {panel title: {model: per-party series}}, drawn
    top to bottom in the order given."""
    panels = [(title, values) for title, values in value_per_model_per_panel.items() if values]
    if len(panels) < 2:
        return
    parties_present = set().union(
        *(series.index for _, values in panels for series in values.values()))
    parties = ordered_parties_present(parties_present)
    # The union, so a model present in only one panel still gets a legend entry and
    # keeps one colour throughout.
    models = sorted({model for _, values in panels for model in values})
    if not parties or not models:
        return

    model_cmap = plt.get_cmap("tab20", max(len(models), 2))
    ylim = ylim or (0.0, min(100.0, PERCENT_HEADROOM + max(
        100.0 * float(np.nanmax(series.to_numpy(dtype=float)))
        for _, values in panels for series in values.values())))

    figure_width = max(13, len(parties) * 2.0)
    fig, axes = plt.subplots(len(panels), 1, figsize=(figure_width, 5.5 * len(panels)),
                             sharex=True)
    for index, (ax, (title, values)) in enumerate(zip(axes, panels)):
        draw_models_party_bars(ax, values, parties, models, model_cmap, value_label,
                               ylim, label_x=index == len(panels) - 1)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Hung off the bottom panel, so the offset is measured against one panel's
    # height rather than the whole stack's.
    axes[-1].legend(bbox_to_anchor=(0.5, -0.30), ncol=min(len(models), MODEL_LEGEND_NCOL),
                    title="model", **LEGEND_BELOW)
    save_figure(fig, output_path, bbox_inches="tight")


def draw_models_party_boxes(ax, replicates_per_model, parties, models, model_cmap,
                            value_label, ylim, label_x=True, scale=1.0, unit="%"):
    """One box panel: a box per (party, model) over that model's bootstrap
    distribution -- box = IQR, whiskers = the usual 1.5 x IQR, line = the median.

    Like draw_models_party_bars, `models` is passed in so a panel missing a model
    still colours the rest the way its neighbours do."""
    party_positions = np.arange(len(parties))
    slot_width = GROUP_WIDTH / len(models)

    lowest, highest = np.inf, -np.inf
    for model_index, model in enumerate(models):
        if model not in replicates_per_model:
            continue
        values = 100.0 * replicates_per_model[model].reindex(parties).to_numpy(dtype=float)
        offsets = party_positions - GROUP_WIDTH / 2 + slot_width * (model_index + 0.5)
        drawn = [(row[np.isfinite(row)], offset) for row, offset in zip(values, offsets)]
        drawn = [(row, offset) for row, offset in drawn if row.size]
        if not drawn:
            continue
        parts = ax.boxplot(
            [row for row, _ in drawn], positions=[offset for _, offset in drawn],
            widths=slot_width * 0.78, patch_artist=True, showfliers=False,
            medianprops=BOX_MEDIAN_STYLE, manage_ticks=False,
        )
        style_boxes(parts, model_cmap(model_index))
        for whisker in parts["whiskers"]:
            lowest = min(lowest, float(np.min(whisker.get_ydata())))
            highest = max(highest, float(np.max(whisker.get_ydata())))

    ax.set_ylim(*(ylim or padded_range(lowest, highest)))
    ax.set_xticks(party_positions)
    ax.set_xticklabels(parties if label_x else [""] * len(parties),
                       rotation=20, ha="right", fontsize=TICK_FONTSIZE * scale)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE * scale)
    ax.set_xlim(-0.5, len(parties) - 0.5)
    for boundary in party_positions[:-1] + 0.5:
        ax.axvline(boundary, color="black", linewidth=0.5, alpha=0.15)
    ax.set_ylabel(f"{value_label} ({unit})", fontsize=AXIS_LABEL_FONTSIZE * scale)
    mark_zero_if_signed(ax, unit)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def model_legend_handles(models, model_cmap, label_for=str):
    """Patches matching the box faces, so a box panel (which has no bar artists to
    label) can carry the same legend a bar panel draws for itself."""
    return [Patch(facecolor=model_cmap(index), edgecolor="black", linewidth=0.7,
                  alpha=0.85, label=label_for(model)) for index, model in enumerate(models)]


def fitted_legend(ax, handles, fontsize, bbox_to_anchor, max_ncol):
    """The legend at `fontsize`, in as many columns (up to `max_ncol`) as fit the figure's
    own width.

    A legend wider than the figure is not clipped: `bbox_inches="tight"` widens the saved
    image to hold it while the panels keep the width they were laid out with, so an
    overlong legend row silently shrinks the panels to a band down the middle of the
    page. Measuring is the only way to know -- a legend shrink-wraps its text, so its
    width depends on the model names it happens to be drawing. The type size is what is
    held fixed rather than the column count: the figure is sized so that this type prints
    legibly, and a column given up costs only height, which a column figure has to spare.

    `bbox_to_anchor` is (x, y) with x a fraction of the FIGURE's width and y a fraction of
    the axes' height: the legend is as wide as the figure, so it is centred under the
    whole of it, y labels included, rather than under the axes."""
    renderer = ax.figure.canvas.get_renderer()
    figure_width_px = ax.figure.get_figwidth() * ax.figure.dpi
    anchor_transform = blended_transform_factory(ax.figure.transFigure, ax.transAxes)
    for ncol in range(max_ncol, 0, -1):
        legend = ax.legend(handles=handles, ncol=ncol, fontsize=fontsize,
                           bbox_to_anchor=bbox_to_anchor, bbox_transform=anchor_transform,
                           **COMBINED_LEGEND_STYLE)
        if ncol == 1 or legend.get_window_extent(renderer).width <= figure_width_px:
            return legend
        legend.remove()


def shrink_texts_to_fit(texts, available_px, dimension, smallest_fontsize):
    """Steps `texts` down together, half a point at a time, until each one's `dimension`
    ("width" or "height") is within `available_px` or `smallest_fontsize` is reached.
    Together, so the panels of one figure keep one title size."""
    renderer = texts[0].figure.canvas.get_renderer()
    fontsize = texts[0].get_fontsize()
    while fontsize > smallest_fontsize and any(
            getattr(text.get_window_extent(renderer), dimension) > available_px
            for text in texts):
        fontsize = max(smallest_fontsize, fontsize - 0.5)
        for text in texts:
            text.set_fontsize(fontsize)


def plot_models_agreement_and_source_panels(vaa_replicates_per_panel, values_per_model_per_source,
                                            agreement_label, probability_label, output_path,
                                            ylim=None, unit="%",
                                            replicates_per_model_per_source=None):
    """The VAA panel and the per-source probability panels drawn as one stacked figure
    sharing a single model legend.

    The panels were printed side by side carrying the same 13-entry model legend twice
    over, which at this type size costs more of the page than the panels themselves. One
    x-axis of EP groups runs down all of them, and a model keeps one colour throughout, so
    a model's VAA score can be read against its own classified shares by eye.

    Every panel is grouped bars on one shared y-range, the VAA one included (`ylim`; by
    default fitted to this figure's own bars and whiskers, see tight_limits): a
    bar per (group, model) at the model's mean score (replicate_means), drawn exactly
    like the classifier's mean probability below it. `vaa_replicates_per_panel` is
    {panel title: bootstrap replicates per model}: one VAA panel with the tracks pooled,
    or one per track. Each bar carries the 95% interval of its statement bootstrap as a
    whisker: the VAA bars from their replicates, the classifier bars from
    `replicates_per_model_per_source` ({panel title: replicates per model}) when given."""
    panels = [(title, values) for title, values in values_per_model_per_source.items() if values]
    vaa_panels = [(title, replicates) for title, replicates in vaa_replicates_per_panel.items()
                  if replicates]
    if not vaa_panels or not panels:
        return
    replicates_per_model_per_source = replicates_per_model_per_source or {}

    parties = ordered_parties_present(
        set().union(
            *(replicates.index for _, per_model in vaa_panels for replicates in per_model.values()),
            *(series.index for _, values in panels for series in values.values()),
        ))
    # The union across all the panels, so a model present in only one still gets a
    # legend entry and one colour.
    models = sorted(
        set().union(*(per_model for _, per_model in vaa_panels),
                    *(values for _, values in panels)))
    if not parties or not models:
        return

    model_cmap = plt.get_cmap("tab20", max(len(models), 2))
    # (title, bar values, whiskers, y label), top to bottom: the VAA panels, then the
    # classifier's.
    drawn = [(title, replicate_means(per_model), replicate_intervals(per_model), agreement_label)
             for title, per_model in vaa_panels]
    drawn += [(title, values,
               replicate_intervals(replicates_per_model_per_source.get(title, {})),
               probability_label)
              for title, values in panels]
    if ylim is None:
        ylim = tight_limits([frames for _, values, intervals, _ in drawn
                             for frames in (values, intervals)], signed=unit == "pp")

    figure_width = max(COMBINED_FIGURE_WIDTH, len(parties) * COMBINED_WIDTH_PER_PARTY)
    panel_count = len(drawn)
    fig, axes = plt.subplots(panel_count, 1,
                             figsize=(figure_width, COMBINED_PANEL_HEIGHT * panel_count),
                             sharex=True)
    for index, (ax, (title, values, intervals, label)) in enumerate(zip(axes, drawn)):
        draw_models_party_bars(ax, values, parties, models, model_cmap, label,
                               ylim, label_x=index == panel_count - 1, unit=unit,
                               intervals_per_model=intervals,
                               tick_rotation=COMBINED_TICK_ROTATION,
                               group_width=COMBINED_GROUP_WIDTH)
        ax.yaxis.label.set_fontsize(COMBINED_YAXIS_FONTSIZE)
        ax.tick_params(axis="y", labelsize=COMBINED_YAXIS_FONTSIZE)
        ax.set_title(title, fontsize=COMBINED_FONTSIZE)

    # Hung off the bottom panel, so the offset is measured against one panel's height
    # rather than the whole stack's. The handles are built rather than collected: the
    # bar panels each label only the models they drew. No title: the entries are
    # model names and say so themselves.
    legend = fitted_legend(
        axes[-1],
        model_legend_handles(models, model_cmap, model_display_name),
        fontsize=COMBINED_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, COMBINED_LEGEND_OFFSET),
        max_ncol=min(len(models), MODEL_LEGEND_NCOL),
    )
    # Kept out of the layout and handed to the save instead: the legend is as tall as a
    # panel here, and tight_layout would make room for it inside the figure by squeezing
    # the panels to a third of their height. The saved image grows downwards to hold it.
    legend.set_in_layout(False)
    fig.tight_layout()
    panel = axes[0].get_window_extent()
    shrink_texts_to_fit([ax.title for ax in axes], panel.width, "width",
                        COMBINED_SMALLEST_FONTSIZE)
    shrink_texts_to_fit([ax.yaxis.label for ax in axes], panel.height, "height",
                        COMBINED_SMALLEST_FONTSIZE)
    save_figure(fig, output_path, bbox_inches="tight", bbox_extra_artists=[legend])


def plot_models_party_boxes(replicates_per_model, value_label, output_path, ylim=None):
    """One box per (party, model) over the bootstrap distribution of that model's
    mean agreement.

    Bars from zero would spend the whole axis on agreement no model is anywhere near:
    the models sit within a few points of each other, so the y-range is clipped to the
    boxes drawn -- shared across the run when `ylim` is given, this figure's own
    otherwise."""
    if not replicates_per_model:
        return
    parties_present = set().union(
        *(replicates.index for replicates in replicates_per_model.values()))
    parties = ordered_parties_present(parties_present)
    models = sorted(replicates_per_model)
    if not parties or not models:
        return

    model_cmap = plt.get_cmap("tab20", max(len(models), 2))
    figure_width = max(13.0, len(parties) * len(models) * 0.34)
    scale = type_scale(figure_width)
    fig, ax = plt.subplots(figsize=(figure_width, 7.5 * scale))
    draw_models_party_boxes(ax, replicates_per_model, parties, models, model_cmap,
                            value_label, ylim, scale=scale)
    ax.legend(
        handles=model_legend_handles(models, model_cmap),
        **{**LEGEND_BELOW,
           "fontsize": LEGEND_FONTSIZE * scale,
           "title_fontsize": LEGEND_TITLE_FONTSIZE * scale},
        bbox_to_anchor=(0.5, -0.26), ncol=min(len(models), 4), title="model",
    )
    save_figure(fig, output_path, bbox_inches="tight")


def style_boxes(parts, facecolor):
    for box in parts["boxes"]:
        box.set_facecolor(facecolor)
        box.set_edgecolor("black")
        box.set_linewidth(0.7)
        box.set_alpha(0.85)
    for key in ("whiskers", "caps"):
        for artist in parts[key]:
            artist.set_color("#444444")
            artist.set_linewidth(0.9)


def plot_cross_model_vaa_bars(replicates_by_model_source_variant, value_label,
                              dataset_plots_dir, filename_stem, ylim=None):
    """The VAA scores as plot_cross_model_party_bars draws the classifier's: one
    grouped-bar figure per scope, a bar per (group, model) at the model's mean score
    (replicate_means) with the 95% interval of its statement bootstrap as a whisker. The
    scopes share a zero-based y-axis; pass `ylim` to widen that to a whole run of calls."""
    plot_cross_model_party_bars(replicate_means(replicates_by_model_source_variant),
                                value_label, dataset_plots_dir, filename_stem, ylim,
                                replicates_by_model_source_variant)


def plot_cross_model_party_bars(values_by_model_source_variant, value_label,
                                dataset_plots_dir, filename_stem, ylim=None,
                                replicates_by_model_source_variant=None):
    """One grouped-bar figure per scope: pooled, per framing, per source, per cell. The
    scopes share a y-axis; pass `ylim` to widen that to a whole run of calls. With
    bootstrap replicates of the same cells, every bar gets the 95% interval of its scope
    as a whisker (the cells of a scope are averaged draw by draw)."""
    ylim = ylim or percent_bar_limits(values_by_model_source_variant,
                                      replicates_by_model_source_variant)
    for source, variant_label, suffix, scope_label in cross_model_scopes(values_by_model_source_variant):
        value_per_model = scoped_mean_per_model(values_by_model_source_variant, source, variant_label)
        if len(value_per_model) < 2:
            print(f"  only {len(value_per_model)} model(s) for '{scope_label}', skipping.")
            continue
        intervals = replicate_intervals(scoped_replicates_per_model(
            replicates_by_model_source_variant or {}, source, variant_label))
        plot_models_party_bars(
            value_per_model, value_label, dataset_plots_dir / f"{filename_stem}{suffix}.png",
            ylim, intervals)


def plot_cross_model_source_panels(values_by_model_source_variant, value_label,
                                   dataset_plots_dir, filename_stem, ylim=None):
    """The per-source figures again, but stacked into one shared-legend figure per
    framing scope -- the layout the thesis prints, where the two panels otherwise
    carry the same 13-entry legend twice.

    One figure per framing (pooled, base, negated) rather than per full scope: the
    panels ARE the sources, so a source-restricted scope has nothing to stack."""
    ylim = ylim or percent_bar_limits(values_by_model_source_variant)
    keys = set(values_by_model_source_variant)
    # Reasons on top, open-ended below -- SOURCE_LABELS' own order is the reverse,
    # and this figure is laid out to match the printed one.
    sources = [source for source in SOURCE_PANEL_ORDER if any(key[1] == source for key in keys)]
    if len(sources) < 2:
        return
    variants = [variant for variant in VARIANT_LABEL_ORDER if any(key[2] == variant for key in keys)]
    scopes = [(None, "")] + ([(variant, f"_{variant}") for variant in variants]
                             if len(variants) > 1 else [])
    for variant_label, suffix in scopes:
        panels = {
            f"Source: {SOURCE_DISPLAY[source]}": scoped_mean_per_model(
                values_by_model_source_variant, source, variant_label)
            for source in sources
        }
        if any(len(values) < 2 for values in panels.values()):
            print(f"  fewer than two models in a source panel for "
                  f"'{variant_label or 'pooled'}', skipping the stacked figure.")
            continue
        plot_models_party_bars_stacked(
            panels, value_label, dataset_plots_dir / f"{filename_stem}{suffix}.png", ylim)


def plot_cross_model_agreement_and_source_panels(
        values_by_model_source_variant, replicates_by_model_source_variant,
        agreement_label, probability_label, dataset_plots_dir, filename_stem,
        ylim=None, unit="%", classifier_replicates_by_model_source_variant=None):
    """plot_cross_model_source_panels' figure with the VAA bars stacked on top of it:
    one figure per framing scope, since the lower panels ARE the sources.

    The VAA panel pools over sources -- it is drawn from the raw answers of both
    tracks, the way plot_cross_model_vaa_bars' pooled figure is -- so it is the
    framing, not the source, that the scopes vary. The panels of one figure share one
    y-range, so a VAA bar and a classifier bar of the same height mean the same number of
    points; it is fitted to that figure's own bars and whiskers (tight_limits) unless
    `ylim` fixes it, so the pooled, base and negated figures each use their full height."""
    classifier_replicates = classifier_replicates_by_model_source_variant or {}
    keys = set(values_by_model_source_variant)
    sources = [source for source in SOURCE_PANEL_ORDER if any(key[1] == source for key in keys)]
    # one classified source is enough here: the VAA bars are a panel of their own
    if not sources:
        return
    variants = [variant for variant in VARIANT_LABEL_ORDER if any(key[2] == variant for key in keys)]
    scopes = [(None, "")] + ([(variant, f"_{variant}") for variant in variants]
                             if len(variants) > 1 else [])
    minus = " minus null" if unit == "pp" else ""
    # Named for what the bars are, like the VAA panel above them; the source is only
    # spelled out when there is more than one classified track to tell apart.
    panel_title = {
        source: (f"{CLASSIFIER_PANEL_TITLE}{minus}" if len(sources) == 1 else
                 f"{SOURCE_DISPLAY[source]}: {CLASSIFIER_PANEL_TITLE}{minus}")
        for source in sources
    }
    for variant_label, suffix in scopes:
        panels = {
            panel_title[source]: scoped_mean_per_model(
                values_by_model_source_variant, source, variant_label)
            for source in sources
        }
        panel_replicates = {
            panel_title[source]: scoped_replicates_per_model(
                classifier_replicates, source, variant_label)
            for source in sources
        }
        replicates_per_model = scoped_replicates_per_model(
            replicates_by_model_source_variant, None, variant_label)
        if any(len(values) < 2 for values in panels.values()) or len(replicates_per_model) < 2:
            print(f"  fewer than two models in a panel for "
                  f"'{variant_label or 'pooled'}', skipping the combined figure.")
            continue
        plot_models_agreement_and_source_panels(
            {VAA_PANEL_TITLE + minus: replicates_per_model}, panels, agreement_label,
            probability_label, dataset_plots_dir / f"{filename_stem}{suffix}.png", ylim, unit,
            panel_replicates)
        # The same figure with the VAA panel split by track: the pooled panel averages
        # the direct and the indirect Likert, which can hide that they disagree.
        per_track = {
            f"{VAA_TRACK_DISPLAY[track]}: {VAA_PANEL_TITLE}{minus}": scoped_replicates_per_model(
                replicates_by_model_source_variant, track, variant_label)
            for track in VAA_TRACK_DISPLAY
        }
        per_track = {title: replicates for title, replicates in per_track.items()
                     if len(replicates) >= 2}
        if len(per_track) > 1:
            plot_models_agreement_and_source_panels(
                per_track, panels, agreement_label, probability_label,
                dataset_plots_dir / f"{filename_stem}_split{suffix}.png", ylim, unit,
                panel_replicates)


def discover_vaa_csvs(model_dir):
    vaa_by_source_and_variant = {}
    for path in model_dir.glob("vaa*.csv"):
        if (match := VAA_SPEECHES_PATTERN.match(path.name)):
            classifier_source = "speeches"
        elif (match := VAA_LIKERT_PATTERN.match(path.name)):
            classifier_source = "reasons"
        else:
            continue
        # every VAA file, classified track or not: the direct Likert ("reasons" key) is a
        # method of its own even when the reasons texts are not classified
        variant_label = VARIANT_SUFFIX_TO_LABEL[match.group("variant")]
        vaa_by_source_and_variant[(classifier_source, variant_label)] = path
    return vaa_by_source_and_variant


def languages_in_vaa_filename(path):
    match = VAA_SPEECHES_PATTERN.match(path.name) or VAA_LIKERT_PATTERN.match(path.name)
    return match.group("languages")


def vaa_groups_are_collapsed(vaa_csv_paths):
    """evaluate_euandi.py may be run with --collapse-ecr-id; its CSVs are the only
    record of which way it went, so the bootstrap has to follow them."""
    return any(
        "ECR+ID" in set(pd.read_csv(path, usecols=["ep_group"])["ep_group"])
        for path in vaa_csv_paths
    )


def load_vaa_agreement_replicates(model_dirs, positions, bootstrap_draws, seed):
    """{(model, source, variant): mean agreement per party per bootstrap draw}.

    The vaa*.csv files hold agreement already averaged over statements, so they cannot
    say how much of a gap between two models is statement-sampling noise. Recompute the
    per-party means from the raw answers and resample statements -- the unit that is
    actually sampled -- as vaa_agreement_ci.py does. Every model and scope is scored on
    the same draws, so the replicates stay comparable and can be averaged across the
    (source, variant) cells a plotted scope pools over.
    """
    vaa_csvs_per_model = {model_dir: discover_vaa_csvs(model_dir) for model_dir in model_dirs}
    every_vaa_csv = [path for csvs in vaa_csvs_per_model.values() for path in csvs.values()]
    if not every_vaa_csv:
        return {}

    party_df = load_party_positions(positions_path(positions), positions)
    if vaa_groups_are_collapsed(every_vaa_csv):
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    party_df = party_df.dropna(subset=["ep_group"])
    statement_index = np.sort(party_df["statement_idx"].unique())
    groups = sorted(party_df["ep_group"].unique())

    draws = statement_draws(len(statement_index), bootstrap_draws, seed)

    replicates_per_model_source_variant = {}
    for model_dir, vaa_csvs in vaa_csvs_per_model.items():
        for (source, variant_label), path in vaa_csvs.items():
            stance_df, languages = stance_frame_for(
                str(model_dir.parent), model_dir.name,
                STANCE_SOURCE_FOR_CLASSIFIER_SOURCE[source],
                VARIANT_LABEL_TO_SUFFIX[variant_label],
                languages_in_vaa_filename(path),
            )
            if stance_df is None:
                continue
            matrices = agreement_matrices(
                stance_df, languages, party_df, groups, statement_index,
                scoring=vaa_scoring(positions))
            if not matrices[1].any():
                continue
            _, replicates = bootstrap_group_means(matrices, draws)
            replicates_per_model_source_variant[(model_dir.name, source, variant_label)] = (
                pd.DataFrame(replicates, index=groups))
    return replicates_per_model_source_variant


def statement_draws(statement_count, bootstrap_draws, seed):
    """(draws, statements) resampled statement positions: one cluster bootstrap over the
    questionnaire, the unit that is actually sampled. The same count and seed give the
    same draws, which is how the VAA and the classifier replicates come to share them."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, statement_count, size=(bootstrap_draws, statement_count))


def load_classifier_probability_replicates(predictions_by_model, bootstrap_draws, seed):
    """{(model, source, variant): mean classifier probability per party per bootstrap
    draw} -- load_vaa_agreement_replicates' counterpart for the classified texts.

    The statements are resampled, not the texts: every text of a drawn statement goes
    with it (all its languages and paraphrases), as on the VAA side, so the two sets of
    whiskers answer the same question -- how far the mean would move on another
    questionnaire of this size. A cell's replicates average to mean_probability_per_party
    up to resampling noise."""
    statements = sorted({int(row) for long_by_source_variant in predictions_by_model.values()
                         for long_df in long_by_source_variant.values()
                         for row in long_df["row_idx"].dropna().unique()})
    if not statements:
        return {}
    draws = statement_draws(len(statements), bootstrap_draws, seed)
    replicates_per_model_source_variant = {}
    for model_dir, long_by_source_variant in predictions_by_model.items():
        for (source, variant_label), long_df in long_by_source_variant.items():
            scored = long_df.dropna(subset=["probability"])
            if scored.empty:
                continue
            cells = scored.groupby(["party", "row_idx"])["probability"].agg(["sum", "count"])
            parties = sorted(scored["party"].unique())
            matrices = tuple(
                cells[column].unstack("row_idx").reindex(index=parties, columns=statements)
                .fillna(0.0).to_numpy(dtype=float)
                for column in ("sum", "count"))
            _, replicates = bootstrap_group_means(matrices, draws)
            replicates_per_model_source_variant[(model_dir.name, source, variant_label)] = (
                pd.DataFrame(replicates, index=parties))
    return replicates_per_model_source_variant


def scoped_replicates_per_model(replicates_by_model_source_variant, source=None, variant_label=None):
    """Per-model replicates per party, averaged over the (source, variant) cells
    matching the scope; None means "every value of that key". The cells share
    bootstrap draws, so averaging them draw by draw gives the pooled figure its own
    sampling distribution rather than a mixture of the cells'."""
    cells_per_model = {}
    for (model, csv_source, csv_variant), cell in replicates_by_model_source_variant.items():
        if source is not None and csv_source != source:
            continue
        if variant_label is not None and csv_variant != variant_label:
            continue
        cells_per_model.setdefault(model, []).append(cell)
    return {
        model: sum(cells) / len(cells)
        for model, cells in cells_per_model.items()
    }


def load_vaa_mean_agreement_per_party(vaa_csv_path):
    df = pd.read_csv(vaa_csv_path)
    if df.empty or "ep_group" not in df.columns or "mean_agreement" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("ep_group")["mean_agreement"].mean()


def spearman_rho_between_party_vectors(series_a, series_b):
    parties = sorted(set(series_a.index) & set(series_b.index))
    if len(parties) < 2:
        return float("nan")
    vector_a = series_a.reindex(parties).to_numpy(dtype=float)
    vector_b = series_b.reindex(parties).to_numpy(dtype=float)
    mask = ~(np.isnan(vector_a) | np.isnan(vector_b))
    if mask.sum() < 2:
        return float("nan")
    vector_a, vector_b = vector_a[mask], vector_b[mask]
    if vector_a.std() == 0 or vector_b.std() == 0:
        return float("nan")
    return float(spearmanr(vector_a, vector_b)[0])


def top_party(series):
    if series.empty or series.isna().all():
        return None, float("nan")
    top_party_label = series.idxmax()
    return top_party_label, float(series.loc[top_party_label])


def build_vaa_comparison_row(model_name, source, variant_label, classifier_mean_prob,
                             classifier_argmax_share, vaa_mean_agreement):
    spearman_mean_prob = spearman_rho_between_party_vectors(classifier_mean_prob, vaa_mean_agreement)
    spearman_argmax = spearman_rho_between_party_vectors(classifier_argmax_share, vaa_mean_agreement)
    classifier_mean_prob_top, classifier_mean_prob_top_value = top_party(classifier_mean_prob)
    classifier_argmax_top, classifier_argmax_top_value = top_party(classifier_argmax_share)
    vaa_top, vaa_top_value = top_party(vaa_mean_agreement)
    return {
        "model": model_name,
        "source": source,
        "variant": variant_label,
        "spearman_rho_mean_prob_vs_vaa": spearman_mean_prob,
        "spearman_rho_argmax_share_vs_vaa": spearman_argmax,
        "classifier_mean_prob_top": classifier_mean_prob_top,
        "classifier_mean_prob_top_value": classifier_mean_prob_top_value,
        "classifier_argmax_top": classifier_argmax_top,
        "classifier_argmax_top_share": classifier_argmax_top_value,
        "vaa_top": vaa_top,
        "vaa_top_mean_agreement": vaa_top_value,
        "mean_prob_top_matches_vaa": classifier_mean_prob_top == vaa_top
            if classifier_mean_prob_top and vaa_top else False,
        "argmax_top_matches_vaa": classifier_argmax_top == vaa_top
            if classifier_argmax_top and vaa_top else False,
    }


def plot_classifier_vs_vaa_scatter(classifier_mean_prob, vaa_mean_agreement,
                                   spearman_rho, output_path):
    parties = sorted(set(classifier_mean_prob.index) & set(vaa_mean_agreement.index))
    if len(parties) < 2:
        return
    classifier_values = classifier_mean_prob.reindex(parties).to_numpy(dtype=float)
    vaa_values = vaa_mean_agreement.reindex(parties).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 8.0))
    for party, vaa_value, classifier_value in zip(parties, vaa_values, classifier_values):
        ax.scatter(vaa_value, classifier_value, s=260, color=color_for_party(party),
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(party, (vaa_value, classifier_value), xytext=(8, 6),
                    textcoords="offset points", fontsize=ANNOTATION_FONTSIZE)

    # Both labels shortened: at 22pt the parenthetical qualifiers ran past the
    # panel. What they averaged over is the caption's job, not the axis's.
    ax.set_xlabel("VAA mean agreement", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Classifier mean probability", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(linestyle="--", alpha=0.4)
    # The correlation is the point of the figure, so it stays on the panel now
    # that the title is gone.
    ax.text(0.02, 0.98, f"Spearman ρ = {spearman_rho:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=ANNOTATION_FONTSIZE)
    save_figure(fig, output_path)


def plot_cross_model_vaa_correlation_heatmap(comparison_rows, value_column, value_label,
                                             output_path):
    if not comparison_rows:
        return
    summary_df = pd.DataFrame(comparison_rows)
    summary_df["source_variant"] = summary_df["source"] + " · " + summary_df["variant"]
    pivoted = summary_df.pivot(index="model", columns="source_variant", values=value_column)
    if pivoted.empty:
        return
    pivoted = pivoted.sort_index().sort_index(axis=1)
    matrix = pivoted.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, pivoted.shape[1] * 1.7),
                                    max(7, pivoted.shape[0] * 0.7)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(value_label, fontsize=AXIS_LABEL_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(range(pivoted.shape[1]), pivoted.columns, rotation=30, ha="right",
                  fontsize=TICK_FONTSIZE)
    ax.set_yticks(range(pivoted.shape[0]), pivoted.index, fontsize=TICK_FONTSIZE)
    for row_index in range(pivoted.shape[0]):
        for column_index in range(pivoted.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=HEATMAP_VALUE_FONTSIZE,
                        color="white" if abs(value) > 0.6 else "black")
    save_figure(fig, output_path)


def plots_dir_for(model_dir, name="plots"):
    """<model_dir>/plots, or a per-null twin such as plots_gaussian."""
    plots_dir = model_dir / name
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def plot_classifier_distributions_for_one_csv(source, variant_label,
                                              classifier_mean_prob, classifier_argmax_share,
                                              long_df, plots_dir):
    plot_party_distribution_bar(
        classifier_mean_prob, classifier_argmax_share,
        plots_dir / f"classified_{source}_{variant_label}_party_distribution.png",
    )
    plot_language_party_heatmap(
        mean_probability_per_language_and_party(long_df),
        plots_dir / f"classified_{source}_{variant_label}_language_party_heatmap.png",
    )


def compare_classifier_to_vaa(model_name, source, variant_label,
                              classifier_mean_prob, classifier_argmax_share,
                              vaa_csv_path, plots_dir):
    if vaa_csv_path is None:
        print(f"  [{source}/{variant_label}] no matching VAA CSV "
              f"({VAA_SOURCE_FOR_CLASSIFIER_SOURCE[source]}), skipping VAA correlation.")
        return None

    vaa_mean_agreement = load_vaa_mean_agreement_per_party(vaa_csv_path)
    if vaa_mean_agreement.empty or vaa_mean_agreement.isna().all():
        print(f"  [{source}/{variant_label}] VAA CSV {vaa_csv_path.name} produced no usable rows.")
        return None

    comparison_row = build_vaa_comparison_row(
        model_name, source, variant_label,
        classifier_mean_prob, classifier_argmax_share, vaa_mean_agreement,
    )
    plot_classifier_vs_vaa_scatter(
        classifier_mean_prob, vaa_mean_agreement,
        comparison_row["spearman_rho_mean_prob_vs_vaa"],
        plots_dir / f"classified_vs_vaa_{source}_{variant_label}_scatter.png",
    )
    return comparison_row


def language_party_matrix_from_classifier(long_by_source_variant):
    if not long_by_source_variant:
        return pd.DataFrame()
    combined = pd.concat(long_by_source_variant.values(), ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    return (
        combined.groupby(["language", "party"])["probability"]
        .mean()
        .unstack("party")
    )


def mean_probability_and_share_per_language(long_by_source_variant):
    frames = [long_df for long_df in long_by_source_variant.values() if not long_df.empty]
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    return {
        language: (mean_probability_per_party(group), predicted_party_share(group))
        for language, group in combined.groupby("language")
    }


def plot_per_language_distributions_for_scope(long_by_source_variant, classified_plots_dir,
                                              file_prefix):
    per_language = mean_probability_and_share_per_language(long_by_source_variant)
    if not per_language:
        return

    for language, (mean_probability, predicted_share) in sorted(per_language.items()):
        plot_party_distribution_bar(
            mean_probability, predicted_share,
            classified_plots_dir / f"classified_{file_prefix}{language}.png",
        )

    plot_all_languages_party_distribution(
        {language: mean_probability for language, (mean_probability, _) in per_language.items()},
        classified_plots_dir / f"classified_{file_prefix}all_languages.png",
    )


def plot_per_language_distributions(long_by_source_variant, plots_dir):
    classified_plots_dir = plots_dir / "classified"
    plot_per_language_distributions_for_scope(
        long_by_source_variant, classified_plots_dir, file_prefix="")
    for source in SOURCE_LABELS:
        long_for_source = {
            (csv_source, variant_label): long_df
            for (csv_source, variant_label), long_df in long_by_source_variant.items()
            if csv_source == source
        }
        plot_per_language_distributions_for_scope(
            long_for_source, classified_plots_dir, file_prefix=f"{source}_")


def language_party_matrix_from_vaa(vaa_csvs):
    frames = []
    for path in vaa_csvs.values():
        df = pd.read_csv(path)
        if not {"language", "ep_group", "mean_agreement"}.issubset(df.columns):
            continue
        normalized = (
            df["language"]
            .str.replace("_negated", "", regex=False)
        )
        frames.append(pd.DataFrame({
            "language": normalized,
            "ep_group": df["ep_group"],
            "mean_agreement": df["mean_agreement"],
        }))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["language", "ep_group"])["mean_agreement"]
        .mean()
        .unstack("ep_group")
    )


def plot_per_language_bar_panel(ax, matrix, parties, ylabel, title):
    languages = sorted(matrix.index.tolist())
    matrix = matrix.reindex(index=languages, columns=parties)
    x_positions = np.arange(len(parties))
    jitter_rng = np.random.default_rng(0)

    mean_values = []
    languages_with_data = set()
    for x_position, party in zip(x_positions, parties):
        column = matrix[party].to_numpy(dtype=float)
        finite_values = column[~np.isnan(column)]
        mean_values.append(float(finite_values.mean()) if finite_values.size else 0.0)
        for language, value in zip(languages, column):
            if not np.isnan(value):
                languages_with_data.add(language)

        # keep the crosslingual spread visible on top of the election-style bar
        if finite_values.size:
            jitter = (jitter_rng.random(finite_values.size) - 0.5) * 0.3
            ax.scatter(np.full(finite_values.size, x_position) + jitter, finite_values,
                       s=12, color="black", alpha=0.35, linewidths=0, zorder=3)

    bar_colors = [color_for_party(party) for party in parties]
    ax.bar(x_positions, mean_values, width=0.7,
           color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.85, zorder=2)
    for x_position, value in zip(x_positions, mean_values):
        ax.text(x_position, value + 0.005, f"{value:.2f}", ha="center", va="bottom",
                fontsize=VALUE_FONTSIZE)

    ax.set_xticks(x_positions, parties, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(bottom=0.0)
    # Two panels side by side, so this is the one title that has to stay short --
    # the per-panel wording is trimmed at the call site for the same reason.
    ax.set_title(f"{title} ({len(languages_with_data)} languages)",
                 fontsize=TITLE_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_pooled_per_language_overview(long_by_source_variant, vaa_csvs, plots_dir):
    classifier_matrix = language_party_matrix_from_classifier(long_by_source_variant)
    vaa_matrix = language_party_matrix_from_vaa(vaa_csvs)
    if classifier_matrix.empty and vaa_matrix.empty:
        return

    parties = ordered_parties_present(set(classifier_matrix.columns) | set(vaa_matrix.columns))
    if not parties or (classifier_matrix.empty and vaa_matrix.empty):
        return

    fig, (ax_vaa, ax_clf) = plt.subplots(
        1, 2, figsize=(max(20, len(parties) * 3.0), 8.0)
    )
    plot_per_language_bar_panel(
        ax_vaa, vaa_matrix, parties,
        ylabel=MEAN_AGREEMENT_LABEL,
        title="VAA agreement",
    )
    plot_per_language_bar_panel(
        ax_clf, classifier_matrix, parties,
        ylabel="Classifier mean probability",
        title="Classifier probability",
    )
    save_figure(fig, plots_dir / "classified_vs_vaa_per_language.png")


def plot_pooled_classifier_vs_vaa(long_by_source_variant, vaa_csvs, plots_dir):
    if not long_by_source_variant or not vaa_csvs:
        return

    classifier_series = [
        mean_probability_per_party(long_df)
        for long_df in long_by_source_variant.values()
        if not long_df.empty
    ]
    vaa_series = [
        series
        for series in (load_vaa_mean_agreement_per_party(p) for p in vaa_csvs.values())
        if not series.empty
    ]
    if not classifier_series or not vaa_series:
        return

    classifier_mean_prob = pd.concat(classifier_series).groupby(level=0).mean()
    vaa_mean_agreement = pd.concat(vaa_series).groupby(level=0).mean()

    spearman_rho = spearman_rho_between_party_vectors(classifier_mean_prob, vaa_mean_agreement)
    plot_classifier_vs_vaa_scatter(
        classifier_mean_prob, vaa_mean_agreement, spearman_rho,
        plots_dir / "classified_vs_vaa_pooled_scatter.png",
    )


def plot_per_source_aggregates(long_by_source_variant, plots_dir):
    mean_probability_per_source = {}
    for source in SOURCE_LABELS:
        mean_probability_by_variant = {
            variant_label: mean_probability_per_party(long_df)
            for (csv_source, variant_label), long_df in long_by_source_variant.items()
            if csv_source == source
        }
        if mean_probability_by_variant:
            plot_party_distribution_across_variants(
                mean_probability_by_variant,
                plots_dir / f"classified_{source}_party_distribution_variants.png",
            )
        long_for_source = [
            long_df for (csv_source, _), long_df in long_by_source_variant.items()
            if csv_source == source
        ]
        if long_for_source:
            combined = pd.concat(long_for_source, ignore_index=True)
            mean_probability_per_source[source] = mean_probability_per_party(combined)

    if len(mean_probability_per_source) >= 2:
        plot_speeches_vs_reasons(
            mean_probability_per_source,
            plots_dir / "classified_speeches_vs_reasons_party_distribution.png",
        )


def write_vaa_summary_csv(model_dir, vaa_comparison_rows):
    if not vaa_comparison_rows:
        return
    summary_path = model_dir / "classified_vs_vaa_summary.csv"
    pd.DataFrame(vaa_comparison_rows).to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path.relative_to(summary_path.parents[2])}")


def load_model_predictions(model_dir):
    """Long predictions per (source, variant), with no plotting: the cross-model
    plots are drawn from these before any per-model figure is generated."""
    print(f"\nLoading: {model_dir.name}")
    classified_csvs = discover_classified_csvs(model_dir)
    if not classified_csvs:
        print("  No *_classified.csv files found, skipping.")
        return {}

    long_by_source_variant = {}
    for (source, variant_label), csv_path in sorted(classified_csvs.items()):
        long_df = load_long_predictions(csv_path)
        if long_df.empty:
            print(f"  [{source}/{variant_label}] no predictions found in {csv_path.name}")
            continue
        long_by_source_variant[(source, variant_label)] = long_df
    return long_by_source_variant


def process_model(model_dir, long_by_source_variant):
    print(f"\nProcessing: {model_dir.name}")
    plots_dir = plots_dir_for(model_dir)
    vaa_csvs = discover_vaa_csvs(model_dir)
    vaa_comparison_rows = []

    for (source, variant_label), long_df in sorted(long_by_source_variant.items()):
        classifier_mean_prob = mean_probability_per_party(long_df)
        classifier_argmax_share = predicted_party_share(long_df)

        plot_classifier_distributions_for_one_csv(
            source, variant_label,
            classifier_mean_prob, classifier_argmax_share, long_df, plots_dir,
        )
        comparison_row = compare_classifier_to_vaa(
            model_dir.name, source, variant_label,
            classifier_mean_prob, classifier_argmax_share,
            vaa_csvs.get((source, variant_label)), plots_dir,
        )
        if comparison_row is not None:
            vaa_comparison_rows.append(comparison_row)

    plot_per_source_aggregates(long_by_source_variant, plots_dir)
    plot_pooled_classifier_vs_vaa(long_by_source_variant, vaa_csvs, plots_dir)
    plot_pooled_per_language_overview(long_by_source_variant, vaa_csvs, plots_dir)
    plot_per_language_distributions(long_by_source_variant, plots_dir)
    write_vaa_summary_csv(model_dir, vaa_comparison_rows)
    return vaa_comparison_rows


def plot_cross_model_comparisons(mean_probability_per_model_source_variant, dataset_plots_dir):
    grouped = {}
    for (model, source, variant_label), series in mean_probability_per_model_source_variant.items():
        grouped.setdefault((source, variant_label), {})[model] = series

    for (source, variant_label), per_model in sorted(grouped.items()):
        if len(per_model) < 2:
            continue
        plot_models_party_distribution(
            per_model, dataset_plots_dir / f"classified_{source}_{variant_label}_models.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Which euandi answers stand for an EP group when the VAA "
                             "confidence intervals are recomputed; must match the basis "
                             "evaluate_euandi.py wrote the vaa*.csv files with.")
    parser.add_argument("--bootstrap", default=10000, type=int,
                        help="Bootstrap draws behind the VAA agreement intervals.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Read this results directory instead of the --dataset one.")
    parser.add_argument("--absolute", action="store_true",
                        help="Draw vaa_agreement_and_classified_sources*.png as raw mean "
                             "agreement and mean probability (into plots/), not minus their "
                             "nulls.")
    parser.add_argument("--null", nargs="+", default=list(SUBTRACTED_NULLS),
                        choices=SUBTRACTED_NULLS,
                        help="The nulls vaa_agreement_and_classified_sources*.png is drawn "
                             "minus, one figure each, into plots_<null>/ (default: "
                             "%(default)s). Every other figure is raw and goes to plots/.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Sigma of the gaussian null (default: the VAA null's).")
    parser.add_argument("--null_dir", type=Path, default=None,
                        help="Default: <results_dir>/tables/null_model")
    return parser.parse_args()


# Which null the combined figure's panels subtract: the classifier method of a classified
# source, and the VAA track a source's agreement replicates were scored from.
CLASSIFIER_METHOD_FOR_SOURCE = {"reasons": "reasons_clf", "speeches": "speeches_clf"}


def minus_null_inputs(mean_probability_per_model_source_variant,
                      vaa_replicates_per_model_source_variant, baseline,
                      classifier_replicates_per_model_source_variant=None):
    """The combined figure's inputs minus one null model: every (model, source, framing)
    cell's mean probability -- and its bootstrap replicates, when given -- minus the
    classifier's mean-probability null of that source and framing, and every agreement
    replicate minus the agreement null of its VAA track. Per cell, before any scope pools
    them, so a pooled panel subtracts the mean of the nulls it pools. The null is a
    constant here, so a whisker is the bootstrap interval of the score alone, shifted."""
    from analysis.null_baseline import minus_null
    probabilities = {
        (model, source, variant): minus_null(series, baseline.mean_probability(
            CLASSIFIER_METHOD_FOR_SOURCE[source], variant))
        for (model, source, variant), series in mean_probability_per_model_source_variant.items()
    }
    replicates = {
        (model, source, variant): cell.sub(
            baseline.agreement(STANCE_SOURCE_FOR_CLASSIFIER_SOURCE[source])
            .reindex(cell.index), axis=0)
        for (model, source, variant), cell in vaa_replicates_per_model_source_variant.items()
    }
    classifier_replicates = {
        (model, source, variant): cell.sub(
            baseline.mean_probability(CLASSIFIER_METHOD_FOR_SOURCE[source], variant)
            .reindex(cell.index).fillna(0.0), axis=0)
        for (model, source, variant), cell
        in (classifier_replicates_per_model_source_variant or {}).items()
    }
    return probabilities, replicates, classifier_replicates


def main():
    global VAA_PANEL_TITLE, VAA_MEAN_LABEL, VAA_DELTA_LABEL, CLASSIFIER_PANEL_TITLE
    args = parse_args()
    scoring = vaa_scoring(args.positions)
    print(f"VAA scoring: {scoring}")
    if scoring == "posterior":
        VAA_PANEL_TITLE = "VAA " + SCORE_NOUN[scoring]
        VAA_MEAN_LABEL = "Mean VAA " + SCORE_NOUN[scoring]
        VAA_DELTA_LABEL = "Δ VAA probability"
        CLASSIFIER_PANEL_TITLE = "Classifier " + SCORE_NOUN[scoring]
    results_dir = args.results_dir or Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")
    sigma = args.sigma or DEFAULT_SIGMA
    baselines = {}
    if not args.absolute:
        # Imported here: null_baseline reaches this module back through
        # classifier_null_model, so a top-level import would be circular.
        from analysis.null_baseline import NullBaseline, output_dir_name
        null_dir = args.null_dir or results_dir / "tables" / "null_model"
        baselines = {output_dir_name("plots", kind, sigma):
                     NullBaseline.load(null_dir, args.positions, sigma, kind)
                     for kind in dict.fromkeys(args.null)}

    model_dirs = sorted(p for p in results_dir.iterdir()
                        if p.is_dir() and not is_output_dir(p.name))
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    predictions_by_model = {}
    for model_dir in model_dirs:
        long_by_source_variant = load_model_predictions(model_dir)
        if long_by_source_variant:
            predictions_by_model[model_dir] = long_by_source_variant

    mean_probability_per_model_source_variant = {
        (model_dir.name, source, variant_label): mean_probability_per_party(long_df)
        for model_dir, long_by_source_variant in predictions_by_model.items()
        for (source, variant_label), long_df in long_by_source_variant.items()
    }
    models_with_data = {model for model, _, _ in mean_probability_per_model_source_variant}
    dataset_plots_dir = results_dir / "plots"
    # The whiskers of the cross-model classifier bars: the same statement bootstrap, on
    # the same draws, as the VAA replicates below.
    classifier_replicates_per_model_source_variant = load_classifier_probability_replicates(
        predictions_by_model, args.bootstrap, args.seed)

    if len(models_with_data) > 1:
        dataset_plots_dir.mkdir(exist_ok=True)
        print("\nCross-model party distributions:")
        plot_cross_model_party_bars(
            mean_probability_per_model_source_variant, MEAN_PROBABILITY_LABEL,
            dataset_plots_dir, "classified_all_models_party_distribution",
            replicates_by_model_source_variant=classifier_replicates_per_model_source_variant,
        )
        plot_cross_model_source_panels(
            mean_probability_per_model_source_variant, MEAN_PROBABILITY_LABEL,
            dataset_plots_dir, "classified_all_models_party_distribution_sources",
        )

    vaa_replicates_per_model_source_variant = load_vaa_agreement_replicates(
        model_dirs, args.positions, args.bootstrap, args.seed)
    if len({model for model, _, _ in vaa_replicates_per_model_source_variant}) > 1:
        dataset_plots_dir.mkdir(exist_ok=True)
        print("\nCross-model VAA agreement:")
        plot_cross_model_vaa_bars(
            vaa_replicates_per_model_source_variant, VAA_MEAN_LABEL,
            dataset_plots_dir, "vaa_all_models_mean_agreement",
        )
        if len(models_with_data) > 1:
            # The agreement boxes and the two per-source probability panels in one
            # three-panel figure: they carry the same model legend, and the thesis
            # prints them together.
            # Minus each null unless --absolute, into plots_<null>/: agreement and
            # probability a group gets beyond what random stances would give it.
            for plots_name, baseline in baselines.items():
                print(f"\n  minus the {baseline.source} -> {plots_name}/")
                null_plots_dir = results_dir / plots_name
                null_plots_dir.mkdir(exist_ok=True)
                probabilities, replicates, classifier_replicates = minus_null_inputs(
                    mean_probability_per_model_source_variant,
                    vaa_replicates_per_model_source_variant, baseline,
                    classifier_replicates_per_model_source_variant)
                plot_cross_model_agreement_and_source_panels(
                    probabilities, replicates,
                    # Short: three stacked panels leave each rotated label one panel of
                    # height, and "Mean VAA agreement − null" runs into its neighbour.
                    VAA_DELTA_LABEL, "Δ probability",
                    null_plots_dir, "vaa_agreement_and_classified_sources", unit="pp",
                    classifier_replicates_by_model_source_variant=classifier_replicates,
                )
            if not baselines:
                plot_cross_model_agreement_and_source_panels(
                    mean_probability_per_model_source_variant,
                    vaa_replicates_per_model_source_variant,
                    VAA_MEAN_LABEL, MEAN_PROBABILITY_LABEL,
                    dataset_plots_dir, "vaa_agreement_and_classified_sources",
                    classifier_replicates_by_model_source_variant=(
                        classifier_replicates_per_model_source_variant),
                )

    vaa_comparison_rows_across_models = []
    for model_dir, long_by_source_variant in predictions_by_model.items():
        vaa_comparison_rows_across_models.extend(process_model(model_dir, long_by_source_variant))

    if len(models_with_data) > 1:
        plot_cross_model_comparisons(mean_probability_per_model_source_variant, dataset_plots_dir)
        plot_cross_model_vaa_correlation_heatmap(
            vaa_comparison_rows_across_models,
            "spearman_rho_mean_prob_vs_vaa", "Spearman ρ (mean prob vs VAA agreement)",
            dataset_plots_dir / "classified_vs_vaa_spearman_models.png",
        )
        if vaa_comparison_rows_across_models:
            summary_path = dataset_plots_dir / "classified_vs_vaa_models_summary.csv"
            pd.DataFrame(vaa_comparison_rows_across_models).to_csv(summary_path, index=False)
            print(f"  Wrote {summary_path.relative_to(summary_path.parents[2])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
