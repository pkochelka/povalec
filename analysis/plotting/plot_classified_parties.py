import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import FALLBACK_PARTY_COLOR, PARTY_COLORS, PARTY_DISPLAY_ORDER, VARIANT_LABELS

from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS, POSITION_CHOICES, load_party_positions, positions_path,
)
from analysis.vaa_agreement_ci import (
    agreement_matrices, bootstrap_group_means, stance_frame_for,
)

VARIANT_SUFFIX_TO_LABEL = VARIANT_LABELS
VARIANT_LABEL_TO_SUFFIX = {label: suffix for suffix, label in VARIANT_SUFFIX_TO_LABEL.items()}
VARIANT_LABEL_ORDER = ["base", "negated", "question"]
SOURCE_LABELS = ["speeches", "reasons"]
SOURCE_DISPLAY = {"speeches": "open-ended", "reasons": "reasons"}
# Top-to-bottom order of the stacked per-source panels, which is the reverse of
# SOURCE_LABELS: the printed figure leads with the reasons track.
SOURCE_PANEL_ORDER = ["reasons", "speeches"]
VAA_SOURCE_FOR_CLASSIFIER_SOURCE = {"speeches": "vaa_speeches", "reasons": "vaa_likert"}
# vaa_agreement_ci.py names the raw answer files by how they were elicited, not by
# what the classifier later reads out of them.
STANCE_SOURCE_FOR_CLASSIFIER_SOURCE = {"speeches": "speeches", "reasons": "likert"}

SPEECHES_CLASSIFIED_PATTERN = re.compile(
    r"^speeches_[a-z]{2}(?:,[a-z]{2})*(?P<variant>|_negated)_classified\.csv$"
)
REASONS_CLASSIFIED_PATTERN = re.compile(
    r"^(?!speeches_)[a-z]{2}(?:,[a-z]{2})*(?P<variant>|_negated)_classified\.csv$"
)
VAA_LIKERT_PATTERN = re.compile(
    r"^vaa(?P<variant>|_negated)_(?P<languages>[a-z]{2}(?:,[a-z]{2})*)\.csv$"
)
VAA_SPEECHES_PATTERN = re.compile(
    r"^vaa_speeches(?P<variant>|_negated)_(?P<languages>[a-z]{2}(?:,[a-z]{2})*)\.csv$"
)

PREDICTED_PARTY_COLUMN_PATTERN = re.compile(
    r"^predicted_party_(?P<language>[a-z]{2})(?P<variant>|_negated)_v(?P<variant_idx>\d+)$"
)
PARTY_PROBABILITY_COLUMN_PATTERN = re.compile(
    r"^party_prob_(?P<party_slug>.+?)_(?P<language>[a-z]{2})(?P<variant>|_negated)_v(?P<variant_idx>\d+)$"
)


MEAN_PROBABILITY_LABEL = "Mean probability"
MEAN_AGREEMENT_LABEL = "Mean VAA agreement"
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

# The three-panel figure is three panels tall but no wider than a one-panel one, so it
# is reduced harder than any other figure here to fit a column -- and its legend, the
# one thing that is not drawn at panel scale, came out well under the body text around
# it. plot_argmax_shares.py sets its legends at roughly its own tick size for the same
# reason; this brings the shared legend past it (25pt against 20pt ticks). The offset
# goes with it: the legend hangs from the bottom panel's axes, so a fraction of ONE
# panel's height, and the old -0.30 left a gap the size of the tick labels twice over.
COMBINED_LEGEND_SCALE = 1.35
COMBINED_LEGEND_OFFSET = -0.19

# Marks and gaps trimmed well in from matplotlib's defaults, which spend some 4.8 em per
# entry on the handle and the column gap alone. The entries are colour swatches next to a
# model name, so the mark carries no shape worth 2 em; what the trim buys is the fourth
# column, which at this type size would not otherwise fit the figure's width.
COMBINED_LEGEND_STYLE = dict(loc="upper center", framealpha=0.9, handlelength=1.4,
                             handletextpad=0.5, columnspacing=1.2, borderpad=0.5)

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


def save_figure(fig, output_path, tight_layout=True, **savefig_kwargs):
    """`tight_layout=False` for a figure laid out some other way -- calling it on a
    constrained-layout figure replaces the engine and undoes what it did."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if FIGURE_FOOTNOTE:
        fig.text(0.99, 0.005, FIGURE_FOOTNOTE, ha="right", va="bottom",
                 fontsize=FOOTNOTE_FONTSIZE, color="#666666")
    if tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=300, **savefig_kwargs)
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
    sources = [s for s in SOURCE_LABELS if any(key[1] == s for key in keys)]
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


def percent_bar_limits(values_by_model_source_variant):
    """(0, tallest bar + headroom) over every scope plot_cross_model_party_bars draws,
    so the figures of one run share a y-axis and can be compared as they are browsed."""
    tallest = 0.0
    for source, variant_label, _, _ in cross_model_scopes(values_by_model_source_variant):
        for series in scoped_mean_per_model(
                values_by_model_source_variant, source, variant_label).values():
            values = 100.0 * series.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                tallest = max(tallest, float(finite.max()))
    return (0.0, min(100.0, tallest + PERCENT_HEADROOM)) if tallest else None


def whisker_extent(values):
    """The whisker ends matplotlib would draw for `values` (the same 1.5 x IQR rule),
    so a shared y-range can be worked out before the first box is drawn."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    stats = boxplot_stats(finite)[0]
    return float(stats["whislo"]), float(stats["whishi"])


def percent_box_limits(replicates_by_model_source_variant):
    """One padded y-range covering the whiskers of every scope plot_cross_model_vaa_boxes
    draws. Not zero-based: the models sit within a few points of each other, so a bar from
    zero would spend the whole axis on agreement no model is anywhere near."""
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
    return padded_range(lowest, highest)


def padded_range(lowest, highest):
    """Room around the drawn values, clamped to the 0-100% a share can actually take."""
    margin = max(0.06 * (highest - lowest), 0.5)
    return max(0.0, lowest - margin), min(100.0, highest + margin)


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
MODEL_DISPLAY_NAME = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini3.5-flash": "Gemini 3.5 Flash",
    "gemma-4-12b": "Gemma 4 12B",
    "gemma-4-31b": "Gemma 4 31B",
    "glm-5.2": "GLM-5.2",
    "gpt-5.6-luna": "GPT 5.6-Luna",
    "gpt-oss-120b": "GPT OSS 120B",
    "granite-4.1-8b": "Granite 4.1 8B",
    "grok-4.5": "Grok 4.5",
    "kimi-k2.7": "Kimi K2.7 Code",
    "kimi-k3": "Kimi K3",
    "mistral-medium-3.5": "Mistral Medium 3.5",
    "qwen3.5-122b": "Qwen3.5 122B",
}


def model_display_name(name):
    """Official name for a result directory, for figures. Takes a name or a Path."""
    return MODEL_DISPLAY_NAME.get(getattr(name, "name", name), str(name))


def draw_models_party_bars(ax, value_per_model, parties, models, model_cmap,
                           value_label, ylim, label_x=True, scale=1.0):
    """One grouped-bar panel: a bar per (party, model), colour-indexed by the
    model's position in `models`.

    `models` is passed in rather than derived from `value_per_model` so a panel
    that is missing a model still colours the ones it has the same as its
    neighbouring panel -- the whole point of a figure that shares one legend."""
    bar_positions = np.arange(len(parties))
    bar_width = GROUP_WIDTH / len(models)
    highest_percentage = 0.0
    for model_index, model in enumerate(models):
        if model not in value_per_model:
            continue
        percentages = 100.0 * value_per_model[model].reindex(parties).fillna(0.0).to_numpy()
        highest_percentage = max(highest_percentage, float(percentages.max()))
        offsets = bar_positions - GROUP_WIDTH / 2 + bar_width * (model_index + 0.5)
        ax.bar(offsets, percentages, width=bar_width,
               color=model_cmap(model_index), edgecolor="black", linewidth=0.3,
               label=model)

    ax.set_xticks(bar_positions)
    # The upper panel of a stacked figure hands its x-labels to the lower one;
    # ticks stay so the party boundaries still read across both panels.
    ax.set_xticklabels(parties if label_x else [""] * len(parties),
                       rotation=20, ha="right", fontsize=TICK_FONTSIZE * scale)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE * scale)
    ax.set_xlim(-0.5, len(parties) - 0.5)
    ax.set_ylabel(f"{value_label} (%)", fontsize=AXIS_LABEL_FONTSIZE * scale)
    ax.set_ylim(*(ylim or (0.0, min(100.0, highest_percentage + PERCENT_HEADROOM))))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def plot_models_party_bars(value_per_model, value_label, output_path, ylim=None):
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
                           plt.get_cmap("tab20", max(len(models), 2)), value_label, ylim)
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
                            value_label, ylim, label_x=True, scale=1.0):
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
    ax.set_ylabel(f"{value_label} (%)", fontsize=AXIS_LABEL_FONTSIZE * scale)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def model_legend_handles(models, model_cmap, label_for=str):
    """Patches matching the box faces, so a box panel (which has no bar artists to
    label) can carry the same legend a bar panel draws for itself."""
    return [Patch(facecolor=model_cmap(index), edgecolor="black", linewidth=0.7,
                  alpha=0.85, label=label_for(model)) for index, model in enumerate(models)]


def fitted_legend(ax, handles, fontsize, header_fontsize, bbox_to_anchor, ncol,
                  smallest_fontsize):
    """The legend at `ncol` columns, set as large as it can be without running past the
    figure's own width -- `fontsize` if it fits, stepped down towards `smallest_fontsize`
    if it does not.

    A legend wider than the figure is not clipped: `bbox_inches="tight"` widens the saved
    image to hold it while the panels keep the width they were laid out with, so an
    overlong legend row silently shrinks the panels to a band down the middle of the
    page. Measuring is the only way to know -- a legend shrink-wraps its text, so its
    width depends on the model names it happens to be drawing. The column count is what
    is held fixed rather than the type size: the columns are what make the legend read as
    a grid against the panels above it, and a column lost costs a whole row of height."""
    renderer = ax.figure.canvas.get_renderer()
    figure_width_px = ax.figure.get_figwidth() * ax.figure.dpi
    header_ratio = header_fontsize / fontsize
    while True:
        legend = ax.legend(handles=handles, ncol=ncol, fontsize=fontsize,
                           bbox_to_anchor=bbox_to_anchor, **COMBINED_LEGEND_STYLE)
        legend.get_texts()[0].set_fontsize(fontsize * header_ratio)
        if legend.get_window_extent(renderer).width <= figure_width_px:
            return legend
        if fontsize <= smallest_fontsize:
            return legend
        legend.remove()
        fontsize = max(smallest_fontsize, fontsize - 0.5)


def legend_header_handle(title):
    """An entry with nothing drawn for it, so a legend's title can sit in the first
    cell of its grid rather than on a row of its own above it -- a row that on a
    figure this tall is the difference between the legend reading with the panels
    and reading as a block under them."""
    return Patch(alpha=0.0, linewidth=0.0, label=title)


def plot_models_agreement_and_source_panels(replicates_per_model, values_per_model_per_source,
                                            agreement_label, probability_label, output_path,
                                            box_ylim=None, bar_ylim=None):
    """The VAA-agreement box figure and the two per-source probability panels drawn as
    one three-panel figure sharing a single model legend.

    The three were printed side by side carrying the same 13-entry model legend twice
    over, which at this type size costs more of the page than the panels themselves. One
    x-axis of EP groups runs down all three, and a model keeps one colour throughout, so
    a model's agreement can be read against its own classified shares by eye.

    Only the y-range is per-panel: agreement sits in a narrow band well away from zero
    (see plot_models_party_boxes), while the probability panels are zero-based and share
    their range with each other."""
    panels = [(title, values) for title, values in values_per_model_per_source.items() if values]
    if not replicates_per_model or not panels:
        return

    parties = ordered_parties_present(
        set().union(
            *(replicates.index for replicates in replicates_per_model.values()),
            *(series.index for _, values in panels for series in values.values()),
        ))
    # The union across all three panels, so a model present in only one still gets a
    # legend entry and one colour.
    models = sorted(
        set(replicates_per_model).union(
            model for _, values in panels for model in values))
    if not parties or not models:
        return

    model_cmap = plt.get_cmap("tab20", max(len(models), 2))
    bar_ylim = bar_ylim or (0.0, min(100.0, PERCENT_HEADROOM + max(
        100.0 * float(np.nanmax(series.to_numpy(dtype=float)))
        for _, values in panels for series in values.values())))

    figure_width = max(13.0, len(parties) * 2.0)
    panel_count = len(panels) + 1
    fig, axes = plt.subplots(panel_count, 1,
                             figsize=(figure_width, 5.5 * panel_count), sharex=True)
    draw_models_party_boxes(axes[0], replicates_per_model, parties, models, model_cmap,
                            agreement_label, box_ylim, label_x=False)
    axes[0].set_title("VAA agreement", fontsize=TITLE_FONTSIZE)
    for index, (ax, (title, values)) in enumerate(zip(axes[1:], panels), start=1):
        draw_models_party_bars(ax, values, parties, models, model_cmap, probability_label,
                               bar_ylim, label_x=index == panel_count - 1)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Hung off the bottom panel, so the offset is measured against one panel's height
    # rather than the whole stack's. The handles are built rather than collected: the
    # box panel labels nothing, and the bar panels each label only the models they drew.
    # "model" is a handle-less entry rather than the legend's title, which puts it in
    # the grid's top-left cell with the first entry of every other column beside it.
    fitted_legend(
        axes[-1],
        [legend_header_handle("model")]
        + model_legend_handles(models, model_cmap, model_display_name),
        fontsize=LEGEND_FONTSIZE * COMBINED_LEGEND_SCALE,
        header_fontsize=LEGEND_TITLE_FONTSIZE * COMBINED_LEGEND_SCALE,
        bbox_to_anchor=(0.5, COMBINED_LEGEND_OFFSET),
        ncol=min(len(models) + 1, MODEL_LEGEND_NCOL),
        smallest_fontsize=LEGEND_FONTSIZE,
    )
    save_figure(fig, output_path, bbox_inches="tight")


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


def plot_cross_model_vaa_boxes(replicates_by_model_source_variant, value_label,
                               dataset_plots_dir, filename_stem, ylim=None):
    """One box figure per scope, mirroring plot_cross_model_party_bars. The scopes share
    a y-axis; pass `ylim` to widen that to a whole run of calls."""
    ylim = ylim or percent_box_limits(replicates_by_model_source_variant)
    for source, variant_label, suffix, scope_label in cross_model_scopes(replicates_by_model_source_variant):
        scoped = scoped_replicates_per_model(
            replicates_by_model_source_variant, source, variant_label)
        if len(scoped) < 2:
            print(f"  only {len(scoped)} model(s) for '{scope_label}', skipping.")
            continue
        plot_models_party_boxes(
            scoped, value_label, dataset_plots_dir / f"{filename_stem}{suffix}.png", ylim)


def plot_cross_model_party_bars(values_by_model_source_variant, value_label,
                                dataset_plots_dir, filename_stem, ylim=None):
    """One grouped-bar figure per scope: pooled, per framing, per source, per cell. The
    scopes share a y-axis; pass `ylim` to widen that to a whole run of calls."""
    ylim = ylim or percent_bar_limits(values_by_model_source_variant)
    for source, variant_label, suffix, scope_label in cross_model_scopes(values_by_model_source_variant):
        value_per_model = scoped_mean_per_model(values_by_model_source_variant, source, variant_label)
        if len(value_per_model) < 2:
            print(f"  only {len(value_per_model)} model(s) for '{scope_label}', skipping.")
            continue
        plot_models_party_bars(
            value_per_model, value_label, dataset_plots_dir / f"{filename_stem}{suffix}.png", ylim)


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
        bar_ylim=None, box_ylim=None):
    """plot_cross_model_source_panels' figure with the VAA agreement boxes stacked on
    top of it: one figure per framing scope, since the lower panels ARE the sources.

    The agreement panel pools over sources -- it is drawn from the raw answers of both
    tracks, the way plot_cross_model_vaa_boxes' pooled figure is -- so it is the
    framing, not the source, that the scopes vary."""
    bar_ylim = bar_ylim or percent_bar_limits(values_by_model_source_variant)
    box_ylim = box_ylim or percent_box_limits(replicates_by_model_source_variant)
    keys = set(values_by_model_source_variant)
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
        replicates_per_model = scoped_replicates_per_model(
            replicates_by_model_source_variant, None, variant_label)
        if any(len(values) < 2 for values in panels.values()) or len(replicates_per_model) < 2:
            print(f"  fewer than two models in a panel for "
                  f"'{variant_label or 'pooled'}', skipping the combined figure.")
            continue
        plot_models_agreement_and_source_panels(
            replicates_per_model, panels, agreement_label, probability_label,
            dataset_plots_dir / f"{filename_stem}{suffix}.png", box_ylim, bar_ylim)


def discover_vaa_csvs(model_dir):
    vaa_by_source_and_variant = {}
    for path in model_dir.glob("vaa*.csv"):
        if (match := VAA_SPEECHES_PATTERN.match(path.name)):
            classifier_source = "speeches"
        elif (match := VAA_LIKERT_PATTERN.match(path.name)):
            classifier_source = "reasons"
        else:
            continue
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

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(statement_index), size=(bootstrap_draws, len(statement_index)))

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
                stance_df, languages, party_df, groups, statement_index)
            if not matrices[1].any():
                continue
            _, replicates = bootstrap_group_means(matrices, draws)
            replicates_per_model_source_variant[(model_dir.name, source, variant_label)] = (
                pd.DataFrame(replicates, index=groups))
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


def plots_dir_for(model_dir):
    plots_dir = model_dir / "plots"
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
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
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

    if len(models_with_data) > 1:
        dataset_plots_dir.mkdir(exist_ok=True)
        print("\nCross-model party distributions:")
        plot_cross_model_party_bars(
            mean_probability_per_model_source_variant, MEAN_PROBABILITY_LABEL,
            dataset_plots_dir, "classified_all_models_party_distribution",
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
        plot_cross_model_vaa_boxes(
            vaa_replicates_per_model_source_variant, MEAN_AGREEMENT_LABEL,
            dataset_plots_dir, "vaa_all_models_mean_agreement",
        )
        if len(models_with_data) > 1:
            # The agreement boxes and the two per-source probability panels in one
            # three-panel figure: they carry the same model legend, and the thesis
            # prints them together.
            plot_cross_model_agreement_and_source_panels(
                mean_probability_per_model_source_variant,
                vaa_replicates_per_model_source_variant,
                MEAN_AGREEMENT_LABEL, MEAN_PROBABILITY_LABEL,
                dataset_plots_dir, "vaa_agreement_and_classified_sources",
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
