import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

VARIANT_SUFFIX_TO_LABEL = {"": "base", "_negated": "negated", "_question": "question"}
VARIANT_LABEL_ORDER = ["base", "negated", "question"]
SOURCE_LABELS = ["speeches", "reasons"]
VAA_SOURCE_FOR_CLASSIFIER_SOURCE = {"speeches": "vaa_speeches", "reasons": "vaa_likert"}

SPEECHES_CLASSIFIED_PATTERN = re.compile(
    r"^speeches_[a-z]{2}(?:,[a-z]{2})*(?P<variant>|_negated|_question)_classified\.csv$"
)
REASONS_CLASSIFIED_PATTERN = re.compile(
    r"^(?!speeches_)[a-z]{2}(?:,[a-z]{2})*(?P<variant>|_negated|_question)_classified\.csv$"
)
VAA_LIKERT_PATTERN = re.compile(
    r"^vaa(?P<variant>|_negated|_question)_[a-z]{2}(?:,[a-z]{2})*\.csv$"
)
VAA_SPEECHES_PATTERN = re.compile(
    r"^vaa_speeches(?P<variant>|_negated|_question)_[a-z]{2}(?:,[a-z]{2})*\.csv$"
)

PREDICTED_PARTY_COLUMN_PATTERN = re.compile(
    r"^predicted_party_(?P<language>[a-z]{2})(?P<variant>|_negated|_question)_v(?P<variant_idx>\d+)$"
)
PARTY_PROBABILITY_COLUMN_PATTERN = re.compile(
    r"^party_prob_(?P<party_slug>.+?)_(?P<language>[a-z]{2})(?P<variant>|_negated|_question)_v(?P<variant_idx>\d+)$"
)

PARTY_DISPLAY_ORDER = ["GUE/NGL", "S&D", "Greens/EFA", "ALDE", "PPE", "ECR", "ID"]
PARTY_COLORS = {
    "GUE/NGL":    "#BB1E10",
    "S&D":        "#E2061D",
    "Greens/EFA": "#5DA13F",
    "ALDE":       "#FAD22D",
    "PPE":        "#3399FF",
    "ECR":        "#0054A5",
    "ID":         "#2B3856",
}
FALLBACK_PARTY_COLOR = "#888888"

MEAN_PROBABILITY_LABEL = "Mean probability"
LEGEND_UPPER_RIGHT = "upper right"


def slugify_party_label(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


def save_figure(fig, output_path, **savefig_kwargs):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, **savefig_kwargs)
    plt.close(fig)
    print(f"  Saved {output_path.relative_to(output_path.parents[3])}")


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


def plot_party_distribution_bar(mean_probability, predicted_share, title, output_path):
    parties_present = set(mean_probability.index) | set(predicted_share.index)
    parties = ordered_parties_present(parties_present)
    if not parties:
        return

    bar_positions = np.arange(len(parties))
    bar_width = 0.4
    probabilities = mean_probability.reindex(parties).fillna(0.0).to_numpy()
    shares = predicted_share.reindex(parties).fillna(0.0).to_numpy()
    bar_colors = [color_for_party(party) for party in parties]

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 1.1), 5.5))
    ax.bar(bar_positions - bar_width / 2, probabilities, width=bar_width,
           color=bar_colors, edgecolor="black", linewidth=0.5, label=MEAN_PROBABILITY_LABEL)
    ax.bar(bar_positions + bar_width / 2, shares, width=bar_width,
           color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.55,
           hatch="//", label="Argmax share")

    for position, probability in zip(bar_positions - bar_width / 2, probabilities):
        ax.text(position, probability + 0.005, f"{probability:.2f}", ha="center", va="bottom", fontsize=7)
    for position, share in zip(bar_positions + bar_width / 2, shares):
        ax.text(position, share + 0.005, f"{share:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean class probability  /  predicted-party share")
    upper_limit = max(probabilities.max(), shares.max(), 0.2) + 0.1
    ax.set_ylim(0.0, min(1.0, upper_limit))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=8, framealpha=0.9)
    save_figure(fig, output_path)


def plot_party_distribution_across_variants(mean_probability_by_variant, title, output_path):
    parties_present = set().union(*(series.index for series in mean_probability_by_variant.values()))
    parties = ordered_parties_present(parties_present)
    variants_present = [v for v in VARIANT_LABEL_ORDER if v in mean_probability_by_variant]
    if not parties or not variants_present:
        return

    bar_positions = np.arange(len(parties))
    group_width = 0.8
    bar_width = group_width / len(variants_present)
    variant_cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 1.2), 5.5))
    for variant_index, variant_label in enumerate(variants_present):
        probabilities = mean_probability_by_variant[variant_label].reindex(parties).fillna(0.0).to_numpy()
        offsets = bar_positions - group_width / 2 + bar_width * (variant_index + 0.5)
        ax.bar(offsets, probabilities, width=bar_width,
               color=variant_cmap(variant_index), edgecolor="black", linewidth=0.4,
               label=variant_label)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean class probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=8, framealpha=0.9, title="variant")
    save_figure(fig, output_path)


def plot_language_party_heatmap(language_party_probability, title, output_path):
    if language_party_probability.empty:
        return
    parties = ordered_parties_present(set(language_party_probability.columns))
    languages = sorted(language_party_probability.index)
    matrix = language_party_probability.reindex(index=languages, columns=parties).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(7, len(parties) * 1.1), max(6, len(languages) * 0.35)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.3, np.nanmax(matrix)))
    fig.colorbar(image, ax=ax, label=MEAN_PROBABILITY_LABEL, fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(parties)), parties, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(languages)), languages, fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    for row_index in range(len(languages)):
        for column_index in range(len(parties)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if value > 0.45 else "black")
    save_figure(fig, output_path)


def plot_speeches_vs_reasons(mean_probability_by_source, title, output_path):
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

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 1.1), 5.5))
    for source_index, source in enumerate(sources_present):
        probabilities = mean_probability_by_source[source].reindex(parties).fillna(0.0).to_numpy()
        offsets = bar_positions - bar_width / 2 + bar_width * source_index
        ax.bar(offsets, probabilities, width=bar_width,
               color=bar_colors, edgecolor="black", linewidth=0.4,
               alpha=1.0 if source == "speeches" else 0.6,
               hatch="" if source == "speeches" else "//",
               label=source)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(parties, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean class probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc=LEGEND_UPPER_RIGHT, fontsize=8, framealpha=0.9, title="source")
    save_figure(fig, output_path)


def plot_models_party_distribution(mean_probability_per_model, source, variant_label, output_path):
    if not mean_probability_per_model:
        return
    parties_present = set().union(*(series.index for series in mean_probability_per_model.values()))
    parties = ordered_parties_present(parties_present)
    models = sorted(mean_probability_per_model)

    matrix = np.array([
        mean_probability_per_model[model].reindex(parties).fillna(0.0).to_numpy()
        for model in models
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 1.1), max(5, len(models) * 0.45)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.4, matrix.max()))
    fig.colorbar(image, ax=ax, label=MEAN_PROBABILITY_LABEL, fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(parties)), parties, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)), models, fontsize=9)
    ax.set_title(f"Mean party probability per model ({source}, {variant_label})", fontsize=11, pad=8)
    for row_index in range(len(models)):
        for column_index in range(len(parties)):
            value = matrix[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value > 0.45 else "black")
    save_figure(fig, output_path)


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


def load_vaa_mean_agreement_per_party(vaa_csv_path):
    df = pd.read_csv(vaa_csv_path)
    if df.empty or "ep_group" not in df.columns or "mean_agreement" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("ep_group")["mean_agreement"].mean()


def correlate_party_vectors(series_a, series_b):
    parties = sorted(set(series_a.index) & set(series_b.index))
    if len(parties) < 2:
        return float("nan"), float("nan")
    vector_a = series_a.reindex(parties).to_numpy(dtype=float)
    vector_b = series_b.reindex(parties).to_numpy(dtype=float)
    mask = ~(np.isnan(vector_a) | np.isnan(vector_b))
    if mask.sum() < 2:
        return float("nan"), float("nan")
    vector_a, vector_b = vector_a[mask], vector_b[mask]
    if vector_a.std() == 0 or vector_b.std() == 0:
        return float("nan"), float("nan")
    pearson_r = float(pearsonr(vector_a, vector_b)[0])
    spearman_rho = float(spearmanr(vector_a, vector_b)[0])
    return pearson_r, spearman_rho


def top_party(series):
    if series.empty or series.isna().all():
        return None, float("nan")
    top_party_label = series.idxmax()
    return top_party_label, float(series.loc[top_party_label])


def build_vaa_comparison_row(model_name, source, variant_label, classifier_mean_prob,
                             classifier_argmax_share, vaa_mean_agreement):
    pearson_mean_prob, spearman_mean_prob = correlate_party_vectors(
        classifier_mean_prob, vaa_mean_agreement,
    )
    pearson_argmax, spearman_argmax = correlate_party_vectors(
        classifier_argmax_share, vaa_mean_agreement,
    )
    classifier_mean_prob_top, classifier_mean_prob_top_value = top_party(classifier_mean_prob)
    classifier_argmax_top, classifier_argmax_top_value = top_party(classifier_argmax_share)
    vaa_top, vaa_top_value = top_party(vaa_mean_agreement)
    return {
        "model": model_name,
        "source": source,
        "variant": variant_label,
        "pearson_r_mean_prob_vs_vaa": pearson_mean_prob,
        "spearman_rho_mean_prob_vs_vaa": spearman_mean_prob,
        "pearson_r_argmax_share_vs_vaa": pearson_argmax,
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
                                   pearson_r, spearman_rho, title, output_path):
    parties = sorted(set(classifier_mean_prob.index) & set(vaa_mean_agreement.index))
    if len(parties) < 2:
        return
    classifier_values = classifier_mean_prob.reindex(parties).to_numpy(dtype=float)
    vaa_values = vaa_mean_agreement.reindex(parties).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for party, vaa_value, classifier_value in zip(parties, vaa_values, classifier_values):
        ax.scatter(vaa_value, classifier_value, s=140, color=color_for_party(party),
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(party, (vaa_value, classifier_value), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(0.3, float(np.nanmax(classifier_values)) + 0.08))
    ax.set_xlabel("VAA mean agreement (averaged over languages)")
    ax.set_ylabel("Classifier mean probability (averaged over texts)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_title(f"{title}\nPearson r = {pearson_r:.3f}   Spearman ρ = {spearman_rho:.3f}",
                 fontsize=11, pad=8)
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

    fig, ax = plt.subplots(figsize=(max(7, pivoted.shape[1] * 1.1),
                                    max(5, pivoted.shape[0] * 0.5)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label=value_label, fraction=0.03, pad=0.02)
    ax.set_xticks(range(pivoted.shape[1]), pivoted.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(pivoted.shape[0]), pivoted.index, fontsize=9)
    ax.set_title(value_label + " between classifier and VAA (per model × source · variant)",
                 fontsize=11, pad=8)
    for row_index in range(pivoted.shape[0]):
        for column_index in range(pivoted.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else "black")
    save_figure(fig, output_path)


def plots_dir_for(model_dir):
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def process_model(model_dir):
    print(f"\nProcessing: {model_dir.name}")
    classified_csvs = discover_classified_csvs(model_dir)
    if not classified_csvs:
        print("  No *_classified.csv files found, skipping.")
        return {}, []

    plots_dir = plots_dir_for(model_dir)
    vaa_csvs = discover_vaa_csvs(model_dir)
    long_by_source_variant = {}
    mean_probability_per_source = {}
    vaa_comparison_rows = []

    for (source, variant_label), csv_path in sorted(classified_csvs.items()):
        long_df = load_long_predictions(csv_path)
        if long_df.empty:
            print(f"  [{source}/{variant_label}] no predictions found in {csv_path.name}")
            continue
        long_by_source_variant[(source, variant_label)] = long_df

        classifier_mean_prob = mean_probability_per_party(long_df)
        classifier_argmax_share = predicted_party_share(long_df)

        plot_party_distribution_bar(
            classifier_mean_prob,
            classifier_argmax_share,
            f"{model_dir.name} – {source} ({variant_label}): party distribution",
            plots_dir / f"classified_{source}_{variant_label}_party_distribution.png",
        )
        plot_language_party_heatmap(
            mean_probability_per_language_and_party(long_df),
            f"{model_dir.name} – {source} ({variant_label}): mean party probability per language",
            plots_dir / f"classified_{source}_{variant_label}_language_party_heatmap.png",
        )

        vaa_csv_path = vaa_csvs.get((source, variant_label))
        if vaa_csv_path is None:
            print(f"  [{source}/{variant_label}] no matching VAA CSV "
                  f"({VAA_SOURCE_FOR_CLASSIFIER_SOURCE[source]}), skipping VAA correlation.")
            continue
        vaa_mean_agreement = load_vaa_mean_agreement_per_party(vaa_csv_path)
        if vaa_mean_agreement.empty:
            print(f"  [{source}/{variant_label}] VAA CSV {vaa_csv_path.name} is empty.")
            continue

        comparison_row = build_vaa_comparison_row(
            model_dir.name, source, variant_label,
            classifier_mean_prob, classifier_argmax_share, vaa_mean_agreement,
        )
        vaa_comparison_rows.append(comparison_row)
        plot_classifier_vs_vaa_scatter(
            classifier_mean_prob, vaa_mean_agreement,
            comparison_row["pearson_r_mean_prob_vs_vaa"],
            comparison_row["spearman_rho_mean_prob_vs_vaa"],
            f"{model_dir.name} – {source} ({variant_label}): classifier mean probability vs VAA mean agreement",
            plots_dir / f"classified_vs_vaa_{source}_{variant_label}_scatter.png",
        )

    for source in SOURCE_LABELS:
        mean_probability_by_variant = {
            variant_label: mean_probability_per_party(long_df)
            for (csv_source, variant_label), long_df in long_by_source_variant.items()
            if csv_source == source
        }
        if mean_probability_by_variant:
            plot_party_distribution_across_variants(
                mean_probability_by_variant,
                f"{model_dir.name} – {source}: party distribution across variants",
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
            f"{model_dir.name} – speeches vs reasons: party distribution (all variants)",
            plots_dir / "classified_speeches_vs_reasons_party_distribution.png",
        )

    if vaa_comparison_rows:
        summary_path = model_dir / "classified_vs_vaa_summary.csv"
        pd.DataFrame(vaa_comparison_rows).to_csv(summary_path, index=False)
        print(f"  Wrote {summary_path.relative_to(summary_path.parents[2])}")

    classifier_mean_prob_per_source_variant = {
        (source, variant_label): mean_probability_per_party(long_df)
        for (source, variant_label), long_df in long_by_source_variant.items()
    }
    return classifier_mean_prob_per_source_variant, vaa_comparison_rows


def plot_cross_model_comparisons(mean_probability_per_model_source_variant, dataset_plots_dir):
    grouped = {}
    for (model, source, variant_label), series in mean_probability_per_model_source_variant.items():
        grouped.setdefault((source, variant_label), {})[model] = series

    for (source, variant_label), per_model in sorted(grouped.items()):
        if len(per_model) < 2:
            continue
        plot_models_party_distribution(
            per_model, source, variant_label,
            dataset_plots_dir / f"classified_{source}_{variant_label}_models.png",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None)
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

    mean_probability_per_model_source_variant = {}
    vaa_comparison_rows_across_models = []
    for model_dir in model_dirs:
        per_source_variant, vaa_comparison_rows = process_model(model_dir)
        for (source, variant_label), series in per_source_variant.items():
            mean_probability_per_model_source_variant[(model_dir.name, source, variant_label)] = series
        vaa_comparison_rows_across_models.extend(vaa_comparison_rows)

    models_with_data = {model for model, _, _ in mean_probability_per_model_source_variant}
    if len(models_with_data) > 1:
        dataset_plots_dir = results_dir / "plots"
        dataset_plots_dir.mkdir(exist_ok=True)
        plot_cross_model_comparisons(mean_probability_per_model_source_variant, dataset_plots_dir)
        plot_cross_model_vaa_correlation_heatmap(
            vaa_comparison_rows_across_models,
            "pearson_r_mean_prob_vs_vaa", "Pearson r (mean prob vs VAA agreement)",
            dataset_plots_dir / "classified_vs_vaa_pearson_models.png",
        )
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
