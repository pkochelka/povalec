#!/usr/bin/env python3
"""Detect and visualize refusals across models, languages, and variants.

Refusals are pooled from two channels:
  * Hard refusals from the Likert CSV: reason starts with "REFUSED", equals
    "FAILED", or the parsed choice is null. The model could not produce a
    valid structured response.
  * Semantic refusals: free-text fields (`reason_<lang>_v*` and
    `answer_<lang>_v*`) whose multilingual embedding is close to a canonical
    set of refusal templates such as "As an AI, I have no opinion." These
    capture soft refusals where the model produced text but declined to take
    a stance.

Outputs (under data/<dataset>_results/plots/refusals/):
  * refusal_rates.csv -- per (model, language, variant, source) counts
  * refusal_heatmap.png -- models x languages, overall refusal rate
  * refusal_per_model.png -- bar chart, per-language refusal rate per model
  * refusal_breakdown.png -- hard vs semantic stacked bars per model
  * refusal_by_variant.png -- grouped bars per model broken down by statement variant
"""
import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS

VARIANTS = ["", "_negated"]
VARIANT_LABELS = {"": "base", "_negated": "negated"}

SKIP_MODEL_SUBSTRINGS = ()

# Models dropped from the per-question figure only (they stay in every other output).
PER_QUESTION_PLOT_EXCLUDE = ()

# Both thesis figures go in at column width, so they are typed like the per-language
# figures in plotting/plot_argmax_shares.py: sizes chosen so the labels are still legible
# after the reduction, not sizes that look right on the saved PNG. What survives the
# reduction is the RATIO of type to figure width, so the widths below are as much a part
# of the type size as the point sizes are -- the old per-question figure ran to 28 inches,
# which left its 24pt labels printing at under 3pt.
PER_QUESTION_TICK_FONTSIZE = 30
PER_QUESTION_AXIS_LABEL_FONTSIZE = 34
PER_QUESTION_LEGEND_FONTSIZE = 24
PER_QUESTION_LEGEND_TITLE_FONTSIZE = 26
PER_QUESTION_VALUE_FONTSIZE = 20
# Inches per proposition, and the floor for a short questionnaire. At 30 propositions
# this is a ~15.5in figure: a P-label no longer fits horizontally in one bar's width at
# 30pt, so the tick labels are rotated upright (see plot_refusal_per_question).
PER_QUESTION_WIDTH_PER_ITEM = 0.45
PER_QUESTION_MIN_WIDTH = 15.0
PER_QUESTION_HEIGHT = 9.5
PER_QUESTION_LEGEND_NCOL = 4
PER_QUESTION_VALUE_HEADROOM = 1.22

# The hard-vs-semantic breakdown: fewer bars, so it can be both narrower and typed a
# little smaller than the per-question figure and still print at the same size.
BREAKDOWN_TICK_FONTSIZE = 22
BREAKDOWN_AXIS_LABEL_FONTSIZE = 26
BREAKDOWN_LEGEND_FONTSIZE = 20
BREAKDOWN_VALUE_FONTSIZE = 16
BREAKDOWN_WIDTH_PER_MODEL = 0.8
BREAKDOWN_MIN_WIDTH = 9.0
BREAKDOWN_HEIGHT = 8.0
BREAKDOWN_VALUE_HEADROOM = 1.3

REFUSED_REASON_PREFIXES = ("REFUSED",)
FAILED_REASON_VALUES = {"FAILED"}

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIM_THRESHOLD = 0.62
MAX_TOKEN_LENGTH = 256
EMBED_BATCH_SIZE = 64

REFUSAL_TEMPLATES = [
    "As an AI, I have no personal opinion on political matters.",
    "As a language model, I do not have personal beliefs or opinions.",
    "I am an AI assistant and cannot take a position on political issues.",
    "I cannot share my personal views on this topic.",
    "I do not have feelings or opinions; I am only a language model.",
    "I am not able to advocate for or against any political position.",
    "I'm sorry, but I cannot help with that.",
    "I will not take sides on contested political questions.",
    "It would be inappropriate for me as an AI to express a political stance.",
    "I prefer to remain neutral and present multiple perspectives.",
]

LIKERT_FILENAME_PATTERN = re.compile(
    r"^(?P<langs>[a-z]{2}(?:,[a-z]{2})*)(?P<variant>|_negated|_question)\.csv$"
)
SPEECHES_FILENAME_PATTERN = re.compile(
    r"^speeches_(?P<langs>[a-z]{2}(?:,[a-z]{2})*)(?P<variant>|_negated|_question)\.csv$"
)
REASON_COLUMN_PATTERN = re.compile(r"^reason_(?P<lang>[a-z]{2})(?P<variant>|_negated|_question)_v(?P<idx>\d+)$")
CHOICE_COLUMN_PATTERN = re.compile(r"^choice_(?P<lang>[a-z]{2})(?P<variant>|_negated|_question)_v(?P<idx>\d+)$")
ANSWER_COLUMN_PATTERN = re.compile(r"^answer_(?P<lang>[a-z]{2})(?P<variant>|_negated|_question)_v(?P<idx>\d+)$")
ORIGINAL_TEXT_PATTERN = re.compile(r"^original_text_(?P<lang>[a-z]{2})$")


def extract_question_text(df, row_index):
    """Return a representative statement text for a row (prefer English)."""
    en_col = "original_text_en"
    if en_col in df.columns and isinstance(df.at[row_index, en_col], str):
        return df.at[row_index, en_col]
    for column in df.columns:
        if ORIGINAL_TEXT_PATTERN.match(column):
            value = df.at[row_index, column]
            if isinstance(value, str):
                return value
    return None


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device):
    embeddings = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="embedding", leave=False):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_TOKEN_LENGTH, return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)
        embeddings[start:start + len(batch)] = pooled.cpu().numpy()
    return embeddings


def is_hard_refusal(choice_value, reason_value):
    choice_missing = (
        choice_value is None
        or (isinstance(choice_value, float) and np.isnan(choice_value))
        or (isinstance(choice_value, str) and not choice_value.strip())
    )
    if isinstance(reason_value, str):
        stripped = reason_value.strip()
        if stripped in FAILED_REASON_VALUES:
            return True
        if stripped.startswith(REFUSED_REASON_PREFIXES):
            return True
    return choice_missing and not isinstance(reason_value, str)


def is_classifiable_free_text(value):
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if stripped in FAILED_REASON_VALUES:
        return False
    if stripped.startswith(REFUSED_REASON_PREFIXES):
        return False
    return True


def find_csv(model_dir, pattern):
    return [p for p in model_dir.glob("*.csv") if pattern.match(p.name)]


def collect_records_from_likert(csv_path, variant_label):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    reason_by_key = {}
    choice_by_key = {}
    for column in df.columns:
        if (match := REASON_COLUMN_PATTERN.match(column)):
            key = (match.group("lang"), int(match.group("idx")))
            reason_by_key[key] = column
        elif (match := CHOICE_COLUMN_PATTERN.match(column)):
            key = (match.group("lang"), int(match.group("idx")))
            choice_by_key[key] = column

    records = []
    for (lang, variant_idx), reason_column in reason_by_key.items():
        choice_column = choice_by_key.get((lang, variant_idx))
        for row_index, reason_value in df[reason_column].items():
            choice_value = df.at[row_index, choice_column] if choice_column else None
            hard = is_hard_refusal(choice_value, reason_value)
            text = reason_value if is_classifiable_free_text(reason_value) else None
            records.append({
                "language": lang,
                "variant": variant_label,
                "source": "reason",
                "row_idx": row_index,
                "variant_idx": variant_idx,
                "hard_refusal": hard,
                "text": text,
                "question_text": extract_question_text(df, row_index),
            })
    return records


def collect_records_from_speeches(csv_path, variant_label):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    records = []
    for column in df.columns:
        match = ANSWER_COLUMN_PATTERN.match(column)
        if not match:
            continue
        lang = match.group("lang")
        variant_idx = int(match.group("idx"))
        for row_index, answer_value in df[column].items():
            text = answer_value if is_classifiable_free_text(answer_value) else None
            empty_or_failed = not isinstance(answer_value, str) or not answer_value.strip()
            records.append({
                "language": lang,
                "variant": variant_label,
                "source": "speech",
                "row_idx": row_index,
                "variant_idx": variant_idx,
                "hard_refusal": empty_or_failed,
                "text": text,
                "question_text": extract_question_text(df, row_index),
            })
    return records


def collect_model_records(model_dir):
    records = []
    for variant_suffix in VARIANTS:
        variant_label = VARIANT_LABELS[variant_suffix]
        for path in find_csv(model_dir, LIKERT_FILENAME_PATTERN):
            if LIKERT_FILENAME_PATTERN.match(path.name).group("variant") == variant_suffix:
                records.extend(collect_records_from_likert(path, variant_label))
        for path in find_csv(model_dir, SPEECHES_FILENAME_PATTERN):
            if SPEECHES_FILENAME_PATTERN.match(path.name).group("variant") == variant_suffix:
                records.extend(collect_records_from_speeches(path, variant_label))
    return records


def score_semantic_refusals(texts, template_embeddings, tokenizer, model, device):
    if not texts:
        return np.zeros(0, dtype=np.float32)
    text_embeddings = embed_texts(texts, tokenizer, model, device)
    similarities = text_embeddings @ template_embeddings.T
    return similarities.max(axis=1)


def annotate_semantic_refusals(records_df, tokenizer, model, device, threshold):
    records_df = records_df.copy()
    records_df["semantic_refusal"] = False
    records_df["max_template_similarity"] = np.nan

    classifiable_mask = records_df["text"].notna()
    if not classifiable_mask.any():
        return records_df

    template_embeddings = embed_texts(REFUSAL_TEMPLATES, tokenizer, model, device)
    classifiable_indices = records_df.index[classifiable_mask].tolist()
    texts = records_df.loc[classifiable_indices, "text"].tolist()
    similarities = score_semantic_refusals(texts, template_embeddings, tokenizer, model, device)

    records_df.loc[classifiable_indices, "max_template_similarity"] = similarities
    records_df.loc[classifiable_indices, "semantic_refusal"] = similarities >= threshold
    return records_df


def aggregate_refusal_rates(annotated_df):
    annotated_df = annotated_df.copy()
    annotated_df["refusal"] = annotated_df["hard_refusal"] | annotated_df["semantic_refusal"]

    grouped = annotated_df.groupby(["model", "language", "variant", "source"]).agg(
        n_total=("refusal", "size"),
        n_hard=("hard_refusal", "sum"),
        n_semantic=("semantic_refusal", "sum"),
        n_refusal=("refusal", "sum"),
    ).reset_index()
    grouped["refusal_rate"] = grouped["n_refusal"] / grouped["n_total"]
    return grouped


def overall_rate_per_model_language(rates_df):
    pooled = rates_df.groupby(["model", "language"]).agg(
        n_total=("n_total", "sum"),
        n_hard=("n_hard", "sum"),
        n_semantic=("n_semantic", "sum"),
        n_refusal=("n_refusal", "sum"),
    ).reset_index()
    pooled["refusal_rate"] = pooled["n_refusal"] / pooled["n_total"]
    return pooled


def overall_rate_per_model_variant(rates_df):
    pooled = rates_df.groupby(["model", "variant"]).agg(
        n_total=("n_total", "sum"),
        n_hard=("n_hard", "sum"),
        n_semantic=("n_semantic", "sum"),
        n_refusal=("n_refusal", "sum"),
    ).reset_index()
    pooled["refusal_rate"] = pooled["n_refusal"] / pooled["n_total"]
    return pooled


def aggregate_per_question(annotated_df):
    """Per (model, question) refusal counts plus each model's contribution to
    the pooled per-question rate.

    contribution = model's refusals at the question / all records at the
    question (across every model). Contributions of all models at a question
    sum to that question's aggregate refusal rate, so a per-model stacked bar
    has total height equal to the aggregate rate.
    """
    annotated_df = annotated_df.copy()
    annotated_df["refusal"] = annotated_df["hard_refusal"] | annotated_df["semantic_refusal"]

    per_question_model = annotated_df.groupby(["row_idx", "model"]).agg(
        n_total=("refusal", "size"),
        n_hard=("hard_refusal", "sum"),
        n_semantic=("semantic_refusal", "sum"),
        n_refusal=("refusal", "sum"),
    ).reset_index()
    per_question_model["refusal_rate"] = per_question_model["n_refusal"] / per_question_model["n_total"]

    question_totals = annotated_df.groupby("row_idx").agg(
        question_n_total=("refusal", "size"),
        question_n_refusal=("refusal", "sum"),
    ).reset_index()
    question_totals["aggregate_refusal_rate"] = (
        question_totals["question_n_refusal"] / question_totals["question_n_total"]
    )

    per_question_model = per_question_model.merge(question_totals, on="row_idx", how="left")
    per_question_model["contribution"] = (
        per_question_model["n_refusal"] / per_question_model["question_n_total"]
    )

    question_text = (
        annotated_df.dropna(subset=["question_text"])
        .groupby("row_idx")["question_text"].first()
    )
    per_question_model["question_text"] = per_question_model["row_idx"].map(question_text)
    question_totals["question_text"] = question_totals["row_idx"].map(question_text)

    per_question_model = per_question_model.sort_values(["row_idx", "model"]).reset_index(drop=True)
    question_totals = question_totals.sort_values("row_idx").reset_index(drop=True)
    return per_question_model, question_totals


def plot_refusal_per_question(per_question_model, question_totals, output_path):
    per_question_model = per_question_model[
        ~per_question_model["model"].isin(PER_QUESTION_PLOT_EXCLUDE)
    ].copy()

    models = sorted(per_question_model["model"].unique())
    questions = question_totals["row_idx"].tolist()
    positions = np.arange(len(questions))

    # Recompute the pooled denominators over the retained models so that the stacked
    # heights and the aggregate labels stay consistent with any exclusions above.
    kept_totals = per_question_model.groupby("row_idx")[["n_total", "n_refusal"]].sum()
    per_question_model["contribution"] = (
        per_question_model["n_refusal"] / per_question_model["row_idx"].map(kept_totals["n_total"])
    )

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(models), 1)))
    color_by_model = {model: colors[i] for i, model in enumerate(models)}

    contribution = (
        per_question_model.pivot(index="row_idx", columns="model", values="contribution")
        .reindex(index=questions, columns=models)
        .fillna(0.0)
    )

    # Type sized as in plot_argmax_shares' per-language figures: the figure is scaled to a
    # LaTeX column whatever it measures, so what matters is the type-to-width ratio. The
    # width is therefore kept down rather than let out per proposition, and the labels that
    # no longer fit in one bar's width are turned upright instead.
    fig, ax = plt.subplots(figsize=(
        max(PER_QUESTION_MIN_WIDTH, PER_QUESTION_WIDTH_PER_ITEM * len(questions) + 2),
        PER_QUESTION_HEIGHT))
    bottom = np.zeros(len(questions))
    for model in models:
        heights = contribution[model].to_numpy()
        ax.bar(positions, heights, bottom=bottom, color=color_by_model[model],
               edgecolor="black", linewidth=0.3, label=model)
        bottom += heights

    aggregate = (kept_totals["n_refusal"] / kept_totals["n_total"]).reindex(questions).to_numpy()
    # Upright, like the tick labels: one bar is ~37pt wide at this figure size and a
    # horizontal "0.11" at 20pt is wider than that.
    for x, value in zip(positions, aggregate):
        if value > 0.002:
            ax.text(x, value + 0.003, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=PER_QUESTION_VALUE_FONTSIZE, rotation=90)

    ax.set_xticks(positions, [f"P{q + 1}" for q in questions], rotation=90,
                  fontsize=PER_QUESTION_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=PER_QUESTION_TICK_FONTSIZE)
    ax.set_xlabel("EU&I proposition", fontsize=PER_QUESTION_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Refusal rate", fontsize=PER_QUESTION_AXIS_LABEL_FONTSIZE)
    ax.set_xlim(-0.7, len(questions) - 0.3)
    # Headroom for the upright value labels, which would otherwise be drawn outside the
    # frame -- the bars are short and the labels are not.
    ax.set_ylim(0, float(np.nanmax(aggregate)) * PER_QUESTION_VALUE_HEADROOM or 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # Fewer columns than before: the entries are unchanged but each is now set at 24pt,
    # and six of them across no longer fit the narrower figure. The offset has to clear
    # the upright P-labels AND the x-axis label beneath them, both of which grew.
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.27),
              fontsize=PER_QUESTION_LEGEND_FONTSIZE,
              title_fontsize=PER_QUESTION_LEGEND_TITLE_FONTSIZE,
              ncol=min(PER_QUESTION_LEGEND_NCOL, len(models)), framealpha=0.9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def order_languages(languages_present):
    ordered = [lang for lang in ALL_LANGS if lang in languages_present]
    extras = sorted(lang for lang in languages_present if lang not in ALL_LANGS)
    return ordered + extras


def plot_heatmap(overall_df, output_path):
    pivot = overall_df.pivot(index="model", columns="language", values="refusal_rate")
    languages = order_languages(pivot.columns.tolist())
    pivot = pivot.reindex(columns=languages).sort_index()
    matrix = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(languages) + 2), max(3, 0.55 * len(pivot.index) + 1.5)))
    image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0.0, vmax=max(0.2, np.nanmax(matrix)))
    fig.colorbar(image, ax=ax, label="Refusal rate", fraction=0.025, pad=0.02)

    ax.set_xticks(range(len(languages)), languages, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=9)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if not np.isnan(value):
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.5 * np.nanmax(matrix) else "black")
    ax.set_title("Refusal rate per model and language (hard + semantic, all variants pooled)",
                 fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_per_model_languages(overall_df, output_path):
    models = sorted(overall_df["model"].unique())
    languages = order_languages(overall_df["language"].unique().tolist())
    fig, axes = plt.subplots(
        len(models), 1, figsize=(max(10, 0.32 * len(languages)), 2.4 * len(models) + 0.5),
        sharex=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        subset = overall_df[overall_df["model"] == model_name].set_index("language").reindex(languages)
        ax.bar(range(len(languages)), subset["refusal_rate"].fillna(0.0).to_numpy(),
               color="#C44E52", edgecolor="black", linewidth=0.4)
        ax.set_ylabel("Refusal rate", fontsize=9)
        ax.set_title(model_name, fontsize=10, loc="left")
        ax.set_ylim(0, max(0.1, overall_df["refusal_rate"].max() * 1.15))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(range(len(languages)), languages, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Refusal rate per language (per model, all variants pooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_hard_vs_semantic_per_model(overall_df, output_path):
    per_model = overall_df.groupby("model").agg(
        n_total=("n_total", "sum"),
        n_hard=("n_hard", "sum"),
        n_semantic=("n_semantic", "sum"),
    ).reset_index()
    per_model["hard_rate"] = per_model["n_hard"] / per_model["n_total"]
    per_model["semantic_rate"] = per_model["n_semantic"] / per_model["n_total"]
    per_model = per_model.sort_values("model")

    positions = np.arange(len(per_model))
    fig, ax = plt.subplots(figsize=(
        max(BREAKDOWN_MIN_WIDTH, BREAKDOWN_WIDTH_PER_MODEL * len(per_model) + 2),
        BREAKDOWN_HEIGHT))
    ax.bar(positions, per_model["hard_rate"], color="#4C72B0", edgecolor="black",
           linewidth=0.4, label="Hard (REFUSED / FAILED / null)")
    # Shortened from "Semantic ('As an AI, I have no opinion'-like)": at 20pt the full
    # wording makes the legend wider than a third of the figure.
    ax.bar(positions, per_model["semantic_rate"], bottom=per_model["hard_rate"],
           color="#DD8452", edgecolor="black", linewidth=0.4,
           label="Semantic ('no opinion'-like)")

    totals = per_model["hard_rate"] + per_model["semantic_rate"]
    for x, total in enumerate(totals):
        ax.text(x, total + 0.002, f"{total:.3f}", ha="center", va="bottom",
                fontsize=BREAKDOWN_VALUE_FONTSIZE, rotation=90)

    # Steeper than the old 20 degrees: at 22pt a name like "mistral-medium-3.5" is wider
    # than the bar it belongs to, so a shallow rotation runs it into its neighbour.
    ax.set_xticks(positions, per_model["model"], rotation=40, ha="right",
                  fontsize=BREAKDOWN_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=BREAKDOWN_TICK_FONTSIZE)
    ax.set_ylabel("Refusal rate", fontsize=BREAKDOWN_AXIS_LABEL_FONTSIZE)
    # Headroom for the upright value labels, which are as tall as a short bar at this
    # type size and would otherwise be drawn outside the frame.
    ax.set_ylim(0, float(totals.max()) * BREAKDOWN_VALUE_HEADROOM or 1.0)
    # No title: it restated the caption, and at this type size it cost a line the bars
    # could use.
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # Below the axes rather than "upper right": at 20pt the two entries reach a third of
    # the way across and covered both the tallest bar and its value label.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.48),
              fontsize=BREAKDOWN_LEGEND_FONTSIZE, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_refusal_by_variant(variant_df, output_path):
    models = sorted(variant_df["model"].unique())
    variants = ["base", "question", "negated"]
    variant_colors = {"base": "#4C72B0", "question": "#DD8452", "negated": "#55A868"}

    n_models = len(models)
    n_variants = len(variants)
    group_width = 0.8
    bar_width = group_width / n_variants

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_models + 2), 5))
    positions = np.arange(n_models)

    for i, variant in enumerate(variants):
        subset = variant_df[variant_df["variant"] == variant].set_index("model").reindex(models)
        rates = subset["refusal_rate"].fillna(0.0).to_numpy()
        offset = (i - n_variants / 2 + 0.5) * bar_width
        bars = ax.bar(
            positions + offset, rates, bar_width * 0.9,
            color=variant_colors[variant], edgecolor="black", linewidth=0.4,
            label=variant,
        )
        for bar, rate in zip(bars, rates):
            if rate > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, rate + 0.001,
                    f"{rate:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(positions, models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Refusal rate")
    ax.set_title("Refusal rate per statement variant per model (all languages pooled)", fontsize=11, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Variant", loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                        help="Cosine-similarity cutoff for marking a free-text response as a semantic refusal.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model directory under data/<dataset>_results/.")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: data/<dataset>_results/plots/refusals).")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    else:
        skipped = [d.name for d in model_dirs if any(s in d.name for s in SKIP_MODEL_SUBSTRINGS)]
        model_dirs = [d for d in model_dirs if not any(s in d.name for s in SKIP_MODEL_SUBSTRINGS)]
        if skipped:
            print(f"Skipping models: {skipped}")
    if not model_dirs:
        raise SystemExit("No model directories found.")

    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")

    all_records = []
    for model_dir in model_dirs:
        print(f"\nCollecting from {model_dir.name}...")
        model_records = collect_model_records(model_dir)
        for record in model_records:
            record["model"] = model_dir.name
        print(f"  {len(model_records)} text fields ({sum(r['hard_refusal'] for r in model_records)} hard refusals)")
        all_records.extend(model_records)

    if not all_records:
        raise SystemExit("No likert or speeches CSVs matched the expected filename patterns.")

    records_df = pd.DataFrame.from_records(all_records)

    print(f"\nLoading embedding model: {args.embed_model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model = AutoModel.from_pretrained(args.embed_model).to(args.device)
    embed_model.eval()

    print(f"Scoring {records_df['text'].notna().sum()} free-text responses for semantic refusals "
          f"(threshold = {args.threshold})...")
    annotated_df = annotate_semantic_refusals(records_df, tokenizer, embed_model, args.device, args.threshold)

    rates_df = aggregate_refusal_rates(annotated_df)
    overall_df = overall_rate_per_model_language(rates_df)
    variant_df = overall_rate_per_model_variant(rates_df)
    per_question_model, question_totals = aggregate_per_question(annotated_df)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots" / "refusals"
    output_dir.mkdir(parents=True, exist_ok=True)

    rates_csv = output_dir / "refusal_rates.csv"
    rates_df.to_csv(rates_csv, index=False)
    print(f"\n  Saved {rates_csv}")

    overall_csv = output_dir / "refusal_rates_overall.csv"
    overall_df.to_csv(overall_csv, index=False)
    print(f"  Saved {overall_csv}")

    examples_path = output_dir / "semantic_refusal_examples.csv"
    examples = (
        annotated_df[annotated_df["semantic_refusal"]]
        .sort_values("max_template_similarity", ascending=False)
        .groupby(["model", "language"])
        .head(3)[["model", "language", "variant", "source", "max_template_similarity", "text"]]
    )
    examples.to_csv(examples_path, index=False)
    print(f"  Saved {examples_path}")

    variant_csv = output_dir / "refusal_rates_by_variant.csv"
    variant_df.to_csv(variant_csv, index=False)
    print(f"  Saved {variant_csv}")

    per_question_csv = output_dir / "refusal_rates_per_question.csv"
    per_question_model.to_csv(per_question_csv, index=False)
    print(f"  Saved {per_question_csv}")

    per_question_overall_csv = output_dir / "refusal_rates_per_question_overall.csv"
    question_totals.to_csv(per_question_overall_csv, index=False)
    print(f"  Saved {per_question_overall_csv}")

    plot_heatmap(overall_df, output_dir / "refusal_heatmap.png")
    plot_per_model_languages(overall_df, output_dir / "refusal_per_model.png")
    plot_hard_vs_semantic_per_model(overall_df, output_dir / "refusal_breakdown.png")
    plot_refusal_by_variant(variant_df, output_dir / "refusal_by_variant.png")
    plot_refusal_per_question(per_question_model, question_totals, output_dir / "refusal_per_question.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
