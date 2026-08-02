import argparse
import re
import sys
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS_DIR.parent))
sys.path.insert(0, str(_ANALYSIS_DIR))

from utils import flip_likert
from sample_speeches_for_labeling import STANCE_BIN_EDGES

VARIANTS = ["", "_question", "_negated"]
VARIANT_LABELS = {"": "base", "_question": "question", "_negated": "negated"}
VARIANT_COLORS = {"base": "#2CA02C", "question": "#FFC107", "negated": "#D62728"}
# On the cross-model panel the colour is spent on the model, so the framing split
# has to be carried by the texture instead.
VARIANT_HATCH = {"base": "", "question": "///", "negated": "xxx"}
LIKERT_VALUES = [1, 2, 3, 4, 5]
LIKERT_XLABEL = "Likert (1 = agree → 5 = disagree)"
KIND_LABEL = {"choices": "Likert choices", "stance": "speech stances (NLI-scored)"}


def save_figure(fig, out_path, **savefig_kwargs):
    fig.savefig(out_path, dpi=300, **savefig_kwargs)
    plt.close(fig)
    try:
        shown = out_path.relative_to(Path.cwd())
    except ValueError:
        shown = out_path
    print(f"  Saved {shown}")


def find_survey_csv(model_dir, suffix):
    pattern = re.compile(rf"^[a-z]{{2}}(?:,[a-z]{{2}})*{re.escape(suffix)}$")
    return next((p for p in model_dir.glob("*.csv") if pattern.match(p.stem)), None)


def find_scored_csv(model_dir, suffix):
    pattern = re.compile(rf"^speeches_[a-z]{{2}}(?:,[a-z]{{2}})*{re.escape(suffix)}_scored$")
    return next((p for p in model_dir.glob("speeches_*_scored.csv") if pattern.match(p.stem)), None)


def columns_by_language(df, kind, suffix):
    pattern = re.compile(rf"^{kind}_([a-z]+){re.escape(suffix)}_v(\d+)$")
    grouped = {}
    for column in df.columns:
        if match := pattern.match(column):
            grouped.setdefault(match.group(1), []).append(column)
    return grouped


def columns_by_question(df, kind, suffix):
    pattern = re.compile(rf"^{kind}_([a-z]+){re.escape(suffix)}_v(\d+)$")
    grouped = {}
    for column in df.columns:
        if match := pattern.match(column):
            grouped.setdefault(int(match.group(2)), []).append(column)
    return grouped


def choice_likerts(csv_path, suffix, flip):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    result = {}
    for language, columns in columns_by_language(df, "choice", suffix).items():
        values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        values = values[np.isin(values, LIKERT_VALUES)]
        result[language] = flip_likert(values) if flip else values
    return result


def choice_likerts_per_question(csv_path, suffix, flip):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    result = {}
    for q_idx, columns in columns_by_question(df, "choice", suffix).items():
        values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        values = values[np.isin(values, LIKERT_VALUES)]
        result[q_idx] = flip_likert(values) if flip else values
    return result


def stance_likerts(csv_path, suffix, flip_sign):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    result = {}
    for language, columns in columns_by_language(df, "stance", suffix).items():
        stance = flip_sign * df[columns].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        bins = pd.cut(stance, bins=STANCE_BIN_EDGES, right=False, labels=False)
        bins = bins[~np.isnan(bins)]
        result[language] = (5 - bins).astype(int)
    return result


def stance_likerts_per_question(csv_path, suffix, flip_sign):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    result = {}
    for q_idx, columns in columns_by_question(df, "stance", suffix).items():
        stance = flip_sign * df[columns].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        bins = pd.cut(stance, bins=STANCE_BIN_EDGES, right=False, labels=False)
        bins = bins[~np.isnan(bins)]
        result[q_idx] = (5 - bins).astype(int)
    return result


def collect_likerts(model_dir, kind):
    by_language = {}
    for suffix in VARIANTS:
        if kind == "choices":
            csv_path = find_survey_csv(model_dir, suffix)
            per_language = choice_likerts(csv_path, suffix, suffix == "_negated") if csv_path else {}
        else:
            csv_path = find_scored_csv(model_dir, suffix)
            per_language = stance_likerts(csv_path, suffix, -1.0 if suffix == "_negated" else 1.0) if csv_path else {}
        for language, values in per_language.items():
            by_language.setdefault(language, {})[VARIANT_LABELS[suffix]] = values
    return by_language


def collect_likerts_per_question(model_dir, kind):
    by_question = {}
    for suffix in VARIANTS:
        if kind == "choices":
            csv_path = find_survey_csv(model_dir, suffix)
            per_question = choice_likerts_per_question(csv_path, suffix, suffix == "_negated") if csv_path else {}
        else:
            csv_path = find_scored_csv(model_dir, suffix)
            per_question = stance_likerts_per_question(csv_path, suffix, -1.0 if suffix == "_negated" else 1.0) if csv_path else {}
        for q_idx, values in per_question.items():
            by_question.setdefault(q_idx, {})[VARIANT_LABELS[suffix]] = values
    return by_question


def pool_over_languages(by_language):
    """{language: {variant: values}} -> {variant: values}, concatenated over languages."""
    pooled = {}
    for per_variant in by_language.values():
        for label, values in per_variant.items():
            pooled.setdefault(label, []).append(values)
    return {label: np.concatenate(chunks) for label, chunks in pooled.items()}


def variant_shares(per_variant, variant_order):
    """Per-Likert share of the model's answers, split by variant.

    Normalised by the total over all variants, exactly like the per-model grids, so
    a model's five stacks sum to 1 and heights compare across models.
    """
    total = sum(len(values) for values in per_variant.values())
    shares = {}
    for label in variant_order:
        values = per_variant.get(label, np.array([]))
        counts = np.array([(values == likert).sum() for likert in LIKERT_VALUES], dtype=float)
        shares[label] = counts / total if total else counts
    return shares, int(total)


def plot_models_distribution(pooled_by_model, kind, out_path):
    """All models on one panel: colour = model, hatch = framing, languages pooled."""
    models = sorted(pooled_by_model)
    variant_order = list(VARIANT_LABELS.values())
    cmap = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")

    x = np.arange(len(LIKERT_VALUES))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(10.0, len(models) * 1.0), 5.5))
    model_handles, y_max = [], 0.0
    for index, model in enumerate(models):
        color = cmap(index % cmap.N)
        shares, total = variant_shares(pooled_by_model[model], variant_order)
        offsets = x - 0.4 + bar_width * (index + 0.5)
        bottom = np.zeros(len(LIKERT_VALUES))
        for label in variant_order:
            ax.bar(offsets, shares[label], width=bar_width * 0.9, bottom=bottom, color=color,
                   hatch=VARIANT_HATCH[label], edgecolor="black", linewidth=0.4)
            bottom += shares[label]
        y_max = max(y_max, bottom.max())
        model_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black",
                                           linewidth=0.4, label=f"{model} (n={total})"))

    ax.set_xticks(x, LIKERT_VALUES, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, y_max * 1.1 or 1.0)
    ax.set_xlabel(LIKERT_XLABEL, fontsize=9)
    ax.set_ylabel("Proportion", fontsize=9)
    ax.set_title(f"{KIND_LABEL[kind]}: all models, languages pooled", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    variant_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", edgecolor="black",
                                     linewidth=0.4, hatch=VARIANT_HATCH[label], label=label)
                       for label in variant_order]
    variant_legend = ax.legend(handles=variant_handles, loc="upper right", fontsize=8,
                               title="Variant", title_fontsize=8, framealpha=0.9)
    ax.add_artist(variant_legend)
    # 13 model entries do not fit beside the bars, so they go under the axes.
    ax.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=min(4, len(models)), fontsize=8, title="Model", title_fontsize=8,
              frameon=False)
    save_figure(fig, out_path, bbox_inches="tight")


def plot_distributions(likerts_by_language, out_path):
    languages = sorted(likerts_by_language)
    variant_order = list(VARIANT_LABELS.values())
    proportions = {}
    y_max = 0.0
    for language in languages:
        shares, total = variant_shares(likerts_by_language[language], variant_order)
        proportions[language] = (shares, total)
        stacked = np.sum([shares[label] for label in variant_order], axis=0)
        y_max = max(y_max, stacked.max() if total else 0.0)

    n_cols = min(6, len(languages))
    n_rows = ceil(len(languages) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.1),
                             squeeze=False, constrained_layout=True)
    for index, language in enumerate(languages):
        ax = axes[index // n_cols][index % n_cols]
        shares, total = proportions[language]
        bottom = np.zeros(len(LIKERT_VALUES))
        for label in variant_order:
            ax.bar(LIKERT_VALUES, shares[label], bottom=bottom, color=VARIANT_COLORS[label],
                   edgecolor="black", linewidth=0.3, label=label)
            bottom += shares[label]
        ax.set_title(f"{language} (n={total})", fontsize=8)
        ax.set_xticks(LIKERT_VALUES)
        ax.set_xticklabels(LIKERT_VALUES, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(0, y_max * 1.1 or 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    for index in range(len(languages), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[label]) for label in variant_order]
    fig.legend(handles, variant_order, loc="upper right", fontsize=8, title="Variant")
    fig.supxlabel(LIKERT_XLABEL, fontsize=9)
    fig.supylabel("Proportion", fontsize=9)
    save_figure(fig, out_path)


def plot_distributions_per_question(likerts_by_question, out_path):
    question_ids = sorted(likerts_by_question)
    variant_order = list(VARIANT_LABELS.values())

    counts_per_question = {}
    y_max = 0.0
    for q_idx in question_ids:
        shares, total = variant_shares(likerts_by_question[q_idx], variant_order)
        counts_per_question[q_idx] = (shares, total)
        stacked = np.sum([shares[label] for label in variant_order], axis=0)
        y_max = max(y_max, stacked.max() if total else 0.0)

    n_cols = min(6, len(question_ids))
    n_rows = ceil(len(question_ids) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.1),
                             squeeze=False, constrained_layout=True)
    for i, q_idx in enumerate(question_ids):
        ax = axes[i // n_cols][i % n_cols]
        shares, total = counts_per_question[q_idx]
        bottom = np.zeros(len(LIKERT_VALUES))
        for label in variant_order:
            ax.bar(LIKERT_VALUES, shares[label], bottom=bottom, color=VARIANT_COLORS[label],
                   edgecolor="black", linewidth=0.3, label=label)
            bottom += shares[label]
        ax.set_title(f"Q{q_idx + 1} (n={total})", fontsize=8)
        ax.set_xticks(LIKERT_VALUES)
        ax.set_xticklabels(LIKERT_VALUES, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(0, y_max * 1.1 or 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i in range(len(question_ids), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[label]) for label in variant_order]
    fig.legend(handles, variant_order, loc="upper right", fontsize=8, title="Variant")
    fig.supxlabel(LIKERT_XLABEL, fontsize=9)
    fig.supylabel("Proportion", fontsize=9)
    save_figure(fig, out_path)


def process_model(model_dir):
    """Writes this model's own grids and returns {kind: {variant: values}} for the
    cross-model panel, so the CSVs are read once."""
    print(f"\nProcessing: {model_dir.name}")
    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    pooled = {}

    choices = collect_likerts(model_dir, "choices")
    if choices:
        pooled["choices"] = pool_over_languages(choices)
        plot_distributions(
            choices,
            out_dir / "likert_distribution_choices_per_language.png",
        )
    else:
        print("  No choice columns found, skipping choices plot.")

    choices_per_question = collect_likerts_per_question(model_dir, "choices")
    if choices_per_question:
        plot_distributions_per_question(
            choices_per_question,
            out_dir / "likert_distribution_choices_per_question.png",
        )

    stance = collect_likerts(model_dir, "stance")
    if stance:
        pooled["stance"] = pool_over_languages(stance)
        plot_distributions(
            stance,
            out_dir / "likert_distribution_stance_per_language.png",
        )
    else:
        print("  No scored speeches found, skipping stance plot.")

    stance_per_question = collect_likerts_per_question(model_dir, "stance")
    if stance_per_question:
        plot_distributions_per_question(
            stance_per_question,
            out_dir / "likert_distribution_stance_per_question.png",
        )
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None, help="Restrict to one model directory.")
    args = parser.parse_args()

    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir()
                        if p.is_dir() and p.name not in ("plots", "tables"))
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    pooled_by_kind = {}
    for model_dir in model_dirs:
        for kind, per_variant in process_model(model_dir).items():
            pooled_by_kind.setdefault(kind, {})[model_dir.name] = per_variant

    # Only worth a panel once there is more than one model on it, so --model still
    # produces just that model's own grids.
    out_dir = results_dir / "plots"
    for kind, pooled_by_model in pooled_by_kind.items():
        if len(pooled_by_model) > 1:
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_models_distribution(pooled_by_model, kind,
                                     out_dir / f"likert_distribution_{kind}_models.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
