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
LIKERT_VALUES = [1, 2, 3, 4, 5]


def save_figure(fig, out_path, **savefig_kwargs):
    fig.savefig(out_path, dpi=150, **savefig_kwargs)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


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


def choice_likerts(csv_path, suffix, flip):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", low_memory=False)
    result = {}
    for language, columns in columns_by_language(df, "choice", suffix).items():
        values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
        values = values[np.isin(values, LIKERT_VALUES)]
        result[language] = flip_likert(values) if flip else values
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


def plot_distributions(likerts_by_language, title, out_path):
    languages = sorted(likerts_by_language)
    variant_order = list(VARIANT_LABELS.values())
    proportions = {}
    y_max = 0.0
    for language in languages:
        per_variant = likerts_by_language[language]
        total = sum(len(values) for values in per_variant.values())
        shares = {}
        for label in variant_order:
            values = per_variant.get(label, np.array([]))
            counts = np.array([(values == likert).sum() for likert in LIKERT_VALUES], dtype=float)
            shares[label] = counts / total if total else counts
        proportions[language] = (shares, int(total))
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
    fig.suptitle(title, fontsize=12)
    fig.supxlabel("Likert (1 = agree → 5 = disagree)", fontsize=9)
    fig.supylabel("Proportion", fontsize=9)
    save_figure(fig, out_path)


def process_model(model_dir):
    print(f"\nProcessing: {model_dir.name}")
    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    choices = collect_likerts(model_dir, "choices")
    if choices:
        plot_distributions(
            choices, f"{model_dir.name} – Choice Likert distribution per language",
            out_dir / "likert_distribution_choices_per_language.png",
        )
    else:
        print("  No choice columns found, skipping choices plot.")

    stance = collect_likerts(model_dir, "stance")
    if stance:
        plot_distributions(
            stance, f"{model_dir.name} – Speech stance (binned) distribution per language",
            out_dir / "likert_distribution_stance_per_language.png",
        )
    else:
        print("  No scored speeches found, skipping stance plot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None, help="Restrict to one model directory.")
    args = parser.parse_args()

    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    for model_dir in model_dirs:
        process_model(model_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
