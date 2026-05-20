import argparse
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import LIKERT_MAX, LIKERT_MIDPOINT, flip_likert, likert_to_stance


RED_WHITE_GREEN = mcolors.LinearSegmentedColormap.from_list(
    "red_white_green", ["red", "white", "green"]
)


def load_variant_answers(csv_path: Path, variant_suffix: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path, sep=";")
    col_pattern = re.compile(
        r"^choice_([a-z]+)" + re.escape(variant_suffix) + r"_v(\d+)$"
    )
    lang_to_version_cols: dict[str, dict[int, str]] = {}
    for col in df.columns:
        match = col_pattern.match(col)
        if match:
            lang, version = match.group(1), int(match.group(2))
            lang_to_version_cols.setdefault(lang, {})[version] = col

    answers_by_lang: dict[str, np.ndarray] = {}
    for lang, version_cols in lang_to_version_cols.items():
        sorted_cols = [version_cols[v] for v in sorted(version_cols)]
        answers = (
            df[sorted_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(LIKERT_MIDPOINT)
            .to_numpy(dtype=float)
        )
        if variant_suffix == "_negated":
            answers = flip_likert(answers)
        answers_by_lang[lang] = answers

    return answers_by_lang


def find_variant_csv(model_dir: Path, variant_suffix: str) -> Path | None:
    stem_pattern = re.compile(r"^[a-z]{2}(?:,[a-z]{2})*" + re.escape(variant_suffix) + r"$")
    for p in model_dir.glob("*.csv"):
        if stem_pattern.match(p.stem):
            return p
    return None


def fraction_of_most_frequent_answer(responses: np.ndarray) -> float:
    valid = responses[~np.isnan(responses)].astype(int)
    if len(valid) == 0:
        return np.nan
    counts = np.bincount(valid, minlength=LIKERT_MAX + 1)
    return counts.max() / len(valid)


def mean_pairwise_pearson_r(variant_arrays: list[np.ndarray], question_index: int) -> float:
    variant_vectors = [arr[question_index] for arr in variant_arrays]
    pair_correlations = []
    for vec_a, vec_b in combinations(variant_vectors, 2):
        mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
        if mask.sum() < 2:
            continue
        a, b = vec_a[mask], vec_b[mask]
        a_const, b_const = a.std() == 0, b.std() == 0
        if a_const and b_const:
            pair_correlations.append(1.0 if a[0] == b[0] else -1.0)
        elif a_const or b_const:
            pair_correlations.append(0.0)
        else:
            r, _ = pearsonr(a, b)
            pair_correlations.append(r)
    return np.mean(pair_correlations) if pair_correlations else np.nan


def compute_consistency_matrices(
    answers_per_variant: list[dict[str, np.ndarray]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_languages = sorted(set().union(*[d.keys() for d in answers_per_variant]))
    n_questions = next(
        arr.shape[0]
        for variant in answers_per_variant
        for arr in variant.values()
    )
    n_langs = len(all_languages)

    fraction_matrix    = np.full((n_questions, n_langs), np.nan)
    mean_matrix        = np.full((n_questions, n_langs), np.nan)
    variance_matrix    = np.full((n_questions, n_langs), np.nan)
    correlation_matrix = np.full((n_questions, n_langs), np.nan)

    for j, lang in enumerate(all_languages):
        available_variants = [v[lang] for v in answers_per_variant if lang in v]
        all_responses = np.concatenate(available_variants, axis=1)
        all_responses_normalized = likert_to_stance(all_responses)

        for i in range(n_questions):
            raw_row = all_responses[i]
            norm_row = all_responses_normalized[i]
            valid_norm = norm_row[~np.isnan(norm_row)]
            if len(valid_norm) == 0:
                continue
            fraction_matrix[i, j]    = fraction_of_most_frequent_answer(raw_row)
            mean_matrix[i, j]        = valid_norm.mean()
            variance_matrix[i, j]    = valid_norm.var()
            if len(available_variants) >= 2:
                correlation_matrix[i, j] = mean_pairwise_pearson_r(available_variants, i)

    return all_languages, fraction_matrix, mean_matrix, variance_matrix, correlation_matrix


def save_heatmap(
    matrix: np.ndarray,
    languages: list[str],
    title: str,
    colorbar_label: str,
    out_path: Path,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> None:
    n_q, n_l = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_l * 0.55), max(6, n_q * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.02, pad=0.02)
    ax.set_xticks(range(n_l))
    ax.set_xticklabels(languages, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels([f"Q{i+1}" for i in range(n_q)], fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    if annotate:
        for i in range(n_q):
            for j in range(n_l):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def save_boxplot_by_language(
    matrix: np.ndarray,
    languages: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    data_per_language = [matrix[:, j][~np.isnan(matrix[:, j])] for j in range(len(languages))]
    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 0.55), 5))
    ax.boxplot(data_per_language, tick_labels=languages, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"),
               medianprops=dict(color="navy", linewidth=1.5))
    ax.set_xticklabels(languages, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def save_boxplot_by_question(
    matrix: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    n_q = matrix.shape[0]
    data_per_question = [matrix[i, :][~np.isnan(matrix[i, :])] for i in range(n_q)]
    question_labels = [f"Q{i+1}" for i in range(n_q)]
    fig, ax = plt.subplots(figsize=(max(10, n_q * 0.55), 5))
    ax.boxplot(data_per_question, tick_labels=question_labels, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"),
               medianprops=dict(color="navy", linewidth=1.5))
    ax.set_xticklabels(question_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")

    variant_specs = [("", "base"), ("_negated", "negated"), ("_question", "question")]
    answers_per_variant: list[dict[str, np.ndarray]] = []

    for suffix, label in variant_specs:
        csv_path = find_variant_csv(model_dir, suffix)
        if csv_path is None:
            print(f"  No {label} CSV found, skipping that variant.")
            continue
        answers_per_variant.append(load_variant_answers(csv_path, suffix))

    if not answers_per_variant:
        print("  No data loaded, skipping.")
        return

    languages, fraction_mat, mean_mat, variance_mat, correlation_mat = \
        compute_consistency_matrices(answers_per_variant)

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    save_heatmap(
        fraction_mat, languages,
        title=f"{model_dir.name} – Fraction of most-frequent answer",
        colorbar_label="Fraction of most-frequent answer",
        out_path=out_dir / "language_fraction_consistency.png",
        cmap="RdYlGn", vmin=0.2, vmax=1.0,
    )

    save_heatmap(
        mean_mat, languages,
        title=f"{model_dir.name} – Mean answer (−1 disagree → +1 agree)",
        colorbar_label="Mean answer",
        out_path=out_dir / "language_mean_consistency.png",
        cmap=RED_WHITE_GREEN, vmin=-1.0, vmax=1.0,
    )

    save_heatmap(
        variance_mat, languages,
        title=f"{model_dir.name} – Variance of answers (−1 to +1 scale)",
        colorbar_label="Variance",
        out_path=out_dir / "language_variance_consistency.png",
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
    )

    save_heatmap(
        correlation_mat, languages,
        title=f"{model_dir.name} – Mean pairwise Pearson r between variant answer vectors",
        colorbar_label="Mean pairwise Pearson r",
        out_path=out_dir / "language_correlation_consistency.png",
        cmap="RdYlGn", vmin=-1.0, vmax=1.0,
    )

    save_boxplot_by_language(
        fraction_mat, languages,
        title=f"{model_dir.name} – Fraction of most-frequent answer per language (distribution over questions)",
        ylabel="Fraction of most-frequent answer",
        out_path=out_dir / "boxplot_fraction_by_language.png",
        ymin=0.2, ymax=1.0,
    )

    save_boxplot_by_question(
        fraction_mat,
        title=f"{model_dir.name} – Fraction of most-frequent answer per question (distribution over languages)",
        ylabel="Fraction of most-frequent answer",
        out_path=out_dir / "boxplot_fraction_by_question.png",
        ymin=0.2, ymax=1.0,
    )

    save_boxplot_by_language(
        correlation_mat, languages,
        title=f"{model_dir.name} – Mean pairwise Pearson r per language (distribution over questions)",
        ylabel="Mean pairwise Pearson r",
        out_path=out_dir / "boxplot_correlation_by_language.png",
        ymin=-1.0, ymax=1.0,
    )

    save_boxplot_by_question(
        correlation_mat,
        title=f"{model_dir.name} – Mean pairwise Pearson r per question (distribution over languages)",
        ylabel="Mean pairwise Pearson r",
        out_path=out_dir / "boxplot_correlation_by_question.png",
        ymin=-1.0, ymax=1.0,
    )

    save_boxplot_by_question(
        mean_mat,
        title=f"{model_dir.name} – Mean answer per question (distribution over languages)",
        ylabel="Mean answer (−1 disagree → +1 agree)",
        out_path=out_dir / "boxplot_mean_by_question.png",
        ymin=-1.0, ymax=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-question/language consistency heatmaps for all models."
    )
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