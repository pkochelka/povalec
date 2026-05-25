import argparse
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

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS_DIR.parent))
sys.path.insert(0, str(_ANALYSIS_DIR))

from utils import likert_to_stance
from evaluate_euandi import detect_likert_languages, likert_means_per_statement

VARIANTS = [("", "base"), ("_negated", "negated"), ("_question", "question")]
VARIANT_LABELS = dict(VARIANTS)
DESIRED_ALPHA = 0.7
PAIRWISE_R_LABEL = "Mean pairwise Pearson r"
RED_WHITE_GREEN = mcolors.LinearSegmentedColormap.from_list(
    "red_white_green", ["red", "white", "green"]
)


def save_figure(fig: plt.Figure, out_path: Path, **savefig_kwargs) -> None:
    fig.savefig(out_path, dpi=150, **savefig_kwargs)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def find_scored_csv(model_dir: Path, variant_suffix: str) -> Path | None:
    pattern = re.compile(
        rf"^speeches_[a-z]{{2}}(?:,[a-z]{{2}})*{re.escape(variant_suffix)}_scored$"
    )
    return next((p for p in model_dir.glob("speeches_*_scored.csv") if pattern.match(p.stem)), None)


def find_likert_csv(model_dir: Path) -> Path | None:
    pattern = re.compile(r"^[a-z]{2}(?:,[a-z]{2})*$")
    return next((p for p in model_dir.glob("*.csv") if pattern.match(p.stem)), None)


def load_speech_stance_arrays(csv_path: Path, variant_suffix: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    col_pattern = re.compile(rf"^stance_([a-z]+){re.escape(variant_suffix)}_v(\d+)$")
    flip_sign = -1.0 if variant_suffix == "_negated" else 1.0

    cols_by_lang: dict[str, dict[int, str]] = {}
    for col in df.columns:
        if match := col_pattern.match(col):
            cols_by_lang.setdefault(match.group(1), {})[int(match.group(2))] = col

    arrays: dict[str, np.ndarray] = {}
    for lang, version_cols in cols_by_lang.items():
        ordered = [version_cols[v] for v in sorted(version_cols)]
        stance = df[ordered].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        arrays[lang] = flip_sign * stance
    return arrays


def load_speech_mean_stance(csv_path: Path, variant_suffix: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    pattern = re.compile(rf"^stance_([a-z]+){re.escape(variant_suffix)}_mean$")
    return pd.DataFrame({
        match.group(1): pd.to_numeric(df[col], errors="coerce")
        for col in df.columns
        if (match := pattern.match(col))
    })


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return pearsonr(a, b)[0]


def mean_pairwise_correlation(variant_arrays: list[np.ndarray], question_index: int) -> float:
    vectors = [arr[question_index] for arr in variant_arrays]
    correlations = []
    for a, b in combinations(vectors, 2):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            continue
        a, b = a[mask], b[mask]
        a_const, b_const = a.std() == 0, b.std() == 0
        if a_const and b_const:
            correlations.append(1.0 if a[0] == b[0] else -1.0)
        elif a_const or b_const:
            correlations.append(0.0)
        else:
            correlations.append(pearsonr(a, b)[0])
    return float(np.mean(correlations)) if correlations else np.nan


def compute_speech_matrices(
    arrays_per_variant: list[dict[str, np.ndarray]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    languages = sorted(set().union(*(d.keys() for d in arrays_per_variant)))
    n_questions = next(arr.shape[0] for variant in arrays_per_variant for arr in variant.values())

    mean_matrix = np.full((n_questions, len(languages)), np.nan)
    variance_matrix = np.full((n_questions, len(languages)), np.nan)
    correlation_matrix = np.full((n_questions, len(languages)), np.nan)

    for col, lang in enumerate(languages):
        variants = [v[lang] for v in arrays_per_variant if lang in v]
        responses = np.concatenate(variants, axis=1)
        for row in range(n_questions):
            valid = responses[row][~np.isnan(responses[row])]
            if len(valid) == 0:
                continue
            mean_matrix[row, col] = valid.mean()
            variance_matrix[row, col] = valid.var()
            if len(variants) >= 2:
                correlation_matrix[row, col] = mean_pairwise_correlation(variants, row)

    return languages, mean_matrix, variance_matrix, correlation_matrix


def save_heatmap(
    matrix: np.ndarray, languages: list[str], title: str, colorbar_label: str,
    out_path: Path, cmap, vmin: float, vmax: float,
) -> None:
    n_questions, n_langs = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_langs * 0.55), max(6, n_questions * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.02, pad=0.02)
    ax.set_xticks(range(n_langs), languages, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_questions), [f"Q{i + 1}" for i in range(n_questions)], fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    for row in range(n_questions):
        for col in range(n_langs):
            if not np.isnan(matrix[row, col]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=5.5)
    save_figure(fig, out_path)


def save_boxplot(
    data: list[np.ndarray], labels: list[str], title: str, ylabel: str,
    out_path: Path, ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5))
    ax.boxplot(data, tick_labels=labels, patch_artist=True,
               boxprops={"facecolor": "lightsteelblue"},
               medianprops={"color": "navy", "linewidth": 1.5})
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11, pad=8)
    save_figure(fig, out_path)


def columns_without_nan(matrix: np.ndarray) -> list[np.ndarray]:
    return [col[~np.isnan(col)] for col in matrix.T]


def rows_without_nan(matrix: np.ndarray) -> list[np.ndarray]:
    return [row[~np.isnan(row)] for row in matrix]


def plot_speech_consistency(model_dir: Path, out_dir: Path) -> None:
    arrays_per_variant: list[dict[str, np.ndarray]] = []
    for suffix, label in VARIANTS:
        csv_path = find_scored_csv(model_dir, suffix)
        if csv_path is None:
            print(f"  No scored {label} speeches found, skipping that variant.")
            continue
        arrays_per_variant.append(load_speech_stance_arrays(csv_path, suffix))

    if not arrays_per_variant:
        print("  No scored speeches loaded, skipping consistency plots.")
        return

    languages, mean_mat, variance_mat, correlation_mat = compute_speech_matrices(arrays_per_variant)
    name = model_dir.name
    question_labels = [f"Q{i + 1}" for i in range(mean_mat.shape[0])]

    save_heatmap(mean_mat, languages,
                 f"{name} – Speech mean stance (−1 disagree → +1 agree)", "Mean stance",
                 out_dir / "speech_language_mean_consistency.png",
                 RED_WHITE_GREEN, vmin=-1.0, vmax=1.0)
    save_heatmap(variance_mat, languages,
                 f"{name} – Speech stance variance (−1 to +1 scale)", "Variance",
                 out_dir / "speech_language_variance_consistency.png",
                 "YlOrRd", vmin=0.0, vmax=1.0)
    save_heatmap(correlation_mat, languages,
                 f"{name} – {PAIRWISE_R_LABEL} between speech-variant stance vectors",
                 PAIRWISE_R_LABEL,
                 out_dir / "speech_language_correlation_consistency.png",
                 "RdYlGn", vmin=-1.0, vmax=1.0)

    save_boxplot(columns_without_nan(mean_mat), languages,
                 f"{name} – Speech mean stance per language (distribution over questions)",
                 "Mean stance (−1 disagree → +1 agree)",
                 out_dir / "speech_boxplot_mean_by_language.png", ylim=(-1.0, 1.0))
    save_boxplot(rows_without_nan(mean_mat), question_labels,
                 f"{name} – Speech mean stance per question (distribution over languages)",
                 "Mean stance (−1 disagree → +1 agree)",
                 out_dir / "speech_boxplot_mean_by_question.png", ylim=(-1.0, 1.0))
    save_boxplot(columns_without_nan(correlation_mat), languages,
                 f"{name} – Speech variant correlation per language (distribution over questions)",
                 PAIRWISE_R_LABEL,
                 out_dir / "speech_boxplot_correlation_by_language.png", ylim=(-1.0, 1.0))
    save_boxplot(rows_without_nan(correlation_mat), question_labels,
                 f"{name} – Speech variant correlation per question (distribution over languages)",
                 PAIRWISE_R_LABEL,
                 out_dir / "speech_boxplot_correlation_by_question.png", ylim=(-1.0, 1.0))


CRONBACH_SPEECH_PATTERN = re.compile(
    r"^cronbach_speeches(?P<variant>|_negated|_question)_(?P<langs>[a-z,]+)\.csv$"
)


def load_speech_alphas(model_dir: Path) -> dict[str, pd.Series]:
    alphas: dict[str, pd.Series] = {}
    for path in model_dir.glob("cronbach_speeches*.csv"):
        match = CRONBACH_SPEECH_PATTERN.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  {path.name} is empty, skipping.")
            continue
        alphas[VARIANT_LABELS[match.group("variant")]] = df.set_index("language")["cronbach_alpha"]
    return dict(sorted(alphas.items()))


def plot_cronbach_per_language(alphas: dict[str, pd.Series], title: str, out_path: Path) -> None:
    languages = sorted(set().union(*(series.index for series in alphas.values())))
    pivot = pd.DataFrame({label: series.reindex(languages) for label, series in alphas.items()})

    n_variants = len(pivot.columns)
    group_width = 0.85
    bar_width = group_width / max(n_variants, 1)
    x = np.arange(len(languages))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 0.9), 5.5))
    for i, label in enumerate(pivot.columns):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        ax.bar(offsets, pivot[label].values, width=bar_width, color=cmap(i % cmap.N),
               edgecolor="black", linewidth=0.3, label=label)

    ax.axhline(DESIRED_ALPHA, color="red", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(len(languages) - 0.5, DESIRED_ALPHA, f" desired ≥ {DESIRED_ALPHA}",
            color="red", va="bottom", ha="right", fontsize=9)
    ax.set_xticks(x, languages, fontsize=9)
    ax.set_ylabel("Cronbach's α (across axes)")
    ax.set_ylim(0.0, max(1.0, float(pivot.max().max()) + 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_variants, 6),
              fontsize=8, framealpha=0.9)
    save_figure(fig, out_path, bbox_inches="tight")


def plot_speech_cronbach(model_dir: Path, out_dir: Path) -> None:
    alphas = load_speech_alphas(model_dir)
    if not alphas:
        print("  No cronbach_speeches*.csv files found, skipping cronbach plot.")
        return
    plot_cronbach_per_language(
        alphas,
        f"{model_dir.name} – Speech Cronbach's α per language",
        out_dir / "cronbach_speeches_per_language.png",
    )


def load_likert_stance(model_dir: Path) -> pd.DataFrame | None:
    csv_path = find_likert_csv(model_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    languages = detect_likert_languages(df)
    if not languages:
        return None
    mean_likert = likert_means_per_statement(df, languages)
    return pd.DataFrame({lang: likert_to_stance(mean_likert[lang]) for lang in languages})


def plot_correlation_bar(per_language_r: pd.Series, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(per_language_r) * 0.5), 5))
    colors = ["seagreen" if r >= 0 else "indianred" for r in per_language_r.values]
    x = np.arange(len(per_language_r))
    ax.bar(x, per_language_r.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pearson r (speech stance vs Likert stance)")
    ax.set_xticks(x, per_language_r.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save_figure(fig, out_path)


def plot_correlation_scatter(
    speech_stance: pd.DataFrame, likert_stance: pd.DataFrame,
    languages: list[str], pooled_r: float, title: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("tab20")
    for i, lang in enumerate(languages):
        ax.scatter(likert_stance[lang], speech_stance[lang], s=12, alpha=0.6,
                   color=cmap(i % cmap.N), label=lang, edgecolors="none")
    ax.plot([-1, 1], [-1, 1], color="black", linestyle="--", linewidth=1.0, zorder=5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Likert stance (−1 disagree → +1 agree)")
    ax.set_ylabel("Speech stance (NLI, −1 → +1)")
    ax.set_title(f"{title}\npooled Pearson r = {pooled_r:.3f}", fontsize=11, pad=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=10, fontsize=6, framealpha=0.9)
    save_figure(fig, out_path, bbox_inches="tight")


def plot_langxlang_heatmap(
    speech_stance: pd.DataFrame, likert_stance: pd.DataFrame,
    languages: list[str], title: str, out_path: Path,
) -> None:
    n = len(languages)
    matrix = np.array([
        [safe_pearson(
            speech_stance[speech_lang].to_numpy(dtype=float),
            likert_stance[likert_lang].to_numpy(dtype=float),
        ) for likert_lang in languages]
        for speech_lang in languages
    ])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.02)
    ax.set_xticks(range(n), languages, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), languages, fontsize=8)
    ax.set_xlabel("Likert language")
    ax.set_ylabel("Speech language")
    ax.set_title(title, fontsize=11, pad=8)
    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(value) > 0.6 else "black")
    save_figure(fig, out_path)


def plot_speech_vs_likert(model_dir: Path, out_dir: Path) -> None:
    scored_csv = find_scored_csv(model_dir, "")
    if scored_csv is None:
        print("  No base scored speeches found, skipping correlation plots.")
        return
    speech_stance = load_speech_mean_stance(scored_csv, "")
    likert_stance = load_likert_stance(model_dir)
    if likert_stance is None or speech_stance.empty:
        print("  Missing speech or Likert stance, skipping correlation plots.")
        return

    languages = sorted(set(speech_stance.columns) & set(likert_stance.columns))
    if not languages:
        print("  No overlapping languages between speeches and Likert, skipping.")
        return

    n = min(len(speech_stance), len(likert_stance))
    speech_stance = speech_stance.iloc[:n].reset_index(drop=True)
    likert_stance = likert_stance.iloc[:n].reset_index(drop=True)

    per_language_r = pd.Series({
        lang: safe_pearson(
            speech_stance[lang].to_numpy(dtype=float),
            likert_stance[lang].to_numpy(dtype=float),
        ) for lang in languages
    })
    name = model_dir.name
    plot_correlation_bar(
        per_language_r,
        f"{name} – Speech stance vs Likert stance correlation per language",
        out_dir / "speech_vs_likert_correlation_bar.png",
    )

    pooled_speech = np.concatenate([speech_stance[lang].to_numpy(dtype=float) for lang in languages])
    pooled_likert = np.concatenate([likert_stance[lang].to_numpy(dtype=float) for lang in languages])
    plot_correlation_scatter(
        speech_stance, likert_stance, languages, safe_pearson(pooled_speech, pooled_likert),
        f"{name} – Speech stance vs Likert stance",
        out_dir / "speech_vs_likert_scatter.png",
    )
    plot_langxlang_heatmap(
        speech_stance, likert_stance, languages,
        f"{name} – Speech vs Likert stance correlation (language × language)",
        out_dir / "speech_vs_likert_langxlang.png",
    )


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")
    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_speech_consistency(model_dir, out_dir)
    plot_speech_cronbach(model_dir, out_dir)
    plot_speech_vs_likert(model_dir, out_dir)


def main() -> None:
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
