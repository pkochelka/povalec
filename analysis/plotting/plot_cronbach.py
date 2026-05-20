import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CRONBACH_CSV_PATTERN = re.compile(r"^cronbach(?P<variant>|_negated|_question)_(?P<langs>[a-z,]+)\.csv$")
DESIRED_THRESHOLD = 0.7
VARIANT_LABELS = {"": "base", "_negated": "negated", "_question": "question"}


def find_cronbach_csvs(model_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in model_dir.glob("cronbach*.csv"):
        match = CRONBACH_CSV_PATTERN.match(path.name)
        if match:
            found[VARIANT_LABELS[match.group("variant")]] = path
    return found


def load_variant_alphas(model_dir: Path) -> dict[str, pd.Series]:
    alphas: dict[str, pd.Series] = {}
    for variant_label, csv_path in sorted(find_cronbach_csvs(model_dir).items()):
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  {csv_path.name} is empty, skipping.")
            continue
        suffix = "" if variant_label == "base" else f"_{variant_label}"
        df["language"] = df["language"].str.removesuffix(suffix)
        alphas[variant_label] = df.set_index("language")["cronbach_alpha"]
    return alphas


def plot_cronbach_per_language(alphas: dict[str, pd.Series], title: str, out_path: Path) -> None:
    languages = sorted(set().union(*[s.index for s in alphas.values()]))
    pivot = pd.DataFrame({label: series.reindex(languages) for label, series in alphas.items()})

    n_variants = len(pivot.columns)
    group_width = 0.85
    bar_width = group_width / max(n_variants, 1)
    x = np.arange(len(languages))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 0.9), 5.5))
    for i, variant_label in enumerate(pivot.columns):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        ax.bar(
            offsets,
            pivot[variant_label].values,
            width=bar_width,
            color=cmap(i % cmap.N),
            edgecolor="black",
            linewidth=0.3,
            label=variant_label,
        )

    ax.axhline(DESIRED_THRESHOLD, color="red", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(
        len(languages) - 0.5, DESIRED_THRESHOLD, f" desired ≥ {DESIRED_THRESHOLD}",
        color="red", va="bottom", ha="right", fontsize=9,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(languages, fontsize=9)
    ax.set_ylabel("Cronbach's α (across axes)")
    ax.set_ylim(0.0, max(1.0, float(pivot.max().max()) + 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_variants, 6),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")
    alphas = load_variant_alphas(model_dir)
    if not alphas:
        print("  No cronbach*.csv files found, skipping.")
        return

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_cronbach_per_language(
        alphas,
        f"{model_dir.name} – Cronbach's α per language",
        out_dir / "cronbach_per_language.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Cronbach's alpha per language for every model, highlighting the desired reliability threshold."
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
