import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


VAA_CSV_PATTERN = re.compile(r"^vaa(?P<variant>|_negated|_question)_(?P<langs>[a-z,]+)\.csv$")


def find_vaa_csvs(model_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in model_dir.glob("vaa*.csv"):
        match = VAA_CSV_PATTERN.match(path.name)
        if not match:
            continue
        variant_key = match.group("variant") or "base"
        if variant_key == "_negated":
            variant_key = "negated"
        elif variant_key == "_question":
            variant_key = "question"
        found[variant_key] = path
    return found


def per_language_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("language")["mean_agreement"]
        .agg(["max", "mean", "min"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return stats


def plot_per_language(
    stats: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    languages = stats["language"].tolist()
    x = range(len(languages))

    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 0.45), 5.5))

    ax.vlines(x, stats["min"], stats["max"], color="lightgray", linewidth=2, zorder=1)
    ax.scatter(x, stats["max"], color="seagreen", s=45, zorder=3, label="Max (best-aligned group)")
    ax.scatter(x, stats["mean"], color="steelblue", s=45, zorder=3, label="Mean over groups")
    ax.scatter(x, stats["min"], color="indianred", s=45, zorder=3, label="Min (worst-aligned group)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(languages, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean agreement")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")
    vaa_csvs = find_vaa_csvs(model_dir)
    if not vaa_csvs:
        print("  No vaa*.csv files found, skipping.")
        return

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    for variant_label, csv_path in sorted(vaa_csvs.items()):
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  {csv_path.name} is empty, skipping.")
            continue
        stats = per_language_stats(df)
        title = f"{model_dir.name} – VAA agreement per language ({variant_label})"
        out_path = out_dir / f"vaa_per_language_{variant_label}.png"
        plot_per_language(stats, title, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-language max/mean/min VAA agreement across EP groups, for every model and variant."
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
