import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from plot_vaa_per_language import (
    party_order_and_colors,
    plot_party_ranking,
    plot_per_language_bars,
    plot_per_language_scatter,
)
from plot_vaa_variants_grouped import (
    plot_party_ranking_per_variant,
    plot_per_language_bars_per_variant,
    plot_party_sensitivity,
)

VAA_SPEECHES_PATTERN = re.compile(
    r"^vaa_speeches(?P<variant>|_negated|_question)_(?P<langs>[a-z,]+)\.csv$"
)
VARIANT_LABELS = {"": "base", "_negated": "negated", "_question": "question"}


def find_speech_vaa_csvs(model_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in model_dir.glob("vaa_speeches*.csv"):
        match = VAA_SPEECHES_PATTERN.match(path.name)
        if match:
            found[VARIANT_LABELS[match.group("variant")]] = path
    return found


def load_variant_dfs(model_dir: Path) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for variant_label, csv_path in sorted(find_speech_vaa_csvs(model_dir).items()):
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  {csv_path.name} is empty, skipping.")
            continue
        dfs[variant_label] = df
    return dfs


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")
    dfs = load_variant_dfs(model_dir)
    if not dfs:
        print("  No vaa_speeches*.csv files found, skipping.")
        return

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    combined = pd.concat(dfs.values(), ignore_index=True)
    parties, colors = party_order_and_colors(combined)

    for variant_label, df in dfs.items():
        plot_party_ranking(
            df, parties, colors,
            out_dir / f"vaa_speeches_party_ranking_{variant_label}.png",
        )
        plot_per_language_bars(
            df, parties, colors,
            out_dir / f"vaa_speeches_per_language_bars_{variant_label}.png",
        )
        plot_per_language_scatter(
            df, parties, colors,
            out_dir / f"vaa_speeches_per_language_scatter_{variant_label}.png",
        )

    plot_party_ranking_per_variant(
        dfs, parties, colors, out_dir / "vaa_speeches_variants_party_ranking.png")
    plot_per_language_bars_per_variant(
        dfs, parties, colors, out_dir / "vaa_speeches_variants_per_language_bars.png")
    plot_party_sensitivity(
        dfs, parties, colors, out_dir / "vaa_speeches_variants_party_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot speech-based VAA party agreement (vaa_speeches*.csv) for every model."
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
