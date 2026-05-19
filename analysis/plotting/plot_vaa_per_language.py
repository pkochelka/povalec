import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def party_order_and_colors(df: pd.DataFrame) -> tuple[list[str], dict[str, tuple]]:
    overall = (
        df.groupby("ep_group")["mean_agreement"]
        .mean()
        .sort_values(ascending=False)
    )
    parties = overall.index.tolist()
    cmap = plt.get_cmap("tab10" if len(parties) <= 10 else "tab20")
    colors = {p: cmap(i % cmap.N) for i, p in enumerate(parties)}
    return parties, colors


def plot_party_ranking(
    df: pd.DataFrame,
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    means = df.groupby("ep_group")["mean_agreement"].mean().reindex(parties)

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 0.7), 5.5))
    x = np.arange(len(parties))
    bar_colors = [colors[p] for p in parties]
    ax.bar(x, means.values, color=bar_colors, edgecolor="black", linewidth=0.5)

    for xi, v in zip(x, means.values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(parties, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean agreement (averaged over languages)")
    ax.set_ylim(0.0, min(1.0, float(means.max()) + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def plot_per_language_bars(
    df: pd.DataFrame,
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    languages = sorted(df["language"].unique())
    pivot = (
        df.pivot_table(index="language", columns="ep_group", values="mean_agreement")
        .reindex(index=languages, columns=parties)
    )

    n_parties = len(parties)
    n_langs = len(languages)
    group_width = 0.85
    bar_width = group_width / max(n_parties, 1)
    x = np.arange(n_langs)

    fig, ax = plt.subplots(figsize=(max(10, n_langs * 0.9), 5.5))
    for i, party in enumerate(parties):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        ax.bar(
            offsets,
            pivot[party].values,
            width=bar_width,
            color=colors[party],
            edgecolor="black",
            linewidth=0.3,
            label=party,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(languages, fontsize=9)
    ax.set_ylabel("Mean agreement")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_parties, 6),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def plot_per_language_scatter(
    df: pd.DataFrame,
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    languages = sorted(df["language"].unique())
    pivot = (
        df.pivot_table(index="language", columns="ep_group", values="mean_agreement")
        .reindex(index=languages, columns=parties)
    )

    x = np.arange(len(languages))
    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 0.6), 5.5))

    lo = pivot.min(axis=1).values
    hi = pivot.max(axis=1).values
    ax.vlines(x, lo, hi, color="lightgray", linewidth=1.2, zorder=1)

    for party in parties:
        ax.scatter(
            x,
            pivot[party].values,
            color=colors[party],
            s=55,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
            label=party,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(languages, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean agreement")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(parties), 6),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def load_variant_dfs(model_dir: Path) -> dict[str, pd.DataFrame]:
    # The VAA CSVs (incl. vaa_negated_*) are produced by evaluate_euandi.py,
    # which already applies utils.likert.flip_likert to the negated variant's
    # answers before computing stance/agreement — so mean_agreement is
    # comparable across variants without any further flipping here.
    vaa_csvs = find_vaa_csvs(model_dir)
    dfs: dict[str, pd.DataFrame] = {}
    for variant_label, csv_path in sorted(vaa_csvs.items()):
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
        print("  No vaa*.csv files found, skipping.")
        return

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    combined = pd.concat(dfs.values(), ignore_index=True)
    parties, colors = party_order_and_colors(combined)

    for variant_label, df in dfs.items():
        base_title = f"{model_dir.name} – VAA ({variant_label})"

        plot_party_ranking(
            df,
            parties,
            colors,
            f"{base_title}: party ranking (avg over languages)",
            out_dir / f"vaa_party_ranking_{variant_label}.png",
        )
        plot_per_language_bars(
            df,
            parties,
            colors,
            f"{base_title}: per-language party agreement (bars)",
            out_dir / f"vaa_per_language_bars_{variant_label}.png",
        )
        plot_per_language_scatter(
            df,
            parties,
            colors,
            f"{base_title}: per-language party agreement (scatter)",
            out_dir / f"vaa_per_language_scatter_{variant_label}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot VAA party agreement: overall ranking (avg over languages) and per-language breakdowns, for every model and variant."
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
