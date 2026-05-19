import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_vaa_per_language import (
    find_vaa_csvs,
    load_variant_dfs,
    party_order_and_colors,
)


VARIANT_ORDER = ["base", "question", "negated"]


def ordered_variants(dfs: dict[str, pd.DataFrame]) -> list[str]:
    return [v for v in VARIANT_ORDER if v in dfs] + [v for v in dfs if v not in VARIANT_ORDER]


def plot_party_ranking_per_variant(
    dfs: dict[str, pd.DataFrame],
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    variants = ordered_variants(dfs)
    n_variants = len(variants)

    fig, axes = plt.subplots(
        n_variants,
        1,
        figsize=(max(8, len(parties) * 0.7), 3.2 * n_variants),
        sharex=True,
        sharey=True,
    )
    if n_variants == 1:
        axes = [axes]

    x = np.arange(len(parties))
    bar_colors = [colors[p] for p in parties]

    for ax, variant in zip(axes, variants):
        means = dfs[variant].groupby("ep_group")["mean_agreement"].mean().reindex(parties)
        ax.bar(x, means.values, color=bar_colors, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x, means.values):
            if pd.notna(v):
                ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(f"{variant}\nmean agreement", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(parties, rotation=30, ha="right", fontsize=9)
    axes[0].set_title(title, fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def plot_per_language_bars_per_variant(
    dfs: dict[str, pd.DataFrame],
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    variants = ordered_variants(dfs)
    n_variants = len(variants)

    languages = sorted(set().union(*(set(df["language"].unique()) for df in dfs.values())))
    n_parties = len(parties)
    n_langs = len(languages)
    group_width = 0.85
    bar_width = group_width / max(n_parties, 1)
    x = np.arange(n_langs)

    fig, axes = plt.subplots(
        n_variants,
        1,
        figsize=(max(10, n_langs * 0.9), 3.6 * n_variants),
        sharex=True,
        sharey=True,
    )
    if n_variants == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        pivot = (
            dfs[variant]
            .pivot_table(index="language", columns="ep_group", values="mean_agreement")
            .reindex(index=languages, columns=parties)
        )
        for i, party in enumerate(parties):
            offsets = x - group_width / 2 + bar_width * (i + 0.5)
            ax.bar(
                offsets,
                pivot[party].values,
                width=bar_width,
                color=colors[party],
                edgecolor="black",
                linewidth=0.3,
                label=party if ax is axes[0] else None,
            )
        ax.set_ylabel(f"{variant}\nmean agreement", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(languages, fontsize=9)
    axes[0].set_title(title, fontsize=11, pad=8)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 + 0.18 / n_variants),
        ncol=min(n_parties, 6),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def plot_party_sensitivity(
    dfs: dict[str, pd.DataFrame],
    parties: list[str],
    colors: dict[str, tuple],
    title: str,
    out_path: Path,
) -> None:
    variants = ordered_variants(dfs)
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.5), 5.5))

    for party in parties:
        ys = []
        for variant in variants:
            means = dfs[variant].groupby("ep_group")["mean_agreement"].mean()
            ys.append(means.get(party, np.nan))
        ax.plot(
            x,
            ys,
            marker="o",
            color=colors[party],
            linewidth=1.6,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=party,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylabel("Mean agreement (averaged over languages)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.3, len(variants) - 0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(len(parties), 6),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(out_path.parents[3])}")


def process_model(model_dir: Path) -> None:
    print(f"\nProcessing: {model_dir.name}")
    if not find_vaa_csvs(model_dir):
        print("  No vaa*.csv files found, skipping.")
        return
    dfs = load_variant_dfs(model_dir)
    if not dfs:
        return

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    combined = pd.concat(dfs.values(), ignore_index=True)
    parties, colors = party_order_and_colors(combined)

    base_title = f"{model_dir.name} – VAA across variants"

    plot_party_ranking_per_variant(
        dfs,
        parties,
        colors,
        f"{base_title}: party ranking per variant",
        out_dir / "vaa_variants_party_ranking.png",
    )
    plot_per_language_bars_per_variant(
        dfs,
        parties,
        colors,
        f"{base_title}: per-language party agreement",
        out_dir / "vaa_variants_per_language_bars.png",
    )
    plot_party_sensitivity(
        dfs,
        parties,
        colors,
        f"{base_title}: party sensitivity to variant",
        out_dir / "vaa_variants_party_sensitivity.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VAA party agreement with all variants (base/question/negated) grouped "
            "into a single comparison figure per model. Party color map is shared with "
            "plot_vaa_per_language.py outputs."
        )
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
