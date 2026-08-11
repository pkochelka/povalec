#!/usr/bin/env python3
"""Speech counts per EP group in the cleaned EuroParl splits, BEFORE ECR+ID collapse.

Reads data/EuroParl Custom/cleaned/{train,dev,test}.parquet and plots one bar per
party, the height being the sum over the three splits. Only the party column is
read, so the 2.2 GB train file costs a fraction of a second.

This is the *pre-collapse* picture on purpose: it is the motivation for
build_collapsed_splits.py. ID is the reason that script exists -- it has roughly
a quarter of ECR's speeches and an order of magnitude fewer than PPE, so a
7-party classifier is learning its rarest class from very little. ECR and ID are
therefore hatched and their merged total is drawn as a dashed outline, showing
where the collapsed track's ECR+ID lands relative to the other six.

Note the splits are not comparable in kind: dev and test are class-balanced by
construction (7,142 rows per party each), so essentially all of the variation
below comes from train's natural, heavily imbalanced priors.

    ./venv/Scripts/python.exe analysis/plotting/plot_speech_counts.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from utils import FALLBACK_PARTY_COLOR, PARTY_COLORS, PARTY_DISPLAY_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "cleaned"
DEFAULT_OUTPUT = PROJECT_ROOT / "plots" / "speech_counts_by_party.png"

PARTY_COLUMN = "EU Party"
SPLITS = ["train", "dev", "test"]
# The two the collapsed track merges; hatched so the reader can see at a glance
# which bars the ECR+ID marker is built from.
MERGE_PARTIES = ("ECR", "ID")
MERGED_LABEL = "ECR+ID"

GRID_STYLE = dict(axis="y", linestyle=":", alpha=0.35, zorder=0)


def party_color(party):
    return PARTY_COLORS.get(party, FALLBACK_PARTY_COLOR)


def split_counts(input_dir):
    """{split: {party: count}} -- only the party column is read off disk."""
    counts = {}
    for split in SPLITS:
        path = input_dir / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing split: {path}")
        column = pq.read_table(path, columns=[PARTY_COLUMN]).column(0).to_pylist()
        counts[split] = {party: column.count(party) for party in sorted(set(column))}
        print(f"{split}: {len(column):,} speeches, {len(counts[split])} parties")
    return counts


def total_counts(counts):
    totals = {}
    for split in SPLITS:
        for party, count in counts[split].items():
            totals[party] = totals.get(party, 0) + count
    return totals


def order_parties(totals, order):
    """Bar order: by count (clearest for a magnitude plot) or the shared
    left-to-right ideological order the other party figures use."""
    if order == "count":
        return sorted(totals, key=lambda party: -totals[party])
    known = [party for party in PARTY_DISPLAY_ORDER if party in totals]
    return known + sorted(set(totals) - set(known))


def format_count(count):
    return f"{count / 1000:.0f}k" if count >= 10_000 else f"{count:,}"


def plot_counts(totals, parties, output_path, log_scale):
    grand = sum(totals.values())
    values = [totals[party] for party in parties]
    positions = np.arange(len(parties))

    merged = [party for party in MERGE_PARTIES if party in totals]
    merged_total = sum(totals[party] for party in merged) if len(
        merged) == len(MERGE_PARTIES) else None
    # A merged bar whose top sits within roughly a label's height of the ECR+ID
    # outline gets its label INSIDE the bar instead; an outside one would land on
    # the dashed line. ECR clears the test, ID (far shorter than the outline)
    # does not and keeps the normal outside label -- its bar is too short to hold
    # two lines of text.
    label_height = 0.07 * max(values) * 1.18
    inside = set() if log_scale or merged_total is None else {
        party for party in merged if merged_total - totals[party] < label_height}

    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(parties)), 5.6))
    ax.grid(**GRID_STYLE)
    for position, party, value in zip(positions, parties, values):
        ax.bar(position, value, width=0.72, color=party_color(party),
               edgecolor="black", linewidth=0.7, zorder=3,
               hatch="///" if party in MERGE_PARTIES else None)
        indented = party in inside
        ax.annotate(f"{format_count(value)}\n{value / grand:.1%}",
                    (position, value), textcoords="offset points",
                    xytext=(0, -4 if indented else 4), ha="center",
                    va="top" if indented else "bottom", fontsize=9, zorder=5,
                    color="white" if indented else "black",
                    fontweight="bold" if indented else "normal")

    # Where the collapsed track's single ECR+ID class would sit: an outline
    # SPANNING both merged bars rather than a bar of its own, so it cannot be
    # mistaken for a party that exists in these files and so it does not sit on
    # top of either bar's own count label.
    if merged_total is not None:
        slots = [positions[parties.index(party)] for party in merged]
        left, right = min(slots) - 0.36, max(slots) + 0.36
        ax.add_patch(mpatches.Rectangle(
            (left, 0), right - left, merged_total, facecolor="none",
            edgecolor=party_color(MERGED_LABEL), linewidth=1.6,
            linestyle="--", zorder=4))
        ax.annotate(f"{MERGED_LABEL} = {format_count(merged_total)}",
                    ((left + right) / 2, merged_total), textcoords="offset points",
                    xytext=(0, 5), ha="center", va="bottom", fontsize=9,
                    color=party_color(MERGED_LABEL), fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(parties, rotation=20, ha="right")
    ax.set_ylabel("Speeches (train + dev + test)")
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, max(values) * 1.18)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///",
                       label="merged into ECR+ID by build_collapsed_splits.py"),
        mpatches.Patch(facecolor="none", edgecolor=party_color(MERGED_LABEL),
                       linestyle="--", linewidth=1.6, label="ECR+ID combined total"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved {output_path}")


def print_table(totals, counts, parties):
    grand = sum(totals.values())
    width = max(len(party) for party in parties)
    header = f"{'Party':<{width}}  " + "".join(f"{split:>12}" for split in SPLITS)
    print("\n" + header + f"{'total':>12}{'share':>9}")
    print("-" * len(header + f"{'total':>12}{'share':>9}"))
    for party in parties:
        cells = "".join(f"{counts[split].get(party, 0):>12,}" for split in SPLITS)
        print(f"{party:<{width}}  {cells}{totals[party]:>12,}"
              f"{totals[party] / grand:>9.1%}")
    merged = [party for party in MERGE_PARTIES if party in totals]
    if len(merged) == len(MERGE_PARTIES):
        merged_total = sum(totals[party] for party in merged)
        cells = "".join(f"{sum(counts[split].get(p, 0) for p in merged):>12,}"
                        for split in SPLITS)
        print(f"{MERGED_LABEL:<{width}}  {cells}{merged_total:>12,}"
              f"{merged_total / grand:>9.1%}   (post-collapse)")
    print(f"\n{'total':<{width}}  " + "".join(f"{sum(counts[s].values()):>12,}"
                                              for s in SPLITS) + f"{grand:>12,}")
    print(f"Imbalance ratio, largest : smallest = "
          f"{max(totals.values()) / min(totals.values()):.1f}x")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--order", default="count", choices=["count", "ideological"],
                        help="Bar order: descending count, or the shared "
                             "left-to-right party order the other figures use.")
    parser.add_argument("--log", action="store_true",
                        help="Log y-axis; the smallest party is ~25x below the "
                             "largest, so the linear default squashes it.")
    return parser.parse_args()


def main():
    args = parse_args()
    counts = split_counts(args.input_dir)
    totals = total_counts(counts)
    parties = order_parties(totals, args.order)
    print_table(totals, counts, parties)
    plot_counts(totals, parties, args.output, args.log)


if __name__ == "__main__":
    main()
