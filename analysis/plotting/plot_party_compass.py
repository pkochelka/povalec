import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import EP_GROUP_BY_PARTY, FALLBACK_PARTY_COLOR, PARTY_DISPLAY_ORDER
from analysis.plotting.plot_political_bias import (
    LEGEND_MARKERSIZE,
    MAIN_LEGEND_LOC,
    covariance_ellipse,
    load_questionnaire,
    setup_compass,
)

# The compasses plot every group separately, so the collapsed ECR+ID bucket does not
# apply here, and unmapped parties fall into "Other".
EP_GROUP_ORDER = [g for g in PARTY_DISPLAY_ORDER if g != "ECR+ID"] + ["Other"]
EP_GROUP_COLOR = {
    # A second palette, deliberately: only GUE/NGL's magenta (which separates it from
    # S&D) is shared with utils.PARTY_COLORS; the other six hexes are this figure
    # family's own and differ from the bar/box figures'. Unifying them would change how
    # every compass looks, so it is a standing choice rather than an oversight.
    "GUE/NGL": "#8E1B6B", "S&D": "#e8112d", "Greens/EFA": "#3eb049", "ALDE": "#f6b40e",
    "PPE": "#3a86c8", "ECR": "#0a4ea3", "ID": "#1b1f3b", "Other": FALLBACK_PARTY_COLOR,
}


def load_parties(dataset):
    path = Path("data") / f"{dataset}_data" / f"{dataset}_parties.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def party_stance_vector(party):
    responses = sorted(party["responses"], key=lambda r: r["statement_idx"])
    return np.array([
        r["normalized_answer"] if isinstance(r.get("normalized_answer"), (int, float)) else np.nan
        for r in responses
    ], dtype=float)


def build_party_positions(parties, questionnaire, dims):
    signs = questionnaire[dims].to_numpy(dtype=float)
    records = []
    for party in parties:
        stance = party_stance_vector(party)
        length = min(len(stance), len(signs))
        stance, party_signs = stance[:length], signs[:length]
        record = {"ep_group": EP_GROUP_BY_PARTY.get(party["short_name"], "Other")}
        for column, dimension in enumerate(dims):
            sign = party_signs[:, column]
            projected = np.where(sign != 0, stance * sign, np.nan)
            record[dimension] = np.nanmean(projected) if np.any(~np.isnan(projected)) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def aggregate_ep_groups(positions, dims):
    grouped = positions.groupby("ep_group")
    means = grouped[dims].mean()
    means["n_parties"] = grouped.size()
    return means.reset_index()


def plot_ep_group_compass(positions, group_means, x_dim, y_dim, one_sided, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_compass(ax, x_dim, y_dim, one_sided)

    present = [group for group in EP_GROUP_ORDER if group in set(group_means["ep_group"])]
    for group in present:
        color = EP_GROUP_COLOR[group]
        members = positions[positions["ep_group"] == group][[x_dim, y_dim]].dropna().to_numpy()
        if len(members) >= 3:
            ax.add_patch(covariance_ellipse(members, 2, edgecolor=color, facecolor=color,
                                            alpha=0.10, lw=1.2))
        row = group_means[group_means["ep_group"] == group].iloc[0]
        gx, gy = row[x_dim], row[y_dim]
        if np.isnan(gx) or np.isnan(gy):
            continue
        ax.scatter(gx, gy, s=420, color=color, edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(f"{group} (n={int(row['n_parties'])})", (gx, gy), xytext=(8, 8),
                    textcoords="offset points", fontsize=11, color=color, fontweight="bold",
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=3.5, foreground="white")], zorder=9)

    handles = [Line2D([], [], color=EP_GROUP_COLOR[g], marker="o", ls="", label=g,
                      markersize=LEGEND_MARKERSIZE) for g in present]
    ax.legend(handles=handles, loc=MAIN_LEGEND_LOC, fontsize=8, title="EP group", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--x-dim", default="Left-Right")
    parser.add_argument("--y-dim", default="Europe")
    args = parser.parse_args()

    questionnaire, dims, one_sided = load_questionnaire(args.dataset)
    for dimension in (args.x_dim, args.y_dim):
        if dimension not in dims:
            raise SystemExit(f"Dimension {dimension!r} not in questionnaire. Available: {dims}")

    parties = load_parties(args.dataset)
    positions = build_party_positions(parties, questionnaire, dims)
    group_means = aggregate_ep_groups(positions, dims)
    n_unmapped = int(group_means.loc[group_means["ep_group"] == "Other", "n_parties"].sum())
    group_means = group_means[group_means["ep_group"] != "Other"].reset_index(drop=True)
    note = f" ({n_unmapped} unmapped parties excluded)" if n_unmapped else ""
    print(f"Aggregated {len(positions)} parties into {len(group_means)} EP groups{note}.")

    out_dir = Path("data") / f"{args.dataset}_results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_ep_group_compass(
        positions, group_means, args.x_dim, args.y_dim, one_sided,
        out_dir / f"ep_group_compass_{args.x_dim}_{args.y_dim}.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()