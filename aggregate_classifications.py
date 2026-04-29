#!/usr/bin/env python3
"""
Aggregate predicted political group labels (English only) across prompt variants.

Reads a classified CSV (output of classify_speeches.py) and computes per-row:
  - predicted_label_en_majority   : majority-vote label across v0..vN
  - predicted_label_en_agreement  : fraction of variants voting for the majority
  - label_en_frac_<group>         : fraction of variants predicting each group
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

LABELS = ["PPE", "S&D", "ALDE", "Greens/EFA", "ID"]


def load_dataframe(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def _safe(label):
    return label.replace("/", "_").replace("&", "and")


def aggregate_labels(df, variant=""):
    lang_variant = f"en{variant}"
    vcols = sorted(c for c in df.columns if c.startswith(f"predicted_label_{lang_variant}_v"))
    if not vcols:
        print(f"[{lang_variant}] no predicted_label columns found, skipping.")
        return df

    label_matrix = df[vcols].values  # (n_rows, n_variants)

    for label in LABELS:
        df[f"label_{lang_variant}_frac_{_safe(label)}"] = (
            (label_matrix == label).sum(axis=1) / len(vcols)
        )

    majorities, agreements = [], []
    for row in label_matrix:
        valid = [x for x in row if pd.notna(x) and str(x).strip()]
        if not valid:
            majorities.append(None)
            agreements.append(None)
            continue
        top_label, top_count = Counter(valid).most_common(1)[0]
        majorities.append(top_label)
        agreements.append(top_count / len(valid))

    df[f"predicted_label_{lang_variant}_majority"] = majorities
    df[f"predicted_label_{lang_variant}_agreement"] = agreements

    print(f"[{lang_variant}] aggregated {len(vcols)} variants over {len(df)} rows")
    return df


def print_summary(df, variant=""):
    lang_variant = f"en{variant}"
    majority_col = f"predicted_label_{lang_variant}_majority"
    agreement_col = f"predicted_label_{lang_variant}_agreement"
    if majority_col not in df.columns:
        return
    print("=" * 60)
    print(f"Label aggregation summary  ({lang_variant})")
    print("=" * 60)
    print("Majority-vote distribution:")
    print(df[majority_col].value_counts().to_string())
    print(f"\nMean agreement:            {df[agreement_col].mean():.3f}")
    print(f"Full-agreement questions:  {(df[agreement_col] == 1.0).sum()} / {len(df)}")


def _q_labels(df, statement_col):
    if statement_col and statement_col in df.columns:
        return [s[:45] + ("…" if len(s) > 45 else "") for s in df[statement_col]]
    return [f"Q{i+1}" for i in range(len(df))]


def plot_label_heatmap(df, variant="", outfile=None, statement_col=None):
    lang_variant = f"en{variant}"
    frac_cols = [f"label_{lang_variant}_frac_{_safe(l)}" for l in LABELS]
    if any(c not in df.columns for c in frac_cols):
        return

    mat = df[frac_cols].values.T  # (n_labels, n_questions)
    q_labels = _q_labels(df, statement_col)

    fig, ax = plt.subplots(
        figsize=(max(12, 0.55 * len(q_labels)), 4), constrained_layout=True
    )
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Predicted label fractions per question  ({lang_variant})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of variants")

    thresh = 0.65
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outfile}")
    else:
        plt.show()


def plot_majority_bar(df, variant="", outfile=None):
    lang_variant = f"en{variant}"
    col = f"predicted_label_{lang_variant}_majority"
    if col not in df.columns:
        return

    counts = df[col].value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    counts.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Majority-vote label counts  ({lang_variant})")
    ax.set_ylabel("Questions")
    ax.set_xlabel("Predicted political group")
    ax.tick_params(axis="x", rotation=30)

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outfile}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate predicted political group labels (English only)."
    )
    ap.add_argument("--model", default="gpt-oss-120b", choices=["gpt-oss-120b", "qwen3.5-122b"])
    ap.add_argument("--results_dir", default=None,
                    help="Results directory (default: ./data/euandi_2019_results/{model})")
    ap.add_argument("--input", default=None,
                    help="Classified CSV path (default: {results_dir}/speeches_en,de,el,es,fr,it{variant}_classified.csv)")
    ap.add_argument("--variant", default="", choices=["", "_question", "_negated"])
    ap.add_argument("--statement_col", default=None,
                    help="Column holding statement text (used for axis labels)")
    ap.add_argument("--heatmap_out", default=None)
    ap.add_argument("--bar_out", default=None)
    args = ap.parse_args()

    results_dir = args.results_dir or f"./data/euandi_2019_results/{args.model}"
    os.makedirs(results_dir, exist_ok=True)

    input_path = (
        args.input
        or f"{results_dir}/speeches_en,de,el,es,fr,it{args.variant}_classified.csv"
    )
    stem, ext = os.path.splitext(input_path)
    output_path = f"{stem}_aggregated{ext}"
    heatmap_out = args.heatmap_out or f"{results_dir}/label_heatmap_en{args.variant}.png"
    bar_out = args.bar_out or f"{results_dir}/label_bar_en{args.variant}.png"

    print(f"Loading {input_path}")
    df = load_dataframe(input_path)
    df = aggregate_labels(df, variant=args.variant)
    print_summary(df, variant=args.variant)

    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nAggregated output saved to {output_path}")

    plot_label_heatmap(df, variant=args.variant, outfile=heatmap_out,
                       statement_col=args.statement_col)
    plot_majority_bar(df, variant=args.variant, outfile=bar_out)


if __name__ == "__main__":
    main()
