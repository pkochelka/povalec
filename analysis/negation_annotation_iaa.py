#!/usr/bin/env python3
"""Inter-annotator agreement on the hand-rated quality of the negated statements.

The dataset (data/negation_annotation_iaa/negation_annotations_150.csv) holds the
30 EU&I statements in five languages, each paired with its automatically negated
variant and rated 1-5 for how well the negation preserves the statement while
reversing its polarity (5 = best). Both annotators rated all 150 items.

The labels are heavily skewed: in practice only 4 and 5 were ever used, and the
great majority are 5. That breaks Cohen's kappa -- with one category this
dominant, chance agreement is estimated as almost the whole of the observed
agreement, so kappa collapses even though the raters almost always match (the
kappa prevalence paradox). Gwet's AC1 is reported alongside precisely because it
does not degenerate this way, and raw percent agreement is the figure to quote
next to it. Read `alpha_ord` and `kappa*` here as lower bounds, not as evidence
that the annotation was unreliable.
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.iaa_metrics import agreement_metrics, breakdown, confusion, show

DATASET = os.path.join(PROJECT_ROOT, "data", "negation_annotation_iaa",
                       "negation_annotations_150.csv")
SCALE = [1, 2, 3, 4, 5]


def threshold_counts(df, thresholds=(3, 4, 5)):
    """How many negations clear each quality bar -- on both annotators' ratings
    (the strict reading) and on at least one (the lenient one)."""
    a, b = df.annotator_a, df.annotator_b
    return pd.DataFrame([{
        "threshold": f">= {t}",
        "both": int(((a >= t) & (b >= t)).sum()),
        "at_least_one": int(((a >= t) | (b >= t)).sum()),
        "annotator_a": int((a >= t).sum()),
        "annotator_b": int((b >= t).sum()),
    } for t in thresholds])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="Bootstrap resamples for the alpha CI on the headline pair (0 = off).")
    parser.add_argument("--out", default=None, help="Write the per-item disagreements here.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset, encoding="utf-8-sig", dtype={"item_id": str})
    print(f"Loaded {len(df)} items from {args.dataset}")

    rows = [agreement_metrics(df.annotator_a, df.annotator_b, "all items", SCALE,
                              n_boot=args.n_boot)]
    # the German items were rated identically by both annotators on every row, which
    # pins any agreement statistic at its ceiling -- so report the rest separately too
    other = df[df.language != "de"]
    rows.append(agreement_metrics(other.annotator_a, other.annotator_b,
                                  "excluding de", SCALE, n_boot=args.n_boot))
    show(rows, "Inter-annotator agreement")

    confusion(df.annotator_a, df.annotator_b, "Annotator_a", "Annotator_b", SCALE,
              note=", 1 = worst negation ... 5 = best")

    print("\nLabel distribution (1 = worst negation ... 5 = best)")
    print(pd.DataFrame({
        "annotator_a": df.annotator_a.value_counts().reindex(SCALE, fill_value=0),
        "annotator_b": df.annotator_b.value_counts().reindex(SCALE, fill_value=0),
    }).to_string())

    print(f"\n=== Items at or above each quality threshold (n={len(df)}) ===")
    print(threshold_counts(df).to_string(index=False))

    print("\nPer language, both annotators at 5 / at least one at 5 / n")
    per_language = df.groupby("language").apply(lambda g: pd.Series({
        "both_5": int(((g.annotator_a >= 5) & (g.annotator_b >= 5)).sum()),
        "either_5": int(((g.annotator_a >= 5) | (g.annotator_b >= 5)).sum()),
        "n": len(g),
    }), include_groups=False)
    print(per_language.to_string())

    breakdown(df, "annotator_a", "annotator_b", ["language"], SCALE,
              "Inter-annotator agreement by language")

    disagreements = df[df.annotator_a != df.annotator_b]
    print(f"\n{len(disagreements)} of {len(df)} items disagree; "
          f"{(df.annotator_a - df.annotator_b).abs().max():.0f} is the widest gap.")
    print(disagreements[["item_id", "statement_idx", "language",
                         "annotator_a", "annotator_b"]].to_string(index=False))

    if args.out:
        disagreements.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nDisagreements -> {args.out}")


if __name__ == "__main__":
    main()
