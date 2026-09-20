#!/usr/bin/env python3
"""Inter-annotator agreement on the hand-labelled stance sample, and optionally
the stance cross-encoder's agreement with each annotator.

The dataset (data/stance_annotation_iaa/stance_annotations_160.csv) carries two
independent Likert stance labels per item -- `annotator_a` and `annotator_b`,
1 = totally agree with the statement ... 5 = totally disagree. Both label the
same 160 (statement, answer) pairs, so agreement is the plain two-rater case.

Because the scale is ordinal and most disagreements are one step wide, the
headline number is Krippendorff's ordinal alpha (quadratic-weighted Cohen's
kappa is reported alongside); unweighted kappa is kept only for reference, since
it counts a 1-vs-2 miss the same as a 1-vs-5 miss.

The IAA half needs nothing but the CSV. With --crossencoder, the checkpoint is
scored against both annotators: its continuous stance in [-1, 1] is cut onto the
5 Likert bins with the sampler's edges, exactly as eval_crossencoder_human does.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.iaa_metrics import agreement_metrics, confusion, show

DATASET = os.path.join(PROJECT_ROOT, "data", "stance_annotation_iaa",
                       "stance_annotations_160.csv")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
LIKERT = [1, 2, 3, 4, 5]
DIRECTIONS = ["agree", "neutral", "disagree"]


def direction(choice):
    """1-2 -> agree, 3 -> neutral, 4-5 -> disagree (the +/-0.2 neutral band of
    validate_stance_judge, read on the Likert side)."""
    choice = np.asarray(choice)
    return np.where(choice < 3, "agree", np.where(choice > 3, "disagree", "neutral"))


def pair_metrics(a, b, name, n_boot=0, seed=0):
    """The shared ordinal metrics, plus the stance-specific direction collapse."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    metrics = agreement_metrics(a, b, name, LIKERT, n_boot=n_boot, seed=seed)
    metrics.update({
        "dir_acc": np.mean(direction(a) == direction(b)),
        "dir_kappa": cohen_kappa_score(direction(a), direction(b), labels=DIRECTIONS),
        "flips": np.mean(((a < 3) & (b > 3)) | ((a > 3) & (b < 3))),
    })
    # keep the bootstrap CI last in the printed table
    for key in ("alpha_lo", "alpha_hi"):
        if key in metrics:
            metrics[key] = metrics.pop(key)
    return metrics


def choice_confusion(a, b, row_name, col_name):
    confusion(a, b, row_name, col_name, LIKERT,
              note=", 1 = totally agree ... 5 = totally disagree")


def breakdown(df, a_col, b_col, name):
    rows = []
    for key in ("language", "answer_model"):
        for value, group in df.groupby(key):
            rows.append(pair_metrics(group[a_col], group[b_col], f"{key}={value}"))
    show(rows, f"{name} by slice")


def run_crossencoder(df, model_dir, batch_size, device):
    """Score the checkpoint on the same items and compare it to both annotators."""
    import torch
    from analysis.stance_crossencoder import load_crossencoder, predict_stances
    from analysis.eval_crossencoder_human import bin_stance
    from utils import stance_to_likert

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, max_len = load_crossencoder(model_dir, device)
    stance = predict_stances(df["statement"].tolist(), df["answer"].tolist(),
                             tokenizer, model, device, max_len, batch_size)
    df = df.assign(
        ce_stance=stance,
        ce_choice=stance_to_likert(bin_stance(np.asarray(stance))).round().astype(int),
    )

    print(f"\nCheckpoint: {model_dir}")
    show([pair_metrics(df.annotator_a, df.annotator_b, "annotator_a vs annotator_b"),
          pair_metrics(df.ce_choice, df.annotator_a, "cross-encoder vs annotator_a"),
          pair_metrics(df.ce_choice, df.annotator_b, "cross-encoder vs annotator_b")],
         "Cross-encoder agreement")

    # correlations on the continuous score, against each annotator and their mean
    mean_human = (df.annotator_a + df.annotator_b) / 2
    print("\nContinuous stance vs Likert labels (negated: stance +1 == choice 1)")
    for label, human in (("annotator_a", df.annotator_a), ("annotator_b", df.annotator_b),
                         ("mean(a, b)", mean_human)):
        print(f"  {label:<12} pearson={pearsonr(-df.ce_stance, human)[0]:.3f}  "
              f"spearman={spearmanr(-df.ce_stance, human)[0]:.3f}")

    choice_confusion(df.ce_choice, df.annotator_a, "Cross-encoder", "Annotator_a")
    choice_confusion(df.ce_choice, df.annotator_b, "Cross-encoder", "Annotator_b")
    breakdown(df, "ce_choice", "annotator_a", "Cross-encoder vs annotator_a")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--crossencoder", nargs="?", const=DEFAULT_MODEL_DIR, default=None,
                        metavar="MODEL_DIR",
                        help="Also score this stance cross-encoder checkpoint against both "
                             f"annotators (default checkpoint: {DEFAULT_MODEL_DIR}).")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="Bootstrap resamples for the alpha CI on the headline pair (0 = off).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Write per-item predictions/disagreements here.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataset, encoding="utf-8-sig", dtype={"item_id": str})
    print(f"Loaded {len(df)} items from {args.dataset}")

    show([pair_metrics(df.annotator_a, df.annotator_b, "annotator_a vs annotator_b",
                       n_boot=args.n_boot)],
         "Inter-annotator agreement (all items)")
    choice_confusion(df.annotator_a, df.annotator_b, "Annotator_a", "Annotator_b")

    print("\nLabel distribution (1 = totally agree ... 5 = totally disagree)")
    print(pd.DataFrame({
        "annotator_a": df.annotator_a.value_counts().reindex(LIKERT, fill_value=0),
        "annotator_b": df.annotator_b.value_counts().reindex(LIKERT, fill_value=0),
    }).to_string())

    breakdown(df, "annotator_a", "annotator_b", "Inter-annotator agreement")

    if args.crossencoder:
        df = run_crossencoder(df, args.crossencoder, args.batch_size, args.device)

    if args.out:
        columns = [c for c in ("item_id", "language", "answer_model", "annotator_a",
                               "annotator_b", "ce_choice", "ce_stance", "statement", "answer")
                   if c in df.columns]
        df.assign(gap=(df.annotator_a - df.annotator_b).abs()) \
          .sort_values("gap", ascending=False)[columns + ["gap"]] \
          .to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nPer-item table -> {args.out}")


if __name__ == "__main__":
    main()
