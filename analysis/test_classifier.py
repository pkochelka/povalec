#!/usr/bin/env python3
"""Evaluate the trained mmBERT EU-party classifier on a uniform eval split.

Defaults to the production ECR+ID-collapsed checkpoint and its own collapsed split
track. Loads the saved model + manifest, runs argmax(logits) inference on
<data_dir>/<split>.parquet (the same inference rule classify_speeches.py uses),
and reports:
  * the confusion matrix (counts and row-normalised recall),
  * the true vs. predicted class histogram — the view that tells whether a class
    (e.g. PPE) is over-predicted where the ground truth is uniform, i.e. residual
    model bias rather than input distribution shift,
  * per-class and macro F1.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import (
    PARTY_COLUMN,
    classification_report_text,
    fit_uniform_bias,
    load_split,
    per_language_f1_report,
)

# The model and the split track have to match: a 6-class collapsed checkpoint fed the
# 7-party splits would silently drop every ECR and ID row via load_split(keep_labels=).
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmBERT-base-balanced-collapsed")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "collapsed")


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, manifest


@torch.no_grad()
def predict_logits(texts, tokenizer, model, max_len, device, batch_size):
    logits = []
    for start in tqdm(range(0, len(texts), batch_size), desc="classifying test"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len,
        ).to(device)
        logits.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(logits, axis=0)


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    counts = pd.DataFrame(cm, index=[f"t:{l}" for l in labels], columns=[f"p:{l}" for l in labels])
    print("\n=== CONFUSION MATRIX (rows = true, cols = predicted) ===")
    print(counts.to_string())

    row_sums = cm.sum(axis=1, keepdims=True)
    recall = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    recall_df = pd.DataFrame(
        (recall * 100).round(1), index=[f"t:{l}" for l in labels], columns=[f"p:{l}" for l in labels],
    )
    print("\n=== ROW-NORMALISED (% of each true class, i.e. recall) ===")
    print(recall_df.to_string())


def plot_confusion_matrix(y_true, y_pred, labels, output_path, subtitle=""):
    """Row-normalised confusion matrix as a heatmap: cell (i, j) is the share of true
    class i predicted as j, so the diagonal is recall and every row sums to 100%.

    Recall is a magnitude, so it gets a single-hue sequential ramp light->dark rather
    than a rainbow -- the only thing a reader has to decode is "darker = more mass",
    and the off-diagonal darkness then shows which pairs the model confuses. Counts
    ride along under each percentage so a cell can be checked against the support."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    row_sums = cm.sum(axis=1, keepdims=True)
    recall = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(1.35 * len(labels) + 3.4, 1.2 * len(labels) + 3.0))
    image = ax.imshow(recall, cmap="Blues", vmin=0.0, vmax=1.0)

    positions = np.arange(len(labels))
    ax.set_xticks(positions, labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(positions, labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("True", fontsize=13, labelpad=10)
    title = "Confusion matrix (row-normalised)"
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=14, pad=14)

    for i in positions:
        for j in positions:
            # Ink, not the series colour: white only where the fill is dark enough to
            # need it, so the numbers stay legible across the whole ramp.
            color = "#FFFFFF" if recall[i, j] > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{100 * recall[i, j]:.1f}%", ha="center", va="center",
                    fontsize=11, color=color)
            ax.text(j, i + 0.28, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=8, color=color, alpha=0.75)

    # A 2px surface gap between cells, and the diagonal ringed so recall reads at a glance.
    ax.set_xticks(positions - 0.5, minor=True)
    ax.set_yticks(positions - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in positions:
        # Above the grid: the white cell gaps are drawn over the patches otherwise, and
        # only the corner cell keeps a visible ring.
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1A1A1A", linewidth=1.6, zorder=5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Share of true class (%)", fontsize=11)
    bar.set_ticks(np.linspace(0, 1, 6))
    bar.ax.set_yticklabels([f"{100 * t:.0f}" for t in np.linspace(0, 1, 6)], fontsize=10)
    bar.outline.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved {output_path}")


def print_class_histogram(y_true, y_pred, labels):
    num_labels = len(labels)
    true_counts = np.bincount(y_true, minlength=num_labels)
    pred_counts = np.bincount(y_pred, minlength=num_labels)
    total = true_counts.sum()
    print("\n=== TRUE vs. PREDICTED CLASS HISTOGRAM ===")
    print(f"(test truth is uniform; a large pred-true gap = the model leans toward that class)\n")
    header = f"{'party':<14}{'true_n':>8}{'pred_n':>8}{'pred-true':>11}{'pred/true':>11}{'pred_%':>9}"
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        ratio = pred_counts[i] / true_counts[i] if true_counts[i] else float("nan")
        print(f"{label:<14}{true_counts[i]:>8d}{pred_counts[i]:>8d}"
              f"{pred_counts[i] - true_counts[i]:>+11d}{ratio:>11.3f}{100 * pred_counts[i] / total:>8.1f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit a per-class logit bias on --calibrate_split to flatten the "
                             "predicted marginal toward uniform, then apply it to --split.")
    parser.add_argument("--calibrate_split", default="dev")
    parser.add_argument("--plot", default=None, nargs="?", const="",
                        help="Save the row-normalised confusion matrix as a PNG. With no "
                             "value, writes <model_dir>/plots/confusion_matrix_<split>.png.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"--- Device: {args.device} | model: {args.model_dir} ---")
    tokenizer, model, manifest = load_model(args.model_dir, args.device)

    label2id = manifest["label2id"]
    labels = sorted(label2id, key=label2id.get)
    max_len = manifest["max_len"]

    def logits_for(split):
        df = load_split(split, args.data_dir, keep_labels=set(label2id))
        df = df[df[PARTY_COLUMN].isin(label2id)].reset_index(drop=True)
        y = df[PARTY_COLUMN].map(label2id).to_numpy()
        langs = df["language"].to_numpy()
        print(f"Loaded {len(df)} {split} rows")
        logits = predict_logits(df["text"].tolist(), tokenizer, model, max_len, args.device, args.batch_size)
        return logits, y, langs

    print(f"{len(labels)} classes: {labels}")
    logits, y_true, languages = logits_for(args.split)

    bias = np.zeros(len(labels))
    if args.calibrate:
        dev_logits, _, _ = logits_for(args.calibrate_split)
        bias = fit_uniform_bias(dev_logits, len(labels))
        print("\n=== Per-class logit bias (fit on " + args.calibrate_split + ") ===")
        for label, b in zip(labels, bias):
            print(f"  {label:<14}{b:+.4f}")

    y_pred = (logits + bias).argmax(axis=1)  # bias=0 → argmax(logits), the deployed rule

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    print(f"\nAccuracy: {accuracy:.4f} | macro F1: {macro_f1:.4f}")

    print("\n=== OVERALL CLASSIFICATION REPORT ===")
    print(classification_report_text(y_true, y_pred, labels))
    print_class_histogram(y_true, y_pred, labels)
    print_confusion_matrix(y_true, y_pred, labels)

    summary, _ = per_language_f1_report(y_true, y_pred, languages, labels)
    print(summary)

    if args.plot is not None:
        path = Path(args.plot) if args.plot else (
            Path(args.model_dir) / "plots" / f"confusion_matrix_{args.split}.png")
        model_name = os.path.basename(os.path.normpath(args.model_dir))
        plot_confusion_matrix(
            y_true, y_pred, labels, path,
            subtitle=f"{model_name} | {args.split} | "
                     f"accuracy {accuracy:.3f} | macro F1 {macro_f1:.3f}",
        )


if __name__ == "__main__":
    main()
