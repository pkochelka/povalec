#!/usr/bin/env python3
"""Evaluate the trained mmBERT EU-party classifier on the uniform test split.

Loads the saved model + manifest, runs argmax(logits) inference on test.parquet
(the same inference rule classify_speeches.py uses), and reports:
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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import (
    PARTY_COLUMN,
    classification_report_text,
    fit_uniform_bias,
    load_split,
    per_language_f1_report,
)

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmBERT-base-balanced-train-first")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom")


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


if __name__ == "__main__":
    main()
