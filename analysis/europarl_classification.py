"""Shared helpers for the EuroParl EU-party classification task.

Used by the trained mmBERT classifier, the hashing-vectorizer baseline, and the
zero-shot LLM baseline so they load data and report metrics identically.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

PARTY_COLUMN = "EU Party"
MIN_LANGUAGE_SAMPLES = 10


def load_split(split, data_dir, max_chars=None, keep_labels=None):
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    df = df.dropna(subset=[PARTY_COLUMN, "text", "language"]).copy()
    text = df["text"].astype(str).str.strip()
    if max_chars is not None:
        text = text.str.slice(0, max_chars)
    df["text"] = text
    df["language"] = df["language"].astype(str)
    df = df[df["text"].str.len() > 0]
    if keep_labels is not None:
        df = df[df[PARTY_COLUMN].isin(keep_labels)]
    return df.reset_index(drop=True)


def fit_uniform_bias(logits, num_labels, iters=500, lr=0.5, eps=1e-9):
    """Additive per-class logit bias so that argmax(logits + bias) has a uniform
    marginal. Iterative prior-shift matching; keeps the bias whose hard prediction
    histogram is closest (L1) to uniform. Fit on the uniform dev split, applied at
    inference to undo the trained model's residual majority-class lean."""
    target = np.full(num_labels, 1.0 / num_labels)
    target_log = np.log(target)
    bias = np.zeros(num_labels)
    best_bias, best_dist = bias.copy(), np.inf
    n = len(logits)
    for _ in range(iters):
        q = np.bincount((logits + bias).argmax(axis=1), minlength=num_labels) / n
        dist = np.abs(q - target).sum()
        if dist < best_dist:
            best_dist, best_bias = dist, bias.copy()
        bias += lr * (target_log - np.log(q + eps))
    return best_bias


def build_label_maps(labels):
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def attach_labels(df, label_to_id):
    df = df[df[PARTY_COLUMN].isin(label_to_id)].copy()
    df["labels"] = df[PARTY_COLUMN].map(label_to_id).astype(int)
    return df[["text", "labels", "language"]].reset_index(drop=True)


def classification_report_text(y_true, y_pred, target_names):
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def per_language_f1_report(y_true, y_pred, languages, target_names, note=""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    languages = np.asarray(languages)

    f1_by_language = {}
    per_language_reports = []
    for language in sorted(np.unique(languages)):
        mask = languages == language
        if mask.sum() < MIN_LANGUAGE_SAMPLES:
            continue
        f1_macro = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        f1_by_language[language] = (f1_macro, int(mask.sum()))
        per_language_reports.append(
            f"\n=== {language} (n={mask.sum()}) | macro F1 = {f1_macro:.4f} ===\n"
            + classification_report_text(y_true[mask], y_pred[mask], target_names)
        )

    prefix = f"{note}, " if note else ""
    summary = f"\n=== PER-LANGUAGE SUMMARY ({prefix}sorted by macro F1) ===\n"
    summary += f"{'lang':<8}{'n':>10}{'macro_f1':>14}\n" + "-" * 32 + "\n"
    for language, (f1_macro, n) in sorted(f1_by_language.items(), key=lambda item: item[1][0]):
        summary += f"{language:<8}{n:>10d}{f1_macro:>14.4f}\n"

    mean_f1 = float(np.mean([f1_macro for f1_macro, _ in f1_by_language.values()]))
    worst_language, (worst_f1, _) = min(f1_by_language.items(), key=lambda item: item[1][0])
    summary += "-" * 32 + "\n"
    summary += f"MEAN across languages: {mean_f1:.4f}\n"
    summary += f"MIN  across languages: {worst_f1:.4f}  ({worst_language})\n"

    return summary, per_language_reports
