#!/usr/bin/env python3
"""Evaluate the trained stance cross-encoder on the OOD source model's judged speeches.

Reuses the training module's loaders so the dev/test statement split is byte-identical
to the one training selected on: as long as the set of judged statement ids is the same,
split_ood is seeded and deterministic, so the "test" statements here are exactly the ones
that never touched model selection -- a clean OOD number, now over however many speeches
have been judged since training. Reports test (clean), dev (selection, for reference), and
the full set.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.stance_crossencoder_training import (
    OOD_SOURCE_MODELS, OUTPUT_DIR, MAX_LEN, load_pool, split_ood,
)
from analysis.stance_detector_training import stance_metrics, select_device
from analysis.eval_crossencoder_human import bin_stance
from analysis.validate_stance_judge import direction
from utils import stance_to_likert
from sklearn.metrics import cohen_kappa_score

BATCH_SIZE = 64


@torch.no_grad()
def predict(model, tokenizer, df, device):
    preds = []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        enc = tokenizer(batch["statement"].tolist(), batch["text"].tolist(),
                        truncation="only_second", max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(**enc).logits.reshape(-1)
        preds.append(logits.float().cpu().numpy())
    return np.clip(np.concatenate(preds), -1.0, 1.0)


def report(name, df, preds):
    labels = df["stance"].to_numpy()
    metrics = stance_metrics((preds, labels))
    metrics["sign_acc"] = float(np.mean(np.sign(np.round(preds, 3)) == np.sign(labels)))
    judge_choice = stance_to_likert(labels).round().astype(int)
    pred_choice = stance_to_likert(bin_stance(preds)).round().astype(int)
    metrics["quad_kappa"] = cohen_kappa_score(judge_choice, pred_choice, weights="quadratic",
                                              labels=[1, 2, 3, 4, 5])
    metrics["dir_kappa"] = cohen_kappa_score(direction(labels), direction(preds),
                                             labels=["disagree", "neutral", "agree"])
    print(f"  [{name}] n={len(df)} stmts={df['statement_id'].nunique()} | "
          + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=OUTPUT_DIR,
                        help="Checkpoint to evaluate (default: the training script's output dir).")
    parser.add_argument("--ood_models", default=None,
                        help="Comma-separated held-out source models whose judged speeches to "
                             f"score (default: {OOD_SOURCE_MODELS}). Must match the checkpoint's "
                             "held_out_models for the split to be the one it was selected on.")
    return parser.parse_args()


def main():
    args = parse_args()
    ood_models = args.ood_models.split(",") if args.ood_models else OOD_SOURCE_MODELS
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(PROJECT_ROOT, args.model_dir)
    device = select_device()
    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"No trained model at {args.model_dir}; train it first.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    ood = load_pool(ood_models, "speeches")
    if ood.empty:
        raise SystemExit(f"No judged speeches from {ood_models} to evaluate on.")
    dev, test = split_ood(ood)

    print(f"\n=== {args.model_dir} on {'+'.join(ood_models)} judged speeches "
          f"({len(ood)} / {ood['statement_id'].nunique()} statements) ===")
    ood_preds = predict(model, tokenizer, ood, device)
    report("TEST  (clean, held out from selection)", test, predict(model, tokenizer, test, device))
    report("dev   (used for stage-2 selection)    ", dev, predict(model, tokenizer, dev, device))
    report("full  (dev+test)                      ", ood, ood_preds)

    print("\n  per source model (full):")
    for source, group in ood.assign(pred=ood_preds).groupby("source"):
        report(f"{source:<22}", group, group["pred"].to_numpy())


if __name__ == "__main__":
    main()
