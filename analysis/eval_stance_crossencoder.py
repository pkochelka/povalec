#!/usr/bin/env python3
"""Evaluate the trained stance cross-encoder on the OOD source model's judged speeches.

Reuses the training module's loaders so the dev/test statement split is byte-identical
to the one training selected on: as long as the set of judged statement ids is the same,
split_ood is seeded and deterministic, so the "test" statements here are exactly the ones
that never touched model selection -- a clean OOD number, now over however many speeches
have been judged since training. Reports test (clean), dev (selection, for reference), and
the full set.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.stance_crossencoder_training import (
    OOD_SOURCE_MODEL, OUTPUT_DIR, MAX_LEN, load_judged, split_ood,
)
from analysis.stance_detector_training import stance_metrics, select_device

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
    print(f"  [{name}] n={len(df)} stmts={df['statement_id'].nunique()} | "
          + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main():
    device = select_device()
    if not os.path.isdir(OUTPUT_DIR):
        raise SystemExit(f"No trained model at {OUTPUT_DIR}; train it first.")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device).eval()

    ood = load_judged(OOD_SOURCE_MODEL, "speeches").drop_duplicates(
        subset=["statement", "text"]).reset_index(drop=True)
    if ood.empty:
        raise SystemExit(f"No judged {OOD_SOURCE_MODEL} speeches to evaluate on.")
    dev, test = split_ood(ood)

    print(f"\n=== {OOD_SOURCE_MODEL} cross-encoder eval "
          f"({len(ood)} speeches / {ood['statement_id'].nunique()} statements) ===")
    report("TEST  (clean, held out from selection)", test, predict(model, tokenizer, test, device))
    report("dev   (used for stage-2 selection)    ", dev, predict(model, tokenizer, dev, device))
    report("full  (dev+test)                      ", ood, predict(model, tokenizer, ood, device))


if __name__ == "__main__":
    main()
