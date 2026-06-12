#!/usr/bin/env python3
"""Evaluate the trained stance cross-encoder on a held-out LLM-labeled CSV."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import load_dataframe
from analysis.stance_detector_training import select_device, stance_metrics

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
DEFAULT_TEST_CSV = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results",
                                "stance_speeches_llm_labeled_kimi_on_glm.csv")
MAX_LEN = 512
BATCH_SIZE = 64


def load_eval_pairs(path):
    frame = load_dataframe(path).drop(columns=["statement"], errors="ignore").rename(
        columns={"statement_text": "statement", "answer_text": "text"})
    frame["statement"] = frame["statement"].astype("string").str.strip()
    frame["text"] = frame["text"].astype("string").str.strip()
    frame["stance"] = pd.to_numeric(frame["llm_stance"], errors="coerce").astype("float32")
    frame = frame.dropna(subset=["statement", "text", "stance"])
    frame = frame[(frame["statement"].str.len() > 0) & (frame["text"].str.len() > 0)]
    return frame.reset_index(drop=True)


@torch.no_grad()
def predict_stances(frame, model, tokenizer, device):
    predictions = []
    for start in range(0, len(frame), BATCH_SIZE):
        chunk = frame.iloc[start:start + BATCH_SIZE]
        inputs = tokenizer(
            chunk["statement"].tolist(), chunk["text"].tolist(),
            truncation="only_second", max_length=MAX_LEN, padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.view(-1).float().cpu().numpy()
        predictions.append(logits)
    return np.clip(np.concatenate(predictions), -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    args = parser.parse_args()

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    frame = load_eval_pairs(args.test_csv)
    predictions = predict_stances(frame, model, tokenizer, device)
    gold = frame["stance"].to_numpy()

    overall = stance_metrics((predictions, gold))
    print(f"\n=== {os.path.basename(args.test_csv)} (n={len(frame)}) ===")
    print(", ".join(f"{key}={value:.4f}" for key, value in overall.items()))

    rows = []
    for language, group in frame.groupby("language"):
        metrics = stance_metrics((predictions[group.index.to_numpy()], gold[group.index.to_numpy()]))
        rows.append({"language": language, "n": len(group), **metrics})
    print(pd.DataFrame(rows).sort_values("language").to_string(index=False))


if __name__ == "__main__":
    main()
