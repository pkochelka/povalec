#!/usr/bin/env python3
"""Refit a trained classifier's softmax temperature on the correct dev split.

mmBERT-base-balanced-calibrated-second's manifest.json temperature was fit on
data/EuroParl Custom/dev.parquet instead of the canonical
data/EuroParl Custom/cleaned/dev.parquet. This recomputes the temperature on the
given dev split and overwrites manifest.json in place (the previous manifest is
kept as manifest.json.bak).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import attach_labels, load_split

DEFAULT_MODEL_DIR = "./mmBERT-base-balanced-calibrated-second"
DEFAULT_DEV_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "cleaned")
DEFAULT_BATCH_SIZE = 32
EPSILON = 1e-12
CALIBRATION_BINS = 15


def softmax_np(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibration_metrics(probs, labels, num_labels, n_bins=CALIBRATION_BINS):
    """Expected calibration error, NLL and Brier score for predicted probabilities."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())

    nll = float(-np.log(probs[np.arange(len(labels)), labels] + EPSILON).mean())
    onehot = np.eye(num_labels)[labels]
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    return {"ece": float(ece), "nll": nll, "brier": brier}


def fit_temperature(logits, labels, device):
    """Single-parameter temperature that minimises dev NLL (Guo et al. 2017)."""
    logits_t = torch.tensor(logits, dtype=torch.float, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / log_temperature.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dev_dir", default=DEFAULT_DEV_DIR)
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def compute_dev_logits(texts, tokenizer, model, device, max_len, batch_size):
    logits = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc="scoring dev"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len,
        ).to(device)
        logits[start:start + len(batch)] = model(**inputs).logits.float().cpu().numpy()
    return logits


def main():
    args = parse_args()
    manifest_path = os.path.join(args.model_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"--- Loading model from: {args.model_dir} ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    print(f"--- Scoring dev split: {args.dev_dir} ---")
    dev_df = attach_labels(load_split("dev", args.dev_dir), manifest["label2id"])
    dev_logits = compute_dev_logits(
        dev_df["text"].tolist(), tokenizer, model, args.device, manifest["max_len"], args.batch_size,
    )
    dev_labels = dev_df["labels"].to_numpy()
    num_labels = len(manifest["label2id"])

    old_temperature = float(manifest.get("temperature", 1.0))
    new_temperature = fit_temperature(dev_logits, dev_labels, args.device)

    old_metrics = calibration_metrics(softmax_np(dev_logits / old_temperature), dev_labels, num_labels)
    new_metrics = calibration_metrics(softmax_np(dev_logits / new_temperature), dev_labels, num_labels)

    print(f"Old temperature: {old_temperature:.4f} | dev ece={old_metrics['ece']:.4f} nll={old_metrics['nll']:.4f} brier={old_metrics['brier']:.4f}")
    print(f"New temperature: {new_temperature:.4f} | dev ece={new_metrics['ece']:.4f} nll={new_metrics['nll']:.4f} brier={new_metrics['brier']:.4f}")

    shutil.copyfile(manifest_path, manifest_path + ".bak")
    manifest["temperature"] = new_temperature
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {manifest_path} (previous version backed up to {manifest_path}.bak)")


if __name__ == "__main__":
    main()
