#!/usr/bin/env python3
"""Compare the zero-shot LLM party baseline against the trained mmBERT classifier.

classify_europarl_api.py caches each LLM prediction as it goes. This script rebuilds
the exact same speeches it indexed into (same split, limit and seed -> same stratified
subsample), runs the mmBERT-base seed-42 model on them, and reports accuracy / macro-F1
for both head to head -- an apples-to-apples quality check on whatever the LLM finished.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.europarl_classification import (
    PARTY_COLUMN, classification_report_text, load_split, per_language_f1_report,
)
from analysis.classify_europarl_api import (
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, LABELS, LABEL_TO_ID, subsample,
)

MODEL_DIR = PROJECT_ROOT / "mmBERT-base-trainer_strat" / "seed-42"
MAX_LEN = 512


def cache_path(output_dir, model, split, limit):
    base_name = f"{model.replace('/', '_')}_{split}"
    if limit is not None:
        base_name += f"_n{limit}"
    return Path(output_dir) / f"{base_name}.cache.json"


@torch.no_grad()
def predict_mmbert(texts, device, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    preds = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN,
        ).to(device)
        preds[start:start + len(chunk)] = model(**inputs).logits.argmax(-1).cpu().numpy()
        print(f"  mmBERT {min(start + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return preds


def scores(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def row_normalize(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)


def heatmap(title, color_values, cell_text, cmap, vmin, vmax, cbar_label, out_png):
    fig, ax = plt.subplots(figsize=(max(7, len(LABELS)), max(6, len(LABELS) * 0.9)))
    image = ax.imshow(color_values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    span = max(abs(vmin), abs(vmax))
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=8,
                    color="white" if abs(color_values[i, j]) < span * 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def confusion(name, y_true, y_pred, out_png):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))

    print(f"\n=== {name} confusion matrix (rows=true, cols=pred) ===")
    print("true\\pred ".ljust(12) + "".join(f"{label[:8]:>9}" for label in LABELS))
    for label, row in zip(LABELS, matrix):
        print(label.ljust(12) + "".join(f"{count:>9d}" for count in row))

    normalized = row_normalize(matrix)
    text = [[str(count) for count in row] for row in matrix]
    heatmap(name, normalized, text, "viridis", 0.0, 1.0, "row-normalized (recall)", out_png)
    return matrix


def difference(name_a, matrix_a, name_b, matrix_b, out_png):
    diff = row_normalize(matrix_a) - row_normalize(matrix_b)

    print(f"\n=== {name_a} - {name_b} recall difference (rows=true, cols=pred) ===")
    print("true\\pred ".ljust(12) + "".join(f"{label[:8]:>9}" for label in LABELS))
    for label, row in zip(LABELS, diff):
        print(label.ljust(12) + "".join(f"{value:>+9.2f}" for value in row))

    limit = max(abs(diff.min()), abs(diff.max()), 1e-6)
    text = [[f"{value:+.2f}" for value in row] for row in diff]
    heatmap(f"{name_a} - {name_b} (recall difference)", diff, text, "RdBu_r",
            -limit, limit, f"recall: {name_a} - {name_b}", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Rebuild the exact frame classify_europarl_api indexed into.
    df = subsample(load_split(args.split, args.data_dir, keep_labels=LABELS), args.limit, args.seed)
    cache = json.loads(cache_path(args.output_dir, args.model, args.split, args.limit).read_text("utf-8"))
    rows = sorted(int(k) for k in cache)
    print(f"LLM has classified {len(rows)} / {len(df)} {args.split} speeches so far.")

    sub = df.iloc[rows].reset_index(drop=True)
    y_true = sub[PARTY_COLUMN].map(LABEL_TO_ID).to_numpy()
    langs = sub["language"].to_numpy()

    llm_choice = [cache[str(i)]["choice"] for i in rows]
    parsable = np.array([c in LABEL_TO_ID for c in llm_choice])
    y_llm = np.array([LABEL_TO_ID.get(c, -1) for c in llm_choice])
    print(f"LLM parsable predictions: {parsable.sum()} / {len(rows)}")

    y_mmbert = predict_mmbert(sub["text"].tolist(), args.device, args.batch_size)

    yt, yl, ym, lg = y_true[parsable], y_llm[parsable], y_mmbert[parsable], langs[parsable]
    llm_acc, llm_f1 = scores(yt, yl)
    mm_acc, mm_f1 = scores(yt, ym)

    print("\n" + "=" * 70)
    print(f"Head-to-head on {parsable.sum()} shared, parsable speeches:")
    print(f"  {'model':<16}{'accuracy':>12}{'macro_f1':>12}")
    print(f"  {args.model:<16}{llm_acc:>12.4f}{llm_f1:>12.4f}")
    print(f"  {'mmBERT seed-42':<16}{mm_acc:>12.4f}{mm_f1:>12.4f}")
    print(f"  agreement (LLM == mmBERT): {(yl == ym).mean():.4f}")

    print(f"\n--- {args.model} per-class ---")
    print(classification_report_text(yt, yl, LABELS))
    print("--- mmBERT seed-42 per-class ---")
    print(classification_report_text(yt, ym, LABELS))

    base_name = f"{args.model.replace('/', '_')}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    llm_matrix = confusion(args.model, yt, yl, args.output_dir / f"{base_name}_confusion_llm.png")
    mm_matrix = confusion("mmBERT seed-42", yt, ym, args.output_dir / f"{base_name}_confusion_mmbert.png")
    difference(args.model, llm_matrix, "mmBERT seed-42", mm_matrix,
               args.output_dir / f"{base_name}_confusion_diff.png")

    print(f"=== {args.model} per-language macro-F1 ===")
    print(per_language_f1_report(yt, yl, lg, LABELS, note=args.model)[0])
    print("=== mmBERT per-language macro-F1 ===")
    print(per_language_f1_report(yt, ym, lg, LABELS, note="mmBERT")[0])


if __name__ == "__main__":
    main()
