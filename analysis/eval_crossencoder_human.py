#!/usr/bin/env python3
"""Agreement of a stance cross-encoder checkpoint with the hand annotations.

The other validator scores the cross-encoder against silver labels from the LLM
judge (compare_mmbert_vs_labeled.py). This one scores the hand-labeled speeches,
whose `choice` column is already stance in [-1, 1] in 5 bins (+1 = totally agree).

The cross-encoder emits a continuous stance; for the Likert-only metrics
(quadratic kappa, 5x5 confusion) it is cut into the same 5 bins the sample was
drawn with (sample_speeches_for_labeling.STANCE_BIN_EDGES). Pearson/Spearman/MAE
stay on the continuous score. Human is ground truth throughout.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import stance_to_likert, configure_stdout
from analysis.stance_crossencoder import DATA_DIR, load_regressor, predict_stances
from analysis.sample_speeches_for_labeling import STANCE_BIN_EDGES
from analysis.validate_stance_judge import NEUTRAL_BAND, direction, read_csv_any

configure_stdout()

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
LABELED = os.path.join(DATA_DIR, "stance_speeches_labeled_en-cz-sk-fr.csv")
META = os.path.join(DATA_DIR, "stance_speeches_to_label_meta_en-cz-sk-fr.csv")
OUT = os.path.join(DATA_DIR, "crossencoder_human_eval_preds.csv")

BIN_STANCES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
DIRECTIONS = ["disagree", "neutral", "agree"]


def bin_stance(stance):
    """Cut a continuous stance onto the 5 human bins using the sampler's edges
    (+/-0.2 neutral band, +/-0.6 to the extremes)."""
    index = pd.cut(np.clip(stance, -1.0, 1.0), bins=STANCE_BIN_EDGES, right=False, labels=False)
    return BIN_STANCES[index.astype(int)]


def describe_checkpoint(model_dir):
    config_path = os.path.join(model_dir, "stance_config.json")
    if not os.path.exists(config_path):
        return
    config = json.load(open(config_path, encoding="utf-8"))
    print(f"Checkpoint: {model_dir}")
    for key in ("training_source_models", "training", "held_out_models", "neutral_weight",
                "selected_on", "max_len"):
        if key in config:
            print(f"  {key}: {config[key]}")


def load_gold(labeled_path, meta_path):
    df = read_csv_any(labeled_path)
    df["choice"] = pd.to_numeric(df["choice"], errors="coerce")
    if os.path.exists(meta_path):
        meta = read_csv_any(meta_path)
        df = df.merge(meta[["id", "statement", "paraphrase"]], on="id", how="left")
    for column in ("statement_text", "answer_text"):
        df[column] = df[column].astype("string").str.strip()
    df = df.dropna(subset=["choice", "statement_text", "answer_text"])
    df = df[(df["statement_text"].str.len() > 0) & (df["answer_text"].str.len() > 0)]
    return df.reset_index(drop=True)


def report(df):
    human, ce = df["human_stance"].to_numpy(), df["ce_stance"].to_numpy()
    human_choice, ce_choice = df["human_choice"].to_numpy(), df["ce_choice"].to_numpy()

    print(f"\n=== Cross-encoder vs human gold  (n={len(df)}) ===")
    print(f"  Pearson r   : {pearsonr(human, ce)[0]:.3f}")
    print(f"  Spearman r  : {spearmanr(human, ce)[0]:.3f}")
    print(f"  MAE (stance): {np.mean(np.abs(human - ce)):.3f}")
    print(f"  Mean signed : {np.mean(ce - human):+.3f}  (cross-encoder - human; +ve = more 'agree')")
    print(f"  Exact choice agreement  : {np.mean(human_choice == ce_choice):.3f}")
    print(f"  Within-1 agreement      : {np.mean(np.abs(human_choice - ce_choice) <= 1):.3f}")
    print(f"  Quadratic-weighted kappa: "
          f"{cohen_kappa_score(human_choice, ce_choice, weights='quadratic', labels=[1, 2, 3, 4, 5]):.3f}")

    human_dir, ce_dir = direction(human), direction(ce)
    print(f"\n  3-class direction (agree/neutral/disagree at +/-{NEUTRAL_BAND})")
    print(f"    accuracy        : {np.mean(human_dir == ce_dir):.3f}")
    print(f"    directional kappa: {cohen_kappa_score(human_dir, ce_dir, labels=DIRECTIONS):.3f}")

    print("\n=== 5x5 choice confusion (rows=human, cols=cross-encoder) ===")
    print("(choice 1=totally agree ... 5=totally disagree)")
    cm = confusion_matrix(human_choice, ce_choice, labels=[1, 2, 3, 4, 5])
    print(pd.DataFrame(cm, index=[f"h{i}" for i in range(1, 6)],
                       columns=[f"c{i}" for i in range(1, 6)]).to_string())

    print("\n=== 3x3 direction confusion (rows=human, cols=cross-encoder) ===")
    dcm = confusion_matrix(human_dir, ce_dir, labels=DIRECTIONS)
    print(pd.DataFrame(dcm, index=[f"h_{d}" for d in DIRECTIONS],
                       columns=[f"c_{d}" for d in DIRECTIONS]).to_string())

    print("\n=== Choice distribution (1=totally agree ... 5=totally disagree) ===")
    counts = pd.DataFrame({
        "human": pd.Series(human_choice).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0),
        "cross-encoder": pd.Series(ce_choice).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0),
    })
    print(counts.to_string())

    print("\n=== Per-language ===")
    per_language = df.groupby("language").apply(lambda g: pd.Series({
        "n": len(g),
        "within1": np.mean(np.abs(g["human_choice"] - g["ce_choice"]) <= 1),
        "pearson": pearsonr(g["human_stance"], g["ce_stance"])[0],
        "mae": np.mean(np.abs(g["human_stance"] - g["ce_stance"])),
    }), include_groups=False)
    print(per_language.round(3).to_string())

    print("\n=== Per source model ===")
    per_model = df.groupby("model").apply(lambda g: pd.Series({
        "n": len(g),
        "within1": np.mean(np.abs(g["human_choice"] - g["ce_choice"]) <= 1),
        "pearson": pearsonr(g["human_stance"], g["ce_stance"])[0],
        "mae": np.mean(np.abs(g["human_stance"] - g["ce_stance"])),
    }), include_groups=False)
    print(per_model.round(3).to_string())


def write_predictions(df, out_path):
    diffs = df.assign(abs_diff=(df["human_stance"] - df["ce_stance"]).abs())
    diffs = diffs.sort_values("abs_diff", ascending=False)
    columns = [c for c in ["id", "model", "language", "statement", "paraphrase", "abs_diff",
                           "human_choice", "ce_choice", "human_stance", "ce_stance",
                           "statement_text", "answer_text"] if c in diffs.columns]
    diffs[columns].to_csv(out_path, sep=";", encoding="utf-8-sig", index=False)

    print(f"\n=== Largest absolute disagreements -> {out_path} ===")
    preview = diffs[diffs["abs_diff"] > 0].copy()
    preview["statement_text"] = preview["statement_text"].str.slice(0, 60)
    preview["answer_text"] = preview["answer_text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 80)
    print(preview[["id", "language", "abs_diff", "human_choice", "ce_choice",
                   "statement_text", "answer_text"]].head(20).to_string(index=False))
    print(f"\n{len(preview)} of {len(df)} pairs differ at all; "
          f"{int((preview['abs_diff'] >= 1.0).sum())} differ by >= 1.0 stance.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--labeled", default=LABELED)
    parser.add_argument("--meta", default=META)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    describe_checkpoint(args.model_dir)

    df = load_gold(args.labeled, args.meta)
    print(f"\nLoaded {len(df)} annotated rows from {args.labeled}")

    tokenizer, model, max_len = load_regressor(args.device, args.model_dir)
    df["ce_stance"] = predict_stances(
        df["statement_text"].tolist(), df["answer_text"].tolist(),
        tokenizer, model, args.device, max_len, args.batch_size,
    )

    df["human_stance"] = df["choice"]
    df["human_choice"] = stance_to_likert(df["choice"]).round().astype(int)
    df["ce_choice"] = stance_to_likert(bin_stance(df["ce_stance"].to_numpy())).round().astype(int)

    report(df)
    write_predictions(df, args.out)


if __name__ == "__main__":
    main()
