#!/usr/bin/env python3
"""Run the stance cross-encoder on a labeled speeches CSV and compare to its stored scoring.

Reads a stance_speeches_llm_labeled_*.csv and compares the cross-encoder against the score
already in that file, which is the LLM judge's.
"""
import argparse
import os

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from analysis.stance_crossencoder import DATA_DIR, load_regressor, predict_stances

# Languages a human reviewer of this script can actually read the examples in. Was shared
# with compare_stance_scoring.py until that was deleted; this is now its only consumer.
READABLE_LANGUAGES = ["en", "cz", "sk", "pl", "fr", "lu", "be"]

DEFAULT_CSV = os.path.join(DATA_DIR, "stance_speeches_llm_labeled_kimi_on_glm_full.csv")


def load_labeled(path, reference):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    df = df.drop(columns=["statement"], errors="ignore").rename(
        columns={"statement_text": "statement", "answer_text": "answer"})
    df["reference"] = pd.to_numeric(df[reference], errors="coerce")
    df = df.dropna(subset=["reference"])
    df = df[
        df["answer"].apply(lambda x: isinstance(x, str) and x.strip() != "")
        & df["statement"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    ]
    return df.reset_index(drop=True)


def report_totals(df, ref_label):
    diff = df["mmbert_stance"] - df["reference"]
    abs_diff = diff.abs()
    print("=" * 70)
    print(f"Speeches compared: {len(df):,}")
    print(f"Total absolute difference:  {abs_diff.sum():.1f}")
    print(f"Mean absolute difference:   {abs_diff.mean():.4f}")
    print(f"Mean signed difference:     {diff.mean():+.4f}  (mmBERT - {ref_label}; +ve = mmBERT more 'agree')")
    print(f"Pearson(mmBERT, {ref_label}):    {pearsonr(df['mmbert_stance'], df['reference'])[0]:.4f}")
    print(f"Spearman(mmBERT, {ref_label}):   {spearmanr(df['mmbert_stance'], df['reference'])[0]:.4f}")
    print("\nMean |difference| by readable language:")
    per_language = (df[df["language"].isin(READABLE_LANGUAGES)]
                    .assign(abs_diff=abs_diff)
                    .groupby("language")["abs_diff"].agg(["mean", "count"]))
    for language in READABLE_LANGUAGES:
        if language in per_language.index:
            row = per_language.loc[language]
            print(f"  {language}: mean |diff|={row['mean']:.4f}  (n={int(row['count'])})")


def report_largest(df, language, top_k, ref_label):
    subset = df[df["language"] == language].copy()
    subset["abs_diff"] = (subset["mmbert_stance"] - subset["reference"]).abs()
    largest = subset.nlargest(top_k, "abs_diff")
    print("\n" + "=" * 70)
    print(f"Top {top_k} largest disagreements in '{language}':")
    for _, row in largest.iterrows():
        print("-" * 70)
        print(f"[{row['model']} | {row['paraphrase']} | v{row['variant']}]  "
              f"{ref_label}={row['reference']:+.3f}  mmBERT={row['mmbert_stance']:+.3f}  "
              f"diff={row['mmbert_stance'] - row['reference']:+.3f}")
        print(f"  STATEMENT: {row['statement'].strip()}")
        speech = " ".join(row["answer"].split())
        print(f"  SPEECH:    {speech[:400]}{'...' if len(speech) > 400 else ''}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_CSV)
    parser.add_argument("--reference", default="llm_stance", choices=["llm_stance", "nli_stance"])
    parser.add_argument("--language", default="en", choices=READABLE_LANGUAGES)
    parser.add_argument("--top_k", default=15, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model, max_len = load_regressor(args.device)

    df = load_labeled(args.input, args.reference)
    df["mmbert_stance"] = predict_stances(
        df["statement"].tolist(), df["answer"].tolist(),
        tokenizer, model, args.device, max_len, args.batch_size,
    )

    ref_label = args.reference.replace("_stance", "").upper()
    report_totals(df, ref_label)
    report_largest(df, args.language, args.top_k, ref_label)


if __name__ == "__main__":
    main()
