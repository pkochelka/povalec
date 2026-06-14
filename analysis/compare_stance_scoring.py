#!/usr/bin/env python3
"""Compare the fine-tuned stance cross-encoder against the original NLI stance scores.

The NLI scorer (nli_stance_scoring.py) reads each speech *against its statement*
and stored stance_{lang}_v{N} in the *_scored.csv files. The cross-encoder also
scores each (statement, speech) pair, the same input format it was trained on.
This script runs it on the same speeches, then reports the total divergence from
the NLI scores plus the largest per-speech disagreements in a readable language.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import ALL_LANGS, ALL_LANGS_STR
from analysis.nli_stance_scoring import speech_variant_indices

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
SOURCE_MODELS = ["kimi-k2.6", "deepseek-v4-pro"]
PARAPHRASES = {"": "base", "_question": "neutral", "_negated": "negated"}
READABLE_LANGUAGES = ["en", "cz", "sk", "pl", "fr", "lu", "be"]


def load_regressor(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    max_len = json.load(open(os.path.join(MODEL_DIR, "stance_config.json")))["max_len"]
    return tokenizer, model, max_len


@torch.no_grad()
def predict_stances(statements, texts, tokenizer, model, device, max_len, batch_size):
    scores = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        statement_chunk = statements[start:start + batch_size]
        text_chunk = texts[start:start + batch_size]
        inputs = tokenizer(
            statement_chunk, text_chunk,
            return_tensors="pt", truncation="only_second", padding=True, max_length=max_len,
        ).to(device)
        logits = model(**inputs).logits.view(-1).float().cpu().numpy()
        scores[start:start + len(text_chunk)] = np.clip(logits, -1.0, 1.0)
    return scores


def collect_comparisons():
    blocks = []
    for source_model in SOURCE_MODELS:
        for suffix, paraphrase in PARAPHRASES.items():
            path = os.path.join(DATA_DIR, source_model, f"speeches_{ALL_LANGS_STR}{suffix}_scored.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
            for language in ALL_LANGS:
                lang_variant = f"{language}{suffix}"
                statement_column = f"original_text_{lang_variant}"
                if statement_column not in df.columns:
                    continue
                for variant in speech_variant_indices(df, lang_variant):
                    answer_column = f"answer_{lang_variant}_v{variant}"
                    stance_column = f"stance_{lang_variant}_v{variant}"
                    if answer_column not in df.columns or stance_column not in df.columns:
                        continue
                    blocks.append(pd.DataFrame({
                        "model": source_model,
                        "paraphrase": paraphrase,
                        "language": language,
                        "variant": variant,
                        "statement": df[statement_column],
                        "answer": df[answer_column],
                        "nli_stance": pd.to_numeric(df[stance_column], errors="coerce"),
                    }))
    comparisons = pd.concat(blocks, ignore_index=True)
    comparisons = comparisons.dropna(subset=["nli_stance"])
    comparisons = comparisons[
        comparisons["answer"].apply(lambda x: isinstance(x, str) and x.strip() != "")
        & comparisons["statement"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    ]
    return comparisons.reset_index(drop=True)


def report_totals(comparisons):
    diff = comparisons["regressor_stance"] - comparisons["nli_stance"]
    abs_diff = diff.abs()
    print("=" * 70)
    print(f"Speeches compared: {len(comparisons):,}")
    print(f"Total absolute difference:  {abs_diff.sum():.1f}")
    print(f"Mean absolute difference:   {abs_diff.mean():.4f}")
    print(f"Mean signed difference:     {diff.mean():+.4f}  (regressor - NLI; +ve = regressor more 'agree')")
    print(f"Pearson(regressor, NLI):    {pearsonr(comparisons['regressor_stance'], comparisons['nli_stance'])[0]:.4f}")
    print(f"Spearman(regressor, NLI):   {spearmanr(comparisons['regressor_stance'], comparisons['nli_stance'])[0]:.4f}")
    print("\nMean |difference| by readable language:")
    per_language = (comparisons[comparisons["language"].isin(READABLE_LANGUAGES)]
                    .assign(abs_diff=abs_diff)
                    .groupby("language")["abs_diff"].agg(["mean", "count"]))
    for language in READABLE_LANGUAGES:
        if language in per_language.index:
            row = per_language.loc[language]
            print(f"  {language}: mean |diff|={row['mean']:.4f}  (n={int(row['count'])})")


def report_largest(comparisons, language, top_k):
    subset = comparisons[comparisons["language"] == language].copy()
    subset["abs_diff"] = (subset["regressor_stance"] - subset["nli_stance"]).abs()
    largest = subset.nlargest(top_k, "abs_diff")
    print("\n" + "=" * 70)
    print(f"Top {top_k} largest disagreements in '{language}':")
    for _, row in largest.iterrows():
        print("-" * 70)
        print(f"[{row['model']} | {row['paraphrase']} | v{row['variant']}]  "
              f"NLI={row['nli_stance']:+.3f}  regressor={row['regressor_stance']:+.3f}  "
              f"diff={row['regressor_stance'] - row['nli_stance']:+.3f}")
        print(f"  STATEMENT: {row['statement'].strip()}")
        speech = " ".join(row["answer"].split())
        print(f"  SPEECH:    {speech[:400]}{'...' if len(speech) > 400 else ''}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", choices=READABLE_LANGUAGES)
    parser.add_argument("--top_k", default=15, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model, max_len = load_regressor(args.device)

    comparisons = collect_comparisons()
    comparisons["regressor_stance"] = predict_stances(
        comparisons["statement"].tolist(), comparisons["answer"].tolist(),
        tokenizer, model, args.device, max_len, args.batch_size,
    )

    report_totals(comparisons)
    report_largest(comparisons, args.language, args.top_k)


if __name__ == "__main__":
    main()
