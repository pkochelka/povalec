#!/usr/bin/env python3
import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--transformer", default="./answerdotai/ModernBERT-large-trainer/en-6epochs-full-data-debates-filtered-europarl-unfiltered", type=str)
parser.add_argument("--llm", default="gpt-oss-120b", type=str, choices=["gpt-oss-120b", "qwen3.5-122b"])
parser.add_argument("--input", default=None, type=str, help="Input CSV; defaults to ./data/euandi_2019_results/{llm}/speeches_{languages}{variant}.csv")
parser.add_argument("--variant", default="_negated", type=str, choices=["", "_question", "_negated"])
parser.add_argument("--languages", default="en", type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def classify_dataframe(df, languages, variant, classifier, batch_size):
    for language in languages:
        lang_variant = f"{language}{variant}"
        v = 0
        found = False
        while True:
            col = f"answer_{lang_variant}_v{v}"
            if col not in df.columns:
                break
            found = True
            valid_mask = df[col].notna() & df[col].astype(str).str.strip().ne("")
            valid_texts = df.loc[valid_mask, col].astype(str).tolist()
            if valid_texts:
                results = classifier(valid_texts, batch_size=batch_size)
                predictions = [res["label"] for res in results]
                df.loc[valid_mask, f"predicted_label_{lang_variant}_v{v}"] = predictions
            v += 1
        if not found:
            print(f"[{lang_variant}] no answer columns found, skipping.")
    return df


def run_evaluation(transformer_path, input_path, output_path, languages, variant, batch_size, device):
    print(f"--- Loading model from: {transformer_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_path)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )

    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {input_path}")

    df = classify_dataframe(df, languages, variant, classifier, batch_size)

    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")
    input_path = args.input or f"./data/euandi_2019_results/{args.llm}/speeches_en,de,el,es,fr,it{args.variant}.csv"
    stem, ext = os.path.splitext(input_path)
    output_path = f"{stem}_classified{ext}"
    run_evaluation(args.transformer, input_path, output_path, languages, args.variant, args.batch_size, args.device)
