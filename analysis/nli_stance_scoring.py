import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, VARIANTS

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ENTAILMENT_LOGIT_INDEX = 0
CONTRADICTION_LOGIT_INDEX = 2
MAX_TOKEN_LENGTH = 512


def load_hypothesis_templates(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_nli_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def entailment_minus_contradiction(speeches, hypotheses, tokenizer, model, device):
    inputs = tokenizer(
        speeches, hypotheses,
        return_tensors="pt", truncation=True, padding=True,
        max_length=MAX_TOKEN_LENGTH,
    ).to(device)
    label_probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
    entailment_probability = label_probabilities[:, ENTAILMENT_LOGIT_INDEX]
    contradiction_probability = label_probabilities[:, CONTRADICTION_LOGIT_INDEX]
    return entailment_probability - contradiction_probability


def speech_variant_indices(df, lang_variant):
    answer_column_prefix = f"answer_{lang_variant}_v"
    return sorted({
        int(column[len(answer_column_prefix):])
        for column in df.columns
        if column.startswith(answer_column_prefix)
        and column[len(answer_column_prefix):].isdigit()
    })


def collect_speech_hypothesis_pairs(df, lang_variant, hypothesis_template):
    statement_column = f"original_text_{lang_variant}"
    variant_indices = speech_variant_indices(df, lang_variant)
    pairs = []
    for row_index, row in df.iterrows():
        statement = row[statement_column]
        if not isinstance(statement, str) or not statement.strip():
            continue
        hypothesis = hypothesis_template.format(statement)
        for variant_index in variant_indices:
            speech = row.get(f"answer_{lang_variant}_v{variant_index}")
            if isinstance(speech, str) and speech.strip():
                pairs.append((row_index, variant_index, speech, hypothesis))
    return pairs, variant_indices


def score_pairs_in_batches(pairs, tokenizer, model, device, batch_size, description):
    scores = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), batch_size), desc=description):
        chunk = pairs[start:start + batch_size]
        chunk_speeches = [speech for _, _, speech, _ in chunk]
        chunk_hypotheses = [hypothesis for _, _, _, hypothesis in chunk]
        scores[start:start + len(chunk)] = entailment_minus_contradiction(
            chunk_speeches, chunk_hypotheses, tokenizer, model, device,
        )
    return scores


def write_stance_columns(df, pairs, scores, lang_variant, variant_indices):
    for (row_index, variant_index, _, _), score in zip(pairs, scores):
        df.at[row_index, f"stance_{lang_variant}_v{variant_index}"] = score
    per_variant_stance_columns = [f"stance_{lang_variant}_v{i}" for i in variant_indices]
    df[f"stance_{lang_variant}_mean"] = df[per_variant_stance_columns].mean(axis=1)


def score_speech_stances(df, languages, variant, hypothesis_templates, tokenizer, model, device, batch_size):
    for language in languages:
        lang_variant = f"{language}{variant}"
        if f"original_text_{lang_variant}" not in df.columns:
            print(f"[{lang_variant}] no statement column, skipping.")
            continue
        if language not in hypothesis_templates:
            print(f"[{lang_variant}] no hypothesis template for '{language}', skipping.")
            continue
        pairs, variant_indices = collect_speech_hypothesis_pairs(
            df, lang_variant, hypothesis_templates[language],
        )
        if not pairs:
            print(f"[{lang_variant}] no speeches found, skipping.")
            continue
        scores = score_pairs_in_batches(
            pairs, tokenizer, model, device, batch_size,
            description=f"scoring {lang_variant}",
        )
        write_stance_columns(df, pairs, scores, lang_variant, variant_indices)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--hypothesis_templates", default="./prompts/nli_hypotheses.json")
    parser.add_argument("--llm", default="qwen3.5-122b")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--variant", default="_negated", choices=VARIANTS)
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_input_path(args, languages):
    if args.input:
        return args.input
    return f"./data/{args.dataset}_results/{args.llm}/speeches_{','.join(languages)}{args.variant}.csv"


def main():
    args = parse_args()
    languages = args.languages.split(",")
    input_path = resolve_input_path(args, languages)

    hypothesis_templates = load_hypothesis_templates(args.hypothesis_templates)
    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    tokenizer, model = load_nli_model(args.nli_model, args.device)

    df = score_speech_stances(
        df, languages, args.variant, hypothesis_templates,
        tokenizer, model, args.device, args.batch_size,
    )

    stem, ext = os.path.splitext(input_path)
    output_path = f"{stem}_scored{ext}"
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()