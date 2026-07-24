import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import ALL_LANGS_STR

DEFAULT_CROSSENCODER_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
DEFAULT_MAX_TOKEN_LENGTH = 512

REFUSED_REASON_PREFIXES = ("REFUSED",)
FAILED_REASON_VALUES = {"FAILED"}

SOURCE_TEXT_COLUMN_PREFIX = {"speeches": "answer", "reasons": "reason"}
SOURCE_INPUT_FILENAME = {
    "speeches": "speeches_{langs}{variant}.csv",
    "reasons": "{langs}{variant}.csv",
}


def load_crossencoder(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    config_path = os.path.join(model_dir, "stance_config.json")
    max_length = DEFAULT_MAX_TOKEN_LENGTH
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            max_length = json.load(f).get("max_len", DEFAULT_MAX_TOKEN_LENGTH)
    return tokenizer, model, max_length


@torch.no_grad()
def agreement_scores(statements, speeches, tokenizer, model, device, max_length):
    inputs = tokenizer(
        statements, speeches,
        return_tensors="pt", truncation="only_second", padding=True,
        max_length=max_length,
    ).to(device)
    logits = model(**inputs).logits.view(-1).float().cpu().numpy()
    return np.clip(logits, -1.0, 1.0)


def text_variant_indices(df, lang_variant, text_prefix):
    column_prefix = f"{text_prefix}_{lang_variant}_v"
    return sorted({
        int(column[len(column_prefix):])
        for column in df.columns
        if column.startswith(column_prefix)
        and column[len(column_prefix):].isdigit()
    })


def is_scorable_text(value, source):
    if not isinstance(value, str) or not value.strip():
        return False
    if source == "reasons":
        text = value.strip()
        if text in FAILED_REASON_VALUES or text.startswith(REFUSED_REASON_PREFIXES):
            return False
    return True


def collect_statement_text_pairs(df, lang_variant, text_prefix, source):
    statement_column = f"original_text_{lang_variant}"
    variant_indices = text_variant_indices(df, lang_variant, text_prefix)
    pairs = []
    for row_index, row in df.iterrows():
        statement = row[statement_column]
        if not isinstance(statement, str) or not statement.strip():
            continue
        for variant_index in variant_indices:
            text = row.get(f"{text_prefix}_{lang_variant}_v{variant_index}")
            if is_scorable_text(text, source):
                pairs.append((row_index, variant_index, statement, text.strip()))
    return pairs, variant_indices


def score_pairs_in_batches(pairs, tokenizer, model, device, max_length, batch_size, description):
    scores = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), batch_size), desc=description):
        chunk = pairs[start:start + batch_size]
        chunk_statements = [statement for _, _, statement, _ in chunk]
        chunk_speeches = [speech for _, _, _, speech in chunk]
        scores[start:start + len(chunk)] = agreement_scores(
            chunk_statements, chunk_speeches, tokenizer, model, device, max_length,
        )
    return scores


def write_stance_columns(df, pairs, scores, lang_variant, variant_indices):
    for (row_index, variant_index, _, _), score in zip(pairs, scores):
        df.at[row_index, f"stance_{lang_variant}_v{variant_index}"] = score
    per_variant_stance_columns = [f"stance_{lang_variant}_v{i}" for i in variant_indices]
    df[f"stance_{lang_variant}_mean"] = df[per_variant_stance_columns].mean(axis=1)


def reorder_stance_columns(df, text_prefix):
    text_column_start = f"{text_prefix}_"
    per_variant_stance = {
        column for column in df.columns
        if column.startswith("stance_") and not column.endswith("_mean")
    }
    ordered, placed = [], set()
    for column in df.columns:
        if column in per_variant_stance:
            continue
        ordered.append(column)
        if column.startswith(text_column_start):
            stance_column = f"stance_{column[len(text_column_start):]}"
            if stance_column in per_variant_stance:
                ordered.append(stance_column)
                placed.add(stance_column)
    ordered.extend(column for column in per_variant_stance if column not in placed)
    return df[ordered]


def score_text_agreement(df, languages, variant, source, tokenizer, model, device, max_length, batch_size):
    text_prefix = SOURCE_TEXT_COLUMN_PREFIX[source]
    for language in languages:
        lang_variant = f"{language}{variant}"
        if f"original_text_{lang_variant}" not in df.columns:
            print(f"[{lang_variant}] no statement column, skipping.")
            continue
        pairs, variant_indices = collect_statement_text_pairs(df, lang_variant, text_prefix, source)
        if not pairs:
            print(f"[{lang_variant}] no {source} texts found, skipping.")
            continue
        scores = score_pairs_in_batches(
            pairs, tokenizer, model, device, max_length, batch_size,
            description=f"scoring {lang_variant}",
        )
        write_stance_columns(df, pairs, scores, lang_variant, variant_indices)
    return reorder_stance_columns(df, SOURCE_TEXT_COLUMN_PREFIX[source])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossencoder_dir", default=DEFAULT_CROSSENCODER_DIR)
    parser.add_argument("--llm", default="qwen3.5-122b")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--source", default="speeches", choices=["speeches", "reasons"])
    parser.add_argument("--variant", default="", choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", action="store_true",
                        help="Recompute and overwrite an existing _scored file instead of skipping.")
    return parser.parse_args()


def resolve_input_path(args, languages):
    if args.input:
        return args.input
    filename = SOURCE_INPUT_FILENAME[args.source].format(
        langs=",".join(languages), variant=args.variant,
    )
    return f"./data/{args.dataset}_results/{args.llm}/{filename}"


def resolve_output_path(input_path):
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_scored{ext}"


def main():
    args = parse_args()
    languages = args.languages.split(",")
    input_path = resolve_input_path(args, languages)
    output_path = resolve_output_path(input_path)

    if os.path.exists(output_path) and not args.override:
        print(f"Output exists, skipping (use --override to recompute): {output_path}")
        return

    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig", low_memory=False)
    tokenizer, model, max_length = load_crossencoder(args.crossencoder_dir, args.device)

    df = score_text_agreement(
        df, languages, args.variant, args.source, tokenizer, model, args.device, max_length, args.batch_size,
    )

    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
