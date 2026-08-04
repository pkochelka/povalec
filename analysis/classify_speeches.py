#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, FAILED_REASON_VALUES, REFUSED_REASON_PREFIXES, SOURCE_INPUT_FILENAME, SOURCE_OUTPUT_FILENAME, SOURCE_TEXT_COLUMN_PREFIX, VARIANTS

DEFAULT_MODEL_DIR = "./mmBERT-base-balanced-collapsed"


@dataclass
class Classifier:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    labels: list
    temperature: float
    max_len: int
    biases: np.ndarray = None


def slugify_party_label(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


def load_classifier(model_dir, device):
    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    # This script reads the softmax head only. A sigmoid checkpoint carries
    # "task": "multi_label_classification" and would otherwise be misread as
    # softmax -- plausible-looking probabilities, silently wrong. The trainer
    # that produced such checkpoints is gone; archived ones may still exist.
    if manifest.get("task") == "multi_label_classification":
        raise SystemExit(
            f"{model_dir} is a multilabel checkpoint; this script only supports the "
            f"softmax head (e.g. {DEFAULT_MODEL_DIR})."
        )

    label2id = manifest["label2id"]
    labels = sorted(label2id, key=label2id.get)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    biases = manifest.get("biases")
    return Classifier(
        tokenizer=tokenizer,
        model=model,
        labels=labels,
        temperature=float(manifest.get("temperature", 1.0)),
        max_len=manifest["max_len"],
        biases=np.asarray(biases, dtype=np.float32) if biases is not None else None,
    )


@torch.no_grad()
def predict_class_logits(texts, classifier, device):
    inputs = classifier.tokenizer(
        texts,
        return_tensors="pt", truncation=True, padding=True,
        max_length=classifier.max_len,
    ).to(device)
    return classifier.model(**inputs).logits.float().cpu().numpy()


def is_classifiable_text(value, source):
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if source == "reasons":
        if text in FAILED_REASON_VALUES:
            return False
        if text.startswith(REFUSED_REASON_PREFIXES):
            return False
    return True


def discover_variant_indices(df, column_prefix):
    return sorted({
        int(column[len(column_prefix):])
        for column in df.columns
        if column.startswith(column_prefix)
        and column[len(column_prefix):].isdigit()
    })


def collect_texts_to_classify(df, lang_variant, source):
    text_column_kind = SOURCE_TEXT_COLUMN_PREFIX[source]
    column_prefix = f"{text_column_kind}_{lang_variant}_v"
    variant_indices = discover_variant_indices(df, column_prefix)
    texts_to_classify = []
    for row_index, row in df.iterrows():
        for variant_index in variant_indices:
            text = row.get(f"{text_column_kind}_{lang_variant}_v{variant_index}")
            if is_classifiable_text(text, source):
                texts_to_classify.append((row_index, variant_index, text.strip()))
    return texts_to_classify, variant_indices


def classify_in_batches(texts_to_classify, classifier, device, batch_size, progress_description):
    class_logits = np.zeros((len(texts_to_classify), len(classifier.labels)), dtype=np.float32)
    for batch_start in tqdm(range(0, len(texts_to_classify), batch_size), desc=progress_description):
        batch = texts_to_classify[batch_start:batch_start + batch_size]
        batch_texts = [text for _, _, text in batch]
        class_logits[batch_start:batch_start + len(batch)] = predict_class_logits(
            batch_texts, classifier, device,
        )
    return class_logits


def ensure_prediction_columns_exist(df, lang_variant, variant_indices, label_slugs):
    for variant_index in variant_indices:
        predicted_party_column = f"predicted_party_{lang_variant}_v{variant_index}"
        if predicted_party_column not in df.columns:
            df[predicted_party_column] = pd.Series(dtype="object")
        for slug in label_slugs:
            probability_column = f"party_prob_{slug}_{lang_variant}_v{variant_index}"
            if probability_column not in df.columns:
                df[probability_column] = np.nan


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def write_predictions(df, texts_to_classify, class_logits, lang_variant, classifier):
    label_slugs = [slugify_party_label(label) for label in classifier.labels]
    # The manifest's inference spec: softmax((logits + biases) / T) with the
    # dev-fitted uniform-marginal biases. The hard label is the argmax of the
    # calibrated probabilities.
    adjusted = class_logits if classifier.biases is None else class_logits + classifier.biases
    class_probabilities = softmax(adjusted / classifier.temperature)
    predicted_label_per_text = [classifier.labels[i] for i in class_probabilities.argmax(axis=1)]
    variant_indices = sorted({variant_index for _, variant_index, _ in texts_to_classify})
    ensure_prediction_columns_exist(df, lang_variant, variant_indices, label_slugs)

    for (row_index, variant_index, _), predicted_label, probabilities in zip(
        texts_to_classify, predicted_label_per_text, class_probabilities,
    ):
        df.at[row_index, f"predicted_party_{lang_variant}_v{variant_index}"] = predicted_label
        for slug, probability in zip(label_slugs, probabilities):
            df.at[row_index, f"party_prob_{slug}_{lang_variant}_v{variant_index}"] = float(probability)


def classify_all_languages(df, languages, variant, source, classifier, device, batch_size):
    for language in languages:
        lang_variant = f"{language}{variant}"
        texts_to_classify, _ = collect_texts_to_classify(df, lang_variant, source)
        if not texts_to_classify:
            print(f"[{lang_variant}] no usable {source} texts, skipping.")
            continue
        class_logits = classify_in_batches(
            texts_to_classify, classifier, device, batch_size,
            progress_description=f"classifying {source} {lang_variant}",
        )
        write_predictions(df, texts_to_classify, class_logits, lang_variant, classifier)
    return df


def resolve_input_and_output_paths(args, languages, source):
    user_override = args.speeches_input if source == "speeches" else args.reasons_input
    if user_override:
        stem, extension = os.path.splitext(user_override)
        return user_override, f"{stem}_classified{extension}"

    langs_in_filename = ",".join(languages)
    results_directory = f"./data/{args.dataset}_results/{args.llm}"
    input_path = os.path.join(results_directory, SOURCE_INPUT_FILENAME[source].format(
        langs=langs_in_filename, variant=args.variant,
    ))
    output_path = os.path.join(results_directory, SOURCE_OUTPUT_FILENAME[source].format(
        langs=langs_in_filename, variant=args.variant,
    ))
    return input_path, output_path


def classify_source(args, languages, source, classifier):
    input_path, output_path = resolve_input_and_output_paths(args, languages, source)
    if not os.path.exists(input_path):
        print(f"[{source}] input not found, skipping: {input_path}")
        return
    if os.path.exists(output_path) and not args.override:
        print(f"[{source}] output already exists, skipping (use --override to rerun): {output_path}")
        return

    print(f"[{source}] reading {input_path}")
    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    df = classify_all_languages(
        df, languages, args.variant, source, classifier, args.device, args.batch_size,
    )
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"[{source}] wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                         help=f"Softmax-head classifier checkpoint. Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--llm", default="qwen3.5-122b")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--source", default="both", choices=["both", "speeches", "reasons"])
    parser.add_argument("--speeches_input", default=None)
    parser.add_argument("--reasons_input", default=None)
    parser.add_argument("--variant", default="", choices=VARIANTS)
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", default=True, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages.split(",")
    sources_to_classify = ["speeches", "reasons"] if args.source == "both" else [args.source]

    print(f"--- Loading classifier from: {args.model_dir} ---")
    classifier = load_classifier(args.model_dir, args.device)
    head = (f"softmax, T={classifier.temperature:.4f}, "
            f"biases {'applied' if classifier.biases is not None else 'absent'}")
    print(f"Loaded model ({head}) | labels (index order): {classifier.labels}")

    for source in sources_to_classify:
        classify_source(args, languages, source, classifier)


if __name__ == "__main__":
    main()
