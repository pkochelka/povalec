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

from utils import ALL_LANGS_STR

DEFAULT_MULTILABEL_MODEL_DIR = "./mmBERT-base-multilabel-collapsed"
DEFAULT_LOGITADJ_MODEL_DIR = "./mmBERT-base-balanced-collapsed"
REFUSED_REASON_PREFIXES = ("REFUSED",)
FAILED_REASON_VALUES = {"FAILED"}

SOURCE_TEXT_COLUMN_PREFIX = {"speeches": "answer", "reasons": "reason"}
SOURCE_INPUT_FILENAME = {
    "speeches": "speeches_{langs}{variant}.csv",
    "reasons": "{langs}{variant}.csv",
}
SOURCE_OUTPUT_FILENAME = {
    "speeches": "speeches_{langs}{variant}_classified.csv",
    "reasons": "{langs}{variant}_classified.csv",
}


@dataclass
class Classifier:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    labels: list
    temperature: float
    max_len: int
    multilabel: bool = False
    platt_scale: np.ndarray = None
    platt_bias: np.ndarray = None
    biases: np.ndarray = None


def slugify_party_label(label):
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


def load_classifier(model_dir, device):
    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    label2id = manifest["label2id"]
    labels = sorted(label2id, key=label2id.get)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    platt_scale = manifest.get("platt_scale")
    platt_bias = manifest.get("platt_bias")
    biases = manifest.get("biases")
    return Classifier(
        tokenizer=tokenizer,
        model=model,
        labels=labels,
        temperature=float(manifest.get("temperature", 1.0)),
        max_len=manifest["max_len"],
        # The head type comes from the manifest, not a CLI flag: only the
        # multilabel trainers write the "task" key.
        multilabel=manifest.get("task") == "multi_label_classification",
        platt_scale=np.asarray(platt_scale, dtype=np.float32) if platt_scale is not None else None,
        platt_bias=np.asarray(platt_bias, dtype=np.float32) if platt_bias is not None else None,
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


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def write_predictions(df, texts_to_classify, class_logits, lang_variant, classifier):
    label_slugs = [slugify_party_label(label) for label in classifier.labels]
    # Each head follows its manifest's inference spec. Multilabel: per-head
    # Platt scaling when present (score_c = sigmoid(scale_c * logit_c +
    # bias_c), the calibrated (1+rho)/2 estimate), else sigmoid(logits / T).
    # Softmax: softmax((logits + biases) / T) with the dev-fitted uniform-
    # marginal biases. The hard label is the argmax of the calibrated
    # probabilities.
    if classifier.multilabel:
        if classifier.platt_scale is not None:
            class_probabilities = sigmoid(
                class_logits * classifier.platt_scale + classifier.platt_bias
            )
        else:
            class_probabilities = sigmoid(class_logits / classifier.temperature)
    else:
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
    parser.add_argument("--multilabel", default=True, action=argparse.BooleanOptionalAction,
                         help="Only selects the default --model_dir (--no-multilabel for the "
                              "softmax one); the head type itself is read from the manifest")
    parser.add_argument("--model_dir", default=None,
                         help=f"Defaults to {DEFAULT_MULTILABEL_MODEL_DIR} with --multilabel, "
                              f"else {DEFAULT_LOGITADJ_MODEL_DIR}")
    parser.add_argument("--llm", default="qwen3.5-122b")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--source", default="both", choices=["both", "speeches", "reasons"])
    parser.add_argument("--speeches_input", default=None)
    parser.add_argument("--reasons_input", default=None)
    parser.add_argument("--variant", default="", choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", default=True, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages.split(",")
    sources_to_classify = ["speeches", "reasons"] if args.source == "both" else [args.source]
    model_dir = args.model_dir or (
        DEFAULT_MULTILABEL_MODEL_DIR if args.multilabel else DEFAULT_LOGITADJ_MODEL_DIR
    )

    print(f"--- Loading classifier from: {model_dir} ---")
    classifier = load_classifier(model_dir, args.device)
    if classifier.multilabel:
        head = ("multilabel sigmoid, per-head Platt" if classifier.platt_scale is not None
                else f"multilabel sigmoid, T={classifier.temperature:.4f}")
    else:
        head = (f"softmax, T={classifier.temperature:.4f}, "
                f"biases {'applied' if classifier.biases is not None else 'absent'}")
    print(f"Loaded model ({head}) | labels (index order): {classifier.labels}")

    for source in sources_to_classify:
        classify_source(args, languages, source, classifier)


if __name__ == "__main__":
    main()
