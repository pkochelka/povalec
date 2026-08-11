#!/usr/bin/env python3
"""Train an mmBERT EU-party classifier on the balanced ECR+ID-collapsed track.

Over the 6 collapsed parties: train is
data/EuroParl Custom/collapsed/train_balanced.parquet (party- and
language-balanced, ECR and ID merged into "ECR+ID"), dev and test are the uniform
collapsed splits from the same directory. All three splits are uniform across
parties, so the model is trained with plain cross-entropy, and dev and test are
scored through the identical plain argmax(logits) path -- no bias or temperature
correction. That keeps the two numbers directly comparable to each other: on a
balanced train the label prior is already uniform, so argmax(logits) is
Bayes-optimal for the uniform dev/test and a prior-shift correction is neither
needed nor principled.

The sibling classifier_training.py trains the same 6 classes on the same track
with logit-adjusted CE off the natural-prior collapsed/train.parquet instead.
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analysis.europarl_classification import (
    classification_report_text,
    per_language_f1_report,
)
from analysis.classifier_training import (
    BEST_METRIC,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MAX_LEN,
    MODEL_NAME,
    MODEL_SLUG,
    SEED,
    LanguageAwareMetrics,
    build_model,
    prepare_training_data,
    release,
    select_device,
    training_arguments,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "collapsed")
OUTPUT_DIR = f"{MODEL_SLUG}-balanced-collapsed"


def predict_labels(trainer, dataset, languages, metrics_fn):
    metrics_fn.current_languages = languages
    output = trainer.predict(dataset)
    return output.label_ids, np.argmax(output.predictions, axis=-1)


def report_split(name, y_true, y_pred, languages, target_names):
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    overall = classification_report_text(y_true, y_pred, target_names)
    summary, per_language = per_language_f1_report(y_true, y_pred, languages, target_names, note=name)
    print(f"\n=== {name.upper()} (f1_macro={f1:.4f}) ===\n" + overall + summary)
    print("\n".join(per_language))
    return f1, overall, summary, per_language


def train_and_evaluate(data, tokenizer, hf_token, device):
    model = build_model(data, hf_token, device)
    metrics_fn = LanguageAwareMetrics()
    metrics_fn.current_languages = data.dev_langs

    trainer = Trainer(
        model=model,
        args=training_arguments(OUTPUT_DIR, MAX_EPOCHS, evaluate_each_epoch=True),
        train_dataset=data.train,
        eval_dataset=data.dev,
        data_collator=data.collator,
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()  # load_best_model_at_end restores the best-dev checkpoint

    scored = [entry for entry in trainer.state.log_history if f"eval_{BEST_METRIC}" in entry]
    best_epoch = max(1, round(max(scored, key=lambda e: e[f"eval_{BEST_METRIC}"])["epoch"]))
    print(f"Best dev {BEST_METRIC} at epoch {best_epoch}")

    dev = predict_labels(trainer, data.dev, data.dev_langs, metrics_fn)
    test = predict_labels(trainer, data.test, data.test_langs, metrics_fn)
    trainer.save_model(OUTPUT_DIR)
    release(model, trainer, device)
    return best_epoch, dev, test


def save_manifest(output_dir, data, num_epochs):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_epochs": num_epochs,
        "label2id": data.label2id,
        "inference": "argmax(model logits)",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_results_file(output_tag, num_epochs, dev_report, test_report):
    dev_f1, dev_overall, dev_summary, dev_per_language = dev_report
    test_f1, test_overall, test_summary, test_per_language = test_report
    path = f"results_{output_tag}.txt"
    with open(path, "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Best epoch from dev search: {num_epochs}\n")
        f.write(f"Dev f1_macro:  {dev_f1:.4f}\n")
        f.write(f"Test f1_macro: {test_f1:.4f}\n\n")
        f.write("=== DEV ===\n" + dev_overall + dev_summary + "\n".join(dev_per_language) + "\n\n")
        f.write("=== TEST ===\n" + test_overall + test_summary + "\n".join(test_per_language))
    print(f"\nSaved {path}")


def main():
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    data = prepare_training_data(DATA_DIR, tokenizer, train_split="train_balanced")

    best_epoch, dev, test = train_and_evaluate(data, tokenizer, hf_token, device)

    dev_report = report_split("dev", *dev, data.dev_langs, data.target_names)
    test_report = report_split("test", *test, data.test_langs, data.target_names)

    save_manifest(OUTPUT_DIR, data, best_epoch)
    output_tag = f"{MODEL_SLUG}_balanced_collapsed_ep{best_epoch}_len{MAX_LEN}"
    write_results_file(output_tag, best_epoch, dev_report, test_report)


if __name__ == "__main__":
    main()
