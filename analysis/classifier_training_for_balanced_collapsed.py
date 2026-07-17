#!/usr/bin/env python3
"""Train an mmBERT EU-party classifier on the balanced ECR+ID-collapsed track.

Same setup as classifier_training_for_balanced, but over the 6 collapsed
parties: train is data/EuroParl Custom/collapsed/train_balanced.parquet
(party- and language-balanced, ECR and ID merged into "ECR+ID"), dev and test
are the rebalanced uniform collapsed splits from the same directory. All
three splits are uniform across parties, so the model is trained with plain
cross-entropy and no imbalance correction.

The model is then calibrated on dev only, in two steps applied at inference
as probs = softmax((logits + biases) / temperature):
  1. per-class additive biases flatten the argmax marginal to uniform,
     undoing any residual class lean (see classifier_training.py);
  2. a single softmax temperature (Guo et al. 2017, NLL-minimised as in
     recalibrate_temperature.py) rescales the probabilities so each class's
     mean probability matches its argmax share as closely as possible.
Dev and test are scored through the identical argmax(logits + biases) path.
"""
import json
import os
import sys

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict
from dotenv import load_dotenv
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import (
    PARTY_COLUMN,
    attach_labels,
    build_label_maps,
    fit_uniform_bias,
    load_split,
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
    TrainingData,
    build_model,
    release,
    select_device,
    training_arguments,
)
from analysis.classifier_training_for_balanced import report_split
from analysis.recalibrate_temperature import calibration_metrics, fit_temperature, softmax_np

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "collapsed")
OUTPUT_DIR = f"{MODEL_SLUG}-balanced-collapsed"


def prepare_training_data(data_dir, tokenizer):
    raw_splits = {
        "train": load_split("train_balanced", data_dir),
        "dev": load_split("dev", data_dir),
        "test": load_split("test", data_dir),
    }

    label_list = sorted(raw_splits["train"][PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    print(f"{num_labels} classes: {label2id}")

    splits = {name: attach_labels(df, label2id) for name, df in raw_splits.items()}
    print("Train class counts:\n", splits["train"]["labels"].value_counts().sort_index().to_dict())
    print("Train languages:\n",    splits["train"]["language"].value_counts().to_dict())

    dataset = DatasetDict({
        "train":      Dataset.from_pandas(splits["train"], preserve_index=False),
        "validation": Dataset.from_pandas(splits["dev"],   preserve_index=False),
        "test":       Dataset.from_pandas(splits["test"],  preserve_index=False),
    })
    for split in dataset:
        dataset[split] = dataset[split].cast_column("labels", ClassLabel(num_classes=num_labels))
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text", "language"],
    )

    return TrainingData(
        train=tokenized["train"],
        dev=tokenized["validation"],
        test=tokenized["test"],
        collator=DataCollatorWithPadding(tokenizer=tokenizer),
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        target_names=[id2label[i] for i in range(num_labels)],
        dev_langs=splits["dev"]["language"].to_numpy(),
        test_langs=splits["test"]["language"].to_numpy(),
    )


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
    trainer.train()

    scored = [entry for entry in trainer.state.log_history if f"eval_{BEST_METRIC}" in entry]
    best_epoch = max(1, round(max(scored, key=lambda e: e[f"eval_{BEST_METRIC}"])["epoch"]))
    print(f"Best dev {BEST_METRIC} at epoch {best_epoch}")

    # Both calibration steps are fit on dev only; the temperature is fit on the
    # already bias-adjusted logits, matching the inference formula.
    metrics_fn.current_languages = data.dev_langs
    dev_out = trainer.predict(data.dev)
    biases = fit_uniform_bias(dev_out.predictions, data.num_labels)
    print("Uniform-marginal bias (fit on dev): "
          + ", ".join(f"{name}={b:+.3f}" for name, b in zip(data.target_names, biases)))
    temperature = fit_temperature(dev_out.predictions + biases, dev_out.label_ids, device)
    print(f"Temperature (fit on dev): {temperature:.4f}")

    metrics_fn.current_languages = data.test_langs
    test_out = trainer.predict(data.test)
    trainer.save_model(OUTPUT_DIR)
    release(model, trainer, device)
    return best_epoch, biases, temperature, dev_out, test_out


def calibration_summary(name, logits, y_true, biases, temperature, num_labels, target_names):
    """Per-class argmax share vs mean calibrated probability, plus ECE/NLL/Brier."""
    adjusted = logits + biases
    probs = softmax_np(adjusted / temperature)
    shares = np.bincount(adjusted.argmax(axis=-1), minlength=num_labels) / len(adjusted)
    mean_probs = probs.mean(axis=0)
    metrics = calibration_metrics(probs, y_true, num_labels)

    lines = [f"=== CALIBRATION ({name}) ===",
             f"temperature={temperature:.4f}  ece={metrics['ece']:.4f}  "
             f"nll={metrics['nll']:.4f}  brier={metrics['brier']:.4f}",
             f"{'party':<12}{'argmax_share':>13}{'mean_prob':>11}{'gap':>9}"]
    for c, party in enumerate(target_names):
        lines.append(f"{party:<12}{shares[c]:>13.4f}{mean_probs[c]:>11.4f}"
                     f"{mean_probs[c] - shares[c]:>+9.4f}")
    lines.append(f"max |gap|: {np.abs(mean_probs - shares).max():.4f}")
    return "\n".join(lines)


def save_manifest(output_dir, data, num_epochs, biases, temperature):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_epochs": num_epochs,
        "label2id": data.label2id,
        "biases": [float(b) for b in biases],
        "temperature": float(temperature),
        "inference": "probs = softmax((model logits + biases) / temperature); "
                     "argmax for the hard label",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_results_file(output_tag, num_epochs, dev_report, test_report, calibration):
    dev_f1, dev_overall, dev_summary, dev_per_language = dev_report
    test_f1, test_overall, test_summary, test_per_language = test_report
    path = f"results_{output_tag}.txt"
    with open(path, "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Best epoch from dev search: {num_epochs}\n")
        f.write(f"Dev f1_macro:  {dev_f1:.4f}\n")
        f.write(f"Test f1_macro: {test_f1:.4f}\n\n")
        f.write(calibration + "\n\n")
        f.write("=== DEV ===\n" + dev_overall + dev_summary + "\n".join(dev_per_language) + "\n\n")
        f.write("=== TEST ===\n" + test_overall + test_summary + "\n".join(test_per_language))
    print(f"\nSaved {path}")


def main():
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    data = prepare_training_data(DATA_DIR, tokenizer)

    best_epoch, biases, temperature, dev_out, test_out = train_and_evaluate(
        data, tokenizer, hf_token, device)

    dev_pred = (dev_out.predictions + biases).argmax(axis=-1)
    test_pred = (test_out.predictions + biases).argmax(axis=-1)

    dev_report = report_split("dev", dev_out.label_ids, dev_pred, data.dev_langs, data.target_names)
    test_report = report_split("test", test_out.label_ids, test_pred, data.test_langs, data.target_names)

    calibration = "\n\n".join([
        calibration_summary("dev", dev_out.predictions, dev_out.label_ids,
                            biases, temperature, data.num_labels, data.target_names),
        calibration_summary("test", test_out.predictions, test_out.label_ids,
                            biases, temperature, data.num_labels, data.target_names),
    ])
    print("\n" + calibration)

    save_manifest(OUTPUT_DIR, data, best_epoch, biases, temperature)
    output_tag = f"{MODEL_SLUG}_balanced_collapsed_ep{best_epoch}_len{MAX_LEN}"
    write_results_file(output_tag, best_epoch, dev_report, test_report, calibration)


if __name__ == "__main__":
    main()
