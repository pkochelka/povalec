#!/usr/bin/env python3
"""Train the multilabel mmBERT classifier on the ECR+ID-collapsed split track.

Same soft-label setup as classifier_training_multilabel, but on the 6 parties
from data/EuroParl Custom/collapsed. The euandi target matrix is likewise 6x6:
ECR and ID per-statement positions are first merged into one "ECR+ID" position
as a weighted average, weighted by the two groups' speech-count ratio in the
origin (cleaned, uncollapsed) train split — the same mix the merged class has
in training. Correlations are then per-statement-centered over the 6 collapsed
parties. Everything downstream (prior-adjusted BCE, per-head Platt scaling,
calibration reporting) is unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import PARTY_COLUMN, build_label_maps, load_split
from analysis.classifier_training import (
    EARLY_STOPPING_PATIENCE, MAX_EPOCHS, MAX_LEN, MODEL_NAME, MODEL_SLUG,
    release, select_device, training_arguments,
)
from analysis.classifier_training_multilabel import (
    BEST_METRIC, AdjustedBCETrainer, MultiLabelMetrics, TrainingData,
    attach_party, build_model, calibrated_probs, calibration_report, fit_platt,
    logit_adjustment_from, report_split, save_manifest, write_results_file,
)
from analysis.evaluate_euandi import PARTY_POSITIONS_PATH, load_party_positions

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "collapsed")
OUTPUT_DIR = f"{MODEL_SLUG}-multilabel-collapsed"
MERGE_PARTIES = ("ECR", "ID")
MERGED_LABEL = "ECR+ID"

# Speech counts in the uncollapsed cleaned/train.parquet; their ratio is the
# composition of the merged ECR+ID class in the collapsed train split.
ORIGIN_SPEECH_COUNTS = {"ECR": 112_366, "ID": 32_149}
MERGE_WEIGHTS = {p: n / sum(ORIGIN_SPEECH_COUNTS.values())
                 for p, n in ORIGIN_SPEECH_COUNTS.items()}


def euandi_targets(target_names, weights):
    positions = load_party_positions(os.path.join(PROJECT_ROOT, PARTY_POSITIONS_PATH))
    positions = positions.dropna(subset=["ep_group", "normalized_answer"])
    grid = (positions.groupby(["ep_group", "statement_idx"])["normalized_answer"]
            .mean().unstack())

    merged = sum(weights[p] * grid.loc[p] for p in MERGE_PARTIES)
    grid = grid.drop(index=list(MERGE_PARTIES))
    grid.loc[MERGED_LABEL] = merged
    assert sorted(grid.index) == target_names, "collapsed euandi EP groups != train parties"

    deviations = grid.loc[target_names] - grid.loc[target_names].mean(axis=0)
    corr = np.corrcoef(deviations.to_numpy())
    targets = (1.0 + corr) / 2.0
    np.fill_diagonal(targets, 1.0)
    return targets.astype(np.float32)


def save_targets(targets, target_names, weights):
    for i, name in enumerate(target_names):
        neighbours = sorted(
            ((target_names[j], targets[i, j]) for j in range(len(target_names)) if j != i),
            key=lambda kv: kv[1], reverse=True,
        )
        print(f"  {name}: " + ", ".join(f"{n}={t:.2f}" for n, t in neighbours))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "party_targets.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labels": target_names,
            "target_matrix": targets.tolist(),
            "merge_weights": weights,
            "source": "euandi_corr: (1 + rho) / 2 with rho the per-statement-centered "
                      "Pearson of party positions; 0.5 = unrelated, < 0.5 = opposed. "
                      "ECR and ID positions merged into ECR+ID as a speech-count-"
                      "weighted mean before centering/correlation",
        }, f, ensure_ascii=False, indent=2)


def prepare_training_data(data_dir, tokenizer, weights):
    raw = {name: load_split(name, data_dir) for name in ("train", "dev", "test")}

    label_list = sorted(raw["train"][PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    target_names = [id2label[i] for i in range(num_labels)]
    print(f"{num_labels} parties: {label2id}")

    splits = {name: attach_party(df, label2id) for name, df in raw.items()}
    print("Train party counts:\n", splits["train"]["party"].value_counts().sort_index().to_dict())

    targets = euandi_targets(target_names, weights)
    save_targets(targets, target_names, weights)

    tokenized = DatasetDict({
        name: Dataset.from_pandas(df, preserve_index=False) for name, df in splits.items()
    }).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True, remove_columns=["text", "language"],
    ).map(
        lambda batch: {"labels": [targets[p].tolist() for p in batch["party"]]},
        batched=True, remove_columns=["party"],
    )

    return TrainingData(
        train=tokenized["train"],
        dev=tokenized["dev"],
        test=tokenized["test"],
        collator=DataCollatorWithPadding(tokenizer=tokenizer),
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        target_names=target_names,
        targets=targets,
        logit_adjustment=logit_adjustment_from(
            targets, splits["train"]["party"].to_numpy(), num_labels
        ),
        dev_langs=splits["dev"]["language"].to_numpy(),
        test_langs=splits["test"]["language"].to_numpy(),
    )


def train_and_evaluate(data, tokenizer, hf_token, device):
    model = build_model(data, hf_token, device)
    metrics_fn = MultiLabelMetrics(data.targets)
    metrics_fn.current_languages = data.dev_langs

    trainer = AdjustedBCETrainer(
        logit_adjustment=torch.tensor(data.logit_adjustment, device=device),
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

    scored = [e for e in trainer.state.log_history if f"eval_{BEST_METRIC}" in e]
    best_epoch = max(1, round(max(scored, key=lambda e: e[f"eval_{BEST_METRIC}"])["epoch"]))
    print(f"Best dev {BEST_METRIC} at epoch {best_epoch}")

    dev_out = trainer.predict(data.dev)
    metrics_fn.current_languages = data.test_langs
    test_out = trainer.predict(data.test)
    trainer.save_model(OUTPUT_DIR)
    release(model, trainer, device)
    return best_epoch, dev_out, test_out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    print("Merge weights from origin speech counts: " + ", ".join(
        f"{p}={ORIGIN_SPEECH_COUNTS[p]:,} (w={w:.4f})" for p, w in MERGE_WEIGHTS.items()))
    data = prepare_training_data(args.data_dir, tokenizer, MERGE_WEIGHTS)

    best_epoch, dev_out, test_out = train_and_evaluate(data, tokenizer, hf_token, device)

    dev_true = dev_out.label_ids.argmax(axis=-1)
    scale, bias = fit_platt(dev_out.predictions, data.targets[dev_true])
    print("Platt per head: " + ", ".join(
        f"{n}: a={a:.3f} b={b:+.3f}" for n, a, b in zip(data.target_names, scale, bias)
    ))
    dev_probs = calibrated_probs(dev_out.predictions, scale, bias)
    test_probs = calibrated_probs(test_out.predictions, scale, bias)

    dev_report = report_split("dev", dev_probs, dev_out.label_ids, data.dev_langs, data.target_names)
    test_report = report_split("test", test_probs, test_out.label_ids, data.test_langs, data.target_names)

    test_true = test_out.label_ids.argmax(axis=-1)
    calibration = calibration_report(test_probs, test_true, data.targets, data.target_names)
    print("\n" + calibration)

    save_manifest(OUTPUT_DIR, data, best_epoch, scale, bias)
    output_tag = f"{MODEL_SLUG}_multilabel_collapsed_ep{best_epoch}_len{MAX_LEN}"
    write_results_file(output_tag, best_epoch, dev_report, test_report, calibration)


if __name__ == "__main__":
    main()
