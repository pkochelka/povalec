#!/usr/bin/env python3
"""Train a multilabel mmBERT EU-party classifier with euandi-derived soft labels.

Each speech keeps its single gold EU Party, but the training target is soft: 1.0
for the gold party and, for every other party, (1 + rho) / 2 where rho is the
euandi 2024 ideological correlation with the gold party (per-statement-centered
Pearson of the party positions). 0.5 means unrelated, below 0.5 means opposed,
so no information is clipped away. Sigmoid heads are trained with prior-adjusted
BCE; per-head Platt scaling fitted on the uniform dev split then makes the scores
commensurable: s_c estimates (1 + rho) / 2 for party c on the same scale for every
head, and 2 * s_c - 1 reads directly as the signed estimated euandi correlation.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.europarl_classification import (
    MIN_LANGUAGE_SAMPLES, PARTY_COLUMN, build_label_maps,
    classification_report_text, load_split, per_language_f1_report,
)
from analysis.classifier_training import (
    EARLY_STOPPING_PATIENCE, MAX_EPOCHS, MAX_LEN, MODEL_NAME, MODEL_SLUG, SEED,
    release, select_device, training_arguments,
)
from analysis.evaluate_euandi import PARTY_POSITIONS_PATH, load_party_positions

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom")
OUTPUT_DIR = f"{MODEL_SLUG}-multilabel"
BEST_METRIC = "f1_macro_mean_lang"
RELIABILITY_BINS = 10


@dataclass
class TrainingData:
    train: Dataset
    dev: Dataset
    test: Dataset
    collator: DataCollatorWithPadding
    label2id: dict
    id2label: dict
    num_labels: int
    target_names: list
    targets: np.ndarray
    logit_adjustment: np.ndarray
    dev_langs: np.ndarray
    test_langs: np.ndarray


class MultiLabelMetrics:
    def __init__(self, targets):
        self.targets = targets
        self.current_languages = None

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        y_true = np.asarray(labels).argmax(axis=-1)
        preds = logits.argmax(axis=-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        gold_prob = probs[np.arange(len(probs)), y_true]
        row_rhos = [spearmanr(p, t)[0] for p, t in zip(probs, self.targets[y_true])]

        results = {
            "accuracy": float((preds == y_true).mean()),
            "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, preds, average="weighted", zero_division=0),
            "mean_gold_prob": float(gold_prob.mean()),
            "spearman_targets": float(np.nanmean(row_rhos)),
        }
        if self.current_languages is None:
            return results

        f1_by_language = {}
        for language in np.unique(self.current_languages):
            mask = self.current_languages == language
            if mask.sum() < MIN_LANGUAGE_SAMPLES:
                continue
            f1_by_language[language] = f1_score(
                y_true[mask], preds[mask], average="macro", zero_division=0
            )
            results[f"f1_macro_{language}"] = f1_by_language[language]
        if f1_by_language:
            results["f1_macro_mean_lang"] = float(np.mean(list(f1_by_language.values())))
            results["f1_macro_min_lang"] = float(min(f1_by_language.values()))
        return results


class AdjustedBCETrainer(Trainer):
    def __init__(self, logit_adjustment, **kwargs):
        super().__init__(**kwargs)
        self.logit_adjustment = logit_adjustment

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(
            outputs.logits + self.logit_adjustment, labels
        )
        return (loss, outputs) if return_outputs else loss


def attach_party(df, label2id):
    df = df[df[PARTY_COLUMN].isin(label2id)].copy()
    df["party"] = df[PARTY_COLUMN].map(label2id).astype(int)
    return df[["text", "party", "language"]].reset_index(drop=True)


def euandi_targets(target_names):
    positions = load_party_positions(os.path.join(PROJECT_ROOT, PARTY_POSITIONS_PATH))
    positions = positions.dropna(subset=["ep_group", "normalized_answer"])
    grid = (positions.groupby(["ep_group", "statement_idx"])["normalized_answer"]
            .mean().unstack())
    assert sorted(grid.index) == target_names, "euandi EP groups != train parties"

    deviations = grid.loc[target_names] - grid.loc[target_names].mean(axis=0)
    corr = np.corrcoef(deviations.to_numpy())
    targets = (1.0 + corr) / 2.0
    np.fill_diagonal(targets, 1.0)
    return targets.astype(np.float32)


def save_targets(targets, target_names):
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
            "source": "euandi_corr: (1 + rho) / 2 with rho the per-statement-centered "
                      "Pearson of party positions; 0.5 = unrelated, < 0.5 = opposed",
        }, f, ensure_ascii=False, indent=2)


def logit_adjustment_from(targets, party_ids, num_labels):
    priors = np.bincount(party_ids, minlength=num_labels) / len(party_ids)
    positive_rate = priors @ targets
    return np.log(positive_rate / (1.0 - positive_rate)).astype(np.float32)


def prepare_training_data(data_dir, tokenizer):
    raw = {name: load_split(name, data_dir) for name in ("train", "dev", "test")}

    label_list = sorted(raw["train"][PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    target_names = [id2label[i] for i in range(num_labels)]
    print(f"{num_labels} parties: {label2id}")

    splits = {name: attach_party(df, label2id) for name, df in raw.items()}
    print("Train party counts:\n", splits["train"]["party"].value_counts().sort_index().to_dict())

    targets = euandi_targets(target_names)
    save_targets(targets, target_names)

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


def build_model(data, hf_token, device):
    set_seed(SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token,
        num_labels=data.num_labels, id2label=data.id2label, label2id=data.label2id,
        problem_type="multi_label_classification",
    ).to(device)


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


def fit_platt(dev_logits, dev_soft_labels):
    logits = torch.tensor(dev_logits, dtype=torch.float32)
    labels = torch.tensor(dev_soft_labels, dtype=torch.float32)
    scale = torch.ones(logits.shape[1], requires_grad=True)
    bias = torch.zeros(logits.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([scale, bias], lr=1.0, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits * scale + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scale.detach().numpy(), bias.detach().numpy()


def calibrated_probs(logits, scale, bias):
    return 1.0 / (1.0 + np.exp(-(logits * scale + bias)))


def calibration_report(probs, y_true, targets, target_names):
    soft = targets[y_true]
    lines = ["=== CALIBRATION vs euandi targets, per head (test) ===",
             f"{'party':<12}{'soft_ECE':>9}{'mean_pred':>10}{'mean_target':>12}"]
    for c, name in enumerate(target_names):
        p, t = probs[:, c], soft[:, c]
        bins = np.clip((p * RELIABILITY_BINS).astype(int), 0, RELIABILITY_BINS - 1)
        ece = sum(
            (bins == b).mean() * abs(p[bins == b].mean() - t[bins == b].mean())
            for b in range(RELIABILITY_BINS) if (bins == b).any()
        )
        lines.append(f"{name:<12}{ece:>9.4f}{p.mean():>10.4f}{t.mean():>12.4f}")

    lines.append("\n=== NEIGHBOUR STRUCTURE (mean calibrated scores vs target row) ===")
    rhos = []
    for g, name in enumerate(target_names):
        head_mask = np.arange(len(target_names)) != g
        mean_scores = probs[y_true == g].mean(axis=0)
        rho = spearmanr(mean_scores[head_mask], targets[g][head_mask])[0]
        rhos.append(rho)
        ranking = sorted(
            ((target_names[j], mean_scores[j]) for j in np.flatnonzero(head_mask)),
            key=lambda kv: kv[1], reverse=True,
        )
        lines.append(f"{name:<12}rho={rho:+.3f}  "
                     + ", ".join(f"{n}={s:.2f}" for n, s in ranking))
    lines.append(f"mean rho: {np.nanmean(rhos):+.3f}")
    return "\n".join(lines)


def report_split(name, probs, label_ids, languages, target_names):
    y_true = label_ids.argmax(axis=-1)
    y_pred = probs.argmax(axis=-1)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    overall = classification_report_text(y_true, y_pred, target_names)
    summary, per_language = per_language_f1_report(y_true, y_pred, languages, target_names, note=name)
    print(f"\n=== {name.upper()} (f1_macro={f1:.4f}) ===\n" + overall + summary)
    print("\n".join(per_language))
    return f1, overall, summary, per_language


def save_manifest(output_dir, data, num_epochs, scale, bias):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_epochs": num_epochs,
        "task": "multi_label_classification",
        "label2id": data.label2id,
        "targets": "euandi_corr as (1 + rho) / 2; matrix in party_targets.json",
        "logit_adjustment": [float(a) for a in data.logit_adjustment],
        "platt_scale": [float(s) for s in scale],
        "platt_bias": [float(b) for b in bias],
        "inference": "score_c = sigmoid(platt_scale[c] * logit_c + platt_bias[c]); "
                     "estimates (1 + rho) / 2, so 2 * score_c - 1 is the signed euandi "
                     "correlation to party c; argmax for the hard label",
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
    data = prepare_training_data(args.data_dir, tokenizer)

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
    output_tag = f"{MODEL_SLUG}_multilabel_ep{best_epoch}_len{MAX_LEN}"
    write_results_file(output_tag, best_epoch, dev_report, test_report, calibration)


if __name__ == "__main__":
    main()
