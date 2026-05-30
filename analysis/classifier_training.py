#!/usr/bin/env python3
"""Train an mmBERT EU-party classifier that is robust to class imbalance.

Three techniques combine to lift minority-class and per-language macro-F1:
logit-adjusted cross-entropy during training, a multi-seed softmax ensemble,
and post-hoc per-class bias tuning on the dev set.
"""
import os
import sys
import json
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.europarl_classification import (
    MIN_LANGUAGE_SAMPLES,
    PARTY_COLUMN,
    attach_labels,
    build_label_maps,
    classification_report_text,
    load_split,
    per_language_f1_report,
)

MODEL_NAME = "jhu-clsp/mmBERT-base"
MODEL_SLUG = MODEL_NAME.split("/")[-1]
DATA_DIR = os.path.expanduser("~/data/Europarl Custom")
ENSEMBLE_DIR = f"{MODEL_SLUG}-logitadj-ensemble"
MAX_LEN = 512
SEEDS = [42, 1337, 2024]
NUM_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2
LOGIT_ADJUSTMENT_TAU = 1.0
EPSILON = 1e-12


@dataclass
class TrainingData:
    tokenized: DatasetDict
    collator: DataCollatorWithPadding
    label2id: dict
    id2label: dict
    num_labels: int
    target_names: list
    val_langs: np.ndarray
    test_langs: np.ndarray
    logit_adjustment: torch.Tensor


@dataclass
class RunSummary:
    per_seed_f1: list
    ensemble_f1_raw: float
    dev_f1_raw: float
    dev_f1_tuned: float
    ensemble_f1_tuned: float
    biases: np.ndarray
    overall_report: str
    language_summary: str
    per_language_reports: list


class LanguageAwareMetrics:
    def __init__(self):
        self.current_languages = None

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        top2 = np.argsort(logits, axis=-1)[:, -2:]

        results = {
            "accuracy":      float((preds == labels).mean()),
            "top2_accuracy": float(np.mean([label in t for label, t in zip(labels, top2)])),
            "f1_macro":      f1_score(labels, preds, average="macro",    zero_division=0),
            "f1_weighted":   f1_score(labels, preds, average="weighted", zero_division=0),
        }

        if self.current_languages is None:
            return results

        f1_by_language = {}
        for language in np.unique(self.current_languages):
            mask = self.current_languages == language
            if mask.sum() < MIN_LANGUAGE_SAMPLES:
                continue
            f1_by_language[language] = f1_score(
                labels[mask], preds[mask], average="macro", zero_division=0
            )
            results[f"f1_macro_{language}"] = f1_by_language[language]

        if f1_by_language:
            results["f1_macro_mean_lang"]     = float(np.mean(list(f1_by_language.values())))
            results["f1_macro_min_lang"]      = float(min(f1_by_language.values()))
            results["f1_macro_min_lang_name"] = min(f1_by_language, key=f1_by_language.get)

        return results


class LogitAdjustedTrainer(Trainer):
    def __init__(self, logit_adjustment, val_langs, test_langs, metrics_fn, **kwargs):
        super().__init__(**kwargs)
        self.logit_adjustment = logit_adjustment
        self.val_langs = val_langs
        self.test_langs = test_langs
        self.metrics_fn = metrics_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        adjusted_logits = outputs.logits + self.logit_adjustment
        loss = F.cross_entropy(
            adjusted_logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.metrics_fn.current_languages = (
            self.test_langs if metric_key_prefix == "test" else self.val_langs
        )
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        self.metrics_fn.current_languages = (
            self.val_langs if metric_key_prefix == "dev" else self.test_langs
        )
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)


def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def compute_logit_adjustment(labels, num_labels, tau, device):
    counts = np.bincount(labels, minlength=num_labels)
    priors = counts / counts.sum()
    return torch.tensor(tau * np.log(priors + EPSILON), dtype=torch.float, device=device)


def prepare_training_data(data_dir, tokenizer, device):
    raw_splits = {name: load_split(name, data_dir) for name in ("train", "dev", "test")}

    label_list = sorted(raw_splits["train"][PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    print(f"{num_labels} classes: {label2id}")

    splits = {name: attach_labels(df, label2id) for name, df in raw_splits.items()}
    print("Train class counts:\n", splits["train"]["labels"].value_counts().sort_index().to_dict())
    print("Train languages:\n",    splits["train"]["language"].value_counts().to_dict())

    logit_adjustment = compute_logit_adjustment(
        splits["train"]["labels"].to_numpy(), num_labels, LOGIT_ADJUSTMENT_TAU, device,
    )
    print(f"Logit adjustment (tau={LOGIT_ADJUSTMENT_TAU}): {logit_adjustment.cpu().numpy().round(3)}")

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
        tokenized=tokenized,
        collator=DataCollatorWithPadding(tokenizer=tokenizer),
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        target_names=[id2label[i] for i in range(num_labels)],
        val_langs=splits["dev"]["language"].to_numpy(),
        test_langs=splits["test"]["language"].to_numpy(),
        logit_adjustment=logit_adjustment,
    )


def train_one_seed(seed, data, tokenizer, hf_token, device):
    print(f"\n{'=' * 60}\n  Training seed={seed}\n{'=' * 60}")
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token,
        num_labels=data.num_labels, id2label=data.id2label, label2id=data.label2id,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"{MODEL_SLUG}-logitadj-s{seed}",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro_mean_lang",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        seed=seed,
    )

    metrics_fn = LanguageAwareMetrics()
    trainer = LogitAdjustedTrainer(
        logit_adjustment=data.logit_adjustment,
        val_langs=data.val_langs,
        test_langs=data.test_langs,
        metrics_fn=metrics_fn,
        model=model,
        args=training_args,
        train_dataset=data.tokenized["train"],
        eval_dataset=data.tokenized["validation"],
        data_collator=data.collator,
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()
    test_predictions = trainer.predict(data.tokenized["test"])
    dev_predictions = trainer.predict(data.tokenized["validation"], metric_key_prefix="dev")

    trainer.save_model(os.path.join(ENSEMBLE_DIR, f"seed-{seed}"))
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_predictions, dev_predictions


def softmax_probabilities(predictions):
    return torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()


def collect_ensemble_predictions(seeds, data, tokenizer, hf_token, device):
    test_probs_per_seed = []
    dev_probs_per_seed = []
    y_test = None
    y_dev = None
    for seed in seeds:
        test_predictions, dev_predictions = train_one_seed(seed, data, tokenizer, hf_token, device)
        test_probs_per_seed.append(softmax_probabilities(test_predictions))
        dev_probs_per_seed.append(softmax_probabilities(dev_predictions))
        y_test = test_predictions.label_ids
        y_dev = dev_predictions.label_ids
    return test_probs_per_seed, dev_probs_per_seed, y_test, y_dev


def report_per_seed(seeds, test_probs_per_seed, y_test):
    print("\n=== PER-SEED TEST (sanity check) ===")
    per_seed_f1 = []
    for seed, probs in zip(seeds, test_probs_per_seed):
        f1 = f1_score(y_test, np.argmax(probs, axis=-1), average="macro", zero_division=0)
        print(f"  seed={seed}: f1_macro={f1:.4f}")
        per_seed_f1.append((seed, f1))
    return per_seed_f1


def tune_per_class_biases(probs, labels, max_passes=20, grid_size=41, grid_range=3.0,
                          min_improvement=1e-5):
    log_probs = np.log(probs + EPSILON)
    biases = np.zeros(probs.shape[1])
    grid = np.linspace(-grid_range, grid_range, grid_size)

    def macro_f1(candidate_biases):
        predictions = np.argmax(log_probs + candidate_biases, axis=-1)
        return f1_score(labels, predictions, average="macro", zero_division=0)

    best_f1 = macro_f1(biases)
    for _ in range(max_passes):
        f1_before_pass = best_f1
        for class_index in range(probs.shape[1]):
            best_bias = biases[class_index]
            for candidate in grid:
                biases[class_index] = candidate
                f1 = macro_f1(biases)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = candidate
            biases[class_index] = best_bias
        if best_f1 - f1_before_pass < min_improvement:
            break
    return biases, best_f1


def save_ensemble(ensemble_dir, data, biases):
    os.makedirs(ensemble_dir, exist_ok=True)
    manifest = {
        "model_name": MODEL_NAME,
        "seeds": SEEDS,
        "seed_model_dirs": [f"seed-{seed}" for seed in SEEDS],
        "max_len": MAX_LEN,
        "logit_adjustment_tau": LOGIT_ADJUSTMENT_TAU,
        "label2id": data.label2id,
        "biases": biases.tolist(),
        "inference": "mean softmax across seed models, then argmax(log(mean_prob + 1e-12) + biases)",
    }
    with open(os.path.join(ensemble_dir, "ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved ensemble to {ensemble_dir}/ ({len(SEEDS)} models + ensemble.json)")


def write_results_file(output_tag, summary):
    path = f"results_{output_tag}.txt"
    with open(path, "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(f"Ensemble seeds: {SEEDS}\n")
        f.write(f"Logit adjustment tau: {LOGIT_ADJUSTMENT_TAU}\n")
        f.write("Per-seed test f1_macro: "
                + ", ".join(f"s{seed}={f1:.4f}" for seed, f1 in summary.per_seed_f1) + "\n")
        f.write(f"Ensemble test f1_macro (no bias): {summary.ensemble_f1_raw:.4f}\n")
        f.write(f"Ensemble dev  f1_macro: {summary.dev_f1_raw:.4f} -> "
                f"{summary.dev_f1_tuned:.4f} (with biases)\n")
        f.write(f"Ensemble test f1_macro (with biases): {summary.ensemble_f1_tuned:.4f}\n")
        f.write(f"Per-class biases: {summary.biases.round(4).tolist()}\n\n")
        f.write("=== OVERALL ===\n" + summary.overall_report)
        f.write(summary.language_summary)
        f.write("\n".join(summary.per_language_reports))
    print(f"\nSaved {path}")


def main():
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    data = prepare_training_data(DATA_DIR, tokenizer, device)

    test_probs_per_seed, dev_probs_per_seed, y_test, y_dev = collect_ensemble_predictions(
        SEEDS, data, tokenizer, hf_token, device,
    )
    mean_test_probs = np.mean(test_probs_per_seed, axis=0)
    mean_dev_probs = np.mean(dev_probs_per_seed, axis=0)

    per_seed_f1 = report_per_seed(SEEDS, test_probs_per_seed, y_test)
    ensemble_f1_raw = f1_score(
        y_test, np.argmax(mean_test_probs, axis=-1), average="macro", zero_division=0,
    )
    print(f"ENSEMBLE (no bias tuning): f1_macro={ensemble_f1_raw:.4f}")

    dev_f1_raw = f1_score(
        y_dev, np.argmax(mean_dev_probs, axis=-1), average="macro", zero_division=0,
    )
    biases, dev_f1_tuned = tune_per_class_biases(mean_dev_probs, y_dev)
    y_pred = np.argmax(np.log(mean_test_probs + EPSILON) + biases, axis=-1)
    ensemble_f1_tuned = f1_score(y_test, y_pred, average="macro", zero_division=0)

    save_ensemble(ENSEMBLE_DIR, data, biases)

    print("\n=== BIAS TUNING ===")
    print(f"  Per-class biases: {biases.round(3).tolist()}")
    print(f"  Dev   f1_macro:  {dev_f1_raw:.4f}  ->  {dev_f1_tuned:.4f}")
    print(f"  Test  f1_macro:  {ensemble_f1_raw:.4f}  ->  {ensemble_f1_tuned:.4f}")

    overall_report = classification_report_text(y_test, y_pred, data.target_names)
    print("\n=== OVERALL (ensemble) ===\n" + overall_report)
    language_summary, per_language_reports = per_language_f1_report(
        y_test, y_pred, data.test_langs, data.target_names, note="ensemble",
    )
    print(language_summary)
    print("\n".join(per_language_reports))

    output_tag = f"{MODEL_SLUG}_logitadj_ens{len(SEEDS)}_ep{NUM_EPOCHS}_len{MAX_LEN}"
    write_results_file(output_tag, RunSummary(
        per_seed_f1=per_seed_f1,
        ensemble_f1_raw=ensemble_f1_raw,
        dev_f1_raw=dev_f1_raw,
        dev_f1_tuned=dev_f1_tuned,
        ensemble_f1_tuned=ensemble_f1_tuned,
        biases=biases,
        overall_report=overall_report,
        language_summary=language_summary,
        per_language_reports=per_language_reports,
    ))


if __name__ == "__main__":
    main()
