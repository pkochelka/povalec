#!/usr/bin/env python3
"""Train a single mmBERT EU-party classifier that is robust to class imbalance.

Train is heavily imbalanced, but dev/test are uniform across parties (and
language-stratified). Logit-adjusted cross-entropy subtracts the train log-priors
during the loss, so the model is optimised for a uniform label distribution: at
inference the plain argmax of the logits is Bayes-optimal for the uniform dev/test.
The best checkpoint is picked on the uniform dev split and evaluated once on test.
"""
import os
import sys
import json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom")
OUTPUT_DIR = f"{MODEL_SLUG}-logitadj"
MAX_LEN = 512
SEED = 42
MAX_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2
LOGIT_ADJUSTMENT_TAU = 1.0
EPSILON = 1e-12
BEST_METRIC = "f1_macro_mean_lang"


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
    dev_langs: np.ndarray
    test_langs: np.ndarray


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
    def __init__(self, logit_adjustment, **kwargs):
        super().__init__(**kwargs)
        self.logit_adjustment = logit_adjustment

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        adjusted_logits = outputs.logits + self.logit_adjustment
        loss = F.cross_entropy(adjusted_logits, labels)
        return (loss, outputs) if return_outputs else loss


def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def logit_adjustment_for(dataset, num_labels, device):
    counts = np.bincount(np.asarray(dataset["labels"]), minlength=num_labels)
    priors = counts / counts.sum()
    return torch.tensor(
        LOGIT_ADJUSTMENT_TAU * np.log(priors + EPSILON), dtype=torch.float, device=device,
    )


def prepare_training_data(data_dir, tokenizer):
    raw_splits = {name: load_split(name, data_dir) for name in ("train", "dev", "test")}

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


def build_model(data, hf_token, device):
    set_seed(SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token,
        num_labels=data.num_labels, id2label=data.id2label, label2id=data.label2id,
    ).to(device)


def training_arguments(output_dir, num_epochs, evaluate_each_epoch):
    return TrainingArguments(
        output_dir=output_dir,
        fp16=True,
        eval_strategy="epoch" if evaluate_each_epoch else "no",
        save_strategy="epoch" if evaluate_each_epoch else "no",
        logging_steps=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=evaluate_each_epoch,
        metric_for_best_model=BEST_METRIC if evaluate_each_epoch else None,
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        seed=SEED,
    )


def release(model, trainer, device):
    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_and_evaluate(data, tokenizer, hf_token, device):
    print(f"\n{'=' * 60}\n  Training on imbalanced train, selecting on uniform dev\n{'=' * 60}")
    model = build_model(data, hf_token, device)
    metrics_fn = LanguageAwareMetrics()
    metrics_fn.current_languages = data.dev_langs

    trainer = LogitAdjustedTrainer(
        logit_adjustment=logit_adjustment_for(data.train, data.num_labels, device),
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
    best = max(scored, key=lambda entry: entry[f"eval_{BEST_METRIC}"])
    best_epoch = max(1, round(best["epoch"]))
    print(f"Best dev {BEST_METRIC}={best[f'eval_{BEST_METRIC}']:.4f} at epoch {best_epoch}")

    metrics_fn.current_languages = data.test_langs
    predictions = trainer.predict(data.test)
    trainer.save_model(OUTPUT_DIR)

    y_test = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    release(model, trainer, device)
    return best_epoch, y_test, y_pred


def save_manifest(output_dir, data, num_epochs):
    manifest = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_epochs": num_epochs,
        "logit_adjustment_tau": LOGIT_ADJUSTMENT_TAU,
        "label2id": data.label2id,
        "inference": "argmax(model logits)",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_results_file(output_tag, num_epochs, test_f1, overall_report,
                       language_summary, per_language_reports):
    path = f"results_{output_tag}.txt"
    with open(path, "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Logit adjustment tau: {LOGIT_ADJUSTMENT_TAU}\n")
        f.write(f"Best epoch from dev search: {num_epochs}\n")
        f.write(f"Test f1_macro: {test_f1:.4f}\n\n")
        f.write("=== OVERALL ===\n" + overall_report)
        f.write(language_summary)
        f.write("\n".join(per_language_reports))
    print(f"\nSaved {path}")


def main():
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    data = prepare_training_data(DATA_DIR, tokenizer)

    best_epoch, y_test, y_pred = train_and_evaluate(data, tokenizer, hf_token, device)

    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    save_manifest(OUTPUT_DIR, data, best_epoch)

    overall_report = classification_report_text(y_test, y_pred, data.target_names)
    print(f"\n=== OVERALL (test f1_macro={test_f1:.4f}) ===\n" + overall_report)
    language_summary, per_language_reports = per_language_f1_report(
        y_test, y_pred, data.test_langs, data.target_names,
    )
    print(language_summary)
    print("\n".join(per_language_reports))

    output_tag = f"{MODEL_SLUG}_logitadj_ep{best_epoch}_len{MAX_LEN}"
    write_results_file(
        output_tag, best_epoch, test_f1, overall_report, language_summary, per_language_reports,
    )


if __name__ == "__main__":
    main()
