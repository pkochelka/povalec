#!/usr/bin/env python3
"""Fine-tune an mmBERT regressor that maps a reason to a stance score in [-1, 1].

Training pairs come from the EU&I reasons track of Kimi K2.6 and DeepSeek V4 Pro
(+1 = totally agree, -1 = totally disagree). The goal is general stance detection
on opinionated writing, so the model is selected on an out-of-domain set of
hand-labeled speeches rather than on held-out reasons. To resist overfitting to
the formulaic LLM-reason register, identical reasons are deduplicated, the bottom
encoder layers are frozen, and neutrals are down-weighted in the loss.
"""
import os
import re
import json
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import ALL_LANGS_STR, VARIANTS, likert_to_stance, load_dataframe

MODEL_NAME = "jhu-clsp/mmBERT-small"
MODEL_SLUG = MODEL_NAME.split("/")[-1]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, f"{MODEL_SLUG}-stance-regressor")
SOURCE_MODELS = ["kimi-k2.7", "deepseek-v4-pro"]
PARAPHRASE_SUFFIXES = VARIANTS
OOD_LABEL_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label.csv")
OOD_META_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label_meta.csv")

CHOICE_PATTERN = re.compile(r"^choice_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
REASON_PATTERN = re.compile(r"^reason_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
LAYER_PATTERN = re.compile(r"layers?\.(\d+)\.")

MAX_LEN = 256
SEED = 42
NUM_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2
DEV_FRACTION = 0.1
TEST_FRACTION = 0.1
OOD_DEV_FRACTION = 0.5
NEUTRAL_WEIGHT = 0.5
FREEZE_BOTTOM_LAYERS = 12
CLASSIFIER_DROPOUT = 0.2
WEIGHT_DECAY = 0.05


class WeightedRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").view(-1)
        outputs = model(**inputs)
        predictions = outputs.logits.view(-1)
        weights = torch.where(labels.abs() < 0.15, NEUTRAL_WEIGHT, 1.0)
        loss = (weights * (predictions - labels) ** 2).mean()
        return (loss, outputs) if return_outputs else loss


def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def melt_pairs(frame, model):
    frame.columns = [c.replace("_negated", "") for c in frame.columns]
    reasons, choices = {}, {}
    for column in frame.columns:
        reason_match = REASON_PATTERN.match(column)
        if reason_match:
            reasons[(reason_match["lang"], int(reason_match["variant"]))] = frame[column]
            continue
        choice_match = CHOICE_PATTERN.match(column)
        if choice_match:
            key = (choice_match["lang"], int(choice_match["variant"]))
            choices[key] = pd.to_numeric(frame[column], errors="coerce")

    records = []
    for key, reason_series in reasons.items():
        if key not in choices:
            continue
        language, variant = key
        block = pd.DataFrame({
            "text": reason_series.astype("string").str.strip(),
            "choice": choices[key],
            "statement": np.arange(len(reason_series)),
        })
        block["model"] = model
        block["language"] = language
        block["variant"] = variant
        records.append(block)
    return pd.concat(records, ignore_index=True)


def load_pairs():
    frames = []
    for model in SOURCE_MODELS:
        for suffix in PARAPHRASE_SUFFIXES:
            path = os.path.join(DATA_DIR, model, f"{ALL_LANGS_STR}{suffix}.csv")
            frames.append(melt_pairs(load_dataframe(path), model))
    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.dropna(subset=["text", "choice"])
    pairs = pairs[pairs["text"].str.len() > 0]
    before = len(pairs)
    pairs = pairs.drop_duplicates(subset=["text", "choice"])
    pairs["stance"] = likert_to_stance(pairs["choice"]).astype("float32")
    print(f"Loaded {len(pairs)} (reason, stance) pairs "
          f"({before - len(pairs)} duplicate reasons dropped) from {SOURCE_MODELS}")
    print("Stance distribution:\n", pairs["stance"].round(2).value_counts().sort_index().to_dict())
    return pairs.reset_index(drop=True)


def split_by_statement(pairs):
    statements = np.sort(pairs["statement"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(statements)
    n_test = max(1, round(len(statements) * TEST_FRACTION))
    n_dev = max(1, round(len(statements) * DEV_FRACTION))
    test_ids = set(statements[:n_test])
    dev_ids = set(statements[n_test:n_test + n_dev])
    split = pairs["statement"].map(
        lambda s: "test" if s in test_ids else "dev" if s in dev_ids else "train")
    for name in ("train", "dev", "test"):
        subset = pairs[split == name]
        print(f"reasons {name}: {len(subset)} rows, {subset['statement'].nunique()} statements")
    return {name: pairs[split == name].reset_index(drop=True) for name in ("train", "dev", "test")}


def load_ood_eval():
    if not (os.path.exists(OOD_LABEL_PATH) and os.path.exists(OOD_META_PATH)):
        return None
    labels = load_dataframe(OOD_LABEL_PATH)
    meta = load_dataframe(OOD_META_PATH)
    labels["choice"] = pd.to_numeric(labels["choice"], errors="coerce")
    labels = labels.dropna(subset=["choice"])
    if labels.empty:
        return None
    merged = labels.merge(meta[["id", "statement"]], on="id", how="left")
    merged["text"] = merged["answer_text"].astype("string").str.strip()
    merged["stance"] = likert_to_stance(merged["choice"]).astype("float32")
    merged = merged.dropna(subset=["text", "statement"])
    print(f"OOD eval: {len(merged)} hand-labeled speeches "
          f"across {merged['statement'].nunique()} statements")
    return merged[["text", "stance", "statement"]].reset_index(drop=True)


def split_ood(ood):
    statements = np.sort(ood["statement"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(statements)
    n_dev = max(1, round(len(statements) * OOD_DEV_FRACTION))
    dev_ids = set(statements[:n_dev])
    dev = ood[ood["statement"].isin(dev_ids)].reset_index(drop=True)
    test = ood[~ood["statement"].isin(dev_ids)].reset_index(drop=True)
    print(f"OOD dev: {len(dev)} rows ({dev['statement'].nunique()} statements) | "
          f"OOD test: {len(test)} rows ({test['statement'].nunique()} statements)")
    return dev, test


def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(
        df[["text", "stance"]].rename(columns={"stance": "labels"}), preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )


def freeze_bottom_layers(model, n_layers):
    frozen = 0
    for name, param in model.base_model.named_parameters():
        index_match = LAYER_PATTERN.search(name)
        index = int(index_match.group(1)) if index_match else None
        if "embeddings" in name or (index is not None and index < n_layers):
            param.requires_grad = False
            frozen += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Froze embeddings + bottom {n_layers} encoder layers "
          f"({frozen} tensors); {trainable:,} trainable params remain")


def stance_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions.reshape(-1), -1.0, 1.0)
    labels = labels.reshape(-1)
    results = {
        "mse": float(np.mean((predictions - labels) ** 2)),
        "mae": float(np.mean(np.abs(predictions - labels))),
    }
    if predictions.std() > 0 and labels.std() > 0:
        results["pearson"] = float(pearsonr(predictions, labels)[0])
        results["spearman"] = float(spearmanr(predictions, labels)[0])
    else:
        results["pearson"] = 0.0
        results["spearman"] = 0.0
    return results


def build_trainer(model, train_ds, val_ds, collator, tokenizer):
    training_args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, f"{MODEL_SLUG}-stance-tmp"),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=200,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-5,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        seed=SEED,
    )
    return WeightedRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=stance_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )


def evaluate_split(trainer, df, tokenizer, prefix):
    predictions = trainer.predict(tokenize_dataset(df, tokenizer), metric_key_prefix=prefix)
    return {key[len(prefix) + 1:]: value
            for key, value in predictions.metrics.items()
            if key.startswith(prefix + "_") and isinstance(value, float)}


def write_results(selection_note, results):
    path = os.path.join(PROJECT_ROOT, f"results_{MODEL_SLUG}_stance.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Stance regressor: {MODEL_NAME}\n")
        f.write(f"Source models: {SOURCE_MODELS}\n")
        f.write(f"Paraphrases: {PARAPHRASE_SUFFIXES}\n")
        f.write(f"Frozen bottom layers: {FREEZE_BOTTOM_LAYERS} | neutral weight: {NEUTRAL_WEIGHT}\n")
        f.write(f"Model selected on: {selection_note}\n" + "-" * 40 + "\n")
        for split_name, metrics in results.items():
            f.write(f"\n[{split_name}]\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    print(f"\nSaved {path}")


def main():
    device = select_device()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))
    hf_token = os.getenv("HF_TOKEN")
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    reasons = split_by_statement(load_pairs())
    train_ds = tokenize_dataset(reasons["train"], tokenizer)

    ood = load_ood_eval()
    if ood is not None:
        ood_dev, ood_test = split_ood(ood)
        val_ds = tokenize_dataset(ood_dev, tokenizer)
        selection_note = "out-of-domain hand-labeled speeches"
    else:
        ood_test = None
        val_ds = tokenize_dataset(reasons["dev"], tokenizer)
        selection_note = "in-domain reasons dev (OOD speeches not labeled yet)"
        print("\n!! No labeled OOD speeches found -- selecting on in-domain reasons dev.")
        print("   Run analysis/sample_speeches_for_labeling.py, fill 'choice', then rerun.\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token, num_labels=1, problem_type="regression",
        classifier_dropout=CLASSIFIER_DROPOUT,
    ).to(device)
    freeze_bottom_layers(model, FREEZE_BOTTOM_LAYERS)
    model.enable_input_require_grads()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = build_trainer(model, train_ds, val_ds, collator, tokenizer)
    trainer.train()

    results = {}
    if ood_test is not None and len(ood_test) > 0:
        results["OOD speeches (test)"] = evaluate_split(trainer, ood_test, tokenizer, "oodtest")
    results["in-domain reasons (test)"] = evaluate_split(trainer, reasons["test"], tokenizer, "indtest")

    print(f"\n=== RESULTS (selected on {selection_note}) ===")
    for split_name, metrics in results.items():
        print(f"  [{split_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "stance_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "source_models": SOURCE_MODELS,
            "paraphrases": PARAPHRASE_SUFFIXES,
            "max_len": MAX_LEN,
            "frozen_bottom_layers": FREEZE_BOTTOM_LAYERS,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "neutral_weight": NEUTRAL_WEIGHT,
            "selected_on": selection_note,
            "output_range": [-1.0, 1.0],
            "inference": "clip(model(text).logits, -1, 1)",
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved model to {OUTPUT_DIR}/")

    shutil.rmtree(trainer.args.output_dir, ignore_errors=True)
    write_results(selection_note, results)


if __name__ == "__main__":
    main()
