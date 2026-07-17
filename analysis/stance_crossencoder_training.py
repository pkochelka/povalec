#!/usr/bin/env python3
"""Train a (statement, text) -> stance cross-encoder via two-stage fine-tuning.

Unlike the text-only regressor, this model sees BOTH the statement and the text,
so "I agree that X" is unambiguous. An mmBERT-small base is fine-tuned in two
stages: first on the abundant reasons track (statement = original_text, free
choice labels) as a warmup, then continued on the LLM-judged speeches (the target
creative-writing register). The second stage is selected on the hand-labeled
speeches, which neither training source touches.
"""
import os
import re
import sys
import json
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import ALL_LANGS_STR, likert_to_stance, load_dataframe
from analysis.stance_detector_training import stance_metrics, freeze_bottom_layers, select_device

MODEL_NAME = "jhu-clsp/mmBERT-small"
MODEL_SLUG = "mmbert-small-stance-crossencoder"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, MODEL_SLUG)
SOURCE_MODELS = ["kimi-k2.6", "deepseek-v4-pro"]
PARAPHRASE_SUFFIXES = ["", "_question", "_negated"]
LLM_LABELS_PATH = os.path.join(DATA_DIR, "stance_speeches_llm_labeled.csv")
OOD_LABEL_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label.csv")
OOD_META_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label_meta.csv")

CHOICE_PATTERN = re.compile(r"^choice_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
REASON_PATTERN = re.compile(r"^reason_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
ORIGINAL_PATTERN = re.compile(r"^original_text_(?P<lang>[a-z]{2})$")

MAX_LEN = 512
SEED = 42
REASON_EPOCHS = 3
SPEECH_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
OOD_DEV_FRACTION = 0.5
NEUTRAL_WEIGHT = 0.9
FREEZE_BOTTOM_LAYERS = 12
WEIGHT_DECAY = 0.05


class WeightedRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").view(-1)
        outputs = model(**inputs)
        predictions = outputs.logits.view(-1)
        weights = torch.where(labels.abs() < 0.15, NEUTRAL_WEIGHT, 1.0)
        loss = (weights * (predictions - labels) ** 2).mean()
        return (loss, outputs) if return_outputs else loss


def melt_reasons(frame, model):
    frame.columns = [c.replace("_question", "").replace("_negated", "") for c in frame.columns]
    reasons, choices, statements = {}, {}, {}
    for column in frame.columns:
        reason_match = REASON_PATTERN.match(column)
        if reason_match:
            reasons[(reason_match["lang"], int(reason_match["variant"]))] = frame[column]
            continue
        choice_match = CHOICE_PATTERN.match(column)
        if choice_match:
            choices[(choice_match["lang"], int(choice_match["variant"]))] = pd.to_numeric(
                frame[column], errors="coerce")
            continue
        original_match = ORIGINAL_PATTERN.match(column)
        if original_match:
            statements[original_match["lang"]] = frame[column]

    records = []
    for (language, variant), reason_series in reasons.items():
        if (language, variant) not in choices or language not in statements:
            continue
        records.append(pd.DataFrame({
            "statement": statements[language].astype("string").str.strip(),
            "text": reason_series.astype("string").str.strip(),
            "choice": choices[(language, variant)],
        }))
    return pd.concat(records, ignore_index=True)


def load_reason_pairs():
    frames = []
    for model in SOURCE_MODELS:
        for suffix in PARAPHRASE_SUFFIXES:
            path = os.path.join(DATA_DIR, model, f"{ALL_LANGS_STR}{suffix}.csv")
            frames.append(melt_reasons(load_dataframe(path), model))
    pairs = pd.concat(frames, ignore_index=True).dropna(subset=["statement", "text", "choice"])
    pairs = pairs[(pairs["statement"].str.len() > 0) & (pairs["text"].str.len() > 0)]
    pairs["stance"] = likert_to_stance(pairs["choice"]).astype("float32")
    return pairs[["statement", "text", "stance"]]


def load_llm_speeches():
    if not os.path.exists(LLM_LABELS_PATH):
        return pd.DataFrame(columns=["statement", "text", "stance"])
    frame = load_dataframe(LLM_LABELS_PATH).drop(columns=["statement"], errors="ignore")
    frame = frame.rename(columns={"statement_text": "statement", "answer_text": "text"})
    frame["stance"] = pd.to_numeric(frame["llm_stance"], errors="coerce").astype("float32")
    frame = frame.dropna(subset=["statement", "text", "stance"])
    return frame[["statement", "text", "stance"]]


def build_stage_pairs(loader, name):
    pairs = loader().drop_duplicates(subset=["statement", "text"]).reset_index(drop=True)
    print(f"{name} pairs: {len(pairs)} after dedup")
    print("Stance distribution:\n", pairs["stance"].round(2).value_counts().sort_index().to_dict())
    return pairs


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
    merged = merged.rename(columns={"statement_text": "statement", "answer_text": "text",
                                    "statement": "statement_id"})
    merged["stance"] = likert_to_stance(merged["choice"]).astype("float32")
    merged = merged.dropna(subset=["statement", "text", "statement_id"])
    print(f"OOD eval: {len(merged)} hand-labeled speeches "
          f"across {merged['statement_id'].nunique()} statements")
    return merged[["statement", "text", "stance", "statement_id"]].reset_index(drop=True)


def split_ood(ood):
    statements = np.sort(ood["statement_id"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(statements)
    n_dev = max(1, round(len(statements) * OOD_DEV_FRACTION))
    dev_ids = set(statements[:n_dev])
    dev = ood[ood["statement_id"].isin(dev_ids)].reset_index(drop=True)
    test = ood[~ood["statement_id"].isin(dev_ids)].reset_index(drop=True)
    print(f"OOD dev: {len(dev)} rows ({dev['statement_id'].nunique()} statements) | "
          f"OOD test: {len(test)} rows ({test['statement_id'].nunique()} statements)")
    return dev, test


def tokenize_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(
        df[["statement", "text", "stance"]].rename(columns={"stance": "labels"}),
        preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["statement"], batch["text"],
                                truncation="only_second", max_length=MAX_LEN),
        batched=True,
        remove_columns=["statement", "text"],
    )


def build_trainer(model, train_ds, val_ds, collator, tokenizer, num_epochs, output_subdir, select):
    training_args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, output_subdir),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if select else "no",
        save_strategy="epoch" if select else "no",
        logging_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        num_train_epochs=num_epochs,
        learning_rate=1e-5,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=select,
        metric_for_best_model="mse" if select else None,
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
        eval_dataset=val_ds if select else None,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=stance_metrics if select else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if select else None,
    )


def evaluate_split(trainer, df, tokenizer, prefix):
    predictions = trainer.predict(tokenize_dataset(df, tokenizer), metric_key_prefix=prefix)
    return {key[len(prefix) + 1:]: value
            for key, value in predictions.metrics.items()
            if key.startswith(prefix + "_") and isinstance(value, float)}


def main():
    device = select_device()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))
    hf_token = os.getenv("HF_TOKEN")
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    reason_pairs = build_stage_pairs(load_reason_pairs, "Stage 1 (reasons)")
    speech_pairs = build_stage_pairs(load_llm_speeches, "Stage 2 (LLM-judged speeches)")
    if speech_pairs.empty:
        raise SystemExit("No LLM-judged speeches found; run llm_stance_judge.py first.")
    reason_ds = tokenize_dataset(reason_pairs, tokenizer)
    speech_ds = tokenize_dataset(speech_pairs, tokenizer)

    ood = load_ood_eval()
    if ood is not None:
        ood_dev, ood_test = split_ood(ood)
        val_ds = tokenize_dataset(ood_dev, tokenizer)
        selection_note = "out-of-domain hand-labeled speeches"
    else:
        ood_test = None
        holdout = speech_pairs.sample(frac=0.05, random_state=SEED)
        val_ds = tokenize_dataset(holdout, tokenizer)
        selection_note = "5% random speech holdout (OOD speeches not labeled yet)"
        print("\n!! No labeled OOD speeches -- selecting on a random speech holdout.\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token, num_labels=1, problem_type="regression",
        ignore_mismatched_sizes=True,
    ).to(device)
    freeze_bottom_layers(model, FREEZE_BOTTOM_LAYERS)
    model.enable_input_require_grads()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"\n{'=' * 60}\n  Stage 1: warmup on reasons for {REASON_EPOCHS} epochs\n{'=' * 60}")
    warmup = build_trainer(model, reason_ds, None, collator, tokenizer,
                           REASON_EPOCHS, f"{MODEL_SLUG}-stage1", select=False)
    warmup.train()
    shutil.rmtree(warmup.args.output_dir, ignore_errors=True)

    print(f"\n{'=' * 60}\n  Stage 2: fine-tune on speeches, selected on {selection_note}\n{'=' * 60}")
    trainer = build_trainer(model, speech_ds, val_ds, collator, tokenizer,
                            SPEECH_EPOCHS, f"{MODEL_SLUG}-stage2", select=True)
    trainer.train()

    results = {}
    if ood_test is not None and len(ood_test) > 0:
        results["OOD speeches (test)"] = evaluate_split(trainer, ood_test, tokenizer, "oodtest")
    print(f"\n=== RESULTS (selected on {selection_note}) ===")
    for split_name, metrics in results.items():
        print(f"  [{split_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "stance_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "architecture": "cross-encoder (statement, text) -> stance",
            "training": f"two-stage: {REASON_EPOCHS} reason warmup epochs -> "
                        f"{SPEECH_EPOCHS} speech epochs",
            "max_len": MAX_LEN,
            "frozen_bottom_layers": FREEZE_BOTTOM_LAYERS,
            "neutral_weight": NEUTRAL_WEIGHT,
            "selected_on": selection_note,
            "output_range": [-1.0, 1.0],
            "inference": "clip(model(statement, text).logits, -1, 1)",
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved model to {OUTPUT_DIR}/")

    shutil.rmtree(trainer.args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
