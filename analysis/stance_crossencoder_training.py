#!/usr/bin/env python3
"""Train a (statement, text) -> stance cross-encoder via two-stage fine-tuning.

Unlike the text-only regressor, this model sees BOTH the statement and the text,
so "I agree that X" is unambiguous. An mmBERT-small base is fine-tuned in two
stages, both on texts pooled over TRAIN_MODELS as scored by the LLM stance judge
(judge_all_speeches.py): first a warmup on the abundant reasons track (the short
justifications), then continued on the speeches track (the target creative-writing
register). The second stage is selected on LLM-judged speeches from held-out source
models (OOD_SOURCE_MODELS), which the training data never touches.
"""
import os
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

from utils import likert_to_stance, load_dataframe
from analysis.stance_detector_training import stance_metrics, freeze_bottom_layers, select_device
from analysis.judge_all_speeches import load_speech_pool, load_reason_pool
from analysis.llm_stance_judge import load_checkpoint

MODEL_NAME = "jhu-clsp/mmBERT-small"
MODEL_SLUG = "mmbert-small-stance-crossencoder"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, MODEL_SLUG)
# Both training stages read the LLM-judge outputs from judge_all_speeches.py, pooled
# over several source models; the reasons track warms the model up, the speeches track
# is the target register. Held-out speeches come from models that are never trained on
# (OOD_SOURCE_MODELS), used only to select/evaluate stage 2. Models with no judge output
# yet are reported and skipped, so this list may run ahead of judge_all_speeches.py.
TRAIN_MODELS = ["deepseek-v4-pro", "glm-5.2", "qwen3.5-122b", "gpt-oss-120b",
                "gemini3.5-flash", "gemma-4-31b"]
OOD_SOURCE_MODELS = ["kimi-k2.7", "mistral-medium-3.5"]
# per-track pool reconstruction + output filename used when a judge run is unfinished
POOL_LOADERS = {"speeches": load_speech_pool, "reasons": load_reason_pool}
STANCE_NAME = {"speeches": "speeches_llm_stance.csv", "reasons": "reasons_llm_stance.csv"}

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


def load_judged(model, track):
    """One model/track's LLM-judged texts as (statement, text, stance, statement_id).

    Prefers the finished speeches_/reasons_llm_stance.csv; while a judge run is still
    going, reconstructs labels from the checkpoint that indexes the pool the matching
    pool loader rebuilds, so partial progress is usable. "statement_id" is the statement
    index (shared across languages and the base/negated framings of one proposition),
    the leakage-safe group for the OOD split.
    """
    output_path = os.path.join(DATA_DIR, model, STANCE_NAME[track])
    checkpoint_path = output_path + ".ckpt.json"
    empty = pd.DataFrame(columns=["statement", "text", "stance", "statement_id", "source"])
    if os.path.exists(output_path):
        pool = load_dataframe(output_path)
        pool["llm_choice"] = pd.to_numeric(pool["llm_choice"], errors="coerce")
    elif os.path.exists(checkpoint_path):
        pool = POOL_LOADERS[track](model).copy()
        pool["llm_choice"] = pool.index.map(load_checkpoint(checkpoint_path))
    else:
        print(f"  !! {model}/{track}: not judged yet (no {STANCE_NAME[track]}); skipping.")
        return empty
    pool = pool.dropna(subset=["llm_choice"]).reset_index(drop=True)
    if pool.empty:
        return empty
    out = pd.DataFrame({
        "statement": pool["statement_text"].astype("string").str.strip(),
        "text": pool["answer_text"].astype("string").str.strip(),
        "stance": likert_to_stance(pool["llm_choice"].astype(int)).astype("float32"),
        "statement_id": pool["statement"].values,
        "source": model,
    })
    out = out.dropna(subset=["statement", "text", "stance", "statement_id"])
    out = out[(out["statement"].str.len() > 0) & (out["text"].str.len() > 0)]
    return out


def load_pool(models, track):
    """Judged (statement, text, stance) pairs pooled over models, deduped.

    The same text can surface under two source models only by coincidence; dedup on
    (statement, text) keeps one copy so no pair is weighted twice.
    """
    frames = [load_judged(model, track) for model in models]
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.empty:
        return pooled
    return pooled.drop_duplicates(subset=["statement", "text"]).reset_index(drop=True)


def build_stage_pairs(track, name):
    pairs = load_pool(TRAIN_MODELS, track)
    if pairs.empty:
        print(f"{name} pairs: none judged yet.")
        return pairs
    per_model = pairs["source"].value_counts().to_dict()
    print(f"{name} pairs: {len(pairs)} after dedup -- " +
          ", ".join(f"{model}={per_model.get(model, 0)}" for model in TRAIN_MODELS))
    print("Stance distribution:\n", pairs["stance"].round(2).value_counts().sort_index().to_dict())
    return pairs


def load_ood_eval():
    """Judged speeches from OOD_SOURCE_MODELS (never trained on) for stage-2 selection."""
    ood = load_pool(OOD_SOURCE_MODELS, "speeches")
    if ood.empty:
        return None
    for model, count in ood["source"].value_counts().items():
        print(f"OOD eval ({model}): {count} judged speeches")
    print(f"OOD eval total: {len(ood)} speeches across "
          f"{ood['statement_id'].nunique()} statements")
    return ood


def split_ood(ood):
    """Split the pooled OOD speeches by statement, one split shared by all sources.

    Splitting statements once (rather than per source) keeps every statement that
    stage 2 is selected on out of every source's test half.
    """
    statements = np.sort(ood["statement_id"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(statements)
    n_dev = max(1, round(len(statements) * OOD_DEV_FRACTION))
    dev_ids = set(statements[:n_dev])
    dev = ood[ood["statement_id"].isin(dev_ids)].reset_index(drop=True)
    test = ood[~ood["statement_id"].isin(dev_ids)].reset_index(drop=True)
    print(f"OOD dev: {len(dev)} rows ({dev['statement_id'].nunique()} statements) | "
          f"OOD test: {len(test)} rows ({test['statement_id'].nunique()} statements)")
    print("  test rows by source: ", test["source"].value_counts().to_dict())
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
    reason_pairs = build_stage_pairs("reasons", "Stage 1 (LLM-judged reasons)")
    speech_pairs = build_stage_pairs("speeches", "Stage 2 (LLM-judged speeches)")
    if speech_pairs.empty:
        raise SystemExit(f"No LLM-judged speeches for {TRAIN_MODELS}; "
                         "run judge_all_speeches.py first.")
    # only the models that actually had judge output, so the saved config does not
    # claim training data that was skipped
    trained_on = [model for model in TRAIN_MODELS
                  if model in set(reason_pairs.get("source", [])) | set(speech_pairs["source"])]
    reason_ds = tokenize_dataset(reason_pairs, tokenizer)
    speech_ds = tokenize_dataset(speech_pairs, tokenizer)

    ood = load_ood_eval()
    if ood is not None:
        ood_dev, ood_test = split_ood(ood)
        val_ds = tokenize_dataset(ood_dev, tokenizer)
        selection_note = ("out-of-domain LLM-judged speeches from "
                          f"{', '.join(sorted(ood['source'].unique()))}")
    else:
        ood_test = None
        holdout = speech_pairs.sample(frac=0.05, random_state=SEED)
        val_ds = tokenize_dataset(holdout, tokenizer)
        selection_note = "5% random speech holdout (OOD speeches not judged yet)"
        print("\n!! No judged OOD speeches -- selecting on a random speech holdout.\n")

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
        results["OOD speeches (test, pooled)"] = evaluate_split(
            trainer, ood_test, tokenizer, "oodtest")
        # per-source too: the sources differ in register, and a pooled number can hide
        # one of them being much worse than the other.
        for model in OOD_SOURCE_MODELS:
            subset = ood_test[ood_test["source"] == model]
            if len(subset) > 0:
                results[f"OOD speeches (test, {model})"] = evaluate_split(
                    trainer, subset, tokenizer, "oodtest")
    print(f"\n=== RESULTS (selected on {selection_note}) ===")
    for split_name, metrics in results.items():
        print(f"  [{split_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "stance_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "architecture": "cross-encoder (statement, text) -> stance",
            "training": f"two-stage on {', '.join(trained_on)} LLM-judged texts: "
                        f"{REASON_EPOCHS} reason warmup epochs -> "
                        f"{SPEECH_EPOCHS} speech epochs",
            "held_out_models": OOD_SOURCE_MODELS,
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
