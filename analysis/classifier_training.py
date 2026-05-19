#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, ClassLabel
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback,
)

MODEL_NAME = "jhu-clsp/mmBERT-base"
MAX_LEN = 512
SEED = 42
MIN_LANG_SAMPLES = 10
MAX_CLASS_WEIGHT = 10.0
PARTY_COL = "EU Party"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {device} ---")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

load_dotenv(os.path.expanduser("~/.env.local"))
HF_TOKEN = os.getenv("HF_TOKEN")


def load_split(name):
    df = pd.read_parquet(os.path.expanduser(f"~/data/Europarl Custom/{name}.parquet"))
    df = df.dropna(subset=[PARTY_COL, "text", "language"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["language"] = df["language"].astype(str)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


df_train = load_split("train")
df_dev   = load_split("dev")
df_test  = load_split("test")

label_list = sorted(df_train[PARTY_COL].unique().tolist())
label2id = {lab: i for i, lab in enumerate(label_list)}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label_list)
print(f"{num_labels} classes: {label2id}")


def attach_labels(df):
    df = df[df[PARTY_COL].isin(label2id)].copy()
    df["labels"] = df[PARTY_COL].map(label2id).astype(int)
    return df[["text", "labels", "language"]].reset_index(drop=True)


df_train = attach_labels(df_train)
df_dev   = attach_labels(df_dev)
df_test  = attach_labels(df_test)

print("Train class counts:\n", df_train["labels"].value_counts().sort_index().to_dict())
print("Train languages:\n",    df_train["language"].value_counts().to_dict())

val_langs  = df_dev["language"].to_numpy()
test_langs = df_test["language"].to_numpy()

raw_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_labels),
    y=df_train["labels"].values,
)
capped_class_weights = np.clip(raw_class_weights, None, MAX_CLASS_WEIGHT)
class_weights = torch.tensor(capped_class_weights, dtype=torch.float, device=device)
print(f"Class weights (capped at {MAX_CLASS_WEIGHT}): {capped_class_weights.round(3)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, token=HF_TOKEN,
    num_labels=num_labels, id2label=id2label, label2id=label2id,
).to(device)

dataset = DatasetDict({
    "train":      Dataset.from_pandas(df_train, preserve_index=False),
    "validation": Dataset.from_pandas(df_dev,   preserve_index=False),
    "test":       Dataset.from_pandas(df_test,  preserve_index=False),
})
for split in dataset:
    dataset[split] = dataset[split].cast_column("labels", ClassLabel(num_classes=num_labels))

tokenized = dataset.map(
    lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
    batched=True,
    remove_columns=["text", "language"],
)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


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
        for lang in np.unique(self.current_languages):
            mask = self.current_languages == lang
            if mask.sum() < MIN_LANG_SAMPLES:
                continue
            f1_by_language[lang] = f1_score(
                labels[mask], preds[mask], average="macro", zero_division=0
            )
            results[f"f1_macro_{lang}"] = f1_by_language[lang]

        if f1_by_language:
            results["f1_macro_mean_lang"]     = float(np.mean(list(f1_by_language.values())))
            results["f1_macro_min_lang"]      = float(min(f1_by_language.values()))
            results["f1_macro_min_lang_name"] = min(f1_by_language, key=f1_by_language.get)

        return results


metrics_fn = LanguageAwareMetrics()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, val_langs, test_langs, metrics_fn, **kw):
        super().__init__(**kw)
        self.val_langs = val_langs
        self.test_langs = test_langs
        self.metrics_fn = metrics_fn
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(
            outputs.logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.metrics_fn.current_languages = (
            self.test_langs if metric_key_prefix == "test" else self.val_langs
        )
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        self.metrics_fn.current_languages = self.test_langs
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)


training_args = TrainingArguments(
    output_dir=f"{MODEL_NAME.split('/')[-1]}-trainer_strat",
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=2000,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro_mean_lang",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
    optim="adamw_torch_fused",
    seed=SEED,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    val_langs=val_langs,
    test_langs=test_langs,
    metrics_fn=metrics_fn,
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

trainer.train()

print("\n=== TEST ===")
test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

predictions = trainer.predict(tokenized["test"])
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
target_names = [id2label[i] for i in range(num_labels)]


def make_report(true_labels, predicted_labels):
    return classification_report(
        true_labels, predicted_labels,
        labels=list(range(num_labels)),
        target_names=target_names,
        digits=4, zero_division=0,
    )


overall_report = make_report(y_true, y_pred)
print("\n=== OVERALL ===\n" + overall_report)

f1_by_language = {}
per_language_reports = []
for lang in sorted(np.unique(test_langs)):
    mask = test_langs == lang
    if mask.sum() < MIN_LANG_SAMPLES:
        continue
    f1m = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
    f1_by_language[lang] = (f1m, int(mask.sum()))
    per_language_reports.append(
        f"\n=== {lang} (n={mask.sum()}) | macro F1 = {f1m:.4f} ===\n"
        + make_report(y_true[mask], y_pred[mask])
    )

summary = "\n=== PER-LANGUAGE SUMMARY (sorted by macro F1) ===\n"
summary += f"{'lang':<8}{'n':>10}{'macro_f1':>14}\n" + "-" * 32 + "\n"
for lang, (f1m, n) in sorted(f1_by_language.items(), key=lambda x: x[1][0]):
    summary += f"{lang:<8}{n:>10d}{f1m:>14.4f}\n"

mean_lang_f1 = float(np.mean([v[0] for v in f1_by_language.values()]))
worst_lang, (worst_f1, _) = min(f1_by_language.items(), key=lambda x: x[1][0])
summary += "-" * 32 + "\n"
summary += f"MEAN across languages: {mean_lang_f1:.4f}\n"
summary += f"MIN  across languages: {worst_f1:.4f}  ({worst_lang})\n"

print(summary)
print("\n".join(per_language_reports))

output_tag = f"{MODEL_NAME.split('/')[-1]}_lr{training_args.learning_rate}_ep{training_args.num_train_epochs}_len{MAX_LEN}_strat"
with open(f"results_{output_tag}.txt", "w") as f:
    f.write(output_tag + "\n" + "-" * 40 + "\n")
    f.write(json.dumps(
        {k: v for k, v in test_results.items() if isinstance(v, (int, float, str))},
        indent=2,
    ))
    f.write("\n\n=== OVERALL ===\n" + overall_report)
    f.write(summary)
    f.write("\n".join(per_language_reports))
print(f"\nSaved results_{output_tag}.txt")