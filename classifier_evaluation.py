#!/usr/bin/env python3
import evaluate
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


HF_TOKEN = os.environ.get("HF_TOKEN")

#model_name = 'distilbert-base-uncased'
#model_name = 'distilbert_trainer/checkpoint-2310'
model_name = "answerdotai/ModernBERT-base-trainer/checkpoint-7851/"

COLUMN_NAME_EU_PARTY = "EU Party"

df = pd.read_csv("data/EuroParl/preprocessed/full.csv")

df[COLUMN_NAME_EU_PARTY] = df[COLUMN_NAME_EU_PARTY].apply(lambda x: x.split('/')[-1])
df = df[~df[COLUMN_NAME_EU_PARTY].str.contains(r"\bNA\b", case=False, na=False)]
df["labels"] = df[COLUMN_NAME_EU_PARTY].astype("category").cat.codes
df = df[["en", "labels"]]
df["labels"] = df["labels"].astype("category").cat.codes

def load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

print(df["labels"].unique())

tokenizer, model = load_model()
dataset = Dataset.from_pandas(df, preserve_index=False)

num_labels = len(df["labels"].unique())

dataset = dataset.cast_column(
    "labels",
    ClassLabel(num_classes=num_labels)
)

train_test = dataset.train_test_split(
    test_size=0.2, seed=42, stratify_by_column="labels"
)

val_test = train_test["test"].train_test_split(
    test_size=0.5, seed=42, stratify_by_column="labels"
)

dataset = DatasetDict({
    "train": train_test["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
})

def tokenize_function(example):
  return tokenizer(example["en"], padding='max_length', truncation=True, max_length=512)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds_top1 = np.argmax(logits, axis=-1)
    
    top2_preds = np.argsort(logits, axis=-1)[:, -2:]
    top2_correct = [label in top2 for label, top2 in zip(labels, top2_preds)]
    top2_accuracy = np.mean(top2_correct)
    
    return {
        "accuracy": accuracy.compute(predictions=preds_top1, references=labels)["accuracy"],
        "top2_accuracy": top2_accuracy,
        "f1_macro": f1.compute(predictions=preds_top1, references=labels, average="macro")["f1"]
    }

trainer = Trainer(
    model,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print(trainer.evaluate(eval_dataset = tokenized_datasets["test"]))
