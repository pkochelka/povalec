#!/usr/bin/env python3
import evaluate
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.utils.class_weight import compute_class_weight

HF_TOKEN = os.environ.get("HF_TOKEN")

#model_name = 'distilbert-base-uncased'
model_name = 'answerdotai/ModernBERT-base'
#model_name = 'answerdotai/ModernBERT-large'

df_train = pd.read_csv("data/combined/train.csv")
df_train["labels"] = df_train["speaker_party"].astype("category").cat.codes
df_train = df_train[["text", "labels"]]
df_train["text"] = df_train["text"].astype(str)
df_train["labels"] = df_train["labels"].astype("category").cat.codes

df_dev = pd.read_csv("data/combined/dev.csv")
df_dev["labels"] = df_dev["speaker_party"].astype("category").cat.codes
df_dev = df_dev[["text", "labels"]]
df_dev["text"] = df_dev["text"].astype(str)
df_dev["labels"] = df_dev["labels"].astype("category").cat.codes

df_test = pd.read_csv("data/combined/test.csv")
df_test["labels"] = df_test["speaker_party"].astype("category").cat.codes
df_test = df_test[["text", "labels"]]
df_test["text"] = df_test["text"].astype(str)
df_test["labels"] = df_test["labels"].astype("category").cat.codes

X_train = df_train["text"]
y_train = df_train["labels"]

X_dev = df_dev["text"]
y_dev = df_dev["labels"]

X_test = df_test["text"]
y_test = df_test["labels"]


def download_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token = HF_TOKEN,
        num_labels = len(df_train["labels"].unique())
    )
    
    model.save_pretrained(f'./local_model_{model_name}')
    tokenizer.save_pretrained(f'./local_model_{model_name}')

    return tokenizer, model

def load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(f"./local_model_{model_name}/", num_labels = len(df_train["labels"].unique()))
    model = AutoModelForSequenceClassification.from_pretrained(f"./local_model_{model_name}/", num_labels = len(df_train["labels"].unique()))
    return tokenizer, model

tokenizer, model = download_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)

ds_train = Dataset.from_pandas(df_train, preserve_index=False)
ds_val = Dataset.from_pandas(df_dev, preserve_index=False)
ds_test = Dataset.from_pandas(df_test, preserve_index=False)

dataset = DatasetDict({
    "train": ds_train,
    "validation": ds_val,
    "test": ds_test
})

num_labels = len(df_train["labels"].unique())
dataset = dataset.cast_column("labels", ClassLabel(num_classes=num_labels))

train_labels = dataset["train"]["labels"]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights.to(model.device)


def weighted_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    
    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights.to(logits.device)
    )
    
    return loss

def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        return_overflowing_tokens=True,
        stride=50,
    )

    # propagate labels to all chunks
    sample_map = tokens.pop("overflow_to_sample_mapping")
    tokens["labels"] = [example["labels"][i] for i in sample_map]

    return tokens

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds_top1 = np.argmax(logits, axis=-1)
    
    # Compute top-2 predictions
    top2_preds = np.argsort(logits, axis=-1)[:, -2:]  # take indices of two largest logits
    
    # Check if true labels are in top-2 predictions
    top2_correct = [label in top2 for label, top2 in zip(labels, top2_preds)]
    top2_accuracy = np.mean(top2_correct)
    
    return {
        "accuracy": accuracy.compute(predictions=preds_top1, references=labels)["accuracy"],
        "top2_accuracy": top2_accuracy,
        "f1_macro": f1.compute(predictions=preds_top1, references=labels, average="macro")["f1"]
    }

training_args = TrainingArguments(f"{model_name}-trainer", fp16=True, eval_strategy="epoch", 
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=64,
    weight_decay=0.1,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    compute_loss_func=weighted_loss,
)

trainer.train()

def predict_speech(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.argmax(logits, dim=-1).item()

def evaluate_full_speeches(dataset_split):
    preds, labels = [], []

    for example in dataset_split:
        preds.append(predict_speech(example["text"]))
        labels.append(example["labels"])

    acc = (np.array(preds) == np.array(labels)).mean()
    print("Speech-level accuracy:", acc)

evaluate_full_speeches(dataset["test"])

