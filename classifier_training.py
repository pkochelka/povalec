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

#df = pd.read_csv("data/EuroParl/scored_llama/full.csv")
df = pd.read_csv("data/EuroParl Custom/en_parties_classifier.csv")
df = df.dropna(subset=["en", "party_group_std"])
df["en"] = df["en"].astype(str)

df["labels"] = df["party_group_std"].astype("category").cat.codes
print(df["party_group_std"].unique())
df = df[["en", "labels"]]
df["labels"] = df["labels"].astype("category").cat.codes



def download_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token = HF_TOKEN,
        num_labels = len(df["labels"].unique())
    )
    
    model.save_pretrained(f'./local_model_{model_name}')
    tokenizer.save_pretrained(f'./local_model_{model_name}')

    return tokenizer, model

def load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(f"./local_model_{model_name}/", num_labels = len(df["labels"].unique()))
    model = AutoModelForSequenceClassification.from_pretrained(f"./local_model_{model_name}/", num_labels = len(df["labels"].unique()))
    return tokenizer, model

print(df["labels"].unique())

tokenizer, model = download_model()
dataset = Dataset.from_pandas(df, preserve_index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)


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
        example["en"],
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
    remove_columns=["en"]
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def predict_speech(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_overflowing_tokens=True,
        stride=50,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    avg_logits = logits.mean(dim=0)
    pred = torch.argmax(avg_logits).item()

    return pred

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
    per_device_train_batch_size=8,
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

def evaluate_full_speeches(dataset_split):
    preds = []
    labels = []

    for example in dataset_split:
        text = example["en"]
        label = example["labels"]

        pred = predict_speech(text)

        preds.append(pred)
        labels.append(label)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = (preds == labels).mean()

    print("Speech-level accuracy:", acc)

evaluate_full_speeches(dataset["test"])