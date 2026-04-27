#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import evaluate
import json
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.metrics import confusion_matrix, classification_report

model_name = "answerdotai/ModernBERT-base-trainer/2epochs-full-data-debates-filtered-europarl-unfiltered"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_and_labels(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "speaker_party"])
    df["text"] = df["text"].astype(str)
    df["speaker_party"] = df["speaker_party"].astype(str).str.strip()
    
    unique_parties = sorted(df["speaker_party"].unique())
    label2id = {party: i for i, party in enumerate(unique_parties)}
    id2label = dict(enumerate(unique_parties))
    
    df["labels"] = df["speaker_party"].map(label2id)
    return df[["text", "labels"]], id2label, label2id

df, id2label, label2id = load_data_and_labels("data/combined/test.csv")
num_labels = len(id2label)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

dataset = Dataset.from_pandas(df, preserve_index=False)
dataset = dataset.cast_column("labels", ClassLabel(names=list(id2label.values())))

# 4. Tokenization (Replaced sliding window with Script 2 logic)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024, # Increased max_length as per script 2
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

# 6. Trainer Initialization
training_args = TrainingArguments(
    output_dir="output/modern_bert_model",
    per_device_eval_batch_size=32,
    fp16=True if torch.cuda.is_available() else False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

'''
# 7. Evaluate and Export Results
print("--- Running Evaluation ---")
results = trainer.evaluate(tokenized_datasets)

print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test Top-2 Accuracy: {results['eval_top2_accuracy']:.4f}")

# Save results to JSON (Script 2 style)
os.makedirs("output", exist_ok=True)
with open("output/test_results.json", "w") as f:
    json.dump(results, f, indent=4)

# 8. Visualization (Confusion Matrix)
def plot_simple_cm():
    predictions = trainer.predict(tokenized_datasets)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=list(id2label.values()), 
                yticklabels=list(id2label.values()))
    plt.title('Confusion Matrix (Standard Truncation @ 1024)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("output/confusion_matrix.png")
    plt.show()

plot_simple_cm()
'''
# 9. Classification Report
print("\n--- Classification Report ---")

predictions = trainer.predict(tokenized_datasets)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# Generate report
report = classification_report(
    y_true,
    y_pred,
    target_names=list(id2label.values()),
    digits=4
)

print(report)

# Save as text file
with open("output/classification_report.txt", "w") as f:
    f.write(report)

# Save as structured JSON
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=list(id2label.values()),
    output_dict=True
)

with open("output/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=4)