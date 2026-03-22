#!/usr/bin/env python3
import evaluate
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()


HF_TOKEN = os.environ.get("HF_TOKEN")

#model_name = 'distilbert-base-uncased'
#model_name = 'distilbert_trainer/checkpoint-2310'
#model_name = "answerdotai/ModernBERT-base-trainer/checkpoint-4236/"
#model_name = "answerdotai/ModernBERT-large-trainer/checkpoint-8364/"
#model_name = "answerdotai/ModernBERT-large-trainer/checkpoint-1082/"
model_name = "answerdotai/ModernBERT-large-trainer/checkpoint-8206/"


df = pd.read_csv("data/EU Debates/preprocessed/filtered.csv")

df["labels"] = df["speaker_party"].astype("category").cat.codes
df = df[["text", "labels"]]
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
  return tokenizer(example["text"], padding='max_length', truncation=True, max_length=512)

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

predictions_output = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions_output.predictions, axis=-1)
labels = predictions_output.label_ids

# 2. Map IDs back to Party Names
# We extract the label names from the ClassLabel feature we created earlier
id2label = dataset["train"].features["labels"].names
pred_names = [id2label[p] for p in preds]
actual_names = [id2label[l] for l in labels]

# 3. Create the output DataFrame
results_df = dataset["test"].to_pandas()
results_df["predicted_label"] = preds
results_df["predicted_party"] = pred_names
results_df["actual_party"] = actual_names

# Save to CSV
os.makedirs("data/EU Debates/predictions", exist_ok=True)
results_df.to_csv("data/EU Debates/predictions/test_predictions.csv", index=False)
print("Predictions saved to data/EU Debates/predictions/test_predictions.csv")

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: EU Party Classification')
    plt.ylabel('Actual Party')
    plt.xlabel('Predicted Party')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/EU Debates/predictions/confusion_matrix.png")
    plt.show()

# Generate the plot
plot_confusion_matrix(labels, preds, id2label)

print(trainer.evaluate(eval_dataset = tokenized_datasets["test"]))