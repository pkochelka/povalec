#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix

model_path = "./answerdotai/ModernBERT-base-trainer/2epochs-full-data-debates-filtered-europarl-unfiltered" 
test_data_path = "./data/euandi_2019_results/LLaMa_speeches.csv"
device = 0 if torch.cuda.is_available() else -1

print(f"--- Loading model from: {model_path} ---")

def run_evaluation():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    classifier = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device,
        truncation=True,
        max_length=1024
    )

    df_test = pd.read_csv(test_data_path, sep=";", encoding="utf-8-sig")
    
    texts = df_test["answer"].astype(str).tolist()

    print(f"Classifying {len(texts)} samples...")

    results = classifier(texts, batch_size=16)
    print(results)

    predictions = [int(res['label'].split('_')[-1]) for res in results]
    
    df_test["predicted_label"] = predictions
    df_test.to_csv("./data/euandi_2019_results/test_predictions_output_llama.csv", sep=";", index=False, encoding="utf-8-sig")
    print("Predictions saved to test_predictions_output.csv")

if __name__ == "__main__":
    run_evaluation()