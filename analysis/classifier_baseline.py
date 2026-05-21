#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
MIN_LANG_SAMPLES = 10
PARTY_COL = "EU Party"
MAX_CHARS = 2000
CHUNK_SIZE = 50_000
EPOCHS = 3
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "EuroParl Custom")


def load_split(name):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{name}.parquet"))
    df = df.dropna(subset=[PARTY_COL, "text", "language"]).copy()
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, MAX_CHARS)
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

word_vec = HashingVectorizer(
    analyzer="word", ngram_range=(1, 2), n_features=2**21,
    alternate_sign=False, norm="l2", strip_accents="unicode",
)
char_vec = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), n_features=2**21,
    alternate_sign=False, norm="l2", strip_accents="unicode",
)


def vectorize(texts):
    return hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()


classes = np.arange(num_labels)
train_texts = df_train["text"].to_numpy()
train_labels = df_train["labels"].to_numpy()
class_weight = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=train_labels)))

clf = SGDClassifier(loss="hinge", alpha=1e-5, class_weight=class_weight,
                    random_state=SEED, n_jobs=-1)

print("\n--- Fitting hashed features + linear SGD (out-of-core) ---")
rng = np.random.default_rng(SEED)
for epoch in range(EPOCHS):
    order = rng.permutation(len(train_texts))
    for start in range(0, len(order), CHUNK_SIZE):
        idx = order[start:start + CHUNK_SIZE]
        X_chunk = vectorize(train_texts[idx].tolist())
        clf.partial_fit(X_chunk, train_labels[idx], classes=classes)
    print(f"epoch {epoch + 1}/{EPOCHS} done")

target_names = [id2label[i] for i in range(num_labels)]


def make_report(true_labels, predicted_labels):
    return classification_report(
        true_labels, predicted_labels,
        labels=list(range(num_labels)),
        target_names=target_names,
        digits=4, zero_division=0,
    )


def predict_in_chunks(texts):
    preds = []
    for start in range(0, len(texts), CHUNK_SIZE):
        preds.append(clf.predict(vectorize(texts[start:start + CHUNK_SIZE])))
    return np.concatenate(preds)


def evaluate(split_name, df):
    y_true = df["labels"].to_numpy()
    langs = df["language"].to_numpy()
    y_pred = predict_in_chunks(df["text"].tolist())

    results = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n=== {split_name.upper()} ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    overall_report = make_report(y_true, y_pred)
    print("\n=== OVERALL ===\n" + overall_report)

    f1_by_language = {}
    per_language_reports = []
    for lang in sorted(np.unique(langs)):
        mask = langs == lang
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

    output_tag = f"baseline_hashing_sgd_{split_name}"
    with open(f"results_{output_tag}.txt", "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n\n=== OVERALL ===\n" + overall_report)
        f.write(summary)
        f.write("\n".join(per_language_reports))
    print(f"\nSaved results_{output_tag}.txt")


evaluate("dev",  df_dev)
evaluate("test", df_test)
