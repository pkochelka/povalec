#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from analysis.europarl_classification import (
    PARTY_COLUMN,
    attach_labels,
    build_label_maps,
    classification_report_text,
    load_split,
    per_language_f1_report,
)

SEED = 42
MAX_CHARS = 2000
CHUNK_SIZE = 50_000
EPOCHS = 3
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "EuroParl Custom")

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


def predict_in_chunks(clf, texts):
    preds = []
    for start in range(0, len(texts), CHUNK_SIZE):
        preds.append(clf.predict(vectorize(texts[start:start + CHUNK_SIZE])))
    return np.concatenate(preds)


def evaluate(clf, split_name, df, target_names, tag, extra=None):
    y_true = df["labels"].to_numpy()
    langs = df["language"].to_numpy()
    y_pred = predict_in_chunks(clf, df["text"].tolist())

    results = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if extra:
        results.update(extra)
    print(f"\n=== {split_name.upper()} ===")
    for key, value in results.items():
        print(f"{key}: {value}" if not isinstance(value, float) else f"{key}: {value:.4f}")

    overall_report = classification_report_text(y_true, y_pred, target_names)
    print("\n=== OVERALL ===\n" + overall_report)

    summary, per_language_reports = per_language_f1_report(y_true, y_pred, langs, target_names)
    print(summary)
    print("\n".join(per_language_reports))

    output_tag = f"{tag}_{split_name}"
    with open(f"results_{output_tag}.txt", "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n\n=== OVERALL ===\n" + overall_report)
        f.write(summary)
        f.write("\n".join(per_language_reports))
    print(f"\nSaved results_{output_tag}.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--tag", default="baseline_hashing_sgd")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    df_train = load_split(args.train_split, args.data_dir, max_chars=MAX_CHARS)
    df_dev   = load_split(args.dev_split, args.data_dir, max_chars=MAX_CHARS)
    df_test  = load_split(args.test_split, args.data_dir, max_chars=MAX_CHARS)

    label_list = sorted(df_train[PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    print(f"{num_labels} classes: {label2id}")

    df_train = attach_labels(df_train, label2id)
    df_dev   = attach_labels(df_dev, label2id)
    df_test  = attach_labels(df_test, label2id)

    print("Train class counts:\n", df_train["labels"].value_counts().sort_index().to_dict())
    print("Train languages:\n",    df_train["language"].value_counts().to_dict())

    classes = np.arange(num_labels)
    train_texts = df_train["text"].to_numpy()
    train_labels = df_train["labels"].to_numpy()
    class_weight = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=train_labels)))

    clf = SGDClassifier(loss="hinge", alpha=1e-5, class_weight=class_weight,
                        random_state=SEED, n_jobs=-1)

    dev_texts = df_dev["text"].tolist()
    dev_true = df_dev["labels"].to_numpy()

    print("\n--- Fitting hashed features + linear SGD (out-of-core) ---")
    rng = np.random.default_rng(SEED)
    best_epoch, best_dev_f1, best_state = None, -np.inf, None
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_texts))
        for start in range(0, len(order), CHUNK_SIZE):
            idx = order[start:start + CHUNK_SIZE]
            X_chunk = vectorize(train_texts[idx].tolist())
            clf.partial_fit(X_chunk, train_labels[idx], classes=classes)
        dev_f1 = f1_score(dev_true, predict_in_chunks(clf, dev_texts),
                          average="macro", zero_division=0)
        print(f"epoch {epoch + 1}/{args.epochs} done | dev macro F1 = {dev_f1:.4f}")
        if dev_f1 > best_dev_f1:
            best_epoch, best_dev_f1 = epoch + 1, dev_f1
            best_state = (clf.coef_.copy(), clf.intercept_.copy())

    clf.coef_, clf.intercept_ = best_state
    print(f"\nSelected epoch {best_epoch}/{args.epochs} (dev macro F1 = {best_dev_f1:.4f})")

    target_names = [id2label[i] for i in range(num_labels)]
    selection = {"selected_epoch": best_epoch, "epochs": args.epochs,
                 "selection_dev_f1_macro": best_dev_f1}
    evaluate(clf, "dev",  df_dev,  target_names, args.tag, extra=selection)
    evaluate(clf, "test", df_test, target_names, args.tag, extra=selection)


if __name__ == "__main__":
    main()
