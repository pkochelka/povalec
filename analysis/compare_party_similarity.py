#!/usr/bin/env python3
"""Compare the party-similarity target matrix T from two independent sources.

  ngram  : symmetrized cross-validated confusion of the char-ngram bag-of-words
           model on the EuroParl train split (what classifier_training_multilabel
           currently uses; loaded from its party_similarity.json cache when valid,
           recomputed otherwise).
  mmbert : soft confusion of the trained mmBERT party classifier — mean
           temperature-calibrated softmax posterior per gold party on the uniform
           dev split (skipped when --mmbert-dir does not exist).
  euandi : similarity of the 7 EP groups' euandi 2024 position vectors
           (national parties aggregated to EP group). Two variants:
             corr      Pearson over statements after per-statement centering
                       across groups (consensus statements carry no signal)
             agreement 1 - mean(|pos_i - pos_j|) / 2 over shared statements

Both are mapped to targets exactly like confusion_to_targets (off-diagonal peak
scaled to NEIGHBOUR_TARGET, diagonal 1) and compared on the 21 off-diagonal
pairs: Pearson/Spearman, an exact Mantel permutation test (all 7! relabelings),
per-party neighbour rankings, and a heatmap + scatter figure.

The euandi matrix stays out of training either way — this is only a sanity check
of how well the data-driven ngram similarity agrees with the ideology-based one.

Usage:
  python analysis/compare_party_similarity.py [--recompute]
"""
import argparse
import itertools
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.classifier_training_multilabel import DATA_DIR, OUTPUT_DIR, attach_party
from analysis.classifier_training import SEED, select_device
from analysis.europarl_classification import PARTY_COLUMN, build_label_maps, load_split
from analysis.evaluate_euandi import PARTY_POSITIONS_PATH, load_party_positions

NGRAM_CACHE = os.path.join(OUTPUT_DIR, "party_similarity.json")
MMBERT_DIR = "mmBERT-base-balanced-train-first"
MMBERT_CACHE_NAME = "party_soft_confusion.json"
NEIGHBOUR_TARGET = 0.5
SIMILARITY_SAMPLES_PER_PARTY = 3000
SIMILARITY_FOLDS = 5
MIN_SHARED_STATEMENTS = 8
MIN_GROUPS_PER_STATEMENT = 3
SPEECH_SOURCES = ("ngram", "mmbert_hard", "mmbert")
IDEOLOGY_REFERENCE = "euandi_corr"
PLOT_PATH = os.path.join(PROJECT_ROOT, "plots", "party_similarity_comparison.png")


def balanced_subsample(party_ids, num_labels, rng):
    indices = []
    for party in range(num_labels):
        party_indices = np.flatnonzero(party_ids == party)
        if len(party_indices) > SIMILARITY_SAMPLES_PER_PARTY:
            party_indices = rng.choice(party_indices, SIMILARITY_SAMPLES_PER_PARTY, replace=False)
        indices.append(party_indices)
    return np.sort(np.concatenate(indices))


def party_confusion(texts, party_ids, num_labels):
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        memory=None,
    )
    folds = StratifiedKFold(n_splits=SIMILARITY_FOLDS, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(pipeline, texts, party_ids, cv=folds)
    return confusion_matrix(party_ids, predictions, labels=list(range(num_labels)))


def confusion_to_targets(confusion):
    rates = confusion / confusion.sum(axis=1, keepdims=True)
    similarity = (rates + rates.T) / 2.0
    np.fill_diagonal(similarity, 0.0)
    peak = similarity.max()
    targets = NEIGHBOUR_TARGET * similarity / peak if peak > 0 else np.zeros_like(similarity)
    np.fill_diagonal(targets, 1.0)
    return targets.astype(np.float32)


def train_party_names(data_dir):
    train = load_split("train", data_dir)
    return sorted(train[PARTY_COLUMN].unique().tolist()), train


def ngram_targets(parties, train, recompute):
    if not recompute and os.path.exists(NGRAM_CACHE):
        with open(NGRAM_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("labels") == parties and "confusion" in cache:
            print(f"ngram targets: loaded cache {NGRAM_CACHE}")
            return np.asarray(cache["target_matrix"], dtype=np.float64)
        print("ngram targets: cache invalid for ngram source, recomputing")

    label2id, _ = build_label_maps(parties)
    df = attach_party(train, label2id)
    rng = np.random.default_rng(SEED)
    subsample = balanced_subsample(df["party"].to_numpy(), len(parties), rng)
    print(f"ngram targets: cross-validating on {len(subsample):,} speeches")
    confusion = party_confusion(
        df["text"].to_numpy()[subsample], df["party"].to_numpy()[subsample], len(parties),
    )
    return confusion_to_targets(confusion).astype(np.float64)


def mmbert_confusions(parties, data_dir, model_dir, batch_size, recompute):
    """Per-gold-party confusion of the trained mmBERT model on uniform dev, both:
      soft  mean temperature-calibrated softmax posterior (full vector per speech)
      hard  argmax-only counts, row-normalized (each speech votes for one party)
    Returns (soft, hard); one dev pass accumulates both."""
    cache_path = os.path.join(model_dir, MMBERT_CACHE_NAME)
    if not recompute and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("labels") == parties and "confusion_hard" in cache:
            print(f"mmbert confusion: loaded cache {cache_path}")
            return (np.asarray(cache["confusion"], dtype=np.float64),
                    np.asarray(cache["confusion_hard"], dtype=np.float64))

    with open(os.path.join(model_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    label2id = manifest["label2id"]
    assert sorted(label2id, key=label2id.get) == parties, "model labels != train labels"

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    dev = attach_party(load_split("dev", data_dir), label2id)
    print(f"mmbert confusion: scoring {len(dev):,} dev speeches from {model_dir}")
    n = len(parties)
    soft_sums = np.zeros((n, n))
    hard_counts = np.zeros((n, n))
    counts = np.zeros(n)
    with torch.no_grad():
        for start in tqdm(range(0, len(dev), batch_size), desc="mmbert dev"):
            batch = dev.iloc[start:start + batch_size]
            encoded = tokenizer(
                batch["text"].tolist(), truncation=True, padding=True,
                max_length=manifest["max_len"], return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits / manifest["temperature"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            for gold, posterior, pred in zip(batch["party"].to_numpy(), probs, preds):
                soft_sums[gold] += posterior
                hard_counts[gold, pred] += 1
                counts[gold] += 1
    soft = soft_sums / counts[:, None]
    hard = hard_counts / counts[:, None]

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"labels": parties, "confusion": soft.tolist(),
                   "confusion_hard": hard.tolist(), "n_dev": int(counts.sum())},
                  f, ensure_ascii=False, indent=2)
    return soft, hard


def euandi_position_matrix(parties):
    """(group x statement) mean normalized_answer in [-1, 1]; NaN where unanswered."""
    pp = load_party_positions(os.path.join(PROJECT_ROOT, PARTY_POSITIONS_PATH))
    pp = pp.dropna(subset=["ep_group", "normalized_answer"])
    gt = pp.groupby(["ep_group", "statement_idx"])["normalized_answer"].mean()
    statements = sorted(gt.index.get_level_values("statement_idx").unique())
    Y = np.array([[gt.get((p, s), np.nan) for s in statements] for p in parties])
    print(f"euandi positions: {len(parties)} groups x {len(statements)} statements, "
          f"{np.isfinite(Y).sum()}/{Y.size} cells")
    return Y


def center_per_statement(Y):
    """Remove the across-group mean per statement -> stance deviations."""
    centered = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j])
        if m.sum() >= MIN_GROUPS_PER_STATEMENT:
            centered[m, j] = Y[m, j] - Y[m, j].mean()
    return centered


def pairwise(Y, stat):
    n = Y.shape[0]
    S = np.full((n, n), np.nan)
    np.fill_diagonal(S, 1.0)
    for i, j in itertools.combinations(range(n), 2):
        m = np.isfinite(Y[i]) & np.isfinite(Y[j])
        if m.sum() >= MIN_SHARED_STATEMENTS:
            S[i, j] = S[j, i] = stat(Y[i, m], Y[j, m])
    return S


def euandi_similarities(Y):
    corr = pairwise(center_per_statement(Y), lambda a, b: pearsonr(a, b)[0])
    agreement = pairwise(Y, lambda a, b: 1.0 - np.abs(a - b).mean() / 2.0)
    return {"euandi_corr": corr, "euandi_agreement": agreement}


def similarity_to_targets(similarity):
    """Same mapping as confusion_to_targets: peak off-diagonal -> NEIGHBOUR_TARGET."""
    targets = np.clip(similarity.copy(), 0.0, None)
    np.fill_diagonal(targets, 0.0)
    peak = np.nanmax(targets)
    targets = NEIGHBOUR_TARGET * targets / peak if peak > 0 else np.zeros_like(targets)
    np.fill_diagonal(targets, 1.0)
    return targets


def off_diagonal(targets):
    n = targets.shape[0]
    return np.array([targets[i, j] for i, j in itertools.combinations(range(n), 2)])


def mantel_test(reference, other):
    """Exact Mantel test: correlate off-diagonals under all party relabelings of `other`."""
    a = off_diagonal(reference)
    observed = pearsonr(a, off_diagonal(other))[0]
    count, total = 0, 0
    for perm in itertools.permutations(range(reference.shape[0])):
        permuted = other[np.ix_(perm, perm)]
        if pearsonr(a, off_diagonal(permuted))[0] >= observed - 1e-12:
            count += 1
        total += 1
    return observed, count / total


def matrix_frame(targets, parties):
    return pd.DataFrame(np.round(targets, 3), index=parties, columns=parties)


def neighbour_ranking(targets, parties, i):
    order = sorted((j for j in range(len(parties)) if j != i),
                   key=lambda j: targets[i, j], reverse=True)
    return ", ".join(f"{parties[j]}={targets[i, j]:.2f}" for j in order)


def report(parties, targets_by_source):
    for name, targets in targets_by_source.items():
        print(f"\n=== targets: {name} ===")
        print(matrix_frame(targets, parties).to_string())

    print("\n=== off-diagonal agreement (21 party pairs) ===")
    for name_a, name_b in itertools.combinations(targets_by_source, 2):
        a, b = targets_by_source[name_a], targets_by_source[name_b]
        r_mantel, p_mantel = mantel_test(a, b)
        rho = spearmanr(off_diagonal(a), off_diagonal(b))[0]
        print(f"{name_a:>16s} vs {name_b:<16s} pearson={r_mantel:+.3f}  "
              f"spearman={rho:+.3f}  "
              f"mantel p={p_mantel:.4f} (exact, {math.factorial(len(parties))} perms)")

    reference = targets_by_source[IDEOLOGY_REFERENCE]
    print(f"\n=== per-party neighbour rankings (row spearman vs {IDEOLOGY_REFERENCE}) ===")
    for i, party in enumerate(parties):
        print(f"\n{party}:")
        mask = np.arange(len(parties)) != i
        for name, targets in targets_by_source.items():
            if name == IDEOLOGY_REFERENCE:
                tag = "(ref)     "
            else:
                rho = spearmanr(reference[i, mask], targets[i, mask])[0]
                tag = f"rho={rho:+.3f} "
            print(f"  {name:18s} {tag} {neighbour_ranking(targets, parties, i)}")


def plot(parties, targets_by_source, path):
    names = list(targets_by_source)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(5.2 * (len(names) + 1), 4.6))
    for ax, name in zip(axes, names):
        targets = targets_by_source[name]
        im = ax.imshow(targets, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(parties)), parties, rotation=45, ha="right")
        ax.set_yticks(range(len(parties)), parties)
        ax.set_title(name)
        for i in range(len(parties)):
            for j in range(len(parties)):
                ax.text(j, i, f"{targets[i, j]:.2f}", ha="center", va="center",
                        color="white" if targets[i, j] < 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[-1]
    pairs = list(itertools.combinations(range(len(parties)), 2))
    x = off_diagonal(targets_by_source[IDEOLOGY_REFERENCE])
    speech_sources = [s for s in SPEECH_SOURCES if s in targets_by_source]
    for k, name in enumerate(speech_sources):
        y = off_diagonal(targets_by_source[name])
        ax.scatter(x, y, label=name, alpha=0.8)
        if k == 0:
            for (i, j), xv, yv in zip(pairs, x, y):
                ax.annotate(f"{parties[i]}–{parties[j]}", (xv, yv), fontsize=6, alpha=0.8)
    ax.set_xlabel(f"{IDEOLOGY_REFERENCE} target")
    ax.set_ylabel("speech-based target")
    ax.set_title("off-diagonal pairs")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"\nSaved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--mmbert-dir", default=MMBERT_DIR,
                        help="trained classifier dir for the soft confusion; skipped if missing")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the cached ngram/mmbert confusions and recompute them")
    return parser.parse_args()


def main():
    args = parse_args()
    parties, train = train_party_names(args.data_dir)
    print(f"{len(parties)} parties: {parties}")

    targets_by_source = {"ngram": ngram_targets(parties, train, args.recompute)}
    if os.path.isdir(args.mmbert_dir):
        soft, hard = mmbert_confusions(
            parties, args.data_dir, args.mmbert_dir, args.batch_size, args.recompute,
        )
        for name, confusion in (("mmbert", soft), ("mmbert_hard", hard)):
            kind = "mean calibrated posterior" if name == "mmbert" else "argmax counts"
            print(f"\n=== raw similarity: {name} ({kind}) ===")
            print(matrix_frame(confusion, parties).to_string())
            targets_by_source[name] = confusion_to_targets(confusion).astype(np.float64)
    else:
        print(f"mmbert source skipped: {args.mmbert_dir} not found")
    similarities = euandi_similarities(euandi_position_matrix(parties))
    for name, similarity in similarities.items():
        print(f"\n=== raw similarity: {name} ===")
        print(matrix_frame(similarity, parties).to_string())
        targets_by_source[name] = similarity_to_targets(similarity)

    report(parties, targets_by_source)
    plot(parties, targets_by_source, PLOT_PATH)


if __name__ == "__main__":
    main()
