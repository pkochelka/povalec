#!/usr/bin/env python3
"""Does a trained classifier lean on person names? Remove them at test time and see.

Runs a trained model on one eval split three ways, with no retraining:

  raw       the split as it is
  names     titled person names deleted (preprocessing/clean_person_names.strip_names,
            names only -- "the report by Mr Morillon" -> "the report by")
  control   the same number of words deleted, but at a random place in the text

Only rows that contain a titled name ("affected") differ between the conditions, so
every comparison is reported on those rows as well as on the whole split. Reading it:

  names ~ raw                 names carry no signal; removing them costs nothing
  names < raw, control ~ raw  the model uses names as a cue: dev/test overstate how it
                              does on name-free text (LLM answers), worth removing them
  names ~ control < raw       any deletion of that size hurts; not name-specific

Prediction is argmax(logits + manifest biases), the trainer's own rule. The 95% CIs
are paired bootstrap intervals over affected rows. Names beyond the model's max_len
tokens are never seen by it, so rows whose only name sits past the cut dilute the
effect slightly toward zero.

    python analysis/name_ablation.py --model_dir runs/national-k4-logitadj/model \\
        --data_dir "data/EuroParl Custom/clusters_k4_national" --split test
"""
import argparse
import difflib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from analysis.europarl_classification import PARTY_COLUMN, load_split
from analysis.test_classifier import load_model, predict_logits
from preprocessing.clean_person_names import strip_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "runs" / "national-k4-logitadj" / "model"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "clusters_k4_national"


def control_text(text, k, rng):
    """`text` with `k` consecutive words deleted at a random position."""
    words = text.split()
    if len(words) <= k:
        return text
    start = int(rng.integers(0, len(words) - k + 1))
    return " ".join(words[:start] + words[start + k:])


def removed_words(before, after):
    a, b = before.split(), after.split()
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return [" ".join(a[i1:i2]) for tag, i1, i2, _, _ in ops if tag in ("delete", "replace")]


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def paired_bootstrap(correct_a, correct_b, n_boot, rng):
    """95% CI of mean(correct_b) - mean(correct_a) over rows resampled in pairs."""
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_b[idx].mean() - correct_a[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def summarise(name, y, pred):
    return {"condition": name, "n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="the track the model was trained on")
    parser.add_argument("--split", default="test",
                        help="test by default: dev chose the epoch and fit the biases")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--examples", type=int, default=10,
                        help="affected rows printed whose prediction flipped")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <model_dir>/../name_ablation_<split>.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"--- device: {args.device} | model: {args.model_dir} | data: {args.data_dir} [{args.split}] ---")
    tokenizer, model, manifest = load_model(str(args.model_dir), args.device)
    label2id = manifest["label2id"]
    labels = sorted(label2id, key=label2id.get)
    bias = np.asarray(manifest.get("biases", [0.0] * len(labels)))
    max_len = manifest["max_len"]

    df = load_split(args.split, args.data_dir, keep_labels=set(label2id))
    texts = df["text"].tolist()
    y = df[PARTY_COLUMN].map(label2id).to_numpy()
    langs = df["language"].to_numpy()

    stripped = [strip_names(t, lead_punct=False) for t in texts]
    affected = np.array([s != t for s, t in zip(stripped, texts)])
    idx = np.flatnonzero(affected)
    k_removed = [max(1, len(texts[i].split()) - len(stripped[i].split())) for i in idx]
    control = [control_text(texts[i], k, rng) for i, k in zip(idx, k_removed)]
    print(f"{len(df):,} rows; {affected.sum():,} ({affected.mean():.1%}) contain a titled name; "
          f"mean {np.mean(k_removed):.1f} words removed per affected row")

    def run(batch):
        return predict_logits(batch, tokenizer, model, max_len, args.device, args.batch_size)

    logits_raw = run(texts)
    logits_names = run([stripped[i] for i in idx])
    logits_ctrl = run(control)

    pred_raw = (logits_raw + bias).argmax(axis=1)
    pred_names_aff = (logits_names + bias).argmax(axis=1)
    pred_ctrl_aff = (logits_ctrl + bias).argmax(axis=1)
    pred_names_all = pred_raw.copy()
    pred_names_all[idx] = pred_names_aff

    y_aff, raw_aff = y[idx], pred_raw[idx]
    ok_raw, ok_names, ok_ctrl = raw_aff == y_aff, pred_names_aff == y_aff, pred_ctrl_aff == y_aff
    p_true = lambda logits: softmax(logits + bias)[np.arange(len(idx)), y_aff]
    p_raw, p_names, p_ctrl = p_true(logits_raw[idx]), p_true(logits_names), p_true(logits_ctrl)

    ci_names = paired_bootstrap(ok_raw.astype(float), ok_names.astype(float), args.bootstrap, rng)
    ci_ctrl = paired_bootstrap(ok_raw.astype(float), ok_ctrl.astype(float), args.bootstrap, rng)
    # The name-specific part: names removed vs the same amount of random deletion.
    ci_names_vs_ctrl = paired_bootstrap(ok_ctrl.astype(float), ok_names.astype(float), args.bootstrap, rng)

    result = {
        "model_dir": str(args.model_dir), "data_dir": str(args.data_dir), "split": args.split,
        "rows": int(len(df)), "affected_rows": int(affected.sum()),
        "whole_split": [summarise("raw", y, pred_raw), summarise("names removed", y, pred_names_all)],
        "affected": [summarise("raw", y_aff, raw_aff), summarise("names removed", y_aff, pred_names_aff),
                     summarise("control deletion", y_aff, pred_ctrl_aff)],
        "affected_accuracy_delta": {
            "names_minus_raw": float(ok_names.mean() - ok_raw.mean()), "names_ci95": ci_names,
            "control_minus_raw": float(ok_ctrl.mean() - ok_raw.mean()), "control_ci95": ci_ctrl,
            "names_minus_control": float(ok_names.mean() - ok_ctrl.mean()),
            "names_vs_control_ci95": ci_names_vs_ctrl,
        },
        "affected_flips": {
            "names_changed_prediction": float((pred_names_aff != raw_aff).mean()),
            "names_right_to_wrong": int((ok_raw & ~ok_names).sum()),
            "names_wrong_to_right": int((~ok_raw & ok_names).sum()),
            "control_changed_prediction": float((pred_ctrl_aff != raw_aff).mean()),
        },
        "affected_mean_p_true": {"raw": float(p_raw.mean()), "names removed": float(p_names.mean()),
                                 "control deletion": float(p_ctrl.mean())},
        "per_language_affected": {},
        "per_class_affected_recall": {},
    }
    for lang in sorted(set(langs[idx])):
        m = langs[idx] == lang
        if m.sum() >= 50:
            result["per_language_affected"][lang] = {
                "n": int(m.sum()), "acc_raw": float(ok_raw[m].mean()),
                "acc_names": float(ok_names[m].mean()), "acc_control": float(ok_ctrl[m].mean())}
    for c, label in enumerate(labels):
        m = y_aff == c
        if m.any():
            result["per_class_affected_recall"][label] = {
                "n": int(m.sum()), "raw": float(ok_raw[m].mean()),
                "names": float(ok_names[m].mean()), "control": float(ok_ctrl[m].mean())}

    print("\n=== whole split ===")
    for r in result["whole_split"]:
        print(f"  {r['condition']:<18} n={r['n']:>7,}  acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")
    print("\n=== rows with a titled name ===")
    for r in result["affected"]:
        print(f"  {r['condition']:<18} n={r['n']:>7,}  acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")
    d = result["affected_accuracy_delta"]
    print(f"\n  accuracy change, names removed: {d['names_minus_raw']:+.4f}  "
          f"95% CI [{ci_names[0]:+.4f}, {ci_names[1]:+.4f}]")
    print(f"  accuracy change, control:       {d['control_minus_raw']:+.4f}  "
          f"95% CI [{ci_ctrl[0]:+.4f}, {ci_ctrl[1]:+.4f}]")
    print(f"  names vs control (name-specific): {d['names_minus_control']:+.4f}  "
          f"95% CI [{ci_names_vs_ctrl[0]:+.4f}, {ci_names_vs_ctrl[1]:+.4f}]")
    f = result["affected_flips"]
    print(f"  predictions changed: names {f['names_changed_prediction']:.1%} "
          f"(right->wrong {f['names_right_to_wrong']:,}, wrong->right {f['names_wrong_to_right']:,}), "
          f"control {f['control_changed_prediction']:.1%}")
    p = result["affected_mean_p_true"]
    print(f"  mean p(true class): raw {p['raw']:.4f}, names removed {p['names removed']:.4f}, "
          f"control {p['control deletion']:.4f}")

    print("\n=== affected rows by language (acc raw -> names / control) ===")
    for lang, r in result["per_language_affected"].items():
        print(f"  {lang:<4} n={r['n']:>6,}  {r['acc_raw']:.3f} -> {r['acc_names']:.3f} / {r['acc_control']:.3f}")
    print("\n=== affected rows by true class (recall raw -> names / control) ===")
    for label, r in result["per_class_affected_recall"].items():
        print(f"  {label:<34} n={r['n']:>6,}  {r['raw']:.3f} -> {r['names']:.3f} / {r['control']:.3f}")

    flipped = np.flatnonzero(ok_raw & ~ok_names)[:args.examples]
    if len(flipped):
        print(f"\n=== {len(flipped)} rows the model got right only with the names in ===")
    for j in flipped:
        i = idx[j]
        print(f"  [{langs[i]}] true {labels[y[i]]} -> {labels[pred_names_aff[j]]}; "
              f"removed: {removed_words(texts[i], stripped[i])[:6]}")

    out = args.out or args.model_dir.parent / f"name_ablation_{args.split}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
