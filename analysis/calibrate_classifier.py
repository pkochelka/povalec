#!/usr/bin/env python3
"""Fit and compare calibrations of the classifier's logits on its human dev split.

The manifest's per-class biases are fit so that the argmax shares on dev match the dev
label shares. On a model whose logits are near one-hot that takes offsets of +-15 and
says nothing about the probabilities. The alternatives here are fit by likelihood on
dev and scored on test:

  raw          softmax(z)                      no bias, no temperature
  manifest     softmax(z + b_manifest)         the shipped inference rule
  ts           softmax(z / T)                  temperature only: argmax = raw labels
  manifest+ts  softmax((z + b_manifest) / T)   shipped labels, calibrated probabilities
  bcts         softmax((z + b) / T)            bias and temperature fit jointly by NLL

All are written as {biases, temperature} in the pipeline's convention
(analysis/classify_speeches.py: softmax((logits + biases) / temperature)), so any of
them can be dropped into a copy of manifest.json or passed to
analysis/classifier_ngram_analysis.py --calibration.

Needs <classifier>/eval_logits/{dev,test}.npz from analysis/dump_eval_logits.py, named
after the manifest's dev/test splits when it records them (dev_langmatched.npz, ...).

--apply METHOD writes that method's biases and temperature into manifest.json, so
classify_speeches and everything downstream use them; the manifest as it was is kept
once as manifest.uncalibrated.json, and the fit always starts from that copy's biases.
manifest+ts leaves every argmax label (and the trainer's test report) unchanged and only
calibrates the probabilities; bcts also moves labels.

Usage, from the repository root:
  python -m analysis.calibrate_classifier --classifier models/national-k4-logitadj_ni/model
  python -m analysis.calibrate_classifier --classifier runs/<run>/model_epoch2 --apply manifest+ts
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import f1_score


def load(path):
    data = np.load(path, allow_pickle=False)
    return data["logits"].astype(np.float64), data["labels"], data["language"], [str(n) for n in data["label_names"]]


def nll(z, y):
    return float(-log_softmax(z, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(z, y):
    res = minimize(lambda p: nll(z / np.exp(p[0]), y), x0=[0.0], method="L-BFGS-B")
    return float(np.exp(res.x[0]))


def fit_bcts(z, y):
    k = z.shape[1]

    def loss(p):
        t, c = np.exp(p[0]), np.r_[0.0, p[1:]]   # class 0 pinned: softmax is shift-invariant
        return nll(z / t + c, y)

    res = minimize(loss, x0=np.zeros(k), method="L-BFGS-B")
    t, c = float(np.exp(res.x[0])), np.r_[0.0, res.x[1:]]
    biases = c * t                               # softmax(z / t + c) = softmax((z + c t) / t)
    return biases - biases.mean(), t


def ece(probs, y, bins=15):
    conf, hit = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def evaluate(z, y, lang, biases, t, names):
    logp = log_softmax((z + biases) / t, axis=1)
    probs, pred = np.exp(logp), logp.argmax(1)
    per_lang = [f1_score(y[lang == l], pred[lang == l], average="macro") for l in np.unique(lang)]
    return {
        "accuracy": float((pred == y).mean()),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "f1_macro_mean_lang": float(np.mean(per_lang)),
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "ece": ece(probs, y),
        "mean_max_prob": float(probs.max(1).mean()),
        "pred_shares": dict(zip(names, np.round(np.bincount(pred, minlength=len(names)) / len(y), 3).tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--logits-dir", default="", help="default: <classifier>/eval_logits")
    parser.add_argument("--out", default="", help="default: <classifier>/calibration.json")
    parser.add_argument("--apply", default="", choices=("", "manifest+ts", "bcts", "ts", "manifest", "raw"),
                        help="write this method's biases and temperature into manifest.json")
    args = parser.parse_args()

    classifier = Path(args.classifier)
    logits_dir = Path(args.logits_dir or classifier / "eval_logits")
    manifest_path, original_path = classifier / "manifest.json", classifier / "manifest.uncalibrated.json"
    # After an --apply, manifest.json holds calibrated biases; refit from the trainer's.
    manifest = json.loads((original_path if original_path.exists() else manifest_path).read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    z_dev, y_dev, _, names = load(logits_dir / f"{splits.get('dev', 'dev')}.npz")
    z_test, y_test, lang_test, test_names = load(logits_dir / f"{splits.get('test', 'test')}.npz")
    assert names == test_names == sorted(manifest["label2id"], key=manifest["label2id"].get), "label order mismatch"
    b_manifest = np.asarray(manifest["biases"], dtype=float)
    zero = np.zeros(len(names))

    b_bcts, t_bcts = fit_bcts(z_dev, y_dev)
    methods = {
        "raw": (zero, 1.0),
        "manifest": (b_manifest, 1.0),
        "ts": (zero, fit_temperature(z_dev, y_dev)),
        "manifest+ts": (b_manifest, fit_temperature(z_dev + b_manifest, y_dev)),
        "bcts": (b_bcts, t_bcts),
    }
    out = {"classifier": str(classifier), "labels": names, "dev_n": int(len(y_dev)), "test_n": int(len(y_test)),
           "convention": "softmax((logits + biases) / temperature)", "methods": {}}
    print(f"dev {len(y_dev)} texts, test {len(y_test)} texts; labels {names}\n")
    print(f"{'method':12s} {'T':>7s}  biases{'':26s} {'acc':>6s} {'F1':>6s} {'F1lang':>6s} {'NLL':>6s} {'ECE':>6s} {'maxp':>6s}  test pred shares")
    for name, (biases, t) in methods.items():
        metrics = evaluate(z_test, y_test, lang_test, biases, t, names)
        out["methods"][name] = {"biases": np.round(biases, 4).tolist(), "temperature": round(t, 4), "test": metrics}
        print(f"{name:12s} {t:7.3f}  {str(np.round(biases, 2).tolist()):32s} {metrics['accuracy']:6.3f} "
              f"{metrics['f1_macro']:6.3f} {metrics['f1_macro_mean_lang']:6.3f} {metrics['nll']:6.3f} "
              f"{metrics['ece']:6.3f} {metrics['mean_max_prob']:6.3f}  {metrics['pred_shares']}")
    path = Path(args.out or classifier / "calibration.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    if args.apply:
        if not original_path.exists():
            original_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        biases, t = methods[args.apply]   # unrounded, so manifest+ts keeps every argmax label
        calibrated = dict(manifest, biases=[float(b) for b in biases], temperature=float(t),
                          calibration=args.apply,
                          inference="argmax(logits + biases); probabilities softmax((logits + biases) / temperature)")
        manifest_path.write_text(json.dumps(calibrated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"applied {args.apply} (T={t:.4f}) to {manifest_path}; "
              f"original kept as {original_path.name}")


if __name__ == "__main__":
    main()
