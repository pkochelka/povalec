#!/usr/bin/env python3
"""Raw logits of a trained classifier on its human dev and test splits.

The trainer keeps only the dev-fitted biases and the test report, not the logits, so
nothing downstream can refit a calibration (temperature, per-class bias) without
rerunning the model. This dumps them once: logits (no bias, no temperature), gold
label ids in the manifest's label order, and language, one .npz per split.

Usage, from the repository root (GPU; see analysis/slurm_dump_eval_logits.sh):
  python analysis/dump_eval_logits.py --classifier runs/national-k4-logitadj_ni/model \
      --data-dir "data/EuroParl Custom/clusters_k4_national_ni"
or locally on CPU with --sample 4000; then fit and compare calibrations locally with analysis/calibrate_classifier.py.

Manifests written by the trainer since the langmatched runs record their data_dir and
dev/test split names (e.g. dev_langmatched, test_langmatched); --data-dir and --splits
default to those, so the logits come from the dev the manifest biases were fit on.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from analysis.classify_speeches import load_classifier, predict_class_logits
from analysis.europarl_classification import PARTY_COLUMN, load_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", required=True, help="model dir with manifest.json")
    parser.add_argument("--data-dir", default="", help="track dir with <split>.parquet (default: the manifest's data_dir)")
    parser.add_argument("--splits", default="", help="default: the manifest's dev,test splits, else dev,test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample", type=int, default=0,
                        help="random texts per split (0 = all); a few thousand fit a calibration on CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="", help="default: <classifier>/eval_logits")
    args = parser.parse_args()
    manifest = json.loads((Path(args.classifier) / "manifest.json").read_text(encoding="utf-8"))
    args.data_dir = args.data_dir or manifest.get("data_dir", "")
    if not args.data_dir:
        parser.error("--data-dir is required: the manifest records no data_dir")
    splits = manifest.get("splits", {})
    args.splits = args.splits or f"{splits.get('dev', 'dev')},{splits.get('test', 'test')}"
    return args


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = load_classifier(args.classifier, device)
    label2id = {label: i for i, label in enumerate(clf.labels)}
    out_dir = Path(args.out_dir or Path(args.classifier) / "eval_logits")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"classifier {args.classifier} on {device}; labels {clf.labels}")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        df = load_split(split, args.data_dir, keep_labels=list(label2id))
        if 0 < args.sample < len(df):
            df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        texts = df["text"].tolist()
        # longest first: padding per batch stays small and an OOM shows up at once
        order = np.argsort([-len(t) for t in texts], kind="stable")
        logits = np.zeros((len(texts), len(clf.labels)), dtype=np.float32)
        for start in tqdm(range(0, len(texts), args.batch_size), desc=split):
            rows = order[start:start + args.batch_size]
            logits[rows] = predict_class_logits([texts[i] for i in rows], clf, device)
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, logits=logits, labels=df[PARTY_COLUMN].map(label2id).to_numpy(),
                            language=df["language"].to_numpy(dtype=str), label_names=np.array(clf.labels))
        print(f"wrote {path} ({len(texts)} texts)")


if __name__ == "__main__":
    main()
