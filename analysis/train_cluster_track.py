#!/usr/bin/env python3
"""Train one of the existing EU-party classifiers on a cluster track, unmodified.

Both trainers hard-code the ECR+ID track (data/EuroParl Custom/collapsed) and a
fixed output directory and results-file name. They read those as module globals at
call time, so this wrapper points them elsewhere instead of forking them:

    --trainer balanced   classifier_training_for_balanced_collapsed.py
                         trains on train_balanced.parquet (uniform classes)
    --trainer logitadj   classifier_training.py
                         trains on train.parquet (natural priors) with logit-adjusted
                         cross-entropy, so the imbalance is corrected in the loss

    --track group        data/EuroParl Custom/clusters_k4            (build_cluster_splits.py)
    --track national     data/EuroParl Custom/clusters_k{k}_national (build_national_cluster_splits.py)

Everything a run writes -- the model directory, manifest.json and the trainer's
results_<tag>.txt, which is written to the working directory under a name that does
not mention the track -- goes to runs/<track>-k<k>-<trainer>/, so runs on different
tracks, and the existing ECR+ID results, never overwrite each other.

    python analysis/train_cluster_track.py --track group --trainer balanced
    python analysis/train_cluster_track.py --track national --trainer logitadj --dry-run
"""
import argparse
import importlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EuroParl Custom"

TRAINERS = {
    "balanced": ("analysis.classifier_training_for_balanced_collapsed", "train_balanced"),
    "logitadj": ("analysis.classifier_training", "train"),
}


def track_dir(track, k):
    if track == "group":
        if k != 4:
            raise SystemExit("the group track exists for k=4 only")
        return DATA_ROOT / "clusters_k4"
    return DATA_ROOT / f"clusters_k{k}_national"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", choices=("group", "national"), required=True)
    parser.add_argument("--trainer", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default: runs/<track>-k<k>-<trainer>")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and check paths, then stop before loading the trainer")
    args = parser.parse_args()

    module_name, train_split = TRAINERS[args.trainer]
    data_dir = track_dir(args.track, args.k)
    needed = [data_dir / f"{name}.parquet" for name in (train_split, "dev", "test")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"missing split files (build the track first): {missing}")

    run_dir = (args.run_dir or PROJECT_ROOT / "runs" / f"{args.track}-k{args.k}-{args.trainer}").resolve()
    print(f"trainer:   {module_name} (train split: {train_split}.parquet)")
    print(f"data:      {data_dir}")
    print(f"run dir:   {run_dir}")
    if args.dry_run:
        return

    trainer = importlib.import_module(module_name)
    trainer.DATA_DIR = str(data_dir)
    trainer.OUTPUT_DIR = str(run_dir / "model")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)          # results_<tag>.txt is written to the working directory
    trainer.main()


if __name__ == "__main__":
    main()
