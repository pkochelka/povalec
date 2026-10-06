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
    --trainer langmatched
                         classifier_training.py on train_langmatched.parquet
                         (preprocessing/build_language_matched_split.py): every language
                         downsampled to the pooled cluster mix, so the language carries no
                         cluster information; logit-adjusted loss as above

    --track group        data/EuroParl Custom/clusters_k4            (build_cluster_splits.py)
    --track national     data/EuroParl Custom/clusters_k{k}_national (build_national_cluster_splits.py)
    --suffix _nohr       ... clusters_k4_nohr / clusters_k{k}_national_nohr, run dir likewise

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
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EuroParl Custom"

TRAINERS = {
    "balanced": ("analysis.classifier_training_for_balanced_collapsed", "train_balanced"),
    "logitadj": ("analysis.classifier_training", "train"),
    "langmatched": ("analysis.classifier_training", "train_langmatched"),
}
# Dev/test files per trainer; the langmatched ones get the same (language, original/
# translated) matching as its train file (build_language_matched_split.py).
EVAL_SPLITS = {"langmatched": ("dev_langmatched", "test_langmatched")}


def track_dir(track, k, suffix=""):
    if track == "group":
        if k != 4:
            raise SystemExit("the group track exists for k=4 only")
        return DATA_ROOT / f"clusters_k4{suffix}"
    return DATA_ROOT / f"clusters_k{k}_national{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", choices=("group", "national"), required=True)
    parser.add_argument("--trainer", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--suffix", default="",
                        help="track variant: reads clusters_k4<suffix>/ or clusters_k<k>_national<suffix>/ "
                             "(slurm_preprocess_clusters.sh SUFFIX=) and writes runs/<track>-k<k>-<trainer><suffix>")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default: runs/<track>-k<k>-<trainer><suffix>")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and check paths, then stop before loading the trainer")
    speed = parser.add_argument_group("speed settings (defaults reproduce the original runs)")
    speed.add_argument("--max-len", type=int, default=None, help="token limit (default 512)")
    speed.add_argument("--no-grad-ckpt", action="store_true", help="turn gradient checkpointing off")
    speed.add_argument("--group-by-length", action="store_true",
                       help="batch texts of similar length (logit-adjusted trainers only)")
    speed.add_argument("--workers", type=int, default=None,
                       help="dataloader workers and tokenizer processes (default 2 / 1)")
    speed.add_argument("--snapshot-epochs", default="",
                       help="e.g. '3,6' or '3:6' (sbatch --export splits on commas): also keep the "
                            "model after these epochs as model_epoch<N>, with its own biases, "
                            "manifest and results (logit-adjusted trainers only)")
    args = parser.parse_args()
    snapshot_epochs = tuple(int(e) for e in re.split(r"[,:\s]+", args.snapshot_epochs) if e)
    if args.trainer == "balanced" and (args.group_by_length or snapshot_epochs):
        raise SystemExit("--group-by-length and --snapshot-epochs are only wired into the logit-adjusted trainer")

    module_name, train_split = TRAINERS[args.trainer]
    dev_split, test_split = EVAL_SPLITS.get(args.trainer, ("dev", "test"))
    data_dir = track_dir(args.track, args.k, args.suffix)
    needed = [data_dir / f"{name}.parquet" for name in (train_split, dev_split, test_split)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"missing split files (build the track first): {missing}")

    run_dir = (args.run_dir or PROJECT_ROOT / "runs" / f"{args.track}-k{args.k}-{args.trainer}{args.suffix}").resolve()
    print(f"trainer:   {module_name} (splits: {train_split}, {dev_split}, {test_split})")
    print(f"data:      {data_dir}")
    print(f"run dir:   {run_dir}")
    print(f"speed:     max_len={args.max_len or 512} grad_ckpt={not args.no_grad_ckpt} "
          f"group_by_length={args.group_by_length} workers={args.workers or 2} snapshots={snapshot_epochs}")
    if args.dry_run:
        return

    # Shared settings live in classifier_training and are read at call time by both
    # trainers; MAX_LEN is also copied into the balanced module's namespace, so set both.
    shared = importlib.import_module("analysis.classifier_training")
    overrides = {"GRADIENT_CHECKPOINTING": not args.no_grad_ckpt,
                 "GROUP_BY_LENGTH": args.group_by_length,
                 "SNAPSHOT_EPOCHS": snapshot_epochs}
    if args.max_len:
        overrides["MAX_LEN"] = args.max_len
    if args.workers:
        overrides.update(DATALOADER_WORKERS=args.workers, TOKENIZE_PROC=args.workers)

    trainer = importlib.import_module(module_name)
    for module in {shared, trainer}:
        for name, value in overrides.items():
            if hasattr(module, name):
                setattr(module, name, value)
    trainer.DATA_DIR = str(data_dir)
    trainer.OUTPUT_DIR = str(run_dir / "model")
    if hasattr(trainer, "TRAIN_SPLIT"):
        trainer.TRAIN_SPLIT = train_split
    if args.trainer in EVAL_SPLITS:
        trainer.DEV_SPLIT, trainer.TEST_SPLIT = dev_split, test_split
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)          # results_<tag>.txt is written to the working directory
    trainer.main()


if __name__ == "__main__":
    main()
