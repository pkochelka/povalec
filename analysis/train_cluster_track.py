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

--adversary origin adds a gradient-reversal origin adversary to the logit-adjusted trainers
(classifier_training.ADVERSARY): within each language the encoder is trained so that EP-
translated, original and machine-translated text cannot be told apart. --adv-weight sets
the maximum reversal strength; the default run dir gets _adv<weight> appended.
--adversary language removes language identity altogether (a head predicting the text's
language), --adversary both runs the language and origin heads together; their run dirs
get _advlanguage<weight> / _advboth<weight>. --adv-lr-mult trains the heads at a multiple
of the learning rate (run dir: _lr<mult>), so they keep up with the encoder.

--init-from runs/<run>/model starts from that fine-tuned checkpoint instead of mmBERT-base,
--train-per-cell N trains on at most N rows per (language, original/translated/mt) cell
(fixed by the seed, so two runs see the same rows), --max-epochs caps the epochs: together
a short warm-start experiment, e.g. an adversary run and its control. The default run dir
gets _warm / _pc<N> appended.

--evaluate-checkpoint runs/<run>/model/checkpoint-<step> skips training: it copies that
checkpoint's weights and tokenizer to runs/<run>/model_epoch<N> (N from its
trainer_state.json) and gives it dev-fitted biases, a manifest and test results, like a
--snapshot-epochs snapshot. Pass the same --track/--trainer/--suffix as the training run.
"""
import argparse
import importlib
import json
import multiprocessing
import os
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EuroParl Custom"

TRAINERS = {
    "balanced": ("analysis.classifier_training_for_balanced_collapsed", "train_balanced"),
    "logitadj": ("analysis.classifier_training", "train"),
    "langmatched": ("analysis.classifier_training", "train_langmatched"),
    # langmatched train plus machine-translated rows as (language, "mt") cells
    # (build_mt_train.py, then build_language_matched_split.py --mt --out-suffix _mt)
    "langmatched_mt": ("analysis.classifier_training", "train_langmatched_mt"),
}
# Dev/test files per trainer; the langmatched ones get the same (language, original/
# translated) matching as its train file (build_language_matched_split.py). The MT
# variant is evaluated on exactly the same files.
EVAL_SPLITS = {"langmatched": ("dev_langmatched", "test_langmatched"),
               "langmatched_mt": ("dev_langmatched", "test_langmatched")}


def safetensors_problem(path):
    """Why a .safetensors file is unreadable (missing, or shorter than its header says), else None."""
    if not path.exists():
        return f"{path} is missing"
    size = path.stat().st_size
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if size < 8 + header_len:
            return f"{path} is truncated inside its header ({size:,} bytes)"
        header = json.loads(f.read(header_len))
    expected = 8 + header_len + max((t["data_offsets"][1] for k, t in header.items() if k != "__metadata__"),
                                    default=0)
    if size != expected:
        return f"{path} is {size:,} bytes, its header says {expected:,} (incomplete copy or write)"
    return None


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
    speed.add_argument("--eval-batch", type=int, default=None,
                       help="per-GPU eval batch (default 32); larger only speeds up dev/test passes")
    speed.add_argument("--global-batch", type=int, default=None,
                       help="global train batch over all GPUs (default 32); changes the optimization, "
                            "so compare only runs with the same value")
    speed.add_argument("--lr", type=float, default=None, help="peak learning rate (default 2e-5)")
    speed.add_argument("--bf16", action="store_true", help="bf16 autocast instead of fp16")
    adv = parser.add_argument_group("gradient-reversal adversary (logit-adjusted trainers only)")
    adv.add_argument("--adversary", choices=("origin", "language", "both"), default=None,
                     help="origin: original/translated/mt within each language; language: the language "
                          "itself; both: the two heads together")
    adv.add_argument("--adv-lr-mult", type=float, default=1.0,
                     help="learning rate of the adversary heads as a multiple of --lr (default 1)")
    adv.add_argument("--adv-weight", type=float, default=0.3,
                     help="maximum gradient-reversal strength, reached ~30%% into training (default 0.3)")
    adv.add_argument("--adv-min-rows", type=int, default=200,
                     help="train rows an origin needs in a language to take part (default 200)")
    short = parser.add_argument_group("short warm-start experiments (logit-adjusted trainers only)")
    short.add_argument("--init-from", type=Path, default=None,
                       help="fine-tuned model dir to start from (same labels), instead of mmBERT-base")
    short.add_argument("--train-per-cell", type=int, default=0,
                       help="at most this many train rows per (language, origin) cell (0 = all)")
    short.add_argument("--max-epochs", type=int, default=None, help="default 6, with early stopping")
    short.add_argument("--save-only-model", action="store_true",
                       help="epoch checkpoints without optimizer state (~1.8 GB less each; cannot be resumed)")
    parser.add_argument("--evaluate-checkpoint", type=Path, default=None,
                        help="no training: turn this Trainer checkpoint into model_epoch<N> with "
                             "biases, manifest and results (logit-adjusted trainers only)")
    args = parser.parse_args()
    if "EVAL_CHECKPOINT" in os.environ and not args.evaluate_checkpoint:
        # Set but empty (an unset shell variable at submit time), or a stale
        # slurm_train_cluster_track.sh that drops it: either way it would retrain.
        raise SystemExit(f"EVAL_CHECKPOINT={os.environ['EVAL_CHECKPOINT']!r} but --evaluate-checkpoint "
                         "was not passed (empty path, or a stale slurm_train_cluster_track.sh): refusing to train")
    snapshot_epochs = tuple(int(e) for e in re.split(r"[,:\s]+", args.snapshot_epochs) if e)
    if args.trainer == "balanced" and (args.adversary or args.init_from or args.train_per_cell or args.max_epochs):
        raise SystemExit("--adversary/--init-from/--train-per-cell/--max-epochs are only wired into the "
                         "logit-adjusted trainers")
    if args.init_from:
        args.init_from = args.init_from.resolve()   # the trainer runs from the run dir
        if not (args.init_from / "config.json").exists():
            raise SystemExit(f"--init-from {args.init_from} has no config.json")
    if args.trainer == "balanced" and (args.group_by_length or snapshot_epochs):
        raise SystemExit("--group-by-length and --snapshot-epochs are only wired into the logit-adjusted trainer")
    if args.evaluate_checkpoint:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise SystemExit("--evaluate-checkpoint runs on one GPU; drop GPUS for it")
        if args.trainer == "balanced":
            raise SystemExit("--evaluate-checkpoint is only wired into the logit-adjusted trainer")
        state_file = args.evaluate_checkpoint / "trainer_state.json"
        if not state_file.exists():
            raise SystemExit(f"not a Trainer checkpoint (no trainer_state.json): {args.evaluate_checkpoint}")
        with open(state_file, encoding="utf-8") as f:
            checkpoint_epoch = round(json.load(f)["epoch"])
        snapshot_epochs = (checkpoint_epoch,)
        problem = safetensors_problem(args.evaluate_checkpoint / "model.safetensors")
        if problem:
            raise SystemExit(f"the checkpoint itself is broken: {problem}")

    module_name, train_split = TRAINERS[args.trainer]
    dev_split, test_split = EVAL_SPLITS.get(args.trainer, ("dev", "test"))
    data_dir = track_dir(args.track, args.k, args.suffix)
    needed = [data_dir / f"{name}.parquet" for name in (train_split, dev_split, test_split)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"missing split files (build the track first): {missing}")

    tags = (("_warm" if args.init_from else "") + (f"_pc{args.train_per_cell}" if args.train_per_cell else "")
            + (f"_adv{'' if args.adversary == 'origin' else args.adversary}{args.adv_weight:g}"
               if args.adversary else "")
            + (f"_lr{args.adv_lr_mult:g}" if args.adversary and args.adv_lr_mult != 1.0 else ""))
    run_dir = (args.run_dir or PROJECT_ROOT / "runs"
               / f"{args.track}-k{args.k}-{args.trainer}{args.suffix}{tags}").resolve()
    print(f"trainer:   {module_name} (splits: {train_split}, {dev_split}, {test_split})")
    print(f"data:      {data_dir}")
    print(f"run dir:   {run_dir}")
    print(f"speed:     max_len={args.max_len or 512} grad_ckpt={not args.no_grad_ckpt} "
          f"group_by_length={args.group_by_length} workers={args.workers or 2} snapshots={snapshot_epochs} "
          f"gpus={os.environ.get('WORLD_SIZE', '1')} eval_batch={args.eval_batch or 32} "
          f"global_batch={args.global_batch or 32} lr={args.lr or 2e-5} bf16={args.bf16}")
    if args.adversary:
        print(f"adversary: {args.adversary}, max reversal {args.adv_weight}, lr x{args.adv_lr_mult:g}, "
              f"min rows {args.adv_min_rows}")
    if args.init_from or args.train_per_cell or args.max_epochs:
        print(f"short:     init_from={args.init_from} train_per_cell={args.train_per_cell or 'all'} "
              f"max_epochs={args.max_epochs or 6}")
    if args.global_batch and args.global_batch % int(os.environ.get("WORLD_SIZE", "1")):
        raise SystemExit(f"--global-batch {args.global_batch} does not split over {os.environ['WORLD_SIZE']} GPUs")
    if args.evaluate_checkpoint:
        print(f"evaluate:  {args.evaluate_checkpoint} -> {run_dir / f'model_epoch{checkpoint_epoch}'} (no training)")
    if args.dry_run:
        return

    # Shared settings live in classifier_training and are read at call time by both
    # trainers; MAX_LEN is also copied into the balanced module's namespace, so set both.
    shared = importlib.import_module("analysis.classifier_training")
    overrides = {"GRADIENT_CHECKPOINTING": not args.no_grad_ckpt,
                 "GROUP_BY_LENGTH": args.group_by_length,
                 "SNAPSHOT_EPOCHS": snapshot_epochs,
                 "EVALUATE_ONLY": bool(args.evaluate_checkpoint)}
    if args.max_len:
        overrides["MAX_LEN"] = args.max_len
    if args.workers:
        overrides.update(DATALOADER_WORKERS=args.workers, TOKENIZE_PROC=args.workers)
    if args.eval_batch:
        overrides["EVAL_BATCH_SIZE"] = args.eval_batch
    if args.global_batch:
        overrides["GLOBAL_TRAIN_BATCH_SIZE"] = args.global_batch
    if args.lr:
        overrides["LEARNING_RATE"] = args.lr
    if args.bf16:
        overrides["BF16"] = True
    if args.adversary:
        overrides.update(ADVERSARY=args.adversary, ADVERSARY_WEIGHT=args.adv_weight,
                         ADVERSARY_MIN_ROWS=args.adv_min_rows, ADVERSARY_LR_MULT=args.adv_lr_mult)
    if args.init_from:
        overrides["INIT_FROM"] = str(args.init_from)
    if args.train_per_cell:
        overrides["TRAIN_PER_CELL"] = args.train_per_cell
    if args.max_epochs:
        overrides["MAX_EPOCHS"] = args.max_epochs
    if args.save_only_model:
        overrides["SAVE_ONLY_MODEL"] = True

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
    if args.evaluate_checkpoint:
        # Weights, config and tokenizer only; the optimizer state is not needed to predict.
        snapshot = run_dir / f"model_epoch{checkpoint_epoch}"
        snapshot.mkdir(exist_ok=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "trainer_state.json"):
            source, target = args.evaluate_checkpoint / name, snapshot / name
            # A copy cut short by an earlier job leaves a file of the wrong size: redo it,
            # via a temporary name so an interrupted copy never looks finished.
            if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(source, partial)
                os.replace(partial, target)
        problem = safetensors_problem(snapshot / "model.safetensors")
        if problem:
            raise SystemExit(f"copying the checkpoint failed (disk quota?): {problem}")
    os.chdir(run_dir)          # results_<tag>.txt is written to the working directory
    # Python 3.14 starts DataLoader workers with forkserver, which pickles the in-memory
    # tokenized train set into every worker: 8 workers = 9 copies, OOM at 48G. fork
    # shares it copy-on-write, as every Python up to 3.13 did.
    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    trainer.main()


if __name__ == "__main__":
    main()
