"""Build the ECR+ID-collapsed split track under data/EuroParl Custom/collapsed.

Reads  data/EuroParl Custom/cleaned/{train,dev,test}.parquet
Writes data/EuroParl Custom/collapsed/{train,dev,test,train_balanced}.parquet

ECR and ID are merged into a single "ECR+ID" label BEFORE splitting. The three
cleaned splits are reunioned into one pool, relabelled, and then re-split from
scratch with the same class-balanced, language-stratified, group-disjoint carve
that split_preprocessed_data uses. ECR+ID is therefore a first-class party from
the very start: dev and test come out balanced by construction (no post-hoc
downsampling, no discarded eval rows), and every speech group (speaker+date)
lands in exactly one split.

NOTE: because this track RE-SPLITS, its dev/test are NOT subsets of the original
7-party splits -- a group in collapsed-train may have sat in the original
dev/test and vice versa. The two tracks' eval sets are independent; never
evaluate a model trained on one track against the other track's test set.

train.parquet keeps the natural, imbalanced priors (relabelled only), for the
logit-adjusted trainer. train_balanced.parquet downsamples it to equal rows per
party (default: the smallest party's count), most-recent-first within each
(party, language) cell with a seeded random tiebreak.

The per-language quota is SHARED: it is water-filled once over the elementwise
MINIMUM capacity across parties, so every party is handed the identical language
profile. Balancing each party independently (--free-language-mix) instead lets a
party that is short in a small language (bg/ro/et/hu) re-spread its deficit onto
the languages where it is plentiful (en/de/fr), which makes language a leaky cue
for party -- the exact confound a party classifier should not be able to use.
The shared quota costs rows only when no single party is the floor in every
language; it is otherwise free.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import water_fill, write_parquet_chunked
from split_preprocessed_data import split_dataframe

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_INPUT_DIR = DATA_DIR / "cleaned"
DEFAULT_OUTPUT_DIR = DATA_DIR / "collapsed"

COLUMNS = ["date", "EU Party", "text", "language", "speaker"]
PARTY_COLUMN = "EU Party"
MERGE_PARTIES = ("ECR", "ID")
MERGED_LABEL = "ECR+ID"


def build_merged_splits(input_dir, seed):
    """Reunion the cleaned splits, merge ECR+ID, and re-split from scratch.

    Returns {"train","dev","test"} DataFrames. Because ECR+ID is merged before
    the carve, it is treated as a normal party and dev/test come out
    class-balanced with no downsampling and no discarded eval rows.
    """
    frames = [pd.read_parquet(input_dir / f"{name}.parquet")[COLUMNS]
              for name in ("train", "dev", "test")]
    pool = pd.concat(frames, ignore_index=True)
    before = len(pool)
    # Name-cleaning can turn once-distinct speeches identical (the party name it
    # stripped was the only difference), so dedup the reunioned pool again.
    pool = pool.drop_duplicates(subset=["text"]).reset_index(drop=True)
    pool[PARTY_COLUMN] = np.where(
        pool[PARTY_COLUMN].isin(MERGE_PARTIES), MERGED_LABEL, pool[PARTY_COLUMN])
    print(f"Pool: {before:,} -> {len(pool):,} rows after reunion+dedup; "
          f"parties after collapse: {pool[PARTY_COLUMN].value_counts().to_dict()}")

    train, dev, test = split_dataframe(pool, seed=seed)
    return {"train": train, "dev": dev, "test": test}


def shared_alloc(df, target):
    """Water-fill one per-language quota that EVERY party can satisfy.

    Capacity per language is the minimum count across parties, so the returned
    allocation is simultaneously feasible for all of them and hands each the
    identical language profile. The reachable per-party total is therefore
    sum(min capacity), which may be below `target`; water_fill clamps to it.
    """
    caps = df.groupby([PARTY_COLUMN, "language"], observed=True).size().unstack(fill_value=0)
    return water_fill(caps.min(axis=0).to_dict(), target)


def balance_party(pdf, target, rng, alloc=None):
    """Select rows from one party's frame, most-recent-first within each language.

    `alloc` is a {language: quota} plan; when omitted the party is water-filled
    on its own capacities to `target` rows (the --free-language-mix behaviour).
    """
    if alloc is None:
        alloc = water_fill(pdf["language"].value_counts().to_dict(), target)

    work = pdf.assign(_tie=rng.random(len(pdf)))
    work = work.sort_values(["date", "_tie"], ascending=[False, True], kind="stable")
    work["_rank"] = work.groupby("language").cumcount()
    work["_quota"] = work["language"].map(alloc).fillna(0)
    return work[work["_rank"] < work["_quota"]].drop(columns=["_tie", "_rank", "_quota"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-party", type=int, default=None,
                        help="Balanced-train rows per party (default: smallest party's count).")
    parser.add_argument("--free-language-mix", action="store_true",
                        help="Water-fill each party's language quota independently (legacy). "
                             "Default shares one quota across parties so every party gets the "
                             "identical language profile.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_merged_splits(args.input_dir, args.seed)
    for name in ("train", "dev", "test"):
        counts = splits[name][PARTY_COLUMN].value_counts()
        print(f"{name}: {len(splits[name]):,} rows, party balance "
              f"min={counts.min():,} max={counts.max():,}")

    # carve guarantees group disjointness; these asserts document the invariant.
    group = {name: set(df["speaker"].astype(str) + "_" + df["date"].astype(str))
             for name, df in splits.items()}
    assert not group["train"] & (group["dev"] | group["test"]), \
        "speech groups leak between train and dev/test"
    assert not group["dev"] & group["test"], "speech groups leak between dev and test"

    train = splits["train"]
    party_counts = train[PARTY_COLUMN].value_counts()
    target = args.per_party if args.per_party is not None else int(party_counts.min())
    print("\nTrain per-party counts:")
    for party in party_counts.sort_values().index:
        print(f"  {party}: {party_counts[party]:,}")
    alloc = None if args.free_language_mix else shared_alloc(train, target)
    if alloc is not None:
        target = sum(alloc.values())
    print(f"\nTarget per party: {target:,}  ({len(party_counts)} parties "
          f"-> {target * len(party_counts):,} rows)")

    parts = []
    for party in sorted(party_counts.index):
        sub = train[train[PARTY_COLUMN] == party]
        parts.append(balance_party(sub, min(target, len(sub)), rng, alloc))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nBalanced train: {len(balanced):,} rows")
    print("Party balance:")
    print(balanced[PARTY_COLUMN].value_counts().to_string())
    print("\nParty x language:")
    print(balanced.groupby([PARTY_COLUMN, "language"], observed=True)
          .size().unstack(fill_value=0).to_string())
    print(f"\nDate range kept: {balanced['date'].min()} .. {balanced['date'].max()}")

    for df, name in [(splits["train"], "train"), (splits["dev"], "dev"),
                     (splits["test"], "test"), (balanced, "train_balanced")]:
        path = args.output_dir / f"{name}.parquet"
        write_parquet_chunked(df[COLUMNS], path)
        print(f"Saved {name} -> {path}")


if __name__ == "__main__":
    main()
