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
party (default: the smallest party's count), water-filled toward an equal
per-language split, most-recent-first within each (party, language) cell with a
seeded random tiebreak.

--with-translations instead builds ONLY balanced_train_with_translations.parquet:
the same merge-then-split produces the collapsed train, then originals and cache
translations are balanced as separate strata, so every party contributes the
same, maximal number of original rows AND the same, maximal number of translated
rows (each stratum's target is its min count across parties). Rows carry a
boolean `translated` provenance column.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import write_parquet_chunked
from split_preprocessed_data import split_dataframe

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_INPUT_DIR = DATA_DIR / "cleaned"
DEFAULT_OUTPUT_DIR = DATA_DIR / "collapsed"
DEFAULT_CACHE = DATA_DIR / "augmented" / "translations.cache.json"

COLUMNS = ["date", "EU Party", "text", "language", "speaker"]
PARTY_COLUMN = "EU Party"
MERGE_PARTIES = ("ECR", "ID")
MERGED_LABEL = "ECR+ID"

# augment_translations plans distinct same-speaker-same-day speeches as
# `group#r<rank>`; the suffix must be stripped to resolve the source group.
RANK_SUFFIX = re.compile(r"#r\d+$")


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


def reconstruct_augmented(train, cache_path):
    """Rebuild augmented rows from the translation cache, keeping only
    train-origin groups.

    Cache keys are `group|||language`, where group is `speaker_date` with an
    optional `#r<rank>` suffix; metadata (party, date, speaker) is inherited
    from the source group, so relabelled parties carry over. Dropping keys
    whose group is absent from train removes dev/test-origin translations
    (leakage) and orphans in one check.
    """
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)

    group = train["speaker"].astype(str) + "_" + train["date"].astype(str)
    meta = train.assign(group=group).drop_duplicates("group").set_index("group")
    party_of = meta[PARTY_COLUMN].to_dict()
    date_of = meta["date"].to_dict()
    speaker_of = meta["speaker"].to_dict()

    rows = []
    dropped = 0
    for key, value in cache.items():
        grp, language = key.rsplit("|||", 1)
        grp = RANK_SUFFIX.sub("", grp)
        if grp not in party_of:
            dropped += 1
            continue
        # New cache stores {"text", "model"}; legacy stored a bare string.
        text = value["text"] if isinstance(value, dict) else value
        rows.append((date_of[grp], party_of[grp], text, language, speaker_of[grp]))

    aug = pd.DataFrame(rows, columns=COLUMNS)
    aug["date"] = pd.to_datetime(aug["date"])
    print(f"Cache: {len(cache):,} entries -> {len(aug):,} train-origin rows "
          f"({dropped:,} dropped as dev/test/orphan)")
    return aug


def water_fill(capacities, total):
    """Distribute `total` over languages, equal shares capped at each capacity.

    Returns {language: allocation}. Languages that cannot meet the equal share
    are taken in full; the freed-up deficit is re-spread over the rest until the
    full `total` is placed (assumes sum(capacities) >= total).
    """
    alloc = dict.fromkeys(capacities, 0)
    active = [lang for lang, cap in capacities.items() if cap > 0]
    remaining = min(total, sum(capacities.values()))

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Hand out the final remainder one row at a time, fullest cells first.
            for lang in sorted(active, key=lambda x: capacities[x] - alloc[x], reverse=True):
                if remaining == 0:
                    break
                alloc[lang] += 1
                remaining -= 1
            break
        for lang in list(active):
            give = min(share, capacities[lang] - alloc[lang])
            alloc[lang] += give
            remaining -= give
            if alloc[lang] >= capacities[lang]:
                active.remove(lang)
    return alloc


def balance_party(pdf, target, rng):
    """Select `target` rows from one party's frame, language-water-filled,
    most-recent-first within each language."""
    capacities = pdf["language"].value_counts().to_dict()
    alloc = water_fill(capacities, target)

    work = pdf.assign(_tie=rng.random(len(pdf)))
    work = work.sort_values(["date", "_tie"], ascending=[False, True], kind="stable")
    work["_rank"] = work.groupby("language").cumcount()
    work["_quota"] = work["language"].map(alloc)
    return work[work["_rank"] < work["_quota"]].drop(columns=["_tie", "_rank", "_quota"])


def build_translated_balanced(args, rng):
    """Build balanced_train_with_translations.parquet: originals and cache
    translations balanced as separate strata, each to its own maximal common
    per-party count (the stratum's min across parties)."""
    splits = build_merged_splits(args.input_dir, args.seed)
    train = splits["train"][COLUMNS]
    print(f"train: {len(train):,} rows, parties after collapse: "
          f"{train[PARTY_COLUMN].value_counts().to_dict()}")

    augmented = reconstruct_augmented(train, args.cache)

    combined = pd.concat([train.assign(translated=False),
                          augmented.assign(translated=True)], ignore_index=True)
    before = len(combined)
    # keep="first" + originals-first concat order: originals win collisions.
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Text dedup: {before:,} -> {len(combined):,} rows")

    parts = []
    for translated, stratum in combined.groupby("translated"):
        label = "translated" if translated else "original"
        counts = stratum[PARTY_COLUMN].value_counts()
        target = int(counts.min())
        print(f"\n{label} per-party counts:")
        for party in counts.sort_values().index:
            print(f"  {party}: {counts[party]:,}")
        print(f"{label} target per party: {target:,}")
        for party in sorted(counts.index):
            parts.append(balance_party(
                stratum[stratum[PARTY_COLUMN] == party], target, rng))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nBalanced train with translations: {len(balanced):,} rows")
    print("Party x origin balance:")
    print(balanced.groupby([PARTY_COLUMN, "translated"], observed=True)
          .size().unstack(fill_value=0).to_string())
    print("\nLanguage balance (overall):")
    print(balanced["language"].value_counts().to_string())
    print(f"\nDate range kept: {balanced['date'].min()} .. {balanced['date'].max()}")

    path = args.output_dir / "balanced_train_with_translations.parquet"
    write_parquet_chunked(balanced[COLUMNS + ["translated"]], path)
    print(f"Saved balanced_train_with_translations -> {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-party", type=int, default=None,
                        help="Balanced-train rows per party (default: smallest party's count).")
    parser.add_argument("--with-translations", action="store_true",
                        help="Build ONLY balanced_train_with_translations.parquet (originals and "
                             "cache translations balanced as separate strata); the four standard "
                             "outputs are left untouched.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Translation cache for --with-translations.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.with_translations:
        build_translated_balanced(args, rng)
        return

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
    print(f"\nTarget per party: {target:,}  ({len(party_counts)} parties "
          f"-> {target * len(party_counts):,} rows)")

    parts = []
    for party in sorted(party_counts.index):
        sub = train[train[PARTY_COLUMN] == party]
        parts.append(balance_party(sub, min(target, len(sub)), rng))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nBalanced train: {len(balanced):,} rows")
    print("Party balance:")
    print(balanced[PARTY_COLUMN].value_counts().to_string())
    print("\nLanguage balance (overall):")
    print(balanced["language"].value_counts().to_string())
    print(f"\nDate range kept: {balanced['date'].min()} .. {balanced['date'].max()}")

    for df, name in [(splits["train"], "train"), (splits["dev"], "dev"),
                     (splits["test"], "test"), (balanced, "train_balanced")]:
        path = args.output_dir / f"{name}.parquet"
        write_parquet_chunked(df[COLUMNS], path)
        print(f"Saved {name} -> {path}")


if __name__ == "__main__":
    main()
