"""Combine the glm translation cache with train.parquet and downsample to a
party-balanced (and, within each party, language-balanced) training set.

Augmented rows are reconstructed straight from the translation cache. A cache
key is `group|||language` where `group = speaker_date`. We only keep cache
entries whose group is present in train.parquet: this single check drops any
translation whose source speech now lives in dev/test (leakage) as well as
orphan groups that were filtered out of every split during preprocessing.

Downsampling makes every EU Party contribute the same number of rows
(`--per-party`, default = the smallest party's combined count). Inside a party
we aim for an equal per-language split, water-filling the deficit from rare
languages onto languages that still have surplus so the party still totals N.
Within each (party, language) cell the most recent speeches (by full `date`)
are kept first, with a seeded random tiebreak among rows sharing a date.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import write_parquet_chunked

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_TRAIN = DATA_DIR / "cleaned" / "train.parquet"
DEFAULT_CACHE = DATA_DIR / "augmented" / "translations.cache.json"
DEFAULT_OUTPUT = DATA_DIR / "cleaned" / "train_balanced.parquet"

COLUMNS = ["date", "EU Party", "text", "language", "speaker"]
PARTY_COLUMN = "EU Party"


def reconstruct_augmented(train, cache_path):
    """Rebuild augmented rows from the cache, keeping only train-origin groups."""
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-party", type=int, default=None,
                        help="Rows per party (default: smallest party's combined count).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    train = pd.read_parquet(args.train)[COLUMNS]
    print(f"Loaded {len(train):,} train rows from {args.train}")

    augmented = reconstruct_augmented(train, args.cache)

    combined = pd.concat([train, augmented], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Combined: {before:,} rows -> {len(combined):,} after text dedup")

    party_counts = combined[PARTY_COLUMN].value_counts()
    target = args.per_party if args.per_party is not None else int(party_counts.min())
    print("\nCombined per-party counts:")
    for party in party_counts.sort_values().index:
        print(f"  {party}: {party_counts[party]:,}")
    print(f"\nTarget per party: {target:,}  ({len(party_counts)} parties "
          f"-> {target * len(party_counts):,} rows)")

    parts = []
    for party in sorted(party_counts.index):
        sub = combined[combined[PARTY_COLUMN] == party]
        parts.append(balance_party(sub, min(target, len(sub)), rng))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nBalanced set: {len(balanced):,} rows")
    print("Party balance:")
    print(balanced[PARTY_COLUMN].value_counts().to_string())
    print("\nLanguage balance (overall):")
    print(balanced["language"].value_counts().to_string())
    print(f"\nDate range kept: {balanced['date'].min()} .. {balanced['date'].max()}")

    write_parquet_chunked(balanced[COLUMNS], args.output)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
