import numpy as np
import pandas as pd
from pathlib import Path
from utils import write_parquet_chunked

pd.options.future.infer_string = False

MULTIPARL_PARQUET  = Path("data/EuroParl Custom/preprocessed.parquet")
PARLEE_PARQUET     = Path("data/EuroParl Custom/preprocessed_parlee.parquet")
EU_DEBATES_PARQUET = Path("data/EuroParl Custom/preprocessed_eu_debates.parquet")
OUTPUT_DIR         = Path("data/EuroParl Custom")

MIN_TEXT_LEN     = 50
MIN_LANG_SAMPLES = 10_000
EVAL_SET_SIZE    = 50_000   # target rows for EACH of dev and test
RANDOM_STATE     = 42

# Dev and test are class-balanced: every EU Party contributes the same number of
# rows, and within each party the language mix follows the corpus-wide language
# distribution (so no party or language dominates the evaluation). Train keeps
# whatever is left over and so retains the natural, imbalanced priors that the
# logit-adjusted trainer needs. Speech groups (all translations of one speech,
# keyed on speaker+date) never cross between train/dev/test, preventing leakage.

df = pd.concat([
    pd.read_parquet(MULTIPARL_PARQUET),
    pd.read_parquet(PARLEE_PARQUET),
    pd.read_parquet(EU_DEBATES_PARQUET),
], ignore_index=True)
print(f"Combined: {len(df):,} rows")

df = df.drop_duplicates(subset=["text"])
df = df[df["text"].str.len() >= MIN_TEXT_LEN]
lang_counts = df["language"].value_counts()
kept_langs = lang_counts[lang_counts >= MIN_LANG_SAMPLES].index
df = df[df["language"].isin(kept_langs)]
df = df.dropna(subset=["EU Party"]).reset_index(drop=True)
print(f"After dedup + length + language filter: {len(df):,} rows ({len(kept_langs)} languages kept)")

# Group key: all translations of the same speech share speaker+date.
df["group"] = df["speaker"].astype(str) + "_" + df["date"].astype(str)
print(f"Unique speech groups: {df['group'].nunique():,}")

parties = sorted(df["EU Party"].unique())
per_party = EVAL_SET_SIZE // len(parties)
lang_frac = df["language"].value_counts(normalize=True)
target_per_lang = {lang: max(1, round(per_party * frac)) for lang, frac in lang_frac.items()}
print(f"\n{len(parties)} parties x {sum(target_per_lang.values()):,} rows = "
      f"~{len(parties) * sum(target_per_lang.values()):,} rows per eval split")


def carve(sub, targets, rng):
    """Pull one class-balanced, language-stratified evaluation chunk out of `sub`.

    Whole speech groups are claimed for the chunk (so they can't leak into the
    remainder), then exactly `targets[lang]` rows per language are sampled from
    the claimed pool. Returns the selected row labels and the untouched rows.
    """
    groups = sub["group"].unique()
    rank = dict(zip(groups, rng.permutation(len(groups))))
    work = sub.assign(_grank=sub["group"].map(rank)).sort_values("_grank", kind="stable")
    work["_lang_cum"] = work.groupby("language").cumcount() + 1

    # Claim groups in random order until every language can meet its target.
    cutoff = 0
    for lang, target in targets.items():
        col = work[work["language"] == lang]
        if col.empty:
            continue
        reached = col[col["_lang_cum"] >= target]
        grank = reached["_grank"].iloc[0] if not reached.empty else col["_grank"].iloc[-1]
        cutoff = max(cutoff, int(grank))

    pool = work[work["_grank"] <= cutoff]
    leftover = work[work["_grank"] > cutoff].drop(columns=["_grank", "_lang_cum"])

    selected = []
    for lang, target in targets.items():
        rows = pool.index[pool["language"] == lang].to_numpy()
        take = min(len(rows), target)
        selected.append(rng.choice(rows, size=take, replace=False))
    return np.concatenate(selected), leftover


rng = np.random.default_rng(RANDOM_STATE)
dev_parts, test_parts, train_parts = [], [], []
for party in parties:
    sub = df[df["EU Party"] == party]
    dev_idx,  rest  = carve(sub,  target_per_lang, rng)
    test_idx, rest2 = carve(rest, target_per_lang, rng)
    dev_parts.append(dev_idx)
    test_parts.append(test_idx)
    train_parts.append(rest2.index.to_numpy())

dev   = df.loc[np.concatenate(dev_parts)]
test  = df.loc[np.concatenate(test_parts)]
train = df.loc[np.concatenate(train_parts)]

print(f"\nTrain: {len(train):,}  Dev: {len(dev):,}  Test: {len(test):,}")
for name, split in [("Dev", dev), ("Test", test)]:
    counts = split["EU Party"].value_counts()
    print(f"{name} party balance: min={counts.min():,} max={counts.max():,}")

cols = ["date", "EU Party", "text", "language", "speaker"]
for split, name in [(train, "train"), (dev, "dev"), (test, "test")]:
    path = OUTPUT_DIR / f"{name}.parquet"
    write_parquet_chunked(split[cols], path)
    print(f"  Saved {name} -> {path}")
