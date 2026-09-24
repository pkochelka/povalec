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
MIN_LANG_SAMPLES = 15_000   # languages with fewer rows are dropped (was 10k; raised to drop hr)
EVAL_SET_SIZE    = 50_000   # target rows for EACH of dev and test
RANDOM_STATE     = 42
# An eval split may take at most this share of a party's rows in any one language
# (see carve). A party short in a language gets a smaller quota there rather than
# having nearly all of its speech groups swept into the eval split and discarded.
MAX_EVAL_SHARE   = 0.25

COLUMNS = ["date", "EU Party", "text", "language", "speaker"]

# Dev and test are class-balanced: every EU Party contributes the same number of
# rows, and within each party the language mix follows the corpus-wide language
# distribution (so no party or language dominates the evaluation). Train keeps
# whatever is left over and so retains the natural, imbalanced priors that the
# logit-adjusted trainer needs. Speech groups (all translations of one speech,
# keyed on speaker+date) never cross between train/dev/test, preventing leakage.
#
# The carve/split logic below is imported by build_collapsed_splits so the
# ECR+ID-collapsed track can merge first and re-split with identical machinery.


def carve(sub, targets, rng, max_share=MAX_EVAL_SHARE):
    """Pull one class-balanced, language-stratified evaluation chunk out of `sub`.

    Whole speech groups are claimed for the chunk (so they can't leak into the
    remainder), then exactly `targets[lang]` rows per language are sampled from
    the claimed pool. Returns the selected row labels and the untouched rows.

    Claimed-but-unsampled rows are discarded (their groups belong to this chunk, so
    they cannot go to train), which is why each language's target is capped at
    `max_share` of the party's rows in that language. Uncapped, a language the party
    barely speaks -- a GUE/NGL row in Croatian -- cannot meet its target, pushes the
    cutoff to its LAST row, claims nearly every group and discards everything not
    sampled: that is how GUE/NGL lost ~90% of its rows before this cap.
    """
    available = sub["language"].value_counts()
    targets = {lang: min(target, int(available.get(lang, 0) * max_share))
               for lang, target in targets.items()}
    targets = {lang: target for lang, target in targets.items() if target > 0}
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


def load_combined():
    """Merge the three preprocessed sources and apply dedup / length / language
    filters.

    Deduplication is two-stage, by design:
      1. WITHIN each source, every text that occurs more than once is removed
         ENTIRELY (keep=False). An intra-source repeat is ambiguous -- the same
         wording surfaces under different speakers/parties/languages -- so no
         copy of it is trusted and all are dropped.
      2. The cleaned sources are THEN merged and a final cross-source dedup keeps
         ONE copy of each remaining text (keep="first"; by concat order
         multiparl > ParlEE > EU Debates), collapsing legitimate cross-source
         overlaps to a single row.
    """
    sources = {
        "multiparl":  MULTIPARL_PARQUET,
        "ParlEE":     PARLEE_PARQUET,
        "EU Debates": EU_DEBATES_PARQUET,
    }
    frames = []
    for name, path in sources.items():
        src = pd.read_parquet(path)
        before = len(src)
        src = src.drop_duplicates(subset=["text"], keep=False)
        print(f"  {name}: {before:,} -> {len(src):,} rows "
              f"(dropped {before - len(src):,} rows with >1 in-source occurrence)")
        frames.append(src)

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["text"])   # keep="first": cross-source overlaps
    print(f"Combined: {before:,} rows -> {len(df):,} after cross-source dedup")

    df = df[df["text"].str.len() >= MIN_TEXT_LEN]
    lang_counts = df["language"].value_counts()
    kept_langs = lang_counts[lang_counts >= MIN_LANG_SAMPLES].index
    dropped = lang_counts[lang_counts < MIN_LANG_SAMPLES]
    print(f"Languages below {MIN_LANG_SAMPLES:,} rows, dropped: {dropped.to_dict()}")
    df = df[df["language"].isin(kept_langs)]
    df = df.dropna(subset=["EU Party"]).reset_index(drop=True)
    print(f"After length + language filter: {len(df):,} rows "
          f"({len(kept_langs)} languages kept)")
    return df


def split_dataframe(df, eval_set_size=EVAL_SET_SIZE, seed=RANDOM_STATE):
    """Carve class-balanced, language-stratified, group-disjoint dev/test out of
    `df` and return (train, dev, test).

    `df` must already be deduped/filtered and carry the date/EU Party/text/
    language/speaker columns. A speech `group` (speaker+date) never crosses
    splits, so no translation of a speech can leak between train and dev/test.
    Every EU Party present in `df` is treated as a first-class label, so the
    eval splits are balanced by construction regardless of how the parties were
    labelled (e.g. after an ECR+ID merge).
    """
    df = df.copy()
    # Group key: all translations of the same speech share speaker+date.
    df["group"] = df["speaker"].astype(str) + "_" + df["date"].astype(str)
    print(f"Unique speech groups: {df['group'].nunique():,}")

    parties = sorted(df["EU Party"].unique())
    per_party = eval_set_size // len(parties)
    lang_frac = df["language"].value_counts(normalize=True)
    target_per_lang = {lang: max(1, round(per_party * frac)) for lang, frac in lang_frac.items()}
    print(f"{len(parties)} parties x {sum(target_per_lang.values()):,} rows = "
          f"~{len(parties) * sum(target_per_lang.values()):,} rows per eval split")

    rng = np.random.default_rng(seed)
    dev_parts, test_parts, train_parts = [], [], []
    for party in parties:
        sub = df[df["EU Party"] == party]
        dev_idx,  rest  = carve(sub,  target_per_lang, rng)
        test_idx, rest2 = carve(rest, target_per_lang, rng)
        # Rows of claimed groups that were not sampled are lost to every split; report
        # them so a party draining away is visible rather than silent.
        lost = len(sub) - len(dev_idx) - len(test_idx) - len(rest2)
        print(f"  {party}: {len(sub):,} rows -> dev {len(dev_idx):,}, test {len(test_idx):,}, "
              f"train {len(rest2):,}, discarded {lost:,} ({lost / len(sub):.1%})")
        dev_parts.append(dev_idx)
        test_parts.append(test_idx)
        train_parts.append(rest2.index.to_numpy())

    train = df.loc[np.concatenate(train_parts)]
    dev   = df.loc[np.concatenate(dev_parts)]
    test  = df.loc[np.concatenate(test_parts)]
    return train, dev, test


def main():
    df = load_combined()
    train, dev, test = split_dataframe(df)

    print(f"\nTrain: {len(train):,}  Dev: {len(dev):,}  Test: {len(test):,}")
    for name, split in [("Dev", dev), ("Test", test)]:
        counts = split["EU Party"].value_counts()
        print(f"{name} party balance: min={counts.min():,} max={counts.max():,}")

    for split, name in [(train, "train"), (dev, "dev"), (test, "test")]:
        path = OUTPUT_DIR / f"{name}.parquet"
        write_parquet_chunked(split[COLUMNS], path)
        print(f"  Saved {name} -> {path}")


if __name__ == "__main__":
    main()
