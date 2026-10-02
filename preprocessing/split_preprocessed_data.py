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
MIN_LANG_SAMPLES = 15_000   # languages with fewer rows are dropped (was 10k). hr passes
                            # this; the cluster tracks drop it (build_cluster_splits.MIN_LANG_ROWS)
EVAL_SET_SIZE    = 50_000   # target rows for EACH of dev and test
RANDOM_STATE     = 42
# An eval split may take at most this share of a party's rows in any one language
# (see carve). A party short in a language gets a smaller quota there rather than
# having nearly all of its speech groups swept into the eval split and discarded.
MAX_EVAL_SHARE   = 0.25
# ... and may claim at most this share of a party's speech groups. Claimed groups are
# lost to train whole, so this bounds what each eval split can discard: at 0.25 alone,
# a language GUE/NGL barely speaks still made dev and test each claim a quarter of its
# groups and discard 59.5% of its rows. Languages that cannot be filled within the
# cap are filled partially (see carve).
MAX_EVAL_GROUP_SHARE = 0.10

COLUMNS = ["date", "EU Party", "text", "language", "speaker"]
# preprocess_data.NI_LABEL; not imported, so this module does not pull in the chair step.
NI_LABEL     = "NI"
NON_INSCRITS = "non_inscrits"   # <OUTPUT_DIR>/non_inscrits.parquet, then cleaned/ likewise

# Dev and test are class-balanced: every EU Party contributes the same number of
# rows, and within each party the language mix follows the corpus-wide language
# distribution (so no party or language dominates the evaluation). Train keeps
# whatever is left over and so retains the natural, imbalanced priors that the
# logit-adjusted trainer needs. Speech groups (all translations of one speech,
# keyed on speaker+date) never cross between train/dev/test, preventing leakage.
#
# The carve/split logic below is imported by build_collapsed_splits so the
# ECR+ID-collapsed track can merge first and re-split with identical machinery.


def carve(sub, targets, rng, max_share=MAX_EVAL_SHARE, max_group_share=MAX_EVAL_GROUP_SHARE,
          required=(), forced_groups=()):
    """Pull one class-balanced, language-stratified evaluation chunk out of `sub`.

    Whole speech groups are claimed for the chunk (so they can't leak into the
    remainder), then exactly `targets[lang]` rows per language are sampled from
    the claimed pool. Returns the selected row labels and the untouched rows.

    Claimed-but-unsampled rows are discarded (their groups belong to this chunk, so
    they cannot go to train), which is why each language's target is capped at
    `max_share` of the party's rows in that language. Uncapped, a language the party
    barely speaks -- a GUE/NGL row in Croatian -- cannot meet its target, pushes the
    cutoff to its LAST row, claims nearly every group and discards everything not
    sampled: that is how GUE/NGL lost ~90% of its rows before this cap. The cutoff
    is also capped at `max_group_share` of the party's groups; a language whose
    target does not fit inside that many groups is filled with what is there.

    `required` row labels (pinned rows, see split_dataframe) are always selected: their
    groups are claimed first, whatever the caps, and they fill their language's target
    before the random draw tops it up. `forced_groups` are claimed first too, so none of
    their rows can reach the leftover (train), even when no row of theirs is required.
    """
    required = pd.Index(required).intersection(sub.index)
    available = sub["language"].value_counts()
    targets = {lang: min(target, int(available.get(lang, 0) * max_share))
               for lang, target in targets.items()}
    targets = {lang: target for lang, target in targets.items() if target > 0}
    groups = sub["group"].unique()
    forced = (set(forced_groups) & set(groups)) | set(sub.loc[required, "group"])
    order = rng.permutation(len(groups))
    # forced groups rank first (0..n-1), the rest keep their random order behind them
    rank = {g: (-1 if g in forced else 0, int(r)) for g, r in zip(groups, order)}
    rank = {g: i for i, g in enumerate(sorted(rank, key=rank.get))}
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
    cutoff = min(cutoff, max(0, int(np.ceil(len(groups) * max_group_share)) - 1))
    cutoff = max(cutoff, len(forced) - 1)

    pool = work[work["_grank"] <= cutoff]
    leftover = work[work["_grank"] > cutoff].drop(columns=["_grank", "_lang_cum"])

    selected = [required.to_numpy()] if len(required) else []
    for lang, target in targets.items():
        rows = pool.index[(pool["language"] == lang) & ~pool.index.isin(required)].to_numpy()
        take = min(len(rows), max(0, target - int((sub.loc[required, "language"] == lang).sum())))
        selected.append(rng.choice(rows, size=take, replace=False))
    return np.concatenate(selected), leftover


def load_combined(with_non_inscrits=False):
    """Merge the three preprocessed sources and apply dedup / length / language
    filters.

    Non-attached (NI) rows are taken out of every source before anything else, so the
    returned pool is the one the group tracks were always built from. With
    `with_non_inscrits`, returns (pool, non_inscrits): the NI rows under the same
    filters, minus any text that also occurs elsewhere in its source or in the pool.

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
    frames, ni_frames = [], []
    for name, path in sources.items():
        src = pd.read_parquet(path)
        is_ni = src["EU Party"] == NI_LABEL
        repeated = src["text"].duplicated(keep=False)
        ni_frames.append(src[is_ni & ~repeated])
        src = src[~is_ni]
        before = len(src)
        src = src.drop_duplicates(subset=["text"], keep=False)
        print(f"  {name}: {before:,} -> {len(src):,} rows "
              f"(dropped {before - len(src):,} rows with >1 in-source occurrence); "
              f"{(is_ni & ~repeated).sum():,} NI rows set aside")
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
    if not with_non_inscrits:
        return df

    ni = pd.concat(ni_frames, ignore_index=True)
    ni = ni[~ni["text"].isin(set(df["text"]))].drop_duplicates(subset=["text"])
    ni = ni[(ni["text"].str.len() >= MIN_TEXT_LEN) & ni["language"].isin(kept_langs)]
    ni = ni.reset_index(drop=True)
    print(f"Non-attached (NI) rows after the same filters: {len(ni):,}")
    return df, ni


def pinned_rows(df, pinned):
    """Where the `pinned` speeches (columns speaker, date, language, text) sit in `df`.

    Returns (rows, groups). rows: the pinned speeches themselves -- exact text first; a
    speech whose text is not found (cleaned differently) falls back to the same speaker,
    day and language. groups: the speech group (speaker+date) of every pinned speech,
    matched on speaker and day, so its translations stay out of train/dev even when the
    annotated language version itself is not in this pool."""
    by_text = df.index[df["text"].isin(set(pinned["text"]))]
    missing = pinned[~pinned["text"].isin(set(df.loc[by_text, "text"]))]
    day = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    speaker_day = df["speaker"].astype(str) + "|" + day
    key = speaker_day + "|" + df["language"]
    pin_day = pinned["speaker"].astype(str) + "|" + pd.to_datetime(pinned["date"]).dt.strftime("%Y-%m-%d")
    want = set(pin_day[missing.index] + "|" + missing["language"])
    by_key = df.index[key.isin(want)]
    found_keys = set(key[by_key])
    groups = set(df.loc[speaker_day.isin(set(pin_day)), "group"])
    print(f"Pinned to test: {len(pinned)} speeches -> {len(by_text)} by exact text, "
          f"{len(found_keys)} by speaker/day/language, "
          f"{len(want - found_keys)} not in this pool; "
          f"{len(groups)} speech groups held out of train/dev")
    return by_text.union(by_key), groups


def split_dataframe(df, eval_set_size=EVAL_SET_SIZE, seed=RANDOM_STATE, pinned=None):
    """Carve class-balanced, language-stratified, group-disjoint dev/test out of
    `df` and return (train, dev, test).

    `pinned` (optional DataFrame: speaker, date, language, text) lists speeches that must
    land in test, e.g. a human-annotated sample: their speech groups are kept out of dev,
    claimed first by the test carve, and the pinned rows themselves are always selected.
    They count toward test's per-party, per-language quotas, so test stays balanced.

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
    # carve() runs per party, so it keeps a group whole only if the group has ONE label.
    # A group under two labels (a missing speaker lumping a whole day together, two
    # namesakes, sources disagreeing on the speaker's group) would be split across
    # train and dev/test. Its label is ambiguous anyway: drop it.
    labels_per_group = df.groupby("group")["EU Party"].nunique()
    mixed = labels_per_group.index[labels_per_group > 1]
    if len(mixed):
        is_mixed = df["group"].isin(mixed)
        print(f"Dropping {len(mixed):,} speech groups ({is_mixed.sum():,} rows, "
              f"{is_mixed.mean():.2%}) that carry more than one label; largest: "
              f"{df.loc[is_mixed, 'group'].value_counts().head(5).to_dict()}")
        df = df[~is_mixed]

    parties = sorted(df["EU Party"].unique())
    per_party = eval_set_size // len(parties)
    lang_frac = df["language"].value_counts(normalize=True)
    target_per_lang = {lang: max(1, round(per_party * frac)) for lang, frac in lang_frac.items()}
    print(f"{len(parties)} parties x {sum(target_per_lang.values()):,} rows = "
          f"~{len(parties) * sum(target_per_lang.values()):,} rows per eval split")

    if pinned is not None and len(pinned):
        required, pinned_groups = pinned_rows(df, pinned)
    else:
        required, pinned_groups = pd.Index([]), set()

    rng = np.random.default_rng(seed)
    dev_parts, test_parts, train_parts = [], [], []
    for party in parties:
        sub = df[df["EU Party"] == party]
        held = sub["group"].isin(pinned_groups)
        dev_idx,  rest  = carve(sub[~held], target_per_lang, rng)
        rest = pd.concat([rest, sub[held]]) if held.any() else rest
        test_idx, rest2 = carve(rest, target_per_lang, rng, required=required,
                                forced_groups=pinned_groups)
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
    df, non_inscrits = load_combined(with_non_inscrits=True)
    train, dev, test = split_dataframe(df)

    print(f"\nTrain: {len(train):,}  Dev: {len(dev):,}  Test: {len(test):,}")
    for name, split in [("Dev", dev), ("Test", test)]:
        counts = split["EU Party"].value_counts()
        print(f"{name} party balance: min={counts.min():,} max={counts.max():,}")

    for split, name in [(train, "train"), (dev, "dev"), (test, "test")]:
        path = OUTPUT_DIR / f"{name}.parquet"
        write_parquet_chunked(split[COLUMNS], path)
        print(f"  Saved {name} -> {path}")
    # Unsplit: only the national-party track reads these, and it re-splits its own pool.
    path = OUTPUT_DIR / f"{NON_INSCRITS}.parquet"
    write_parquet_chunked(non_inscrits[COLUMNS], path)
    print(f"  Saved {len(non_inscrits):,} non-attached rows -> {path}")


if __name__ == "__main__":
    main()
