#!/usr/bin/env python3
"""Write {train,dev,test}_langmatched.parquet: every language gets the same cluster mix.

train.parquet keeps the natural priors, and those differ by language -- Slovak is 25%
Sovereigntist right where most languages are ~10%, Romanian 73% Liberal-conservative.
The logit-adjusted loss only corrects the pooled class prior, so a model trained on it
still learns P(cluster | language) and applies it to any text in that language
(analysis/classifier_ngram_analysis.py, block 1).

This split removes that shortcut without throwing away as much as train_balanced does.
Each cell is downsampled to the *pooled* cluster mix of train.parquet: for a cell with
counts n(cell, c) and pooled shares g(c), it keeps s * g(c) rows of cluster c, where
s = min_c n(cell, c) / g(c) is the largest size the cell can fill at that mix. Rows are
drawn uniformly at random within each (cell, cluster), so nothing is duplicated and the
speaker mix inside a cell is preserved in expectation. The pooled prior stays what the
logit-adjusted trainer expects.

**Cells are (language, origin), not just language.** preprocess_data.py spreads each
LinkedEP speech over every language it was translated into, and translations stop after
2012; LinkedEP 2013-17, ParlEE and EU Debates are original-language only. So the
translated text of a language carries the EU-wide cluster mix, while its original text
comes from that country's own MEPs -- on the national NI track, original Slovak is 99%
Sovereigntist right, original Romanian 98% Liberal-conservative. Matching the language
as a whole leaves that intact, and AI-written text, which reads as original, inherits
the country's party mix. Origin is read off the data: a speech (speaker + date) present
in at most --original-max-languages languages is original, the rest translated (the
distribution is bimodal: 1-2 languages vs 11-21). A cell that cannot give every cluster
at least one row keeps nothing, which drops the original text of most smaller languages;
their translated text stays.
--ignore-origin gives the old per-language matching.

**--mt adds machine-translated train rows** (preprocessing/build_mt_train.py) as a third
origin, (language, "mt"), matched like the other cells but to train's own pooled mix
(without the MT rows), and writes train_langmatched<--out-suffix>.parquet. Those cells
hold non-translationese text of every cluster in every language -- what original text
lacks. Dev and test are not touched; the MT run reuses dev/test_langmatched.

**Dev and test get the same treatment**, but at a uniform mix: every (language, origin)
cell keeps equal rows per cluster, as many as its thinnest cluster has, so each split is
exactly class-uniform in every language. Otherwise test would still reward the shortcut
on the original text train no longer has, and dev would pick the checkpoint and fit the
biases on it. The annotated sample (pinned_test_speeches.csv) stays in test: those rows
fill their cell's quota first and are kept even beyond it; the count is printed and
stored in the .json. dev.parquet and test.parquet themselves are left as they are.

Usage, from the repository root:
  python preprocessing/build_language_matched_split.py
  python preprocessing/build_language_matched_split.py --track-dir "data/EuroParl Custom/clusters_k4"
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.split_preprocessed_data import pinned_rows

PARTY_COLUMN = "EU Party"
DEFAULT_TRACK_DIR = Path("data") / "EuroParl Custom" / "clusters_k4_national_ni"
PINNED_TEST_FILE = Path(__file__).resolve().parent / "pinned_test_speeches.csv"


def origin_of(keys, max_languages):
    """'original' where the speech (speaker + date) exists in <= max_languages languages.

    Speech groups never straddle train, dev and test (split_and_write asserts it), so
    each split alone holds every language version of its speeches."""
    n_lang = keys.groupby(["speaker", "date"])["language"].transform("nunique")
    return np.where(n_lang <= max_languages, "original", "translated")


def cell_targets(counts, uniform=False, pooled=None):
    """Rows to keep per (cell, cluster): each cell at the target mix, as large as it can fill.

    The target mix is the split's pooled mix, or equal shares with `uniform`, or `pooled`."""
    if uniform:
        pooled = pd.Series(1 / counts.shape[1], index=counts.columns)
    elif pooled is None:
        pooled = counts.sum(axis=0) / counts.to_numpy().sum()
    scale = (counts / pooled).min(axis=1)
    target = pd.DataFrame(np.floor(np.outer(scale, pooled)), index=counts.index,
                          columns=counts.columns).astype(int).clip(upper=counts)
    # A cell too small to give every cluster a row would keep a skewed remnant; drop it.
    target[target.min(axis=1) == 0] = 0
    return pooled, target


def match_split(track_dir, split, args, pinned=None):
    """Write <split>_langmatched.parquet (+ .json); return the json payload.

    `pinned` rows (test only: the annotated sample) are always kept. They fill their
    (cell, cluster) quota first; any beyond it, or in a cell that keeps nothing, are kept
    on top, reported, and taken back from the same cluster's other cells in that language,
    so the language as a whole keeps its mix."""
    source = track_dir / f"{split}.parquet"
    columns = ["language", PARTY_COLUMN, "speaker", "date"] + (["text"] if pinned is not None else [])
    keys = pq.read_table(source, columns=columns).to_pandas()
    keys["origin"] = "any" if args.ignore_origin else origin_of(keys, args.original_max_languages)
    n_split = len(keys)
    # The split's own pooled mix, before any MT rows: the MT variant targets the same mix
    # as the plain langmatched train, so the two runs differ only by the MT rows.
    own_mix = None
    if args.mt is not None:
        own_mix = keys[PARTY_COLUMN].value_counts() / n_split
        mt_keys = pq.read_table(args.mt, columns=["language", PARTY_COLUMN, "speaker", "date"]).to_pandas()
        mt_keys["origin"] = "mt"
        keys = pd.concat([keys, mt_keys], ignore_index=True)
    cell = ["language", "origin"]
    counts = pd.crosstab([keys["language"], keys["origin"]], keys[PARTY_COLUMN])
    pooled, target = cell_targets(counts, uniform=split != "train",
                                  pooled=None if own_mix is None else own_mix.reindex(counts.columns).fillna(0))

    forced = pd.Index([])
    if pinned is not None:
        keys["group"] = keys["speaker"].astype(str) + "_" + keys["date"].astype(str)
        forced, _ = pinned_rows(keys, pinned)
    is_forced = keys.index.isin(forced)

    # Pinned rows beyond their quota, per (language, cluster); the other cells pay for them.
    pinned_counts = (keys[is_forced].groupby(cell + [PARTY_COLUMN]).size()
                     .unstack(PARTY_COLUMN).reindex(index=target.index, columns=target.columns, fill_value=0)
                     .fillna(0).astype(int))
    excess = (pinned_counts - target).clip(lower=0)
    over_quota = int(excess.to_numpy().sum())
    owed = excess.groupby(level="language").sum()

    rng = np.random.default_rng(args.seed)
    keep = []
    for (language, origin, cluster), rows in keys.groupby(cell + [PARTY_COLUMN]).groups.items():
        rows = np.asarray(rows)
        must, free = rows[is_forced[rows]], rows[~is_forced[rows]]
        room = max(0, target.loc[(language, origin), cluster] - len(must))
        paid = min(room, owed.loc[language, cluster])
        owed.loc[language, cluster] -= paid
        keep += [must, rng.choice(free, size=min(room - paid, len(free)), replace=False)]
    keep = np.sort(np.concatenate(keep)).astype(int)

    # filter, not take: take concatenates each column's chunks first, which overflows the
    # 32-bit string offsets once the text column passes 2 GB. keep is sorted and unique, so
    # the mask selects the same rows in the same order.
    mask = np.zeros(len(keys), dtype=bool)
    mask[keep] = True
    table = pq.read_table(source).filter(mask[:n_split])
    if args.mt is not None:
        mt_table = pq.read_table(args.mt, columns=table.schema.names).filter(mask[n_split:])
        table = pa.concat_tables([table, mt_table.cast(table.schema)])
    out = track_dir / f"{split}_langmatched{args.out_suffix}.parquet"
    # Written aside and renamed, so a concurrent job never reads a half-written file.
    partial = out.with_suffix(".parquet.partial")
    pq.write_table(table, partial)
    os.replace(partial, out)

    kept = keys.iloc[keep]
    print(f"\n===== {split}: target mix {pooled.round(4).to_dict()}")
    print(f"kept {len(keep):,} of {len(keys):,} rows ({len(keep) / len(keys):.1%})")
    if pinned is not None:
        print(f"pinned rows kept: {int(is_forced.sum())} ({over_quota} beyond their cell's quota)")
    print("rows kept per cell (before -> after):")
    after = kept.groupby(cell).size().reindex(counts.index, fill_value=0)
    print(pd.DataFrame({"before": counts.sum(axis=1), "after": after})
          .unstack("origin").fillna(0).astype(int).to_string())
    deviation = {}
    for by in (["language"], cell):
        mix = pd.crosstab([kept[c] for c in by], kept[PARTY_COLUMN], normalize="index")
        deviation[" x ".join(by)] = float((mix - pooled).abs().to_numpy().max())
        print(f"max deviation of any {' x '.join(by)} mix from target: {deviation[' x '.join(by)]:.4f}")
    payload = {
        "source": str(source), "seed": args.seed, "rows": int(len(keep)),
        "by_origin": not args.ignore_origin, "original_max_languages": args.original_max_languages,
        "target_mix": pooled.round(6).to_dict(), "max_deviation": deviation,
        "pinned_kept": int(is_forced.sum()), "pinned_beyond_quota": int(over_quota),
        "mt": None if args.mt is None else str(args.mt),
        "mt_rows": int((kept["origin"] == "mt").sum()),
        "rows_per_cell_cluster": {f"{language}|{origin}": row
                                  for (language, origin), row in target.to_dict(orient="index").items()},
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        help="each <split>.parquet is matched into <split>_langmatched.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-origin", action="store_true",
                        help="match per language only, not per (language, original/translated)")
    parser.add_argument("--original-max-languages", type=int, default=2)
    parser.add_argument("--pinned", type=Path, default=PINNED_TEST_FILE,
                        help="speeches always kept in test (the annotated sample)")
    parser.add_argument("--mt", type=Path, default=None,
                        help="machine-translated train rows (preprocessing/build_mt_train.py), added to "
                             "train as (language, 'mt') cells at train's own pooled mix; train only")
    parser.add_argument("--out-suffix", default="",
                        help="write <split>_langmatched<suffix>.parquet, e.g. _mt next to the plain files")
    args = parser.parse_args()
    if args.mt is not None and args.splits != ["train"]:
        raise SystemExit("--mt adds rows to train only: pass --splits train")

    pinned = pd.read_csv(args.pinned, encoding="utf-8-sig", keep_default_na=False)
    # train goes last: the training job treats train_langmatched.json as "all three built".
    for split in sorted(args.splits, key=lambda name: name == "train"):
        match_split(args.track_dir, split, args, pinned if split == "test" else None)


if __name__ == "__main__":
    main()
