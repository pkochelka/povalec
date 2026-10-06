#!/usr/bin/env python3
"""Write train_langmatched.parquet: every language gets the same cluster mix.

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
import pyarrow.parquet as pq

PARTY_COLUMN = "EU Party"
DEFAULT_TRACK_DIR = Path("data") / "EuroParl Custom" / "clusters_k4_national_ni"


def origin_of(keys, max_languages):
    """'original' where the speech (speaker + date) exists in <= max_languages languages.

    Speech groups never straddle train and dev/test (split_and_write asserts it), so
    train.parquet alone holds every language version of its speeches."""
    n_lang = keys.groupby(["speaker", "date"])["language"].transform("nunique")
    return np.where(n_lang <= max_languages, "original", "translated")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-name", default="train_langmatched.parquet")
    parser.add_argument("--ignore-origin", action="store_true",
                        help="match per language only, not per (language, original/translated)")
    parser.add_argument("--original-max-languages", type=int, default=2)
    args = parser.parse_args()

    source = args.track_dir / "train.parquet"
    keys = pq.read_table(source, columns=["language", PARTY_COLUMN, "speaker", "date"]).to_pandas()
    keys["origin"] = "any" if args.ignore_origin else origin_of(keys, args.original_max_languages)
    cell = ["language", "origin"]
    counts = pd.crosstab([keys["language"], keys["origin"]], keys[PARTY_COLUMN])
    pooled = counts.sum(axis=0) / counts.to_numpy().sum()
    scale = (counts / pooled).min(axis=1)
    target = pd.DataFrame(np.floor(np.outer(scale, pooled)), index=counts.index,
                          columns=counts.columns).astype(int).clip(upper=counts)
    # A cell too small to give every cluster a row would keep a skewed remnant; drop it.
    target[target.min(axis=1) == 0] = 0

    rng = np.random.default_rng(args.seed)
    keep = []
    for (language, origin, cluster), rows in keys.groupby(cell + [PARTY_COLUMN]).groups.items():
        keep.append(rng.choice(np.asarray(rows), size=target.loc[(language, origin), cluster], replace=False))
    keep = np.sort(np.concatenate(keep))

    table = pq.read_table(source).take(keep)
    out = args.track_dir / args.out_name
    # Written aside and renamed, so a concurrent job never reads a half-written file.
    partial = out.with_suffix(".parquet.partial")
    pq.write_table(table, partial)
    os.replace(partial, out)

    kept = keys.iloc[keep]
    before, after = counts.sum(axis=1), target.sum(axis=1)
    print(f"pooled mix: {pooled.round(4).to_dict()}")
    print(f"kept {len(keep):,} of {len(keys):,} rows ({len(keep) / len(keys):.1%})")
    print("rows kept per cell (before -> after):")
    print(pd.DataFrame({"before": before, "after": after}).unstack("origin").fillna(0).astype(int).to_string())
    for by in (["language"], cell):
        mix = pd.crosstab([kept[c] for c in by], kept[PARTY_COLUMN], normalize="index")
        print(f"max deviation of any {' x '.join(by)} mix from pooled: "
              f"{(mix - pooled).abs().to_numpy().max():.4f}")
    (args.track_dir / (Path(args.out_name).stem + ".json")).write_text(json.dumps({
        "source": str(source), "seed": args.seed, "rows": int(len(keep)),
        "by_origin": not args.ignore_origin, "original_max_languages": args.original_max_languages,
        "pooled_mix": pooled.round(6).to_dict(),
        "rows_per_cell_cluster": {f"{language}|{origin}": row
                                  for (language, origin), row in target.to_dict(orient="index").items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
