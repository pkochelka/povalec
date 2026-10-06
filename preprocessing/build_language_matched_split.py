#!/usr/bin/env python3
"""Write train_langmatched.parquet: every language gets the same cluster mix.

train.parquet keeps the natural priors, and those differ by language -- Slovak is 25%
Sovereigntist right where most languages are ~10%, Romanian 73% Liberal-conservative.
The logit-adjusted loss only corrects the pooled class prior, so a model trained on it
still learns P(cluster | language) and applies it to any text in that language
(analysis/classifier_ngram_analysis.py, block 1).

This split removes that shortcut without throwing away as much as train_balanced does.
Each language is downsampled to the *pooled* cluster mix of train.parquet: for language
l with counts n(l, c) and pooled shares g(c), it keeps s_l * g(c) rows of cluster c,
where s_l = min_c n(l, c) / g(c) is the largest size the language can fill at that mix.
Rows are drawn uniformly at random within each (language, cluster) cell, so nothing is
duplicated and the speaker mix inside a cell is preserved in expectation. Language
becomes uninformative of the cluster; the pooled prior stays what the logit-adjusted
trainer expects. Keeps ~76% of train on the k=4 national NI track.

Usage, from the repository root:
  python preprocessing/build_language_matched_split.py
  python preprocessing/build_language_matched_split.py --track-dir "data/EuroParl Custom/clusters_k4_national"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-name", default="train_langmatched.parquet")
    args = parser.parse_args()

    source = args.track_dir / "train.parquet"
    keys = pq.read_table(source, columns=["language", PARTY_COLUMN]).to_pandas()
    counts = pd.crosstab(keys["language"], keys[PARTY_COLUMN])
    pooled = counts.sum(axis=0) / counts.to_numpy().sum()
    scale = (counts / pooled).min(axis=1)
    target = pd.DataFrame(np.floor(np.outer(scale, pooled)), index=counts.index,
                          columns=counts.columns).astype(int).clip(upper=counts)

    rng = np.random.default_rng(args.seed)
    keep = []
    for (language, cluster), rows in keys.groupby(["language", PARTY_COLUMN]).groups.items():
        keep.append(rng.choice(np.asarray(rows), size=target.loc[language, cluster], replace=False))
    keep = np.sort(np.concatenate(keep))

    table = pq.read_table(source).take(keep)
    out = args.track_dir / args.out_name
    # Written aside and renamed, so a concurrent job never reads a half-written file.
    partial = out.with_suffix(".parquet.partial")
    pq.write_table(table, partial)
    os.replace(partial, out)

    kept = keys.iloc[keep]
    mix = pd.crosstab(kept["language"], kept[PARTY_COLUMN], normalize="index")
    print(f"pooled mix: {pooled.round(4).to_dict()}")
    print(f"kept {len(keep):,} of {len(keys):,} rows ({len(keep) / len(keys):.1%})")
    print("per-language share kept:", (target.sum(axis=1) / counts.sum(axis=1)).round(2).to_dict())
    print(f"max deviation of any language's mix from pooled: {(mix - pooled).abs().to_numpy().max():.4f}")
    (args.track_dir / (Path(args.out_name).stem + ".json")).write_text(json.dumps({
        "source": str(source), "seed": args.seed, "rows": int(len(keep)),
        "pooled_mix": pooled.round(6).to_dict(), "rows_per_language_cluster": target.to_dict(orient="index"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
