"""Build the k=4 cluster track labelled by each speech's NATIONAL party.

Reads  data/EuroParl Custom/cleaned/{train,dev,test}.parquet
       data/EuroParl Custom/national_parties/speeches.parquet   (build_national_party_map.py)
Writes data/EuroParl Custom/clusters_k4_national/{train,dev,test,train_balanced}.parquet
       data/EuroParl Custom/clusters_k4_national/labels.json

The group-level track (build_cluster_splits.py) moves whole EP groups into clusters.
This one looks up every row's national party instead, maps it onto the EU&I 2024 party
it became (see build_national_party_map.py), and labels the row with that party's k=4
cluster. A Romanian S&D speech (PSD, in the PSD-PNL list) and a German S&D speech (SPD)
can therefore land in different clusters, as the parties themselves do.

Rows whose national party does not map onto a clustered 2024 party are DROPPED -- UK
parties, parties that no longer exist, independents, and LinkedEP parties last seen
before 2009. There is no fallback to the EP-group label.

Row -> national party, by the row's `speaker` and `date` (the only speech identity that
survives preprocessing):
  LinkedEP rows   speaker is the MEP's URI: exact (URI, date) match against the LinkedEP
                  speeches; failing that, the same MEP's nearest-dated speech.
  ParlEE rows     speaker is a name: exact (name, date) match.
  EU Debates rows speaker is a name, and the corpus has no national party at all: the
                  same speaker's nearest-dated speech in ParlEE or LinkedEP.
Nearest-date matches are limited to --max-gap-days (default five years), so a 2023
speech is not labelled from 2014 when the MEP may have changed party since. An exact
match to a speech whose party is unmapped drops the row; it does not fall through to a
neighbouring date.
"""

import argparse
from pathlib import Path

import pandas as pd

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from analysis import party_kmeans as pk
from preprocessing.build_cluster_splits import (K, PARTY_COLUMN, cluster_parties, load_pool,
                                                split_and_write)
from preprocessing.build_national_party_map import LINKEDEP_PREFIX, name_key
from preprocessing.split_preprocessed_data import EVAL_SET_SIZE

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_INPUT_DIR = DATA_DIR / "cleaned"
DEFAULT_SPEECHES = DATA_DIR / "national_parties" / "speeches.parquet"
DEFAULT_OUTPUT_DIR = DATA_DIR / "clusters_k4_national"


def nearest(rows, lookup, key, max_gap):
    """For each row, the pui of the same `key`'s closest-dated lookup speech.

    Returns (pui, matched, exact) aligned to `rows`. `matched` is False when the key
    has no speech within `max_gap`; `exact` marks same-day matches.
    """
    # merge_asof rejects null keys; an undated or unnamed row simply stays unmatched.
    left = rows[[key, "date"]].dropna().rename_axis("row").reset_index().sort_values("date")
    right = lookup[[key, "date", "pui"]].rename(columns={"date": "match_date"}).sort_values("match_date")
    joined = pd.merge_asof(left, right, left_on="date", right_on="match_date", by=key,
                           direction="nearest", tolerance=max_gap).set_index("row").reindex(rows.index)
    return (joined["pui"], joined["match_date"].notna(),
            joined["match_date"].eq(joined["date"]) & joined["match_date"].notna())


def dedup_lookup(lookup, key):
    # Same speaker, same day, both corpora: one speech. Prefer the copy with a party.
    return (lookup.sort_values("pui", na_position="last")
                  .drop_duplicates([key, "date"]).dropna(subset=[key]))


def resolve(pool, speeches, max_gap_days):
    max_gap = pd.Timedelta(days=max_gap_days)
    pool = pool.copy()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    speeches = speeches.assign(date=pd.to_datetime(speeches["date"]).dt.normalize())

    is_uri = pool["speaker"].astype(str).str.startswith(LINKEDEP_PREFIX)
    pool["route"] = "name"
    pool.loc[is_uri, "route"] = "linkedep-uri"
    pool["pui"], pool["matched"], pool["exact"] = pd.NA, False, False

    uri_lookup = dedup_lookup(speeches[speeches["source"] == "linkedep"], "speaker")
    pui, matched, exact = nearest(pool[is_uri], uri_lookup, "speaker", max_gap)
    pool.loc[is_uri, "pui"], pool.loc[is_uri, "matched"], pool.loc[is_uri, "exact"] = pui, matched, exact

    named = pool[~is_uri].assign(speaker_key=lambda d: d["speaker"].map(name_key))
    name_lookup = dedup_lookup(speeches, "speaker_key")
    pui, matched, exact = nearest(named, name_lookup, "speaker_key", max_gap)
    pool.loc[~is_uri, "pui"], pool.loc[~is_uri, "matched"], pool.loc[~is_uri, "exact"] = pui, matched, exact
    return pool


def report(pool):
    pool = pool.assign(kept=pool["pui"].notna())
    summary = {}
    for route, df in pool.groupby("route"):
        summary[route] = {
            "rows": int(len(df)),
            "kept": round(float(df["kept"].mean()), 4),
            "exact_match": round(float(df["exact"].mean()), 4),
            "no_speaker_match": round(float((~df["matched"]).mean()), 4),
            "matched_but_party_unmapped": round(float((df["matched"] & ~df["kept"]).mean()), 4),
        }
        print(f"  {route:<13} {len(df):>10,} rows  kept {df['kept'].mean():6.1%}  "
              f"(same-day {df['exact'].mean():.1%}, no speaker match {(~df['matched']).mean():.1%})")
    by_year = pool.groupby(pool["date"].dt.year)["kept"].mean().round(3)
    by_lang = pool.groupby("language")["kept"].mean().round(3)
    print(f"  kept by year:     {by_year.to_dict()}")
    print(f"  kept by language: {by_lang.to_dict()}")
    summary["kept_by_year"] = {int(k): float(v) for k, v in by_year.items()}
    summary["kept_by_language"] = by_lang.to_dict()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--speeches", type=Path, default=DEFAULT_SPEECHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gap-days", type=int, default=5 * 365,
                        help="furthest a nearest-date speaker match may reach")
    parser.add_argument("--per-party", type=int, default=None,
                        help="Balanced-train rows per cluster (default: smallest cluster's count).")
    parser.add_argument("--eval-set-size", type=int, default=EVAL_SET_SIZE)
    parser.add_argument("--seed", type=int, default=42, help="split/balancing seed")
    parser.add_argument("--cluster-seed", type=int, default=0,
                        help="k-means seed; CLUSTER_NAMES were read off the seed-0 fit")
    return parser.parse_args()


def main():
    args = parse_args()
    parties = cluster_parties(args.cluster_seed)
    cluster_of = dict(zip(parties["PUI"].astype(int), parties["cluster_name"]))
    # Abbreviations repeat across countries (PS is Portuguese, Belgian and Slovak).
    label_of = dict(zip(parties["PUI"].astype(int),
                        parties["ABBREVIATON"].str.strip() + " (" + parties["COUNTRY"].str.strip() + ")"))

    speeches = pd.read_parquet(args.speeches)
    pool = resolve(load_pool(args.input_dir), speeches, args.max_gap_days)
    print("Row -> national party:")
    coverage = report(pool)

    kept = pool[pool["pui"].notna()].copy()
    kept["pui"] = kept["pui"].astype(int)
    kept[PARTY_COLUMN] = kept["pui"].map(cluster_of)
    missing = kept[PARTY_COLUMN].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum():,} rows map to PUIs outside the clustering: "
                         f"{sorted(kept.loc[missing, 'pui'].unique())[:10]}; rebuild national_parties/")
    print(f"\nKept {len(kept):,} of {len(pool):,} rows; clusters: "
          f"{kept[PARTY_COLUMN].value_counts().to_dict()}")
    rows_by_party = {}
    for (cluster, pui), n in kept.groupby([PARTY_COLUMN, "pui"]).size().sort_values(ascending=False).items():
        rows_by_party.setdefault(cluster, {})[label_of[pui]] = int(n)

    split_and_write(kept, args.output_dir, args.seed, args.per_party, args.eval_set_size, {
        "k": K,
        "cluster_seed": args.cluster_seed,
        "split_seed": args.seed,
        "clusters": pk.CLUSTER_NAMES[K],
        "max_gap_days": args.max_gap_days,
        "coverage": coverage,
        "rows_by_cluster_and_party": rows_by_party,
    }, "labels.json")


if __name__ == "__main__":
    main()
