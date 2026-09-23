"""Build the k=4 cluster-collapsed split track under data/EuroParl Custom/clusters_k4.

Reads  data/EuroParl Custom/cleaned/{train,dev,test}.parquet
       data/euandi_2024_raw/EUandI_2024_party_dataset.csv   (via analysis/party_kmeans.py)
Writes data/EuroParl Custom/clusters_k4/{train,dev,test,train_balanced}.parquet
       data/EuroParl Custom/clusters_k4/label_mapping.json

The seven EP-group labels are replaced by the four "fictional EP groups" that k-means
finds in the EU&I 2024 party answers (analysis/party_kmeans.py, k=4):

    Radical left | Progressive federalists | Sovereigntist right |
    Liberal-conservative centre-right

**Why group-level, not speech-level.** The clusters are made of national parties, but
the speech corpora only carry the speaker's EP group -- no national party survives
preprocessing -- and most speeches predate the 2024 party system the clusters describe.
So every training label is mapped as a whole: onto the cluster that holds the most MEP
seats of its tenth-term successor group(s). This is the ECR+ID collapse again, with the
merge decided by the clustering instead of by hand. The mapping is recomputed from the
raw EU&I file on every run and written to label_mapping.json together with each label's
seat split, so how clean each merge is (its "purity") is on record: some of every
group's seats sit in another cluster, and a group-level mapping cannot follow them.

After relabelling, the cleaned splits are reunioned and re-split from scratch with the
same machinery as build_collapsed_splits.py -- class-balanced, language-stratified,
group-disjoint dev/test, a natural-prior train, and a shared-language-quota
train_balanced. The same caveat applies: these dev/test sets are NOT subsets of any
other track's, so never evaluate across tracks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from utils import write_parquet_chunked
from analysis import party_kmeans as pk
from preprocessing.split_preprocessed_data import EVAL_SET_SIZE, split_dataframe
from preprocessing.build_collapsed_splits import COLUMNS, PARTY_COLUMN, balance_party, shared_alloc

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_INPUT_DIR = DATA_DIR / "cleaned"
DEFAULT_OUTPUT_DIR = DATA_DIR / "clusters_k4"

K = 4
# The training corpus's labels (preprocess_data.py's mappings) -> the tenth-term groups
# that succeeded them, spelled as in the raw EU&I file. ID split in 2024 into Patriots
# (RN, Lega, FPÖ, VB) and ESN (AfD), so it takes both.
SUCCESSOR_GROUPS = {
    "GUE/NGL": ["LEFT"],
    "Greens/EFA": ["G/EFA"],
    "S&D": ["S&D"],
    "ALDE": ["RENEW"],
    "PPE": ["EPP"],
    "ECR": ["ECR"],
    "ID": ["PFE", "ESN"],
}
# party_kmeans.CLUSTER_NAMES is positional, read off one fit. One unambiguous member
# per cluster pins each name to its cluster, so a changed fit (new data, new seed)
# stops the run instead of silently swapping names.
ANCHORS = {
    "Radical left": ("Germany", "Linke"),
    "Progressive federalists": ("Germany", "SPD"),
    "Sovereigntist right": ("France", "RN"),
    "Liberal-conservative centre-right": ("Germany", "CDU/CSU"),
}


def cluster_parties(seed):
    """k=4 k-means over the MEP-holding parties, exactly as party_kmeans.py runs it."""
    df = pk.load_parties(PROJECT_ROOT / pk.RAW_CSV)
    pk.check_raw_columns(df, PROJECT_ROOT / pk.PARTIES_CODEBOOK)
    df = df[df["meps"] >= 1].reset_index(drop=True)
    X = KNNImputer(n_neighbors=5).fit_transform(pk.answer_matrix(df))
    df["cluster"] = pk.order_clusters(pk.fit(X, K, seed).labels_, df)
    df["cluster_name"] = [pk.CLUSTER_NAMES[K][c] for c in df["cluster"]]

    for name, (country, abbrev) in ANCHORS.items():
        row = df[(df["COUNTRY"].str.strip() == country) & (df["ABBREVIATON"].str.strip() == abbrev)]
        if len(row) != 1 or row["cluster_name"].iloc[0] != name:
            found = row["cluster_name"].tolist()
            raise SystemExit(f"cluster names no longer match the fit: {abbrev} ({country}) "
                             f"should be in '{name}', found {found}. Re-read the profiles "
                             f"from analysis/party_kmeans.py --k {K} and update CLUSTER_NAMES.")
    return df


def label_mapping(parties):
    """Each corpus label -> the cluster holding most of its successor groups' seats."""
    seats = pd.DataFrame(
        [(group, name, n) for by_group, name in zip(parties["seats_by_group"], parties["cluster_name"])
         for group, n in by_group.items()],
        columns=["group", "cluster", "seats"],
    ).pivot_table(index="group", columns="cluster", values="seats", aggfunc="sum", fill_value=0)

    mapping, report = {}, {}
    for label, successors in SUCCESSOR_GROUPS.items():
        row = seats.reindex(successors).fillna(0).sum()
        if row.sum() == 0:
            raise SystemExit(f"no MEP seats for {label}'s successor groups {successors}")
        mapping[label] = row.idxmax()
        report[label] = {
            "cluster": mapping[label],
            "successor_groups": successors,
            "seats_by_cluster": {c: int(n) for c, n in row.items() if n},
            "purity": round(float(row.max() / row.sum()), 3),
        }
    unmapped = set(pk.CLUSTER_NAMES[K]) - set(mapping.values())
    if unmapped:
        raise SystemExit(f"no corpus label maps onto {sorted(unmapped)}; the split would "
                         f"have fewer than {K} classes")
    return mapping, report


def load_pool(input_dir):
    """The three cleaned splits reunioned into one pool and deduplicated on text."""
    frames = [pd.read_parquet(input_dir / f"{name}.parquet")[COLUMNS]
              for name in ("train", "dev", "test")]
    pool = pd.concat(frames, ignore_index=True)
    before = len(pool)
    # As in build_collapsed_splits: name-cleaning can make distinct speeches identical.
    pool = pool.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Pool: {before:,} -> {len(pool):,} rows after reunion+dedup")
    return pool


def relabelled_pool(input_dir, mapping):
    pool = load_pool(input_dir)
    unknown = set(pool[PARTY_COLUMN].unique()) - set(mapping)
    if unknown:
        raise SystemExit(f"labels with no cluster mapping: {sorted(unknown)}")
    pool[PARTY_COLUMN] = pool[PARTY_COLUMN].map(mapping)
    print(f"Clusters: {pool[PARTY_COLUMN].value_counts().to_dict()}")
    return pool


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-party", type=int, default=None,
                        help="Balanced-train rows per cluster (default: smallest cluster's count).")
    parser.add_argument("--eval-set-size", type=int, default=EVAL_SET_SIZE,
                        help="Target rows for each of dev and test.")
    parser.add_argument("--seed", type=int, default=42, help="split/balancing seed")
    parser.add_argument("--cluster-seed", type=int, default=0,
                        help="k-means seed; CLUSTER_NAMES were read off the seed-0 fit")
    return parser.parse_args()


def split_and_write(pool, output_dir, seed, per_party, eval_set_size, metadata, metadata_name):
    """Re-split a relabelled pool, balance its train split, and write the track.

    Shared by this group-level track and the national-party one
    (build_national_cluster_splits.py): same carve, same leak checks, same
    shared-language-quota balancing, same four parquet files plus one JSON of metadata.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, dev, test = split_dataframe(pool, eval_set_size=eval_set_size, seed=seed)
    splits = {"train": train, "dev": dev, "test": test}
    for name, df in splits.items():
        counts = df[PARTY_COLUMN].value_counts()
        print(f"{name}: {len(df):,} rows, cluster balance min={counts.min():,} max={counts.max():,}")

    group = {name: set(df["speaker"].astype(str) + "_" + df["date"].astype(str))
             for name, df in splits.items()}
    assert not group["train"] & (group["dev"] | group["test"]), \
        "speech groups leak between train and dev/test"
    assert not group["dev"] & group["test"], "speech groups leak between dev and test"

    counts = train[PARTY_COLUMN].value_counts()
    target = per_party if per_party is not None else int(counts.min())
    alloc = shared_alloc(train, target)
    target = sum(alloc.values())
    print(f"\nTarget per cluster: {target:,}  ({len(counts)} clusters "
          f"-> {target * len(counts):,} rows)")
    balanced = pd.concat(
        [balance_party(train[train[PARTY_COLUMN] == c], target, rng, alloc)
         for c in sorted(counts.index)],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Balanced train: {len(balanced):,} rows")
    print(balanced.groupby([PARTY_COLUMN, "language"], observed=True)
          .size().unstack(fill_value=0).to_string())

    for df, name in [(train, "train"), (dev, "dev"), (test, "test"), (balanced, "train_balanced")]:
        path = output_dir / f"{name}.parquet"
        write_parquet_chunked(df[COLUMNS], path)
        print(f"Saved {name} -> {path}")

    path = output_dir / metadata_name
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved {metadata_name} -> {path}")


def main():
    args = parse_args()
    parties = cluster_parties(args.cluster_seed)
    mapping, report = label_mapping(parties)
    print("Label -> cluster (by MEP seats of the tenth-term successor groups):")
    for label, entry in report.items():
        print(f"  {label:<11} -> {entry['cluster']:<34} purity {entry['purity']:.0%}  "
              f"{entry['seats_by_cluster']}")

    pool = relabelled_pool(args.input_dir, mapping)
    split_and_write(pool, args.output_dir, args.seed, args.per_party, args.eval_set_size, {
        "k": K,
        "cluster_seed": args.cluster_seed,
        "split_seed": args.seed,
        "clusters": pk.CLUSTER_NAMES[K],
        "labels": report,
    }, "label_mapping.json")


if __name__ == "__main__":
    main()
