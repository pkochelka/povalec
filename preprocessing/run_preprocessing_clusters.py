"""Run the whole preprocessing pipeline through to the k=4 national-party cluster track.

The same driver as run_preprocessing.py -- same flags (--from/--to/--only/--skip,
--languages, --fasttext-python, --force, --dry-run, --extra), same skip-if-built
behaviour -- with the last step swapped. Instead of `collapse` (merge ECR+ID), it runs:

    national-map       build_national_party_map.py        each source speech's national
                                                          party -> EU&I 2024 party
    national-clusters  build_national_cluster_splits.py   label rows by that party's k=4
                                                          cluster, drop the rest, re-split

    python preprocessing/run_preprocessing_clusters.py --dry-run
    python preprocessing/run_preprocessing_clusters.py
    python preprocessing/run_preprocessing_clusters.py --only national-map national-clusters
    python preprocessing/run_preprocessing_clusters.py --extra 'national-clusters:--per-party 50000'

Steps 1-6 are shared with run_preprocessing.py, so their outputs are reused if either
driver has built them. The track lands in data/EuroParl Custom/clusters_k4_national/,
next to (not in place of) the ECR+ID track in collapsed/. The group-level variant,
build_cluster_splits.py (whole EP groups -> clusters), is still runnable on its own.
"""

import sys

from preprocessing import run_preprocessing as base

TRACK_DIR = base.EUROPARL_DIR / "clusters_k4_national"
NATIONAL_STEPS = [
    base.Step(
        "national-map", "build_national_party_map.py",
        "map every speech's national party onto an EU&I 2024 party",
        outputs=[base.EUROPARL_DIR / "national_parties" / name
                 for name in ("speeches.parquet", "party_map.csv", "coverage.json")],
    ),
    base.Step(
        "national-clusters", "build_national_cluster_splits.py",
        "label rows by their national party's k=4 cluster, drop the rest, re-split",
        outputs=[TRACK_DIR / f"{name}.parquet" for name in base.SPLITS + ("train_balanced",)]
                + [TRACK_DIR / "labels.json"],
    ),
]

# run_preprocessing reads its step list from these module globals at call time, so
# swapping them reuses its whole runner (selection, skipping, fastText handling).
base.STEPS = [step for step in base.STEPS if step.name != "collapse"] + NATIONAL_STEPS
base.STEP_NAMES = [step.name for step in base.STEPS]


def main():
    base.main()
    if "--dry-run" not in sys.argv and "-h" not in sys.argv and "--help" not in sys.argv:
        print("Cluster track: data/EuroParl Custom/clusters_k4_national/ "
              "(coverage and per-cluster parties in labels.json; party mapping in "
              "national_parties/party_map.csv).")


if __name__ == "__main__":
    main()
