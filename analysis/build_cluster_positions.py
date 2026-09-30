#!/usr/bin/env python3
"""Average each k=4 cluster's member parties into one MEP-weighted position vector.

`--positions cluster` is the fourth basis `analysis/core/positions.py` offers. It swaps
the seven EP groups for the four "fictional EP groups" analysis/party_kmeans.py finds in
the EU&I 2024 party answers -- the same labels the cluster-track classifiers
(runs/<track>-k4-<trainer>) predict, so VAA agreement and classifier probabilities are
reported over one set of groups:

    Radical left | Progressive federalists | Liberal-conservative centre-right |
    Sovereigntist right

A cluster's position on a statement is the mean of its member parties' answers, each
party weighted by its 2024 MEP count, over the members that answered the statement.
This is vaa_null_model_clusters.py's `--pooling mean --weight seats` vector, computed by
the same functions, so the null model and the main pipeline score against one thing.

Member parties are every MEP-holding party in the raw 27-country EU&I file (~164), not
the five countries euandi_2024_parties.jsonl covers: the clusters were fitted on those
parties, and a five-country subset would leave some clusters resting on a handful of
parties. Answers are the raw, un-imputed ones (KNN imputation only exists to give
k-means complete rows).

The output follows build_group_positions.py's schema -- `country_iso` "eu", one record
per cluster -- plus an explicit `ep_group` field carrying the cluster name, which
`load_party_positions` keeps instead of mapping `short_name` through EP_GROUP_BY_PARTY.
Statement text is copied from statements.jsonl, so the file passes
`core.positions.check_statement_order` by construction.

    python analysis/build_cluster_positions.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.vaa_null_model_clusters import K, cluster_parties, cluster_tensor, party_answers
from analysis.build_group_positions import PRECISION, statement_texts
from utils import configure_stdout

configure_stdout()


def build(data_dir: Path, cluster_seed: int) -> list[dict]:
    statements = statement_texts(data_dir)
    parties = cluster_parties(cluster_seed, K)
    answers = party_answers(parties, len(statements))
    P, _, names = cluster_tensor(parties, answers, "mean", "seats")

    records = []
    for ci, name in enumerate(names):
        member = (parties["cluster_name"] == name).to_numpy()
        sub = parties[member]
        answered = (~np.isnan(answers[member])).sum(axis=0)
        # One Latvian party has no abbreviation in the raw file; fall back to its name.
        labels = sorted(f"{(a if isinstance(a, str) else n).strip()} ({c.strip()})"
                        for a, n, c in zip(sub["ABBREVIATON"], sub["PARTY_NAME_ENG"], sub["COUNTRY"]))
        records.append({
            "short_name": name,
            "ep_group": name,
            "full_name": f"{name}: MEP-weighted mean of {len(sub)} parties, {int(sub['meps'].sum())} MEPs",
            "country_iso": "eu",
            "country": "eu",
            "weighting": "meps",
            "meps": int(sub["meps"].sum()),
            "member_parties": labels,
            "responses": [{
                "statement_idx": idx,
                "statement": statement,
                "n_answered": int(answered[idx]),
                "normalized_answer": None if np.isnan(P[ci, idx]) else round(float(P[ci, idx]), PRECISION),
            } for idx, statement in enumerate(statements)],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2024"])
    parser.add_argument("--cluster-seed", type=int, default=0,
                        help="k-means seed; the cluster names and the classifier tracks use 0")
    parser.add_argument("--output", help="default: data/{dataset}_data/{dataset}_cluster_positions.jsonl")
    args = parser.parse_args()

    data_dir = Path("data") / f"{args.dataset}_data"
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} not found (run from the repo root)")

    records = build(data_dir, args.cluster_seed)
    out_path = Path(args.output) if args.output else data_dir / f"{args.dataset}_cluster_positions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(records)} clusters, MEP-weighted)")
    for record in records:
        answered = sum(1 for r in record["responses"] if r["normalized_answer"] is not None)
        values = [r["normalized_answer"] for r in record["responses"] if r["normalized_answer"] is not None]
        rms = float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")
        print(f"  {record['ep_group']:<34} {len(record['member_parties']):>3} parties, "
              f"{record['meps']:>3} MEPs, {answered}/{len(record['responses'])} statements, "
              f"RMS from centre {rms:.3f}")


if __name__ == "__main__":
    main()
