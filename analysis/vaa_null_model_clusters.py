#!/usr/bin/env python3
"""The VAA null model of vaa_null_model.py, scored against the four k-means clusters.

vaa_null_model.py asks whether "models converge on S&D" is just S&D sitting near the
centre of the EP-group position space. This asks the same question of the four "fictional
EP groups" analysis/party_kmeans.py finds in the EU&I 2024 party answers (k=4):

    Radical left | Progressive federalists | Sovereigntist right |
    Liberal-conservative centre-right

Each cluster's position vector is the national mean of its member parties -- the
cluster analogue of `--positions group-mean` (build_group_positions.py): the member
parties' answers are averaged per statement, over the parties that answered it, and the
model is compared against that one vector. `--pooling parties` instead keeps every
member party as its own row and averages the agreements, the analogue of
`--positions national`. `--weight seats` weights parties by their 2024 MEPs.

Member parties are every MEP-holding party in the raw 27-country EU&I file (~164), not
the five countries euandi_2024_parties.jsonl covers, so the cluster positions rest on the
same parties the clusters were fitted on. Their answers are the raw, un-imputed ones:
KNN imputation is only there to give k-means complete rows, and imputed answers would be
the neighbours' answers counted twice. The raw file is in codebook order; its columns
are mapped to results rows through party_kmeans.RAW_COLUMN_FOR_QUESTIONNAIRE and the
`statement_codebook_idx` that align_statement_order.py left on every aligned response.

Everything else -- the real runs, the three nulls, the four aggregation levels, the
scorer, the tables and the figure -- is vaa_null_model.py's, imported unchanged.

Outputs (under --out-dir, next to the other null-model tables):
  null_model_clusters_<pooling>_<weight>_<source>_win_shares.csv / _summary.md / .png
  cluster_positions_k4_<weight>.csv   the cluster position vectors, results-row order

Usage, from the repository root with the package installed:
  python analysis/vaa_null_model_clusters.py
  python analysis/vaa_null_model_clusters.py --source speeches
  python analysis/vaa_null_model_clusters.py --pooling parties --weight seats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import party_kmeans as pk
from analysis.analyze_all import MODEL_DIRS
from analysis.core import PARTY_POSITIONS_PATH
from analysis.vaa_null_model import (
    DEFAULT_SIGMA,
    load_runs,
    make_generators,
    null_levels,
    plot_win_shares,
    real_levels,
    render_summary,
    score_everything,
)
from preprocessing.build_cluster_splits import cluster_parties
from utils import configure_stdout

configure_stdout()

K = 4
# The cluster the headline group (S&D) falls into; build_cluster_splits anchors SPD here.
DEFAULT_FOCUS = "Progressive federalists"


def codebook_to_results_row(path=PARTY_POSITIONS_PATH):
    """{codebook statement index -> results row}, read off the aligned parties file."""
    order = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for response in json.loads(line).get("responses", []):
                codebook, row = response.get("statement_codebook_idx"), response["statement_idx"]
                if codebook is None:
                    raise SystemExit(f"{path} has no statement_codebook_idx; run "
                                     "statement_collection/align_statement_order.py")
                if order.setdefault(int(codebook), int(row)) != int(row):
                    raise SystemExit(f"codebook statement {codebook} maps to two results rows")
    n = len(pk.RAW_COLUMN_FOR_QUESTIONNAIRE)
    if sorted(order) != list(range(n)) or sorted(order.values()) != list(range(n)):
        raise SystemExit(f"codebook -> results order is not a permutation of 0..{n - 1}")
    return order


def party_answers(parties, statement_count):
    """(parties, statements) raw answers on [-1, 1] in results-row order, NaN = no answer."""
    order = codebook_to_results_row()
    if statement_count != len(order):
        raise SystemExit(f"runs have {statement_count} statements, positions {len(order)}")
    raw = pk.answer_matrix(parties)                        # s1..s36, codebook columns
    out = np.full((len(parties), statement_count), np.nan)
    for codebook, column in enumerate(pk.RAW_COLUMN_FOR_QUESTIONNAIRE):
        out[:, order[codebook]] = raw[:, pk.S_COLS.index(column)]
    return out


def cluster_tensor(parties, answers, pooling, weight):
    """(P, party_group, groups) in vaa_null_model.positions_tensor's shape.

    `mean` gives one row per cluster: the (weighted) mean over the members that answered
    each statement, NaN where none did. `parties` keeps the member rows; group_scores
    then averages agreements over them, which is unweighted by construction.
    """
    names = pk.CLUSTER_NAMES[K]
    labels = parties["cluster_name"].to_numpy()
    if pooling == "parties":
        return answers, labels, names
    w = parties["meps"].to_numpy(dtype=float) if weight == "seats" else np.ones(len(parties))
    P = np.full((len(names), answers.shape[1]), np.nan)
    for ci, name in enumerate(names):
        member = labels == name
        values, weights = answers[member], np.where(np.isnan(answers[member]), 0.0, w[member, None])
        total = weights.sum(axis=0)
        with np.errstate(invalid="ignore"):
            P[ci] = np.where(total > 0, (np.nan_to_num(values) * weights).sum(axis=0) / total, np.nan)
    return P, np.array(names), names


def describe(parties, P, names, pooling):
    lines = []
    for name in names:
        sub = parties[parties["cluster_name"] == name]
        seats = {}
        for by_group in sub["seats_by_group"]:
            for g, n in by_group.items():
                seats[g] = seats.get(g, 0) + n
        top = ", ".join(f"{g} {n}" for g, n in sorted(seats.items(), key=lambda x: -x[1])[:4])
        lines.append(f"  {name:<34} {len(sub):>3} parties, {sub['meps'].sum():>3} MEPs ({top})")
    if pooling == "mean":
        dist = np.sqrt(np.nanmean(P ** 2, axis=1))
        lines.append("  RMS distance of each cluster vector from the neutral centre: "
                     + ", ".join(f"{n} {d:.3f}" for n, d in zip(names, dist)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2024"])
    parser.add_argument("--source", default="likert", choices=["likert", "speeches"])
    parser.add_argument("--models", nargs="+", default=MODEL_DIRS)
    parser.add_argument("--pooling", default="mean", choices=["mean", "parties"],
                        help="mean: one averaged vector per cluster (group-mean analogue); "
                             "parties: every member party a row (national analogue).")
    parser.add_argument("--weight", default="equal", choices=["equal", "seats"],
                        help="Party weights in the cluster mean (--pooling mean only).")
    parser.add_argument("--cluster-seed", type=int, default=0,
                        help="k-means seed; party_kmeans.py's default, which the names hold for.")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--focus", default=DEFAULT_FOCUS, help="Cluster for the per-model table.")
    parser.add_argument("--out-dir", default=None,
                        help="default: data/<dataset>_results/tables/null_model")
    args = parser.parse_args()
    if args.pooling == "parties" and args.weight == "seats":
        parser.error("--weight seats needs --pooling mean (party rows are averaged unweighted)")

    results_root = Path("data") / f"{args.dataset}_results"
    model_dirs = [results_root / m for m in args.models]
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "tables" / "null_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading real {args.source} runs")
    runs = load_runs(model_dirs, args.source)
    models = [m for m in args.models if m in runs.index.get_level_values("model")]

    print(f"Clustering MEP-holding EU&I parties (k={K}, seed {args.cluster_seed})")
    parties = cluster_parties(args.cluster_seed, K)
    answers = party_answers(parties, runs.shape[1])
    P, party_group, groups = cluster_tensor(parties, answers, args.pooling, args.weight)
    print(describe(parties, P, groups, args.pooling))
    if args.pooling == "mean":
        pd.DataFrame(P.T, columns=groups).rename_axis("statement_idx").to_csv(
            out_dir / f"cluster_positions_k{K}_{args.weight}.csv")

    paraphrases = int(runs.groupby(level=["model", "framing", "language"]).size().median())
    runs_per_model = int(runs.groupby(level="model").size().median())
    print(f"Scoring {args.draws} draws per null and level")
    null = null_levels(make_generators(runs, args.sigma, rng), args.draws, paraphrases, runs_per_model)
    table = score_everything(real_levels(runs), null, P, party_group, groups, args.source)

    stem = f"null_model_clusters_{args.pooling}_{args.weight}_{args.source}"
    table.to_csv(out_dir / f"{stem}_win_shares.csv", index=False)
    # render_summary reads these two for its "Positions:" line; the basis here is neither.
    args.positions = (f"k={K} clusters, {args.weight}-weighted national-party mean"
                      if args.pooling == "mean" else f"k={K} clusters, member parties as rows")
    args.no_collapse_ecr_id = False
    summary = render_summary(table, groups, models, args, args.sigma, runs, paraphrases,
                             runs_per_model)
    summary = (summary.replace("# VAA null model", f"# VAA null model, k={K} clusters", 1)
                      .replace(", ECR+ID collapsed", "", 1))
    summary += "\nCluster membership:\n\n```\n" + describe(parties, P, groups, args.pooling) + "\n```\n"
    (out_dir / f"{stem}_summary.md").write_text(summary, encoding="utf-8")
    plot_win_shares(table, groups, out_dir / f"{stem}_win_shares.png", f"k={K} clusters, {args.source}")
    print()
    print(summary)
    print(f"\nWrote {out_dir / (stem + '_win_shares.csv')}, {stem}_summary.md, {stem}_win_shares.png")


if __name__ == "__main__":
    main()
