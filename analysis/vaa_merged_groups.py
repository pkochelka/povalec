#!/usr/bin/env python3
"""euandi (VAA) agreement with the seven EP groups merged into four blocs.

    GUE/NGL          on its own
    S&D+Greens/EFA   S&D and Greens/EFA
    ALDE+PPE         ALDE and PPE
    ECR+ID           ECR and ID

Each bloc gets one position vector, the weighted mean of its groups' positions,
computed per statement over the groups that answered it (weights renormalised, so a
group that abstains does not drag the bloc toward 0). The model is then scored against
that vector with the usual `1 - |position - stance| / 2`.

This is *not* what `--collapse-ecr-id` does in evaluate_euandi.py: that relabels the
rows and averages the agreements, which is an equal-weight mean of per-group agreements
rather than agreement with an averaged position. The two differ because the absolute
value is not linear. ECR+ID here uses the position mean like the other blocs.

Weights: `seats` uses EP seats won in the 2019 election (the 9th term, whose groups
the 2024 euandi positions describe); `equal` weighs the two groups the same. Both are
reported by default so the choice can be seen to matter or not.

With `--positions national` the bloc mean is taken over the member *parties* directly
(all parties of both groups pooled), weighted by the seats each party's list won at the
2024 election (`seats`) or equally (`equal`). The six-group reference is then what
`evaluate_euandi.py --positions national --collapse-ecr-id` computes.

Output goes to the tables dir only; the per-model vaa*.csv files are left untouched.
"""
import argparse
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import ALL_LANGS_STR, configure_stdout
from analysis.evaluate_euandi import (
    likert_stance_frame, load_party_positions, positions_path, speech_stance_frame,
)

configure_stdout()

BLOC_BY_GROUP = {
    "GUE/NGL": "GUE/NGL",
    "S&D": "S&D+Greens/EFA", "Greens/EFA": "S&D+Greens/EFA",
    "ALDE": "ALDE+PPE", "PPE": "ALDE+PPE",
    "ECR": "ECR+ID", "ID": "ECR+ID",
}
BLOCS = ["GUE/NGL", "S&D+Greens/EFA", "ALDE+PPE", "ECR+ID"]
# Seats won at the 2019 European Parliament election (constitutive session, 751 MEPs).
SEATS_2019 = {"PPE": 182, "S&D": 154, "ALDE": 108, "Greens/EFA": 74,
              "ID": 73, "ECR": 62, "GUE/NGL": 41}
# Seats won by each national list at the 2024 European Parliament election, for the
# parties euandi_2024_parties.jsonl covers (list totals, so RE is the Besoin d'Europe
# list and PS the PS-Place publique list; CDU excludes the CSU).
SEATS_2024_NATIONAL = {
    "CDU": 23, "SPD": 14, "AfD": 15, "FDP": 5, "Linke": 3, "Grüne": 12,
    "RE": 13, "RN": 30, "LFI": 9, "EELV": 5, "PS": 13, "LR": 6,
    "Lega": 8, "PD": 21, "FDI": 24, "M5S": 8, "FI": 8, "AVS": 6,
    "PSOE": 20, "PP": 22, "Vox": 6, "AR": 3, "Sumar": 3, "Podemos": 2,
    "ND": 7, "SYRIZA": 4, "PASOK": 3, "EL": 2,
}
WEIGHTS = {"seats": SEATS_2019, "equal": {group: 1 for group in BLOC_BY_GROUP}}
NATIONAL_WEIGHTS = {"seats": SEATS_2024_NATIONAL,
                    "equal": {party: 1 for party in SEATS_2024_NATIONAL}}
SIX_GROUPS = {"ECR": "ECR+ID", "ID": "ECR+ID"}
SOURCES = {"likert": "likert", "speeches": "open-ended"}
VARIANTS = {"": "base", "_negated": "negated"}


def group_positions(positions):
    """(ep_group, statement_idx) -> normalized_answer, one vector per group."""
    df = load_party_positions(positions_path(positions), positions).dropna(subset=["ep_group"])
    return df.groupby(["ep_group", "statement_idx"], as_index=False)["normalized_answer"].mean()


def national_positions():
    """(ep_group, short_name, statement_idx) -> normalized_answer, one row per party."""
    df = load_party_positions(positions_path("national"), "national").dropna(subset=["ep_group"])
    unweighted = set(df["short_name"]) - set(SEATS_2024_NATIONAL)
    if unweighted:
        raise SystemExit(f"No 2024 seat count for: {sorted(unweighted)}")
    return df[["ep_group", "short_name", "statement_idx", "normalized_answer"]]


def bloc_positions(groups, weights, unit="ep_group"):
    df = groups.dropna(subset=["normalized_answer"]).copy()
    df["bloc"] = df["ep_group"].map(BLOC_BY_GROUP)
    df["w"] = df[unit].map(weights)
    df["wx"] = df["w"] * df["normalized_answer"]
    sums = df.groupby(["bloc", "statement_idx"], as_index=False)[["wx", "w"]].sum()
    sums["normalized_answer"] = sums["wx"] / sums["w"]
    return sums.rename(columns={"bloc": "ep_group"})[["ep_group", "statement_idx", "normalized_answer"]]


def six_group_positions(groups):
    """The current pipeline's six groups: ECR and ID rows kept apart, agreements pooled."""
    df = groups.copy()
    df["ep_group"] = df["ep_group"].replace(SIX_GROUPS)
    return df


def agreement(stance_df, languages, positions):
    """Mean agreement per group, averaged over statements within a language, then over
    languages -- the same order the vaa*.csv files and their plots use."""
    cols = [f"{lang}_stance" for lang in languages]
    merged = positions.merge(stance_df[["statement_idx", *cols]], on="statement_idx")
    long = merged.melt(id_vars=["ep_group", "normalized_answer"], value_vars=cols,
                       var_name="language", value_name="llm_stance")
    long["agreement"] = 1 - (long["normalized_answer"] - long["llm_stance"]).abs() / 2
    per_lang = long.groupby(["ep_group", "language"])["agreement"].mean()
    return per_lang.groupby("ep_group").mean()


def stance_frame(results_dir, model, source, variant, languages):
    if source == "likert":
        path = os.path.join(results_dir, model, f"{languages}{variant}.csv")
        loader = lambda: likert_stance_frame(path, negated=variant == "_negated")
    else:
        path = os.path.join(results_dir, model, f"speeches_{languages}{variant}_scored.csv")
        loader = lambda: speech_stance_frame(path, variant)
    return loader() if os.path.exists(path) else (None, [])


def models_in(results_dir):
    return sorted(d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d)) and d not in {"plots", "tables"})


def markdown_table(wide, columns):
    header = "| model | scope | " + " | ".join(columns) + " | top |"
    rule = "|" + "---|" * (len(columns) + 3)
    rows = [header, rule]
    for (model, scope), row in wide.iterrows():
        top = row[columns].idxmax()
        cells = [f"**{row[c]:.3f}**" if c == top else f"{row[c]:.3f}" for c in columns]
        rows.append(f"| {model} | {scope} | " + " | ".join(cells) + f" | {top} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--positions", default="ep-group", choices=["ep-group", "group-mean", "national"],
                        help="ep-group/group-mean: one vector per group, blocs average groups. "
                             "national: blocs average all member parties directly.")
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--models", default=None, help="Comma-separated; default: all.")
    parser.add_argument("--common-statements", action="store_true",
                        help="Score only statements every group answered, so all groups "
                             "and blocs are compared on the same subset.")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, "data", f"{args.dataset}_results")
    models = args.models.split(",") if args.models else models_in(results_dir)
    national = args.positions == "national"
    groups = national_positions() if national else group_positions(args.positions)
    suffix = ""
    if args.common_statements:
        answered = groups.dropna(subset=["normalized_answer"]).groupby("statement_idx")["ep_group"].nunique()
        common = answered[answered == groups["ep_group"].nunique()].index
        groups = groups[groups["statement_idx"].isin(common)]
        suffix = "_common"
        print(f"Common statements ({len(common)}): {list(common)}")
    weights, unit = (NATIONAL_WEIGHTS, "short_name") if national else (WEIGHTS, "ep_group")
    bases = {"six": six_group_positions(groups),
             **{name: bloc_positions(groups, w, unit) for name, w in weights.items()}}

    records = []
    for model in models:
        for source, source_label in SOURCES.items():
            for variant, variant_label in VARIANTS.items():
                stance_df, languages = stance_frame(results_dir, model, source, variant, args.languages)
                if stance_df is None:
                    continue
                for basis, positions in bases.items():
                    for group, value in agreement(stance_df, languages, positions).items():
                        records.append({"model": model, "scope": f"{source_label} ({variant_label})",
                                        "basis": basis, "ep_group": group, "mean_agreement": value})
    if not records:
        raise SystemExit("No model results found.")
    long = pd.DataFrame(records)

    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, f"vaa_merged_groups_{args.positions}{suffix}.csv")
    long.to_csv(csv_path, index=False)

    six_cols = ["GUE/NGL", "S&D", "Greens/EFA", "ALDE", "PPE", "ECR+ID"]
    subset = f"{groups['statement_idx'].nunique()} statements answered by every group" if suffix else "all statements"
    sections = [f"Positions basis `{args.positions}`, {subset}. Agreement averaged over statements "
                f"per language, then over {len(args.languages.split(','))} languages."]
    seat_title = ("all member parties, weighted by 2024 national-list seats" if national
                  else "groups weighted by 2019 seats")
    for basis, title in [("seats", f"Four blocs, seat-weighted ({seat_title})"),
                         ("equal", "Four blocs, equal-weighted" + (" parties" if national else " groups")),
                         ("six", "Reference: current six groups (--collapse-ecr-id)")]:
        cols = BLOCS if basis != "six" else six_cols
        wide = (long[long["basis"] == basis]
                .pivot_table(index=["model", "scope"], columns="ep_group", values="mean_agreement"))
        sections += ["", f"### {title}", "", markdown_table(wide, cols)]

    tops = (long.loc[long.groupby(["model", "scope", "basis"])["mean_agreement"].idxmax()]
            .pivot_table(index="basis", columns="ep_group", values="model", aggfunc="count", fill_value=0))
    sections += ["", "### How often each group/bloc is the top match (model x scope)", "",
                 "| basis | " + " | ".join(tops.columns) + " |",
                 "|" + "---|" * (len(tops.columns) + 1),
                 *(f"| {basis} | " + " | ".join(str(n) for n in row) + " |"
                   for basis, row in tops.iterrows())]
    markdown = "\n".join(sections)
    md_path = os.path.join(tables_dir, f"vaa_merged_groups_{args.positions}{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    print(markdown)
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
