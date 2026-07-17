import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, flip_likert, likert_to_stance

PARTY_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_parties.jsonl"
NUM_RESPONSE_VARIANTS = 8
NEUTRAL_LIKERT = 3

EP_GROUP_BY_PARTY = {
    "EPP": "PPE", "ECR": "ECR", "PES": "S&D", "ALDE": "ALDE",
    "EGP": "Greens/EFA", "ID": "ID", "PEL": "GUE/NGL",

    "CDU": "PPE", "SPD": "S&D", "Grüne": "Greens/EFA", "FDP": "ALDE",
    "AfD": "ID", "Linke": "GUE/NGL",

    "RE": "ALDE", "RN": "ID", "PS": "S&D", "LFI": "GUE/NGL",
    "EELV": "Greens/EFA", "LR": "PPE",

    "FDI": "ECR", "Lega": "ID", "FI": "PPE", "PD": "S&D",
    "M5S": "GUE/NGL", "AVS": "Greens/EFA", "AR": "ALDE",

    "ND": "PPE", "PASOK": "S&D", "SYRIZA": "S&D", "EL": "ECR",

    "PP": "PPE", "PSOE": "S&D", "Vox": "ECR",
    "Sumar": "GUE/NGL", "Podemos": "GUE/NGL",
}


def detect_likert_languages(df: pd.DataFrame) -> list[str]:
    return [
        col[len("choice_"):-len("_v0")]
        for col in df.columns
        if col.startswith("choice_") and col.endswith("_v0")
    ]


def detect_speech_languages(df: pd.DataFrame, variant: str) -> list[str]:
    pattern = re.compile(rf"^stance_([a-z]+){re.escape(variant)}_mean$")
    return [m.group(1) for col in df.columns if (m := pattern.match(col))]


def likert_means_per_statement(raw_df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    means = pd.DataFrame({"statement_idx": range(len(raw_df))})
    for lang in languages:
        variant_cols = [f"choice_{lang}_v{v}" for v in range(NUM_RESPONSE_VARIANTS)]
        numeric = raw_df[variant_cols].apply(pd.to_numeric, errors="coerce")
        means[lang] = numeric.fillna(NEUTRAL_LIKERT).mean(axis=1)
    return means


def load_party_positions(path: str) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = {k: v for k, v in obj.items() if k != "responses"}
            for response in obj.get("responses", []):
                records.append({**meta, **response})

    df = pd.DataFrame(records)
    df = df[df["country_iso"] != "eu"].copy()
    df["statement_idx"] = df["statement_idx"].astype(int)
    df["statement"] = df["statement"].str.strip()
    df["ep_group"] = df["short_name"].map(EP_GROUP_BY_PARTY)
    return df


def agreement_by_ep_group(
    stance_df: pd.DataFrame, languages: list[str], party_positions_path: str,
    collapse_ecr_id: bool = False,
) -> pd.DataFrame:
    stance_cols = [f"{lang}_stance" for lang in languages]
    party_df = load_party_positions(party_positions_path)
    if collapse_ecr_id:
        # Pool the national parties of both groups before averaging, matching
        # the collapsed ECR+ID classifier label. EP_GROUP_BY_PARTY itself must
        # stay 7-group: the multilabel trainers rely on separate ECR/ID rows.
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    merged = party_df.merge(stance_df[["statement_idx", *stance_cols]], on="statement_idx")

    long = merged.melt(
        id_vars=["ep_group", "normalized_answer"],
        value_vars=stance_cols,
        var_name="language",
        value_name="llm_stance",
    )
    long["language"] = long["language"].str.removesuffix("_stance")
    long["agreement"] = 1 - (long["normalized_answer"] - long["llm_stance"]).abs() / 2

    return (
        long.groupby(["ep_group", "language"], as_index=False)["agreement"]
        .mean()
        .rename(columns={"agreement": "mean_agreement"})
        .sort_values(["language", "mean_agreement"], ascending=[True, False])
    )


def evaluate_likert(
    llm_responses_path: str, party_positions_path: str, negated: bool = False,
    collapse_ecr_id: bool = False,
) -> pd.DataFrame:
    raw_df = pd.read_csv(llm_responses_path, sep=";", encoding="utf-8-sig")
    languages = detect_likert_languages(raw_df)

    likert_df = likert_means_per_statement(raw_df, languages)
    for lang in languages:
        likert = flip_likert(likert_df[lang]) if negated else likert_df[lang]
        likert_df[f"{lang}_stance"] = likert_to_stance(likert)

    return agreement_by_ep_group(likert_df, languages, party_positions_path, collapse_ecr_id)


def evaluate_speeches(
    scored_path: str, party_positions_path: str, variant: str = "",
    collapse_ecr_id: bool = False,
) -> pd.DataFrame:
    raw_df = pd.read_csv(scored_path, sep=";", encoding="utf-8-sig")
    languages = detect_speech_languages(raw_df, variant)
    flip_sign = -1.0 if variant == "_negated" else 1.0

    stance_df = pd.DataFrame({"statement_idx": range(len(raw_df))})
    for lang in languages:
        speech_stance = pd.to_numeric(raw_df[f"stance_{lang}{variant}_mean"], errors="coerce")
        stance_df[f"{lang}_stance"] = flip_sign * speech_stance

    return agreement_by_ep_group(stance_df, languages, party_positions_path, collapse_ecr_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="qwen3.5-122b")
    parser.add_argument("--variant", default="", choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--source", default="likert", choices=["likert", "speeches"])
    parser.add_argument("--collapse-ecr-id", action="store_true",
                        help="Merge the ECR and ID EP groups into one 'ECR+ID' group, "
                             "matching the collapsed classifier labels.")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    results_dir = f"data/{args.dataset}_results/{args.model_dir}"
    if args.source == "likert":
        input_path = f"{results_dir}/{args.languages}{args.variant}.csv"
        output_path = f"{results_dir}/vaa{args.variant}_{args.languages}.csv"
    else:
        input_path = f"{results_dir}/speeches_{args.languages}{args.variant}_scored.csv"
        output_path = f"{results_dir}/vaa_speeches{args.variant}_{args.languages}.csv"

    if os.path.exists(output_path) and not args.override:
        print(f"Output exists, skipping (use --override to recompute): {output_path}")
        return

    if args.source == "likert":
        summary = evaluate_likert(
            input_path, PARTY_POSITIONS_PATH, negated=args.variant == "_negated",
            collapse_ecr_id=args.collapse_ecr_id,
        )
    else:
        summary = evaluate_speeches(
            input_path, PARTY_POSITIONS_PATH, args.variant,
            collapse_ecr_id=args.collapse_ecr_id,
        )

    summary.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(summary)


if __name__ == "__main__":
    main()
