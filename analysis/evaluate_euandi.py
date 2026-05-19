import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import flip_likert, likert_to_stance

PARTY_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_parties.jsonl"
NUM_RESPONSE_VARIANTS = 8

EP_GROUP_BY_PARTY = {
    # EU-level party federations
    "EPP":  "PPE",       "ECR":  "ECR",
    "PES":  "S&D",       "ALDE": "ALDE",
    "EGP":  "Greens/EFA","ID":   "ID",
    "PEL":  "GUE/NGL",
    # Germany
    "CDU":   "PPE",      "SPD":   "S&D",
    "Grüne": "Greens/EFA","FDP":  "ALDE",
    "AfD":   "ID",       "Linke": "GUE/NGL",
    # France
    "RE":   "ALDE",      "RN":   "ID",
    "PS":   "S&D",       "LFI":  "GUE/NGL",
    "EELV": "Greens/EFA","LR":   "PPE",
    # Italy
    "FDI":  "ECR",       "Lega": "ID",
    "FI":   "PPE",       "PD":   "S&D",
    "M5S":  "GUE/NGL",   "AVS":  "Greens/EFA",
    "AR":   "ALDE",
    # Greece
    "ND":     "PPE",     "PASOK":  "S&D",
    "SYRIZA": "S&D",     "EL":     "ECR",
    # KKE and Niki are non-attached — omitted
    # Spain
    "PP":      "PPE",    "PSOE":    "S&D",
    "Vox":     "ECR",    "Sumar":   "GUE/NGL",
    "Podemos": "GUE/NGL",
}


def detect_languages(df: pd.DataFrame) -> list[str]:
    return [
        col[len("choice_"):-len("_v0")]
        for col in df.columns
        if col.startswith("choice_") and col.endswith("_v0")
    ]


def mean_likert_per_statement(df: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    result = pd.DataFrame({"statement_idx": range(len(df))})
    for lang in languages:
        choice_cols = [f"choice_{lang}_v{v}" for v in range(NUM_RESPONSE_VARIANTS)]
        result[lang] = df[choice_cols].apply(pd.to_numeric, errors="coerce").fillna(3).mean(axis=1)
    return result


def load_party_positions(path: str) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            party_meta = {k: v for k, v in obj.items() if k != "responses"}
            for resp in obj.get("responses", []):
                records.append({**party_meta, **resp})
    df = pd.DataFrame(records)
    df = df[df["country_iso"] != "eu"].copy()
    df["statement_idx"] = df["statement_idx"].astype(int)
    df["statement"] = df["statement"].str.strip()
    df["ep_group"] = df["short_name"].map(EP_GROUP_BY_PARTY)
    return df


def compute_mean_agreement_per_ep_group_and_language(
    merged: pd.DataFrame, languages: list[str]
) -> pd.DataFrame:
    per_language_rows = []
    for lang in languages:
        rows = merged[["ep_group", "statement", "normalized_answer", f"{lang}_stance"]].copy()
        rows = rows.rename(columns={
            f"{lang}_stance": "llm_stance",
            "normalized_answer": "party_stance",
        })
        rows["language"] = lang
        rows["agreement"] = 1 - (rows["party_stance"] - rows["llm_stance"]).abs() / 2
        per_language_rows.append(rows[["ep_group", "statement", "language", "party_stance", "llm_stance", "agreement"]])

    per_statement_df = pd.concat(per_language_rows, ignore_index=True)
    return (
        per_statement_df
        .groupby(["ep_group", "language"])["agreement"]
        .mean()
        .reset_index()
        .rename(columns={"agreement": "mean_agreement"})
        .sort_values(["language", "mean_agreement"], ascending=[True, False])
    )


def evaluate(llm_responses_path: str, party_positions_path: str, negated: bool = False) -> pd.DataFrame:
    raw_llm_df = pd.read_csv(llm_responses_path, sep=";", encoding="utf-8-sig")
    languages = detect_languages(raw_llm_df)

    llm_df = mean_likert_per_statement(raw_llm_df, languages)
    if negated:
        for lang in languages:
            llm_df[lang] = flip_likert(llm_df[lang])
    for lang in languages:
        llm_df[f"{lang}_stance"] = likert_to_stance(llm_df[lang])

    party_df = load_party_positions(party_positions_path)

    stance_cols = [f"{lang}_stance" for lang in languages]
    merged = party_df.merge(llm_df[["statement_idx"] + stance_cols], on="statement_idx", how="inner")

    return compute_mean_agreement_per_ep_group_and_language(merged, languages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="qwen3.5-122b", type=str)
    parser.add_argument("--variant", default="", type=str, choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default="at,be,bg,hr,cy,cz,dk,ee,fi,fr,de,gr,hu,ie,it,lv,lt,lu,mt,nl,pl,pt,ro,sk,si,es,se,en", type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    args = parser.parse_args()
    languages_joined = ",".join(args.languages.split(","))

    llm_responses_path = f"data/{args.dataset}_results/{args.model_dir}/{languages_joined}{args.variant}.csv"
    output_path = f"data/{args.dataset}_results/{args.model_dir}/vaa{args.variant}_{languages_joined}.csv"

    summary = evaluate(llm_responses_path, PARTY_POSITIONS_PATH, negated=args.variant == "_negated")
    summary.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(summary)
