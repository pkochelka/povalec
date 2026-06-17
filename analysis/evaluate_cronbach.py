import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, LIKERT_MIDPOINT, flip_likert
from evaluate_euandi import detect_likert_languages, likert_means_per_statement

AXES = ["Ukraine", "Ecology", "Immigration", "Values", "Economy", "Europe", "Left-Right"]
SUPPORTS = 1
OPPOSES = -1


def load_axis_directions(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        records = [{axis: json.loads(line)[axis] for axis in AXES} for line in f]
    return pd.DataFrame(records, columns=AXES)


def align_responses_to_axes(mean_likert: pd.Series, directions: pd.DataFrame) -> pd.DataFrame:
    flipped = flip_likert(mean_likert)
    aligned = {}
    for axis in AXES:
        direction = directions[axis]
        column = pd.Series(LIKERT_MIDPOINT, index=mean_likert.index)
        column = column.where(direction != SUPPORTS, mean_likert)
        column = column.where(direction != OPPOSES, flipped)
        aligned[axis] = column
    return pd.DataFrame(aligned, columns=AXES)


def cronbach_alpha(items: pd.DataFrame) -> float:
    item_count = items.shape[1]
    item_variances = items.var(axis=0, ddof=1).sum()
    total_variance = items.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return item_count / (item_count - 1) * (1 - item_variances / total_variance)


def evaluate(llm_responses_path: str, questionnaire_path: str) -> pd.DataFrame:
    raw_llm_df = pd.read_csv(llm_responses_path, sep=";", encoding="utf-8-sig")
    languages = detect_likert_languages(raw_llm_df)

    mean_likert = likert_means_per_statement(raw_llm_df, languages)
    directions = load_axis_directions(questionnaire_path)

    rows = [
        {"language": lang, "cronbach_alpha": cronbach_alpha(align_responses_to_axes(mean_likert[lang], directions))}
        for lang in languages
    ]
    return pd.DataFrame(rows).sort_values("cronbach_alpha", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="qwen3.5-122b", type=str)
    parser.add_argument("--variant", default="", type=str, choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default=ALL_LANGS_STR, type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    languages_joined = ",".join(args.languages.split(","))

    llm_responses_path = f"data/{args.dataset}_results/{args.model_dir}/{languages_joined}{args.variant}.csv"
    questionnaire_path = f"data/{args.dataset}_data/{args.dataset}_questionnaire.jsonl"
    output_path = f"data/{args.dataset}_results/{args.model_dir}/cronbach{args.variant}_{languages_joined}.csv"

    if os.path.exists(output_path) and not args.override:
        print(f"Output exists, skipping (use --override to recompute): {output_path}")
        sys.exit(0)

    summary = evaluate(llm_responses_path, questionnaire_path)
    summary.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(summary)
