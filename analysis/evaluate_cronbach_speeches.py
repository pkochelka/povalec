import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR
from evaluate_cronbach import AXES, SUPPORTS, OPPOSES, load_axis_directions, cronbach_alpha

# NLI stance scores already live on the [-1, +1] agree/disagree axis
# (+1 = entailment, -1 = contradiction), so the neutral midpoint is 0 here,
# the analog of LIKERT_MIDPOINT in evaluate_cronbach.
STANCE_NEUTRAL = 0.0


def detect_speech_languages(df: pd.DataFrame, variant: str) -> list[str]:
    pattern = re.compile(r"^stance_([a-z]+)" + re.escape(variant) + r"_mean$")
    return [m.group(1) for col in df.columns if (m := pattern.match(col))]


def mean_stance_per_statement(df: pd.DataFrame, languages: list[str], variant: str) -> pd.DataFrame:
    return pd.DataFrame({
        lang: pd.to_numeric(df[f"stance_{lang}{variant}_mean"], errors="coerce")
        for lang in languages
    })


def align_stance_to_axes(mean_stance: pd.Series, directions: pd.DataFrame) -> pd.DataFrame:
    aligned = {}
    for axis in AXES:
        direction = directions[axis]
        column = pd.Series(STANCE_NEUTRAL, index=mean_stance.index)
        column = column.where(direction != SUPPORTS, mean_stance)
        column = column.where(direction != OPPOSES, -mean_stance)
        aligned[axis] = column
    return pd.DataFrame(aligned, columns=AXES)


def evaluate(scored_path: str, questionnaire_path: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(scored_path, sep=";", encoding="utf-8-sig")
    languages = detect_speech_languages(df, variant)

    mean_stance = mean_stance_per_statement(df, languages, variant)
    directions = load_axis_directions(questionnaire_path)

    rows = [
        {"language": lang, "cronbach_alpha": cronbach_alpha(align_stance_to_axes(mean_stance[lang], directions))}
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

    scored_path = f"data/{args.dataset}_results/{args.model_dir}/speeches_{languages_joined}{args.variant}_scored.csv"
    questionnaire_path = f"data/{args.dataset}_data/{args.dataset}_questionnaire.jsonl"
    output_path = f"data/{args.dataset}_results/{args.model_dir}/cronbach_speeches{args.variant}_{languages_joined}.csv"

    if os.path.exists(output_path) and not args.override:
        print(f"Output exists, skipping (use --override to recompute): {output_path}")
        sys.exit(0)

    summary = evaluate(scored_path, questionnaire_path, args.variant)
    summary.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(summary)
