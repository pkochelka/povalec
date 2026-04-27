import argparse
import json
import re

import pandas as pd
from api_caller import call_api

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="kimi-k2.6", type=str, choices=["kimi-k2.6"])
parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)
parser.add_argument("--task_prompts", default="./prompts/negate_and_neutralize.json", type=str)


def load_prompts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def survey_batch(proposition: str, language: str, model: str, prompts: dict) -> dict:
    user_prompt = prompts[language].format(proposition=proposition)
    while True:
        try:
            response = call_api(user_prompt, model)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result and "question" in result and "negation" in result:
                return result
        except Exception as e:
            print(f"Error processing: {e}. Retrying...", flush=True)


def process_survey(
    df: pd.DataFrame,
    model: str,
    prompts: dict,
    languages: list[str],
    output_file: str,
):
    results = {}

    for language in languages:
        print(f">>>>> {language}")
        for i, row in df.iterrows():
            print(f"Progress: {(i / len(df)) * 100:.2f}%", flush=True)
            if i not in results:
                results[i] = {}

            while True:
                try:
                    statement = row["statement"][language]
                    analysis = survey_batch(statement, language, model, prompts)

                    results[i][f"original_text_{language}"] = statement
                    results[i][f"question_{language}"] = analysis["question"]
                    results[i][f"negation_{language}"] = analysis["negation"]
                    break
                except Exception as e:
                    print(f"Error for language {language}, row {i}: {e}", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}_{','.join(languages)}.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    print("Processing complete.")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    prompts = load_prompts(args.task_prompts)
    missing = [lang for lang in languages if lang not in prompts]
    if missing:
        raise ValueError(f"Missing prompts for languages: {missing}")

    df = pd.read_json("data/euandi_2019_data/euandi_2019_questionnaire.jsonl", lines=True)
    process_survey(
        df,
        model=args.model,
        prompts=prompts,
        languages=languages,
        output_file=f"data/euandi_2019_results/{args.model}",
    )
