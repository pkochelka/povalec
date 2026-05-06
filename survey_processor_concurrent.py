import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from api_caller import call_api

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5-122b", type=str)
parser.add_argument("--variant", default="_question", type=str, choices=["", "_question", "_negated"])
parser.add_argument("--max_workers", default=4, type=int)
parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)
parser.add_argument("--task_prompts", default="./prompts/survey_processor_concurrent.json", type=str)
parser.add_argument("--dataset", default="euandi_2019", type=str, choices=["euandi_2019", "euandi_2024"])


def load_prompts(path: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"], data["option_lists"]


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def call_survey(statement: str, model: str, options: str, language: str, prompts: dict) -> dict:
    prompt = prompts[language].format(question=statement, options=options)
    while True:
        try:
            response = call_api(prompt, model)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result:
                return result
        except Exception as e:
            print(f"Error: {e}. Retrying...", flush=True)


def process_survey(
    df: pd.DataFrame,
    model: str,
    variant: str,
    prompts: dict[str, str],
    option_lists: dict[str, list[str]],
    languages: list[str],
    dataset: str = "euandi_2019",
    max_workers: int = 8,
):
    os.makedirs(f"./data/{dataset}_results/{model}", exist_ok=True)
    output_file = f"./data/{dataset}_results/{model}/"

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, options in enumerate(option_lists[language]):
                tasks[(i, language, j)] = (statement, options, lang_variant)

    total = len(tasks)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, language, j), (statement, options, lang_variant) in tasks.items():
            future = pool.submit(call_survey, statement, model, options, language, prompts)
            futures[future] = (i, language, j, lang_variant, statement)
        for done, future in enumerate(as_completed(futures), 1):
            i, language, j, lang_variant, statement = futures[future]
            analysis = future.result()

            if i not in results:
                results[i] = {}
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"choice_{lang_variant}_v{j}"] = analysis["choice"]
            results[i][f"reason_{lang_variant}_v{j}"] = analysis["reason"]

            if done % 100 == 0:
                print(f"  {done}/{total} done", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}{','.join(languages)}{variant}.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )
    print("Processing complete.")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    prompts, option_lists = load_prompts(args.task_prompts)
    missing = [lang for lang in languages if lang not in prompts or lang not in option_lists]
    if missing:
        raise ValueError(f"Missing prompts or option lists for languages: {missing}")

    df = pd.read_json(
        f"data/{args.dataset}_data/statements_negated_neutral.jsonl",
        lines=True,
    )
    process_survey(
        df,
        model=args.model,
        variant=args.variant,
        prompts=prompts,
        option_lists=option_lists,
        languages=languages,
        dataset=args.dataset,
        max_workers=args.max_workers,
    )
