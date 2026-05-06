import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from api_caller import call_api

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5-122b", type=str, choices=["gpt-oss-120b", "qwen3.5-122b"])
parser.add_argument("--variant", default="_negated", type=str, choices=["", "_question", "_negated"])
parser.add_argument("--max_workers", default=3, type=int)
parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)
parser.add_argument("--task_prompts", default="./prompts/generate_speeches_open_ended.json", type=str)
parser.add_argument("--dataset", default="euandi_2019", type=str, choices=["euandi_2019", "euandi_2024"])


def load_task_lists(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_speech(statement: str, task: str, model: str) -> str:
    prompt = f"Task: {task}\n\"{statement}\"\n"
    while True:
        try:
            response = call_api(prompt, model)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception as e:
            print(f"Error: {e}. Retrying...", flush=True)


def generate_speeches(
    df: pd.DataFrame,
    model: str,
    variant: str,
    task_lists: dict[str, list[str]],
    languages: list[str],
    dataset: str = "euandi_2019",
    max_workers: int = 8,
):
    os.makedirs(f"./data/{dataset}_results/{model}", exist_ok=True)
    output_file = f"./data/{dataset}_results/{model}/speeches"

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, task in enumerate(task_lists[language]):
                tasks[(i, language, j)] = (statement, task, lang_variant)

    total = len(tasks)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_speech, statement, task, model): (i, j, lang_variant, statement, task)
            for (i, language, j), (statement, task, lang_variant) in tasks.items()
        }

        for done, future in enumerate(as_completed(futures), 1):
            i, j, lang_variant, statement, task = futures[future]
            answer = future.result()

            results.setdefault(i, {})
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"task_{lang_variant}_v{j}"] = task
            results[i][f"answer_{lang_variant}_v{j}"] = answer

            if done % 100 == 0:
                print(f"  {done}/{total} done", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}_{','.join(languages)}{variant}.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )
    print("Processing complete.")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    task_lists = load_task_lists(args.task_prompts)
    missing = [lang for lang in languages if lang not in task_lists]
    if missing:
        raise ValueError(f"Missing task prompts for languages: {missing}")

    input_file = (
        f"data/{args.dataset}_data/statements.jsonl"
        if args.variant == ""
        else f"data/{args.dataset}_data/statements_negated_neutral.jsonl"
    )
    df = pd.read_json(input_file, lines=True)
    generate_speeches(
        df,
        model=args.model,
        variant=args.variant,
        task_lists=task_lists,
        languages=languages,
        dataset=args.dataset,
        max_workers=args.max_workers,
    )