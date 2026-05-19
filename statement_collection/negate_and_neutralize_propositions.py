import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import ALL_LANGS_STR, call_api, extract_json

def load_prompts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def survey_batch(proposition: str, language: str, model: str, prompts: dict) -> dict:
    user_prompt = prompts[language].format(proposition=proposition)
    attempt = 0
    while True:
        attempt += 1
        try:
            response = call_api(user_prompt, model)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result and "question" in result and "negation" in result:
                return result
            missing = [k for k in ("question", "negation") if not result or k not in result]
            print(
                f"[attempt {attempt}] Bad response (missing {missing}) for [{language}] "
                f"{proposition[:60]!r} — retrying. Content: {content[:200]!r}",
                flush=True,
            )
        except BaseException as e:
            print(f"[attempt {attempt}] Error [{language}] {proposition[:60]!r}: {e}. Retrying...", flush=True)


def process_survey(
    df: pd.DataFrame,
    model: str,
    prompts: dict,
    languages: list[str],
    output_file: str,
    max_workers: int = 4,
):
    tasks = {}
    for language in languages:
        for i, row in df.iterrows():
            tasks[(i, language)] = row["statement"][language]

    total = len(tasks)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(survey_batch, statement, language, model, prompts): (i, language, statement)
            for (i, language), statement in tasks.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            i, language, statement = futures[future]
            analysis = future.result()

            if i not in results:
                results[i] = {}
            results[i][f"original_text_{language}"] = statement
            results[i][f"question_{language}"] = analysis["question"]
            results[i][f"negation_{language}"] = analysis["negation"]

            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} done", flush=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kimi-k2.6", type=str, choices=["kimi-k2.6"])
    parser.add_argument("--languages", default=ALL_LANGS_STR, type=str)
    parser.add_argument("--task_prompts", default="./prompts/negate_and_neutralize.json", type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--max_workers", default=4, type=int)
    args = parser.parse_args()
    languages = args.languages.split(",")

    prompts = load_prompts(args.task_prompts)
    missing = [lang for lang in languages if lang not in prompts]
    if missing:
        raise ValueError(f"Missing prompts for languages: {missing}")

    df = pd.read_json(f"data/{args.dataset}_data/statements.jsonl", lines=True)
    process_survey(
        df,
        model=args.model,
        prompts=prompts,
        languages=languages,
        output_file=f"data/{args.dataset}_results/{args.model}",
        max_workers=args.max_workers,
    )

    
