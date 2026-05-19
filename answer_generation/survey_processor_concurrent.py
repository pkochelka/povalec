import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import call_api, extract_json, save_checkpoint

def load_prompts(path: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"], data["option_lists"]


def call_survey(statement, model, options, language, prompts, max_retries=5, second_provider=False):
    prompt = prompts[language].format(question=statement, options=options)
    last_content = None
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, model, second_provider=second_provider)
            content = response["choices"][0]["message"]["content"]
            if content is None:
                finish_reason = response["choices"][0].get("finish_reason")
                raise ValueError(f"API returned null content (finish_reason={finish_reason!r}), full message: {response['choices'][0]['message']}")
            last_content = content
            result = extract_json(content)
            if result and "choice" in result and "reason" in result:
                return result
            raise ValueError(f"Unexpected response format — could not extract {{choice, reason}} from: {content!r}")
        except Exception as e:
            stmt_snippet = statement[:80].replace("\n", " ")
            print(f"Error (attempt {attempt+1}/{max_retries}) [{model}/{language}] stmt={stmt_snippet!r}: {e}", flush=True)
        time.sleep(min(2 ** (attempt+1), 10))  # exponential backoff
    reason = f"REFUSED: {last_content}" if last_content is not None else "FAILED"
    return {"choice": None, "reason": reason}


def process_survey(
    df: pd.DataFrame,
    model: str,
    variant: str,
    prompts: dict[str, str],
    option_lists: dict[str, list[str]],
    languages: list[str],
    dataset: str = "euandi_2019",
    model_dir: str = None,
    max_workers: int = 8,
    second_provider: bool = False,
):
    model_dir = model_dir or model
    os.makedirs(f"./data/{dataset}_results/{model_dir}", exist_ok=True)
    output_path = f"./data/{dataset}_results/{model_dir}/{','.join(languages)}{variant}.csv"
    checkpoint_path = output_path + ".ckpt.json"

    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        return

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, options in enumerate(option_lists[language]):
                tasks[(i, language, j)] = (statement, options, lang_variant)

    total = len(tasks)
    results = {}

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            raw = json.load(f)
        results = {int(k): v for k, v in raw.items()}
        already_done = sum(
            1 for (i, _, j), (_, _, lang_variant) in tasks.items()
            if i in results and f"choice_{lang_variant}_v{j}" in results[i]
        )
        print(f"Resuming from checkpoint: {already_done}/{total} already done", flush=True)
    else:
        already_done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, language, j), (statement, options, lang_variant) in tasks.items():
            if i in results and f"choice_{lang_variant}_v{j}" in results[i]:
                continue
            future = pool.submit(call_survey, statement, model, options, language, prompts, second_provider=second_provider)
            futures[future] = (i, language, j, lang_variant, statement)

        for completed, future in enumerate(as_completed(futures), already_done + 1):
            i, language, j, lang_variant, statement = futures[future]
            analysis = future.result()

            if i not in results:
                results[i] = {}
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"choice_{lang_variant}_v{j}"] = analysis["choice"]
            results[i][f"reason_{lang_variant}_v{j}"] = analysis["reason"]

            if completed % 100 == 0:
                print(f"  {completed}/{total} done", flush=True)
                save_checkpoint(checkpoint_path, results)

    save_checkpoint(checkpoint_path, results)
    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        output_path,
        sep=";", index=False, encoding="utf-8-sig",
    )
    os.remove(checkpoint_path)
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-122b", type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--variant", default="_question", type=str, choices=["", "_question", "_negated"])
    parser.add_argument("--max_workers", default=4, type=int)
    parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)
    parser.add_argument("--task_prompts", default="./prompts/survey_processor_concurrent.json", type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--second_provider", action="store_true")
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
        model_dir=args.model_dir or args.model,
        max_workers=args.max_workers,
        second_provider=args.second_provider,
    )
