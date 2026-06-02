import json
import os
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pandas as pd

from utils import ALL_LANGS_STR, call_api, save_checkpoint

def load_task_lists(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_speech(statement: str, task: str, model: str, max_retries: int = 5, second_provider: bool = False) -> str | None:
    prompt = f"Task: {task}\n\"{statement}\"\n"
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, model, second_provider=second_provider)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
            raise ValueError(f"API returned empty content, full message: {response['choices'][0]['message']}")
        except Exception as e:
            stmt_snippet = statement[:80].replace("\n", " ")
            print(f"Error (attempt {attempt+1}/{max_retries}) [{model}] stmt={stmt_snippet!r}: {e}", flush=True)
        time.sleep(min(2 ** (attempt + 1), 10))
    return None


def generate_speeches(
    df: pd.DataFrame,
    model: str,
    variant: str,
    task_lists: dict[str, list[str]],
    languages: list[str],
    dataset: str = "euandi_2019",
    model_dir: str = None,
    max_workers: int = 8,
    second_provider: bool = False,
):
    dir_name = model_dir if model_dir is not None else model
    out_dir = os.path.join(_ROOT, "data", f"{dataset}_results", dir_name)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"speeches_{','.join(languages)}{variant}.csv")
    checkpoint_path = output_path + ".ckpt.json"

    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        return

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, task in enumerate(task_lists[language]):
                tasks[(i, language, j)] = (statement, task, lang_variant)

    total = len(tasks)
    results = {}

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            raw = json.load(f)
        results = {int(k): v for k, v in raw.items()}
        already_done = sum(
            1 for (i, _, j), (_, _, lang_variant) in tasks.items()
            if i in results and f"answer_{lang_variant}_v{j}" in results[i]
        )
        print(f"Resuming from checkpoint: {already_done}/{total} already done", flush=True)
    else:
        already_done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, language, j), (statement, task, lang_variant) in tasks.items():
            if i in results and f"answer_{lang_variant}_v{j}" in results[i]:
                continue
            future = pool.submit(call_speech, statement, task, model, second_provider=second_provider)
            futures[future] = (i, j, lang_variant, statement, task)

        for completed, future in enumerate(as_completed(futures), already_done + 1):
            i, j, lang_variant, statement, task = futures[future]
            answer = future.result()

            results.setdefault(i, {})
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"task_{lang_variant}_v{j}"] = task
            results[i][f"answer_{lang_variant}_v{j}"] = answer

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
    parser.add_argument("--model", default="qwen3.5-122b", type=str)#, choices=["gpt-oss-120b", "qwen3.5-122b"])
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--variant", default="_negated", type=str, choices=["", "_question", "_negated"])
    parser.add_argument("--max_workers", default=4, type=int)
    parser.add_argument("--languages", default=ALL_LANGS_STR, type=str)
    parser.add_argument("--task_prompts", default=os.path.join(_ROOT, "prompts", "generate_speeches.json"), type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--second_provider", action="store_true")
    args = parser.parse_args()
    languages = args.languages.split(",")

    task_lists = load_task_lists(args.task_prompts)
    missing = [lang for lang in languages if lang not in task_lists]
    if missing:
        raise ValueError(f"Missing task prompts for languages: {missing}")

    input_file = os.path.join(
        _ROOT, "data", f"{args.dataset}_data",
        "statements.jsonl" if args.variant == "" else "statements_negated_neutral.jsonl",
    )
    df = pd.read_json(input_file, lines=True)
    generate_speeches(
        df,
        model=args.model,
        variant=args.variant,
        task_lists=task_lists,
        languages=languages,
        dataset=args.dataset,
        model_dir=args.model_dir or args.model,
        max_workers=args.max_workers,
        second_provider=args.second_provider,
    )