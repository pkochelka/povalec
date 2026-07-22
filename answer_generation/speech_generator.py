import json
import os
import sys
import argparse
import time
from concurrent.futures import as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pandas as pd

from utils import ALL_LANGS_STR, call_api, save_checkpoint, make_pool

def load_task_lists(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_speech(statement: str, task: str, model: str, max_retries: int = 5, second_provider: bool = False) -> str | None:
    prompt = f"{task}\n\"{statement}\"\n"
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, model, second_provider=second_provider, enable_thinking=False)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if content:
                return content
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise ValueError(
                    "Truncated before any content (finish_reason=length): the token budget was "
                    "exhausted (likely by reasoning). Disable thinking or raise max_tokens."
                )
            raise ValueError(f"API returned empty content (finish_reason={finish_reason}), full message: {choice['message']}")
        except Exception as e:
            stmt_snippet = statement[:80].replace("\n", " ")
            print(f"Error (attempt {attempt+1}/{max_retries}) [{model}] stmt={stmt_snippet!r}: {e}", flush=True)
        time.sleep(min(2 ** (attempt + 1), 10))
    return None


def speeches_output_path(dataset, model_dir, languages, variant):
    return os.path.join(_ROOT, "data", f"{dataset}_results", model_dir, f"speeches_{','.join(languages)}{variant}.csv")


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
    output_path = speeches_output_path(dataset, dir_name, languages, variant)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

    with make_pool(max_workers, second_provider) as pool:
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


def patch_speeches(
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
    output_path = speeches_output_path(dataset, dir_name, languages, variant)
    if not os.path.exists(output_path):
        print(f"Nothing to patch, output does not exist: {output_path}")
        return

    existing = pd.read_csv(
        output_path, sep=";", encoding="utf-8-sig",
        dtype=str, keep_default_na=False, na_filter=False,
    )

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        variant_indices = sorted(
            int(column.rsplit("_v", 1)[1])
            for column in existing.columns
            if column.startswith(f"answer_{lang_variant}_v")
        )
        for i, row in df.iterrows():
            if i not in existing.index:
                continue
            statement = row["statement"][lang_variant]
            for j in variant_indices:
                if j < len(task_lists[language]) and existing.at[i, f"answer_{lang_variant}_v{j}"] == "":
                    tasks[(i, lang_variant, j)] = (statement, task_lists[language][j])

    if not tasks:
        print(f"No failed responses to patch in {output_path}")
        return

    print(f"Patching {len(tasks)} failed responses in {output_path}", flush=True)
    with make_pool(max_workers, second_provider) as pool:
        futures = {
            pool.submit(call_speech, statement, task, model, second_provider=second_provider): (i, lang_variant, j)
            for (i, lang_variant, j), (statement, task) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            i, lang_variant, j = futures[future]
            answer = future.result()
            existing.at[i, f"answer_{lang_variant}_v{j}"] = "" if answer is None else answer
            if completed % 100 == 0:
                print(f"  {completed}/{len(tasks)} patched", flush=True)
                existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    still_failed = sum(
        1 for (i, lang_variant, j) in tasks
        if existing.at[i, f"answer_{lang_variant}_v{j}"] == ""
    )
    print(f"Patch complete: {len(tasks) - still_failed}/{len(tasks)} filled, {still_failed} still failed.", flush=True)


def overwrite_changed_speeches(
    df: pd.DataFrame,
    model: str,
    variant: str,
    task_lists: dict[str, list[str]],
    languages: list[str],
    dataset: str = "euandi_2019",
    model_dir: str = None,
    max_workers: int = 8,
    second_provider: bool = False,
    dry_run: bool = False,
):
    """Regenerate, in place, only the (row, language) cells whose statement text
    changed since the file was written -- i.e. rows whose statement was rewritten.
    Detected by comparing the stored original_text_<lang_variant> against the
    current statements jsonl; unchanged rows (and the whole base variant, whose
    statements were untouched) are a no-op. Updates original_text, task and answer.
    """
    dir_name = model_dir if model_dir is not None else model
    output_path = speeches_output_path(dataset, dir_name, languages, variant)
    if not os.path.exists(output_path):
        print(f"Nothing to overwrite, output does not exist: {output_path}")
        return

    existing = pd.read_csv(
        output_path, sep=";", encoding="utf-8-sig",
        dtype=str, keep_default_na=False, na_filter=False,
    )

    tasks = {}
    changed_rows = set()
    for language in languages:
        lang_variant = f"{language}{variant}"
        original_col = f"original_text_{lang_variant}"
        if original_col not in existing.columns:
            continue
        variant_indices = sorted(
            int(column.rsplit("_v", 1)[1])
            for column in existing.columns
            if column.startswith(f"answer_{lang_variant}_v")
        )
        for i, row in df.iterrows():
            if i not in existing.index:
                continue
            new_statement = row["statement"].get(lang_variant)
            if new_statement is None:
                continue
            if str(new_statement).strip() == str(existing.at[i, original_col]).strip():
                continue
            changed_rows.add(i)
            for j in variant_indices:
                if j < len(task_lists[language]):
                    tasks[(i, lang_variant, j)] = (new_statement, task_lists[language][j], original_col)

    if not tasks:
        print(f"No rewritten statements to overwrite in {output_path}")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Overwriting {len(tasks)} cells "
          f"across {len(changed_rows)} rewritten rows {sorted(changed_rows)} in {output_path}", flush=True)
    if dry_run:
        return

    with make_pool(max_workers, second_provider) as pool:
        futures = {
            pool.submit(call_speech, statement, task, model, second_provider=second_provider): (i, lang_variant, j, statement, task, original_col)
            for (i, lang_variant, j), (statement, task, original_col) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            i, lang_variant, j, statement, task, original_col = futures[future]
            answer = future.result()
            existing.at[i, original_col] = statement
            existing.at[i, f"task_{lang_variant}_v{j}"] = task
            existing.at[i, f"answer_{lang_variant}_v{j}"] = "" if answer is None else answer
            if completed % 100 == 0:
                print(f"  {completed}/{len(tasks)} overwritten", flush=True)
                existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    still_failed = sum(
        1 for (i, lang_variant, j) in tasks
        if existing.at[i, f"answer_{lang_variant}_v{j}"] == ""
    )
    print(f"Overwrite complete: {len(tasks) - still_failed}/{len(tasks)} filled, {still_failed} still failed.", flush=True)


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
    parser.add_argument("--patch", action="store_true",
                        help="Regenerate only the failed responses in an existing output and patch them in place.")
    parser.add_argument("--overwrite_changed", action="store_true",
                        help="Regenerate in place only the rows whose statement text changed since the file was written.")
    parser.add_argument("--dry_run", action="store_true",
                        help="With --overwrite_changed, only report what would be regenerated; make no API calls.")
    args = parser.parse_args()
    languages = args.languages.split(",")

    task_lists = load_task_lists(args.task_prompts)
    missing = [lang for lang in languages if lang not in task_lists]
    if missing:
        raise ValueError(f"Missing task prompts for languages: {missing}")

    input_file = os.path.join(
        _ROOT, "data", f"{args.dataset}_data",
        "statements.jsonl" if args.variant == "" else "statements_negated.jsonl",
    )
    df = pd.read_json(input_file, lines=True)
    if args.overwrite_changed:
        overwrite_changed_speeches(
            df, model=args.model, variant=args.variant, task_lists=task_lists,
            languages=languages, dataset=args.dataset, model_dir=args.model_dir or args.model,
            max_workers=args.max_workers, second_provider=args.second_provider, dry_run=args.dry_run,
        )
    else:
        run = patch_speeches if args.patch else generate_speeches
        run(
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