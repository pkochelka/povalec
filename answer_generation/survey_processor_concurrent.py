import os
import sys
import json
import argparse
from concurrent.futures import as_completed
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pandas as pd
from utils import ALL_LANGS_STR, call_api, extract_json, save_checkpoint, make_pool

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


def survey_output_path(dataset, model_dir, languages, variant):
    return os.path.join(_ROOT, "data", f"{dataset}_results", model_dir, f"{','.join(languages)}{variant}.csv")


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
    output_path = survey_output_path(dataset, model_dir, languages, variant)
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

    with make_pool(max_workers, second_provider) as pool:
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


def patch_survey(
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
    output_path = survey_output_path(dataset, model_dir, languages, variant)
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
            if column.startswith(f"choice_{lang_variant}_v")
        )
        for i, row in df.iterrows():
            if i not in existing.index:
                continue
            statement = row["statement"][lang_variant]
            for j in variant_indices:
                if j < len(option_lists[language]) and existing.at[i, f"choice_{lang_variant}_v{j}"] == "":
                    tasks[(i, language, lang_variant, j)] = (statement, option_lists[language][j])

    if not tasks:
        print(f"No failed or refused responses to patch in {output_path}")
        return

    print(f"Patching {len(tasks)} failed/refused responses in {output_path}", flush=True)
    with make_pool(max_workers, second_provider) as pool:
        futures = {
            pool.submit(call_survey, statement, model, options, language, prompts, second_provider=second_provider): (i, lang_variant, j)
            for (i, language, lang_variant, j), (statement, options) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            i, lang_variant, j = futures[future]
            result = future.result()
            existing.at[i, f"choice_{lang_variant}_v{j}"] = "" if result["choice"] is None else str(result["choice"])
            existing.at[i, f"reason_{lang_variant}_v{j}"] = result["reason"]
            if completed % 100 == 0:
                print(f"  {completed}/{len(tasks)} patched", flush=True)
                existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    still_failed = sum(
        1 for (i, _language, lang_variant, j) in tasks
        if existing.at[i, f"choice_{lang_variant}_v{j}"] == ""
    )
    print(f"Patch complete: {len(tasks) - still_failed}/{len(tasks)} filled, {still_failed} still failed/refused.", flush=True)


def overwrite_changed_survey(
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
    dry_run: bool = False,
):
    """Regenerate, in place, only the (row, language) cells whose statement text
    changed since the file was written -- i.e. rows whose statement was rewritten.
    Detected by comparing the stored original_text_<lang_variant> against the
    current statements jsonl; unchanged rows (and the whole base variant, whose
    statements were untouched) are a no-op. Updates original_text, choice, reason.
    """
    model_dir = model_dir or model
    output_path = survey_output_path(dataset, model_dir, languages, variant)
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
            if column.startswith(f"choice_{lang_variant}_v")
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
                if j < len(option_lists[language]):
                    tasks[(i, language, lang_variant, j)] = (new_statement, option_lists[language][j], original_col)

    if not tasks:
        print(f"No rewritten statements to overwrite in {output_path}")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Overwriting {len(tasks)} cells "
          f"across {len(changed_rows)} rewritten rows {sorted(changed_rows)} in {output_path}", flush=True)
    if dry_run:
        return

    with make_pool(max_workers, second_provider) as pool:
        futures = {
            pool.submit(call_survey, statement, model, options, language, prompts, second_provider=second_provider): (i, lang_variant, j, statement, original_col)
            for (i, language, lang_variant, j), (statement, options, original_col) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            i, lang_variant, j, statement, original_col = futures[future]
            result = future.result()
            existing.at[i, original_col] = statement
            existing.at[i, f"choice_{lang_variant}_v{j}"] = "" if result["choice"] is None else str(result["choice"])
            existing.at[i, f"reason_{lang_variant}_v{j}"] = result["reason"]
            if completed % 100 == 0:
                print(f"  {completed}/{len(tasks)} overwritten", flush=True)
                existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    still_failed = sum(
        1 for (i, _language, lang_variant, j) in tasks
        if existing.at[i, f"choice_{lang_variant}_v{j}"] == ""
    )
    print(f"Overwrite complete: {len(tasks) - still_failed}/{len(tasks)} filled, {still_failed} still failed/refused.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5-122b", type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--variant", default="_question", type=str, choices=["", "_question", "_negated"])
    parser.add_argument("--max_workers", default=10, type=int)
    parser.add_argument("--languages", default=ALL_LANGS_STR, type=str)
    parser.add_argument("--task_prompts", default=os.path.join(_ROOT, "prompts", "survey_processor_concurrent.json"), type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--second_provider", action="store_true")
    parser.add_argument("--patch", action="store_true",
                        help="Regenerate only the failed/refused responses in an existing output and patch them in place.")
    parser.add_argument("--overwrite_changed", action="store_true",
                        help="Regenerate in place only the rows whose statement text changed since the file was written.")
    parser.add_argument("--dry_run", action="store_true",
                        help="With --overwrite_changed, only report what would be regenerated; make no API calls.")
    args = parser.parse_args()
    languages = args.languages.split(",")

    prompts, option_lists = load_prompts(args.task_prompts)
    missing = [lang for lang in languages if lang not in prompts or lang not in option_lists]
    if missing:
        raise ValueError(f"Missing prompts or option lists for languages: {missing}")

    df = pd.read_json(
        os.path.join(_ROOT, "data", f"{args.dataset}_data", "statements_negated.jsonl"),
        lines=True,
    )
    if args.overwrite_changed:
        overwrite_changed_survey(
            df, model=args.model, variant=args.variant, prompts=prompts, option_lists=option_lists,
            languages=languages, dataset=args.dataset, model_dir=args.model_dir or args.model,
            max_workers=args.max_workers, second_provider=args.second_provider, dry_run=args.dry_run,
        )
    else:
        runner = patch_survey if args.patch else process_survey
        runner(
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
