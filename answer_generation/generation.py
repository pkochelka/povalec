"""Shared generation engine for the two elicitation tracks.

The direct track (survey_processor_concurrent) and the indirect one
(speech_generator) differ only in how a prompt is built and what columns a
response fills. Everything else -- the task fan-out over
(statement x language x prompt variant), the resumable checkpoint, the thread
pool, the patch/overwrite-changed modes, the CSV layout -- was duplicated
function-for-function between them. That scaffolding lives here once; each track
supplies a `Track` describing only its differences.

A "unit" is one prompt variant for one language: an (option list, template) pair
for the survey track, a task string for the speeches track. `call` turns a
(statement, unit) into the dict of column prefixes to write, e.g.
{"choice": 4, "reason": "..."} or {"task": "...", "answer": "..."}.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import ALL_LANGS_STR, VARIANTS, save_checkpoint

CHECKPOINT_EVERY = 100


@dataclass
class Track:
    name: str
    # Output basename, formatted with {langs} and {variant}.
    filename: str
    # Column prefix that decides whether a cell is done / still failed.
    primary: str
    # prompt-file path -> {language: [unit, ...]}
    load_units: Callable[[str], dict]
    # (statement, unit, model) -> {column_prefix: value}
    call: Callable[..., dict]
    # Which statements jsonl this track reads for a given variant.
    statements_file: Callable[[str], str]
    default_workers: int = 8
    default_variant: str = ""
    default_prompts: str = ""
    # Noun used in the "nothing to patch" message ("failed" vs "failed/refused").
    failure_noun: str = "failed"


def output_path(track, dataset, model_dir, languages, variant):
    name = track.filename.format(langs=",".join(languages), variant=variant)
    return os.path.join(_ROOT, "data", f"{dataset}_results", model_dir, name)


def _read_existing(path):
    return pd.read_csv(path, sep=";", encoding="utf-8-sig",
                       dtype=str, keep_default_na=False, na_filter=False)


def _variant_indices(existing, track, lang_variant):
    return sorted(
        int(column.rsplit("_v", 1)[1])
        for column in existing.columns
        if column.startswith(f"{track.primary}_{lang_variant}_v")
    )


def _write_cells(existing, row_index, lang_variant, index, cells):
    for prefix, value in cells.items():
        existing.at[row_index, f"{prefix}_{lang_variant}_{'v' + str(index)}"] = (
            "" if value is None else str(value)
        )


def _still_failed(existing, track, keys):
    return sum(1 for (i, _language, lang_variant, j) in keys
               if existing.at[i, f"{track.primary}_{lang_variant}_v{j}"] == "")


def generate(track, df, model, variant, units, languages, dataset,
             model_dir=None, max_workers=8):
    """Full run: every (statement, language, prompt variant) cell, resumable."""
    model_dir = model_dir or model
    path = output_path(track, dataset, model_dir, languages, variant)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint_path = path + ".ckpt.json"

    if os.path.exists(path):
        print(f"Output already exists: {path}")
        return

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, unit in enumerate(units[language]):
                tasks[(i, language, j)] = (statement, unit, lang_variant)

    total = len(tasks)
    results = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            results = {int(k): v for k, v in json.load(f).items()}
        already_done = sum(
            1 for (i, _, j), (_, _, lang_variant) in tasks.items()
            if i in results and f"{track.primary}_{lang_variant}_v{j}" in results[i]
        )
        print(f"Resuming from checkpoint: {already_done}/{total} already done", flush=True)
    else:
        already_done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, _language, j), (statement, unit, lang_variant) in tasks.items():
            if i in results and f"{track.primary}_{lang_variant}_v{j}" in results[i]:
                continue
            future = pool.submit(track.call, statement, unit, model)
            futures[future] = (i, j, lang_variant, statement)

        for completed, future in enumerate(as_completed(futures), already_done + 1):
            i, j, lang_variant, statement = futures[future]
            cells = future.result()

            results.setdefault(i, {})
            results[i][f"original_text_{lang_variant}"] = statement
            for prefix, value in cells.items():
                results[i][f"{prefix}_{lang_variant}_v{j}"] = value

            if completed % CHECKPOINT_EVERY == 0:
                print(f"  {completed}/{total} done", flush=True)
                save_checkpoint(checkpoint_path, results)

    save_checkpoint(checkpoint_path, results)
    pd.DataFrame([results[i] for i in sorted(results)]).to_csv(
        path, sep=";", index=False, encoding="utf-8-sig")
    os.remove(checkpoint_path)
    print("Processing complete.")


def patch(track, df, model, variant, units, languages, dataset,
          model_dir=None, max_workers=8):
    """Refill only the cells that came back empty, in place."""
    model_dir = model_dir or model
    path = output_path(track, dataset, model_dir, languages, variant)
    if not os.path.exists(path):
        print(f"Nothing to patch, output does not exist: {path}")
        return
    existing = _read_existing(path)

    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for j in _variant_indices(existing, track, lang_variant):
            if j >= len(units[language]):
                continue
            for i, row in df.iterrows():
                if i in existing.index and existing.at[i, f"{track.primary}_{lang_variant}_v{j}"] == "":
                    tasks[(i, language, lang_variant, j)] = (
                        row["statement"][lang_variant], units[language][j])

    if not tasks:
        print(f"No {track.failure_noun} responses to patch in {path}")
        return

    print(f"Patching {len(tasks)} {track.failure_noun} responses in {path}", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(track.call, statement, unit, model): key
            for key, (statement, unit) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            i, _language, lang_variant, j = futures[future]
            _write_cells(existing, i, lang_variant, j, future.result())
            if completed % CHECKPOINT_EVERY == 0:
                print(f"  {completed}/{len(tasks)} patched", flush=True)
                existing.to_csv(path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    failed = _still_failed(existing, track, tasks)
    print(f"Patch complete: {len(tasks) - failed}/{len(tasks)} filled, "
          f"{failed} still {track.failure_noun}.", flush=True)


def overwrite_changed(track, df, model, variant, units, languages, dataset,
                      model_dir=None, max_workers=8, dry_run=False):
    """Regenerate, in place, only the cells whose statement text changed since the
    file was written -- i.e. rows whose statement was rewritten. Detected by
    comparing the stored original_text_<lang_variant> against the current statements
    jsonl; unchanged rows (and the whole base variant, whose statements were
    untouched) are a no-op."""
    model_dir = model_dir or model
    path = output_path(track, dataset, model_dir, languages, variant)
    if not os.path.exists(path):
        print(f"Nothing to overwrite, output does not exist: {path}")
        return
    existing = _read_existing(path)

    tasks, changed_rows = {}, set()
    for language in languages:
        lang_variant = f"{language}{variant}"
        original_col = f"original_text_{lang_variant}"
        if original_col not in existing.columns:
            continue
        indices = _variant_indices(existing, track, lang_variant)
        for i, row in df.iterrows():
            if i not in existing.index:
                continue
            new_statement = row["statement"].get(lang_variant)
            if new_statement is None:
                continue
            if str(new_statement).strip() == str(existing.at[i, original_col]).strip():
                continue
            changed_rows.add(i)
            for j in indices:
                if j < len(units[language]):
                    tasks[(i, language, lang_variant, j)] = (
                        new_statement, units[language][j], original_col)

    if not tasks:
        print(f"No rewritten statements to overwrite in {path}")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Overwriting {len(tasks)} cells across "
          f"{len(changed_rows)} rewritten rows {sorted(changed_rows)} in {path}", flush=True)
    if dry_run:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(track.call, statement, unit, model):
                (key, statement, original_col)
            for key, (statement, unit, original_col) in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), 1):
            (i, _language, lang_variant, j), statement, original_col = futures[future]
            existing.at[i, original_col] = statement
            _write_cells(existing, i, lang_variant, j, future.result())
            if completed % CHECKPOINT_EVERY == 0:
                print(f"  {completed}/{len(tasks)} overwritten", flush=True)
                existing.to_csv(path, sep=";", index=False, encoding="utf-8-sig")

    existing.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    failed = _still_failed(existing, track, tasks)
    print(f"Overwrite complete: {len(tasks) - failed}/{len(tasks)} filled, "
          f"{failed} still {track.failure_noun}.", flush=True)


def parse_args(track):
    parser = argparse.ArgumentParser(description=f"Generate the {track.name} track.")
    parser.add_argument("--model", default="qwen3.5-122b", type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--variant", default=track.default_variant, type=str, choices=VARIANTS)
    parser.add_argument("--max_workers", default=track.default_workers, type=int)
    parser.add_argument("--languages", default=ALL_LANGS_STR, type=str)
    parser.add_argument("--task_prompts",
                        default=os.path.join(_ROOT, "prompts", track.default_prompts), type=str)
    parser.add_argument("--dataset", default="euandi_2024", type=str,
                        choices=["euandi_2024"])
    parser.add_argument("--patch", action="store_true",
                        help="Regenerate only the failed/refused responses in an existing "
                             "output and patch them in place.")
    parser.add_argument("--overwrite_changed", action="store_true",
                        help="Regenerate in place only the rows whose statement text changed "
                             "since the file was written.")
    parser.add_argument("--dry_run", action="store_true",
                        help="With --overwrite_changed, only report what would be regenerated; "
                             "make no API calls.")
    return parser.parse_args()


def main(track):
    args = parse_args(track)
    languages = args.languages.split(",")

    units = track.load_units(args.task_prompts)
    missing = [lang for lang in languages if lang not in units]
    if missing:
        raise ValueError(f"Missing prompts for languages: {missing}")

    df = pd.read_json(
        os.path.join(_ROOT, "data", f"{args.dataset}_data", track.statements_file(args.variant)),
        lines=True,
    )
    common = dict(
        df=df, model=args.model, variant=args.variant, units=units, languages=languages,
        dataset=args.dataset, model_dir=args.model_dir or args.model,
        max_workers=args.max_workers,
    )
    if args.overwrite_changed:
        overwrite_changed(track, dry_run=args.dry_run, **common)
    elif args.patch:
        patch(track, **common)
    else:
        generate(track, **common)
