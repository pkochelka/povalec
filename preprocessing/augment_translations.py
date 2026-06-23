import argparse
import json
import sys
import threading
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import call_api, save_checkpoint, make_pool

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "EuroParl Custom" / "train.parquet"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translate_speech.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "augmented"

PARTY_COLUMN = "EU Party"
COLUMNS = ["date", PARTY_COLUMN, "text", "language", "speaker"]

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "it": "Italian", "es": "Spanish",
    "de": "German", "pt": "Portuguese", "sv": "Swedish", "el": "Greek",
    "nl": "Dutch", "fi": "Finnish", "da": "Danish", "pl": "Polish",
    "cs": "Czech", "lt": "Lithuanian", "sk": "Slovak", "hu": "Hungarian",
    "et": "Estonian", "sl": "Slovenian", "lv": "Latvian", "ro": "Romanian",
    "bg": "Bulgarian", "hr": "Croatian", "ga": "Irish", "mt": "Maltese",
}

MAX_RETRIES = 5
CHECKPOINT_INTERVAL_SECONDS = 60
PROGRESS_INTERVAL = 200


@dataclass(frozen=True)
class TranslationConfig:
    model: str
    prompt_template: str
    second_provider: bool
    max_tokens: int
    max_completion_tokens: int
    max_source_chars: int


@dataclass(frozen=True)
class TranslationTask:
    key: str
    target_language: str
    source_text: str
    date: object
    party: str
    speaker: str


def translate(task, config):
    prompt = config.prompt_template.format(
        language=LANGUAGE_NAMES.get(task.target_language, task.target_language),
        text=task.source_text[: config.max_source_chars],
    )
    max_tokens = config.max_tokens
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_api(
                prompt,
                config.model,
                second_provider=config.second_provider,
                max_tokens=max_tokens,
            )
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                if max_tokens >= config.max_completion_tokens:
                    raise ValueError(f"truncated even at {max_tokens} tokens (reasoning too long)")
                max_tokens = min(max_tokens * 2, config.max_completion_tokens)
                raise ValueError(f"truncated; raising budget to {max_tokens} tokens")
            if content is None:
                raise ValueError(f"null content (finish_reason={choice.get('finish_reason')!r})")
            content = content.strip().strip('"').strip()
            if not content:
                raise ValueError("empty translation")
            return content
        except Exception as error:
            snippet = task.source_text[:60].replace("\n", " ")
            print(f"  retry {attempt}/{MAX_RETRIES} [{task.target_language}]: {error} | {snippet!r}", flush=True)
            time.sleep(min(2 ** attempt, 10))
    return None


def plan_party_tasks(party, party_df, target_languages, projected_rows, target_per_party):
    tasks = []
    for group_id, group in sorted(party_df.groupby("group"), key=lambda item: item[0]):
        if target_per_party is not None and projected_rows >= target_per_party:
            break
        present_languages = set(group["language"])
        canonical = group.loc[group["text_length"].idxmax()]
        for language in target_languages:
            if language in present_languages:
                continue
            if target_per_party is not None and projected_rows >= target_per_party:
                break
            tasks.append(TranslationTask(
                key=f"{group_id}|||{language}",
                target_language=language,
                source_text=canonical["text"],
                date=canonical["date"],
                party=party,
                speaker=canonical["speaker"],
            ))
            projected_rows += 1
    return tasks


def plan_tasks(df, target_languages, target_per_party):
    df = df.assign(
        group=df["speaker"].astype(str) + "_" + df["date"].astype(str),
        text_length=df["text"].str.len(),
    )

    party_row_counts = df[PARTY_COLUMN].value_counts()
    parties_fewest_first = party_row_counts.sort_values().index.tolist()

    print("Party processing order (fewest rows first):")
    for party in parties_fewest_first:
        print(f"  {party}: {party_row_counts[party]:,} rows")

    tasks = []
    for party in parties_fewest_first:
        tasks.extend(plan_party_tasks(
            party,
            df[df[PARTY_COLUMN] == party],
            target_languages,
            int(party_row_counts[party]),
            target_per_party,
        ))

    print(f"\nPlanned {len(tasks):,} translation tasks for languages {sorted(target_languages)}")
    return tasks


def load_cache(cache_path):
    if not cache_path.exists():
        return {}
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def run_translations(tasks, results, cache_path, config, max_workers):
    pending = [task for task in tasks if not results.get(task.key)]
    print(f"Pending: {len(pending):,} / {len(tasks):,} (cached: {len(tasks) - len(pending):,})")
    if not pending:
        return

    lock = threading.Lock()
    last_checkpoint = time.time()
    succeeded = 0
    failed = 0

    with make_pool(max_workers, config.second_provider) as pool:
        futures = {pool.submit(translate, task, config): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                translation = future.result()
            except Exception as error:
                translation = None
                print(f"  task {task.key} crashed: {error}", flush=True)
            if translation:
                results[task.key] = translation
                succeeded += 1
            else:
                failed += 1

            if completed % PROGRESS_INTERVAL == 0 or completed == len(pending):
                with lock:
                    save_checkpoint(str(cache_path), results)
                    last_checkpoint = time.time()
                print(f"  {completed:,}/{len(pending):,} done | {succeeded:,} ok | {failed:,} failed", flush=True)
            elif time.time() - last_checkpoint > CHECKPOINT_INTERVAL_SECONDS:
                with lock:
                    save_checkpoint(str(cache_path), results)
                    last_checkpoint = time.time()

    save_checkpoint(str(cache_path), results)


def write_output(tasks, results, output_path):
    rows = [
        {
            "date": task.date,
            PARTY_COLUMN: task.party,
            "text": results[task.key],
            "language": task.target_language,
            "speaker": task.speaker,
        }
        for task in tasks
        if results.get(task.key)
    ]
    augmented = pd.DataFrame(rows, columns=COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path, index=False)

    print(f"\nWrote {len(augmented):,} augmented rows -> {output_path}")
    for party, count in augmented[PARTY_COLUMN].value_counts().items():
        print(f"  {party}: {count:,}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Workers per API key when --second-provider; total pool is this times the number of keys (KEY1, KEY2, ... in .env.local). Plain worker count otherwise.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--max-source-chars", type=int, default=6000)
    parser.add_argument("--min-source-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=150000)
    parser.add_argument("--target-per-party", type=int, default=None)
    parser.add_argument("--target-langs", default=None)
    parser.add_argument("--second-provider", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    config = TranslationConfig(
        model=args.model,
        prompt_template=spec["prompt"],
        second_provider=args.second_provider,
        max_tokens=args.max_tokens,
        max_completion_tokens=args.max_completion_tokens,
        max_source_chars=args.max_source_chars,
    )

    df = pd.read_parquet(args.train)[COLUMNS]
    df = df[df["text"].str.len() >= args.min_source_len]
    print(f"Loaded {len(df):,} rows from {args.train}")

    present_languages = sorted(df["language"].unique())
    target_languages = (
        [code.strip() for code in args.target_langs.split(",") if code.strip()]
        if args.target_langs else present_languages
    )
    print(f"Languages present: {present_languages}")

    tasks = plan_tasks(df, target_languages, args.target_per_party)
    if args.limit is not None:
        tasks = tasks[: args.limit]
        print(f"Applied --limit: {len(tasks):,} tasks")

    base_name = args.model.replace("/", "_")
    cache_path = args.output_dir / f"{base_name}.cache.json"
    output_path = args.output_dir / f"{base_name}.augmented.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_cache(cache_path)
    if results:
        print(f"Loaded cache: {len(results):,} translations already done ({cache_path})")

    run_translations(tasks, results, cache_path, config, args.max_workers)
    write_output(tasks, results, output_path)


if __name__ == "__main__":
    main()