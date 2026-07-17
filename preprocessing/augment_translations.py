import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import call_api, save_checkpoint, make_pool, second_provider_num_keys

DEFAULT_TRAIN = PROJECT_ROOT / "data" / "EuroParl Custom" / "cleaned"/"train.parquet"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "translate_speech.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "augmented"

# Old single-model cache, reused read-only and folded into the joint cache.
LEGACY_GLM_CACHE = "glm-5.2.cache.json"
LEGACY_GLM_MODEL = "glm-5.2"

PARTY_COLUMN = "EU Party"
COLUMNS = ["date", PARTY_COLUMN, "text", "language", "speaker"]
OUTPUT_COLUMNS = COLUMNS + ["model"]

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "it": "Italian", "es": "Spanish",
    "de": "German", "pt": "Portuguese", "sv": "Swedish", "el": "Greek",
    "nl": "Dutch", "fi": "Finnish", "da": "Danish", "pl": "Polish",
    "cs": "Czech", "lt": "Lithuanian", "sk": "Slovak", "hu": "Hungarian",
    "et": "Estonian", "sl": "Slovenian", "lv": "Latvian", "ro": "Romanian",
    "bg": "Bulgarian",
}

MAX_RETRIES = 2
CHECKPOINT_INTERVAL_SECONDS = 60
PROGRESS_INTERVAL = 200


@dataclass(frozen=True)
class TranslationConfig:
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


def translate(task, model, config):
    prompt = config.prompt_template.format(
        language=LANGUAGE_NAMES.get(task.target_language, task.target_language),
        text=task.source_text[: config.max_source_chars],
    )
    max_tokens = config.max_tokens
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_api(
                prompt,
                model,
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
            print(f"  retry {attempt}/{MAX_RETRIES} [{model} {task.target_language}]: {error} | {snippet!r}", flush=True)
            time.sleep(min(2 ** attempt, 10))
    return None


def _iter_candidates(party_df, target_languages):
    for group_id, group in sorted(party_df.groupby("group"), key=lambda item: item[0]):
        present_languages = set(group["language"])
        missing = [lang for lang in target_languages if lang not in present_languages]
        if not missing:
            continue
        # rank r = r-th longest text per language: distinct speeches by the same
        # speaker on the same day. Each rank is planned as its own speech, but
        # still only into languages absent from the whole group, so a language
        # never receives a translation of content it may already contain.
        for rank in sorted(group["rank"].unique()):
            subgroup = group[group["rank"] == rank]
            canonical = subgroup.loc[subgroup["text_length"].idxmax()]
            sub_id = group_id if rank == 0 else f"{group_id}#r{rank}"
            for language in missing:
                yield sub_id, canonical, language


def plan_party_tasks(party, party_df, target_languages, projected_rows, target_per_party,
                     existing_keys, translated_target):
    candidates = []
    n_cached = 0
    for sub_id, canonical, language in _iter_candidates(party_df, target_languages):
        if target_per_party is not None and projected_rows >= target_per_party:
            break
        key = f"{sub_id}|||{language}"
        cached = key in existing_keys
        n_cached += cached
        candidates.append((key, canonical, language, cached))
        projected_rows += 1

    # Cached translations always stay in the plan (and thus the output); the
    # target only controls how many new ones are added on top of them.
    new_quota = None if translated_target is None else max(0, translated_target - n_cached)

    tasks = []
    new_count = 0
    for key, canonical, language, cached in candidates:
        if not cached:
            if new_quota is not None and new_count >= new_quota:
                continue
            new_count += 1
        tasks.append(TranslationTask(
            key=key,
            target_language=language,
            source_text=canonical["text"],
            date=canonical["date"],
            party=party,
            speaker=canonical["speaker"],
        ))
    return tasks, new_count


def parse_party_targets(spec):
    """'ID=97000,ECR=63000,*=10000' -> ({'ID': 97000, 'ECR': 63000}, 10000)."""
    targets, default = {}, None
    if not spec:
        return targets, default
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        if not sep:
            sys.exit(f"Bad --translated-per-party entry {part!r}, expected PARTY=N")
        if name.strip() == "*":
            default = int(value)
        else:
            targets[name.strip()] = int(value)
    return targets, default


def plan_tasks(df, target_languages, target_per_party, existing_keys, party_targets, default_target):
    df = df.assign(
        group=df["speaker"].astype(str) + "_" + df["date"].astype(str),
        text_length=df["text"].str.len(),
    )
    df["rank"] = (df.groupby(["group", "language"])["text_length"]
                  .rank(method="first", ascending=False).astype(int) - 1)

    party_row_counts = df[PARTY_COLUMN].value_counts()
    parties_fewest_first = party_row_counts.sort_values().index.tolist()

    unknown = set(party_targets) - set(party_row_counts.index)
    if unknown:
        sys.exit(f"--translated-per-party names unknown parties: {sorted(unknown)} "
                 f"(data has: {sorted(party_row_counts.index)})")

    print("Party processing order (fewest rows first):")
    for party in parties_fewest_first:
        print(f"  {party}: {party_row_counts[party]:,} rows")

    tasks = []
    total_new = 0
    for party in parties_fewest_first:
        target = party_targets.get(party, default_target)
        party_tasks, party_new = plan_party_tasks(
            party,
            df[df[PARTY_COLUMN] == party],
            target_languages,
            int(party_row_counts[party]),
            target_per_party,
            existing_keys,
            target,
        )
        tasks.extend(party_tasks)
        total_new += party_new
        target_text = f"{target:,}" if target is not None else "none"
        print(f"  {party}: {party_new:,} new + {len(party_tasks) - party_new:,} cached "
              f"= {len(party_tasks):,} planned (target {target_text})")

    print(f"\nPlanned {len(tasks):,} translation tasks ({total_new:,} new) for languages {sorted(target_languages)}")
    return tasks


def _normalize_entry(value, default_model):
    # Old caches stored a bare string; new ones store {"text", "model"}.
    if isinstance(value, dict):
        return {"text": value["text"], "model": value.get("model", default_model)}
    return {"text": value, "model": default_model}


def load_results(cache_path, legacy_glm_cache_path):
    results = {}

    if legacy_glm_cache_path.exists():
        with open(legacy_glm_cache_path, encoding="utf-8") as f:
            legacy = json.load(f)
        for key, value in legacy.items():
            results[key] = _normalize_entry(value, LEGACY_GLM_MODEL)
        print(f"Seeded {len(legacy):,} translations from legacy glm cache ({legacy_glm_cache_path})")

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            joint = json.load(f)
        for key, value in joint.items():
            results[key] = _normalize_entry(value, LEGACY_GLM_MODEL)
        print(f"Loaded joint cache: {len(joint):,} translations ({cache_path})")

    return results


def run_translations(tasks, results, cache_path, config, models, per_key_workers):
    pending = [task for task in tasks if not results.get(task.key)]
    print(f"Pending: {len(pending):,} / {len(tasks):,} (cached: {len(tasks) - len(pending):,})")
    if not pending:
        return

    work = queue.Queue()
    for task in pending:
        work.put(task)

    # Workers share one queue and are split across models round-robin, so the
    # faster model drains more tasks instead of being held to a fixed share.
    if config.second_provider:
        total_workers = per_key_workers * max(1, second_provider_num_keys())
    else:
        total_workers = per_key_workers
    total_workers = min(total_workers, len(pending))

    lock = threading.Lock()
    state = {"done": 0, "ok": 0, "failed": 0, "last_checkpoint": time.time()}
    per_model = {model: 0 for model in models}
    total = len(pending)

    def worker(model):
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            try:
                translation = translate(task, model, config)
            except Exception as error:
                translation = None
                print(f"  task {task.key} crashed: {error}", flush=True)

            with lock:
                state["done"] += 1
                if translation:
                    results[task.key] = {"text": translation, "model": model}
                    state["ok"] += 1
                    per_model[model] += 1
                else:
                    state["failed"] += 1

                done = state["done"]
                due = time.time() - state["last_checkpoint"] > CHECKPOINT_INTERVAL_SECONDS
                if done % PROGRESS_INTERVAL == 0 or done == total or due:
                    save_checkpoint(str(cache_path), results)
                    state["last_checkpoint"] = time.time()
                    if done % PROGRESS_INTERVAL == 0 or done == total:
                        by_model = " | ".join(f"{m}: {c:,}" for m, c in per_model.items())
                        print(f"  {done:,}/{total:,} done | {state['ok']:,} ok | {state['failed']:,} failed | {by_model}", flush=True)

    with make_pool(per_key_workers, config.second_provider) as pool:
        futures = [pool.submit(worker, models[i % len(models)]) for i in range(total_workers)]
        for future in futures:
            future.result()

    save_checkpoint(str(cache_path), results)


def write_output(tasks, results, output_path):
    rows = []
    for task in tasks:
        entry = results.get(task.key)
        if not entry:
            continue
        rows.append({
            "date": task.date,
            PARTY_COLUMN: task.party,
            "text": entry["text"],
            "language": task.target_language,
            "speaker": task.speaker,
            "model": entry["model"],
        })
    augmented = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path, index=False)

    print(f"\nWrote {len(augmented):,} augmented rows -> {output_path}")
    for model, count in augmented["model"].value_counts().items():
        print(f"  [{model}]: {count:,}")
    for party, count in augmented[PARTY_COLUMN].value_counts().items():
        print(f"  {party}: {count:,}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="deepseek-v4-pro",
                        help="Comma-separated model ids translated concurrently from a shared work queue. "
                             "Each translated speech is tagged with the model that produced it.")
    parser.add_argument("--cache-name", default="translations",
                        help="Base name for the joint cache / output files in --output-dir.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Workers per API key when --second-provider; total pool is this times the number of keys (KEY1, KEY2, ... in .env.local). Plain worker count otherwise.")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--max-source-chars", type=int, default=6000)
    parser.add_argument("--min-source-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=300000)
    parser.add_argument("--target-per-party", type=int, default=None,
                        help="Stop planning once a party's projected rows (existing + translated) reach this.")
    parser.add_argument("--translated-per-party", default=None,
                        help='Total translated rows to aim for per party, e.g. "ID=97000,ECR=63000,*=10000" '
                             "('*' = default for unlisted parties). Cached translations count toward the "
                             "target and are always kept; only the shortfall is planned as new work.")
    parser.add_argument("--target-langs", default=None)
    parser.add_argument("--second-provider", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        sys.exit("No models given (--models)")
    print(f"Models: {models} (second_provider={args.second_provider})")

    spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    config = TranslationConfig(
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

    base_name = args.cache_name.replace("/", "_")
    cache_path = args.output_dir / f"{base_name}.cache.json"
    output_path = args.output_dir / f"{base_name}.augmented.parquet"
    legacy_glm_cache_path = args.output_dir / LEGACY_GLM_CACHE
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(cache_path, legacy_glm_cache_path)
    if results:
        print(f"Total cached translations available: {len(results):,}")

    party_targets, default_target = parse_party_targets(args.translated_per_party)
    tasks = plan_tasks(df, target_languages, args.target_per_party, set(results),
                       party_targets, default_target)
    if args.limit is not None:
        tasks = tasks[: args.limit]
        print(f"Applied --limit: {len(tasks):,} tasks")

    run_translations(tasks, results, cache_path, config, models, args.max_workers)
    write_output(tasks, results, output_path)


if __name__ == "__main__":
    main()