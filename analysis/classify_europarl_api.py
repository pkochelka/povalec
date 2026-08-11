#!/usr/bin/env python3
"""Zero-shot LLM baseline for the 6-class EU party classification task.

Prompts an OpenAI-compatible endpoint to assign each EuroParl speech to one of
the six EU groups (ECR and ID collapsed into a single ECR+ID group, matching
the collapsed/ split track), caches each response to disk so runs are
resumable, and prints a classification report comparable to the trained
mmBERT model.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from utils import call_api, extract_json, save_checkpoint
from analysis.europarl_classification import (
    PARTY_COLUMN,
    classification_report_text,
    load_split,
)

LABELS = ["ALDE", "ECR+ID", "GUE/NGL", "Greens/EFA", "PPE", "S&D"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LABEL_ALIASES = {
    "EPP": "PPE",
    "RENEW": "ALDE",
    "RENEW EUROPE": "ALDE",
    "GREENS": "Greens/EFA",
    "GREENS-EFA": "Greens/EFA",
    "GREENS/EFA": "Greens/EFA",
    "THE LEFT": "GUE/NGL",
    "LEFT": "GUE/NGL",
    "GUE-NGL": "GUE/NGL",
    "GUE": "GUE/NGL",
    "S AND D": "S&D",
    "SOCIALISTS AND DEMOCRATS": "S&D",
    "SD": "S&D",
    "ECR": "ECR+ID",
    "ID": "ECR+ID",
    "ECR-ID": "ECR+ID",
    "ECR/ID": "ECR+ID",
    "ECR AND ID": "ECR+ID",
    "ID GROUP": "ECR+ID",
    "IDENTITY AND DEMOCRACY": "ECR+ID",
    "ECR GROUP": "ECR+ID",
}

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "collapsed"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "classify_europarl_party.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "europarl_llm_results"

MAX_TEXT_CHARS = 4000
MAX_RETRIES = 5
REPORT_EVERY = 1000
CACHE_SAVE_INTERVAL_SECONDS = 60


@dataclass(frozen=True)
class ClassifierConfig:
    model: str
    prompt_template: str
    descriptions: dict
    max_tokens: int


def interleave_by_party(df, seed):
    """Round-robin the rows across PARTY_COLUMN so every prefix of the result --
    in particular each ~10k-row chunk served to the API -- is within one row of
    perfectly balanced across parties, however the job gets paused/resumed."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    party_rank = df.groupby(PARTY_COLUMN).cumcount()
    order = pd.DataFrame({"_rank": party_rank, PARTY_COLUMN: df[PARTY_COLUMN]}).sort_values(
        ["_rank", PARTY_COLUMN], kind="stable"
    ).index
    return df.loc[order].reset_index(drop=True)


def subsample(df, limit, seed):
    if limit is not None and limit < len(df):
        fraction = limit / len(df)
        parts = [
            group.sample(max(1, round(len(group) * fraction)), random_state=seed)
            for _, group in df.groupby(["language", PARTY_COLUMN])
        ]
        df = pd.concat(parts).reset_index(drop=True)
    return interleave_by_party(df, seed)


def build_prompt(config, text):
    descriptions = "\n".join(f"- {label}: {config.descriptions[label]}" for label in LABELS)
    return config.prompt_template.format(
        descriptions=descriptions,
        labels=", ".join(LABELS),
        text=text[:MAX_TEXT_CHARS],
    )


def normalize_label(raw):
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().strip('"').strip("'")
    if candidate in LABELS:
        return candidate
    upper = candidate.upper()
    for label in LABELS:
        if upper == label.upper():
            return label
    return LABEL_ALIASES.get(upper)


def classify_speech(text, config):
    prompt = build_prompt(config, text)
    last_content = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_api(
                prompt,
                config.model,
                max_tokens=config.max_tokens,
            )
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if content is None:
                raise ValueError(f"null content (finish_reason={choice.get('finish_reason')!r})")
            last_content = content
            parsed = extract_json(content)
            if not parsed or "choice" not in parsed:
                raise ValueError(f"could not extract choice from: {content!r}")
            return {
                "choice": normalize_label(parsed.get("choice")),
                "raw_choice": parsed.get("choice"),
                "reason": parsed.get("reason", ""),
            }
        except Exception as error:
            snippet = text[:60].replace("\n", " ")
            print(f"  retry {attempt}/{MAX_RETRIES}: {error} | text={snippet!r}", flush=True)
            time.sleep(min(2 ** attempt, 10))
    return {
        "choice": None,
        "raw_choice": last_content,
        "reason": f"FAILED: {last_content}" if last_content else "FAILED",
    }


def load_cache(cache_path):
    if not cache_path.exists():
        return {}
    with open(cache_path, encoding="utf-8") as f:
        return {int(index): prediction for index, prediction in json.load(f).items()}


def classify_pending(df, results, cache_path, config, max_workers):
    pending = [i for i in range(len(df)) if i not in results]
    print(f"Pending: {len(pending)} / {len(df)}")

    lock = threading.Lock()
    last_save = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(classify_speech, df.at[i, "text"], config): i
            for i in pending
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                results[index] = {"choice": None, "raw_choice": None, "reason": f"FAILED: {error}"}

            if len(results) % REPORT_EVERY == 0 or done == len(pending):
                with lock:
                    save_checkpoint(cache_path, results)
                    last_save = time.time()
                print(f"\n=== after {len(results)} total ({done} this run) ===")
                print(evaluation_report(results, df), flush=True)
            elif time.time() - last_save > CACHE_SAVE_INTERVAL_SECONDS:
                with lock:
                    save_checkpoint(cache_path, results)
                    last_save = time.time()

    save_checkpoint(cache_path, results)


def evaluation_report(results, df):
    pairs = [
        (LABEL_TO_ID[df.at[index, PARTY_COLUMN]], LABEL_TO_ID[prediction["choice"]])
        for index, prediction in results.items()
        if prediction.get("choice") in LABEL_TO_ID and df.at[index, PARTY_COLUMN] in LABEL_TO_ID
    ]
    if not pairs:
        return "  no parseable predictions yet."

    y_true = np.array([true for true, _ in pairs])
    y_pred = np.array([pred for _, pred in pairs])
    accuracy = float((y_true == y_pred).mean())
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    header = (
        f"  n={len(results)} parsed={len(pairs)} unparsable={len(results) - len(pairs)} "
        f"acc={accuracy:.4f} f1_macro={f1_macro:.4f}"
    )
    report = classification_report_text(y_true, y_pred, LABELS)
    return f"{header}\n{report}"


def write_predictions_csv(results, df, output_csv):
    rows = [
        {
            "row_index": index,
            "language": df.at[index, "language"],
            "true_party": df.at[index, PARTY_COLUMN],
            "predicted_party": results.get(index, {}).get("choice"),
            "raw_choice": results.get(index, {}).get("raw_choice"),
            "reason": results.get(index, {}).get("reason"),
        }
        for index in range(len(df))
    ]
    pd.DataFrame(rows).to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nWrote per-row predictions: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE, type=Path)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--max_workers", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--max_tokens", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    config = ClassifierConfig(
        model=args.model,
        prompt_template=spec["prompt"],
        descriptions=spec["label_descriptions"],
        max_tokens=args.max_tokens,
    )

    df = load_split(args.split, args.data_dir, keep_labels=LABELS)
    df = subsample(df, args.limit, args.seed)
    print(
        f"Loaded {args.split}: {len(df)} speeches across "
        f"{df['language'].nunique()} languages, {df[PARTY_COLUMN].nunique()} parties"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{args.model.replace('/', '_')}_{args.data_dir.name}_{args.split}"
    if args.limit is not None:
        base_name += f"_n{args.limit}"
    cache_path = args.output_dir / f"{base_name}.cache.json"
    output_csv = args.output_dir / f"{base_name}.csv"
    report_path = args.output_dir / f"{base_name}.report.txt"

    results = load_cache(cache_path)
    if results:
        print(f"Loaded cache: {len(results)} already classified ({cache_path})")

    classify_pending(df, results, cache_path, config, args.max_workers)
    write_predictions_csv(results, df, output_csv)

    report = evaluation_report(results, df)
    print(f"\n=== FINAL ===\n{report}")
    report_path.write_text(
        f"model={args.model} split={args.split} n={len(df)}\n{'-' * 40}\n{report}\n",
        encoding="utf-8",
    )
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
