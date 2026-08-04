#!/usr/bin/env python3
"""Ask a frontier LLM how strongly each survey reason agrees with its statement.

The judge (default kimi-k2.6) rates each reason text on the same 1-5 Likert
scale the generating model used for its own choice, so likert_to_stance maps
both onto [-1, 1] and the judge stance can be correlated against the actual
choice. Samples are stratified so every language gets at least min_per_lang
reasons, balanced across the five choice values and spread over many statements.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import ALL_LANGS_STR, VARIANTS, call_api, extract_json, likert_to_stance, save_checkpoint

PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "stance_judge.json")
LIKERT_CHOICES = (1, 2, 3, 4, 5)
MAX_REASON_CHARS = 6000
SEED = 42


def load_prompt():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return json.load(f)["prompt"]


def reason_variant_indices(df, lang_variant):
    prefix = f"reason_{lang_variant}_v"
    return sorted({
        int(column[len(prefix):])
        for column in df.columns
        if column.startswith(prefix) and column[len(prefix):].isdigit()
    })


def collect_observations(df, languages, variant):
    records = []
    for language in languages:
        lang_variant = f"{language}{variant}"
        statement_column = f"original_text_{lang_variant}"
        if statement_column not in df.columns:
            continue
        for variant_index in reason_variant_indices(df, lang_variant):
            for row_index, row in df.iterrows():
                statement = row[statement_column]
                reason = row.get(f"reason_{lang_variant}_v{variant_index}")
                choice = pd.to_numeric(row.get(f"choice_{lang_variant}_v{variant_index}"), errors="coerce")
                if not isinstance(statement, str) or not statement.strip():
                    continue
                if not isinstance(reason, str) or not reason.strip():
                    continue
                if reason.startswith("REFUSED") or reason == "FAILED":
                    continue
                if pd.isna(choice) or int(choice) not in LIKERT_CHOICES:
                    continue
                records.append({
                    "language": language,
                    "lang_variant": lang_variant,
                    "statement_idx": row_index,
                    "variant_idx": variant_index,
                    "statement": statement.strip(),
                    "reason": reason.strip(),
                    "choice": int(choice),
                })
    return pd.DataFrame.from_records(records)


def allocate_evenly(available, quota):
    allocation = {choice: 0 for choice in available}
    while quota > 0:
        open_choices = [choice for choice in available if allocation[choice] < available[choice]]
        if not open_choices:
            break
        share = max(1, quota // len(open_choices))
        for choice in open_choices:
            take = min(share, available[choice] - allocation[choice], quota)
            allocation[choice] += take
            quota -= take
            if quota == 0:
                break
    return allocation


def diversified_sample(rows, n, rng):
    if len(rows) <= n:
        return rows
    shuffled = rows.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    one_per_statement = shuffled.drop_duplicates(subset="statement_idx", keep="first")
    if len(one_per_statement) >= n:
        return one_per_statement.head(n)
    leftover = shuffled.drop(one_per_statement.index)
    return pd.concat([one_per_statement, leftover.head(n - len(one_per_statement))])


def stratified_sample(observations, target_total, min_per_lang, seed):
    languages = sorted(observations["language"].unique())
    per_lang = max(min_per_lang, -(-target_total // len(languages)))
    rng = np.random.default_rng(seed)
    chosen = []
    for language in languages:
        lang_rows = observations[observations["language"] == language]
        available = lang_rows["choice"].value_counts().to_dict()
        allocation = allocate_evenly(available, min(per_lang, len(lang_rows)))
        for choice, count in allocation.items():
            if count:
                chosen.append(diversified_sample(lang_rows[lang_rows["choice"] == choice], count, rng))
    sample = pd.concat(chosen, ignore_index=True)
    print(f"Sampled {len(sample)} reasons across {len(languages)} languages "
          f"(target {per_lang}/language).")
    print("Per-choice totals:", sample["choice"].value_counts().sort_index().to_dict())
    print("Min per language:", int(sample["language"].value_counts().min()))
    return sample


def judge_one(statement, reason, model, prompt_template, second_provider, max_retries=4):
    prompt = prompt_template.format(statement=statement, text=reason[:MAX_REASON_CHARS])
    last_content = None
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, model, max_tokens=300, temperature=0.0, second_provider=second_provider)
            content = response["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError(f"null content (finish_reason={response['choices'][0].get('finish_reason')!r})")
            last_content = content
            parsed = extract_json(content)
            if parsed and "choice" in parsed and int(parsed["choice"]) in LIKERT_CHOICES:
                return int(parsed["choice"])
            raise ValueError(f"no valid choice in: {content!r}")
        except Exception as error:
            print(f"  retry {attempt + 1}/{max_retries}: {str(error)[:120]}", flush=True)
            time.sleep(min(2 ** (attempt + 1), 10))
    print(f"  FAILED: {str(last_content)[:120]!r}", flush=True)
    return None


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def run_judge(sample, model, prompt_template, second_provider, max_workers, checkpoint_path):
    choices = load_checkpoint(checkpoint_path)
    if choices:
        print(f"Resuming: {len(choices)}/{len(sample)} already judged", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_one, row["statement"], row["reason"],
                            model, prompt_template, second_provider): index
            for index, row in sample.iterrows() if index not in choices
        }
        for done, future in enumerate(as_completed(futures), 1):
            choices[futures[future]] = future.result()
            if done % 50 == 0:
                print(f"  {done}/{len(futures)} judged", flush=True)
                save_checkpoint(checkpoint_path, choices)
    return choices


def build_paired(sample, choices):
    paired = sample.copy()
    paired["judge_choice"] = paired.index.map(choices)
    paired = paired.dropna(subset=["judge_choice"]).copy()
    paired["judge_choice"] = paired["judge_choice"].astype(int)
    paired["choice_stance"] = likert_to_stance(paired["choice"]).astype("float32")
    paired["judge_stance"] = likert_to_stance(paired["judge_choice"]).astype("float32")
    return paired


def correlation_row(label, group):
    valid = group.dropna(subset=["choice_stance", "judge_stance"])
    if len(valid) < 2:
        return {"group": label, "n": len(valid), "pearson_r": np.nan,
                "spearman_rho": np.nan, "mae": np.nan, "exact_match": np.nan}
    return {
        "group": label,
        "n": len(valid),
        "pearson_r": valid["choice_stance"].corr(valid["judge_stance"], method="pearson"),
        "spearman_rho": valid["choice_stance"].corr(valid["judge_stance"], method="spearman"),
        "mae": (valid["choice_stance"] - valid["judge_stance"]).abs().mean(),
        "exact_match": (valid["choice"] == valid["judge_choice"]).mean(),
    }


def compute_correlations(paired):
    rows = [correlation_row("ALL", paired)]
    for language, group in paired.groupby("language", sort=True):
        rows.append(correlation_row(language, group))
    return pd.DataFrame(rows)


def highest_discrepancies(paired, top_n):
    valid = paired.dropna(subset=["choice_stance", "judge_stance"]).copy()
    valid["signed_gap"] = valid["choice_stance"] - valid["judge_stance"]
    valid["discrepancy"] = valid["signed_gap"].abs()
    return valid.sort_values("discrepancy", ascending=False).head(top_n)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge how strongly each survey reason agrees with its statement using a "
                    "frontier LLM, and correlate the judged stance with the actual choice.",
    )
    parser.add_argument("--llm", default="deepseek-v4-pro",
                        help="Generating model whose reason/choice outputs are evaluated.")
    parser.add_argument("--judge_model", default="kimi-k2.6")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--input", default=None,
                        help="Optional explicit path to a survey_processor_concurrent CSV.")
    parser.add_argument("--variant", default="", choices=VARIANTS)
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--target_total", default=3000, type=int)
    parser.add_argument("--min_per_lang", default=100, type=int)
    parser.add_argument("--max_workers", default=4, type=int)
    parser.add_argument("--second_provider", action="store_true")
    parser.add_argument("--top_discrepancies", default=100, type=int)
    return parser.parse_args()


def resolve_input_path(args, languages):
    if args.input:
        return args.input
    return f"./data/{args.dataset}_results/{args.llm}/{','.join(languages)}{args.variant}.csv"


def main():
    args = parse_args()
    languages = args.languages.split(",")
    input_path = resolve_input_path(args, languages)
    stem, ext = os.path.splitext(input_path)
    output_stem = f"{stem}_reason_llmjudge_{args.judge_model}"
    paired_path = f"{output_stem}_paired{ext}"
    correlations_path = f"{output_stem}_correlations{ext}"
    discrepancies_path = f"{output_stem}_top_discrepancies{ext}"
    checkpoint_path = f"{output_stem}.ckpt.json"

    df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    observations = collect_observations(df, languages, args.variant)
    if observations.empty:
        raise SystemExit(f"No usable reasons found in {input_path}")

    sample = stratified_sample(observations, args.target_total, args.min_per_lang, SEED)
    prompt_template = load_prompt()
    choices = run_judge(sample, args.judge_model, prompt_template,
                        args.second_provider, args.max_workers, checkpoint_path)

    paired = build_paired(sample, choices)
    paired.to_csv(paired_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nJudged {len(paired)}/{len(sample)} reasons -> {paired_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    correlations = compute_correlations(paired)
    correlations.to_csv(correlations_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {correlations_path}")
    print(correlations.to_string(index=False))

    if args.top_discrepancies > 0:
        top = highest_discrepancies(paired, args.top_discrepancies)
        top.to_csv(discrepancies_path, sep=";", index=False, encoding="utf-8-sig")
        print(f"Wrote {discrepancies_path}")


if __name__ == "__main__":
    main()
