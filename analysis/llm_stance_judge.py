#!/usr/bin/env python3
"""Label speeches with an LLM judge to distill stance into a cross-encoder.

Each (statement, speech) pair is scored on the same 1-5 Likert scale as the
reasons track, so likert_to_stance maps it to [-1, 1]. The pool already spans
the base, question and negated framings (and every paraphrase variant), and the
hand-labeled eval speeches are excluded to keep that set clean for model
selection. With no --limit the whole pool is labeled; with one, speeches are
sampled stratified across NLI-stance bins for balanced agree/disagree coverage.
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

from utils import call_api, extract_json, likert_to_stance, load_dataframe, save_checkpoint
from analysis.sample_speeches_for_labeling import load_speech_pool, STANCE_BIN_EDGES, LABEL_PATH

PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "stance_judge.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
MAX_TEXT_CHARS = 6000
SEED = 42


def output_paths(judge_model, source_models):
    tag = f"{judge_model}_on_{'+'.join(source_models)}"
    output_path = os.path.join(RESULTS_DIR, f"stance_speeches_llm_labeled_{tag}.csv")
    return output_path, output_path + ".ckpt.json"


def load_prompt():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return json.load(f)["prompt"]


def exclude_handlabeled(pool):
    if not os.path.exists(LABEL_PATH):
        return pool
    handlabeled = set(load_dataframe(LABEL_PATH)["answer_text"].astype("string").str.strip())
    return pool[~pool["answer_text"].isin(handlabeled)].reset_index(drop=True)


def stratified_sample(pool, n_samples):
    pool = pool.copy()
    pool["bin"] = pd.cut(pool["nli_stance"], bins=STANCE_BIN_EDGES, right=False, labels=False)
    per_bin = -(-n_samples // pool["bin"].nunique())
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, group in pool.groupby("bin"):
        chosen.append(group.sample(min(per_bin, len(group)), random_state=rng.integers(1 << 31)))
    sample = pd.concat(chosen, ignore_index=True)
    sample = sample.sample(min(n_samples, len(sample)), random_state=SEED).reset_index(drop=True)
    print("Judging per NLI-stance bin:\n", sample["bin"].value_counts().sort_index().to_dict())
    return sample


def interleave_by_bin(sample, seed):
    """Round-robin rows across NLI-stance bins so every prefix served to the
    judge -- in particular each ~10k-row chunk -- stays within one row of
    balanced agree/disagree coverage, however the run gets paused/resumed."""
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    bin_rank = sample.groupby("bin").cumcount()
    order = pd.DataFrame({"_rank": bin_rank, "bin": sample["bin"]}).sort_values(
        ["_rank", "bin"], kind="stable"
    ).index
    return sample.loc[order].reset_index(drop=True)


def balanced_sample(pool):
    pool = pool.copy()
    pool["bin"] = pd.cut(pool["nli_stance"], bins=STANCE_BIN_EDGES, right=False, labels=False)
    cap = int(pool["bin"].value_counts().median())
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, group in pool.groupby("bin"):
        chosen.append(group.sample(min(cap, len(group)), random_state=rng.integers(1 << 31)))
    sample = pd.concat(chosen, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Downsampling over-represented bins to the median ({cap}):\n",
          sample["bin"].value_counts().sort_index().to_dict())
    return sample


def judge_one(statement, text, model, prompt_template, second_provider, max_retries=4):
    prompt = prompt_template.format(statement=statement, text=text[:MAX_TEXT_CHARS])
    last_content = None
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, model, max_tokens=300, temperature=0.0, second_provider=second_provider)
            content = response["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError(f"null content (finish_reason={response['choices'][0].get('finish_reason')!r})")
            last_content = content
            parsed = extract_json(content)
            if parsed and "choice" in parsed and int(parsed["choice"]) in (1, 2, 3, 4, 5):
                return int(parsed["choice"])
            raise ValueError(f"no valid choice in: {content!r}")
        except Exception as error:
            print(f"  retry {attempt + 1}/{max_retries}: {str(error)[:120]}", flush=True)
            time.sleep(min(2 ** (attempt + 1), 10))
    print(f"  FAILED: {str(last_content)[:120]!r}", flush=True)
    return None


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return {}
    with open(checkpoint_path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", default="deepseek-v4-pro")
    parser.add_argument("--second_provider", default=True, action="store_true")
    parser.add_argument("--limit", default=40000, type=int,
                        help="Max speeches to label; default labels the whole pool.")
    parser.add_argument("--balanced", action="store_true",
                        help="Downsample over-represented NLI-stance bins to the median bin size "
                             "instead of capping the total (ignores --limit).")
    parser.add_argument("--reuse_labels", default=None,
                        help="Path to an existing labeled CSV; speeches whose answer_text is "
                             "already labeled there are reused instead of re-judged.")
    parser.add_argument("--max_workers", default=10, type=int)
    parser.add_argument("--source_models", default="deepseek-v4-pro,glm-5.2,kimi-k2.7",
                        help="Comma-separated models whose scored speech files feed the pool.")
    args = parser.parse_args()
    source_models = args.source_models.split(",")
    output_path, checkpoint_path = output_paths(args.judge_model, source_models)

    prompt_template = load_prompt()
    pool = exclude_handlabeled(load_speech_pool(models=source_models))
    if args.balanced:
        sample = balanced_sample(pool)
    elif args.limit is None:
        pool = pool.copy()
        pool["bin"] = pd.cut(pool["nli_stance"], bins=STANCE_BIN_EDGES, right=False, labels=False)
        sample = pool.reset_index(drop=True)
        print(f"Labeling the full pool: {len(sample)} speeches")
    else:
        sample = stratified_sample(pool, args.limit)
    sample = interleave_by_bin(sample, SEED)

    choices = load_checkpoint(checkpoint_path)
    if choices:
        print(f"Resuming: {len(choices)}/{len(sample)} already judged ({checkpoint_path})", flush=True)
    if args.reuse_labels and os.path.exists(args.reuse_labels):
        prior = load_dataframe(args.reuse_labels)
        prior_choice = dict(zip(prior["answer_text"],
                                pd.to_numeric(prior["llm_choice"], errors="coerce")))
        reused = 0
        for index, answer_text in sample["answer_text"].items():
            if index not in choices and pd.notna(prior_choice.get(answer_text)):
                choices[index] = int(prior_choice[answer_text])
                reused += 1
        print(f"Reused {reused} existing labels from {args.reuse_labels}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool_executor:
        futures = {
            pool_executor.submit(
                judge_one, row["statement_text"], row["answer_text"],
                args.judge_model, prompt_template, args.second_provider,
            ): index
            for index, row in sample.iterrows() if index not in choices
        }
        for done, future in enumerate(as_completed(futures), 1):
            choices[futures[future]] = future.result()
            if done % 50 == 0:
                print(f"  {done}/{len(futures)} judged", flush=True)
                save_checkpoint(checkpoint_path, choices)

    sample["llm_choice"] = sample.index.map(choices)
    labeled = sample.dropna(subset=["llm_choice"]).copy()
    labeled["llm_choice"] = labeled["llm_choice"].astype(int)
    labeled["llm_stance"] = likert_to_stance(labeled["llm_choice"]).astype("float32")

    columns = ["model", "paraphrase", "language", "variant", "statement",
               "statement_text", "answer_text", "nli_stance", "llm_choice", "llm_stance"]
    labeled[columns].to_csv(output_path, sep=";", encoding="utf-8-sig", index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nLabeled {len(labeled)}/{len(sample)} speeches -> {output_path}")
    print("LLM stance distribution:\n",
          labeled["llm_stance"].round(2).value_counts().sort_index().to_dict())
    agreement = labeled[["llm_stance", "nli_stance"]].corr().iloc[0, 1]
    print(f"Pearson(LLM judge, NLI): {agreement:.4f}")


if __name__ == "__main__":
    main()
