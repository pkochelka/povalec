#!/usr/bin/env python3
"""Sample LLM speeches into a hand-labeling template for the OOD stance eval.

Speeches are the opinionated creative-writing track (no Likert column), so they
serve as the out-of-domain set the stance regressor is selected on. Items are
deduplicated and stratified across NLI-stance bins for balanced agree/disagree
coverage. The NLI proxy is written to a sidecar, never the labeling file, so it
cannot anchor the annotator.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import ALL_LANGS_STR, load_dataframe

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
SOURCE_MODELS = ["kimi-k2.7", "deepseek-v4-pro", "glm-5.2"]
PARAPHRASE_SUFFIXES = ["", "_negated"]
TARGET_SAMPLES = 160
STANCE_BIN_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0001]
SEED = 42

LABEL_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label.csv")
META_PATH = os.path.join(DATA_DIR, "stance_speeches_to_label_meta.csv")

ANSWER_PATTERN = re.compile(r"^answer_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
STANCE_PATTERN = re.compile(r"^stance_(?P<lang>[a-z]{2})_v(?P<variant>\d+)$")
ORIGINAL_PATTERN = re.compile(r"^original_text_(?P<lang>[a-z]{2})$")


def normalize_columns(columns):
    return [c.replace("_question", "").replace("_negated", "") for c in columns]


def melt_speeches(frame, model, paraphrase):
    frame.columns = normalize_columns(frame.columns)
    answers, stances, statements = {}, {}, {}
    for column in frame.columns:
        answer_match = ANSWER_PATTERN.match(column)
        if answer_match:
            answers[(answer_match["lang"], int(answer_match["variant"]))] = frame[column]
            continue
        stance_match = STANCE_PATTERN.match(column)
        if stance_match:
            key = (stance_match["lang"], int(stance_match["variant"]))
            stances[key] = pd.to_numeric(frame[column], errors="coerce")
            continue
        original_match = ORIGINAL_PATTERN.match(column)
        if original_match:
            statements[original_match["lang"]] = frame[column]

    records = []
    for (language, variant), answer_series in answers.items():
        block = pd.DataFrame({
            "answer_text": answer_series.astype("string").str.strip(),
            "statement_text": statements[language].astype("string").str.strip(),
            "nli_stance": stances.get((language, variant), np.nan),
            "statement": np.arange(len(answer_series)),
        })
        block["model"] = model
        block["language"] = language
        block["variant"] = variant
        block["paraphrase"] = paraphrase
        records.append(block)
    return pd.concat(records, ignore_index=True)


def load_speech_pool(models=None):
    models = SOURCE_MODELS if models is None else models
    frames = []
    for model in models:
        for suffix in PARAPHRASE_SUFFIXES:
            path = os.path.join(DATA_DIR, model, f"speeches_{ALL_LANGS_STR}{suffix}_scored.csv")
            if not os.path.exists(path):
                print(f"  [{model}{suffix}] no scored file at {path}, skipping.")
                continue
            paraphrase = suffix.lstrip("_") or "base"
            frames.append(melt_speeches(load_dataframe(path), model, paraphrase))
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.dropna(subset=["answer_text", "nli_stance"])
    pool = pool[pool["answer_text"].str.len() > 0]
    pool = pool.drop_duplicates(subset=["answer_text"]).reset_index(drop=True)
    print(f"Speech pool: {len(pool)} unique answers across {models}")
    return pool


def exclude_handlabeled(pool):
    if not os.path.exists(LABEL_PATH):
        return pool
    handlabeled = set(load_dataframe(LABEL_PATH)["answer_text"].astype("string").str.strip())
    return pool[~pool["answer_text"].isin(handlabeled)].reset_index(drop=True)


def stratified_sample(pool):
    pool = pool.copy()
    pool["bin"] = pd.cut(pool["nli_stance"], bins=STANCE_BIN_EDGES, right=False, labels=False)
    per_bin = -(-TARGET_SAMPLES // pool["bin"].nunique())
    rng = np.random.default_rng(SEED)

    chosen, leftovers = [], []
    for _, group in pool.groupby("bin"):
        shuffled = group.sample(frac=1.0, random_state=rng.integers(1 << 31))
        chosen.append(shuffled.iloc[:per_bin])
        leftovers.append(shuffled.iloc[per_bin:])

    sample = pd.concat(chosen, ignore_index=True)
    shortfall = TARGET_SAMPLES - len(sample)
    if shortfall > 0:
        pool_leftover = pd.concat(leftovers, ignore_index=True).sample(
            n=min(shortfall, sum(len(l) for l in leftovers)), random_state=SEED)
        sample = pd.concat([sample, pool_leftover], ignore_index=True)

    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "id", [f"sp{i:04d}" for i in range(len(sample))])
    print("Sampled per NLI-stance bin:\n", sample["bin"].value_counts().sort_index().to_dict())
    return sample


def water_fill(capacities, total):
    """Distribute `total` over keys, equal shares capped at each capacity (see
    preprocessing/build_collapsed_splits.py for the same routine over languages)."""
    alloc = dict.fromkeys(capacities, 0)
    active = [key for key, cap in capacities.items() if cap > 0]
    remaining = min(total, sum(capacities.values()))
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for key in sorted(active, key=lambda k: capacities[k] - alloc[k], reverse=True):
                if remaining == 0:
                    break
                alloc[key] += 1
                remaining -= 1
            break
        for key in list(active):
            give = min(share, capacities[key] - alloc[key])
            alloc[key] += give
            remaining -= give
            if alloc[key] >= capacities[key]:
                active.remove(key)
    return alloc


def stratified_sample_by_language(pool, languages, target_samples, seed=SEED):
    """Water-filled three-level stratification: an even quota per language,
    then within each language an even quota per NLI-stance bin, then within
    each (language, bin) cell an even quota per paraphrase (base/negated) --
    so a scarce cell (e.g. few neutral-agreement French speeches) just caps
    out and the shortfall spreads to the other bins, instead of skewing the
    whole sample toward whichever language/bin/paraphrase is most abundant.
    Prompt variant (v0-v4) is left unconstrained: crossing it too would mostly
    fragment the already-scarce neutral bins into near-empty cells."""
    pool = pool[pool["language"].isin(languages)].copy()
    pool["bin"] = pd.cut(pool["nli_stance"], bins=STANCE_BIN_EDGES, right=False, labels=False)
    rng = np.random.default_rng(seed)

    lang_capacities = {lang: int((pool["language"] == lang).sum()) for lang in languages}
    lang_quota = water_fill(lang_capacities, target_samples)

    chosen = []
    for lang, quota in lang_quota.items():
        lang_pool = pool[pool["language"] == lang]
        bin_quota = water_fill(lang_pool["bin"].value_counts().to_dict(), quota)
        for bin_id, n in bin_quota.items():
            if n == 0:
                continue
            cell_pool = lang_pool[lang_pool["bin"] == bin_id]
            paraphrase_quota = water_fill(cell_pool["paraphrase"].value_counts().to_dict(), n)
            for paraphrase, k in paraphrase_quota.items():
                if k > 0:
                    group = cell_pool[cell_pool["paraphrase"] == paraphrase]
                    chosen.append(group.sample(k, random_state=rng.integers(1 << 31)))

    sample = pd.concat(chosen, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample.insert(0, "id", [f"sp{i:04d}" for i in range(len(sample))])
    print(f"Sampled {len(sample)}/{target_samples} requested, per language x bin:\n",
          sample.groupby(["language", "bin"], observed=True).size().unstack(fill_value=0).to_string())
    print(f"\nper language x paraphrase:\n",
          sample.groupby(["language", "paraphrase"], observed=True).size().unstack(fill_value=0).to_string())
    return sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", default=None,
                        help="Comma-separated language codes to restrict to (default: all). "
                             "Switches to the per-language-balanced sampler and writes to a "
                             "language-tagged file instead of the default OOD label file.")
    parser.add_argument("--target_samples", default=None, type=int,
                        help=f"Total rows to sample (default: {TARGET_SAMPLES}).")
    parser.add_argument("--models", default=None,
                        help=f"Comma-separated source models (default: {SOURCE_MODELS}).")
    return parser.parse_args()


def main():
    args = parse_args()
    models = args.models.split(",") if args.models else None
    languages = args.languages.split(",") if args.languages else None
    target_samples = args.target_samples or TARGET_SAMPLES

    pool = load_speech_pool(models=models)
    if languages is None:
        sample = stratified_sample(pool)
        label_path, meta_path = LABEL_PATH, META_PATH
    else:
        pool = exclude_handlabeled(pool)
        sample = stratified_sample_by_language(pool, languages, target_samples)
        tag = "-".join(languages)
        label_path = LABEL_PATH.replace(".csv", f"_{tag}.csv")
        meta_path = META_PATH.replace(".csv", f"_{tag}.csv")

    labeling = sample[["id", "model", "language", "statement_text", "answer_text"]].copy()
    labeling["choice"] = ""
    labeling.to_csv(label_path, sep=";", encoding="utf-8-sig", index=False)

    meta = sample[["id", "model", "paraphrase", "language", "variant", "statement", "nli_stance"]]
    meta.to_csv(meta_path, sep=";", encoding="utf-8-sig", index=False)

    print(f"\nWrote {len(labeling)} items to label -> {label_path}")
    print("Fill the 'choice' column with Likert 1-5 (1=totally agree ... 5=totally disagree),")
    print("judging how much the speech agrees with its statement_text. Leave blank to skip.")
    print(f"Strata/NLI proxy kept separately -> {meta_path}")


if __name__ == "__main__":
    main()
