#!/usr/bin/env python3
"""Sample LLM speeches into a hand-labeling template for the OOD stance eval.

Speeches are the opinionated creative-writing track (no Likert column), so they
serve as the out-of-domain set the stance regressor is selected on. Items are
deduplicated and stratified across NLI-stance bins for balanced agree/disagree
coverage. The NLI proxy is written to a sidecar, never the labeling file, so it
cannot anchor the annotator.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import ALL_LANGS_STR, load_dataframe

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
SOURCE_MODELS = ["glm-5"]#["kimi-k2.6", "deepseek-v4-pro"]
PARAPHRASE_SUFFIXES = ["", "_question", "_negated"]
TARGET_SAMPLES = 320
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


def load_speech_pool():
    frames = []
    for model in SOURCE_MODELS:
        for suffix in PARAPHRASE_SUFFIXES:
            path = os.path.join(DATA_DIR, model, f"speeches_{ALL_LANGS_STR}{suffix}_scored.csv")
            paraphrase = suffix.lstrip("_") or "base"
            frames.append(melt_speeches(load_dataframe(path), model, paraphrase))
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.dropna(subset=["answer_text", "nli_stance"])
    pool = pool[pool["answer_text"].str.len() > 0]
    pool = pool.drop_duplicates(subset=["answer_text"]).reset_index(drop=True)
    print(f"Speech pool: {len(pool)} unique answers across {SOURCE_MODELS}")
    return pool


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


def main():
    pool = load_speech_pool()
    sample = stratified_sample(pool)

    labeling = sample[["id", "model", "language", "statement_text", "answer_text"]].copy()
    labeling["choice"] = ""
    labeling.to_csv(LABEL_PATH, sep=";", encoding="utf-8-sig", index=False)

    meta = sample[["id", "model", "paraphrase", "language", "variant", "statement", "nli_stance"]]
    meta.to_csv(META_PATH, sep=";", encoding="utf-8-sig", index=False)

    print(f"\nWrote {len(labeling)} items to label -> {LABEL_PATH}")
    print("Fill the 'choice' column with Likert 1-5 (1=totally agree ... 5=totally disagree),")
    print("judging how much the speech agrees with its statement_text. Leave blank to skip.")
    print(f"Strata/NLI proxy kept separately -> {META_PATH}")


if __name__ == "__main__":
    main()
