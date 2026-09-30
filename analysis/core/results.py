"""Reading a model's results directory into scored, comparable frames.

Every loader here returns a long frame keyed by (statement_idx, language, paraphrase),
so the four measurement methods can be stacked and compared on the one thing that *is*
comparable across them — the ordering of the EP groups. Scales are not: VAA agreement
occupies a narrow band and classifier probability is dominated by the class prior.

Moved out of `plotting/plot_ep_group_rank_boxplots.py`, which owned all of this while
`rank_consistency_tables.py` imported fourteen of the names below from it.
"""
import re

import numpy as np
import pandas as pd

from utils import flip_likert, likert_to_stance
from analysis.core.positions import load_party_positions, positions_path
from analysis.core.labels import count
from analysis.core.paths import (
    CHOICE_COLUMN,
    FRAMING_FOR_VARIANT,
    FRAMING_ORIENTATION,
    JUDGE_COLUMNS,
    JUDGE_CSV,
    PARTY_PROB_COLUMN,
    PREDICTED_PARTY_COLUMN,
    REASONS_CLASSIFIED_CSV,
    RESPONSES_CSV,
    SLUG_TO_LABEL,
    SPEECHES_CLASSIFIED_CSV,
    SPEECHES_SCORED_CSV,
    STANCE_COLUMN,
)

NEUTRAL_LIKERT = 3

# What makes one "run": vary one of these and hold the rest, and you have a comparison.
CELL_KEYS = ["model", "method", "framing", "language", "paraphrase"]


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def find_csvs(model_dir, pattern):
    """{framing: path}, keeping the file that covers the most languages."""
    found = {}
    for path in sorted(model_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        framing = FRAMING_FOR_VARIANT[match.group("variant")]
        if framing not in found or len(match.group("langs")) > len(found[framing][1]):
            found[framing] = (path, match.group("langs"))
    return {framing: path for framing, (path, _) in found.items()}


# --------------------------------------------------------------------------- #
# stance frames  (statement_idx, language, paraphrase, stance)
# --------------------------------------------------------------------------- #

def likert_stances(path, framing):
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    frames = []
    for column in raw.columns:
        match = CHOICE_COLUMN.match(column)
        if not match:
            continue
        likert = pd.to_numeric(raw[column], errors="coerce").fillna(NEUTRAL_LIKERT)
        if framing == "negated":
            likert = flip_likert(likert)
        frames.append(pd.DataFrame({
            "statement_idx": np.arange(len(raw)),
            "language": match.group("language"),
            "paraphrase": int(match.group("paraphrase")),
            "stance": likert_to_stance(likert).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def speech_stances(path, framing):
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns
    stance_columns = [c for c in header if STANCE_COLUMN.match(c)]
    if not stance_columns:
        return pd.DataFrame()
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=stance_columns)
    orientation = FRAMING_ORIENTATION[framing]
    frames = []
    for column in stance_columns:
        match = STANCE_COLUMN.match(column)
        frames.append(pd.DataFrame({
            "statement_idx": np.arange(len(raw)),
            "language": match.group("language"),
            "paraphrase": int(match.group("paraphrase")),
            "stance": orientation * pd.to_numeric(raw[column], errors="coerce").to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def judge_stances(model_dir, track):
    """Both framings come out of one file, so this returns {framing: frame}."""
    path = model_dir / JUDGE_CSV[track]
    if not path.exists():
        return {}
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=JUDGE_COLUMNS)
    raw = raw[raw["paraphrase"].isin(FRAMING_ORIENTATION)]
    by_framing = {}
    for framing, group in raw.groupby("paraphrase"):
        by_framing[framing] = pd.DataFrame({
            "statement_idx": group["statement"].astype(int).to_numpy(),
            "language": group["language"].to_numpy(),
            "paraphrase": group["variant"].astype(int).to_numpy(),
            "stance": (FRAMING_ORIENTATION[framing]
                       * pd.to_numeric(group["llm_stance"], errors="coerce")).to_numpy(),
        })
    return by_framing


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def positions_frame(positions, collapse_ecr_id):
    party_df = load_party_positions(positions_path(positions), positions)
    if collapse_ecr_id:
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    return party_df[["ep_group", "statement_idx", "normalized_answer"]].dropna(subset=["ep_group"])


def vaa_scores(party_df, stances):
    """Same agreement as evaluate_euandi.agreement_by_ep_group -- 1 - |position -
    stance| / 2 -- but keeping the paraphrase axis instead of averaging it away."""
    if stances.empty:
        return pd.DataFrame()
    merged = party_df.merge(stances, on="statement_idx")
    merged["score"] = 1 - (merged["normalized_answer"] - merged["stance"]).abs() / 2
    return (merged.groupby(["ep_group", "language", "paraphrase"], as_index=False)["score"]
            .mean())


def slug_labels(df):
    labels = {}
    for column in df.columns:
        if PREDICTED_PARTY_COLUMN.match(column):
            for label in df[column].dropna().unique():
                labels[re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")] = label
    return labels


def classifier_scores(path):
    """Mean party probability per (language, paraphrase, group), over statements."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    labels = slug_labels(raw)
    records = []
    for column in raw.columns:
        match = PARTY_PROB_COLUMN.match(column)
        if not match:
            continue
        probability = pd.to_numeric(raw[column], errors="coerce")
        if probability.isna().all():
            continue
        slug = match.group("slug")
        records.append({
            "ep_group": labels.get(slug, SLUG_TO_LABEL.get(slug, slug)),
            "language": match.group("language"),
            "paraphrase": int(match.group("paraphrase")),
            "score": float(probability.mean()),
        })
    return pd.DataFrame.from_records(records, columns=["ep_group", "language", "paraphrase", "score"])


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

def vaa_method(pattern, stance_loader):
    def load(model_dir, party_df):
        frames = []
        for framing, path in find_csvs(model_dir, pattern).items():
            scores = vaa_scores(party_df, stance_loader(path, framing))
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


def judge_method(track):
    def load(model_dir, party_df):
        frames = []
        for framing, stances in judge_stances(model_dir, track).items():
            scores = vaa_scores(party_df, stances)
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


def classifier_method(pattern):
    def load(model_dir, party_df):
        frames = []
        for framing, path in find_csvs(model_dir, pattern).items():
            scores = classifier_scores(path)
            if not scores.empty:
                frames.append(scores.assign(framing=framing))
        return frames
    return load


METHODS = {
    "vaa-likert": dict(
        track="reasons", scoring="VAA · Likert choice",
        load=vaa_method(RESPONSES_CSV, likert_stances)),
    "vaa-likert-judge": dict(
        track="reasons", scoring="VAA · LLM judge",
        load=judge_method("reasons")),
    "vaa-speeches": dict(
        track="open-ended", scoring="VAA · cross-encoder",
        load=vaa_method(SPEECHES_SCORED_CSV, speech_stances)),
    "vaa-speeches-judge": dict(
        track="open-ended", scoring="VAA · LLM judge",
        load=judge_method("speeches")),
    "clf-reasons": dict(
        track="reasons", scoring="mmBERT classifier",
        load=classifier_method(REASONS_CLASSIFIED_CSV)),
    "clf-speeches": dict(
        track="open-ended", scoring="mmBERT classifier",
        load=classifier_method(SPEECHES_CLASSIFIED_CSV)),
}


def load_scores(model_dirs, methods, party_df):
    frames = []
    for model_dir in model_dirs:
        print(f"\nLoading: {model_dir.name}")
        for method in methods:
            spec = METHODS[method]
            found = spec["load"](model_dir, party_df)
            if not found:
                print(f"  [{method}] no input files, skipping.")
                continue
            scores = pd.concat(found, ignore_index=True)
            frames.append(scores.assign(model=model_dir.name, method=method,
                                        track=spec["track"], scoring=spec["scoring"]))
            print(f"  [{method}] {len(scores)} group scores over "
                  + ", ".join([count(scores["language"].nunique(), "language"),
                               count(scores["paraphrase"].nunique(), "paraphrase"),
                               count(scores["framing"].nunique(), "framing")]))
    if not frames:
        raise SystemExit("No scores could be loaded for any model.")
    return pd.concat(frames, ignore_index=True)
