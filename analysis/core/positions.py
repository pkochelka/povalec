"""Whose euandi answers stand for an EP group, and how to load them.

The positions basis is a project-wide choice, not `evaluate_euandi.py`'s: eleven modules
read it, and every figure and table in a run has to be built on the same one. It lives
here so nothing has to import a CLI script to find out what a group's positions are.

    ep-group     the europarty's own manifesto answers (country_iso == "eu"), one
                 position vector per group, nothing averaged across parties.
    national     the five countries' member parties in the same file, averaged.
    group-mean   every national party that ran in 2024, averaged per group.

**The basis is not recorded in the output filenames.** Change it and the existing
vaa*.csv files are left in place, so every downstream plot silently keeps reading the
old basis unless the run passes `--override`. See findings §6.
"""
import json
from pathlib import Path

import pandas as pd

from utils import EP_GROUP_BY_PARTY

PARTY_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_parties.jsonl"
GROUP_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_group_positions.jsonl"

POSITION_CHOICES = ["ep-group", "national", "group-mean"]
DEFAULT_POSITIONS = "ep-group"


def positions_path(positions: str) -> str:
    return GROUP_POSITIONS_PATH if positions == "group-mean" else PARTY_POSITIONS_PATH


def check_statement_order(df: pd.DataFrame, path: str) -> None:
    """Fail if `statement_idx` does not mean "row of statements.jsonl".

    It did not used to. The positions file arrived in EU&I codebook order while every
    results CSV is written in `statements.jsonl` order, and the two agree on only 11 of
    the 30 statements -- so `merge(..., on="statement_idx")` silently scored each model's
    answers against another statement's official positions, for 19 statements out of 30.
    `statement_collection/align_statement_order.py` reindexed the file; this is the guard
    that stops a re-downloaded or hand-edited positions file from reintroducing the bug
    without anyone noticing, since nothing about the mismatch makes a join fail.
    """
    statements_path = Path(path).parent / "statements.jsonl"
    if not statements_path.exists() or "statement" not in df.columns:
        return
    with open(statements_path, encoding="utf-8") as f:
        statements = [" ".join(json.loads(line)["statement"]["en"].split())
                      for line in f if line.strip()]

    pairs = df[["statement_idx", "statement"]].dropna().drop_duplicates()
    wrong = [(idx, text) for idx, text in pairs.itertuples(index=False)
             if not 0 <= idx < len(statements)
             or " ".join(str(text).split()) != statements[idx]]
    if wrong:
        idx, text = wrong[0]
        raise SystemExit(
            f"{path} is not aligned to statements.jsonl: {len(wrong)} statement(s) sit at "
            f"the wrong index, e.g. idx {idx} carries {' '.join(str(text).split())!r} but "
            f"row {idx} of statements.jsonl is "
            f"{statements[idx] if 0 <= idx < len(statements) else '(out of range)'!r}. "
            "Run: python statement_collection/align_statement_order.py")


def load_party_positions(path: str, positions: str = DEFAULT_POSITIONS) -> pd.DataFrame:
    """One row per (party, statement), with the party's EP group attached."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = {k: v for k, v in obj.items() if k != "responses"}
            for response in obj.get("responses", []):
                records.append({**meta, **response})

    df = pd.DataFrame(records)
    is_europarty = df["country_iso"] == "eu"
    df = df[~is_europarty if positions == "national" else is_europarty].copy()
    df["statement_idx"] = df["statement_idx"].astype(int)
    df["statement"] = df["statement"].str.strip()
    df["ep_group"] = df["short_name"].map(EP_GROUP_BY_PARTY)
    check_statement_order(df, path)
    return df
