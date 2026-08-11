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

import pandas as pd

from utils import EP_GROUP_BY_PARTY

PARTY_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_parties.jsonl"
GROUP_POSITIONS_PATH = "data/euandi_2024_data/euandi_2024_group_positions.jsonl"

POSITION_CHOICES = ["ep-group", "national", "group-mean"]
DEFAULT_POSITIONS = "ep-group"


def positions_path(positions: str) -> str:
    return GROUP_POSITIONS_PATH if positions == "group-mean" else PARTY_POSITIONS_PATH


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
    return df
