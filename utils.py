import json
import re

import pandas as pd


def save_checkpoint(path: str, results: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def load_dataframe(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")
