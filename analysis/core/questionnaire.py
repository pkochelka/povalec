"""Reading the EU&I questionnaire's per-statement axis coding.

Every statement is coded -1/0/+1 on each of the seven axes. The figures that slice
results by topic need those codings lined up with the rows of a results CSV, which is a
join, not a zip -- see `load_axis_directions`.

Rescued from `evaluate_cronbach.py` when the Cronbach scripts were deleted: the alpha
computation went, but this is questionnaire data access that the topic figures still
need, and it belongs in the data layer rather than in an evaluation script.
"""
import json

import pandas as pd

from utils import AXES


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def load_axis_directions(path: str, statements: pd.Series) -> pd.DataFrame:
    """Axis directions in the row order of `statements`, joined on statement text.

    The questionnaire is stored in codebook order (s1..s36) but the results CSVs
    follow statements.jsonl, which is shuffled relative to it -- only 11 of 30 rows
    line up. Joining positionally silently attaches the wrong directions to 19
    statements, so the join is on original_text_en.
    """
    with open(path, encoding="utf-8") as f:
        coding = {normalize(d["statement"]): {axis: d[axis] for axis in AXES}
                  for d in (json.loads(line) for line in f if line.strip())}

    keys = [normalize(s) for s in statements]
    missing = [k for k in keys if k not in coding]
    if missing:
        raise SystemExit(f"{len(missing)} statement(s) in the results CSV have no axis coding "
                         f"in {path}; first: {missing[0]!r}")
    return pd.DataFrame([coding[k] for k in keys], columns=AXES)


def axis_directions_in_admin_order(questionnaire_path: str, dataset: str, n_rows: int) -> pd.DataFrame:
    """Axis directions in the order the statements were administered.

    For result files that carry no English statement column (the *_scored speech
    CSVs only have per-language texts), the order is recovered from
    statements.jsonl, which is the order every result CSV is written in.
    """
    statements_path = f"data/{dataset}_data/statements.jsonl"
    with open(statements_path, encoding="utf-8") as f:
        administered = [json.loads(line)["statement"]["en"] for line in f if line.strip()]
    if len(administered) != n_rows:
        raise SystemExit(f"{statements_path} has {len(administered)} statements but the results "
                         f"file has {n_rows} rows; cannot recover the administration order.")
    return load_axis_directions(questionnaire_path, pd.Series(administered))
