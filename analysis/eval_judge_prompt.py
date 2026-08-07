#!/usr/bin/env python3
"""Score the hand-labeled speeches with a candidate judge prompt and report how
well it recovers the human's 5-point nuance -- before committing API budget to a
full re-judge.

The production judge collapsed to 1/3/5 (columns l2/l4 empty in the confusion
matrix). This harness re-scores just the ~150 annotated speeches with whatever
prompt is at --prompt, then prints exact / within-1 / quadratic-kappa agreement,
the 5x5 confusion, and -- the whole point -- how often the middle bins (2/4) get
used vs. how often the human used them.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import stance_to_likert, likert_to_stance
from analysis.llm_stance_judge import judge_one
from analysis.validate_stance_judge import read_csv_any, LABELED, META, DATA_DIR

DEFAULT_PROMPT = os.path.join(PROJECT_ROOT, "prompts", "stance_judge.json")
# Speeches whose text is embedded in the prompt as worked examples -- scoring them
# would be contaminated, so they are held out of the eval.
ANCHOR_IDS = {"sp0040", "sp0121"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--judge_model", default="deepseek-v4-pro")
    parser.add_argument("--labeled", default=LABELED)
    parser.add_argument("--meta", default=META)
    parser.add_argument("--max_workers", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Score only the first N (debug).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show the plan and the first fully-formatted prompt, make no API calls.")
    args = parser.parse_args()

    with open(args.prompt, encoding="utf-8") as f:
        prompt_template = json.load(f)["prompt"]

    labeled = read_csv_any(args.labeled)
    labeled["choice"] = pd.to_numeric(labeled["choice"], errors="coerce")
    df = labeled[~labeled["id"].isin(ANCHOR_IDS)].dropna(subset=["choice"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    df["human_choice"] = stance_to_likert(df["choice"]).round().astype(int)

    print(f"Prompt   : {args.prompt}")
    print(f"Judge    : {args.judge_model}")
    print(f"Speeches : {len(df)} annotated rows")
    if args.dry_run:
        row = df.iloc[0]
        print("\n--- first formatted prompt ---\n")
        print(prompt_template.format(statement=row["statement_text"], text=row["answer_text"][:6000]))
        print(f"\n[dry run] would make up to {len(df)} calls; no API calls issued.")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        preds = list(ex.map(
            lambda r: judge_one(r["statement_text"], r["answer_text"],
                                args.judge_model, prompt_template),
            [r for _, r in df.iterrows()]))
    df["llm_choice"] = preds

    ok = df.dropna(subset=["llm_choice"]).copy()
    ok["llm_choice"] = ok["llm_choice"].astype(int)
    failed = len(df) - len(ok)
    print(f"\nScored {len(ok)}/{len(df)} ({failed} failed).")

    human, llm = ok["human_choice"].to_numpy(), ok["llm_choice"].to_numpy()
    hs, ls = likert_to_stance(human), likert_to_stance(llm)
    print(f"  Exact choice agreement  : {np.mean(human == llm):.3f}")
    print(f"  Within-1 agreement       : {np.mean(np.abs(human - llm) <= 1):.3f}")
    print(f"  Quadratic-weighted kappa : {cohen_kappa_score(human, llm, weights='quadratic', labels=[1,2,3,4,5]):.3f}")
    print(f"  Pearson r (stance)       : {np.corrcoef(hs, ls)[0,1]:.3f}")
    print(f"  MAE (stance)             : {np.mean(np.abs(hs - ls)):.3f}")

    print("\nMiddle-bin usage (the point of this change):")
    print(f"  human used 2 or 4 : {int(((human==2)|(human==4)).sum())}/{len(ok)}")
    print(f"  llm   used 2 or 4 : {int(((llm==2)|(llm==4)).sum())}/{len(ok)}")

    print("\n5x5 confusion (rows=human, cols=llm; 1=totally agree ... 5=totally disagree):")
    cm = confusion_matrix(human, llm, labels=[1, 2, 3, 4, 5])
    print(pd.DataFrame(cm, index=[f"h{i}" for i in range(1, 6)],
                       columns=[f"l{i}" for i in range(1, 6)]).to_string())

    out = os.path.join(DATA_DIR, "eval_judge_prompt_preds.csv")
    ok["abs_diff"] = np.abs(hs - ls)
    ok.sort_values("abs_diff", ascending=False)[
        ["id", "language", "abs_diff", "human_choice", "llm_choice", "statement_text", "answer_text"]
    ].to_csv(out, sep=";", encoding="utf-8-sig", index=False)
    print(f"\nPer-item predictions (largest diff first) -> {out}")


if __name__ == "__main__":
    main()
