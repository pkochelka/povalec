#!/usr/bin/env python3
"""Agreement of the hand annotations vs. the LLM judge vs. the NLI scorer.

Three stance estimates on the same speeches:
  * human   -- the hand-filled `choice`, already on stance [-1, 1] in 5 bins
               (-1,-0.5,0,0.5,1; +1 = totally agree) in the labeled CSV
  * llm     -- llm_stance_judge's `llm_choice` (Likert 1-5), matched by answer_text
  * nli     -- the NLI proxy `nli_stance` carried in the sampling meta sidecar

Everything is put on stance in [-1, 1] (+1 = totally agree); the human bins map to
a 1-5 choice for the Likert-only metrics via stance_to_likert. We report
pairwise Pearson/Spearman + MAE on that scale; for the two Likert raters (human,
llm) also exact / within-1 agreement and quadratic-weighted Cohen's kappa; and a
3-class direction (agree / neutral / disagree at +/-0.2) confusion + kappa across
all three. Human is treated as ground truth throughout.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import likert_to_stance, stance_to_likert

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
LABELED = os.path.join(DATA_DIR, "stance_speeches_labeled_en-cz-sk-fr.csv")
META = os.path.join(DATA_DIR, "stance_speeches_to_label_meta_en-cz-sk-fr.csv")
JUDGE = os.path.join(
    DATA_DIR, "stance_speeches_llm_labeled_deepseek-v4-pro_on_deepseek-v4-pro+glm-5.2+kimi-k2.7.csv")
NEUTRAL_BAND = 0.2


def read_csv_any(path):
    """Sniff ; vs , -- the labeled file comes back comma-separated from Excel,
    the meta/judge files are the pipeline's native ;-CSV."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") >= header.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def direction(stance):
    return np.where(stance >= NEUTRAL_BAND, "agree",
                    np.where(stance <= -NEUTRAL_BAND, "disagree", "neutral"))


def pair_report(name, a, b, a_choice=None, b_choice=None):
    print(f"\n=== {name}  (n={len(a)}) ===")
    print(f"  Pearson r  : {pearsonr(a, b)[0]:.3f}")
    print(f"  Spearman r : {spearmanr(a, b)[0]:.3f}")
    print(f"  MAE (stance): {np.mean(np.abs(a - b)):.3f}")
    if a_choice is not None and b_choice is not None:
        exact = np.mean(a_choice == b_choice)
        within1 = np.mean(np.abs(a_choice - b_choice) <= 1)
        kappa = cohen_kappa_score(a_choice, b_choice, weights="quadratic",
                                  labels=[1, 2, 3, 4, 5])
        print(f"  Exact choice agreement : {exact:.3f}")
        print(f"  Within-1 agreement     : {within1:.3f}")
        print(f"  Quadratic-weighted kappa: {kappa:.3f}")
    labels = ["disagree", "neutral", "agree"]
    da, db = direction(a), direction(b)
    dir_kappa = cohen_kappa_score(da, db, labels=labels)
    print(f"  3-class direction agreement: {np.mean(da == db):.3f}  (kappa {dir_kappa:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", default=LABELED)
    parser.add_argument("--meta", default=META)
    parser.add_argument("--judge", default=JUDGE)
    args = parser.parse_args()

    labeled = read_csv_any(args.labeled)
    meta = read_csv_any(args.meta)
    judge = read_csv_any(args.judge)

    labeled["choice"] = pd.to_numeric(labeled["choice"], errors="coerce")
    df = labeled.merge(meta[["id", "statement", "nli_stance"]], on="id", how="left")

    judge = judge.copy()
    judge["answer_text"] = judge["answer_text"].astype("string").str.strip()
    judge["llm_choice"] = pd.to_numeric(judge["llm_choice"], errors="coerce")
    judge_map = judge.drop_duplicates("answer_text").set_index("answer_text")["llm_choice"]
    df["answer_text"] = df["answer_text"].astype("string").str.strip()
    df["llm_choice"] = df["answer_text"].map(judge_map)

    print(f"Loaded {len(labeled)} labeled rows.")
    print(f"  annotated (choice filled) : {df['choice'].notna().sum()}")
    print(f"  matched to judge label    : {df['llm_choice'].notna().sum()}")
    print(f"  have NLI proxy            : {df['nli_stance'].notna().sum()}")

    df["human_stance"] = df["choice"]
    df["human_choice"] = stance_to_likert(df["choice"]).round().astype("Int64")
    df["llm_stance"] = likert_to_stance(df["llm_choice"])
    df["nli_stance"] = pd.to_numeric(df["nli_stance"], errors="coerce")

    hl = df.dropna(subset=["human_stance", "llm_stance"])
    pair_report("Human vs LLM judge", hl["human_stance"].to_numpy(), hl["llm_stance"].to_numpy(),
                hl["human_choice"].astype(int).to_numpy(), hl["llm_choice"].astype(int).to_numpy())

    hn = df.dropna(subset=["human_stance", "nli_stance"])
    pair_report("Human vs NLI", hn["human_stance"].to_numpy(), hn["nli_stance"].to_numpy())

    ln = df.dropna(subset=["llm_stance", "nli_stance"])
    pair_report("LLM judge vs NLI", ln["llm_stance"].to_numpy(), ln["nli_stance"].to_numpy())

    print("\n=== Human vs LLM judge: 5x5 choice confusion (rows=human, cols=llm) ===")
    print("(choice 1=totally agree ... 5=totally disagree)")
    cm = confusion_matrix(hl["human_choice"].astype(int), hl["llm_choice"].astype(int), labels=[1, 2, 3, 4, 5])
    print(pd.DataFrame(cm, index=[f"h{i}" for i in range(1, 6)],
                       columns=[f"l{i}" for i in range(1, 6)]).to_string())

    print("\n=== Per-language Human-vs-LLM within-1 agreement ===")
    hl = hl.assign(within1=(hl["human_choice"].astype(int) - hl["llm_choice"].astype(int)).abs() <= 1)
    print(hl.groupby("language")["within1"].agg(["mean", "size"]).round(3).to_string())

    diffs = hl.assign(abs_diff=(hl["human_stance"] - hl["llm_stance"]).abs())
    diffs = diffs.sort_values("abs_diff", ascending=False)
    out_cols = ["id", "language", "statement", "abs_diff", "human_choice", "llm_choice",
                "human_stance", "llm_stance", "statement_text", "answer_text"]
    out_path = os.path.join(DATA_DIR, "stance_judge_human_disagreements.csv")
    diffs[out_cols].to_csv(out_path, sep=";", encoding="utf-8-sig", index=False)

    print(f"\n=== Human vs LLM absolute stance diffs (largest first) -> {out_path} ===")
    preview = diffs[diffs["abs_diff"] > 0].copy()
    preview["statement_text"] = preview["statement_text"].str.slice(0, 60)
    preview["answer_text"] = preview["answer_text"].str.slice(0, 80)
    print(preview[["id", "language", "abs_diff", "human_choice", "llm_choice",
                   "statement_text", "answer_text"]].head(30).to_string(index=False))
    print(f"\n{len(preview)} of {len(hl)} pairs disagree at all; "
          f"{int((preview['abs_diff'] >= 1.0).sum())} disagree by >= 1.0 stance.")


if __name__ == "__main__":
    main()
