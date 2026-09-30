"""Add LLM-written opinion texts to the annotation mix, stratified by stance.

Reads  data/euandi_2024_results/<model>/speeches_<langs>_scored.csv           one row per
       data/euandi_2024_results/<model>/speeches_<langs>_negated_scored.csv   EU&I statement
                                      (negated: the statement reworded to its opposite);
                                      answer_<lang>[_negated]_v<k> is the model's text for
                                      prompt variant k (v0 essay paragraph, v1 opinion report,
                                      v2 short opinion, v3 take a side, v4 EP speech),
                                      stance_... its classifier stance in [-1, 1] toward the
                                      statement the model was given
       data/annotation_mix.csv, data/annotation_mix_key.csv   the speech mix from
                                      build_annotation_mix.py
Writes the same two files. Any LLM items from an earlier run are dropped first, the sheet
       is reshuffled, and every item (speech or LLM) is renumbered am000.. in shuffled
       order, so neither position nor id gives the source away; the key is sorted by id. The key gains source/model/negated/statement_idx/statement/variant/
       stance/stance_bin ("EU Party" is empty for LLM rows; statement and stance refer to the
       negated statement on negated rows).

--total texts (default 204, as many as the speech items) are spread over the (model,
language, negated) cells (13 models x 3 languages x 2 = 78): every cell gets the floor of
the average, and the remainder goes one text at a time to the cell whose language, then
(language, negated), then model, then (model, language) has fewest extra texts so far, so
the languages come out exactly even and the models within one text of each other. A
cell's texts are split evenly over the stance bins against / neutral / for. Model
stances are heavily skewed toward "for", so when a model has too few texts in a bin the
shortfall is refilled from its other bins, scarcest-overall bin first. Within a bin the
next text is the one whose prompt variant is least used (within the model, then overall),
then whose statement is least used, so variants and statements come out as even as the
bins allow. A (model, statement) pair is used at most once, so no model contributes the
same statement twice, whether in two languages or in both framings.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.build_annotation_mix import ANSWER_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "euandi_2024_results"

# results-file code -> annotation-mix code
LANGUAGES = {"en": "en", "cz": "cs", "sk": "sk"}
BINS = ["against", "neutral", "for"]
BIN_EDGES = [-1.01, -1 / 3, 1 / 3, 1.01]


def load_pool() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        for negated in (False, True):
            files = [f for f in model_dir.glob("speeches_*_scored.csv")
                     if ("_negated_" in f.name) == negated]
            if len(files) != 1:
                continue
            d = pd.read_csv(files[0], sep=";")
            suffix = "_negated" if negated else ""
            for code, lang in LANGUAGES.items():
                for col in d.columns:
                    m = re.fullmatch(rf"answer_{code}{suffix}_v(\d+)", col)
                    if not m:
                        continue
                    v = int(m.group(1))
                    rows.append(pd.DataFrame({
                        "model": model_dir.name, "language": lang, "negated": negated,
                        "variant": v, "statement_idx": d.index,
                        "statement": d[f"original_text_{code}{suffix}"],
                        "text": d[col], "stance": d[f"stance_{code}{suffix}_v{v}"],
                    }))
    pool = pd.concat(rows, ignore_index=True).dropna(subset=["text", "stance"])
    pool["stance_bin"] = pd.cut(pool["stance"], BIN_EDGES, labels=BINS).astype(str)
    return pool


def cell_quotas(cell: pd.DataFrame, per_cell: int, bin_order: list[str]) -> dict[str, int]:
    """Even split over bins, capped by availability; shortfall goes to the scarcest bins."""
    avail = cell["stance_bin"].value_counts().reindex(BINS, fill_value=0).to_dict()
    quota = {b: 0 for b in BINS}
    for _ in range(per_cell):
        # next slot goes to the bin furthest below an even share that still has texts
        open_bins = [b for b in bin_order if quota[b] < avail[b]]
        if not open_bins:
            break
        quota[min(open_bins, key=lambda b: quota[b])] += 1
    return quota


def cell_sizes(cells: list[tuple], total: int) -> dict[tuple, int]:
    """Texts per (model, language, negated) cell, summing to `total` (see docstring)."""
    base, extra = divmod(total, len(cells))
    sizes = dict.fromkeys(cells, base)
    got = Counter()
    for _ in range(extra):
        open_cells = [c for c in cells if sizes[c] == base]
        cell = min(open_cells, key=lambda c: (got[c[1]], got[(c[1], c[2])], got[c[0]],
                                              got[(c[0], c[1])], c))
        sizes[cell] += 1
        for k in (cell[1], (cell[1], cell[2]), cell[0], (cell[0], cell[1])):
            got[k] += 1
    return sizes


def sample(pool: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = pool.iloc[rng.permutation(len(pool))]
    bin_order = pool["stance_bin"].value_counts().reindex(BINS, fill_value=0).sort_values().index.tolist()
    stmt_used = pd.Series(0, index=sorted(pool["statement_idx"].unique()))
    variant_used = pd.Series(0, index=sorted(pool["variant"].unique()))
    model_variant_used = {}
    used_pairs, picked = set(), []
    cells = pool.groupby(["model", "language", "negated"])
    sizes = cell_sizes(sorted(cells.groups), total)
    # scarcest cells (fewest against+neutral texts) pick first
    order = sorted(cells.groups, key=lambda c: (cells.get_group(c)["stance_bin"] != "for").sum())
    for cell in order:
        model = cell[0]
        per_cell = sizes[cell]
        group = cells.get_group(cell)
        quota = cell_quotas(group, per_cell, bin_order)
        mv = model_variant_used.setdefault(model, pd.Series(0, index=variant_used.index))
        n = 0
        # second pass tops up from any bin if a quota was blocked by (model, statement) reuse
        for bins in [[(b,) for b in bin_order], [tuple(bin_order)]]:
            for bs in bins:
                cand = group[group["stance_bin"].isin(bs) & ~group.index.isin(picked)]
                want = quota[bs[0]] if len(bs) == 1 else per_cell - n
                for _ in range(want):
                    ok = cand[[(model, s) not in used_pairs for s in cand["statement_idx"]]]
                    if ok.empty:
                        break
                    rank = pd.DataFrame({
                        "mv": mv.loc[ok["variant"]].to_numpy(),
                        "v": variant_used.loc[ok["variant"]].to_numpy(),
                        "s": stmt_used.loc[ok["statement_idx"]].to_numpy(),
                    }, index=ok.index)
                    # stable sort, so ties follow the random pool order
                    idx = rank.sort_values(["mv", "v", "s"], kind="stable").index[0]
                    s, v = pool.at[idx, "statement_idx"], pool.at[idx, "variant"]
                    picked.append(idx)
                    used_pairs.add((model, s))
                    stmt_used[s] += 1
                    variant_used[v] += 1
                    mv[v] += 1
                    cand = cand.drop(idx)
                    n += 1
        if n < per_cell:
            raise SystemExit(f"cell {cell}: only {n} of {per_cell} texts")
    return pool.loc[picked]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=204, help="LLM texts in all")
    ap.add_argument("--min-chars", type=int, default=300, help="drop one-liners")
    ap.add_argument("--max-chars", type=int, default=2500, help="keep items readable in one go")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=Path, default=DATA_DIR / "annotation_mix.csv")
    args = ap.parse_args()
    key_path = args.mix.with_name(args.mix.stem + "_key.csv")

    sheet = pd.read_csv(args.mix, encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(key_path, encoding="utf-8-sig", keep_default_na=False)
    if "source" in key.columns:
        old = set(key.loc[key["source"] == "llm", "item_id"])
        if old:
            print(f"dropping {len(old)} LLM items from an earlier run")
        key = key[key["source"] != "llm"]
        sheet = sheet[~sheet["item_id"].isin(old)]
    key = key[[c for c in key.columns if c in ("item_id", "EU Party", "language", "date", "speaker")]]
    # a canonical speech order (ids get renumbered below), so a rerun reproduces the ids
    sheet = sheet.sort_values(["language", "text"], kind="stable").reset_index(drop=True)
    key = key.set_index("item_id").loc[sheet["item_id"]].reset_index()

    pool = load_pool()
    pool = pool[pool["text"].str.len().between(args.min_chars, args.max_chars)]
    print(f"pool after filters: {len(pool)} texts, {pool['model'].nunique()} models")

    mix = sample(pool, args.total, args.seed).reset_index(drop=True)
    # temporary ids: the speech ids may already be spread over the whole range
    mix.insert(0, "item_id", [f"llm{i:03d}" for i in range(len(mix))])

    new_sheet = mix[["item_id", "language", "text"]].assign(**dict.fromkeys(ANSWER_COLUMNS, ""))
    sheet = pd.concat([sheet, new_sheet], ignore_index=True)
    sheet = sheet.iloc[np.random.default_rng(args.seed + 1).permutation(len(sheet))]
    # renumber in shuffled order: neither the id nor the position gives the source away
    renumber = dict(zip(sheet["item_id"], [f"am{i:03d}" for i in range(len(sheet))]))
    sheet["item_id"] = sheet["item_id"].map(renumber)
    sheet.to_csv(args.mix, index=False, encoding="utf-8-sig")

    key = key.assign(source="speech")
    new_key = mix[["item_id", "language", "model", "negated", "statement_idx", "statement",
                   "variant", "stance", "stance_bin"]].assign(**{"EU Party": "", "source": "llm"})
    key = pd.concat([key, new_key], ignore_index=True)
    key["item_id"] = key["item_id"].map(renumber)
    key = key.sort_values("item_id").reset_index(drop=True)
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    print(pd.crosstab(mix["model"], [mix["language"], mix["negated"]], margins=True))
    print(pd.crosstab(mix["stance_bin"], [mix["negated"]], margins=True))
    print(pd.crosstab(mix["variant"], mix["negated"], margins=True))
    per_model_variant = pd.crosstab(mix["model"], mix["variant"])
    print(f"variants per model: {per_model_variant.min().min()}-{per_model_variant.max().max()}")
    per_stmt = mix["statement_idx"].value_counts()
    print(f"statements: {per_stmt.size} of {pool['statement_idx'].nunique()}, "
          f"{per_stmt.min()}-{per_stmt.max()} texts each")
    print(f"wrote {len(mix)} LLM items to {args.mix} and {key_path} ({len(sheet)} total)")


if __name__ == "__main__":
    main()
