import pandas as pd
import json
import numpy as np
#from scrape_euandi import LANGS

PARTY_POSITIONS_PATH   = "data/euandi_2024_data/euandi_2024_parties.jsonl"
LLM_RESPONSES_PATH = "data/euandi_2024_results/qwen3.5-122b/summary.csv"
LANGS = "en,de,fr,it,es,pt,nl,pl,cz,sk,hu,ro,bg,hr,da,se,fi,ee,lv,lt,mt,gr,si,ie".split(",")

# ── Inverse of stance_to_likert ───────────────────────────────────────────────
def likert_to_stance(likert: float) -> float:
    """Map [1, 5] Likert mean back to [-1, 1] normalized score."""
    return (3 - likert) / 2

# ── Load LLM responses ────────────────────────────────────────────────────────
llm_df = pd.read_csv(LLM_RESPONSES_PATH, sep=";")
llm_df["statement"] = llm_df["statement"].str.strip()

# Convert all language columns from Likert [1-5] → stance [-1, 1]
for lang in LANGS:
    llm_df[f"{lang}_stance"] = llm_df[lang].apply(likert_to_stance)

# ── Load party responses ──────────────────────────────────────────────────────
records = []
with open(PARTY_POSITIONS_PATH) as f:
    for line in f:
        obj = json.loads(line)
        party_meta = {k: v for k, v in obj.items() if k != "responses"}
        for resp in obj.get("responses", []):
            records.append({**party_meta, **resp})

party_df = pd.DataFrame(records)
party_df = party_df[party_df["country_iso"] != "eu"]

party_to_group_map = {
    "EPP":  "PPE",        # European People's Party → Parti Populaire Européen (EP group name)
    "ECR":  "ECR",        # same name in both contexts
    "PES":  "S&D",        # Party of European Socialists → Socialists & Democrats (EP group)
    "ALDE": "ALDE",       # same name in both contexts
    "EGP":  "Greens/EFA", # European Green Party → Greens–European Free Alliance (EP group)
    "ID":   "ID",         # same name in both contexts
    "PEL":  "GUE/NGL",   # Party of the European Left → GUE/NGL (EP group)
}

party_df["EU Party"] = party_df["short_name"].map(party_to_group_map)
party_df["statement"] = party_df["statement"].str.strip()

# ── Match statements ──────────────────────────────────────────────────────────
# Exact match; flag misses for inspection
llm_statements   = set(llm_df["statement"])
party_statements = set(party_df["statement"])
unmatched = llm_statements.symmetric_difference(party_statements)
print(unmatched)
if unmatched:
    print(f"⚠️  {len(unmatched)} unmatched statements:")
    for s in unmatched:
        print(f"  · {s}")

# ── Compute proximity per party × language ────────────────────────────────────
# EuAndI methodology: agreement = 1 - |party_stance - llm_stance| / 2
# (dividing by 2 normalises the max possible distance to 1)

merged = party_df.merge(
    llm_df[["statement"] + [f"{l}_stance" for l in LANGS]],
    on="statement",
    how="inner"
)

party_id_cols = ["EU Party", "country_iso"]  # adjust to your schema

results = []
for lang in LANGS:
    col = f"{lang}_stance"
    tmp = merged[party_id_cols + ["statement", "normalized_answer", col]].copy()
    tmp["lang"]      = lang
    tmp["llm_stance"]    = tmp[col]
    tmp["party_stance"]  = tmp["normalized_answer"]
    tmp["agreement"]     = 1 - (tmp["party_stance"] - tmp["llm_stance"]).abs() / 2
    results.append(tmp[party_id_cols + ["statement", "lang",
                                         "party_stance", "llm_stance", "agreement"]])

eval_df = pd.concat(results, ignore_index=True)

# ── Aggregate: mean agreement per party × language ────────────────────────────
summary = (
    eval_df
    .groupby(party_id_cols + ["lang"])["agreement"]
    .mean()
    .reset_index()
    .rename(columns={"agreement": "mean_agreement"})
    .sort_values(["lang", "mean_agreement"], ascending=[True, False])
)

print(summary)