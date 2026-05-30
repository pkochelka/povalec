import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from utils import write_parquet_chunked

pd.options.future.infer_string = False

MULTIPARL_PARQUET  = Path("data/EuroParl Custom/preprocessed.parquet")
PARLEE_PARQUET     = Path("data/EuroParl Custom/preprocessed_parlee.parquet")
EU_DEBATES_PARQUET = Path("data/EuroParl Custom/preprocessed_eu_debates.parquet")
OUTPUT_DIR         = Path("data/EuroParl Custom")

MIN_TEXT_LEN     = 50
MIN_LANG_SAMPLES = 10_000
MIN_STRATUM_SIZE = 20
RANDOM_STATE     = 42

df = pd.concat([
    pd.read_parquet(MULTIPARL_PARQUET),
    pd.read_parquet(PARLEE_PARQUET),
    pd.read_parquet(EU_DEBATES_PARQUET),
], ignore_index=True)
print(f"Combined: {len(df):,} rows")

df = df.drop_duplicates(subset=["text"])
df = df[df["text"].str.len() >= MIN_TEXT_LEN]
lang_counts = df["language"].value_counts()
kept_langs = lang_counts[lang_counts >= MIN_LANG_SAMPLES].index
df = df[df["language"].isin(kept_langs)]
print(f"After dedup + length + language filter: {len(df):,} rows ({len(kept_langs)} languages kept)")

df["stratum"] = df["EU Party"] + "_" + df["language"]
counts = df["stratum"].value_counts()
rare_strata = counts[counts < MIN_STRATUM_SIZE].index
df["stratum_key"] = df["stratum"].where(~df["stratum"].isin(rare_strata), other="other")
print(f"Strata: {df['stratum_key'].nunique()} (merged {len(rare_strata)} rare combos into 'other')")

# Group key: all translations of the same speech share speaker+date
df["group"] = df["speaker"].astype(str) + "_" + df["date"].astype(str)
print(f"Unique speech groups: {df['group'].nunique():,}")

# 80% train / 20% temp, groups never cross splits
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, temp_idx = next(sgkf.split(df, df["stratum_key"], groups=df["group"]))
train = df.iloc[train_idx]
temp  = df.iloc[temp_idx]

# 50/50 split of temp into dev (10%) and test (10%)
sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
dev_idx, test_idx = next(sgkf2.split(temp, temp["stratum_key"], groups=temp["group"]))
dev  = temp.iloc[dev_idx]
test = temp.iloc[test_idx]

print(f"Train: {len(train):,}  Dev: {len(dev):,}  Test: {len(test):,}")

cols = ["date", "EU Party", "text", "language", "speaker"]
for split, name in [(train, "train"), (dev, "dev"), (test, "test")]:
    path = OUTPUT_DIR / f"{name}.parquet"
    write_parquet_chunked(split[cols], path)
    print(f"  Saved {name} -> {path}")
