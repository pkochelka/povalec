import pandas as pd
import numpy as np
from pathlib import Path

from utils import write_parquet_chunked

pd.options.future.infer_string = False

MULTIPARL_CSV       = Path("data/EuroParl Custom/multi-europarl-lang_id.csv")
MULTIPARL_PARQUET   = Path("data/EuroParl Custom/preprocessed.parquet")
PARLEE_CSV          = Path("data/ParlEE/ParlEE_EP_plenary_speeches.csv")
PARLEE_PARQUET      = Path("data/EuroParl Custom/preprocessed_parlee.parquet")
EU_DEBATES_JSONL    = Path("data/EU Debates/train.jsonl")
EU_DEBATES_PARQUET  = Path("data/EuroParl Custom/preprocessed_eu_debates.parquet")

LANG_COLS = ["fr", "it", "da", "sv", "el", "pt", "en", "lv", "es", "nl",
             "fi", "de", "hu", "pl", "et", "sl", "sk", "lt", "mt", "cs",
             "ro", "bg", "hr"]

MULTIPARL_PARTY_MAPPING = {
    "EPP-ED": "PPE",   "EPP":  "PPE",
    "PES":    "S&D",   "S&D":  "S&D",
    "ELDR":   "ALDE",  "ALDE": "ALDE",  "DEM": "ALDE",
    "EFA":    "Greens/EFA",
    "NGL":    "GUE/NGL",
    "ECR":    "ECR",
    "ITS":    "ID",    "EFD":  "ID",
    "UEN":    "ECR",   "EDD":  "ECR",
    "TGI":    np.nan,  "NA":   np.nan,
}

PARLEE_PARTY_MAPPING = {
    "EPP-ED":     "PPE",        "EPP":        "PPE",
    "PES":        "S&D",        "S&D":        "S&D",
    "ALDE":       "ALDE",       "Renew":      "ALDE",
    "G/EFA":      "Greens/EFA", "Greens/EFA": "Greens/EFA",
    "EUL/NGL":    "GUE/NGL",    "GUE/NGL":    "GUE/NGL",
    "ECR":        "ECR",        "UEN":        "ECR",
    "EFD":        "ID",         "EFDD":       "ID",
    "ENF":        "ID",         "ID":         "ID",         "IND/DEM": "ID",
    "NI":         np.nan,       "PECH":       np.nan,
}

LANG_NAME_TO_CODE = {
    "Arabic": "ar", "Basque": "eu", "Bulgarian": "bg", "Croatian": "hr",
    "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Estonian": "et", "Finnish": "fi", "French": "fr", "German": "de",
    "Greek": "el", "Hungarian": "hu", "Irish": "ga", "Italian": "it",
    "Latin": "la", "Latvian": "lv", "Lithuanian": "lt", "Luxembourgish": "lb",
    "Maltese": "mt", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro",
    "Scottish Gaelic": "gd", "Slovak": "sk", "Slovenian": "sl",
    "Spanish": "es", "Swedish": "sv", "Welsh": "cy", "Yiddish": "yi",
}


def collapse(series):
    unique_vals = series.dropna().unique()
    if len(unique_vals) == 0:
        return np.nan
    if len(unique_vals) == 1:
        return unique_vals[0]
    return " ".join(str(v) for v in series.dropna())


def preprocess_multiparl():
    print("=== multiparl ===")
    _cols = ["Unnamed: 0", "EU Party", "date", "speaker"] + LANG_COLS
    df = pd.read_csv(MULTIPARL_CSV, low_memory=False, usecols=_cols)
    print(f"  Loaded {len(df):,} rows")

    df = df.dropna(subset=["EU Party"])
    df["truncated_party"] = df["EU Party"].apply(lambda x: x.split("/")[-1])
    df["party_group_std"] = df["truncated_party"].map(MULTIPARL_PARTY_MAPPING)

    df = df.groupby("Unnamed: 0", as_index=False).agg(collapse)
    df = df.dropna(subset=["party_group_std"])
    print(f"  {len(df):,} speeches after party filtering")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    id_cols = ["Unnamed: 0", "date", "party_group_std", "speaker"]
    df_long = df[id_cols + LANG_COLS].melt(
        id_vars=id_cols, value_vars=LANG_COLS, var_name="language", value_name="text"
    )
    df_long = df_long.dropna(subset=["text"])
    df_long["text"] = df_long["text"].str.strip()
    df_long = df_long[df_long["text"] != ""]
    df_long["text"] = df_long["text"].str.replace(r"^\([A-Z]{2}\)\s*", "", regex=True)

    df_out = df_long[["date", "party_group_std", "text", "language", "speaker"]].rename(
        columns={"party_group_std": "EU Party"}
    )
    print(f"  {len(df_out):,} rows in output")
    print(f"  Party distribution:\n{df_out['EU Party'].value_counts().to_string()}")
    print(f"  Language distribution:\n{df_out['language'].value_counts().to_string()}")

    write_parquet_chunked(df_out, MULTIPARL_PARQUET)
    print(f"  Saved to {MULTIPARL_PARQUET}\n")


def preprocess_parlee():
    print("=== ParlEE ===")
    df = pd.read_csv(PARLEE_CSV, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # A speech is (date, speechnumber): speechnumber restarts every sitting, so on its own
    # it merges up to 86 unrelated speeches (362,854 speeches, not 88,221).
    df_speeches = (
        df.sort_values(["date", "speechnumber", "sentencenumber"])
          .groupby(["date", "speechnumber"], as_index=False)
          .agg({"text": " ".join, "party": "first", "language": "first", "speaker": "first"})
    )
    print(f"  {len(df_speeches):,} speeches after grouping")

    df_speeches["EU Party"] = df_speeches["party"].map(PARLEE_PARTY_MAPPING)
    df_speeches = df_speeches.dropna(subset=["EU Party"])
    print(f"  {len(df_speeches):,} speeches after party filtering")

    df_speeches["date"] = pd.to_datetime(df_speeches["date"], dayfirst=True, errors="coerce")
    df_speeches["language"] = df_speeches["language"].map(LANG_NAME_TO_CODE)
    df_speeches = df_speeches.dropna(subset=["language", "date"])

    df_speeches["text"] = df_speeches["text"].str.strip()
    df_speeches = df_speeches[df_speeches["text"] != ""]

    df_out = df_speeches[["date", "EU Party", "text", "language", "speaker"]]
    print(f"  {len(df_out):,} rows in output")
    print(f"  Party distribution:\n{df_out['EU Party'].value_counts().to_string()}")
    print(f"  Language distribution:\n{df_out['language'].value_counts().to_string()}")

    write_parquet_chunked(df_out, PARLEE_PARQUET)
    print(f"  Saved to {PARLEE_PARQUET}\n")


def preprocess_eu_debates():
    print("=== EU Debates ===")
    df = pd.read_json(EU_DEBATES_JSONL, lines=True)
    print(f"  Loaded {len(df):,} rows")

    df = df[~df["speaker_party"].isin(["NI", "N/A"])]
    print(f"  {len(df):,} rows after dropping NI and N/A (non-inscrits)")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    df = df.dropna(subset=["date", "text"])

    df_out = df.rename(columns={
        "speaker_party": "EU Party",
        "intervention_language": "language",
        "speaker_name": "speaker",
    })[["date", "EU Party", "text", "language", "speaker"]]
    print(f"  {len(df_out):,} rows in output")

    parlee_texts = set(pd.read_parquet(PARLEE_PARQUET, columns=["text"])["text"])
    eu_texts = set(df_out["text"])
    intersection = eu_texts & parlee_texts
    print(f"  EU Debates unique texts: {len(eu_texts):,}")
    print(f"  ParlEE unique texts:     {len(parlee_texts):,}")
    print(f"  Text intersection size:  {len(intersection):,}")

    print(f"  Party distribution:\n{df_out['EU Party'].value_counts().to_string()}")
    print(f"  Language distribution:\n{df_out['language'].value_counts().to_string()}")

    write_parquet_chunked(df_out, EU_DEBATES_PARQUET)
    print(f"  Saved to {EU_DEBATES_PARQUET}\n")


if __name__ == "__main__":
    preprocess_parlee()
    preprocess_multiparl()
    preprocess_eu_debates()
