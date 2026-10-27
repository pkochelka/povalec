#!/usr/bin/env python3
"""Exact counts of the machine-translated rows in an MT-augmented langmatched train split.

Reads, from --track-dir:
  train_mt.parquet / .json                 every usable MT row (preprocessing/build_mt_train.py)
                                           and its filter report
  train_langmatched_mt.parquet / .json     what training saw (build_language_matched_split.py --mt);
                                           the MT rows are its last `mt_rows` rows (appended
                                           after the human rows in train_mt order)
  train.parquet                            to check the MT rows never duplicate a human row

Prints per target language: MT rows available and used, used rows by period of the source
speech (pre-2013 = the EP still translated every speech; 2013+ = original-language only),
by cluster, and the human rows next to them by origin (original / translated, same rule as
the split builder: a speech present in <= 2 languages is original). Also the source
languages of the used MT rows.

Usage, from the repository root (light: key columns only, ~1 min):
  python -m analysis.mt_train_stats
  python -m analysis.mt_train_stats --track-dir "data/EuroParl Custom/clusters_k4_national_ni_eval25k"
Output: results_mt_train_stats.txt + <track-dir>/mt_train_stats.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.core.labels import short_party
from utils import configure_stdout

configure_stdout()

PARTY = "EU Party"
KEYS = ["language", PARTY, "speaker", "date"]
CUTOFF = 2013
out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


def read_keys(path, columns=KEYS):
    df = pq.read_table(path, columns=columns).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


def period(dates):
    return np.where(dates.dt.year < CUTOFF, f"pre-{CUTOFF}", f"{CUTOFF}+")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track-dir", type=Path,
                        default=Path("data") / "EuroParl Custom" / "clusters_k4_national_ni_eval25k")
    parser.add_argument("--suffix", default="_mt", help="train_langmatched<suffix>.parquet")
    parser.add_argument("--report", type=Path, default=Path("results_mt_train_stats.txt"))
    args = parser.parse_args()
    track = args.track_dir
    pd.set_option("display.width", 250)

    built = json.loads((track / "train_mt.json").read_text(encoding="utf-8"))
    matched_json = json.loads((track / f"train_langmatched{args.suffix}.json").read_text(encoding="utf-8"))
    mt_all = read_keys(track / "train_mt.parquet", KEYS + ["source_language", "translator"])
    matched = read_keys(track / f"train_langmatched{args.suffix}.parquet")
    n_mt = matched_json["mt_rows"]
    human, used = matched.iloc[:len(matched) - n_mt].copy(), matched.iloc[len(matched) - n_mt:].copy()

    # The tail must be MT rows and nothing else: every key in train_mt, none in train.parquet.
    mt_keys = set(zip(mt_all.speaker, mt_all.date, mt_all.language))
    used_keys = list(zip(used.speaker, used.date, used.language))
    train = read_keys(track / "train.parquet")
    train_keys = set(zip(train.speaker, train.date, train.language))
    assert all(k in mt_keys for k in used_keys), "tail rows not in train_mt: file order changed?"
    assert not any(k in train_keys for k in used_keys), "an MT row duplicates a human row"
    # Leakage: no used MT row may come from a speech (speaker + date) held in dev or test.
    used_speeches = set(zip(used.speaker, used.date))
    held_out = {}
    for split in ("dev", "test", "dev_langmatched", "test_langmatched"):
        if (track / f"{split}.parquet").exists():
            keys = read_keys(track / f"{split}.parquet", ["speaker", "date"])
            held_out[split] = len(used_speeches & set(zip(keys.speaker, keys.date)))
    assert not any(held_out.values()), f"MT rows from dev/test speeches: {held_out}"
    src = mt_all.drop_duplicates(["speaker", "date", "language"]).set_index(["speaker", "date", "language"])
    used["source_language"] = src.loc[used_keys, "source_language"].to_numpy()
    used["translator"] = src.loc[used_keys, "translator"].to_numpy()

    say(f"track: {track}")
    say(f"translator(s): {used.translator.value_counts().to_dict()}")
    say(f"leakage check: {len(used_speeches):,} source speeches of the used MT rows; "
        f"overlap with held-out splits {held_out} (speaker + date)")
    say("\n[1] FILTERING (build_mt_train.py, train_mt.json)")
    for k in ("translations", "source_speeches", "sources_by_split", "dropped_language_already_in_track",
              "dropped_exact_duplicates", "dropped_second_translator", "dropped_transliterated",
              "dropped_short", "rows"):
        say(f"  {k}: {built.get(k)}")
    mt_all["period"] = period(mt_all.date)
    say(f"  usable MT rows by source period: {mt_all.period.value_counts().sort_index().to_dict()}")
    say(f"  usable source speeches (speaker+date) by period: "
        f"{mt_all.drop_duplicates(['speaker', 'date']).period.value_counts().sort_index().to_dict()}")

    say(f"\n[2] BALANCING (train_langmatched{args.suffix}.json)")
    say(f"  target mix (train's own pooled mix, without MT): "
        f"{ {short_party(k): round(v, 4) for k, v in matched_json['target_mix'].items()} }")
    say(f"  MT rows used: {n_mt:,} of {len(mt_all):,} usable ({n_mt / len(mt_all):.1%}); "
        f"human rows: {len(human):,}; total train: {len(matched):,} (MT share {n_mt / len(matched):.1%})")
    say(f"  max deviation from target mix: {matched_json['max_deviation']}")

    used["period"] = period(used.date)
    human["period"] = period(human.date)
    n_lang = human.groupby(["speaker", "date"])["language"].transform("nunique")
    human["origin"] = np.where(n_lang <= matched_json.get("original_max_languages", 2), "original", "translated")

    table = pd.DataFrame({
        "mt_usable": mt_all.language.value_counts(),
        "mt_used": used.language.value_counts(),
    })
    table = table.join(pd.crosstab(used.language, used.period).add_prefix("mt_used_"))
    table = table.join(pd.crosstab(used.language, used[PARTY].map(short_party)).add_prefix("mt_"))
    table = table.join(pd.crosstab(human.language, human.origin).add_prefix("human_"))
    table = table.join(pd.crosstab(human.language, human.period).add_prefix("human_"))
    table = table.fillna(0).astype(int).sort_index()
    table.loc["TOTAL"] = table.sum()
    say("\n[3] PER TARGET LANGUAGE (mt_* = DeepSeek rows; human_* = EP rows in the same train split)")
    say(table.to_string())

    say("\n[4] USED MT ROWS: source language (rows) x period")
    say(pd.crosstab(used.source_language, used.period, margins=True, margins_name="TOTAL").to_string())
    say("\n[5] USED MT ROWS: period x cluster")
    say(pd.crosstab(used.period, used[PARTY].map(short_party), margins=True, margins_name="TOTAL").to_string())

    table.to_csv(track / "mt_train_stats.csv")
    args.report.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nreport -> {args.report}; csv -> {track / 'mt_train_stats.csv'}")


if __name__ == "__main__":
    main()
