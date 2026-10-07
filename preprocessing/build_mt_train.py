#!/usr/bin/env python3
"""Machine-translated train rows for a national cluster track: <track>/train_mt.parquet.

Original-language EP text in a language comes only from that country's MEPs, so it
carries their party mix (original Slovak is 99% Sovereigntist right), and the classifier
reads any Slovak that is not EP translationese -- LLM-written or LLM-translated -- as
Sov-right (analysis/translation_probe.py). Machine translations of speeches from every
cluster into every language put counterexamples into exactly that region.

Two translation sets, both keyed by the source speech (speaker + date):

  deepseek  data/augmented/translations.augmented.parquet: DeepSeek-V4-Pro translations
            of ~14.5k speeches (2007-2023) into 21 languages, labelled with the old EP
            groups. The label is replaced by the track's cluster.
  kimi      <mt_augmentation dir>/plan.parquet + translations.jsonl from
            preprocessing/build_mt_augmentation.py (Kimi K3). Left out by default, so Kimi
            stays a held-out translator for analysis/translation_probe.py.

A translation is kept only if its source speech is in the track's TRAIN split (a speech
group never straddles splits, so dev/test translations would leak), it takes the track's
cluster label, and the track does not already hold that speech in the target language
(no MT copy next to the human version or the original). Then every text goes through the
same cleaning as the splits (clean_party_names._clean_chunk: accents, EP group and party
names, residue, titled person names) -- MT re-spells names the cleaned source had lost,
e.g. a Hungarian inflected group name comes back as "PPE".

Output columns: the split columns (date, EU Party, text, language, speaker) plus
translator and source_language. build_language_matched_split.py --mt adds these rows to
train as their own (language, "mt") cells.

Usage, from the repository root:
  python -m preprocessing.build_mt_train --track-dir "data/EuroParl Custom/clusters_k4_national_ni_eval25k"
  python -m preprocessing.build_mt_train --translators deepseek,kimi
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from preprocessing.clean_party_names import _clean_chunk

PARTY = "EU Party"
COLUMNS = ["date", PARTY, "text", "language", "speaker"]
DEFAULT_TRACK = Path("data") / "EuroParl Custom" / "clusters_k4_national_ni_eval25k"
DEFAULT_DEEPSEEK = Path("data") / "augmented" / "translations.augmented.parquet"
DEFAULT_KIMI = Path("data") / "EuroParl Custom" / "clusters_k4_national_ni" / "mt_augmentation"
MIN_WORDS = 5
CHUNK = 500


def load_deepseek(path):
    df = pd.read_parquet(path, columns=["date", "text", "language", "speaker"])
    df["translator"] = "deepseek-v4-pro"
    return df


def load_kimi(directory):
    plan = pd.read_parquet(directory / "plan.parquet", columns=["target", "row", "date", "speaker"])
    lines = (directory / "translations.jsonl").read_text(encoding="utf-8").splitlines()
    done = pd.DataFrame([json.loads(line) for line in lines if line.strip()])
    df = done.merge(plan, on=["target", "row"]).rename(columns={"target": "language"})
    df["translator"] = "kimi-k3"
    return df[["date", "text", "language", "speaker", "translator"]]


def track_speeches(track_dir):
    """(speaker, date) -> split, cluster, languages present, for train/dev/test."""
    parts = []
    for split in ("train", "dev", "test"):
        keys = pd.read_parquet(track_dir / f"{split}.parquet", columns=["date", PARTY, "language", "speaker"])
        keys["split"] = split
        parts.append(keys)
    keys = pd.concat(parts, ignore_index=True)
    keys["date"] = pd.to_datetime(keys["date"])
    speeches = keys.groupby(["speaker", "date"]).agg(
        split=("split", "first"), n_split=("split", "nunique"),
        cluster=(PARTY, "first"), n_cluster=(PARTY, "nunique"),
        languages=("language", lambda s: frozenset(s)),
    ).reset_index()
    bad = speeches[(speeches.n_split > 1) | (speeches.n_cluster > 1)]
    if len(bad):
        raise SystemExit(f"{len(bad)} speeches straddle splits or clusters in {track_dir}")
    # the language the speech was given in: the one language of an original-only speech
    return speeches


def clean(texts, languages, workers):
    chunks = [(texts[i:i + CHUNK], languages[i:i + CHUNK]) for i in range(0, len(texts), CHUNK)]
    cleaned, keep = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part, part_keep, _, _ in pool.map(_clean_chunk, chunks):
            cleaned += part
            keep += part_keep
    return cleaned, keep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--translators", default="deepseek", help="comma list of deepseek, kimi")
    parser.add_argument("--deepseek", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--kimi-dir", type=Path, default=DEFAULT_KIMI)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--out", type=Path, default=None, help="default <track-dir>/train_mt.parquet")
    args = parser.parse_args()
    out = args.out or args.track_dir / "train_mt.parquet"

    translators = [t.strip() for t in args.translators.split(",") if t.strip()]
    frames = []
    if "deepseek" in translators:
        frames.append(load_deepseek(args.deepseek))
    if "kimi" in translators:
        frames.append(load_kimi(args.kimi_dir))
    if not frames:
        raise SystemExit("no translators selected")
    mt = pd.concat(frames, ignore_index=True)
    mt["date"] = pd.to_datetime(mt["date"])
    print(f"{len(mt):,} translations from {translators}")

    speeches = track_speeches(args.track_dir)
    mt = mt.merge(speeches, on=["speaker", "date"], how="left")
    sources = mt.drop_duplicates(["speaker", "date"])
    report = {"translations": int(len(mt)), "source_speeches": int(len(sources)),
              "sources_by_split": sources.split.fillna("not in track").value_counts().to_dict()}
    print("source speeches by split:", report["sources_by_split"])

    mt = mt[mt.split == "train"]
    exists = [lang in langs for lang, langs in zip(mt.language, mt.languages)]
    report["dropped_language_already_in_track"] = int(sum(exists))
    mt = mt[[not e for e in exists]]
    # A (speaker, date) can hold several speeches, so only exact duplicates go; where two
    # translators covered the same (speaker, date, language), the first one listed wins.
    before = len(mt)
    mt = mt.drop_duplicates(["speaker", "date", "language", "text"])
    report["dropped_exact_duplicates"] = int(before - len(mt))
    first = mt.groupby(["speaker", "date", "language"]).translator.transform("first")
    report["dropped_second_translator"] = int((mt.translator != first).sum())
    mt = mt[mt.translator == first]
    mt["source_language"] = [next(iter(l)) if len(l) == 1 else "multi" for l in mt.languages]

    print(f"cleaning {len(mt):,} texts with {args.workers} workers")
    texts, keep = clean(mt.text.astype(str).tolist(), mt.language.tolist(), args.workers)
    mt = mt.assign(text=texts)[keep]
    report["dropped_transliterated"] = int(len(keep) - sum(keep))
    short = mt.text.str.split().str.len() < MIN_WORDS
    report["dropped_short"] = int(short.sum())
    mt = mt[~short]

    mt = mt.rename(columns={"cluster": PARTY})[COLUMNS + ["translator", "source_language"]]
    mt = mt.sort_values(["language", PARTY, "speaker", "date"]).reset_index(drop=True)
    partial = out.with_suffix(".parquet.partial")
    mt.to_parquet(partial, index=False)
    os.replace(partial, out)

    table = pd.crosstab(mt.language, mt[PARTY])
    print(f"\n{len(mt):,} rows -> {out}")
    print(table.to_string())
    print("by translator:", mt.translator.value_counts().to_dict())
    report.update(rows=int(len(mt)), translators=translators,
                  rows_per_language_cluster={lang: row for lang, row in table.to_dict(orient="index").items()},
                  min_cell=int(table.to_numpy().min()))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
