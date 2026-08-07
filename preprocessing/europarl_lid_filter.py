"""Language-ID filtering for the multiparallel EuroParl CSV.

Original author: Paul Lerner
Source:  https://github.com/PaulLerner/21-EuroParl/blob/main/data/lid.ipynb
Licence: MIT, Copyright (c) 2025 Paul Lerner -- see preprocessing/LICENSE.21-EuroParl
Paper:   Lerner and Yvon (2025), "Assessing the Political Fairness of Multilingual
         LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset"

LinkedEP labels each translation with the language it is supposed to be in, and
is sometimes wrong. Every cell is run through fastText lid.176 and blanked when
the predicted language does not match its column; rows left with too few verified
languages are dropped.

    multi-europarl.csv -> [this script] -> multi-europarl-lang_id.csv

Adapted for this project: the filter itself is Lerner's, unchanged. What changed
here is packaging plus one bug fix -- the script was moved out of the gitignored
data directory into preprocessing/, given the repo-standard argparse, and the
year-based train/dev/test split was fixed. The version shipped in the data
directory did `subset["year"] == 2010` although multi-europarl.csv has no `year`
column (its header is `,fr,it,src_lang,date,speaker,EU Party,...`), so it raised
KeyError and the real artifact must have come from Lerner's notebook. The year is
now derived from `date`, and writing that split CSV is opt-in (--write-split-csv)
because nothing downstream reads it: split_preprocessed_data.py re-splits the
corpus from scratch on its own group-disjoint, class-balanced carve.

THE --min-languages QUIRK (do not "fix" it)
-------------------------------------------
Before the row filter, every NON-language column of the boolean mask is set to
True. Those six metadata columns (src_lang, date, speaker, EU Party,
NationalParty, EU Committee) therefore count toward the per-row total, so the
default threshold of 9 really means "at least 3 verified languages". This is
carried over verbatim from the notebook: it is the rule that produced the
corpus every existing classifier in this repo was trained on. Raising it to a
true 9 would silently yield a different, much smaller dataset.

Usage
-----
    python preprocessing/europarl_lid_filter.py
    python preprocessing/europarl_lid_filter.py --input-csv <small.csv> --output-csv <out.csv>

Requires lid.176.bin (fetched by fetch_raw_data.py --source lid-model) and a
fastText binding. Neither `fasttext`/`fasttext-wheel` nor `fasttext-predict`
publishes wheels for Python 3.14 yet and `fasttext` builds from C++ source, so
this step may need an older interpreter; see load_lid_model() and the note in
requirements.txt.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_INPUT_CSV = DATA_DIR / "multi-europarl.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "multi-europarl-lang_id.csv"
DEFAULT_SPLIT_CSV = DATA_DIR / "21-multi-europarl.csv"
DEFAULT_LID_MODEL = DATA_DIR / "lid.176.bin"

DEV_YEAR = 2010
TEST_YEAR = 2011


def load_lid_model(path):
    """Load lid.176.bin through whichever fastText binding is installed.

    `fasttext` (or `fasttext-wheel`) is the reference implementation;
    `fasttext-predict` is a prediction-only build that ships wheels for more
    interpreters. Both expose load_model(path).predict(text) identically.
    """
    if not path.exists():
        raise RuntimeError(
            f"{path} not found; run preprocessing/fetch_raw_data.py --source lid-model"
        )
    for module in ("fasttext", "fasttext_predict"):
        try:
            return __import__(module).load_model(str(path))
        except ImportError:
            continue
    raise RuntimeError(
        "no fastText binding installed. Try 'pip install fasttext-wheel' or "
        "'pip install fasttext-predict'; if neither has a wheel for your Python, "
        "run this one step on an older interpreter (3.10-3.13)."
    )


def write_split_csv(data, path):
    """Lerner's year-based split. Kept for parity; nothing downstream reads it."""
    subset = data.copy()
    year = pd.to_datetime(subset["date"], errors="coerce").dt.year
    subset["split"] = "train"
    subset.loc[year == DEV_YEAR, "split"] = "dev"
    subset.loc[year == TEST_YEAR, "split"] = "test"
    print("Split distribution:", Counter(subset["split"]))
    subset.to_csv(path)
    print(f"Wrote {path}")


def language_id_mask(data, languages, model):
    """Per-cell boolean: does fastText agree with the column's language label?"""
    lang_ids = {}
    for language in tqdm(languages, desc="lid"):
        lang_id = []
        for text in data[language]:
            if pd.isna(text):
                lang_id.append(False)
            else:
                label, prob = model.predict(" ".join(str(text).split()))
                lang_id.append(label[0][-2:] == language)
        lang_ids[language] = lang_id
    return pd.DataFrame(lang_ids, index=data.index)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
                        help=f"default {DEFAULT_INPUT_CSV}")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help=f"default {DEFAULT_OUTPUT_CSV}")
    parser.add_argument("--lid-model", type=Path, default=DEFAULT_LID_MODEL,
                        help=f"fastText lid.176.bin (default {DEFAULT_LID_MODEL})")
    parser.add_argument("--mask-csv", type=Path, default=None,
                        help="also write the raw per-cell boolean mask here")
    parser.add_argument("--write-split-csv", nargs="?", type=Path,
                        const=DEFAULT_SPLIT_CSV, default=None,
                        help=f"write Lerner's year-based split CSV "
                             f"(default path {DEFAULT_SPLIT_CSV}); unused downstream")
    parser.add_argument("--min-languages", type=int, default=9,
                        help="row keep threshold on the mask row-sum; see the module "
                             "docstring before changing this (default 9)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input_csv}")
    data = pd.read_csv(args.input_csv, index_col="Unnamed: 0", low_memory=False)
    print(f"  {len(data):,} rows, {len(data.columns)} columns")

    if args.write_split_csv:
        write_split_csv(data, args.write_split_csv)

    model = load_lid_model(args.lid_model)

    languages = [col for col in data.columns if len(col) == 2]
    print(f"  Language columns: {languages}")

    lang_ids_df = language_id_mask(data, languages, model)

    # Share of non-empty cells fastText disagreed with, per language column.
    # (to_string, not the notebook's to_markdown, which needs `tabulate`.)
    langid_stats = {}
    for language in languages:
        langid_stats[language] = 1 - lang_ids_df[language][~data[language].isna()].mean()
    print("\nLanguage ID error rates:")
    print((pd.Series(langid_stats) * 100).round(1).astype(str).add("%").to_string())

    if args.mask_csv:
        lang_ids_df.to_csv(args.mask_csv)
        print(f"Wrote mask to {args.mask_csv}")

    # Metadata columns are forced True so they survive the mask -- and, as the
    # docstring explains, they also inflate the row-sum below.
    for column in set(data.columns) - set(languages):
        lang_ids_df[column] = True

    data_filtered = data[lang_ids_df]
    data_final = data_filtered[lang_ids_df.T.sum() >= args.min_languages]

    print(f"\nFinal dataset shape: {data_final.shape}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data_final.to_csv(args.output_csv)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
