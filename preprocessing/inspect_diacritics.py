"""Spot-check fix_diacritics on the real corpus: what it changes, and what it misses.

Reads the uncleaned split (data/EuroParl Custom/<split>.parquet, the clean-names
step's input) batch by batch and prints, per language:

  - a few repaired rows, as before -> after snippets around the first changes, to
    confirm the repairs are real damage and not correct text being altered;
  - how many repaired rows still contain a detached accent character afterwards
    (damage the repair map does not cover);
  - the detached accent characters found after a letter in rows it left alone,
    most common first (the macron "¯", for one, is not in the map).

    python preprocessing/inspect_diacritics.py --languages fr it lv --examples 5
"""

import argparse
import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from preprocessing.fix_diacritics import fix_diacritics

DATA_DIR = Path("data/EuroParl Custom")
# Every spacing accent / accent-like character, mapped or not.
DETACHED_RE = re.compile(r"(?<=[^\W\d_])[´`^ˆ˜¨°˚ˇ˘˙˝¸˛¯ˉ~]")


def snippets(before, after, n=3, width=30):
    """Context around the first `n` places where before and after differ."""
    out = []
    ops = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        out.append(f"    {before[max(0, i1 - width):i2 + width]!r}\n"
                   f"  -> {after[max(0, j1 - width):j2 + width]!r}")
        if len(out) == n:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="train")
    parser.add_argument("--languages", nargs="*", default=None, help="default: all")
    parser.add_argument("--examples", type=int, default=4, help="repaired rows shown per language")
    parser.add_argument("--max-rows", type=int, default=400_000, help="rows scanned in total")
    args = parser.parse_args()

    wanted = set(args.languages) if args.languages else None
    rows, repaired, still_detached = Counter(), Counter(), Counter()
    missed = defaultdict(Counter)
    shown = defaultdict(list)
    scanned = 0
    reader = pq.ParquetFile(DATA_DIR / f"{args.split}.parquet")
    for batch in reader.iter_batches(batch_size=20_000, columns=["text", "language"]):
        for text, lang in zip(batch.column("text").to_pylist(), batch.column("language").to_pylist()):
            if not text or (wanted and lang not in wanted):
                continue
            rows[lang] += 1
            fixed = fix_diacritics(text)
            if fixed != text:
                repaired[lang] += 1
                if DETACHED_RE.search(fixed):
                    still_detached[lang] += 1
                if len(shown[lang]) < args.examples:
                    shown[lang].append(snippets(text, fixed))
            else:
                for m in DETACHED_RE.finditer(text):
                    missed[lang][m.group()] += 1
        scanned += batch.num_rows
        if scanned >= args.max_rows:
            break

    for lang in sorted(rows, key=lambda l: -rows[l]):
        print(f"\n=== {lang}: {rows[lang]:,} rows scanned, {repaired[lang]:,} repaired "
              f"({100 * repaired[lang] / rows[lang]:.1f}%), "
              f"{still_detached[lang]:,} of those still hold a detached accent")
        print(f"  detached accents in rows left alone: {dict(missed[lang].most_common(8))}")
        for k, snips in enumerate(shown[lang], 1):
            print(f"  example {k}:")
            print("\n".join(snips))


if __name__ == "__main__":
    main()
