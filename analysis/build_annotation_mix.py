"""Sample a cluster- and language-balanced speech set for human annotation.

Reads  data/test_k4_group.parquet     speech-time EP group -> k=4 cluster ("EU Party")
       data/test_k4_national.parquet  speaker's national party -> its 2024 k=4 cluster
Writes data/annotation_mix.csv        blind sheet for annotators: item_id, language, text,
                                      and the empty ANSWER_COLUMNS (rows shuffled)
       data/annotation_mix_key.csv    item_id -> gold cluster, date, speaker

Every (cluster, language) cell gets the same number of rows (--per-cell, default 17;
4 clusters x 3 languages (--languages, default sk cs en) = 204, as many as the LLM
texts build_annotation_mix_llm.py adds). Languages left out of --languages are
dropped after the shuffle, like the filters below, so they don't reorder the rest.

**Stable parties only.** Neither parquet carries the national party or the EP group, so
"the speaker's party never switched EP group" is approximated as: the speaker's label in
the group track (EP group at speech time) equals their label in the national track
(their national party's 2024 cluster), and the speaker has a single label across all
their group-track rows. A party that moved across a cluster boundary (e.g. EPP -> PfE)
fails this; a move between two groups of the same cluster is invisible, but it doesn't
change the gold label either. Speakers absent from the national test split can't be
checked and are dropped.

**One version per speech.** The corpus holds translations, so one speech appears in
several languages under the same speaker and date. Each (speaker, date) is used at most
once, so no annotator sees the same speech twice in two languages.

**Cleaned again, then filtered.** The pool texts go through the current
clean_party_names (so patterns added since the parquets were built apply), then two kinds
of speech are dropped:
  holes       a removed name left its article or preposition dangling ("Les députés du
              prennent", "the Group of the and", "Skupiny v Evropském parlamentu") --
              broken grammar, and a cue in itself (most often the KKE vote formula);
  procedural  split / roll-call votes, oral amendments, points of order, referral back
              to committee, agenda motions -- no policy position to annotate.
Both are regex heuristics, tuned on a hand-read draw (they caught 25/27 and 12/13 of the
items flagged there, no false positives). Dropped rows are removed after the pool is
shuffled, so a redraw keeps every item that passes and refills only the gaps.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.clean_party_names import clean_text, strip_residue

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

LANGUAGES = ["sk", "cs", "en", "fr"]
LABEL = "EU Party"
# what the annotator fills in, one row per item
ANSWER_COLUMNS = ["choice", "secondary_choice", "confidence (1 = min, 3 = max)",
                  "reason (one short sentence in english)"]

# A function word left dangling where a name was cut: followed by punctuation, a dash,
# another function word or a verb instead of the noun it introduced.
HOLE_PATTERNS = {
    "en": [r"\b(?:the|by the|Group of the)\s*[,.;:)–—]",
           r"\bthe (?:are|were|has|have|and|will always)\b",
           r"\bof the (?:and|are|were|or|in)\b", r"\bof the ,",
           r"\bGroup of the (?:in|and|has|have|is|was)\b",
           r"\bof the Group (?:I|we)\b", r"\(\s*[Gg]roup\s*\)",
           r"(?:^|[.!?]\s)[’']s\b", r"’\.\s+[a-z]", r"\bthe ’\s"],
    "fr": [r"(?<![-’'])\b(?:du|de la|des|le|la|les|au|aux|par le)\s*[,.;:)–—]",
           r"\b(?:députés|membres|élus|eurodéputés)(?: européens)? du "
           r"(?!gouvernement|mouvement|Parlement|parlement)[a-zéè]+(?:ent|ont)\b",
           r"\bdu (?:ont|a|sont|votent|prennent)\b",
           r"\bde la (?:se|a|ont|sont)\b"],
    "cs": [r"\bSkupin\w*\s+v\s+Evropském\s+parlamentu\b",
           r"\b(?:ze|z|v)\s*[,;:–—)]\s",
           r"(?:^|[.!?]\s+)(?:však|a|ale)\s+(?:požádal|navrhl|předložil)\w*",
           r"\bkonfederace a\?", r"\b[Pp]říspěvek skupiny k\b", r"\(\s*[Ss]kupin\w*\s*\)"],
    "sk": [r"\bSkupin\w*\s+v\s+Európskom\s+parlamente\b",
           r"\b(?:zo|z|v)\s*[,;:–—)]\s", r"\bzo Skupiny[,.]",
           r"\bskupin\w*\s*[–—]\s*(?:sa|,|\()", r"\(\s*[Ss]kupin\w*\s*\)",
           r"\bso (?:predložili|navrhli|podali)\b",
           r"^\s*(?:by|je|sa)\s", r"\bokrem,",
           r"^\s*\w\s*$"],
}
# multiline: "^" is a line start (a sentence opened by a cut name)
HOLE_RE = {lang: re.compile("|".join(p), re.M) for lang, p in HOLE_PATTERNS.items()}

PROCEDURAL_RE = {lang: re.compile(p, re.I) for lang, p in {
    "en": r"split vote|separate vote|roll-call vote|oral amendment|point of order|Rule \d+|"
          r"voting list|my vote (?:be|was)|referr(?:ed|al) back to (?:the )?committee|"
          r"(?:put|placed) on the agenda|written reply|English version|give the floor to|"
          r"table a separate resolution|incorrectly recorded|(?:debate|vote) .{0,60}be postponed",
    "fr": r"vote par (?:division|appel nominal)|vote séparé|amendement oral|rappel au règlement|"
          r"article \d+(?:\.\d)? du règlement|conformément aux dispositions de l'article|"
          r"renvo\w+ .{0,30}(?:en|devant la) commission|inscrit\w* à l[’']ordre du jour|"
          r"mon vote soit|vote (?:défavorable|favorable) .{0,40}enregistr",
    "cs": r"dílčí hlasování|hlasování (?:po částech|podle jmen)|jmenovité hlasování|"
          r"ústní pozměňovací|jednacího řádu|harmonogram\w*.{0,40}hlasov|"
          r"hlasov\w*.{0,40}harmonogram|zpět (?:do|výboru)",
    "sk": r"hlasovanie (?:po častiach|podľa mien)|ústny pozmeňujúci|rokovacieho poriadku|"
          r"späť (?:do|výboru)|pozmeňujúci a doplňujúci návrh k odseku",
}.items()}
# Longer speeches that mention a procedure usually argue a position as well.
PROCEDURAL_MAX_CHARS = 1200


def has_hole(text: str, lang: str) -> bool:
    return HOLE_RE[lang].search(text) is not None


def is_procedural(text: str, lang: str) -> bool:
    return len(text) <= PROCEDURAL_MAX_CHARS and PROCEDURAL_RE[lang].search(text) is not None


def stable_speaker_rows(group: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    nat = national.groupby("speaker")[LABEL].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    n_group_labels = group.groupby("speaker")[LABEL].nunique()
    keep = (group["speaker"].map(nat) == group[LABEL]) & (group["speaker"].map(n_group_labels) == 1)
    return group[keep]


def sample(pool: pd.DataFrame, drop: pd.Series, per_cell: int, max_per_speaker_cell: int,
           max_per_speaker: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # shuffle first, drop after: the order of the surviving rows doesn't depend on `drop`
    pool = pool.iloc[rng.permutation(len(pool))]
    pool = pool[~drop.loc[pool.index]]
    cells = pool.groupby([LABEL, "language"])
    # Scarcest cells first, so they get first pick of the shared speeches and speakers.
    order = sorted(cells.groups, key=lambda c: len(cells.groups[c]))
    used_speeches, speaker_total, picked = set(), {}, []
    for cell in order:
        rows, speaker_cell = [], {}
        for idx, row in cells.get_group(cell).iterrows():
            speech = (row["speaker"], row["date"])
            if (speech in used_speeches
                    or speaker_cell.get(row["speaker"], 0) >= max_per_speaker_cell
                    or speaker_total.get(row["speaker"], 0) >= max_per_speaker):
                continue
            rows.append(idx)
            used_speeches.add(speech)
            speaker_cell[row["speaker"]] = speaker_cell.get(row["speaker"], 0) + 1
            speaker_total[row["speaker"]] = speaker_total.get(row["speaker"], 0) + 1
            if len(rows) == per_cell:
                break
        if len(rows) < per_cell:
            raise SystemExit(f"cell {cell}: only {len(rows)} of {per_cell} rows under the caps")
        picked.extend(rows)
    out = pool.loc[picked]
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=17)
    ap.add_argument("--languages", nargs="+", choices=LANGUAGES, default=["sk", "cs", "en"])
    ap.add_argument("--min-chars", type=int, default=300, help="drop one-liners")
    ap.add_argument("--max-chars", type=int, default=2500, help="keep items readable in one go")
    ap.add_argument("--max-per-speaker-cell", type=int, default=2)
    ap.add_argument("--max-per-speaker", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "annotation_mix.csv")
    args = ap.parse_args()

    group = pd.read_parquet(DATA_DIR / "test_k4_group.parquet")
    national = pd.read_parquet(DATA_DIR / "test_k4_national.parquet")

    pool = stable_speaker_rows(group, national)
    pool = pool[pool["language"].isin(LANGUAGES)]
    lengths = pool["text"].str.len()
    pool = pool[lengths.between(args.min_chars, args.max_chars)]
    print(f"pool after filters: {len(pool)} rows, {pool['speaker'].nunique()} speakers")

    counts = Counter()
    cleaned = [strip_residue(clean_text(t, counts)) for t in pool["text"]]
    changed = pool["text"].ne(pd.Series(cleaned, index=pool.index))
    pool = pool.assign(text=cleaned)
    print(f"re-cleaned: {changed.sum()} rows changed {dict(counts)}")

    hole = pd.Series([has_hole(t, l) for t, l in zip(pool["text"], pool["language"])], index=pool.index)
    proc = pd.Series([is_procedural(t, l) for t, l in zip(pool["text"], pool["language"])], index=pool.index)
    print(f"dropped: {hole.sum()} with a name hole, {proc.sum()} procedural, "
          f"{(hole | proc).sum()} in all")
    print(pd.crosstab(pool[LABEL], [hole.rename("hole"), proc.rename("procedural")]))

    other_lang = ~pool["language"].isin(args.languages)
    mix = sample(pool, hole | proc | other_lang, args.per_cell, args.max_per_speaker_cell, args.max_per_speaker, args.seed)
    mix.insert(0, "item_id", [f"am{i:03d}" for i in range(len(mix))])

    sheet = mix[["item_id", "language", "text"]].assign(**dict.fromkeys(ANSWER_COLUMNS, ""))
    sheet.to_csv(args.out, index=False, encoding="utf-8-sig")
    key_path = args.out.with_name(args.out.stem + "_key.csv")
    mix[["item_id", LABEL, "language", "date", "speaker"]].to_csv(key_path, index=False, encoding="utf-8-sig")

    print(pd.crosstab(mix[LABEL], mix["language"], margins=True))
    print(f"speakers: {mix['speaker'].nunique()}, years {mix['date'].dt.year.min()}-{mix['date'].dt.year.max()}")
    print(f"wrote {args.out} and {key_path}")


if __name__ == "__main__":
    main()
