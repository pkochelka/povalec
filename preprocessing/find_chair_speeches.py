"""Find the speeches the sitting's chair made, so preprocess_data.py can drop them.

Reads  data/EuroParl Custom/data/linkedEP/<Language>.ttl   (the LinkedEP corpora)
Writes data/EuroParl Custom/chair_speeches.txt              one LinkedEP speech id per line

Why this exists
---------------
Whoever presides a sitting -- the EP President or a Vice-President -- speaks between
the other speakers: "The next item is ...", "J'appelle la question n° 6 de M. X",
"The debate is closed", "Thank you, Mr X". The corpora attribute those speeches to the
presiding MEP and so label them with that MEP's group, although they carry no
political content -- and a read-out question carries SOMEONE ELSE's (a PES member's
question labelled Radical left because a GUE/NGL Vice-President read it out). In a
400-speech annotation sample 12 were chair speeches, 7 of them Progressive federalists.

Where the chair is marked
-------------------------
LinkedEP (1999 - 2017-07-06): each language file keeps the transcript's speaker line as
`lpv:unclassifiedMetadata`, and for a chair speech that line is just the chair label in
the sitting's language: "President.", "Předsedající. −", "El Presidente.", "Die
Präsidentin.", "Le Président. –", "Πρόεδρος. –", ... CHAIR_LABEL_RE must match the WHOLE
value (after trailing punctuation), so "President-in-Office of the Council", "President
of the Commission" or "Presidente della Commissione" never match. Checked against the
English and Czech files: every chair-co-occurring speaker label is covered, and on the
2009-2012 English speeches the label marks 5.8% of MEP speeches. `lpv:spokenAs` and
Events_and_structure.ttl carry no chair information (checked); this label is the only
source marker.

ParlEE (2009-2019) has no chair flag, but its (date, speechnumber) IS the LinkedEP
speech id ("12/01/2009", "1-004" = 2009-01-12-Speech-1-004, same text), so up to
2017-07-06 the LinkedEP list marks ParlEE too. After that, preprocess_data.py falls back
to chair_formula_mask(): the chair's standard English formulas, propagated to every
speech by the same speaker in the same agenda item (which catches the "Thank you, Mr X"
interjections), except one that addresses the chair (the same MEP after a handover). On
the 2009-2012 LinkedEP English speeches, scored against the label: precision 0.994,
recall 0.79; most "false positives" are chair speeches the label missed ("The debate is
closed."). A structural rule (the speaker who keeps returning between
different speakers) was tried and dropped: precision 0.51, blue-card exchanges look the
same.

EU Debates needs none of this: its chair rows are speaker_role "EUROPARL President"
with speaker_party "N/A", which preprocess_data.py already drops.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LINKEDEP_DIR = PROJECT_ROOT / "data" / "EuroParl Custom" / "data" / "linkedEP"
CHAIR_IDS = PROJECT_ROOT / "data" / "EuroParl Custom" / "chair_speeches.txt"

# The chair's speaker label in each EP language, as the transcripts write it.
CHAIR_LABEL_RE = re.compile(r"""(?:
    (?:Le\ |La\ )?Présidente? | (?:Der\ |Die\ )?Präsident(?:in)?
  | (?:El\ |La\ |Il\ |O\ |A\ )?President[ea] | (?:De\ )?Voorzitter | Talman(?:nen)? | Ordföranden
  | Formand(?:en)? | Puhemies | (?:Ο\ |Η\ )?Πρόεδρος | Przewodnicząc[ya] | Předsedající | Předsed(?:a|kyně)
  | Predsedajúc[ia] | Predsed(?:a|níčka) | (?:Az\ )?Elnök | Predsedujoč[ia] | Predsedni(?:k|ca)
  | (?:Istungi\ )?Juhataja | Priekšsēdētāj[sa] | Sēdes\ vadītāj[sa] | (?:Posėdžio\ )?Pirminink(?:as|ė)
  | Președinte(?:le)? | Președinta | Председател(?:ят|ката)? | Председателстващ(?:ият|ата)?
  | (?:Il-)?Presidenti?                                    # en, and mt "Il-President"
  | Predsjedavajuć[ia] | Predsjedni(?:k|ca) | (?:An\ t)?Uachtarán
)""", re.X | re.I)
_TRAILING_RE = re.compile(r"[\s.:,\-–−—‒]+$")
_METADATA_RE = re.compile(r'lp_eu:(\S+) lpv:unclassifiedMetadata "(.*)" \.\s*$')

# The chair's standard English formulas (ParlEE's EP text is English). At the start of
# the speech, after any stray punctuation, unless marked otherwise.
_LEAD = r"^[\s.\-–−—]*"
CHAIR_FORMULA_RE = re.compile(_LEAD + r"""(?:
      [TΤ]he\ next\ item\ (?:is|on\ the\ agenda\ is)
    | I\ declare\ (?:resumed|open|closed) | The\ sitting\ (?:is|was)\ (?:opened|closed|suspended|resumed)
    | The\ (?:joint\ )?debate\ is\ closed | Th(?:is|e)\ (?:item|debate)\ is\ closed
    | Question\ (?:No|number)\ \d+\ by
    | (?:That|This)\ concludes\ (?:the|Question\ Time|this)
    | (?:We\ )?(?:shall\ )?(?:now\ )?(?:move|proceed|turn)\ (?:on\ )?to\ the\ vote
    | The\ vote\ will\ take\ place | The\ Minutes\ of\ (?:yesterday|the\ previous)
    | I\ have\ received\ .{0,80}motions?\ for\ (?:a\ )?resolution | Voting\ time
    | I\ give\ the\ floor\ to
    | (?:Ladies\ and\ gentlemen|Colleagues),\ (?:I\ would\ like\ to|allow\ me\ to|it\ is\ my)
      \ (?:great\ pleasure\ to\ )?(?:extend\ a\ warm\ )?welcome
    | (?:On\ behalf\ of\ (?:the\ House|Parliament),\ )?I\ (?:would|should)\ like\ to
      \ (?:extend\ a\ warm\ )?welcome\ (?:the\ )?(?:delegation|members\ of)
    | I\ would\ like\ to\ make\ (?:several\ |a\ )?(?:statements?|announcements?)\ at\ the\ (?:beginning|start)\ of\ this
  )
  | The\ (?:joint\ )?debate\ is\ closed\. | The\ vote\ will\ take\ place\ (?:today|tomorrow|on|at|during)
  | [Yy]ou\ have\ (?:exceeded|used\ up)\ your\ (?:speaking\ )?time | Please\ keep\ to\ (?:the\ )?(?:rules|time)
  | You\ do\ not\ have\ the\ floor
""", re.I | re.M | re.X)


def is_chair_label(value: str) -> bool:
    """A LinkedEP speaker-line value that is nothing but the chair label."""
    return bool(CHAIR_LABEL_RE.fullmatch(_TRAILING_RE.sub("", value.strip())))


# A speech that addresses the chair is not the chair's: it is the same MEP after handing
# the chair over ("The next item is ..." then, later in the item, "Madam President, I
# would like us to reflect ..."). Only 21 of 5,300 labelled chair speeches open this way.
ADDRESSES_CHAIR_RE = re.compile(
    r"^[\s.\-–−—]*(?:(?:Thank you(?: very much)?|Many thanks)(?: for the floor)?,?\s+)?"
    r"(?:Mr|Madam|Madame|Mrs|Ms)\.?\s+President\b", re.I)


def chair_formula_mask(df: pd.DataFrame, text: str = "text",
                       by: tuple[str, ...] = ("date", "agenda", "speaker")) -> pd.Series:
    """Speeches the chair made, from the text: a speech opening with a chair formula
    marks its speaker as the chair of that agenda item, and every speech by them in
    the item goes with it -- unless it addresses the chair (ADDRESSES_CHAIR_RE)."""
    texts = df[text].fillna("")
    seed = texts.str.contains(CHAIR_FORMULA_RE)
    chaired = seed.groupby([df[c] for c in by], dropna=False).transform("any")
    return seed | (chaired & ~texts.str.contains(ADDRESSES_CHAIR_RE))


def linkedep_language_files(linkedep_dir: Path) -> list[Path]:
    """The top-level Turtle files. Only the language corpora carry speaker lines, but
    scanning the few support files costs seconds, so none is special-cased."""
    return sorted(p for p in linkedep_dir.glob("*.ttl") if not p.stem.endswith("_prov"))


def find_linkedep_chair_ids(files: list[Path]) -> tuple[set[str], Counter]:
    """Speech ids whose speaker line is a chair label in any language file, and how
    often each label was seen (the report, to spot a label the pattern misses)."""
    ids: set[str] = set()
    labels: Counter = Counter()
    for path in files:
        before = len(ids)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "unclassifiedMetadata" not in line:
                    continue
                m = _METADATA_RE.match(line)
                if m and is_chair_label(m[2]):
                    ids.add(m[1])
                    labels[_TRAILING_RE.sub("", m[2].strip())] += 1
        if len(ids) > before:
            print(f"  {path.name:<18} +{len(ids) - before:>7,} chair speeches")
    return ids, labels


def main() -> None:
    files = linkedep_language_files(LINKEDEP_DIR)
    if not files:
        sys.exit(f"no LinkedEP language files in {LINKEDEP_DIR} (run the fetch step)")
    print(f"=== chair speeches: scanning {len(files)} LinkedEP language files ===")
    ids, labels = find_linkedep_chair_ids(files)
    CHAIR_IDS.write_text("".join(f"{i}\n" for i in sorted(ids)), encoding="utf-8")
    years = Counter(i[:4] for i in ids)
    print(f"  {len(ids):,} chair speeches -> {CHAIR_IDS}")
    print(f"  per year: {dict(sorted(years.items()))}")
    print(f"  labels seen: {labels.most_common(30)}")


def load_chair_ids() -> set[str]:
    """The ids main() wrote; preprocess_data.py stops if the chair step has not run."""
    if not CHAIR_IDS.exists():
        sys.exit(f"{CHAIR_IDS} missing: run the `chair` step (find_chair_speeches.py) first")
    return set(CHAIR_IDS.read_text(encoding="utf-8").split())


if __name__ == "__main__":
    main()
