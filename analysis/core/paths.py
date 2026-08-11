"""The filename and column grammar of the results tree, in both directions.

Results metadata — language set, prompt framing, track, scoring method — is carried by
file and column *names*, not by any manifest. That makes the naming a shared contract,
and it was previously re-derived per module: five filename regexes at three different
levels of strictness, against writers that built the same names with f-strings
elsewhere. Writer and reader could drift with nothing raising.

Both directions live here. The builders (`responses_name` and friends) are what
`analyze_all.py` predicts output paths with; the patterns are what the readers match.
They are written next to each other so a change to one is a visibly incomplete change
until the other follows.

Grammar, for the record:

    [speeches_]{langs}{variant}[_scored|_classified].csv   result files
    vaa[_speeches]{variant}_{langs}.csv                    agreement files
    {kind}_{lang}{variant}_v{paraphrase}                    columns

`{variant}` is the framing suffix from `utils.VARIANT_PATTERN` — "" or "_negated".
"""
import re

from utils import VARIANT_LABELS, VARIANT_PATTERN

# The language set as it appears in a filename: "en" or "bg,cz,dk".
LANGS = r"[a-z]{2}(?:,[a-z]{2})*"
VARIANT = VARIANT_PATTERN

# Framing, as the readers name it. `FRAMING_FOR_VARIANT` maps a matched filename suffix
# onto that name; `FRAMING_ORIENTATION` carries the sign a framing implies, since the
# negated framing shows the model the opposite statement and so flips the stance.
FRAMING_FOR_VARIANT = VARIANT_LABELS
FRAMING_ORDER = ["base", "negated"]
# This also filters the judged CSVs, which carry the framing as a column rather than a
# filename suffix: the retired "question" framing is no longer generated, but judged
# files written before it was dropped still contain those rows.
FRAMING_ORIENTATION = {"base": 1.0, "negated": -1.0}

# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #

RESPONSES_CSV = re.compile(rf"^(?P<langs>{LANGS}){VARIANT}\.csv$")
SPEECHES_SCORED_CSV = re.compile(rf"^speeches_(?P<langs>{LANGS}){VARIANT}_scored\.csv$")
# The negative lookahead is what separates the two tracks: a reasons file is the one
# whose name does *not* start with the speeches prefix.
REASONS_CLASSIFIED_CSV = re.compile(rf"^(?!speeches_)(?P<langs>{LANGS}){VARIANT}_classified\.csv$")
SPEECHES_CLASSIFIED_CSV = re.compile(rf"^speeches_(?P<langs>{LANGS}){VARIANT}_classified\.csv$")

CHOICE_COLUMN = re.compile(rf"^choice_(?P<language>[a-z]{{2}}){VARIANT}_v(?P<paraphrase>\d+)$")
STANCE_COLUMN = re.compile(rf"^stance_(?P<language>[a-z]{{2}}){VARIANT}_v(?P<paraphrase>\d+)$")
PARTY_PROB_COLUMN = re.compile(
    rf"^party_prob_(?P<slug>.+?)_(?P<language>[a-z]{{2}}){VARIANT}_v(?P<paraphrase>\d+)$")
PREDICTED_PARTY_COLUMN = re.compile(rf"^predicted_party_[a-z]{{2}}{VARIANT}_v\d+$")

# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #


def responses_name(langs: str, variant: str = "") -> str:
    """The direct track's raw answers: `{langs}{variant}.csv`."""
    return f"{langs}{variant}.csv"


def speeches_name(langs: str, variant: str = "") -> str:
    """The indirect track's raw answers: `speeches_{langs}{variant}.csv`."""
    return f"speeches_{langs}{variant}.csv"


def scored_name(langs: str, variant: str = "", source: str = "speeches") -> str:
    """Stance recovered from prose, by `agreement_scoring.py`."""
    stem = speeches_name if source == "speeches" else responses_name
    return stem(langs, variant).removesuffix(".csv") + "_scored.csv"


def classified_name(langs: str, variant: str = "", source: str = "speeches") -> str:
    """EP-party predictions, by `classify_speeches.py`."""
    stem = speeches_name if source == "speeches" else responses_name
    return stem(langs, variant).removesuffix(".csv") + "_classified.csv"


def vaa_name(langs: str, variant: str = "", source: str = "reasons") -> str:
    """Agreement per EP group, by `evaluate_euandi.py`.

    Note the argument order flips: agreement files put the framing before the language
    set (`vaa_negated_en,de.csv`), the raw files after it (`en,de_negated.csv`).
    """
    prefix = "vaa_speeches" if source == "speeches" else "vaa"
    return f"{prefix}{variant}_{langs}.csv"


# The judge writes one file per track, with the framing as a column instead.
JUDGE_CSV = {"reasons": "reasons_llm_stance.csv", "speeches": "speeches_llm_stance.csv"}
JUDGE_COLUMNS = ["paraphrase", "language", "variant", "statement", "llm_stance"]

# Party slugs are upper-case, so the non-greedy slug in PARTY_PROB_COLUMN always
# splits correctly; this only restores the punctuation the slug dropped.
SLUG_TO_LABEL = {"ECR_ID": "ECR+ID", "GUE_NGL": "GUE/NGL", "Greens_EFA": "Greens/EFA", "S_D": "S&D"}
