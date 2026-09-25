"""
Repair speeches whose accents were stored as separate characters.

Used by clean_party_names.py (the `clean-names` step, which writes cleaned/), before
any name matching, so the name patterns see the repaired words:

    from preprocessing.fix_diacritics import fix_diacritics
    text = fix_diacritics(text)

What the damage looks like
--------------------------
Some source transcripts store every accented letter as the bare letter followed by a
separate accent character, and some accents as R's escape text for a code point:

    de´cision      (acute as a spacing U+00B4)       -> décision
    Voila`         (grave as a backtick)             -> Voilà
    mo^z<U+030C>u  (circumflex as a caret, caron as the literal text "<U+030C>")
    mu°z<U+030C>e  (ring above as a degree sign)

In dev_stance_mix_400.csv 11 of 280 EP speeches look like this; the classifier sees
such a speech as a string of broken tokens that no LLM answer ever contains.

The repair: "<U+XXXX>" escapes become the character they name; a spacing accent right
after a letter becomes the matching combining accent; then the text is NFC-normalised,
so letter + combining accent turns into the single precomposed letter.

When it applies
---------------
Escapes are always decoded: the text "<U+030C>" is never meant literally. Spacing
accents are converted only in a row that shows the damage -- an escape, or at least
two accents that are not an English apostrophe ending -- because some speeches use
"´" as an apostrophe ("the State´s view", "don´t"), and turning those into accents
would break correct text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_ESCAPE_RE = re.compile(r"<U\+([0-9A-Fa-f]{4,6})>")

# Spacing accent -> combining accent.
_COMBINING = {
    "´": "́",   # acute
    "`": "̀",   # grave
    "^": "̂",   # circumflex
    "ˆ": "̂",
    "˜": "̃",   # tilde
    "¨": "̈",   # diaeresis
    "°": "̊",   # ring above
    "˚": "̊",
    "ˇ": "̌",   # caron
    "˘": "̆",   # breve
    "˙": "̇",   # dot above
    "˝": "̋",   # double acute
    "¸": "̧",   # cedilla
    "˛": "̨",   # ogonek
}
_ACCENTS = re.escape("".join(_COMBINING))
# A spacing accent glued to the end of a letter. The letter is kept, the accent
# replaced; a following letter is not required ("EU´" is a word-final accent).
_ACCENT_RE = re.compile(rf"(?<=[^\W\d_])[{_ACCENTS}]")
# The same, minus the English apostrophe endings "´s", "´t", "´re", "´ve", "´ll", "´d", "´m".
_EVIDENCE_RE = re.compile(rf"(?<=[^\W\d_])(?:[{_ACCENTS}](?!(?:s|t|re|ve|ll|d|m)\b))")


def fix_diacritics(text: str, counts: Counter | None = None) -> str:
    """Repair detached accents in `text`; unchanged text is returned byte-identical.

    `counts`, if given, is incremented under "diacritics" once per repaired row.
    """
    if not text:
        return text
    new = text
    if "<U+" in new:
        new = _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), new)
        # Compose the decoded marks first, so "s" + caron is "š" before the accent test
        # below looks at what follows each spacing accent.
        new = unicodedata.normalize("NFC", new)
    if new != text or len(_EVIDENCE_RE.findall(new)) >= 2:
        new = _ACCENT_RE.sub(lambda m: _COMBINING[m.group(0)], new)
        new = unicodedata.normalize("NFC", new)
    if new != text and counts is not None:
        counts["diacritics"] += 1
    return new
