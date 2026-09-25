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

The repair: "<U+XXXX>" escapes become the character they name; a letter + spacing
accent becomes the single precomposed accented letter -- only where one exists, so
the number sign "n° 5" is never read as an n with a ring.

When it applies
---------------
Escapes are always decoded: the text "<U+030C>" is never meant literally. Spacing
accents are converted only in a row that shows the damage -- an escape, or at least
two convertible accents -- and never where "´" or "`" is an apostrophe: English
endings ("the State´s view", "don´t") and elision ("l´Europe", "c´è"), which
turning into accents would break.
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
# A spacing accent glued to the end of a letter; a following letter is not required
# ("EU´" is a word-final accent).
_ACCENT_RE = re.compile(rf"([^\W\d_])([{_ACCENTS}])")
# "´" and "`" used as an apostrophe rather than an accent:
#   English endings  "the State´s", "don´t", "we´re"
#   elision          "l´Europe", "dell´Unione", "c´è", "d´este" -- a short word-initial
#                    prefix, then a vowel or h
_ENGLISH_APOSTROPHE_RE = re.compile(r"(?:s|t|re|ve|ll|d|m)\b")
_ELISION_PREFIX_RE = re.compile(r"(?i:\b(?:[ldjnmstc]|qu|dell|all|dall|nell|sull|coll|tutt|un|quell|"
                                r"anch|buon|nessun|ciascun|dev|jusqu|lorsqu|puisqu|quoiqu))$")
_VOWEL_OR_H_RE = re.compile(r"(?i:[aeiouyhàáâäãåæèéêëìíîïòóôöõøùúûü])")


def _composed(letter: str, accent: str) -> str | None:
    """The single precomposed letter for letter + accent, or None if there is none
    ("n°" is a number sign, not an n with a ring)."""
    c = unicodedata.normalize("NFC", letter + _COMBINING[accent])
    return c if len(c) == 1 else None


def _is_elision(text: str, m: re.Match) -> bool:
    """A "´"/"`" after a short word-initial prefix and before a vowel: an apostrophe."""
    return bool(m.group(2) in "´`"
                and _VOWEL_OR_H_RE.match(text[m.end():m.end() + 1])
                and _ELISION_PREFIX_RE.search(text[max(0, m.end() - 9):m.end() - 1]))


def _repairs(text: str) -> tuple[list[tuple[re.Match, str]], int]:
    """(every letter + spacing accent that forms a real accented letter, except elision;
    how many of them are also not an English apostrophe ending).

    The second number decides whether a row is damaged at all. It leaves out "´s",
    "´t", "´m", ... because an undamaged English text may use "´" as its apostrophe;
    once a row is judged damaged, those convert too -- in damaged text they are accents.
    """
    out, evidence = [], 0
    for m in _ACCENT_RE.finditer(text):
        c = _composed(m.group(1), m.group(2))
        if c and not _is_elision(text, m):
            out.append((m, c))
            if not (m.group(2) in "´`" and _ENGLISH_APOSTROPHE_RE.match(text, m.end())):
                evidence += 1
    return out, evidence


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
    repairs, evidence = _repairs(new)
    if repairs and (new != text or evidence >= 2):
        parts, pos = [], 0
        for m, c in repairs:
            parts.append(new[pos:m.start()])
            parts.append(c)
            pos = m.end()
        parts.append(new[pos:])
        new = unicodedata.normalize("NFC", "".join(parts))
    if new != text and counts is not None:
        counts["diacritics"] += 1
    return new
