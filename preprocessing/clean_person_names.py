"""
Delete titled person names, and stray leading punctuation, from speech text.

Used by clean_party_names.py (the `clean-names` step, which writes cleaned/):

    from preprocessing.clean_person_names import strip_names
    text = strip_names(text)

Why this exists
---------------
EP speeches name the people they answer -- "Mr Morillon's report", "as Commissioner
Füle said", "I thank Mr Barroso" -- mostly Commission figures and other MEPs, and a
name can leak which side the speaker is on. In dev_stance_mix_400.csv 22% of EP
speeches carry a title + name, against 0% of the LLM answers. Names are deleted, not
masked: LLM answers carry no placeholders either.

Forms of address are deliberately KEPT ("Mr President, ladies and gentlemen, ...",
"Dear colleagues", "Thank you, Commissioner"): only the name after a title goes.

Two passes, both language-independent (one pattern bank covers the 21 corpus
languages, so no language column is needed):

  lead_punct  stray punctuation a speech starts with (". Mr President, ...",
              "- I voted in favour"), left over from the source transcripts.
  name        a title followed by one to three capitalised words. After a courtesy
              title the title goes too ("the report by Mr Morillon" -> "the report
              by"); after an office word only the name goes ("Commissioner Füle said"
              -> "Commissioner said", "President Barroso" -> "President"). A few
              languages put the title after the name ("Barroso Mr"); those are
              handled too. Institution words and "European" are never taken for a
              name, so "President of the Council" and "Chair of the Committee on
              Fisheries" survive.

Bare names with no title ("Barroso said") are NOT removed; that would need NER. The
patterns were checked by hand on en/cs/sk only; the per-language counts that
clean_party_names.py prints are the check for the other languages.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Character classes. `re` has no \p{Lu}, so build the upper-case set from the  #
# Latin, Greek and Cyrillic ranges the corpus uses.                            #
# --------------------------------------------------------------------------- #
_UPPER = "".join(sorted({c for c in map(chr, range(0x0041, 0x0530)) if c.isupper()}))
UP = "[" + re.escape(_UPPER) + "]"
LOW = r"[^\W\d_]"                                   # any letter; used after UP
CAP = rf"(?-i:(?!{UP}+\b){UP}(?:{LOW}|['’\-]{LOW})+)"   # Capitalised word, not ALL-CAPS


def _alt(words):
    return "(?:%s)" % "|".join(words)


# --------------------------------------------------------------------------- #
# Word banks, in all 21 corpus languages.                                      #
# --------------------------------------------------------------------------- #
# Courtesy titles: "Mr", "Mrs", "Madam". Abbreviations are case-sensitive, so a
# sentence ending in "on." before a capitalised word, or "DNA" in capitals, is not
# read as a title.
HONORIFIC = _alt([
    r"mr", r"mrs", r"ms", r"mister", r"madam", r"madame", r"mesdames", r"messieurs", r"monsieur",
    r"herrn?", r"frau",
    r"signor[ae]?", r"onorevole", r"señor[a]?", r"senhor[a]?",
    r"mijnheer", r"mevrouw", r"heer", r"fru",
    r"herra", r"rouva", r"κύρι[εο]ς?", r"κυρία",
    r"pan[aieuy]?", r"panem", r"panią", r"pane", r"paní", r"pán", r"pána", r"pánovi", r"pánom", r"pani",
    r"gospod", r"gospa", r"gospe", r"gospoda", r"gospodine", r"gospođo", r"gospodin", r"gospođa",
    r"domnule", r"domnul", r"domnului", r"doamnă", r"doamna", r"doamnei",
    r"härra", r"proua", r"pone", r"ponia", r"ponas",
    r"господин", r"госпожо", r"госпожа", r"г-н", r"г-жо", r"г-жа", r"gđ[aeo]",
    r"(?-i:Mme|M\.|Sig\.(?:ra)?|On\.|Sr\.(?:ª|a)?|Sra\.|Hr\.|hr\.|Dl\.?|dl\.?|Dna\.?|dna\.?|"
    r"dlui|dnei|p\.|g\.|ga\.|κ\.|Dr)",
    r"mr\.", r"mrs\.", r"ms\.", r"dr\.",
])
# Titles that follow the name (Hungarian, Latvian).
POST_HONORIFIC = _alt([r"úr", r"urat", r"úrnak", r"úrral", r"asszony\w*",
                       r"kungs", r"kungam", r"kunga", r"kundze", r"kundzei", r"kundzes"])

# The Presidency (an institution) is not an office holder; keep it out of OFFICE.
_NOT_PRESIDENCY = (r"(?!presiden(?:c|z|t?ship)|presidên|předsednictv|predsedníctv|präsidentschaft|"
                   r"présidence|voorzitterschap|ordförandeskap|prezydencj|preşedinţi|președinți|"
                   r"puheenjohtaja?kau)")
# Office words: "President", "Commissioner", "Prime Minister", "High Representative".
OFFICE = _NOT_PRESIDENCY + _alt([
    # president / chair / speaker / prime minister
    r"(?:vice-?)?president\w*", r"(?:vize)?präsident\w*", r"(?:vice-)?président\w*",
    r"(?:vice)?presidente\w*", r"(?:vice)?presidenta", r"voorzitter\w*", r"talman\w*",
    r"ordförande\w*", r"formand\w*", r"puhemie\w*", r"(?:αντι)?πρόεδρ\w*",
    r"(?:wice)?przewodnicząc\w*", r"(?:místo)?předsed\w*", r"(?:pod)?predsed\w*",
    r"predsjedni\w*", r"pre[sşș]edin\w*", r"(?:al)?elnök\w*", r"juhataja\w*",
    r"priekšsēdētāj\w*", r"pirminink\w*", r"председател\w*", r"chair\w*",
    r"premi[eé]r\w*", r"premierminister\w*", r"pääministeri\w*", r"statsminister\w*",
    r"πρωθυπουργ\w*", r"miniszterelnök\w*", r"peaminist\w*", r"taoiseach",
    # commissioner
    r"commissioner\w*", r"kommissar\w*", r"commissaire\w*", r"commissari[oa]", r"comisari[oa]",
    r"comissári[oa]", r"commissaris\w*", r"kommissionär\w*", r"kommissær\w*", r"komissaari\w*",
    r"επίτροπ\w*", r"komisarz\w*", r"komisař\w*", r"komisár\w*", r"komisar\w*", r"povjeren\w*",
    r"comisar\w*", r"biztos\w*", r"volinik\w*", r"komisār\w*", r"комисар\w*",
    r"kommissionsledamot\w*", r"jäsen\w*",
    # minister / high representative / in office
    r"minist[er]\w*", r"ministr\w*", r"υπουργ\w*", r"miniszter\w*", r"министр?\w*",
    r"high", r"hoher?", r"haute?", r"alt[oa]", r"hoge", r"vysok\w*", r"wysok\w*",
    r"representative", r"vertreter\w*", r"représentant\w*", r"rappresentant\w*",
    r"representante\w*", r"vertegenwoordiger\w*", r"představitel\w*", r"predstavite\w*",
    r"przedstawiciel\w*", r"in-office", r"amtierende\w*", r"úřadující\w*", r"úradujúc\w*",
    r"urzędując\w*", r"fungerend\w*", r"tjänstgörande", r"exercício", r"ejercicio",
    r"exercice", r"carica",
])

# Words that are capitalised in forms of address but are never a person's name:
# the audience ("Colleagues", "Members"), courtesy adjectives ("Dear", "Honourable"),
# institutions, committees and "European" ("Chair of the Committee on Fisheries").
NOT_A_NAME = _alt([
    # audience
    r"colleagues?", r"collègues?", r"colleghi", r"colegas?", r"collega['’]?s", r"kolleg(?:er|a)",
    r"kollegat", r"kollegor", r"συνάδελφο\w*", r"koleżank\w*", r"koled\w*", r"koleg\w*", r"kolegyn\w*",
    r"colegi", r"kollégá\w*", r"képviselőtárs\w*", r"kolleegid", r"kolēģ\w*", r"колеги",
    r"kolleg(?:inn)?en", r"kollege\w*", r"kollegin\w*",
    r"damen", r"herren", r"ladies", r"gentlemen", r"signore", r"signori", r"señoras",
    r"señores", r"señorías?", r"senhoras", r"senhores", r"deputad[oa]s", r"dames", r"heren",
    r"damer", r"herrar", r"herrer", r"ledamöter", r"κυρίες", r"κύριοι", r"państwo", r"panie",
    r"panowie", r"posłowie", r"posłank\w*", r"dámy", r"pánové", r"páni", r"poslanc\w*",
    r"poslankyn\w*", r"dame", r"gospodo", r"doamnelor", r"domnilor", r"hölgyeim", r"uraim",
    r"daamid", r"härrad", r"dāmas", r"kungi", r"ponios", r"ponai", r"дами", r"господа",
    r"members", r"fellow", r"mep\w*", r"eurodeput\w*", r"europoslanc\w*", r"parlamentsledamöter",
    # courtesy adjectives
    r"dear", r"honou?rable", r"esteemed", r"distinguished", r"respected",
    r"sehr", r"geehrte\w*", r"liebe[rns]?", r"verehrte\w*",
    r"ch[eè]re?s?", r"estimad[oa]s?", r"querid[oa]s?", r"egregi[oa]", r"gentil[ei]",
    r"car[oaie]", r"illustr[ei]", r"onorevol[ei]", r"excelent[ií]ss?im[oa]s?", r"caros?",
    r"beste", r"geachte", r"waarde", r"bäste", r"ärade", r"kære", r"ærede",
    r"arvoisa\w*", r"hyvä\w*", r"αξιότιμ\w*", r"αγαπητ\w*", r"σεβαστ\w*",
    r"szanown\w*", r"drodzy", r"drog[ai]", r"vážen\w*", r"mil[ií]", r"milá",
    r"spoštovan\w*", r"poštovan\w*", r"stimat\w*", r"dragi", r"drag[ăa]",
    r"tisztelt", r"kedves", r"lugupeetud", r"austatud", r"godājam\w*", r"cienījam\w*",
    r"gerbiam\w*", r"mieli", r"brangūs", r"уважаем\w*", r"скъп\w*",
    # institutions
    r"council\w*", r"commission\w*", r"parliament\w*", r"rady", r"rada", r"radě", r"komis[ei]\w*",
    r"parlament\w*", r"conseil\w*", r"consiglio", r"consejo", r"conselho", r"raad", r"råd\w*",
    r"neuvoston", r"komission", r"συμβουλ\w*", r"επιτροπ\w*", r"sveta", r"consiliului",
    r"tanács\w*", r"bizottság\w*", r"nõukogu", r"komisjon\w*", r"padomes", r"komisijas",
    r"tarybos", r"komisijos", r"съвета", r"комисия\w*", r"comisi[oó]n\w*", r"comissão",
    r"commissione", r"komisj\w*", r"komisij\w*", r"eu", r"union\w*", r"unii", r"unie",
    r"europ\w*", r"evrop\w*", r"európ\w*", r"eiropas", r"europos", r"ευρωπ\w*", r"европ\w*",
    r"house", r"hemicycle", r"group\w*", r"skupin\w*",
    # committees and delegations
    r"committee\w*", r"výbor\w*", r"ausschuss\w*", r"comité\w*", r"comitato", r"commissie",
    r"utskott\w*", r"udvalg\w*", r"valiokun\w*", r"odbor\w*", r"komitet\w*", r"komitej\w*",
    r"комитет\w*", r"delegac\w*", r"delegation\w*", r"délégation\w*",
])


def _w(*banks):
    """Whole-word, case-insensitive match of any bank. (?!\\w), not \\b, closes it:
    after the dot of "Mr." there is no word boundary to find."""
    return rf"(?i:\b(?:{'|'.join(banks)})(?!\w))"


# --------------------------------------------------------------------------- #
# Names are found word by word, not with one big regex: a bank alternation     #
# tried at every position of every text ran at ~300 rows/s. Each distinct word #
# is classified once (cached), and words repeat, so the loop is cheap.         #
# --------------------------------------------------------------------------- #
# An elided article is its own token, so a title glued to it is still seen.
_TOKEN_RE = re.compile(r"(?i:\b(?:[ldmtsn]|dell|all|dall|nell|sull|qu)['’])(?=[^\W\d_])|\S+")
# word -> (core, possessive, trailing punctuation): "Morillon's," -> Morillon, 's, ","
_SPLIT_RE = re.compile(r"(?P<core>.*?)(?P<poss>['’]s|['’])?(?P<punct>[,.;:!?)\]»”\"'’…]*)", re.S)
_HON_FULL = re.compile(_w(HONORIFIC))
_OFFICE_FULL = re.compile(_w(OFFICE))
_POST_FULL = re.compile(_w(POST_HONORIFIC))
_BANK_FULL = re.compile(_w(HONORIFIC, OFFICE, POST_HONORIFIC, NOT_A_NAME))
_CAP_FULL = re.compile(CAP)
# Name particles: "De Gucht", "van den Broek".
_PARTICLES = {"de", "da", "di", "del", "della", "dos", "das", "du", "van", "von", "der", "den",
              "ten", "ter", "le", "la", "zu", "De", "Van", "Von", "Le", "La", "Di", "Da", "Del"}
# The German titles for Mr/Mrs are also the nouns "gentleman"/"woman": after a
# determiner ("every woman ...") the next capitalised word is a German noun, not a name.
_GERMAN_TITLES = {"frau", "herr", "herrn"}
_GERMAN_DET_RE = re.compile(r"(?:[Dd](?:ie|er|en|em|es)|[Ee]ine[mnrs]?|[Jj]ede[mnrs]?|[Kk]eine[mnrs]?|"
                            r"[Dd]iese[mnrs]?|[Mm]eine[mnrs]?|[Ss]eine[mnrs]?|[Ii]hre[mnrs]?|jung\w*|alt\w*)")
# German capitalises nouns, so "the President of France" is an office word followed by
# a capitalised genitive ending in -s; that is not a name.
_GERMAN_OFFICE_RE = re.compile(r"(?i:(?:vize)?präsident\w*|kommissar\w*|minister\w*|kanzler\w*)")


@lru_cache(maxsize=500_000)
def _kind(token: str) -> str:
    """'hon', 'office', 'post', 'name' or '' for one whitespace-delimited word as-is."""
    if _HON_FULL.fullmatch(token):
        return "hon"
    if _OFFICE_FULL.fullmatch(token):
        return "office"
    if _POST_FULL.fullmatch(token):
        return "post"
    if _CAP_FULL.fullmatch(token) and not _BANK_FULL.fullmatch(token):
        return "name"
    return ""


def _split(token: str):
    m = _SPLIT_RE.fullmatch(token)
    return m.group("core"), m.group("poss") or "", m.group("punct")


def _name_spans(text: str, counts: Counter) -> list[tuple[int, int]]:
    """Character spans to delete: titled names, and names before a post-title."""
    toks = [(m.start(), m.group()) for m in _TOKEN_RE.finditer(text)]
    n, spans, i = len(toks), [], 0
    while i < n:
        kind = _kind(toks[i][1])
        if kind in ("hon", "office"):
            j = i                                  # the run of titles: "Madam Commissioner"
            while j + 1 < n and _kind(toks[j + 1][1]) in ("hon", "office"):
                j += 1
            k, names, end = j + 1, 0, None
            while k < n and names < 3:
                core, poss, punct = _split(toks[k][1])
                if names and toks[k][1] in _PARTICLES and k + 1 < n \
                        and _kind(_split(toks[k + 1][1])[0]) == "name":
                    k += 1                         # "van den Broek"
                    continue
                if _kind(core) != "name":
                    break
                names += 1
                end = toks[k][0] + len(core) + len(poss)
                k += 1
                if punct:
                    break
            if names:
                h = j                              # trailing courtesy titles go with the name
                while h >= i and _kind(toks[h][1]) == "hon":
                    h -= 1
                if h < j:
                    first = h + 1
                    if toks[first][1].lower() in _GERMAN_TITLES and first > 0 \
                            and _GERMAN_DET_RE.fullmatch(toks[first - 1][1]):
                        i = k
                        continue
                    if toks[first][1].lower() == "heer" and first > 0 and toks[first - 1][1].lower() == "de":
                        first -= 1                 # the Dutch title is two words ("the Mr")
                    start = toks[first][0]
                elif names == 1 and _GERMAN_OFFICE_RE.fullmatch(toks[j][1]) \
                        and _split(toks[j + 1][1])[0].endswith("s"):
                    i = k
                    continue
                else:
                    start = toks[j + 1][0]
                spans.append((start, end))
                counts["name"] += 1
                i = k
                continue
            i = j + 1
            continue
        # Title after the name: "Barroso Mr", "Barroso President Mr".
        if kind == "name" and i + 1 < n:
            nxt_core = _split(toks[i + 1][1])[0]
            if _kind(nxt_core) == "post":
                spans.append((toks[i][0], toks[i + 1][0] + len(nxt_core)))
                counts["name"] += 1
                i += 2
                continue
            if _kind(toks[i + 1][1]) == "office" and i + 2 < n \
                    and _kind(_split(toks[i + 2][1])[0]) == "post":
                spans.append((toks[i][0], toks[i][0] + len(toks[i][1])))
                counts["name"] += 1
        i += 1
    return spans


# Stray punctuation a speech starts with (".\nMr President", "- I voted"). Never an
# opening quote or bracket, and never "*": markdown bold ("**Against.**") starts with it.
LEAD_PUNCT_RE = re.compile(r"^[\s.,;:!?–—\-−•·]+(?=\w)")
_WS_RE = re.compile(r"[^\S\n]{2,}")
_SPACE_PUNCT_RE = re.compile(r"[^\S\n]+([,.;:!?])")
_DOUBLE_COMMA_RE = re.compile(r",(?:[^\S\n]*,)+")
_COMMA_BEFORE_END_RE = re.compile(r",[^\S\n]*(?=[.!?;:])")   # "Mrs Klingvall, Mr Bodström!" -> ",!" -> "!"


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text[:1].islower() else text


def strip_names(text: str, counts: Counter | None = None, lead_punct: bool = True) -> str:
    """Delete stray leading punctuation and titled person names from `text`.

    `counts`, if given, is incremented under "lead_punct" and "name". Text with
    nothing to remove is returned unchanged (byte-identical). lead_punct=False removes
    names only (analysis/name_ablation.py isolates the effect of names that way).
    """
    if not text:
        return text
    counts = counts if counts is not None else Counter()
    new = text

    m = LEAD_PUNCT_RE.match(new) if lead_punct else None
    if m:
        counts["lead_punct"] += 1
        new = new[m.end():]

    spans = _name_spans(new, counts)
    if spans:
        parts, pos = [], 0
        for start, end in spans:
            if start < pos:
                continue
            parts.append(new[pos:start])
            # judged on the output so far, so a name right after a deleted one still
            # counts as starting the sentence ("Mrs Klingvall, Mr Bodström! We ...")
            before = "".join(parts).rstrip().rstrip("\"”»)")
            # "... late. Mr Morillon's report is ..." -> "... late. Report is ...", and
            # "... late. Mr Brok, we ..." / "Mr Brok! We ..." -> "We ...": no orphaned
            # comma or exclamation mark at the start of the sentence.
            if not before or before[-1] in ".!?":
                rest = new[end:].lstrip(" \t,;:!?.")
                pos = len(new) - len(rest)
                if rest[:1].islower():
                    parts.append(rest[0].upper())
                    pos += 1
            else:
                pos = end
        parts.append(new[pos:])
        cleaned = "".join(parts)
        # A text that was nothing but names ("Mrs Klingvall, Mr Bodström!") is kept.
        new = cleaned if re.search(r"[^\W\d_]", cleaned) else new

    if new != text:
        new = _DOUBLE_COMMA_RE.sub(",", new)
        new = _COMMA_BEFORE_END_RE.sub("", new)
        new = _WS_RE.sub(" ", new)
        new = _SPACE_PUNCT_RE.sub(r"\1", new)
        new = re.sub(r"^[\s,;:]+", "", new)
        new = _capitalise(new)
    return new


if __name__ == "__main__":
    # python preprocessing/clean_person_names.py "Mr President, Mr Barroso's plan ..."
    for arg in sys.argv[1:]:
        print(strip_names(arg))
