"""
Strip European Parliament political-group / party names out of the speech text,
after repairing detached accents (fix_diacritics.py), then titled person names and
stray leading punctuation (clean_person_names.py).

Reads  data/EuroParl Custom/{train,dev,test}.parquet
Writes data/EuroParl Custom/cleaned/{train,dev,test}.parquet
and prints, per split, how many name occurrences were removed for each of the
seven canonical EP groups, and per language how many rows had their accents
repaired and how many titled names ("Mr Morillon") and stray leading punctuation
marks were removed. The `cleaned/` subdirectory, same filenames, is what
build_collapsed_splits.py and analysis/plotting/plot_speech_counts.py read.

Why this exists
---------------
The `EU Party` column is the label we classify on. If a speech literally names
its own group ("on behalf of the PPE-DE Group", "Fraktion der Sozialdemokraten",
"groupe des Verts", ...) a classifier can cheat off that string instead of
learning ideology. This pass regex-removes those self-identifying names.

Patterns are organised under the SEVEN canonical groups that appear in the data
(PPE, S&D, ALDE, Greens/EFA, ECR, GUE/NGL, ID). Historical predecessors fold
into their modern successor for counting:

    ELDR / Renew Europe        -> ALDE
    PSE / PES                  -> S&D
    PPE-DE / EPP-ED            -> PPE
    ENF / EFD / EFDD / ITS     -> ID
    UEN / IND/DEM              -> ID   (2004-09 groups; counted here, not mapped)

A few NATIONAL parties are removed too ("national" in the report): the ones whose
MEPs name themselves routinely -- KKE's explanations of vote open with "The
Communist Party of Greece voted against", UKIP's with "UKIP MEPs" -- which made
the party name a near-perfect label proxy (KKE: 88/88 test rows Radical left, UKIP
120/122 Sovereigntist right). The list is short and measured, not exhaustive.

Three precision tiers, chosen so we delete *names*, not ordinary vocabulary:

  acr  - acronyms, matched case-SENSITIVE with word boundaries. They are
         language-independent (free coverage in all 21 languages) and case
         sensitivity stops "argue"/"happen"/"types" matching GUE/PPE/PES.

  name - distinctive MULTI-WORD proper names (and unambiguous compounds like
         "Volkspartei" / "Linksfraktion"), matched case-INSENSITIVE, translated
         into the 21 corpus languages. Multi-word names are not ordinary
         vocabulary, so lower-case matching is safe.

  anch - AMBIGUOUS single words ("socialist", "liberal", "green", "left", ...)
         in many languages. These are removed ONLY when they sit immediately
         next to a word meaning "group" (Fraktion / groupe / grupo / ομάδα /
         група / ...), in either order, separated by at most two short function
         words and no comma. This catches "Socialist Group" / "groupe des Verts"
         without ever deleting the ideological vocabulary from ordinary
         sentences.

  raw  - hand-written regexes, case-SENSITIVE: capitalised plurals that only
         ever name the group ("the Greens", "Zelení"), acronyms that are only
         safe next to a group word ("groupe ID", "ID-Fraktion"), and orphaned
         halves of slash names ("/ALE").

Spaces inside name patterns match any whitespace run: the cs/sk/fr translations
write "socialistov a\xa0demokratov" with a no-break space, which a literal space
never matched. Czech/Slovak/Polish names are written as stems ("Evropsk\\w*
sjednocen\\w* levic\\w*"), because the nominative form alone missed every other
case ("Skupina konfederace Evropské sjednocené levice").

After the names go, their leftovers go too (see strip_residue): empty brackets
"( )", orphaned "-Fraktion", and the transcript header before " – " ("au nom du
groupe ID. –", "on behalf of the Group. –", "in writing. –"), which is session
metadata, not speech, and whose shape still told the groups apart.

The per-group report makes any coverage gaps visible; extend the dicts to close
them. Set REPLACEMENT to e.g. "[GROUP]" if you would rather mask than delete.
"""

from __future__ import annotations

import gc
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:                                    # run as a script from the repo root ...
    from preprocessing.clean_person_names import strip_names
    from preprocessing.fix_diacritics import fix_diacritics, is_transliterated
except ImportError:                     # ... or from inside preprocessing/
    from clean_person_names import strip_names
    from fix_diacritics import fix_diacritics, is_transliterated

DATA_DIR = Path("data/EuroParl Custom")
OUT_DIR = DATA_DIR / "cleaned"      # what build_collapsed_splits.py reads
SPLITS = ["train", "dev", "test"]
TEXT_COL = "text"
PAR_CHUNK = 4_000            # rows per task handed to a worker process
N_WORKERS = max(1, (os.cpu_count() or 2) - 2)   # leave a couple cores for I/O
REPLACEMENT = " "            # what a removed name becomes (whitespace then collapsed)
# Titled person names ("Mr Morillon") stay in the text. Removing them was tested: on the
# rows that have one, a test-time ablation cost -3.4pp accuracy against -2.3pp for
# deleting as many random words (analysis/name_ablation.py), so the name-specific cue
# is ~1pp -- and the title patterns cover the corpus languages unevenly, which matters
# more for a multilingual instrument than that cue. True brings the removal back.
REMOVE_PERSON_NAMES = False

CANON = ["PPE", "S&D", "ALDE", "Greens/EFA", "ECR", "GUE/NGL", "ID", "national"]

# --------------------------------------------------------------------------- #
# Word meaning "(political) group" in each of the 21 corpus languages.         #
# Used only to anchor the ambiguous single ideology words ("anch").            #
# --------------------------------------------------------------------------- #
GROUP_WORD = [
    r"group", r"groupe", r"gruppo", r"grupo", r"grupa", r"grupul", r"grupului",
    r"grupas", r"grupai", r"grupp\w*", r"gruppe\w*", r"fraktion\w*", r"fractie",
    r"fraktsioon\w*", r"frakcij\w*", r"ryhm\w*", r"skupin\w*",
    r"képviselőcsoport\w*", r"csoport\w*", r"rühm\w*",
    r"ομάδ\w*",      # Greek  (omada)
    r"груп\w*",      # Cyrillic (grup-)
]
GW = r"(?:%s)" % "|".join(GROUP_WORD)

# at most two short (<=4-letter) function words between group-word and ideology
# word, but never a coordinating conjunction (which would signal a *second*
# group) and never across a comma/number.
_CONJ = r"(?:and|und|et|och|og|ja|y|e|i|és|ir|и|και)"
_CONN = rf"(?:\s+(?!{_CONJ}\b)[^\W\d_]{{1,4}}){{0,2}}"


# --------------------------------------------------------------------------- #
# Pattern bank, per canonical group.                                           #
# --------------------------------------------------------------------------- #
PATTERNS: dict[str, dict[str, list[str]]] = {
    "PPE": {
        # NB: only EUROPEAN-qualified "People's Party" forms are listed. Bare
        # "People's Party" / "Partido Popular" / "Volkspartij" (Dutch VVD is
        # *liberal*) collide with unrelated national/foreign parties, so they are
        # deliberately excluded. The Christian-Democrat label is firmly EPP.
        "acr": ["PPE-DE", "EPP-ED", "PPE", "EPP"],
        "name": [
            r"European People['’]?s Party",
            r"Christian[\s\-]?Democrat\w*", r"European Democrats",
            r"Europäische Volkspartei", r"Christdemokrat\w*",
            r"christlich[\s\-]?demokratische\w*",
            r"Parti populaire européen",
            r"démocrate[s]?[\s\-]chrétien\w*", r"chrétiens[\s\-]démocrates",
            r"Partito popolare europeo", r"democratici cristiani",
            r"Partido Popular Europeo", r"demócrata[s]?[\s\-]cristiano\w*",
            r"Partido Popular Europeu", r"democratas[\s\-]cristãos",
            r"Europese Volkspartij", r"christen[\s\-]?democraten",
            r"Europeiska folkpartiet", r"kristdemokrat\w*",
            r"Det Europæiske Folkeparti", r"kristelige demokrater",
            r"Euroopan kansanpuolue", r"kristillisdemokraat\w*",
            r"Ευρωπαϊκό Λαϊκό Κόμμα",
            r"Europejskiej Partii Ludowej", r"Chrześcijańscy Demokraci",
            r"Evropsk\w* lidov\w* stran\w*", r"Křesťansk\w* demokrat\w*",
            r"Európsk\w* ľudov\w* stran\w*", r"Kresťansk\w* demokrat\w*",
            r"Europejsk\w* Parti\w* Ludow\w*",
            r"evropsk\w* lidovc\w*", r"európsk\w* ľudovc\w*",   # "evropští lidovci"
            r"Európai Néppárt", r"Kereszténydemokrat\w*",
            r"Euroopa Rahvapartei",
            r"Evropska ljudska stranka",
            r"Eiropas Tautas partija",
            r"Europos liaudies partija",
            r"Partidul Popular European",
            r"Европейската народна партия",
        ],
        "anch": [],
    },
    "S&D": {
        "acr": ["S&D", "PSE", "PES"],
        "name": [
            r"Progressive Alliance of Socialists and Democrats",
            r"Socialists and Democrats", r"Party of European Socialists",
            r"Alliance Progressiste des Socialistes", r"socialistes et démocrates",
            r"Parti socialiste européen",
            r"Alleanza Progressista di Socialisti", r"socialisti e democratici",
            r"Partito del socialismo europeo",
            r"Alianza Progresista de Socialistas", r"socialistas y demócratas",
            r"Partido Socialista Europeo",
            r"socialistas e democratas", r"Partido Socialista Europeu",
            r"Progressieve Alliantie van Socialisten", r"socialisten en democraten",
            r"Σοσιαλιστών και Δημοκρατών",
            r"Socjalistów i Demokratów",
            r"socialist\w* a demokrat\w*",                     # cs/sk, any case
            r"Szocialisták és Demokraták",
            r"socialistų ir demokratų",
            r"Alianța Progresistă a Socialiștilor", r"Socialiștilor și Democraților",
            r"Социалисти и демократи",
        ],
        # ambiguous: only next to a group word
        "anch": [
            r"socialist\w*", r"social[\s\-]?democrat\w*", r"sozialdemokrat\w*",
            r"sozialistische\w*", r"socialiste\w*", r"socialdemokrat\w*",
            r"sosialidemokraat\w*", r"sosiaalidemokraat\w*",
            r"socjalist\w*", r"socjaldemokrat\w*", r"sociálnedemokrat\w*",
            r"szociáldemokrat\w*", r"szocialist\w*",
            r"sotsiaaldemokraat\w*", r"sotsialist\w*", r"sociāldemokrāt\w*",
            r"σοσιαλιστ\w*", r"социалист\w*", r"социалдемократ\w*",
        ],
    },
    "ALDE": {
        "acr": ["ALDE", "ELDR"],
        "name": [
            r"Renew Europe",
            r"Alliance of Liberals and Democrats for Europe",
            r"Liberals and Democrats", r"Liberal Democrat and Reform",
            r"Allianz der Liberalen und Demokraten",
            r"Alliance des démocrates et des libéraux",
            r"Alleanza dei Liberali e dei Democratici",
            r"Alianza de los Liberales y Demócratas",
            r"liberalen en democraten",
            r"Libéraux et (?:des )?Démocrates",
            r"(?:Alianc\w* )?liberál\w* a demokrat\w*(?: (?:pro Evropu|za Európu))?",   # cs/sk
            r"Liberałów i Demokratów", r"Liberálisok és Demokraták",
            r"Liberaalide ja Demokraatide", r"Liberāļu un demokrātu",
            r"Liberalų ir demokratų", r"Liberalilor și Democraților",
            r"Либерали и демократи",
        ],
        "anch": [
            r"liberal\w*", r"libéra\w*", r"liberaal\w*", r"liberál\w*",
            r"λιμπερ\w*", r"λιβερ\w*", r"либерал\w*", r"liberał\w*",
            r"renew",  # "Renew Group" / "groupe Renew" (only next to a group word)
        ],
    },
    "Greens/EFA": {
        "acr": ["Greens/EFA", "Verts/ALE", "G/EFA"],
        "name": [
            r"European Free Alliance",
            r"Greens[\s/–\-]+European Free Alliance", r"Greens[/–\-]EFA",
            r"Freie Europäische Allianz", r"Alliance libre européenne",
            r"Alleanza libera europea", r"Alianza Libre Europea",
            r"Aliança Livre Europeia", r"Vrije Europese Alliantie",
            r"Europeiska fria alliansen", r"Euroopan vapaa allianssi",
            r"Ελεύθερη Ευρωπαϊκή Συμμαχία", r"Wolne Przymierze Europejskie",
            r"Evropsk\w* svobodn\w* alianc\w*", r"Európsk\w* slobodn\w* alianci\w*",
            r"Európai Szabad Szövetség", r"Euroopa Vabaliit",
            r"Evropska svobodna zveza", r"Eiropas Brīvā apvienība",
            r"Europos laisvojo aljanso", r"Alianța Liberă Europeană",
            r"Европейски свободен съюз",
        ],
        "anch": [
            r"green\w*", r"grün\w*", r"vert\w*", r"verd\w*", r"groen\w*",
            r"grön\w*", r"grøn\w*", r"vihre\w*", r"zielon\w*", r"zelen\w*",
            r"zöld\w*", r"roheli\w*", r"zaļ\w*", r"žali\w*",
            r"πρασιν\w*", r"πράσιν\w*", r"зелен\w*",
        ],
        "raw": [
            # capitalised plurals: the group/party, never the colour
            r"\b(?:Greens|Verts|Grünen|Verdes|Groenen|Gröna|Grønne|Vihreät|"
            r"Zieloni|Zelení|Zelených|Zelenými|Zöldek)\b",
            r"/\s?(?:ALE|EFA)\b",               # "Verts/ALE" after "Verts" went
        ],
    },
    "ECR": {
        "acr": ["ECR"],
        "name": [
            r"European Conservatives and Reformists", r"Conservatives and Reformists",
            r"Europäische Konservative und Reformer",
            r"Conservateurs et Réformistes européens",
            r"Conservatori e Riformisti europei",
            r"Conservadores y Reformistas Europeos",
            r"Conservadores e Reformistas Europeus",
            r"Conservatieven en Hervormers",
            r"Ευρωπαίων Συντηρητικών και Μεταρρυθμιστών",
            r"Konserwatystów i Reformatorów",
            r"Konzervativců a reformistů", r"konzervatívcov a reformist\w*",
            r"Konzervatívok és Reformerek",
            r"Konservatiivide ja Reformistide", r"Konservatīvo un reformistu",
            r"Konservatorių ir reformistų",
            r"Conservatorilor și Reformiștilor", r"Консерватори и реформисти",
        ],
        "anch": [
            r"conservativ\w*", r"konservativ\w*", r"conservateur\w*",
            r"conservador\w*", r"conservatori\w*", r"konzervat\w*",
            r"konserwat\w*", r"konservatiiv\w*", r"konservatīv\w*",
            r"συντηρητ\w*", r"консерват\w*",
        ],
    },
    "GUE/NGL": {
        "acr": ["GUE/NGL", "GUE", "NGL"],
        "name": [
            r"European United Left[\s/–\-]*Nordic Green Left",
            r"European United Left", r"Nordic Green Left",
            r"The Left in the European Parliament",
            r"Vereinte Europäische Linke", r"Nordische Grüne Linke", r"Linksfraktion",
            r"Gauche unitaire européenne", r"Gauche verte nordique",
            r"Sinistra unitaria europea", r"Sinistra verde nordica",
            r"Izquierda Unitaria Europea", r"Izquierda Verde Nórdica",
            r"Esquerda Unitária Europeia", r"Esquerda Nórdica Verde",
            r"Verenigd Europees Links", r"Noords Groen Links",
            r"Europeiska enade vänstern", r"Nordisk grön vänster",
            r"Yhtyneen vasemmiston", r"Pohjoismaiden vihreä vasemmisto",
            r"Ενωτικής Αριστεράς",
            r"Zjednoczona Lewica Europejska", r"Nordycką Zieloną Lewicę",
            r"Evropsk\w* sjednocen\w* levic\w*", r"Seversk\w* zelen\w* levic\w*",
            r"Európsk\w* zjednoten\w* ľavic\w*", r"(?:Seversk|Nordick)\w* zelen\w* ľavic\w*",
            r"Zjednoczon\w* Lewic\w* Europejsk\w*", r"Nordyck\w* Zielon\w* Lewic\w*",
            # only GUE/NGL ever called itself "confederal"
            r"conf[eé]d[eé]ral\w*", r"konföderal\w*", r"konfederatívn\w*",
            r"skupin\w* konfederac\w*",
            r"Egységes Európai Baloldal", r"Északi Zöld Baloldal",
            r"Ühendatud Vasakliit", r"Apvienotā kreisā",
            r"Vieningosios kairiųjų", r"Stânga Unită Europeană",
            r"Европейска обединена левица",
        ],
        "anch": [
            r"left", r"linke\w*", r"links\w*", r"gauche", r"sinistra",
            r"izquierda", r"esquerda", r"vänster\w*", r"vasemmist\w*",
            r"αριστερ\w*", r"lewic\w*", r"levic\w*", r"ľavic\w*",
            r"baloldal\w*", r"vasak\w*", r"kreis\w*", r"kairi\w*",
            r"stâng\w*", r"stang\w*", r"левиц\w*", r"ляв\w*",
        ],
    },
    "ID": {
        "acr": ["EFDD", "EFD", "ENF", "ID Group", "ITS", "IND/DEM", "UEN"],
        "raw": [
            # bare "ID" is an identity document; only next to a group word is it the group
            r"\b(?:groupe|gruppo|grupo|grupy|grupa|grupul|skupin\w*|[Gg]roup|ομάδας)\s+ID\b",
            r"\bID(?=[\s\-]?(?:Fraktion|Fractie|[Gg]roup|[Gg]ruppe\w*|fraktion\w*|ryhm\w*))",
        ],
        "name": [
            r"Independence ?(?:/|and) ?Democracy", r"Ind[ée]pendance ?(?:/|et) ?(?:de la )?D[ée]mocratie",
            r"Unabhängigkeit ?(?:/|und) ?Demokratie",
            r"Union pour l['’]Europe des Nations", r"Union für das Europa der Nationen",
            r"Identity and Democracy",
            r"Europe of Freedom and Direct Democracy", r"Europe of Freedom and Democracy",
            r"Europe of Nations and Freedom", r"Union for Europe of the Nations",
            r"Identität und Demokratie", r"Europa der Nationen und der Freiheit",
            r"Europa der Freiheit und der direkten Demokratie",
            r"Identité et Démocratie", r"Europe des Nations et des Libertés",
            r"Europe de la liberté et de la démocratie directe",
            r"Identità e Democrazia", r"Europa delle Nazioni e della Libertà",
            r"Identidad y Democracia", r"Europa de las Naciones y de las Libertades",
            r"Identidade e Democracia",
            r"Identiteit en Democratie", r"Europa van Vrijheid en Directe Democratie",
            r"Ταυτότητα και Δημοκρατία",
            r"Tożsamość i Demokracja", r"Europa Wolności i Demokracji Bezpośredniej",
            r"Identita a demokracie", r"Identita a demokracia",
            r"Identitás és Demokrácia",
            r"Identiteet ja demokraatia", r"Identitāte un demokrātija",
            r"Tapatybė ir demokratija",
            r"Identitate și Democrație", r"Идентичност и демокрация",
        ],
        "anch": [],
    },
    "national": {
        "acr": ["KKE", "UKIP", "PiS", "Fidesz"],
        "name": [
            r"Communist Party of Greece", r"Κομμουνιστικό Κόμμα Ελλάδας",
            r"Kommunistische Partei Griechenlands", r"parti communiste (?:de|grec)\w*(?: Grèce)?",
            r"Partito comunista (?:di Grecia|greco)", r"Partido Comunista (?:de Grecia|Griego|da Grécia)",
            r"Komunistick\w* stran\w* (?:Řecka|Grécka)", r"(?:Řeck|Gréck)\w* komunistick\w* stran\w*",
            r"Komunistyczn\w* Parti\w* Grecji",
            r"Sinn F[ée]in", r"Rassemblement national", r"Prawo i Sprawiedliwo\w*",
            r"Northern League", r"Ligue du Nord", r"Lega Nord",
        ],
        "raw": [
            # generic phrases in lower case; only the capitalised form is the party
            r"\bFront [Nn]ational\b", r"\bLaw and Justice\b", r"\bDroit et [Jj]ustice\b",
            r"\bLega\b",
        ],
        "anch": [],
    },
}


# --------------------------------------------------------------------------- #
# Build ONE combined regex over all groups so each speech is scanned a single  #
# time. Each group is a named capture so we can tally which group a match hit. #
#   acronyms  -> case-sensitive, \b-bounded, longest-first                     #
#   names/anch-> wrapped in (?i:...) so they match case-insensitively          #
# The group word (GW) is factored OUT of the per-stem loop: it appears twice   #
# per group (forward/reverse) instead of once per ideology stem, which is what #
# made the naive build ~10x slower on long texts.                              #
# --------------------------------------------------------------------------- #
# regex group names can't contain / or &, so map safe name -> canonical label
_SAFE = {
    "PPE": "PPE", "SD": "S&D", "ALDE": "ALDE", "GREENS": "Greens/EFA",
    "ECR": "ECR", "GUENGL": "GUE/NGL", "ID": "ID", "NAT": "national",
}


def _any_space(pattern: str) -> str:
    """A literal space in a pattern matches any whitespace run, no-break space included."""
    return pattern.replace(r"\ ", " ").replace(" ", r"\s+")


def _subpattern(spec: dict[str, list[str]]) -> str:
    parts: list[str] = []

    acr = sorted(spec.get("acr", []), key=len, reverse=True)
    if acr:
        parts.append(r"\b(?:%s)\b" % "|".join(_any_space(re.escape(a)) for a in acr))

    parts.extend(_any_space(p) for p in spec.get("raw", []))

    ci = [_any_space(p) for p in spec.get("name", [])]
    anch = spec.get("anch", [])
    if anch:
        ide = "(?:%s)" % "|".join(anch)          # ideology words, matched once
        ci.append(rf"{GW}{_CONN}[\s\-]+{ide}")    # "Fraktion der Sozialdemokraten"
        ci.append(rf"{ide}{_CONN}[\s\-]+{GW}")    # "Socialist Group", "Renew-Fraktion"
    if ci:
        parts.append(r"(?i:\b(?:%s)\b)" % "|".join(ci))

    return "|".join(parts)


COMBINED_RE = re.compile(
    "|".join(rf"(?P<{safe}>{_subpattern(PATTERNS[canon])})" for safe, canon in _SAFE.items())
)
_WS_RE = re.compile(r"[^\S\n]{2,}")          # runs of spaces/tabs (keep newlines)
_SPACE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
MAX_PASSES = 3   # a removal can join two fragments into a new match ("Grupo dos [X] conservadores")

# Leftovers of a removed name (strip_residue).
_EMPTY_BRACKETS_RE = re.compile(r"[(\[]\s*[)\]]")                      # "(PPE-DE)" -> "( )"
_EMPTY_QUOTES_RE = re.compile(r'(?<=\s)"\s+"(?=\s)')                   # 'le groupe " "'
_LONE_SLASH_RE = re.compile(r"(?<![\d\w])\s/\s(?![\d])")                  # "návrhu / o" after "Zelení/ALE"
_ORPHAN_COMPOUND_RE = re.compile(                                       # "der -Fraktion"
    r"(?<![\w/])-(?=(?:Fraktion|Fractie|[Gg]ruppe|gruppen|ryhm|frakcij)\w*\b)")
# The transcript header before the speech: "au nom du groupe ID. – ", "in writing. – ",
# "blue-card answer. – ", or a bare "– " once the whole header was a name. The full stop
# is required: "The EU budget – which ..." is speech, not a header.
_HEADER_RE = re.compile(r"^(?:(?P<head>[^\n–—.]{1,80})\.)?[^\S\n]*[–—][^\S\n]+(?=\S)")
MAX_HEADER_WORDS = 8


def clean_text(text: str, counts: Counter) -> str:
    """Remove every group/party name from `text`, tallying hits into `counts`."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        counts[_SAFE[m.lastgroup]] += 1
        return REPLACEMENT

    new = text
    for _ in range(MAX_PASSES):
        step = COMBINED_RE.sub(_repl, new)
        if step == new:
            break
        new = step
    # Only tidy whitespace if we actually removed something, so untouched rows
    # stay byte-identical to the source.
    if new != text:
        new = _WS_RE.sub(" ", new)
        new = _SPACE_PUNCT_RE.sub(r"\1", new).lstrip()   # a name that opened the text
    return new


def strip_residue(text: str, counts: Counter | None = None) -> str:
    """Remove what a deleted name leaves behind: empty brackets and quotes, a hyphen
    orphaned from "-Fraktion", and a short transcript header before " – ". Runs on
    every row, not only on rows clean_text changed: the source already carries "()"
    where its own tooling dropped a group acronym. Counted once per changed row
    under "residue"."""
    if not text:
        return text
    new = _EMPTY_BRACKETS_RE.sub("", text)
    new = _EMPTY_QUOTES_RE.sub("", new)
    new = _ORPHAN_COMPOUND_RE.sub("", new)
    new = _LONE_SLASH_RE.sub(" ", new)
    m = _HEADER_RE.match(new)
    if m and len((m.group("head") or "").split()) <= MAX_HEADER_WORDS:
        new = new[m.end():]
    if new != text:
        new = _WS_RE.sub(" ", new)
        new = _SPACE_PUNCT_RE.sub(r"\1", new).lstrip()
        if counts is not None:
            counts["residue"] += 1
    return new


def _clean_chunk(chunk: tuple[list, list]) -> tuple[list, Counter, Counter]:
    """Worker: clean a list of texts, returning (cleaned, keep, party_counts, address_counts).

    Accents are repaired first (fix_diacritics), so the name patterns see whole
    words; a row whose accents were stripped upstream (is_transliterated) gets
    keep=False and is dropped by the caller; then party names and their residue
    (strip_residue); then titled person names and stray leading
    punctuation (clean_person_names.strip_names). address_counts is keyed (language, kind), plus
    (language, "rows"), so the report can show a rate per language.

    Runs in a separate process (Python's `re` holds the GIL, so threads do not
    help). The module-level COMBINED_RE is compiled once per worker on import.
    """
    texts, langs = chunk
    local: Counter = Counter()
    addr: Counter = Counter()
    cleaned, keep = [], []
    for text, lang in zip(texts, langs):
        addr[(lang, "rows")] += 1
        if text:
            kinds: Counter = Counter()
            text = fix_diacritics(text, kinds)
            if is_transliterated(text, lang):       # accents lost upstream: drop the row
                addr[(lang, "translit")] += 1
                cleaned.append(text)
                keep.append(False)
                continue
            text = strip_residue(clean_text(text, local), kinds)
            text = strip_names(text, kinds, names=REMOVE_PERSON_NAMES)
            for kind, n in kinds.items():
                addr[(lang, kind)] += n
        cleaned.append(text)
        keep.append(True)
    return cleaned, keep, local, addr


ADDRESS_KINDS = ["diacritics", "translit", "residue", "lead_punct", "name"]


def print_address_report(addr: Counter) -> None:
    """Removals per 100 rows, per language: the coverage check for languages whose
    patterns nobody has read against real text."""
    langs = sorted({lang for lang, _ in addr}, key=lambda l: -addr[(l, "rows")])
    print(f"    {'lang':<6}{'rows':>11}" + "".join(f"{k:>12}" for k in ADDRESS_KINDS)
          + "   (removals per 100 rows)")
    for lang in langs:
        rows = addr[(lang, "rows")]
        print(f"    {str(lang):<6}{rows:>11,}"
              + "".join(f"{100 * addr[(lang, k)] / max(rows, 1):>12.1f}" for k in ADDRESS_KINDS))


def process_split(split: str) -> tuple[Counter, Counter]:
    src = DATA_DIR / f"{split}.parquet"
    dst = OUT_DIR / f"{split}.parquet"
    if not src.exists():
        print(f"[skip] {src} not found")
        return Counter(), Counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    addr: Counter = Counter()
    reader = pq.ParquetFile(src)
    schema = reader.schema_arrow
    field = schema.field(TEXT_COL)
    text_idx = schema.get_field_index(TEXT_COL)
    lang_idx = schema.get_field_index("language")
    pool = pa.default_memory_pool()

    writer = pq.ParquetWriter(dst, schema)
    rows = 0
    # Read one row-group at a time (bounded peak memory), fan its rows out to a
    # process pool for the regex work, then write the group back Arrow-native
    # (no pandas round-trip). ex.map preserves order, so rows stay aligned.
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        try:
            for rg in range(reader.metadata.num_row_groups):
                group = reader.read_row_group(rg)
                texts = group.column(text_idx).to_pylist()
                langs = (group.column(lang_idx).to_pylist() if lang_idx >= 0
                         else [None] * len(texts))
                chunks = [(texts[i:i + PAR_CHUNK], langs[i:i + PAR_CHUNK])
                          for i in range(0, len(texts), PAR_CHUNK)]
                cleaned: list = []
                keep: list = []
                for part, part_keep, local, local_addr in ex.map(_clean_chunk, chunks):
                    cleaned.extend(part)
                    keep.extend(part_keep)
                    counts.update(local)
                    addr.update(local_addr)
                new_col = pa.array(cleaned, type=field.type)
                group = group.set_column(text_idx, field, new_col)
                rows += group.num_rows
                group = group.filter(pa.array(keep, type=pa.bool_()))
                writer.write_table(group)
                print(f"  {split}: {rows:,} rows processed", end="\r")
                del group, texts, langs, chunks, cleaned, keep, new_col
                gc.collect()
                try:
                    pool.release_unused()
                except AttributeError:
                    pass
        finally:
            writer.close()

    total = sum(counts.values())
    print(f"\n[{split}] {rows:,} rows  ->  {dst}  "
          f"({total:,} name occurrences removed)")
    width = max(len(g) for g in CANON)
    for g in CANON:
        print(f"    {g:<{width}} {counts.get(g, 0):>8,}")
    print(f"  accents repaired (rows) / leading punctuation / titled names removed ({split}):")
    print_address_report(addr)
    return counts, addr


def main() -> None:
    splits = sys.argv[1:] or SPLITS      # e.g. `python clean_party_names.py test`
    grand: Counter = Counter()
    grand_addr: Counter = Counter()
    for split in splits:
        counts, addr = process_split(split)
        grand.update(counts)
        grand_addr.update(addr)

    print("\n=== TOTAL across splits ===")
    width = max(len(g) for g in CANON)
    for g in CANON:
        print(f"    {g:<{width}} {grand.get(g, 0):>9,}")
    print(f"    {'ALL':<{width}} {sum(grand.values()):>9,}")
    print("\n=== accents repaired / leading punctuation / titled names, all splits ===")
    print_address_report(grand_addr)


if __name__ == "__main__":
    main()
