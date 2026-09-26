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
_CONJ = r"(?:and|und|et|och|og|ja|y|e|i|és|ir|и|και|un|in|en|a|și|şi)"
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
            r"skupin\w*,? konfederac\w*",                      # "moje skupina, Konfederace"
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
            # Case-SENSITIVE on purpose: every one of these is also an ordinary phrase in
            # lower case ("fought for independence and democracy", "a Europe of freedom and
            # democracy", "questions of identity and democracy"); the group is always
            # written capitalised.
            r"\bIndependence ?(?:/|and) ?Democracy\b", r"\bInd[ée]pendance ?(?:/|et) ?(?:de la )?D[ée]mocratie\b",
            r"\bUnabhängigkeit ?(?:/|und) ?Demokratie\b",
            r"\b(?:%s)\b" % "|".join([
                r"Identity and Democracy",
                r"Europe of Freedom and Direct Democracy", r"Europe of Freedom and Democracy",
                r"Europe of Nations and Freedom",
                r"Identität und Demokratie", r"Europa der Nationen und der Freiheit",
                r"Europa der Freiheit und der direkten Demokratie",
                r"Identité et Démocratie", r"Europe des Nations et des Libertés",
                r"Europe de la [Ll]iberté et de la [Dd]émocratie directe",
                r"Identità e Democrazia", r"Europa delle Nazioni e della Libertà",
                r"Identidad y Democracia", r"Europa de las Naciones y de las Libertades",
                r"Identidade e Democracia",
                r"Identiteit en Democratie", r"Europa van Vrijheid en Directe Democratie",
                r"Ταυτότητα και Δημοκρατία",
                r"Tożsamość i Demokracja", r"Europa Wolności i Demokracji Bezpośredniej",
                r"Identita a [Dd]emokracie", r"Identita a [Dd]emokracia",
                r"Identitás és Demokrácia",
                r"Identiteet ja [Dd]emokraatia", r"Identitāte un [Dd]emokrātija",
                r"Tapatybė ir [Dd]emokratija",
                r"Identitate [șş]i Democra[țţ]ie", r"Идентичност и [Дд]емокрация",
            ]),
        ],
        "name": [
            r"Union pour l['’]Europe des Nations", r"Union für das Europa der Nationen",
            r"Úni\w* za Európu národov", r"Uni\w* pro Evropu národů", r"Uni\w* na rzecz Europy Narodów",
            r"Union for Europe of the Nations",
        ],
        "anch": [
            r"Independence and Democracy", r"Ind[ée]pendance et (?:de la )?D[ée]mocratie",
            r"Unabhängigkeit und Demokratie", r"Identity and Democracy", r"Identität und Demokratie",
        ],
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
# Case forms, word orders and historic group names, found 2026-09-26 by listing #
# what follows and precedes a group word in every corpus language of the       #
# cleaned test split (the names above are mostly nominative-only, and UEN,     #
# ELDR, IND/DEM, EFD, ENF were missing in most languages). Merged into         #
# PATTERNS below. "name" entries are distinctive enough to remove anywhere;    #
# names that double as ordinary phrases ("a Europe of nations", "une Europe de #
# la liberté et de la démocratie", "independence and democracy") go in "anch", #
# removed only next to a group word, so the ideological phrase itself stays.  #
# --------------------------------------------------------------------------- #
PATTERNS_MORE: dict[str, dict[str, list[str]]] = {
    "PPE": {
        "acr": ["EVP-ED", "EVP", "ΕΛΚ-ΕΔ", "ΕΛΚ", "ЕНП"],
        "name": [
            r"Europäisch\w* Volkspartei\w*",
            r"Démocrates européens", r"Demócratas Europeos", r"Democratas Europeus",
            r"Democratic[io][\s\-]cristian[io]", r"Democratici europei",
            r"Europeiska folkpartiet\w*", r"Europademokrat\w*",
            r"Europæisk\w* Folkepart\w*", r"Europæiske Demokrater",
            r"Euroopan kansanpuolue\w*",
            r"Ευρωπαϊκ\w* Λαϊκ\w* Κόμμ\w*", r"Χριστιανοδημοκρ[αά]τ\w*",
            r"Chrześcijańsk\w* Demokrat\w*",
            r"Evropsk\w* stran\w* lidov\w*",                     # cs word order "Evropské strany lidové"
            r"Európai Néppárt\w*",
            r"Evropsk\w* ljudsk\w* strank\w*", r"Krščansk\w* demokrat\w*",
            r"Euroopa Rahvapartei\w*", r"Eiropas Tautas partij\w*", r"Europos liaudies partij\w*",
            r"Partidul\w* Popular\w* European", r"Cre[șş]tin[\s\-]?[Dd]emocra(?:t|[țţ])\w*",
            # the second half of PPE-DE, with its conjunction (safe: never a plain phrase)
            r"und europäischer Demokraten", r"en (?:de )?Europese Democraten",
            r"i Europejskich Demokratów", r"a Evropských demokratů", r"a európskych demokratov",
            r"és (?:az )?Európai Demokraták\w*", r"in Evropskih demokratov", r"ja Euroopa [Dd]emokraatide",
            r"un Eiropas Demokrātu", r"ir Europos demokratų", r"[șş]i (?:al? )?Democra[țţ]ilor Europeni",
            r"и Европейските демократи", r"και (?:των )?Ευρωπα[ιί]ων Δημοκρατ[ώω]ν", r"ja Euroopan demokraattien",
            r"kristlik\w* demokraat\w*", r"krikščionių demokrat\w*", r"Kristīg\w* demokrāt\w*",
            r"Европейск\w* народн\w* парти\w*", r"Християндемократ\w*",
        ],
        "anch": [
            r"Europe\w* Democrat\w*", r"Europäisch\w* Demokrat\w*", r"Europese democraten",
            r"Euroopan demokraat\w*", r"Ευρωπα[ιί]\w* Δημοκρατ\w*", r"Europejsk\w* Demokrat\w*",
            r"Evropsk\w* demokrat\w*", r"Európsk\w* demokrat\w*", r"Európai Demokrat\w*",
            r"Euroopa Demokraat\w*", r"Eiropas Demokrāt\w*", r"Europos demokrat\w*",
        ],
    },
    "S&D": {
        "acr": ["SPE"],
        "name": [
            r"Progressiv\w* Allianz\w*", r"Sozialdemokratisch\w* Partei Europas",
            r"Parti des socialistes européens", r"Partido dos Socialistas Europeus",
            r"Aliança Progressista\w*", r"Partij van de Europese Sociaaldemocraten",
            r"Progressiva förbundet\w*", r"Europeiska socialdemokraternas parti",
            r"Europæiske Socialdemokrat\w*", r"Sosialistien ja demokraattien",
            r"Προοδευτικ\w* Συμμαχ\w*", r"Σοσιαλιστικ\w* Κόμμ\w*",
            r"Postępow\w* Sojusz\w*", r"Pokrokov\w* alianc\w*", r"Progresívn\w* alianci\w*",
            r"Európai Szocialist\w*", r"[Nn]apredn\w* zavezništv\w*",
            r"Sotsiaaldemokraat\w* ja demokraat\w*", r"Sociālistu un demokrātu", r"progresīv\w* alians\w*",
            r"pažang\w* aljans\w*", r"Alian[țţ]\w* Progresist\w*", r"Sociali[șş]tilor Europeni", r"Sociali[șş]tilor [șş]i (?:a )?Democra[țţ]ilor",
            r"Euroopan sosia?alidemokraatti\w* puolue\w*", r"Europeiska socialdemokratiska partiet\w*",
            r"Прогресивни\w* алианс\w*",
        ],
        "anch": [
            r"sociaal[\s\-]?democrat\w*", r"sozialist\w*", r"socialist\w* europe\w*",
            r"europe\w* socialist\w*", r"evropsk\w* socialist\w*", r"sociáln\w* demokrat\w*", r"európsk\w* socialist\w*",
            r"evropskih socialistov", r"Euroopa Sotsialist\w*", r"европейските социалисти",
        ],
    },
    "ALDE": {
        "name": [
            r"European Liberal,? Democrat and Reform Party",
            r"Allianz der Liberalen und Demokraten(?: für Europa)?",
            r"Alliance des démocrates et des libéraux pour l['’]Europe",
            r"Parti européen des libéraux,? démocrates et réformateurs",
            r"Alleanza dei Democratici e dei Liberali per l['’]Europa",
            r"Partito europeo dei liberali,? democratici e riformatori",
            r"Alianza de los Demócratas y Liberales por Europa",
            r"Partido Europeo de los Liberales,? Demócratas y Reformistas",
            r"Aliança dos (?:Democratas e Liberais|Liberais e Democratas) pela Europa",
            r"Alleanza dei Liberali e dei Democratici per l['’]Europa",
            r"Alianza de los Liberales y Demócratas por Europa",
            r"Alliance des libéraux et des démocrates pour l['’]Europe",
            r"Partido Europeu dos Liberais,? Democratas e Reformistas",
            r"Alliantie van Liberalen en Democraten voor Europa",
            r"Alliansen liberaler och demokrater för Europa", r"Alliancen af Liberale og Demokrater for Europa",
            r"Συμμαχ\w* (?:των )?Φιλελε[υύ]θ[εέ]ρ\w*(?: και (?:των )?Δημοκρατ[ώω]ν)?(?: για την Ευρώπη)?",
            r"Porozumieni\w* (?:Liberałów i Demokratów )?na rzecz Europy",
            r"[Zz]avezništv\w* liberalc\w* in demokrat\w* za Evropo",
            r"(?:Euroopa )?Demokraatide ja Liberaalide Liid\w*",
            r"Alian[țţ]\w* Liberalilor (?:[șş]i Democra[țţ]ilor )?pentru Europa",
            r"Алианс\w* на либерал\w* и демократ\w* за Европа",
            r"Liberal\w* und Demokratisch\w* Partei\w*",
            r"Parti européen des libéraux", r"Alleanza dei Democratici\w*", r"Partito europeo dei liberali",
            r"Alianza de los Demócratas", r"Partido Europeo de los Liberales",
            r"Aliança dos Democratas", r"Partido Europeu dos Liberais",
            r"Alliantie van Liberalen", r"Europese Liberale\w*",
            r"Alliansen liberaler", r"Alliancen af Liberale",
            r"liberaalidemokraat\w*", r"liberaali- ja demokraattipuolue\w*",
            r"Συμμαχ\w* των Φιλελε[υύ]θερ\w*", r"Porozumieni\w* Liberałów",
            r"Szövetség\w* Európáért", r"[Zz]avezništv\w* liberalc\w*", r"Liberaalide Liid\w*",
            r"aljans\w* už Europą", r"Alian[țţ]\w* Liberalilor", r"Alianc\w* liberálov",
            r"Алианс\w* на либерал\w*",
        ],
        "anch": [r"φιλελε[υύ]θ[εέ]ρ\w*", r"liberalc\w*", r"liberaal\w*"],
    },
    "Greens/EFA": {
        "name": [
            r"Frei\w* Europäisch\w* Allianz", r"Euroopan vapa\w* allianssi\w*",
            r"Ευρωπαϊκ\w* Ελεύθερ\w* Συμμαχ\w*", r"Ελεύθερ\w* Ευρωπαϊκ\w* Συμμαχ\w*",
            r"Woln\w* Przymierz\w* Europejsk\w*", r"Európai Szabad Szövetség\w*",
            r"Evropsk\w* svobodn\w* zvez\w*", r"Euroopa Vabaliid\w*", r"Eiropas Brīv\w* apvienīb\w*",
            r"Europos laisv\w* aljans\w*", r"Alian[țţ]\w* Liber\w* Europe(?:an|n)\w*",
            r"Европейски\w* свободен алианс\w*", r"Europæiske Fri Alliance",
        ],
        "raw": [r"\bVerzilor\b", r"\bΠρ[αά]σ[ιί]ν(?:ων|οι|ους)\b"],
        "anch": [r"πρ[αά]σ[ιί]ν\w*"],
    },
    "GUE/NGL": {
        "name": [
            r"Verein\w* Europäisch\w* Link\w*", r"Nordisch\w* Grün\w* Link\w*",
            r"Europe\w* Unitair\w* Links", r"Noord\w* Groen\w* Links",
            r"enade vänster\w*", r"Nordisk\w* grön\w* vänster\w*",
            r"Venstrefløjs Fællesgruppe", r"Nordisk Grønne? Venstre",
            r"yhtynee\w* vasemmisto\w*", r"Pohjoismaiden vihre\w* vasemmisto\w*",
            r"Συνομοσπονδιακ\w*", r"(?:Ευρωπαϊκ\w* )?Ενωτικ\w* Αριστερ\w*", r"Αριστερ\w* των Πρ[αά]σ[ιί]νων",
            r"Konfederacyjn\w*", r"konfederativn\w*", r"Konfederaln\w*",
            r"Egységes Európai Baloldal\w*", r"Északi Zöld Baloldal\w*",
            r"Evropsk\w* združen\w* levic\w*", r"nordijsk\w* levic\w*",
            r"Ühendatud Vasak\w*", r"Põhjamaade Roheliste Vasak\w*",
            r"Apvienot\w* kreis\w*(?: un (?:Ziemeļvalstu [Zz]aļ\w* kreis\w* )?spēku)?", r"Ziemeļvalstu zaļ\w* kreis\w*", r"konfederāl\w*",
            r"vieningųjų kairiųjų(?: jungtin\w*)?", r"Šiaurės šalių žali\w*(?: kairi\w*)?",
            r"St[âa]ng\w* Unit\w* European\w*", r"St[âa]ng\w* Verde Nordic\w*",
            r"Европейск\w* обединен\w* левиц\w*", r"Северн\w* зелен\w* левиц\w*", r"Конфедератив\w*",
        ],
    },
    "ECR": {
        "acr": ["EKR", "ЕКР"],
        "name": [
            r"Europäisch\w* Konservativ\w* und Reformer\w*", r"konservativa och reformist\w*",
            r"Konservative og Reformister", r"konservatiiv\w* ja reformist\w*",
            r"Konserwatyst\w* i Reformator\w*", r"konservativc\w* in reformist\w*",
            r"консерватори и реформисти",
        ],
    },
    "ID": {
        "name": [
            # UEN
            r"Union for (?:a )?Europe of (?:the )?Nations", r"Unione per l['’]Europa delle Nazioni",
            r"Unión por la Europa de las Naciones", r"União para a Europa das Nações",
            r"Unie voor (?:een )?Europa van (?:de )?(?:Nati(?:es|ën|onen)|Nationale Staten)", r"Union(?:en)? för nationernas Europa",
            r"Union(?:en)? for Nationernes Europa", r"Unioni (?:kansakuntien )?Euroopan puolesta", r"kansakuntien Euroopan puolesta",
            r"Ένωσ\w* για την Ευρώπη των Εθνών", r"na rzecz Europy Narodów",
            r"Nemzetek Európájáért Unió\w*", r"za Evropo narodov", r"Liit Rahvusriikide Euroopa eest", r"Rahvusriikide Euroopa\w*",
            r"Nāciju Eiropa\w*", r"už tautų Europą", r"Uniun\w* pentru Europa Na[țţ]iunilor",
            r"Съюз\w* за Европа на нациите",
            # EDD (1999-2004)
            r"Europe of Democracies and Diversities", r"Europe des démocraties et des différences",
            r"Europa der Demokratien und der Unterschiede", r"Europa delle democrazie e delle diversità",
            r"Europa de las Democracias y de las Diferencias", r"Europa das Democracias e das Diferenças",
            r"Europa van Democratieën en Diversiteit", r"Demokrati\w* ja monimuotoisuuden Eurooppa\w*",
            r"Demokratiernas och mångfaldens Europa", r"Demokratiernes og Mangfoldighedens Europa",
        ],
        "raw": [
            # IND/DEM: only the capitalised official name, never "independence and democracy"
            r"\bIndepend[eê]nci?[ae]\s*(?:/|e|y|and)\s*Democra\w*",
            r"\bIndipendenza\s*(?:/|e)\s*Democrazia",
            r"\bOnafhankelijkheid\s*(?:/|en)\s*Democratie",
            r"\bSelvstændighed\s*(?:/|og)\s*Demokrati", r"\bSjälvständighet\s*(?:/|och)\s*demokrati",
            r"\bItsenäisyys\s*(?:/|ja)\s*demokratia",
            r"\bΑνεξαρτησία\s*(?:/|και)\s*Δημοκρατία",
            r"\bNiepodległość\s*(?:/|i)\s*Demokracja", r"\bNezávislos[tť]\s*(?:/|a)\s*demokraci[ea]",
            r"\bFüggetlenség\s*(?:/|és)\s*Demokrácia", r"\bNeodvisnost\s*(?:/|in)\s*[Dd]emokracija",
            r"\bIseseisvus\s*(?:/|ja)\s*[Dd]emokraatia", r"\bNeatkarība\s*(?:/|un)\s*[Dd]emokrātija",
            r"\bNepriklausomybė\s*(?:/|ir)\s*[Dd]emokratija",
        ],
        "anch": [
            # the same names without "Union for": a group word must be next to them
            r"Europe of (?:the )?Nations", r"Europe des [Nn]ations", r"Europa delle [Nn]azioni",
            r"Europa de las [Nn]aciones", r"Europa das Nações", r"Europa der Nationen",
            r"Europa van (?:de )?Nati\w*", r"Europe of Freedom and (?:Direct )?Democracy", r"για την Ευρώπη των Εθνών",
            r"Europe de la liberté et de la démocratie(?: directe)?",
            r"Europa der Freiheit und der (?:direkten )?Demokratie",
            r"Evrop\w* svobod\w* a (?:přímé )?demokraci\w*", r"Európ\w* slobod\w* a (?:priamej )?demokraci\w*",
            r"Ευρώπη της Ελευθερίας", r"Europa Wolności", r"Evropa narod\w* a svobod\w*",
            # EFD / EFDD, ENF: ordinary phrases too ("un'Europa della libertà e della democrazia")
            r"Europa della [Ll]ibertà e della [Dd]emocrazia(?: [Dd]iretta)?",
            r"Europa de la Libertad y de la Democracia(?: Directa)?", r"Europa da Liberdade e da Democracia(?: Direta)?",
            r"Frihet och direktdemokrati", r"Frihed og Direkte Demokrati",
            r"Vapauden ja suoran demokratian Eurooppa\w*",
            r"Europa das Nações e da Liberdade", r"Europa van Naties en Vrijheid",
            r"Nationernas och frihetens Europa", r"Nationernes og Frihedens Europa",
        ],
    },
}
for _canon, _spec in PATTERNS_MORE.items():
    for _tier, _pats in _spec.items():
        PATTERNS[_canon].setdefault(_tier, []).extend(_pats)

# --------------------------------------------------------------------------- #
# PATCH block, 2026-09-26: leaks still found in the test splits after a full    #
# clean-names run with everything above (~0.2% of rows; el, sl, hu, da, cs      #
# most). Merged into PATTERNS like the rest, so a run from raw applies it; and  #
# compiled on its own as PATCH_RE, so patch_cleaned.py can apply ONLY this      #
# block to an existing cleaned/ without re-running the full bank. Hence the     #
# "tail" entries: in already-cleaned text the head of a name is gone and only   #
# its tail is left ("groupe de l' et Démocrates"), which the full name no      #
# longer matches. When you add a round of patterns, put them here (and empty    #
# the block into PATTERNS_MORE after the next full run) so a patch stays cheap. #
# --------------------------------------------------------------------------- #
PATTERNS_PATCH: dict[str, dict[str, list[str]]] = {
    "PPE": {
        "name": [r"Cre[șş]tin\s*[\-–]\s*[Dd]emocra(?:t|[țţ])\w*"],   # "Creştin – Democrat"
    },
    "S&D": {
        "name": [
            r"Progres[ií]vn\w* alianc\w*",                            # cs "Progresivní aliance"
            r"(?:Det )?Progressive (?:Forbund|Alliance) af Socialdemokrater(?: og Demokrater)?",
            r"(?:(?:Sotsiaal)?[Dd]emokraatide |Sotsialistide )?(?:ja [Dd]emokraatide )?Progressiivse Liid\w*",
            r"(?:Demokraták )?Progresszív Szövetség\w*",
            r"Alliance [Pp]rogressiste des [Ss]ocialistes et (?:des )?[Dd]émocrates",
            r"Alleanza [Pp]rogressista (?:dei |di )?[Ss]ocialisti e (?:dei )?[Dd]emocratici",
            r"Partido de los Socialistas Europeos", r"Partito dei [Ss]ocialisti [Ee]uropei",
            r"Κόμμ\w* των Ευρωπα[ιί]ων Σοσιαλιστ[ώω]ν",
        ],
        "raw": [
            # tails in already-cleaned text: the head went in an earlier run
            r"(?i:\b(?:groupe|grupo)\s+(?:de\s+l['’]\s*|de\s+la\s+|do\s+|da\s+)?)(?:et|e|y)\s+(?:des\s+|dos\s+)?D[ée]mocra\w*",
            r"(?i:\bgruppo\s+(?:dell['’]\s*|del\s+)?)e\s+(?:dei\s+)?Democratici",
        ],
        "anch": [r"sociālist\w*"],                                   # lv "Sociālistu grupa"
    },
    "ALDE": {
        "name": [r"Συμμαχ\w* (?:των )?Δημοκρατ[ώω]ν και (?:των )?Φιλελε[υύ]θ[εέ]ρ\w*(?: για την Ευρώπη)?"],
    },
    "Greens/EFA": {
        "name": [r"Den Europæiske Fri Alliance"],
        "raw": [r"[/–]\s?Zelen(?:e|ih|i)\b"],                         # sl "Skupina Zelenih/... – Zelene"
    },
    "GUE/NGL": {
        "name": [
            r"(?:Den )?(?:Europæiske )?Venstrefløjs Fællesgruppe",
            r"(?:al )?St[âa]ng\w* Unit\w* Europe(?:an|n)\w*",
        ],
        "raw": [
            r"\b[șş]i Unite Europene\b",                              # ro tail of "Stângii Unite Europene"
            # capital Χ only: "των Βορείων χωρών" is also just "the northern countries"
            r"(?:των\s+)?Β[οό]ρε[ιί]\w*\s+Χωρ\w*",
        ],
    },
    "ECR": {
        "name": [r"Europäisch\w* Konservativ\w* und Reformist\w*"],
    },
    "ID": {
        "acr": ["ENL"],                                               # fr ENF, "Europe des nations et des libertés"
        "name": [
            r"(?:Unió )?a Nemzetek Európájáért(?: Unió\w*)?",
            r"Szabadság és (?:Közvetlen )?Demokrácia Európáj\w*",
            r"Evrop\w* svobode in (?:neposredne )?demokracije",
            r"Europa Libert[ăa][țţ]ii [șş]i (?:a )?Democra[țţ]iei(?: Directe)?",
        ],
        "raw": [
            # ordinary phrases in lower case ("sosiaalinen ja demokraattinen Eurooppa")
            r"\bVaba ja Demokraatliku Euroopa\w*", r"\bVapaa ja demokraattinen Eurooppa\w*",
            r"\bKansakuntien Eurooppa\b-?",
        ],
    },
    "national": {
        "acr": ["Fidesz-KDNP", "KDNP"],                              # "Fidesz – KDNP" left "- KDNP"
    },
}
for _canon, _spec in PATTERNS_PATCH.items():
    for _tier, _pats in _spec.items():
        PATTERNS[_canon].setdefault(_tier, []).extend(_pats)


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

    # longest first: at one start position the regex takes the FIRST alternative, so a
    # short name listed earlier ("Allianz der Liberalen und Demokraten") would otherwise
    # beat the full one and leave its tail ("für Europa") behind
    ci = [_any_space(p) for p in sorted(spec.get("name", []), key=len, reverse=True)]
    anch = [_any_space(p) for p in spec.get("anch", [])]
    if anch:
        ide = "(?:%s)" % "|".join(anch)          # ideology words, matched once
        ci.append(rf"{GW}{_CONN}[\s\-]+{ide}")    # "Fraktion der Sozialdemokraten"
        ci.append(rf"{ide}{_CONN}[\s\-]+{GW}")    # "Socialist Group", "Renew-Fraktion"
    if ci:
        parts.append(r"(?i:\b(?:%s)\b)" % "|".join(ci))

    return "|".join(parts)


def _combine(patterns: dict[str, dict[str, list[str]]]) -> re.Pattern:
    return re.compile("|".join(rf"(?P<{safe}>{_subpattern(patterns[canon])})"
                               for safe, canon in _SAFE.items() if canon in patterns))


COMBINED_RE = _combine(PATTERNS)
PATCH_RE = _combine(PATTERNS_PATCH)      # only the PATCH block, for patch_cleaned.py
_WS_RE = re.compile(r"[^\S\n]{2,}")          # runs of spaces/tabs (keep newlines)
_SPACE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
MAX_PASSES = 3   # a removal can join two fragments into a new match ("Grupo dos [X] conservadores")

# Leftovers of a removed name (strip_residue).
_EMPTY_BRACKETS_RE = re.compile(r"[(\[]\s*[)\]]")                      # "(PPE-DE)" -> "( )"
_EMPTY_QUOTES_RE = re.compile(r'(?<=\s)"\s+"(?=\s)')                   # 'le groupe " "'
# "the / supports", "předloženého /," after "Verts/ALE" or "Zelení/ALE" went; after a digit
# or capital it is real text ("2009 / 2010", "A / B", "EU / NATO") and stays
_LONE_SLASH_RE = re.compile(r"(?<=\w)(?<![\dA-ZÀ-Ý])\s/(?=[\s,.;:])(?!\s?\d)")
_ORPHAN_COMPOUND_RE = re.compile(                                       # "der -Fraktion"
    r"(?<![\w/])-(?=(?:Fraktion|Fractie|[Gg]ruppe|gruppen|ryhm|frakcij)\w*\b)")
# The transcript header before the speech: "au nom du groupe ID. – ", "in writing. – ",
# "blue-card answer. – ", or a bare "– " once the whole header was a name. The full stop
# is required: "The EU budget – which ..." is speech, not a header.
_HEADER_RE = re.compile(r"^(?:(?P<head>[^\n–—.]{1,80})\.)?[^\S\n]*[–—][^\S\n]+(?=\S)")
MAX_HEADER_WORDS = 8


def clean_text(text: str, counts: Counter, regex: re.Pattern | None = None) -> str:
    """Remove every group/party name from `text`, tallying hits into `counts`.
    `regex` defaults to the full bank; patch_cleaned.py passes PATCH_RE."""
    if not text:
        return text
    regex = regex or COMBINED_RE

    def _repl(m: re.Match) -> str:
        counts[_SAFE[m.lastgroup]] += 1
        return REPLACEMENT

    new = text
    for _ in range(MAX_PASSES):
        step = regex.sub(_repl, new)
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
