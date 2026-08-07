"""
Strip European Parliament political-group / party names out of the speech text.

Reads  data/EuroParl Custom/{train,dev,test}.parquet
Writes data/EuroParl Custom/cleaned/{train,dev,test}.parquet
and prints, per split, how many name occurrences were removed for each of the
seven canonical EP groups. The `cleaned/` subdirectory, same filenames, is what
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

DATA_DIR = Path("data/EuroParl Custom")
OUT_DIR = DATA_DIR / "cleaned"      # what build_collapsed_splits.py reads
SPLITS = ["train", "dev", "test"]
TEXT_COL = "text"
PAR_CHUNK = 4_000            # rows per task handed to a worker process
N_WORKERS = max(1, (os.cpu_count() or 2) - 2)   # leave a couple cores for I/O
REPLACEMENT = " "            # what a removed name becomes (whitespace then collapsed)

CANON = ["PPE", "S&D", "ALDE", "Greens/EFA", "ECR", "GUE/NGL", "ID"]

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
            r"Evropská lidová strana", r"Křesťansk\w* demokrat\w*",
            r"Európska ľudová strana", r"Kresťansk\w* demokrat\w*",
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
            r"socialistů a demokratů", r"socialistov a demokratov",
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
            r"Evropská svobodná aliance", r"Európsku slobodnú alianciu",
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
            r"Evropská sjednocená levice", r"Severská zelená levice",
            r"Európska zjednotená ľavica", r"Severská zelená ľavica",
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
        "acr": ["EFDD", "EFD", "ENF", "ID Group", "ITS"],
        "name": [
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
    "ECR": "ECR", "GUENGL": "GUE/NGL", "ID": "ID",
}


def _subpattern(spec: dict[str, list[str]]) -> str:
    parts: list[str] = []

    acr = sorted(spec.get("acr", []), key=len, reverse=True)
    if acr:
        parts.append(r"\b(?:%s)\b" % "|".join(re.escape(a) for a in acr))

    ci = list(spec.get("name", []))
    anch = spec.get("anch", [])
    if anch:
        ide = "(?:%s)" % "|".join(anch)          # ideology words, matched once
        ci.append(rf"{GW}{_CONN}\s+{ide}")        # "Fraktion der Sozialdemokraten"
        ci.append(rf"{ide}{_CONN}\s+{GW}")        # "Socialist Group"
    if ci:
        parts.append(r"(?i:\b(?:%s)\b)" % "|".join(ci))

    return "|".join(parts)


COMBINED_RE = re.compile(
    "|".join(rf"(?P<{safe}>{_subpattern(PATTERNS[canon])})" for safe, canon in _SAFE.items())
)
_WS_RE = re.compile(r"[^\S\n]{2,}")          # runs of spaces/tabs (keep newlines)
_SPACE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def clean_text(text: str, counts: Counter) -> str:
    """Remove every group/party name from `text`, tallying hits into `counts`."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        counts[_SAFE[m.lastgroup]] += 1
        return REPLACEMENT

    new = COMBINED_RE.sub(_repl, text)
    # Only tidy whitespace if we actually removed something, so untouched rows
    # stay byte-identical to the source.
    if new != text:
        new = _WS_RE.sub(" ", new)
        new = _SPACE_PUNCT_RE.sub(r"\1", new)
    return new


def _clean_chunk(texts: list) -> tuple[list, Counter]:
    """Worker: clean a list of texts, returning (cleaned, local_counts).

    Runs in a separate process (Python's `re` holds the GIL, so threads do not
    help). The module-level COMBINED_RE is compiled once per worker on import.
    """
    local: Counter = Counter()
    cleaned = [clean_text(t, local) if t else t for t in texts]
    return cleaned, local


def process_split(split: str) -> Counter:
    src = DATA_DIR / f"{split}.parquet"
    dst = OUT_DIR / f"{split}.parquet"
    if not src.exists():
        print(f"[skip] {src} not found")
        return Counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    reader = pq.ParquetFile(src)
    schema = reader.schema_arrow
    field = schema.field(TEXT_COL)
    text_idx = schema.get_field_index(TEXT_COL)
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
                chunks = [texts[i:i + PAR_CHUNK] for i in range(0, len(texts), PAR_CHUNK)]
                cleaned: list = []
                for part, local in ex.map(_clean_chunk, chunks):
                    cleaned.extend(part)
                    counts.update(local)
                new_col = pa.array(cleaned, type=field.type)
                group = group.set_column(text_idx, field, new_col)
                writer.write_table(group)
                rows += group.num_rows
                print(f"  {split}: {rows:,} rows processed", end="\r")
                del group, texts, chunks, cleaned, new_col
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
    return counts


def main() -> None:
    splits = sys.argv[1:] or SPLITS      # e.g. `python clean_party_names.py test`
    grand: Counter = Counter()
    for split in splits:
        grand.update(process_split(split))

    print("\n=== TOTAL across splits ===")
    width = max(len(g) for g in CANON)
    for g in CANON:
        print(f"    {g:<{width}} {grand.get(g, 0):>9,}")
    print(f"    {'ALL':<{width}} {sum(grand.values()):>9,}")


if __name__ == "__main__":
    main()
