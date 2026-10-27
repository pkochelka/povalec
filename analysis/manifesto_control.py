#!/usr/bin/env python3
"""Out-of-domain control for the mmBERT EU-party classifier: party manifestos.

The classifier is trained and validated on European Parliament plenary speeches but the
paper applies it to LLM-written opinion prose. This script applies the *deployed*
classifier (argmax of logits, as in classify_speeches.py) to text whose EP-group label is
known but which is not a plenary speech -- national party manifestos and manifesto
excerpts -- and reports how much of the in-domain accuracy survives the register change.

Three sources, chosen with --source:

  euandi   EU&I 2024 party justifications already in the repo
           (data/euandi_2024_data/euandi_2024_parties.jsonl). Every national party
           justifies each of its 30 answers with a quote from its own programme. Exact
           ground truth (utils.EP_GROUP_BY_PARTY), no download, runs immediately.
  marpor   Manifesto Project corpus (WZB, https://manifesto-project.wzb.eu). Full national
           election manifestos, quasi-sentence annotated. Needs a free API key in
           .env.local as MANIFESTO_API_KEY. The party -> EP group ground truth is the
           curated 2019-2024 term table EP_GROUP_BY_MARPOR_PARTY below (check it against
           --list_parties, extend/override it with --mapping).
  local    A directory of *.txt manifestos plus a --mapping CSV
           (file;party;country;language;ep_group), e.g. the Euromanifesto Study 2019
           documents (GESIS ZA7892) converted to text, or hand-collected 2024 manifestos.

Manifestos are far longer than the classifier's 512-token window, so each document is cut
into --chunk_words-word chunks (default 200, roughly an EP plenary speech) and scored at two
levels: per chunk (comparable to the in-domain test set) and per document (mean softmax
over the document's chunks, then argmax). Reports accuracy, macro F1, confusion matrix,
the true-vs-predicted class histogram, per-language and per-country F1, chance baselines,
and optionally the same after a uniform-marginal logit bias (--calibrate), which separates
prior shift from genuine confusion.

In-domain reference (paper, collapsed test split): accuracy 0.485, macro F1 0.47.

Usage:
  python analysis/manifesto_control.py --source euandi --plot
  python analysis/manifesto_control.py --source marpor --list_parties --countries Germany France
  python analysis/manifesto_control.py --source marpor --plot
  python analysis/manifesto_control.py --source marpor --model_dir runs/<cluster run>/model --tag <name> --plot

A cluster-track model (its manifest names the four k=4 clusters) is scored against the
parties' clusters instead: euandi through the cluster positions file's member lists, marpor
through PartyFacts (EU&I party -> CHES id -> Manifesto Project party code, all 27 countries).
  python analysis/manifesto_control.py --source local --local_dir data/manifesto_control/local --mapping data/manifesto_control/local/mapping.csv
"""
import argparse
import base64
import io
import json
import os
import re
import time
import zlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")
from analysis.core.labels import use_short_party_labels
use_short_party_labels()   # cluster nicknames in every label (utils.PARTY_SHORT)

from analysis.classify_speeches import softmax
from analysis.core.positions import (
    CLUSTER_POSITIONS_PATH,
    PARTY_POSITIONS_PATH,
    load_party_positions,
)
from analysis.europarl_classification import (
    MIN_LANGUAGE_SAMPLES,
    classification_report_text,
    fit_uniform_bias,
    per_language_f1_report,
)
from analysis.test_classifier import (
    load_model,
    plot_confusion_matrix,
    predict_logits,
    print_class_histogram,
    print_confusion_matrix,
)
from analysis.party_kmeans import COUNTRY_BY_ISO, RAW_CSV, load_parties
from utils import EP_GROUP_BY_PARTY, configure_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "mmBERT-base-balanced-collapsed")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "manifesto_control")

# Same collapse the classifier's training data went through (build_collapsed_splits.py) and
# that evaluate_euandi.agreement_by_ep_group applies to the 7-group party constants.
COLLAPSE = {"ECR": "ECR+ID", "ID": "ECR+ID"}

# --- Manifesto Project (MARPOR) -----------------------------------------------------------

MARPOR_API = "https://manifesto-project.wzb.eu/api/v1"
MARPOR_CORE_VERSION = "MPDS2026a"       # main dataset release (party codes, election dates)
MARPOR_CORPUS_VERSION = "2026-1"        # corpus release (texts and annotations)
# The corpus metadata names a manifesto's language in English; the classifier and the
# per-language report use ISO 639-1 codes.
ISO_BY_MARPOR_LANGUAGE = {
    "bulgarian": "bg", "catalan": "ca", "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "estonian": "et", "finnish": "fi", "french": "fr",
    "galician": "gl", "german": "de", "greek": "el", "hungarian": "hu", "italian": "it",
    "latvian": "lv", "lithuanian": "lt", "maltese": "mt", "polish": "pl",
    "portuguese": "pt", "romanian": "ro", "slovak": "sk", "slovenian": "sl",
    "spanish": "es", "swedish": "sv",
}

# PartyFacts (https://partyfacts.herokuapp.com) links party ids across datasets; its
# external-parties table carries both the CHES ids the raw EU&I file gives every party and
# the Manifesto Project party codes. Public, no key.
PARTYFACTS_URL = "https://partyfacts.herokuapp.com/download/external-parties-csv/"

# EP-group affiliation during the 2019-2024 term, keyed by MPDS `countryname` and one or
# more spellings of `partyabbrev` (matching is case-insensitive, punctuation-insensitive).
# Seeded from utils.EP_GROUP_BY_NATIONAL_PARTY (DE, FR, IT, ES, GR) and extended with
# parties whose affiliation was stable over the term. Parties that changed group mid-term
# (Fidesz, left EPP in 2021) or joined only after the 2024 election (Chega, AUR) are left
# out on purpose. Verify abbreviations with --list_parties; override with --mapping.
EP_GROUP_BY_MARPOR_PARTY = {
    "Germany": {
        ("CDU/CSU", "CDU", "CSU"): "PPE", ("SPD",): "S&D", ("FDP",): "ALDE",
        ("90/Greens", "Greens/90", "Grüne", "B90/Gru", "Greens"): "Greens/EFA",
        ("LINKE", "Die Linke", "PDS"): "GUE/NGL", ("AfD",): "ID",
    },
    "France": {
        ("LR", "UMP"): "PPE", ("PS",): "S&D", ("LREM", "REM", "RE", "Renaissance"): "ALDE",
        ("EELV", "Greens", "Verts"): "Greens/EFA", ("RN", "FN"): "ID", ("LFI", "FI"): "GUE/NGL",
        ("MoDem",): "ALDE",
    },
    "Italy": {
        ("FI", "PdL"): "PPE", ("PD",): "S&D", ("FdI",): "ECR", ("Lega", "LN"): "ID",
        ("M5S",): "GUE/NGL", ("AVS", "SI", "EV"): "Greens/EFA", ("Az", "IV", "AR"): "ALDE",
    },
    "Spain": {
        ("PP",): "PPE", ("PSOE",): "S&D", ("VOX",): "ECR", ("Cs", "C's"): "ALDE",
        ("UP", "Podemos", "Sumar"): "GUE/NGL", ("ERC",): "Greens/EFA", ("PNV", "EAJ-PNV"): "ALDE",
    },
    "Greece": {
        ("ND",): "PPE", ("PASOK", "KINAL", "PASOK-KINAL"): "S&D", ("SYRIZA",): "S&D",
        ("EL",): "ECR",
    },
    "Netherlands": {
        ("VVD",): "ALDE", ("D66",): "ALDE", ("CDA",): "PPE", ("PvdA",): "S&D",
        ("GL",): "Greens/EFA", ("SP",): "GUE/NGL", ("PVV",): "ID", ("SGP",): "ECR",
        ("PvdD",): "GUE/NGL", ("CU",): "PPE", ("FvD",): "ECR", ("Volt",): "Greens/EFA",
    },
    "Austria": {
        ("ÖVP", "OVP"): "PPE", ("SPÖ", "SPO"): "S&D", ("FPÖ", "FPO"): "ID",
        ("Grüne", "GRÜNE", "Greens"): "Greens/EFA", ("NEOS",): "ALDE",
    },
    "Belgium": {
        ("N-VA", "NVA"): "ECR", ("VB",): "ID", ("CD&V", "CVP"): "PPE",
        ("Open VLD", "VLD"): "ALDE", ("MR",): "ALDE", ("PS",): "S&D",
        ("sp.a", "Vooruit"): "S&D", ("Ecolo",): "Greens/EFA", ("Groen", "Groen!"): "Greens/EFA",
        ("PTB-PVDA", "PVDA", "PTB"): "GUE/NGL", ("cdH", "LE"): "PPE",
    },
    "Poland": {
        ("PiS",): "ECR", ("PO", "KO"): "PPE", ("PSL",): "PPE", ("SLD", "Lewica", "NL"): "S&D",
        ("Wiosna",): "S&D",
    },
    "Czech Republic": {
        ("ANO", "ANO 2011"): "ALDE", ("ODS",): "ECR", ("KDU-ČSL", "KDU-CSL"): "PPE",
        ("TOP 09", "TOP09"): "PPE", ("ČSSD", "CSSD"): "S&D", ("KSČM", "KSCM"): "GUE/NGL",
        ("Piráti", "Pirati", "Pirates"): "Greens/EFA", ("SPD",): "ID", ("STAN",): "PPE",
    },
    "Slovakia": {
        ("SMER", "Smer-SD", "SMER-SD"): "S&D", ("PS",): "ALDE", ("OĽaNO", "OLaNO", "OLANO"): "PPE",
        ("KDH",): "PPE", ("SaS",): "ECR",
    },
    "Hungary": {
        ("MSZP",): "S&D", ("DK",): "S&D", ("Momentum",): "ALDE", ("KDNP",): "PPE",
        ("LMP",): "Greens/EFA",
    },
    "Sweden": {
        ("M",): "PPE", ("KD",): "PPE", ("S", "SAP"): "S&D", ("C",): "ALDE", ("L", "FP"): "ALDE",
        ("MP",): "Greens/EFA", ("V",): "GUE/NGL", ("SD",): "ECR",
    },
    "Denmark": {
        ("V",): "ALDE", ("S", "SD"): "S&D", ("SF",): "Greens/EFA", ("DF",): "ID",
        ("KF", "K"): "PPE", ("RV",): "ALDE", ("EL",): "GUE/NGL",
    },
    "Finland": {
        ("KOK",): "PPE", ("SDP", "SSDP"): "S&D", ("PS",): "ID", ("KESK",): "ALDE",
        ("VIHR",): "Greens/EFA", ("VAS",): "GUE/NGL", ("RKP", "SFP", "RKP/SFP"): "ALDE",
        ("KD",): "PPE",
    },
    "Portugal": {
        ("PS",): "S&D", ("PSD", "PPD-PSD", "PPD/PSD"): "PPE", ("CDS-PP", "CDS"): "PPE",
        ("BE",): "GUE/NGL", ("PCP", "CDU"): "GUE/NGL", ("IL",): "ALDE", ("PAN",): "Greens/EFA",
    },
    "Ireland": {
        ("FG",): "PPE", ("FF",): "ALDE", ("SF",): "GUE/NGL", ("GP", "Greens"): "Greens/EFA",
        ("Lab",): "S&D",
    },
    "Romania": {
        ("PSD",): "S&D", ("PNL",): "PPE", ("USR", "USR-PLUS", "USR PLUS"): "ALDE",
        ("PMP",): "PPE", ("UDMR", "RMDSZ"): "PPE",
    },
}

# EU&I country codes -> classifier language codes (the model was trained on lowercase
# ISO 639-1 language tags; the europarties answer in English).
LANGUAGE_BY_COUNTRY_ISO = {"de": "de", "fr": "fr", "it": "it", "es": "es", "gr": "el", "eu": "en"}


# The cluster-track classifiers (runs/<track>-k4-<trainer>) predict the four k=4 party
# clusters, not EP groups. Their ground truth for an EU&I party is the cluster it was
# fitted into, read off the member lists of the cluster positions file, which spells
# parties as the raw 27-country file does. These are the jsonl short names that differ
# from it; everything else matches case-insensitively.
RAW_ABBREV_BY_EUANDI_PARTY = {
    ("de", "CDU"): "CDU/CSU", ("de", "Grüne"): "Grunen", ("fr", "PS"): "PP-PS",
    ("es", "Sumar"): "SMR", ("gr", "Niki"): "NIKH",
    ("gr", "ND"): "ΝΔ", ("gr", "SYRIZA"): "ΣΥΡΙΖΑ", ("gr", "PASOK"): "ΠΑΣΟΚ",
    ("gr", "EL"): "EΛ",   # Latin E + Greek lambda, as the raw file has it
}


# (EU&I country, EU&I abbreviation) -> Manifesto Project party codes PartyFacts does not
# reach through the party's CHES id: no CHES id in the raw file, no PartyFacts link for it,
# or a link that ends at a predecessor's code. Checked against the MPDS2026a party list.
# Same-party matches only -- a joint list (GL-PvdA, PSD-PNL, SPOLU, AD) is not handed down
# to its member parties, and a party is not credited with a joint national list it ran on
# (NUPES for LFI, PCP for the Portuguese CDU).
MARPOR_CODES_BY_EUANDI_PARTY = {
    ("Austria", "NEOS"): [42430],
    ("Belgium", "CD&V"): [21521], ("Belgium", "VB"): [21917],
    ("Germany", "CDU/CSU"): [41521],
    ("Greece", "ΠΑΣΟΚ"): [34315],
    ("Hungary", "MHM"): [86711],
    ("Latvia", "V"): [87063], ("Latvia", "P"): [87312], ("Latvia", "LPV"): [87736],
    ("Latvia", "AS"): [87011], ("Latvia", "SDPS"): [87340],
    ("Netherlands", "Volt"): [22235],
    ("Poland", "KO"): [92040], ("Poland", "Konfederacja"): [92070], ("Poland", "L"): [92022, 92023],
    ("Spain", "Junts"): [33912], ("Spain", "SMR"): [33021],
}


def cluster_members():
    """"<abbreviation> (<country>)", lowercased -> k=4 cluster, for every party the
    clusters were fitted on."""
    member_cluster = {}
    with open(CLUSTER_POSITIONS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for member in record["member_parties"]:
                member_cluster[member.lower()] = record["ep_group"]
    return member_cluster


def cluster_by_euandi_party(parties):
    """`parties`: (country_iso, short_name) pairs -> the k=4 cluster of each, None for a
    party the clusters were not fitted on (no MEPs)."""
    member_cluster = cluster_members()
    clusters = []
    for country_iso, short_name in parties:
        abbrev = RAW_ABBREV_BY_EUANDI_PARTY.get((country_iso, short_name), short_name)
        clusters.append(member_cluster.get(f"{abbrev} ({COUNTRY_BY_ISO.get(country_iso)})".lower()))
    return clusters


def partyfacts_table(cache_dir):
    """PartyFacts' external-parties table, downloaded once into cache_dir."""
    path = Path(cache_dir) / "external-parties.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(PARTYFACTS_URL, timeout=300)
        if response.status_code != 200:
            raise SystemExit(f"PartyFacts download failed ({response.status_code}): {PARTYFACTS_URL}")
        path.write_bytes(response.content)
        print(f"  [partyfacts] downloaded {path}")
    return pd.read_csv(path, low_memory=False)


def cluster_by_marpor_code(cache_dir):
    """Manifesto Project party code -> (k=4 cluster, EU&I party), for the cluster-track
    classifiers. EU&I party -> CHES id (raw EU&I file) -> PartyFacts id -> every Manifesto
    Project code PartyFacts files under it, plus MARPOR_CODES_BY_EUANDI_PARTY. A code two
    EU&I parties of different clusters lay claim to is left out."""
    member_cluster = cluster_members()
    raw = load_parties(RAW_CSV)
    # build_cluster_positions' member label: the abbreviation, or the name where it is missing.
    raw["member"] = [f"{(a if isinstance(a, str) else n).strip()} ({c.strip()})"
                     for a, n, c in zip(raw["ABBREVIATON"], raw["PARTY_NAME_ENG"], raw["COUNTRY"])]
    raw["cluster"] = raw["member"].str.lower().map(member_cluster)
    raw = raw.dropna(subset=["cluster"]).copy()
    raw["ches_id"] = pd.to_numeric(raw["CHES_ID"], errors="coerce")

    partyfacts = partyfacts_table(cache_dir)
    ches = partyfacts[partyfacts["dataset_key"] == "ches"].copy()
    ches["ches_id"] = pd.to_numeric(ches["dataset_party_id"], errors="coerce")
    ches = ches.dropna(subset=["ches_id", "partyfacts_id"]).drop_duplicates("ches_id")
    marpor = partyfacts[partyfacts["dataset_key"] == "manifesto"].dropna(subset=["partyfacts_id"])
    linked = (raw.merge(ches[["ches_id", "partyfacts_id"]], on="ches_id")
              .merge(marpor[["partyfacts_id", "dataset_party_id"]], on="partyfacts_id"))
    claims = [(int(code), cluster, member) for code, cluster, member
              in zip(linked["dataset_party_id"], linked["cluster"], linked["member"])]
    by_key = {(country.strip(), str(abbrev).strip()): (cluster, member) for country, abbrev, cluster, member
              in zip(raw["COUNTRY"], raw["ABBREVIATON"], raw["cluster"], raw["member"])}
    for key, codes in MARPOR_CODES_BY_EUANDI_PARTY.items():
        if key not in by_key:
            raise SystemExit(f"MARPOR_CODES_BY_EUANDI_PARTY names {key}, which no cluster holds")
        claims += [(code, *by_key[key]) for code in codes]

    code_cluster, contested = {}, set()
    for code, cluster, member in claims:
        if code in code_cluster and code_cluster[code][0] != cluster:
            contested.add(code)
        code_cluster.setdefault(code, (cluster, member))
    for code in contested:
        del code_cluster[code]
    parties = {member for _, member in code_cluster.values()}
    print(f"  [partyfacts] {len(code_cluster)} Manifesto Project codes for {len(parties)} of "
          f"{len(raw)} clustered EU&I parties ({len(contested)} contested codes left out)")
    return code_cluster


# The 21 languages of the classifier's train/dev/test splits. A manifesto in any other
# language (Catalan, Croatian, Maltese in the Manifesto Project corpus) is out of the
# classifier's domain twice over, register and language, so it is left out by default.
CLASSIFIER_LANGUAGES = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu",
                        "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]


def classifier_languages(model_dir):
    """The languages the classifier was evaluated on: read off the run's dumped eval logits
    where it has them, CLASSIFIER_LANGUAGES otherwise."""
    found = set()
    for path in Path(model_dir, "eval_logits").glob("*.npz"):
        with np.load(path) as data:
            if "language" in data.files:
                found |= set(data["language"].tolist())
    return found or set(CLASSIFIER_LANGUAGES)


# --- text units --------------------------------------------------------------------------

def chunk_words(text, chunk_size):
    """Cut `text` into consecutive windows of `chunk_size` words. 0 = whole text."""
    words = text.split()
    if chunk_size <= 0 or len(words) <= chunk_size:
        return [" ".join(words)]
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def explode_into_chunks(docs, chunk_size, min_chars):
    """docs: one row per document -> one row per chunk, with `doc_id` and `chunk_idx`."""
    rows = []
    for doc in docs.itertuples(index=False):
        for idx, chunk in enumerate(chunk_words(doc.text, chunk_size)):
            if len(chunk) < min_chars:
                continue
            rows.append({**doc._asdict(), "text": chunk, "chunk_idx": idx})
    return pd.DataFrame(rows)


# --- loaders: each returns doc_id, party, country, language, ep_group, text ---------------

def load_euandi(include_europarties, clusters=False):
    df = load_party_positions(PARTY_POSITIONS_PATH, "national")
    if clusters:
        df["ep_group"] = cluster_by_euandi_party(zip(df["country_iso"], df["short_name"]))
    if include_europarties:
        eu = load_party_positions(PARTY_POSITIONS_PATH, "ep-group")
        df = pd.concat([df, eu], ignore_index=True)
    df = df.dropna(subset=["justification"])
    df["justification"] = df["justification"].astype(str).str.strip()
    df = df[df["justification"].str.len() > 0]
    # One "document" per (party, statement) justification; the excerpts are short (a few
    # sentences), so with the default chunk size document == chunk.
    return pd.DataFrame({
        "doc_id": df["short_name"] + "#" + df["statement_idx"].astype(str),
        "party": df["short_name"],
        "country": df["country_iso"],
        "language": df["country_iso"].map(LANGUAGE_BY_COUNTRY_ISO).fillna("en"),
        "ep_group": df["ep_group"],
        "text": df["justification"],
    }).reset_index(drop=True)


def load_local(local_dir, mapping_path):
    if not mapping_path:
        raise SystemExit("--source local needs --mapping <csv> with columns "
                         "file;party;country;language;ep_group")
    mapping = pd.read_csv(mapping_path, sep=";", dtype=str).fillna("")
    required = {"file", "party", "country", "language", "ep_group"}
    missing = required - set(mapping.columns)
    if missing:
        raise SystemExit(f"--mapping is missing columns: {sorted(missing)}")
    rows = []
    for row in mapping.itertuples(index=False):
        path = Path(local_dir) / row.file
        if not path.exists():
            print(f"  [local] missing file, skipped: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({
            "doc_id": row.file, "party": row.party, "country": row.country,
            "language": row.language, "ep_group": row.ep_group,
            "text": re.sub(r"\s+", " ", text).strip(),
        })
    return pd.DataFrame(rows)


def _normalise_abbrev(s):
    return re.sub(r"[^0-9a-zà-ÿ]", "", str(s).lower())


def build_marpor_lookup(mapping_path=None):
    """(countryname, normalised partyabbrev) -> 7-group label."""
    lookup = {}
    for country, parties in EP_GROUP_BY_MARPOR_PARTY.items():
        for abbrevs, group in parties.items():
            for abbrev in abbrevs:
                lookup[(country, _normalise_abbrev(abbrev))] = group
    if mapping_path:
        extra = pd.read_csv(mapping_path, sep=";", dtype=str).fillna("")
        for row in extra.itertuples(index=False):
            lookup[(row.country, _normalise_abbrev(row.party_abbrev))] = row.ep_group
        print(f"  [marpor] {len(extra)} mapping rows loaded from {mapping_path}")
    return lookup


class MarporClient:
    """Thin cached client for the Manifesto Project REST API. Every response is written
    under cache_dir so that re-runs never touch the daily request quota."""

    def __init__(self, api_key, cache_dir, sleep=0.5):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.session = requests.Session()

    def _get(self, endpoint, params, cache_name):
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        if not self.api_key:    # only a request needs the key: a synced cache runs without it
            raise SystemExit(f"{cache_path} is not cached and MANIFESTO_API_KEY is not set (put it "
                             "in .env.local; free key at https://manifesto-project.wzb.eu after "
                             "registering)")
        response = self.session.get(
            f"{MARPOR_API}/{endpoint}", params={"api_key": self.api_key, **params}, timeout=120,
        )
        if response.status_code != 200:
            raise SystemExit(f"MARPOR {endpoint} failed ({response.status_code}): "
                             f"{response.text[:300]}")
        payload = response.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        time.sleep(self.sleep)
        return payload

    def core(self, version):
        """The main dataset as a DataFrame (party codes, abbreviations, election dates)."""
        payload = self._get("get_core", {"key": version}, f"core_{version}.json")
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            return pd.DataFrame(payload[1:], columns=payload[0])  # header row first
        if isinstance(payload, dict) and "content" in payload:
            raw = base64.b64decode(payload["content"])
            try:
                return pd.read_csv(io.BytesIO(raw))
            except Exception:  # noqa: BLE001 - a Stata/other binary; let pandas try
                return pd.read_stata(io.BytesIO(raw))
        raise SystemExit(f"Unexpected get_core payload shape: {type(payload)}")

    def metadata(self, keys, version):
        items = []
        for start in range(0, len(keys), 50):
            batch = keys[start:start + 50]
            payload = self._get(
                "metadata", {"keys[]": batch, "version": version},
                f"metadata_{version}_{start:05d}_{zlib.crc32(' '.join(batch).encode()):08x}.json",
            )
            items.extend(payload.get("items", []))
        return items

    def text(self, manifesto_id, version):
        payload = self._get(
            "texts_and_annotations", {"keys[]": [manifesto_id], "version": version},
            f"text_{version}_{manifesto_id}.json",
        )
        items = payload.get("items", [])
        if not items:
            return None
        sentences = []
        for entry in items[0].get("items", []):
            content = entry.get("text", entry.get("content", ""))
            if content and str(content).strip():
                sentences.append(str(content).strip())
        return " ".join(sentences)


def load_marpor(args, clusters=False):
    code_cluster = cluster_by_marpor_code(Path(args.out_dir) / "partyfacts") if clusters else {}
    load_dotenv(os.path.join(PROJECT_ROOT, ".env.local"))
    client = MarporClient(os.getenv("MANIFESTO_API_KEY"), Path(args.out_dir) / "marpor")
    core = client.core(args.core_version)
    core["edate"] = pd.to_datetime(core["edate"], dayfirst=True, errors="coerce")   # dd/mm/yyyy
    core["date"] = pd.to_numeric(core["date"], errors="coerce")
    core["party"] = pd.to_numeric(core["party"], errors="coerce")
    if args.countries:
        countries = set(args.countries)
    elif clusters:      # every country a clustered party's code turns up in
        countries = set(core.loc[core["party"].isin(code_cluster), "countryname"])
    else:
        countries = set(EP_GROUP_BY_MARPOR_PARTY)
    core = core[core["countryname"].isin(countries)].copy()

    if args.list_parties:
        latest = (core.sort_values("edate")
                  .groupby(["countryname", "party"], as_index=False).last()
                  [["countryname", "party", "partyabbrev", "partyname", "edate"]])
        print(latest.to_string(index=False))
        raise SystemExit(0)

    since = pd.Timestamp(args.since)
    until = pd.Timestamp(args.until)
    recent = core[(core["edate"] >= since) & (core["edate"] <= until)].copy()
    if clusters:
        # By party code, not abbreviation; --mapping (naming clusters) fills in the rest.
        lookup = build_marpor_lookup(args.mapping) if args.mapping else {}
        cluster_names = set(cluster_members().values())
        lookup = {key: group for key, group in lookup.items() if group in cluster_names}
        recent["ep_group"] = [
            code_cluster[p][0] if p in code_cluster else lookup.get((c, _normalise_abbrev(a)))
            for p, c, a in zip(recent["party"], recent["countryname"], recent["partyabbrev"])
        ]
        matched = recent[recent["party"].isin(code_cluster)].drop_duplicates("party")
        print("  [marpor] Manifesto Project party <- EU&I party, cluster (check the pairs):")
        for row in matched.sort_values(["countryname", "partyabbrev"]).itertuples(index=False):
            cluster, member = code_cluster[row.party]
            print(f"    {row.countryname:<16}{str(row.partyabbrev):<14}{str(row.partyname)[:44]:<46}"
                  f"<- {member:<26}{cluster}")
    else:
        lookup = build_marpor_lookup(args.mapping)
        recent["ep_group"] = [
            lookup.get((c, _normalise_abbrev(a))) for c, a in zip(recent["countryname"], recent["partyabbrev"])
        ]
    unmatched = recent[recent["ep_group"].isna()][["countryname", "partyabbrev", "partyname"]]
    unmatched = unmatched.drop_duplicates().sort_values(["countryname", "partyabbrev"])
    print(f"  [marpor] {recent['ep_group'].notna().sum()} manifesto rows matched to "
          f"{'a cluster' if clusters else 'an EP group'}, "
          f"{len(unmatched)} distinct parties unmatched (not an error; extend --mapping if needed):")
    print(unmatched.to_string(index=False))
    recent = recent[recent["ep_group"].notna()].copy()
    recent["key"] = recent["party"].astype(int).astype(str) + "_" + recent["date"].astype(int).astype(str)

    meta = client.metadata(sorted(recent["key"].unique()), args.corpus_version)
    meta_by_key = {}
    for item in meta:
        if not item.get("annotations"):
            continue
        if item.get("is_primary_doc") is False:
            continue
        meta_by_key[f"{item['party_id']}_{item['election_date']}"] = item
    print(f"  [marpor] {len(meta_by_key)} of {recent['key'].nunique()} manifestos have machine-readable text")

    rows = []
    for row in recent.drop_duplicates("key").itertuples(index=False):
        item = meta_by_key.get(row.key)
        if item is None:
            continue
        text = client.text(item.get("manifesto_id", row.key), args.corpus_version)
        if not text:
            continue
        language = str(item.get("language", "")).strip().lower()
        rows.append({
            "doc_id": row.key, "party": f"{row.partyabbrev} ({row.countryname})",
            "country": row.countryname,
            "language": ISO_BY_MARPOR_LANGUAGE.get(language, language[:2]) or "xx",
            "ep_group": row.ep_group, "text": text,
        })
    return pd.DataFrame(rows)


# --- evaluation --------------------------------------------------------------------------

def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def chance_baselines(y_true, num_labels, seed=0):
    majority = np.full_like(y_true, np.bincount(y_true, minlength=num_labels).argmax())
    uniform = np.random.default_rng(seed).integers(0, num_labels, size=len(y_true))
    return {"majority": score(y_true, majority), "uniform_random": score(y_true, uniform)}


def per_group_table(df, group_col, labels, y_true, y_pred):
    lines = [f"\n=== PER-{group_col.upper()} ===",
             f"{group_col:<20}{'n':>8}{'accuracy':>10}{'macro_f1':>10}", "-" * 48]
    for value, index in df.groupby(group_col).indices.items():
        s = score(y_true[index], y_pred[index])
        lines.append(f"{str(value):<20}{len(index):>8d}{s['accuracy']:>10.4f}{s['f1_macro']:>10.4f}")
    return "\n".join(lines)


def document_level(chunks, probs, labels, label2id):
    """Mean softmax over a document's chunks -> argmax. Returns a per-document frame."""
    chunks = chunks.reset_index(drop=True)
    records = []
    for doc_id, index in chunks.groupby("doc_id", sort=False).indices.items():
        first = chunks.iloc[index[0]]
        mean_probs = probs[index].mean(axis=0)
        true_id = label2id[first["ep_group"]]
        records.append({
            "doc_id": doc_id, "party": first["party"], "country": first["country"],
            "language": first["language"], "ep_group": first["ep_group"],
            "n_chunks": len(index), "pred": labels[int(mean_probs.argmax())],
            "p_true": float(mean_probs[true_id]),
            **{f"p_{label}": float(p) for label, p in zip(labels, mean_probs)},
        })
    return pd.DataFrame(records)


def evaluate(chunks, logits, labels, label2id, tag, bias=None, temperature=1.0):
    """Chunk- and document-level report for one scoring rule. Returns (text, metrics)."""
    adjusted = logits if bias is None else logits + bias
    probs = softmax(adjusted / temperature)
    y_true = chunks["ep_group"].map(label2id).to_numpy()
    y_pred = adjusted.argmax(axis=1)

    out = [f"\n\n##### {tag} #####"]
    chunk_scores = score(y_true, y_pred)
    out.append(f"\n[chunks n={len(y_true)}] accuracy {chunk_scores['accuracy']:.4f} | "
               f"macro F1 {chunk_scores['f1_macro']:.4f}")
    out.append("\n=== CHUNK-LEVEL CLASSIFICATION REPORT ===\n"
               + classification_report_text(y_true, y_pred, labels))
    out.append(per_group_table(chunks, "country", labels, y_true, y_pred))
    languages = chunks["language"].to_numpy()
    if (pd.Series(languages).value_counts() >= MIN_LANGUAGE_SAMPLES).any():
        summary, _ = per_language_f1_report(y_true, y_pred, languages, labels)
        out.append(summary)
    else:
        out.append(f"\n(per-language summary skipped: no language has >= {MIN_LANGUAGE_SAMPLES} chunks)")

    docs = document_level(chunks, probs, labels, label2id)
    d_true = docs["ep_group"].map(label2id).to_numpy()
    d_pred = docs["pred"].map(label2id).to_numpy()
    doc_scores = score(d_true, d_pred)
    out.append(f"\n[documents n={len(docs)}] accuracy {doc_scores['accuracy']:.4f} | "
               f"macro F1 {doc_scores['f1_macro']:.4f}")
    out.append("\n=== DOCUMENT-LEVEL CLASSIFICATION REPORT ===\n"
               + classification_report_text(d_true, d_pred, labels))
    party = (docs.groupby(["party", "country", "ep_group"], as_index=False)
             .agg(n_docs=("doc_id", "size"), n_chunks=("n_chunks", "sum"),
                  p_true=("p_true", "mean"),
                  correct=("pred", lambda s: float((s == docs.loc[s.index, "ep_group"]).mean())))
             .sort_values(["ep_group", "country", "party"]))
    out.append("\n=== PER-PARTY (document level) ===\n" + party.to_string(index=False))

    metrics = {"chunks": chunk_scores, "documents": doc_scores,
               "n_chunks": int(len(y_true)), "n_documents": int(len(docs))}
    return "\n".join(out), metrics, (y_true, y_pred), (d_true, d_pred), docs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["euandi", "marpor", "local"], default="euandi")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--chunk_words", default=200, type=int,
                        help="Words per chunk (0 = whole document, truncated at 512 tokens).")
    parser.add_argument("--min_chars", default=50, type=int,
                        help="Drop chunks shorter than this (training used MIN_TEXT_LEN=50).")
    parser.add_argument("--languages", default=None, nargs="+",
                        help="Keep only documents in these languages (ISO 639-1). Default: the "
                             "languages the classifier was trained on; 'all' keeps every document.")
    parser.add_argument("--max_chunks_per_doc", default=None, type=int,
                        help="Score a random sample of this many chunks per document (seed 0) "
                             "instead of all of them; the outputs are tagged _s<N>.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Also report results after a uniform-marginal logit bias fitted "
                             "on these chunks (prior shift vs. genuine confusion).")
    parser.add_argument("--tag", default=None,
                        help="Appended to the output file names, so a second model does not "
                             "overwrite the first one's results.")
    parser.add_argument("--plot", default=None, nargs="?", const="",
                        help="Save the document-level confusion matrix PNG (default path under --out_dir).")
    # euandi
    parser.add_argument("--include_europarties", action="store_true",
                        help="euandi: also score the 7 europarties' own English justifications.")
    # local
    parser.add_argument("--local_dir", default=os.path.join(DEFAULT_OUT_DIR, "local"))
    parser.add_argument("--mapping", default=None,
                        help="CSV (;-separated). local: file;party;country;language;ep_group. "
                             "marpor: country;party_abbrev;ep_group (extends the built-in table).")
    # marpor
    parser.add_argument("--countries", nargs="*", default=None, help="marpor: MPDS countryname filter.")
    parser.add_argument("--since", default="2019-01-01", help="marpor: earliest election date.")
    parser.add_argument("--until", default="2024-06-30", help="marpor: latest election date.")
    parser.add_argument("--core_version", default=MARPOR_CORE_VERSION)
    parser.add_argument("--corpus_version", default=MARPOR_CORPUS_VERSION)
    parser.add_argument("--list_parties", action="store_true",
                        help="marpor: print the parties of the selected countries and exit.")
    return parser.parse_args()


def main():
    configure_stdout()
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"--- source: {args.source} | device: {args.device} | model: {args.model_dir} ---")
    tokenizer, model, manifest = load_model(args.model_dir, args.device)
    label2id = manifest["label2id"]
    labels = sorted(label2id, key=label2id.get)
    max_len = manifest["max_len"]
    # A cluster-track model: score against the parties' k=4 clusters (euandi source; a
    # local --mapping can name clusters in its ep_group column itself).
    cluster_labels = "Radical left" in label2id
    # The manifest's inference spec, as classify_speeches.py applies it.
    deployed_bias = manifest.get("biases")
    if deployed_bias is not None:
        deployed_bias = np.asarray(deployed_bias, dtype=np.float32)
    temperature = float(manifest.get("temperature", 1.0))

    if args.source == "euandi":
        docs = load_euandi(args.include_europarties, clusters=cluster_labels)
    elif args.source == "marpor":
        docs = load_marpor(args, clusters=cluster_labels)
    else:
        docs = load_local(args.local_dir, args.mapping)

    languages = classifier_languages(args.model_dir) if args.languages is None else set(args.languages)
    if "all" not in languages:
        outside = docs[~docs["language"].isin(languages)]
        if len(outside):
            kept = "the classifier's languages" if args.languages is None else "--languages"
            print(f"Dropping {len(outside)} documents outside {kept}: "
                  f"{outside['language'].value_counts().to_dict()} (--languages all keeps them)")
        docs = docs[docs["language"].isin(languages)].reset_index(drop=True)

    docs["ep_group"] = docs["ep_group"].replace(COLLAPSE)
    dropped = docs[~docs["ep_group"].isin(label2id)]
    if len(dropped):
        print(f"Dropping {len(dropped)} documents whose EP group is not a classifier label "
              f"(NI/unmapped): {sorted(dropped['party'].astype(str).unique())}")
    docs = docs[docs["ep_group"].isin(label2id)].reset_index(drop=True)
    if docs.empty:
        raise SystemExit("No documents with a usable EP-group label.")
    print(f"{len(docs)} documents from {docs['party'].nunique()} parties, "
          f"{docs['country'].nunique()} countries; class counts (documents):")
    print(docs["ep_group"].value_counts().reindex(labels).fillna(0).astype(int).to_string())

    chunks = explode_into_chunks(docs, args.chunk_words, args.min_chars)
    if args.max_chunks_per_doc:
        chunks = (chunks.sample(frac=1, random_state=0).groupby("doc_id", sort=False)
                  .head(args.max_chunks_per_doc).sort_index().reset_index(drop=True))
    print(f"{len(chunks)} chunks of <= {args.chunk_words or 'all'} words "
          f"(mean {chunks['text'].str.split().str.len().mean():.0f} words)")

    logits = predict_logits(chunks["text"].tolist(), tokenizer, model, max_len, args.device, args.batch_size)

    deployed_rule = "argmax(logits)" if deployed_bias is None else "argmax(logits + manifest biases)"
    report, metrics, (y_true, y_pred), (d_true, d_pred), doc_frame = evaluate(
        chunks, logits, labels, label2id, f"DEPLOYED RULE: {deployed_rule}",
        deployed_bias, temperature)
    print(report)
    print_class_histogram(y_true, y_pred, labels)
    print_confusion_matrix(y_true, y_pred, labels)
    print("\n(document level)")
    print_confusion_matrix(d_true, d_pred, labels)

    baselines = chance_baselines(y_true, len(labels))
    print("\n=== CHANCE BASELINES (chunk level) ===")
    for name, s in baselines.items():
        print(f"{name:<16} accuracy {s['accuracy']:.4f} | macro F1 {s['f1_macro']:.4f}")
    if not cluster_labels:
        print("in-domain reference (EP speeches, collapsed test): accuracy 0.485 | macro F1 0.47")

    results = {"source": args.source, "chunk_words": args.chunk_words,
               "languages": sorted(chunks["language"].unique()), "deployed": metrics,
               "baselines": baselines}
    calibrated_report = ""
    if args.calibrate:
        bias = fit_uniform_bias(logits, len(labels))
        calibrated_report, cal_metrics, (cy_true, cy_pred), _, _ = evaluate(
            chunks, logits, labels, label2id, "CALIBRATED: argmax(logits + uniform-marginal bias)", bias)
        print(calibrated_report)
        print_class_histogram(cy_true, cy_pred, labels)
        results["calibrated"] = cal_metrics
        results["bias"] = {label: float(b) for label, b in zip(labels, bias)}

    # outputs
    suffix = args.source if args.chunk_words == 200 else f"{args.source}_w{args.chunk_words}"
    if args.max_chunks_per_doc:
        suffix = f"{suffix}_s{args.max_chunks_per_doc}"
    if args.tag:
        suffix = f"{suffix}_{args.tag}"
    predictions_path = Path(args.out_dir) / f"predictions_{suffix}.csv"
    pred_frame = chunks.drop(columns=["text"]).copy()
    pred_frame["pred"] = [labels[i] for i in y_pred]
    for i, label in enumerate(labels):
        pred_frame[f"logit_{label}"] = logits[:, i]
    pred_frame.to_csv(predictions_path, sep=";", index=False, encoding="utf-8-sig")
    doc_frame.to_csv(Path(args.out_dir) / f"documents_{suffix}.csv", sep=";", index=False, encoding="utf-8-sig")

    results_path = Path(f"results_manifesto_control_{suffix}.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"manifesto_control_{suffix}\n" + "-" * 40 + "\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n" + report + "\n" + calibrated_report + "\n")
    print(f"\nSaved {results_path}, {predictions_path}")

    if args.plot is not None:
        path = Path(args.plot) if args.plot else Path(args.out_dir) / f"confusion_matrix_{suffix}.png"
        model_name = os.path.basename(os.path.normpath(args.model_dir))
        plot_confusion_matrix(
            d_true, d_pred, labels, path,
            subtitle=f"{model_name} | {args.source} manifestos, document level | "
                     f"accuracy {metrics['documents']['accuracy']:.3f} | "
                     f"macro F1 {metrics['documents']['f1_macro']:.3f}",
        )


if __name__ == "__main__":
    main()
