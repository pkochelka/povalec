"""Download the three raw corpora the classifier track is built from.

Everything under data/ is gitignored, so a fresh clone starts with nothing. This
script fetches the inputs that preprocessing/preprocess_data.py expects, plus the
fastText model europarl_lid_filter.py needs:

  data/ParlEE/ParlEE_EP_plenary_speeches.csv          851 MB
  data/EU Debates/train.jsonl                         286 MB
  data/EuroParl Custom/data/linkedEP/*.ttl            ~7.8 GB
  data/EuroParl Custom/lid.176.bin                    131 MB

Sources, citations and licences
------------------------------
ParlEE (EP plenary speeches, 2009-2019), sentence-level
    Harvard Dataverse, dataset DOI 10.7910/DVN/VOPK0E, file id 6936027.
    Rauh, Christian and Jan Schwalbach (2020), "The ParlSpeech V2 data set" /
    ParlEE plenary speeches. Terms: see the Dataverse record.

EU Debates (EP plenary speeches, 2009-2023)
    HuggingFace dataset coastalcph/eu_debates, licence CC-BY-NC-SA-4.0.
    Chalkidis, Ilias and Stephanie Brandl (2024), "Llama meets EU: Investigating
    the European political spectrum through the lens of LLMs", NAACL 2024.

LinkedEP / Talk of Europe (EP debates as Linked Open Data)
    DANS Data Station SSH, DOI 10.17026/dans-x62-ew3m, licence CC0-1.0.
    van Aggelen, Astrid et al. (2016), "The debates of the European Parliament
    as Linked Open Data", Semantic Web Journal.
    The RDF-query and language-ID steps applied to it here are Paul Lerner's;
    see preprocessing/europarl_rdf_query.py for the attribution.

fastText language identification model (lid.176.bin)
    Joulin et al. (2016), distributed by Meta AI under CC-BY-SA-3.0.

Usage
-----
    python preprocessing/fetch_raw_data.py --dry-run
    python preprocessing/fetch_raw_data.py --source eu-debates --source lid-model
    python preprocessing/fetch_raw_data.py --source linkedep --languages en de

Downloads resume: an interrupted transfer leaves a .part file and the next run
continues it with an HTTP Range request. Completed files are checksum-verified
against the repository API and then skipped on re-runs unless --force is given.
"""

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from preprocessing.preprocess_data import LANG_NAME_TO_CODE

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

SOURCES = ("parlee", "eu-debates", "linkedep", "lid-model")

HARVARD = "https://dataverse.harvard.edu"
PARLEE_FILE_ID = 6936027
PARLEE_DOI = "10.7910/DVN/VOPK0E"

EU_DEBATES_REPO = "coastalcph/eu_debates"
EU_DEBATES_ZIP = "eu_debates.zip"
EU_DEBATES_MEMBER = "train.jsonl"

DANS = "https://ssh.datastations.nl"
LINKEDEP_DOI = "doi:10.17026/dans-x62-ew3m"

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Non-language LinkedEP files the SPARQL in europarl_rdf_query.py actually needs.
# vocabulary.ttl carries the party-type labels ("EU Party", "NationalParty",
# "EU Committee") that become column names; spokenAs.ttl the lpv:spokenAs links;
# MembersOfParliament_background.ttl the lpv:institution edges, speaker names,
# dates of birth and party acronyms; Events_and_structure.ttl the agenda dates
# and lpv:speaker edges.
LINKEDEP_SUPPORT_FILES = (
    "Events_and_structure.ttl",
    "spokenAs.ttl",
    "MembersOfParliament_background.ttl",
    "vocabulary.ttl",
    "Countries_in_Geonames.ttl",
)

# Deliberately not fetched by default: *_prov.ttl (provenance; no triple in them
# matches the queries), eurovoc_skos.rdf (282 MB, and europarl_rdf_query.py only
# globs *.ttl so it was never parsed), topics/ subdirectories (also not globbed),
# mep_wikidata / MEPs_in_* (owl:sameAs links only). --all-files overrides this
# and mirrors every root-level file in the DOI.

CHUNK_SIZE = 1 << 20

# Harvard Dataverse answers 403 to the default "python-requests/x.y" User-Agent
# on /api/access/datafile, so identify ourselves as something it will serve.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; povalec-fetch/1.0)"}


def human(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.2f} GB"


def checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, expected_size=None, algorithm=None, expected_digest=None,
             force=False, dry_run=False, label=None):
    """Stream url to dest, resuming from a .part sidecar and verifying afterwards."""
    label = label or dest.name
    if dry_run:
        size = f" ({human(expected_size)})" if expected_size else ""
        print(f"  [dry-run] {label}{size}\n            {url}\n         -> {dest}")
        return "dry-run"

    if dest.exists() and not force:
        if expected_size is None or dest.stat().st_size == expected_size:
            print(f"  skipping {label}: already present ({human(dest.stat().st_size)})")
            return "skipped"
        print(f"  {label}: size mismatch "
              f"({human(dest.stat().st_size)} vs {human(expected_size)}), re-downloading")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() and not force else 0
    if force and part.exists():
        part.unlink()

    headers = dict(HEADERS)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    if offset and response.status_code != 206:
        # Server ignored the Range request; start over rather than concatenate.
        print(f"  {label}: server refused resume, restarting from 0")
        offset = 0

    total = expected_size
    if total is None:
        length = response.headers.get("Content-Length")
        total = int(length) + offset if length is not None else None

    mode = "ab" if offset else "wb"
    with open(part, mode) as handle, tqdm(
        total=total, initial=offset, unit="B", unit_scale=True,
        unit_divisor=1024, desc=f"  {label}",
    ) as bar:
        for block in response.iter_content(chunk_size=CHUNK_SIZE):
            handle.write(block)
            bar.update(len(block))

    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(
            f"{label}: got {part.stat().st_size} bytes, expected {expected_size}; "
            f"partial file left at {part}"
        )
    if algorithm and expected_digest:
        actual = checksum(part, algorithm)
        if actual.lower() != expected_digest.lower():
            raise RuntimeError(
                f"{label}: {algorithm} mismatch ({actual} != {expected_digest}); "
                f"partial file left at {part}"
            )
        print(f"  {label}: {algorithm} verified")

    os.replace(part, dest)
    return "downloaded"


def api_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_parlee(data_dir, force, dry_run):
    print("=== ParlEE ===")
    print(f"  Harvard Dataverse {PARLEE_DOI}, file {PARLEE_FILE_ID}")
    meta = api_json(f"{HARVARD}/api/files/{PARLEE_FILE_ID}")["data"]["dataFile"]
    dest = data_dir / "ParlEE" / meta["filename"]
    download(
        f"{HARVARD}/api/access/datafile/{PARLEE_FILE_ID}?format=original",
        dest,
        expected_size=meta["filesize"],
        algorithm=meta["checksum"]["type"].replace("-", "").lower(),
        expected_digest=meta["checksum"]["value"],
        force=force, dry_run=dry_run,
    )
    print()


def fetch_eu_debates(data_dir, force, dry_run):
    print("=== EU Debates ===")
    print(f"  HuggingFace {EU_DEBATES_REPO} ({EU_DEBATES_ZIP} -> {EU_DEBATES_MEMBER})")
    dest = data_dir / "EU Debates" / EU_DEBATES_MEMBER

    if dry_run:
        print(f"  [dry-run] hf_hub_download({EU_DEBATES_REPO!r}, {EU_DEBATES_ZIP!r})"
              f"\n         -> {dest}")
        print()
        return
    if dest.exists() and not force:
        print(f"  skipping {dest.name}: already present ({human(dest.stat().st_size)})")
        print()
        return

    # Not load_dataset(): eu_debates is a script-based dataset (unsupported from
    # datasets>=3) and its loader drops the intervention_language field that
    # preprocess_eu_debates() renames to "language".
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id=EU_DEBATES_REPO, filename=EU_DEBATES_ZIP, repo_type="dataset",
    )
    print(f"  cached archive: {archive}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        member = next(
            (name for name in bundle.namelist()
             if Path(name).name == EU_DEBATES_MEMBER and not name.endswith("/")),
            None,
        )
        if member is None:
            raise RuntimeError(f"{EU_DEBATES_ZIP} contains no {EU_DEBATES_MEMBER}")
        with bundle.open(member) as source, open(dest, "wb") as handle:
            while block := source.read(CHUNK_SIZE):
                handle.write(block)
    print(f"  extracted {member} -> {dest} ({human(dest.stat().st_size)})")
    print()


def select_linkedep_files(entries, languages, all_files):
    """Pick the root-level files to fetch, in (support first, then language) order."""
    root = [entry for entry in entries if not entry.get("directoryLabel")]
    by_name = {entry["dataFile"]["filename"]: entry for entry in root}

    if all_files:
        return [by_name[name] for name in sorted(by_name)]

    wanted = [name for name in LINKEDEP_SUPPORT_FILES if name in by_name]
    missing_support = set(LINKEDEP_SUPPORT_FILES) - set(wanted)
    if missing_support:
        raise RuntimeError(f"LinkedEP is missing expected files: {sorted(missing_support)}")

    available_langs = {
        name: LANG_NAME_TO_CODE[name.removesuffix(".ttl")]
        for name in by_name
        if name.endswith(".ttl") and name.removesuffix(".ttl") in LANG_NAME_TO_CODE
    }
    if languages:
        unknown = set(languages) - set(available_langs.values())
        if unknown:
            raise RuntimeError(
                f"LinkedEP has no corpus for language code(s) {sorted(unknown)}; "
                f"available: {sorted(available_langs.values())}"
            )
        wanted += sorted(name for name, code in available_langs.items() if code in languages)
    else:
        wanted += sorted(available_langs)

    return [by_name[name] for name in wanted]


def fetch_linkedep(data_dir, languages, all_files, force, dry_run):
    print("=== LinkedEP (Talk of Europe) ===")
    print(f"  DANS Data Station SSH {LINKEDEP_DOI} (CC0-1.0)")
    entries = api_json(
        f"{DANS}/api/datasets/:persistentId/versions/:latest/files",
        params={"persistentId": LINKEDEP_DOI},
    )["data"]
    selected = select_linkedep_files(entries, languages, all_files)
    total = sum(entry["dataFile"]["filesize"] for entry in selected)
    print(f"  {len(selected)} files, {human(total)}"
          f"{' (--all-files)' if all_files else ''}")

    out_dir = data_dir / "EuroParl Custom" / "data" / "linkedEP"
    for entry in selected:
        info = entry["dataFile"]
        download(
            f"{DANS}/api/access/datafile/{info['id']}",
            out_dir / info["filename"],
            expected_size=info["filesize"],
            algorithm=info["checksum"]["type"].replace("-", "").lower(),
            expected_digest=info["checksum"]["value"],
            force=force, dry_run=dry_run,
        )
    print()


def fetch_lid_model(data_dir, force, dry_run):
    print("=== fastText LID model ===")
    download(
        LID_URL,
        data_dir / "EuroParl Custom" / "lid.176.bin",
        force=force, dry_run=dry_run, label="lid.176.bin",
    )
    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", choices=SOURCES + ("all",),
                        help="repeatable; default all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"default {DEFAULT_DATA_DIR}")
    parser.add_argument("--languages", nargs="+", metavar="CODE",
                        help="LinkedEP language corpora to fetch, as 2-letter codes "
                             "(default: all 23). Each is ~250-630 MB.")
    parser.add_argument("--all-files", action="store_true",
                        help="fetch every root-level LinkedEP file, not just the ones "
                             "the SPARQL needs (adds ~410 MB of provenance and EuroVoc)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the target already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print URLs, sizes and destinations; download nothing")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = set(args.source or ["all"])
    if "all" in sources:
        sources = set(SOURCES)

    if args.languages and "linkedep" not in sources:
        print("note: --languages only affects the linkedep source")

    print(f"Data directory: {args.data_dir}\n")
    for source in SOURCES:
        if source not in sources:
            continue
        try:
            if source == "parlee":
                fetch_parlee(args.data_dir, args.force, args.dry_run)
            elif source == "eu-debates":
                fetch_eu_debates(args.data_dir, args.force, args.dry_run)
            elif source == "linkedep":
                fetch_linkedep(args.data_dir, args.languages, args.all_files,
                               args.force, args.dry_run)
            elif source == "lid-model":
                fetch_lid_model(args.data_dir, args.force, args.dry_run)
        except Exception as error:
            print(f"  FAILED ({source}): {type(error).__name__}: {error}\n")

    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to download.")
    else:
        print("Next: preprocessing/europarl_rdf_query.py, then europarl_lid_filter.py, "
              "then preprocess_data.py.")


if __name__ == "__main__":
    main()
