"""Map every speech's NATIONAL party onto an EU&I 2024 party, for the cluster track.

Reads  data/ParlEE/ParlEE_EP_plenary_speeches.csv
       data/EuroParl Custom/data/linkedEP/{MembersOfParliament_background,spokenAs,
                                           Events_and_structure}.ttl
       data/euandi_2024_raw/EUandI_2024_party_dataset.csv   (via analysis/party_kmeans.py)
       preprocessing/national_party_overrides.csv
       PartyFacts external-parties table (downloaded on first run)
Writes data/EuroParl Custom/national_parties/
         speeches.parquet   one row per source speech: speaker, speaker_key, date, party, pui
         party_map.csv      one row per source party: how (or why not) it was mapped
         coverage.json      speech shares per source and outcome

Only parties that map onto an EU&I 2024 party holding at least one MEP -- the parties
analysis/party_kmeans.py clusters -- get a `pui`; everything else is left unmapped and
dropped downstream. There is no EP-group fallback.

How a party is mapped, in order:

  ParlEE     1. ids: parlgov_party (else cmp_party) -> PartyFacts -> CHES id -> the EU&I
                row with that CHES_ID. Name-free, so spelling variants are free.
             2. national_party_overrides.csv: the hand-checked table, for 2024 joint lists
                (PSD + CDS-PP -> AD, ODS/KDU-ČSL/TOP 09 -> SPOLU, ...), renames (UMP -> LR,
                FN -> RN), formal mergers (PDL -> PNL), and EU&I rows with no CHES id.
                Every row carries its reason. Overrides win over ids.
  LinkedEP   Its parties are only "<acronym>_<Country>" codes, with no ids. They are
             bridged through ParlEE: a LinkedEP speech and a ParlEE speech by the same
             speaker (normalised name) on the same day are the same speech, so each
             LinkedEP party takes the 2024 party its bridged speeches resolved to -- when
             at least --bridge-min-share of them agree and there are --bridge-min-n.
             Parties with no speech from 2009 on (before ParlEE starts, so no bridge is
             possible) are dropped outright, as decided -- not hand-matched.

EU Debates carries no national party at all; its speeches are resolved downstream by
speaker name against this file's speeches (see build_national_cluster_splits.py).
"""

import argparse
import json
import re
import unicodedata
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

pd.options.future.infer_string = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from analysis import party_kmeans as pk

DATA_DIR = PROJECT_ROOT / "data"
PARLEE_CSV = DATA_DIR / "ParlEE" / "ParlEE_EP_plenary_speeches.csv"
LINKEDEP_DIR = DATA_DIR / "EuroParl Custom" / "data" / "linkedEP"
OVERRIDES_CSV = PROJECT_ROOT / "preprocessing" / "national_party_overrides.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "EuroParl Custom" / "national_parties"
PARTYFACTS_URL = "https://partyfacts.herokuapp.com/download/external-parties-csv/"
# How LinkedEP speaker URIs appear in multi-europarl.csv (str() of the rdflib URIRef),
# and so in the cleaned splits' `speaker` column.
LINKEDEP_PREFIX = "http://purl.org/linkedpolitics/"
BRIDGE_FROM_YEAR = 2009


def name_key(name):
    """Accent-, case-, punctuation- and order-insensitive speaker/party key."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(sorted(re.sub(r"[^a-z ]", " ", text.lower()).split()))


# --------------------------------------------------------------------------- inputs

def load_partyfacts(path):
    if not path.exists():
        print(f"Downloading PartyFacts external parties -> {path}")
        response = requests.get(PARTYFACTS_URL, timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    pf = pd.read_csv(path, low_memory=False).dropna(subset=["partyfacts_id"])
    pf["dataset_party_id"] = pd.to_numeric(pf["dataset_party_id"], errors="coerce")
    return {key: dict(zip(group["dataset_party_id"], group["partyfacts_id"].astype(int)))
            for key, group in pf.groupby("dataset_key") if key in ("parlgov", "manifesto", "ches")}


def clustered_parties():
    """The EU&I rows k-means clusters (>= 1 MEP), with their PartyFacts id."""
    df = pk.load_parties(PROJECT_ROOT / pk.RAW_CSV)
    df = df[df["meps"] >= 1].copy()
    df["PUI"] = df["PUI"].astype(int)
    df["abbr"] = df["ABBREVIATON"].str.strip()
    df["ches"] = pd.to_numeric(df["CHES_ID"], errors="coerce")
    return df[["PUI", "abbr", "COUNTRY", "ches"]]


def load_parlee():
    cols = ["date", "speechnumber", "speaker", "country", "national_party",
            "parlgov_party", "cmp_party"]
    df = pd.read_csv(PARLEE_CSV, usecols=cols, low_memory=False)
    # speechnumber restarts every sitting ("1-004-000" recurs on 86 dates), so a speech
    # is (date, speechnumber): 362,854 speeches, not the 88,221 speechnumber alone gives.
    df = df.drop_duplicates(["date", "speechnumber"])
    # preprocess_data.py parses ParlEE dates the same way; the lookup must agree with it.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"])


def load_linkedep():
    """speech -> (speaker URI, speaker name, national-party code, date), streamed.

    rdflib would need the whole graph in memory; these three support files are plain
    one-triple-per-line Turtle, so a line regex is enough. The `.` in the NationalParty
    pattern is the Turtle-escaped `\\` before the slash.
    """
    functions, names = {}, {}
    with open(LINKEDEP_DIR / "MembersOfParliament_background.ttl", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"lp:(pf\w+) lpv:institution lp:NationalParty./(\S+) \.", line)
            if m:
                functions[m[1]] = urllib.parse.unquote(urllib.parse.unquote(m[2]))
                continue
            m = re.match(r'lp:(EUmember_\d+) foaf:name "(.*)" \.', line)
            if m:
                names[m[1]] = m[2]
    party = {}
    with open(LINKEDEP_DIR / "spokenAs.ttl", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"lp_eu:(\S+) lpv:spokenAs lp:(pf\w+) \.", line)
            if m and m[2] in functions:
                party[m[1]] = functions[m[2]]
    speaker = {}
    with open(LINKEDEP_DIR / "Events_and_structure.ttl", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"lp_eu:(\S+) lpv:speaker lp:(EUmember_\d+) \.", line)
            if m:
                speaker[m[1]] = m[2]

    df = pd.DataFrame({"speech": list(party), "party": list(party.values())})
    member = df["speech"].map(speaker)
    df["speaker"] = LINKEDEP_PREFIX + member
    df["name"] = member.map(names)
    df["date"] = pd.to_datetime(df["speech"].str[:10], errors="coerce")
    return df.dropna(subset=["speaker", "date"])


# --------------------------------------------------------------------------- mapping

def map_parlee(parlee, eu, partyfacts, overrides):
    ches_to_pui = {}
    for pf_id, rows in eu.assign(pf=eu["ches"].map(partyfacts["ches"])).dropna(subset=["pf"]).groupby("pf"):
        # A PartyFacts id shared by several clustered rows would be a guess; leave it out.
        if len(rows) == 1:
            ches_to_pui[pf_id] = rows["PUI"].iloc[0]
    pf_id = pd.to_numeric(parlee["parlgov_party"], errors="coerce").map(partyfacts["parlgov"])
    pf_id = pf_id.fillna(pd.to_numeric(parlee["cmp_party"], errors="coerce").map(partyfacts["manifesto"]))
    parlee["pui"] = pf_id.map(ches_to_pui)
    parlee["method"] = np.where(parlee["pui"].notna(), "ids", "unmatched")

    table = overrides[overrides["source"] == "parlee"]
    keyed = {(c, name_key(p)): pui for c, p, pui in table[["country", "party", "eu_pui"]].itertuples(index=False)}
    hit = [keyed.get((c, name_key(p))) for c, p in zip(parlee["country"], parlee["national_party"])]
    hit = pd.Series(hit, index=parlee.index, dtype="float")
    parlee.loc[hit.notna(), "pui"] = hit[hit.notna()]
    parlee.loc[hit.notna(), "method"] = "override"

    used = {(c, name_key(p)) for c, p in zip(parlee["country"], parlee["national_party"])}
    unused = [f"{c}: {p}" for c, p in table[["country", "party"]].itertuples(index=False)
              if (c, name_key(p)) not in used]
    if unused:
        print(f"  note: {len(unused)} override row(s) match no ParlEE party: {unused}")
    return parlee


def map_linkedep(linkedep, parlee, overrides, min_share, min_n):
    last_seen = linkedep.groupby("party")["date"].max()
    active = linkedep["party"].map(last_seen).dt.year >= BRIDGE_FROM_YEAR

    # Same speaker, same day, both corpora: the same speech.
    resolved = parlee.dropna(subset=["pui"]).assign(key=lambda d: d["speaker"].map(name_key))
    resolved = resolved.drop_duplicates(["key", "date"])[["key", "date", "pui"]]
    joined = linkedep[active].assign(key=lambda d: d["name"].map(name_key)).merge(
        resolved, on=["key", "date"], how="inner")
    votes = joined.groupby("party")["pui"].agg(
        top=lambda s: s.value_counts().index[0],
        share=lambda s: s.value_counts(normalize=True).iloc[0],
        n="size")
    bridged = votes[(votes["share"] >= min_share) & (votes["n"] >= min_n)]

    linkedep["pui"] = linkedep["party"].map(bridged["top"])
    linkedep["method"] = np.select(
        [~active, linkedep["pui"].notna()], ["dropped-pre2009", "bridge"], "unmatched")

    table = overrides[overrides["source"] == "linkedep"].set_index("party")["eu_pui"]
    hit = linkedep["party"].map(table)
    linkedep.loc[hit.notna(), "pui"] = hit[hit.notna()]
    linkedep.loc[hit.notna(), "method"] = "override"
    return linkedep, votes


def party_table(parlee, linkedep, votes, eu):
    abbr = eu.set_index("PUI")["abbr"]
    rows = [
        parlee.groupby(["country", "national_party"], dropna=False).agg(
            speeches=("date", "size"), first=("date", "min"), last=("date", "max"),
            pui=("pui", "first"), method=("method", "first"))
        .reset_index().rename(columns={"national_party": "party"}).assign(source="parlee"),
        linkedep.groupby("party").agg(
            speeches=("date", "size"), first=("date", "min"), last=("date", "max"),
            pui=("pui", "first"), method=("method", "first"))
        .join(votes.rename(columns={"top": "bridge_pui", "share": "bridge_share", "n": "bridge_n"}))
        .reset_index().assign(source="linkedep", country=None),
    ]
    table = pd.concat(rows, ignore_index=True)
    table["eu_abbr"] = table["pui"].map(abbr)
    return table.sort_values(["source", "speeches"], ascending=[True, False])


def coverage(parlee, linkedep):
    report = {}
    for source, df in (("parlee", parlee), ("linkedep", linkedep)):
        uk = df["party"].str.endswith("_UnitedKingdom") if source == "linkedep" else df["country"].eq("UK")
        report[source] = {
            "speeches": int(len(df)),
            "mapped": round(float(df["pui"].notna().mean()), 4),
            "mapped_excl_uk": round(float(df.loc[~uk, "pui"].notna().mean()), 4),
            "by_method": {k: int(v) for k, v in df["method"].value_counts().items()},
            "mapped_by_year": {int(y): round(float(v), 3) for y, v in
                               df.groupby(df["date"].dt.year)["pui"].apply(lambda s: s.notna().mean()).items()},
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bridge-min-share", type=float, default=0.8,
                        help="share of a LinkedEP party's bridged speeches that must agree")
    parser.add_argument("--bridge-min-n", type=int, default=3,
                        help="bridged speeches a LinkedEP party needs")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partyfacts = load_partyfacts(args.output_dir / "partyfacts_external.csv")
    overrides = pd.read_csv(OVERRIDES_CSV)
    eu = clustered_parties()
    unknown = set(overrides["eu_pui"]) - set(eu["PUI"])
    if unknown:
        raise SystemExit(f"{OVERRIDES_CSV.name} targets PUIs that are not clustered parties: {sorted(unknown)}")

    print("Reading ParlEE ...")
    parlee = map_parlee(load_parlee(), eu, partyfacts, overrides)
    print("Reading LinkedEP support files ...")
    linkedep, votes = map_linkedep(load_linkedep(), parlee, overrides,
                                   args.bridge_min_share, args.bridge_min_n)

    speeches = pd.concat([
        parlee.assign(source="parlee", name=parlee["speaker"], party=parlee["national_party"]),
        linkedep.assign(source="linkedep"),
    ], ignore_index=True)
    speeches["speaker_key"] = speeches["name"].map(name_key)
    speeches = speeches[["source", "speaker", "speaker_key", "date", "party", "pui", "method"]]
    speeches["pui"] = speeches["pui"].astype("Int64")
    speeches.to_parquet(args.output_dir / "speeches.parquet", index=False)

    table = party_table(parlee, linkedep, votes, eu)
    table.to_csv(args.output_dir / "party_map.csv", index=False)
    report = coverage(parlee, linkedep)
    (args.output_dir / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for source, entry in report.items():
        print(f"\n{source}: {entry['speeches']:,} speeches, mapped {entry['mapped']:.1%} "
              f"({entry['mapped_excl_uk']:.1%} excluding UK); {entry['by_method']}")
    print(f"\nWrote speeches.parquet, party_map.csv, coverage.json to {args.output_dir}")


if __name__ == "__main__":
    main()
