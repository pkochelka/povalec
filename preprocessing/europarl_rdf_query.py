"""Query the LinkedEP RDF dumps into the wide multiparallel CSV.

Original author: Paul Lerner
Source:  https://github.com/PaulLerner/21-EuroParl/blob/main/data/rdf.py
Licence: MIT, Copyright (c) 2025 Paul Lerner -- see preprocessing/LICENSE.21-EuroParl
Paper:   Lerner and Yvon (2025), "Assessing the Political Fairness of Multilingual
         LLMs: A Case Study based on a 21-way Multiparallel EuroParl Dataset"

Input data: LinkedEP / Talk of Europe, van Aggelen et al. (2016), DANS
DOI 10.17026/dans-x62-ew3m, CC0-1.0. Fetch it with fetch_raw_data.py.

Adapted for this project: the three SPARQL queries and the output shape are
Lerner's, unchanged. What changed here is packaging only -- the script was moved
out of the gitignored data directory into preprocessing/, the jsonargparse CLI
was replaced with the repo-standard argparse, and --languages was added so the
graph can be built on a subset of the language dumps.

WHERE THIS PIPELINE STOPS
-------------------------
Lerner's pipeline continues past this point: bertalign sentence alignment
(github.com/PaulLerner/bertalign) and then merge_align.ipynb, which reduce the
corpus to sentence-aligned multiparallel tuples. We deliberately DO NOT run
those steps. We keep every speech, at speech granularity, in
multi-europarl-lang_id.csv -- the party classifier wants volume and natural
per-language coverage, not alignment. Do not "finish" the pipeline: it would
throw away most of the training data.

    LinkedEP *.ttl -> [this script] -> multi-europarl.csv
                   -> europarl_lid_filter.py -> multi-europarl-lang_id.csv
                   -> preprocess_data.py (STOP -- no alignment)

MEMORY
------
parse() loads every *.ttl in the input directory into a single in-memory
rdflib.Graph. At full scale that is ~7.8 GB of Turtle and the dominant cost of
the whole rebuild -- expect many hours and a very large RAM requirement. Use
--languages to build the graph on a subset when testing.

Usage
-----
    python preprocessing/europarl_rdf_query.py --languages en de
    python preprocessing/europarl_rdf_query.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdflib import Graph
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_data import LANG_NAME_TO_CODE

DATA_DIR = PROJECT_ROOT / "data" / "EuroParl Custom"
DEFAULT_LINKEDEP_DIR = DATA_DIR / "data" / "linkedEP"
DEFAULT_OUTPUT_DIR = DATA_DIR

# Non-language dumps that carry the speaker, party and agenda triples; always
# parsed, regardless of --languages.
SUPPORT_STEMS = frozenset({
    "Events_and_structure", "spokenAs", "MembersOfParliament_background",
    "vocabulary", "Countries_in_Geonames",
})


def select_ttl_files(root, languages):
    """Root-level *.ttl to parse. Non-recursive: topics/ is not part of the query."""
    paths = sorted(root.glob("*.ttl"))
    if not languages:
        return paths
    keep = {name for name, code in LANG_NAME_TO_CODE.items() if code in set(languages)}
    return [path for path in paths
            if path.stem in keep or path.stem in SUPPORT_STEMS]


def parse(paths):
    g = Graph()
    for path in tqdm(paths):
        g.parse(path)
        print(path.name, len(g))
    return g


def verbose_query(g, query):
    print(f"Querying...\n{query}\n")
    result = g.query(query)
    print(f"Got {len(result)} results")
    return result


def multiparallel(g):
    """
    Main query for multiparallel data.
    Retrieves, for a given speech:
        - all of its translations
        - its source language
        - speaker identifier
        - for each type of party:
            - party identifier
        - national party identifier
        - its date
    """
    query="""
    SELECT ?text ?translation ?speech ?party ?partytypelabel ?date ?speaker
    WHERE {
        ?speech lpv:spokenText ?text.
        ?speech dcterms:isPartOf ?agenda.
        ?agenda dcterms:date ?date.
        ?speech lpv:spokenAs ?function.
        ?function lpv:institution ?party.
        ?party rdf:type ?partytype.
        ?partytype rdfs:label ?partytypelabel.
        ?speech lpv:speaker ?speaker.
        ?speech lpv:translatedText ?translation.
    }
    """
    df = {}
    result = verbose_query(g, query)
    for row in result:
        df.setdefault(row.speech, {})
        df[row.speech][row.text.language] = row.text.value
        df[row.speech][row.translation.language] = row.translation.value
        df[row.speech]["src_lang"] = row.text.language
        df[row.speech]["date"] = str(row.date.value)
        df[row.speech]["speaker"] = str(row.speaker)
        df[row.speech][row.partytypelabel] = str(row.party)

    df = pd.DataFrame(df).T
    print(len(df), df.columns)
    return df


def query_speakers(g):
    query = """
    SELECT ?speaker ?name ?dob ?country ?countryLabel
    WHERE {
        ?speaker foaf:name ?name.
        ?speaker lpv:dateOfBirth ?dob.
        ?speaker lpv:countryOfRepresentation ?country.
        ?country rdfs:label ?countryLabel.
    }
    """
    speakers = {}
    result = verbose_query(g, query)
    for row in result:
        speaker = str(row.speaker)
        speakers[speaker] = {}
        speakers[speaker]["name"] = row.name.value
        speakers[speaker]["dateOfBirth"] = str(row.dob.value)
        speakers[speaker]["countryOfRepresentation"] = str(row.country)
        speakers[speaker]["countryOfRepresentationLabel"] = row.countryLabel.value
    print(len(speakers))
    return speakers


def query_parties(g):
    query = """
    SELECT ?party ?partylabel ?partytypelabel ?partyAcronym
    WHERE {
        ?party rdf:type ?partytype.
        ?party rdfs:label ?partylabel.
        ?partytype rdfs:label ?partytypelabel.
        ?party lpv:acronym ?partyAcronym.
    }
    """
    parties = {}
    result = verbose_query(g, query)
    for row in result:
        party = str(row.party)
        parties[party] = {}
        parties[party]["label"] = row.partylabel.value
        parties[party]["type"] = row.partytypelabel.value
        parties[party]["acronym"] = row.partyAcronym.value
    return parties


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--linkedep-dir", type=Path, default=DEFAULT_LINKEDEP_DIR,
                        help=f"directory holding the *.ttl dumps "
                             f"(default {DEFAULT_LINKEDEP_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"default {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--languages", nargs="+", metavar="CODE",
                        help="restrict the graph to these language dumps, as 2-letter "
                             "codes (default: all). Support dumps are always parsed.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = select_ttl_files(args.linkedep_dir, args.languages)
    if not paths:
        raise RuntimeError(f"no *.ttl found in {args.linkedep_dir}; "
                           f"run preprocessing/fetch_raw_data.py --source linkedep")
    print(f"Parsing {len(paths)} Turtle files from {args.linkedep_dir}")

    g = parse(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = multiparallel(g)
    df.to_csv(args.output_dir / "multi-europarl.csv")
    speakers = query_speakers(g)
    with open(args.output_dir / "speakers.json", "wt") as file:
        json.dump(speakers, file)
    parties = query_parties(g)
    with open(args.output_dir / "parties.json", "wt") as file:
        json.dump(parties, file)
    print(f"Wrote multi-europarl.csv, speakers.json, parties.json to {args.output_dir}")


if __name__ == "__main__":
    main()
