#!/usr/bin/env python3
"""Put the EU&I positions and questionnaire into the order the results are written in.

Three files describe the same 30 statements and two of them disagree about the order.
`statements.jsonl` is what `scrape_euandi.py` writes and what every results CSV is laid
out in -- row *i* of a results CSV holds the answers to `statements.jsonl[i]`, which its
`original_text_en` column records. `euandi_2024_parties.jsonl` and
`euandi_2024_questionnaire.jsonl` are in the EU&I codebook order instead, and the two
orders agree on only 11 of the 30 statements.

Nothing in the data marks the difference: both sides number statements 0..29 (the
questionnaire 1..30), so every `merge(..., on="statement_idx")` in the analysis code
succeeds and silently pairs a model's answers about one statement with the groups'
official positions on another -- AI regulation scored against protecting farmers, abortion
against immigration, for 19 of the 30 statements.

This script rewrites the two codebook-order files into `statements.jsonl` order, once, so
that `statement_idx` means the same thing everywhere and the existing positional joins
become correct instead of being replaced by text joins scattered across six call sites.

What it guarantees:

  parties        `statement_idx` becomes the results row. `statement` becomes the
                 `statements.jsonl` wording, so downstream text joins are exact; the
                 codebook's own index and wording are preserved on every response as
                 `statement_codebook_idx` / `statement_codebook_text`.
  questionnaire  rows are reordered the same way and renumbered, but stay **1-based**.
                 The pipeline's existing `statement_idx - 1` convention
                 (`rank_consistency_tables.load_statement_topics`,
                 `plot_party_axis_compass`) then lands on the right results row without
                 any code change. Its `statement` text is already the scraped wording and
                 is left alone.

Provenance: the first run moves each original to `*_codebook_order.jsonl` and every run
reads from there, so it is idempotent and re-runnable, and the raw source stays in the
repository (and in git history) rather than being overwritten in place.

Two statements are worded differently in the codebook files than in `statements.jsonl`
("EU member states should increase military spending" against "European Union should
increase military spending"). That is not drift between snapshots: `scrape_euandi.py`
rewrites the country name out of the English text with a literal
`replace(country_name, "European Union")`. Those two are matched by similarity, with a
margin over the runner-up so an ambiguous case raises instead of being guessed, and every
inexact match is printed for review.

Run this before `analysis/analyze_all.py`. Changing the positions file makes it newer
than every `vaa_*.csv`, and `analyze_all`'s staleness check lists the positions path as an
input to `evaluate_euandi.py`, so the VAA layer and everything downstream re-derive on the
next run while the cross-encoder and classifier steps -- which do not read positions --
are skipped.

Usage, from the repository root:
  python statement_collection/align_statement_order.py --check
  python statement_collection/align_statement_order.py
"""
import argparse
import difflib
import json
from pathlib import Path

DATA = Path("data")
STATEMENTS = "statements.jsonl"
PARTIES = "{dataset}_parties.jsonl"
QUESTIONNAIRE = "{dataset}_questionnaire.jsonl"
PRESERVED = "_codebook_order"


def normalize(text):
    return " ".join(str(text).split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def source_path(directory, name):
    """The codebook-order original, moved aside on the first run.

    Reading from the preserved copy rather than from the canonical path is what makes
    this idempotent: running twice reorders the same source twice and lands in the same
    place, instead of reordering an already-reordered file.
    """
    canonical = directory / name
    preserved = directory / name.replace(".jsonl", f"{PRESERVED}.jsonl")
    if preserved.exists():
        return preserved
    if not canonical.exists():
        raise SystemExit(f"{canonical} not found.")
    canonical.rename(preserved)
    print(f"  preserved original as {preserved.name}")
    return preserved


def build_order(statements, wordings, cutoff, margin=0.10):
    """{codebook row -> results row}, joined on statement text.

    The codebook rows are the authority on *which* statements exist and the
    `statements.jsonl` order is the authority on *where* each one goes.

    `wordings` maps a codebook row to *every* wording the parties file gives it, because
    it gives some of them more than one: the national parties carry the scraped English
    ("Immigration into European Union...") and the europarties the original ("...into EU
    member states..."). Matching against all of them keeps the exact-match path working
    regardless of which party happens to come first in the file.
    """
    pool = {}
    for row, texts in wordings.items():
        for text in texts:
            if normalize(text) not in ("nan", "none", ""):
                pool.setdefault(normalize(text), row)
    order, inexact = {}, []
    for row, statement in enumerate(statements):
        key = normalize(statement)
        if key in pool:
            order[pool[key]] = row
            continue
        scored = sorted(((difflib.SequenceMatcher(None, key, candidate).ratio(), candidate)
                         for candidate in pool), reverse=True)
        if not scored or scored[0][0] < cutoff:
            best = f" (best {scored[0][0]:.2f}: {scored[0][1]!r})" if scored else ""
            raise SystemExit(f"No codebook statement matches {key!r}{best}.")
        if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
            raise SystemExit(f"{key!r} matches {scored[0][1]!r} and {scored[1][1]!r} about "
                             f"equally well ({scored[0][0]:.2f} vs {scored[1][0]:.2f}); "
                             "refusing to guess.")
        order[pool[scored[0][1]]] = row
        inexact.append((scored[0][0], key, scored[0][1]))

    if len(order) != len(statements):
        raise SystemExit(f"{len(statements)} statements but {len(order)} codebook rows "
                         "matched; the join is not one-to-one.")
    for score, wanted, found in inexact:
        print(f"  matched by wording ({score:.2f}): {found!r}\n"
              f"                  -> {wanted!r}")
    return order


def align_parties(source, statements, order):
    """Reindex every party's responses onto the results row order."""
    aligned = []
    for party in read_jsonl(source):
        responses = []
        for response in party.get("responses", []):
            codebook_idx = int(response["statement_idx"])
            if codebook_idx not in order:
                continue
            row = order[codebook_idx]
            responses.append({**response,
                              "statement_idx": row,
                              "statement": statements[row],
                              "statement_codebook_idx": codebook_idx,
                              "statement_codebook_text": response.get("statement")})
        responses.sort(key=lambda r: r["statement_idx"])
        aligned.append({**{k: v for k, v in party.items() if k != "responses"},
                        "responses": responses})
    return aligned


def align_questionnaire(source, order):
    """Reorder the axis coding, keeping the file's 1-based numbering.

    No provenance field is added here, unlike the parties file. `plot_political_bias.
    load_questionnaire` decides what an axis is by scanning for numeric columns it does
    not recognise, so any extra number in this file becomes an eighth policy axis and
    shows up as a column in Table 4. The codebook index stays in the preserved
    `*_codebook_order.jsonl` copy instead.
    """
    rows = read_jsonl(source)
    aligned = []
    for row in rows:
        codebook_idx = int(row["statement_idx"]) - 1   # the file is 1-based
        if codebook_idx not in order:
            raise SystemExit(f"Questionnaire row {row['statement_idx']} has no place in "
                             "the results order.")
        aligned.append({**row, "statement_idx": order[codebook_idx] + 1})
    aligned.sort(key=lambda r: r["statement_idx"])
    return aligned


def verify(dataset, statements):
    """Confirm both canonical files now agree with the results order.

    The parties check reads `statement_codebook_text` -- the wording the source file
    actually had -- not the `statement` this script wrote, which would only confirm that
    an assignment was copied correctly.

    The test is that the original wording is a *better* match for the statement it was
    assigned to than for any of the other 29. That beats a fixed similarity threshold:
    the two reworded statements only score 0.73 against their true counterpart, which no
    useful threshold would admit while still rejecting a wrong assignment, but they are
    still far and away the closest of the thirty.
    """
    directory = DATA / f"{dataset}_data"
    problems = []
    targets = [normalize(s) for s in statements]

    best_row = {}

    def closest(text):
        if text not in best_row:
            scores = [difflib.SequenceMatcher(None, text, target).ratio()
                      for target in targets]
            best_row[text] = max(range(len(scores)), key=scores.__getitem__)
        return best_row[text]

    parties = read_jsonl(directory / PARTIES.format(dataset=dataset))
    for party in parties:
        for response in party.get("responses", []):
            row = int(response["statement_idx"])
            if not 0 <= row < len(statements):
                problems.append(f"{party['short_name']}: statement_idx {row} out of range")
                continue
            original = normalize(response.get("statement_codebook_text"))
            if original.lower() in ("nan", "none", ""):
                continue          # the source leaves a few answers unworded
            if closest(original) != row:
                problems.append(f"{party['short_name']} row {row}: {original!r} is closer "
                                f"to {targets[closest(original)]!r}")
    questionnaire = read_jsonl(directory / QUESTIONNAIRE.format(dataset=dataset))
    for row in questionnaire:
        index = int(row["statement_idx"]) - 1
        if normalize(row["statement"]) != normalize(statements[index]):
            problems.append(f"questionnaire {row['statement_idx']}: "
                            f"{normalize(row['statement'])!r}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--cutoff", type=float, default=0.75,
                        help="similarity a reworded statement needs to be matched")
    parser.add_argument("--check", action="store_true",
                        help="only report whether the files are already aligned")
    args = parser.parse_args()

    directory = DATA / f"{args.dataset}_data"
    statements = [record["statement"]["en"]
                  for record in read_jsonl(directory / STATEMENTS)]
    print(f"{len(statements)} statements in {STATEMENTS} (the results row order)")

    if args.check:
        problems = verify(args.dataset, statements)
        if problems:
            print(f"\nNOT aligned: {len(problems)} mismatch(es); first few:")
            for problem in problems[:5]:
                print(f"  {problem}")
            raise SystemExit(1)
        print("\nAligned: every positions and questionnaire row matches its results row.")
        return

    parties_source = source_path(directory, PARTIES.format(dataset=args.dataset))
    questionnaire_source = source_path(directory, QUESTIONNAIRE.format(dataset=args.dataset))

    wordings = {}
    for party in read_jsonl(parties_source):
        for response in party.get("responses", []):
            wordings.setdefault(int(response["statement_idx"]), set()).add(
                response.get("statement"))
    order = build_order(statements, wordings, args.cutoff)
    moved = sum(1 for codebook_idx, row in order.items() if codebook_idx != row)
    print(f"  {moved} of {len(order)} statements change position")

    write_jsonl(directory / PARTIES.format(dataset=args.dataset),
                align_parties(parties_source, statements, order))
    write_jsonl(directory / QUESTIONNAIRE.format(dataset=args.dataset),
                align_questionnaire(questionnaire_source, order))
    print(f"  wrote {PARTIES.format(dataset=args.dataset)} and "
          f"{QUESTIONNAIRE.format(dataset=args.dataset)}")

    problems = verify(args.dataset, statements)
    if problems:
        print(f"\nVerification FAILED: {len(problems)} mismatch(es); first few:")
        for problem in problems[:5]:
            print(f"  {problem}")
        raise SystemExit(1)
    print("\nVerified: every positions and questionnaire row matches its results row.")
    print("Now re-run: python analysis/analyze_all.py")


if __name__ == "__main__":
    main()
