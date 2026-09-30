#!/usr/bin/env python3
"""Average each EP group's member parties into one position vector per group.

`--positions group-mean` is the third basis `analysis/core/positions.py` offers, and the
only one that is not read straight out of `euandi_2024_parties.jsonl`: it needs one
synthetic answer vector per EP group, which this script precomputes into
`data/{dataset}_data/{dataset}_group_positions.jsonl`.

Why it is not just `national` again. Both start from the same member parties, but they
average at different points, and `agreement = 1 - |position - stance| / 2` is not linear
through the absolute value:

    national    agreement is computed against every member party separately and the
                results averaged, so a group is only as close to the model as its
                parties are on average, and a group with six member parties contributes
                six rows against another's three.
    group-mean  the member parties are averaged *first*, into one position per
                statement, and the model is compared against that single vector. Every
                group then weighs the same regardless of how many of its parties EU&I
                covers, and within-group disagreement cancels instead of accumulating.

A group whose members split evenly on a statement lands near 0 here -- an absence of a
common position -- where `national` would record every member's distance from the model
and report the group as uniformly mediocre. Which of the two is wanted is a question
about the claim being made, so both remain available.

**Scope.** The average is over the member parties `euandi_2024_parties.jsonl` covers,
which is the five countries EU&I published party answers for in this snapshot (DE, ES,
FR, GR, IT) -- 28 parties across the seven groups, not every national party that ran in
2024. Parties outside the seven groups (KKE, Niki) are dropped, matching the
`ep_group != "Other"` filter every consumer already applies.

The output is written in `euandi_2024_parties.jsonl`'s own schema, with `country_iso`
set to `"eu"` and `short_name` set to the group's europarty, so that
`load_party_positions` selects these rows and `EP_GROUP_BY_PARTY` maps them back onto
the group without any special case. Statement text is copied from `statements.jsonl`, so
the file satisfies `core.positions.check_statement_order` by construction.

Re-run it after anything changes the statement order; it is idempotent.
"""
import argparse
import json
from pathlib import Path

from utils import EP_GROUP_BY_PARTY, EUROPARTY_BY_EP_GROUP

# The mean is reported to this many places. The inputs are on a five-point grid
# (-1, -0.5, 0, 0.5, 1), so the averages are exact ratios and rounding only keeps
# the file from carrying float noise that would make diffs unreadable.
PRECISION = 6


def statement_texts(data_dir: Path) -> list[str]:
    """Row *i* of `statements.jsonl`, in English -- the order everything else uses."""
    path = data_dir / "statements.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["statement"]["en"] for line in f if line.strip()]


def load_parties(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def member_parties(parties: list[dict]) -> dict[str, list[dict]]:
    """EP group -> its national member parties. Europarties and unmapped parties drop."""
    members: dict[str, list[dict]] = {}
    for party in parties:
        if party.get("country_iso") == "eu":
            continue
        group = EP_GROUP_BY_PARTY.get(party["short_name"])
        if group is None:
            continue
        members.setdefault(group, []).append(party)
    return members


def mean_responses(members: list[dict], statements: list[str]) -> list[dict]:
    """One averaged response per statement index, over the members that answered it.

    A statement no member answered keeps a `normalized_answer` of `null` rather than a
    zero: zero is a real position on this scale ("neutral") and inventing it would pull
    the group toward the centre of every axis the statement loads on.
    """
    answers: dict[int, list[float]] = {}
    for party in members:
        for response in party["responses"]:
            value = response.get("normalized_answer")
            if isinstance(value, (int, float)):
                answers.setdefault(int(response["statement_idx"]), []).append(float(value))

    responses = []
    for idx, statement in enumerate(statements):
        values = answers.get(idx, [])
        responses.append({
            "statement_idx": idx,
            "statement": statement,
            # Provenance: how many of the group's member parties actually answered, so a
            # mean resting on one party is visible in the file rather than only in the
            # spread of the numbers.
            "n_answered": len(values),
            "normalized_answer": round(sum(values) / len(values), PRECISION) if values else None,
        })
    return responses


def build(data_dir: Path) -> list[dict]:
    statements = statement_texts(data_dir)
    parties = load_parties(data_dir / f"{data_dir.name.removesuffix('_data')}_parties.jsonl")
    members = member_parties(parties)

    records = []
    # EUROPARTY_BY_EP_GROUP fixes the group order and is the invertible half of the
    # mapping, so iterating it keeps the output stable and guarantees every short_name
    # written here maps back through EP_GROUP_BY_PARTY.
    for group, europarty in EUROPARTY_BY_EP_GROUP.items():
        group_members = members.get(group, [])
        if not group_members:
            continue
        names = sorted(party["short_name"] for party in group_members)
        countries = sorted({party["country_iso"] for party in group_members})
        records.append({
            "short_name": europarty,
            "full_name": f"{group} member-party mean ({len(names)} parties: {', '.join(names)})",
            "country_iso": "eu",
            "country": "eu",
            "member_parties": names,
            "member_countries": countries,
            "responses": mean_responses(group_members, statements),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--output", help="default: data/{dataset}_data/{dataset}_group_positions.jsonl")
    args = parser.parse_args()

    data_dir = Path("data") / f"{args.dataset}_data"
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} not found (run from the repo root)")

    records = build(data_dir)
    out_path = Path(args.output) if args.output else data_dir / f"{args.dataset}_group_positions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(records)} EP groups)")
    for record in records:
        answered = sum(1 for r in record["responses"] if r["normalized_answer"] is not None)
        group = EP_GROUP_BY_PARTY[record["short_name"]]
        print(f"  {group:<11} {len(record['member_parties'])} parties "
              f"({', '.join(record['member_countries'])}), "
              f"{answered}/{len(record['responses'])} statements answered")


if __name__ == "__main__":
    main()
