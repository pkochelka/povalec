#!/usr/bin/env python3
"""A two-axis compass of the euandi EP group positions alone.

plot_party_compass.py draws the same groups as coloured means with covariance ellipses
over their member parties. This is the same projection with everything else stripped
out: one black star per EP group, labelled in bold, on whichever two questionnaire axes
are asked for (Ukraine x Immigration by default).

A coordinate on an axis is the mean of the euandi answers to the statements loading on
that axis, each signed by the direction coded in the questionnaire -- the same
projection build_party_positions uses. Both default axes are one-sided (every statement
loads with the same sign), which setup_compass flags on the axis label.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS_DIR.parent))
sys.path.insert(0, str(_ANALYSIS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_euandi import EP_GROUP_BY_PARTY
from plot_political_bias import load_questionnaire, setup_compass
from plot_party_compass import EP_GROUP_ORDER, load_parties, party_stance_vector

STAR_SIZE = 520
LABEL_FONTSIZE = 12
LIMIT_PADDING = 1.18
# Labels are placed around the marker; coincident parties share one star, so these
# only have to separate genuinely nearby points.
LABEL_OFFSETS = [(9, 9), (9, -14), (-9, 9), (-9, -14)]
# Which answers stand for an EP group, in the vocabulary evaluate_euandi.py uses:
#   ep-group   -- the europarty's own euandi answers, one position vector per group
#   national   -- its member parties in the five countries euandi_2024_parties.jsonl
#                 covers (DE, FR, IT, ES, GR), averaged within the group
#   group-mean -- every national party that ran in 2024, averaged per group; read from
#                 the file build_group_positions.py writes, not recomputed here
POSITION_CHOICES = ["ep-group", "national", "group-mean"]
DEFAULT_POSITIONS = "ep-group"

# Seven axes plus a group column do not fit a text-width table under their full names.
AXIS_ABBREVIATION = {
    "Ukraine": "UA", "Ecology": "Eco", "Immigration": "Imm", "Values": "Val",
    "Economy": "Econ", "Europe": "Eu", "Left-Right": "L-R",
}
TABCOLSEP = "2.75pt"


def position_rows(dataset, basis):
    """The euandi answer vectors the basis is built from: the parties file for
    `ep-group` and `national`, the precomputed group means for `group-mean`."""
    if basis != "group-mean":
        return load_parties(dataset)
    path = Path("data") / f"{dataset}_data" / f"{dataset}_group_positions.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found; build it with analysis/build_group_positions.py.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def party_positions(rows, dims, questionnaire):
    """One row per euandi party: its name, country, EP group and axis coordinates."""
    signs = questionnaire[dims].to_numpy(dtype=float)
    records = []
    for party in rows:
        stance = party_stance_vector(party)
        length = min(len(stance), len(signs))
        stance, party_signs = stance[:length], signs[:length]
        record = {
            "short_name": party["short_name"],
            "country_iso": party["country_iso"],
            "ep_group": EP_GROUP_BY_PARTY.get(party["short_name"], "Other"),
        }
        for column, dimension in enumerate(dims):
            sign = party_signs[:, column]
            projected = np.where(sign != 0, stance * sign, np.nan)
            record[dimension] = np.nanmean(projected) if np.any(~np.isnan(projected)) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def ep_group_positions(positions, dims, basis):
    """One row per EP group. `ep-group` takes the europarty's own answers and
    `group-mean` the precomputed member means -- both are already one row per group --
    while `national` averages the member parties here. Parties outside the seven groups
    are dropped."""
    in_group = positions[positions["ep_group"] != "Other"]
    if basis in ("ep-group", "group-mean"):
        selected = in_group[in_group["country_iso"] == "eu"]
        return selected[["ep_group", *dims]].reset_index(drop=True)

    members = in_group[in_group["country_iso"] != "eu"]
    grouped = members.groupby("ep_group")[dims].mean().reset_index()
    return grouped


def merge_coincident(positions, x_dim, y_dim, label_column):
    """[(x, y, [labels])] -- groups landing on the same point share one star.

    With only a handful of statements per axis the coordinates are coarse and exact
    ties happen (PPE and ECR both sit on (1, 1) for Ukraine x Immigration); drawing
    their labels separately would just overprint them."""
    usable = positions.dropna(subset=[x_dim, y_dim])
    merged = []
    for (x, y), group in usable.groupby([usable[x_dim].round(6), usable[y_dim].round(6)]):
        merged.append((float(x), float(y), list(group[label_column])))
    return sorted(merged)


def label_text(names):
    return "\n".join(names)


def ordered_groups(groups):
    present = set(groups["ep_group"])
    ordered = [group for group in EP_GROUP_ORDER if group in present and group != "Other"]
    return ordered + sorted(present - set(ordered) - {"Other"})


def format_position(value):
    return "--" if pd.isna(value) else f"{value:+.2f}"


def position_table_rows(groups, dims):
    indexed = groups.set_index("ep_group")
    return [
        [group, *(format_position(indexed.at[group, dimension]) for dimension in dims)]
        for group in ordered_groups(groups)
    ]


def statements_per_axis(questionnaire, dims):
    return {dimension: int((questionnaire[dimension] != 0).sum()) for dimension in dims}


def axis_header(dims, counts, latex):
    """Each axis abbreviated, carrying how many statements it rests on -- a column built
    from two statements is not the same evidence as one built from twelve, and that has
    to travel with the numbers.

    `counts=None` leaves the abbreviations bare, for a table that carries the statement
    counts elsewhere or not at all."""
    return ["Group"] + [
        f"{AXIS_ABBREVIATION.get(dimension, dimension)}"
        + ("" if counts is None else
           (f"$^{{{counts[dimension]}}}$" if latex else f" ({counts[dimension]})"))
        for dimension in dims
    ]


def one_sided_note(dims, one_sided):
    """Which columns are one-sided, named by position when they are the leading ones --
    the header is too narrow to carry a marker as well as the axis abbreviation."""
    present = [dimension for dimension in dims if dimension in one_sided]
    if not present:
        return None
    leading = dims[:len(present)]
    where = (f"The first {len(present)} columns are" if present == leading
             else f"{', '.join(present)} are")
    return (f"{where} one-sided axes: every statement loads with the same sign, so the "
            f"scale measures agreement with one pole rather than a contrast between two.")


def render_markdown_table(header, rows, notes):
    widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join(lines) + "\n\n" + "\n".join(notes)


def render_latex_table(header, rows, caption, label, provenance):
    alignment = "l" + "r" * (len(header) - 1)
    lines = [f"% {line}" for line in provenance]
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\setlength{{\\tabcolsep}}{{{TABCOLSEP}}}",
        f"  \\begin{{tabular}}{{{alignment}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + " & ".join(row) + r" \\" for row in rows]
    lines += [r"    \bottomrule", r"  \end{tabular}",
              f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def write_position_table(groups, dims, one_sided, basis, dataset, tables_dir):
    rows = position_table_rows(groups, dims)
    notes = [
        f"Each cell is the mean of the group's euandi answers to the statements loading "
        f"on that axis, signed by the direction coded in the questionnaire; +1 is the "
        f"axis' positive pole, -1 its negative one. Basis: {basis}.",
    ]
    if (one_sided_text := one_sided_note(dims, one_sided)):
        notes.append(one_sided_text)

    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"ep_group_positions_{basis}.csv"
    groups.set_index("ep_group").reindex(ordered_groups(groups))[dims].to_csv(csv_path)
    print(f"  Wrote {csv_path}")

    tex_path = tables_dir / f"ep_group_positions_{basis}.tex"
    tex_path.write_text(render_latex_table(
        [column.replace("&", r"\&") for column in axis_header(dims, None, latex=True)],
        [[cell.replace("&", r"\&") for cell in row] for row in rows],
        caption=" ".join(notes).replace("%", r"\%"),
        label=f"tab:ep-group-positions-{basis}",
        provenance=[
            "Generated by analysis/plotting/plot_party_axis_compass.py -- do not edit by hand.",
            f"dataset={dataset}  positions={basis}  axes={','.join(dims)}",
        ],
    ) + "\n", encoding="utf-8")
    print(f"  Wrote {tex_path}")

    print()
    print(render_markdown_table(axis_header(dims, None, latex=False), rows, notes))


def plot_party_compass(merged, x_dim, y_dim, one_sided, out_path, subtitle):
    fig, ax = plt.subplots(figsize=(11, 11))
    setup_compass(ax, x_dim, y_dim, one_sided)
    # Both axes are bounded at +-1 and the parties pile up on those bounds, so the
    # panel is widened past them to leave the labels somewhere to sit.
    ax.set_xlim(-LIMIT_PADDING, LIMIT_PADDING)
    ax.set_ylim(-LIMIT_PADDING, LIMIT_PADDING)

    for index, (x, y, names) in enumerate(merged):
        ax.scatter(x, y, s=STAR_SIZE, marker="*", color="black", zorder=5)
        offset = LABEL_OFFSETS[index % len(LABEL_OFFSETS)]
        ax.annotate(
            label_text(names), (x, y), xytext=offset, textcoords="offset points",
            fontsize=LABEL_FONTSIZE, fontweight="bold", color="black", zorder=9,
            path_effects=[matplotlib.patheffects.withStroke(
                linewidth=3.0, foreground="white")],
        )

    ax.set_title(subtitle, fontsize=11, pad=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--x-dim", default="Ukraine")
    parser.add_argument("--y-dim", default="Immigration")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Whose euandi answers stand for an EP group: the europarty's "
                             "own, or its national member parties averaged.")
    return parser.parse_args()


def main():
    args = parse_args()
    questionnaire, dims, one_sided = load_questionnaire(args.dataset)
    for dimension in (args.x_dim, args.y_dim):
        if dimension not in dims:
            raise SystemExit(f"Dimension {dimension!r} not in questionnaire. Available: {dims}")

    # The figure shows two axes; the table alongside it reports all of them.
    positions = party_positions(position_rows(args.dataset, args.positions), dims, questionnaire)
    groups = ep_group_positions(positions, dims, args.positions)
    if groups.empty:
        raise SystemExit(f"No EP groups left on the {args.positions} basis.")

    results_dir = Path("data") / f"{args.dataset}_results"
    write_position_table(groups, dims, one_sided, args.positions,
                         args.dataset, results_dir / "tables")
    print()

    merged = merge_coincident(groups, args.x_dim, args.y_dim, "ep_group")
    drawn = sum(len(names) for _, _, names in merged)
    print(f"{drawn} EP groups ({args.positions}) on {len(merged)} distinct points.")
    if drawn < len(groups):
        print(f"  {len(groups) - drawn} group(s) have no answer on one of the axes.")

    out_dir = results_dir / "plots"
    plot_party_compass(
        merged, args.x_dim, args.y_dim, one_sided,
        out_dir / f"ep_group_stars_{args.x_dim}_{args.y_dim}_{args.positions}.png",
        subtitle=f"euandi {args.dataset.split('_')[-1]} EP group positions "
                 f"({drawn} groups, {args.positions} basis)",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
