#!/usr/bin/env python3
"""The paper's compact cross-method consistency table, one per null.

Two rows, base and negated framing; three columns, Kendall's W, ICC(3,1) and argmax
consistency over the measurement methods' cluster rankings, pooled over models, each
with its 95% statement-bootstrap interval in square brackets:

    Base    & \\Corr{.64} {\\scriptsize[.38, .91]} & ...

Nothing is recomputed: the numbers are the pooled rows of rank_consistency_tables.py's
"internal" table, read back from <results_dir>/tables_<null>/rank_consistency.csv, so
this runs after that script. It writes cross_method_consistency.tex next to the CSV it
read, for every null in --null whose CSV exists.

    python analysis/tables/cross_method_consistency_table.py --dataset euandi_2024
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from analysis.null_baseline import SUBTRACTED_NULLS, null_description, output_dir_name
from analysis.tables.labels import count_word
from analysis.vaa_null_model import DEFAULT_SIGMA
from utils import configure_stdout

configure_stdout()

POOLED = "All"
FRAMINGS = ["Base", "Negated"]   # the order rank_consistency_tables writes the rows in
COLUMNS = [("Kendall's $W$", "w"), ("ICC(3,1)", "icc"), ("Argmax consistency", "top1")]


def number(value, in_brackets=False):
    """.64 / 1.0 / -.36: two decimals, no leading zero, as the paper's tables print
    correlations. Inside the brackets a negative gets a real minus sign; inside \\Corr the
    plain value is kept, since the macro reads it as a number."""
    text = f"{value:.2f}"
    if text in ("1.00", "-1.00"):
        text = text[:-1]
    elif text == "-0.00":
        text = ".00"
    else:
        text = text.replace("0.", ".", 1)
    return text.replace("-", "$-$") if in_brackets else text


def cell(row, key):
    return (rf"\Corr{{{number(row[key])}}} "
            rf"{{\scriptsize[{number(row[key + '_low'], True)}, {number(row[key + '_high'], True)}]}}")


def bootstrap_draws(tex_path):
    """The bootstrap size rank_consistency_tables recorded in its provenance comment."""
    if tex_path.exists():
        match = re.search(r"bootstrap=(\d+)", tex_path.read_text(encoding="utf-8"))
        if match:
            return int(match.group(1))
    return None


def render(rows, kind, sigma, draws):
    methods = int(rows.iloc[0]["raters"])
    samples = f", {draws} samples" if draws else ""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        "    Framing & " + " & ".join(title for title, _ in COLUMNS) + r" \\",
        r"    \midrule",
    ]
    for framing, (_, row) in zip(FRAMINGS, rows.iterrows()):
        lines.append(f"    {framing:7s} & " + " & ".join(cell(row, key) for _, key in COLUMNS)
                     + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{",
        r"  \textbf{Cross-method ``annotator consistency'' across the "
        f"{count_word(methods)} methods' cluster rankings, pooled over models}} "
        rf"(95\% CI, cluster-bootstrapped over statements{samples}). Every score is taken "
        f"relative to its null, with stances drawn from {null_description(kind, sigma)}.",
        r"  }",
        r"  \label{tab:cross-method-consistency" + ("" if kind == "gaussian" else f"-{kind}") + "}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Read this results directory instead of the --dataset one.")
    parser.add_argument("--null", nargs="+", default=list(SUBTRACTED_NULLS),
                        choices=SUBTRACTED_NULLS)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Sigma of the gaussian null the tables were built with.")
    parser.add_argument("--positions", default=None, help="Accepted for analyze_all; unused.")
    args = parser.parse_args()
    results_dir = args.results_dir or Path("data") / f"{args.dataset}_results"

    for kind in dict.fromkeys(args.null):
        tables_dir = results_dir / output_dir_name("tables", kind, args.sigma)
        csv_path = tables_dir / "rank_consistency.csv"
        if not csv_path.exists():
            print(f"{csv_path} not found -- run rank_consistency_tables.py --null {kind} first; "
                  "skipping.")
            continue
        table = pd.read_csv(csv_path)
        rows = table[(table["table"] == "internal") & (table["model"] == POOLED)]
        if len(rows) != len(FRAMINGS):
            print(f"{csv_path}: expected {len(FRAMINGS)} pooled 'internal' rows (base, "
                  f"negated), found {len(rows)}; skipping.")
            continue
        tex = render(rows, kind, args.sigma, bootstrap_draws(tables_dir / "rank_consistency.tex"))
        output = tables_dir / "cross_method_consistency.tex"
        output.write_text(tex, encoding="utf-8")
        print(tex)
        print(f"Wrote {output}\n")


if __name__ == "__main__":
    main()
