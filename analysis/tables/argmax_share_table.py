#!/usr/bin/env python3
"""Per-model argmax shares as a LaTeX table: the print-size replacement for
argmax_all_models_party_distribution.png.

One block per model, one row per evaluation method plus the mean over the four methods,
one column per party (cluster or EP group). The winning party of every row is set in
bold, so the argmax reads off the table directly instead of off bar heights.

Reads <results_dir>/plots/argmax_share_methods.csv, which plot_argmax_shares.py writes;
nothing is recomputed here. The blocks are laid out side by side in `--columns` panels
so the table fills the page width rather than running down a whole column.

    python -m analysis.tables.argmax_share_table --dataset euandi_2024
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core import model_display_name, short_party
from analysis.plotting.plot_argmax_shares import AVERAGE_LABEL, METHODS
from analysis.tables.render import latex_escape
from utils import PARTY_SHORT

# plot_argmax_shares' method labels -> the shorter row labels used in the tables.
ROW_LABEL = {
    "direct Likert": "Direct",
    "indirect Likert": "Indirect",
    "classified reasons": "Reasons",
    "classified open-ended": "Prose",
    AVERAGE_LABEL: "Mean",
}
METHOD_ROWS = [label for _, label, _ in METHODS]


def load_shares(csv_path, variant):
    df = pd.read_csv(csv_path)
    df = df[df["variant"] == variant]
    if df.empty:
        raise SystemExit(f"No rows for variant '{variant}' in {csv_path}")
    return df.pivot_table(index=["model", "method"], columns="ep_group",
                          values="argmax_share", aggfunc="first")


def format_row(values):
    """Integer percentages, every maximum of the row in bold. Ties are judged on the
    unrounded shares, so two printed 31s are only both bold if they really tie."""
    finite = values[np.isfinite(values)]
    top = finite.max() if finite.size else np.nan
    cells = []
    for value in values:
        if not np.isfinite(value):
            cells.append("--")
            continue
        text = f"{100 * value:.0f}"
        cells.append(rf"\textbf{{{text}}}" if np.isclose(value, top) else text)
    return cells


RULE = None   # marks the dashed rule between the methods and their mean


def model_block(model, shares, parties):
    """The rows of one model: a name line, the four methods, a dashed rule, the mean.
    A missing method still gets its (empty-valued) row, so every block has the same
    height and the dashed rules of side-by-side panels fall on the same line."""
    ncols = len(parties) + 1
    name = latex_escape(model_display_name(model))
    rows = [rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{name}}}}}"]
    for method in METHOD_ROWS + [AVERAGE_LABEL]:
        if method == AVERAGE_LABEL:
            rows.append(RULE)
        if (model, method) in shares.index:
            values = shares.loc[(model, method), parties].to_numpy(dtype=float)
        else:
            values = np.full(len(parties), np.nan)
        rows.append(" & ".join([ROW_LABEL[method]] + format_row(values)))
    return rows


def panel_columns(panel, ncols):
    """1-based tabular columns of a panel; panels are separated by one spacer column."""
    first = panel * (ncols + 1) + 1
    return first, first + ncols - 1


def side_by_side(blocks, panels, ncols):
    """Deal whole model blocks into `panels` side-by-side panels, top to bottom then
    left to right, and join them line by line; a short last panel is padded blank."""
    per_panel = int(np.ceil(len(blocks) / panels))
    columns = [sum(blocks[i:i + per_panel], []) for i in range(0, len(blocks), per_panel)]
    empty = " & ".join([""] * ncols)
    lines = []
    for index in range(max(len(column) for column in columns)):
        cells = [column[index] if index < len(column) else empty for column in columns]
        if RULE in cells:
            lines.append("".join(
                r"\cdashline{%d-%d}" % panel_columns(panel, ncols)
                for panel, cell in enumerate(cells) if cell is RULE))
        else:
            lines.append(" & & ".join(cells) + r" \\")
    return lines


def render(shares, parties, panels, caption, label):
    models = sorted(shares.index.get_level_values("model").unique(),
                    key=lambda m: model_display_name(m).casefold())
    blocks = [model_block(model, shares, parties) for model in models]
    ncols = len(parties) + 1
    panels = min(panels, len(blocks))
    header = " & ".join(["Method"] + [latex_escape(short_party(p)) for p in parties])
    # Panels are joined by an empty spacer column, which the rules below skip.
    spec = "@{}" + "@{\\hspace{1.5em}}c@{}".join(["l" + "r" * len(parties)] * panels) + "@{}"
    midrules = "".join(r"\cmidrule(lr){%d-%d}" % panel_columns(p, ncols) for p in range(panels))
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        rf"  \begin{{tabular}}{{{spec}}}",
        r"    \toprule",
        "    " + " & & ".join([header] * panels) + r" \\",
        "    " + midrules,
    ]
    lines += ["    " + line for line in side_by_side(blocks, panels, ncols)]
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="euandi_2024",
                        help="Reads data/<dataset>_results/plots/argmax_share_methods.csv, "
                             "as plot_argmax_shares.py --dataset names it.")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Read this results directory instead of the --dataset one.")
    parser.add_argument("--variant", default="pooled",
                        help="Prompt variant to tabulate (base, negated or pooled).")
    parser.add_argument("--columns", type=int, default=2,
                        help="Model blocks are laid out in this many side-by-side panels.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <results_dir>/plots/argmax_share_table.tex")
    return parser.parse_args()


def main():
    args = parse_args()
    plots_dir = (args.results_dir or Path("data") / f"{args.dataset}_results") / "plots"
    csv_path = plots_dir / "argmax_share_methods.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run plot_argmax_shares.py first.")
    shares = load_shares(csv_path, args.variant)
    parties = list(shares.columns)
    legend = "; ".join(f"{short}: {full}" for full, short in PARTY_SHORT.items()
                       if full in parties)
    caption = (
        "Argmax share (\\%) of each group per evaluation method, and their mean, for every "
        "model; the largest share in each row is in bold. Direct: Likert answers scored "
        "against party positions; Indirect: stance of the open-ended answers scored the "
        "same way; Reasons / Prose: classifier on the Likert reasons / open-ended answers."
        + (f" {latex_escape(legend)}." if legend else ""))
    tex = render(shares, parties, args.columns, caption, "tab:argmax_shares")
    output = args.output or plots_dir / "argmax_share_table.tex"
    output.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
