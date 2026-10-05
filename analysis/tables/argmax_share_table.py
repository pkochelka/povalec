#!/usr/bin/env python3
"""Per-model argmax shares as a LaTeX table: the print-size replacement for
argmax_all_models_party_distribution.png.

One block per model, one row per evaluation method plus the mean over the four methods,
one column per party (cluster or EP group). The winning party of every row is set in
bold, so the argmax reads off the table directly instead of off bar heights.

Reads <results_dir>/plots/argmax_share_methods.csv, which plot_argmax_shares.py writes;
nothing is recomputed here. The blocks are laid out side by side in `--columns` panels
so the table fills the page width rather than running down a whole column.

When the null-model outputs are present it also writes a second table, every share minus
its 0-centered normal null: vaa_null_model_argmax.py's for the Direct and Indirect rows,
classifier_null_model.py's (the same stance distribution, carried over to the
classifier) for Reasons and Prose. The Mean row is then the mean of the four
differences. A positive cell is a share a group wins beyond what random stances around
the center would already give it.

    python -m analysis.tables.argmax_share_table --dataset euandi_2024
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core import model_display_name, party_sort_key, short_party
from analysis.plotting.plot_argmax_shares import AVERAGE_LABEL, METHODS
from analysis.tables.render import latex_escape
from analysis.classifier_null_model import output_stem
from analysis.vaa_null_model_argmax import WHOLE, output_suffix
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
# vaa_null_model_argmax names its two tracks by source; the table by method label.
VAA_NULL_METHOD = {"likert": "direct Likert", "speeches": "indirect Likert"}
NULL_SIDE = "normal"


def load_shares(csv_path, variant):
    df = pd.read_csv(csv_path)
    df = df[df["variant"] == variant]
    if df.empty:
        raise SystemExit(f"No rows for variant '{variant}' in {csv_path}")
    return df.pivot_table(index=["model", "method"], columns="ep_group",
                          values="argmax_share", aggfunc="first")


def load_nulls(null_dir, positions, variant):
    """Null share per method label (rows) and group (columns), or None when either null
    is missing. The VAA null is framing-free (a 0-centered null is symmetric under
    negation), so it serves every variant; the classifier null is read per variant."""
    vaa_path = null_dir / f"null_model_argmax{output_suffix(positions)}_shares.csv"
    classifier_path = null_dir / f"{output_stem(positions)}_shares.csv"
    missing = [path.name for path in (vaa_path, classifier_path) if not path.exists()]
    if missing:
        print(f"No null-model table: {', '.join(missing)} not in {null_dir} -- run "
              "vaa_null_model_argmax.py and classifier_null_model.py first.")
        return None
    vaa = pd.read_csv(vaa_path)
    vaa = vaa[(vaa["side"] == NULL_SIDE) & (vaa["topic"] == WHOLE)
              & vaa["method"].isin(VAA_NULL_METHOD)]
    vaa = vaa.assign(method=vaa["method"].map(VAA_NULL_METHOD))
    classifier = pd.read_csv(classifier_path)
    classifier = classifier[classifier["variant"] == variant].rename(
        columns={"null_share": "argmax_share"})
    nulls = pd.concat([vaa[["method", "ep_group", "argmax_share"]],
                       classifier[["method", "ep_group", "argmax_share"]]])
    return nulls.pivot_table(index="method", columns="ep_group", values="argmax_share")


def minus_null(shares, nulls):
    """Each method's share minus its null; the Mean row becomes the mean of the method
    differences a model has, so it averages the same rows its share did."""
    rows = {}
    for (model, method), values in shares.iterrows():
        if method in nulls.index:
            rows[(model, method)] = values - nulls.loc[method].reindex(values.index)
    difference = pd.DataFrame.from_dict(rows, orient="index")
    difference.index = pd.MultiIndex.from_tuples(difference.index, names=["model", "method"])
    means = difference.groupby(level="model").mean()
    means.index = pd.MultiIndex.from_tuples([(m, AVERAGE_LABEL) for m in means.index],
                                            names=["model", "method"])
    return pd.concat([difference, means]).sort_index()


def signed_text(value):
    """+4.2 / $-$3.1 in percentage points: a real minus sign, not a hyphen, and no
    "-0.0" for a difference that rounds to zero."""
    rounded = round(100 * value, 1) + 0.0
    return f"+{rounded:.1f}" if rounded >= 0 else f"$-${-rounded:.1f}"


def format_row(values, signed=False):
    """Percentages to one decimal, every maximum of the row in bold. Ties are judged on the
    unrounded shares, so two printed 31.0s are only both bold if they really tie. `signed`
    prints differences from the null with their sign: +4.2, -3.1."""
    finite = values[np.isfinite(values)]
    top = finite.max() if finite.size else np.nan
    cells = []
    for value in values:
        if not np.isfinite(value):
            cells.append("--")
            continue
        text = signed_text(value) if signed else f"{100 * value:.1f}"
        cells.append(rf"\textbf{{{text}}}" if np.isclose(value, top) else text)
    return cells


RULE = None   # marks the dashed rule between the methods and their mean


def model_block(model, shares, parties, signed=False):
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
        rows.append(" & ".join([ROW_LABEL[method]] + format_row(values, signed)))
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


def render(shares, parties, panels, caption, label, signed=False):
    models = sorted(shares.index.get_level_values("model").unique(),
                    key=lambda m: model_display_name(m).casefold())
    blocks = [model_block(model, shares, parties, signed) for model in models]
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
                        help="Default: <results_dir>/plots/argmax_share_table.tex. The "
                             "minus-null table goes next to it, as <stem>_minus_null.tex.")
    parser.add_argument("--positions", default="ep-group",
                        help="Basis the VAA null was run on; picks its file name.")
    parser.add_argument("--null_dir", type=Path, default=None,
                        help="Default: <results_dir>/tables/null_model")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir or Path("data") / f"{args.dataset}_results"
    plots_dir = results_dir / "plots"
    csv_path = plots_dir / "argmax_share_methods.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run plot_argmax_shares.py first.")
    shares = load_shares(csv_path, args.variant)
    parties = sorted(shares.columns, key=party_sort_key)   # left to right, as the figures
    legend = "; ".join(f"{short}: {full}" for full, short in PARTY_SHORT.items()
                       if full in parties)
    methods_note = (
        "Direct: Likert answers scored against party positions; Indirect: stance of the "
        "open-ended answers scored the same way; Reasons / Prose: classifier on the Likert "
        "reasons / open-ended answers." + (f" {latex_escape(legend)}." if legend else ""))
    caption = ("Argmax share (\\%) of each group per evaluation method, and their mean, for "
               "every model; the largest share in each row is in bold. " + methods_note)
    tex = render(shares, parties, args.columns, caption, "tab:argmax_shares")
    output = args.output or plots_dir / "argmax_share_table.tex"
    output.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"Wrote {output}")

    nulls = load_nulls(args.null_dir or results_dir / "tables" / "null_model",
                       args.positions, args.variant)
    if nulls is None:
        return
    caption = (
        "Argmax share of each group minus its 0-centered null (percentage points), per "
        "evaluation method and their mean; the largest value in each row is in bold. The null "
        "draws every stance from a normal distribution around the neutral answer "
        "($\\sigma$ = one Likert step): for Direct and Indirect as random Likert answers "
        "scored against the positions, for Reasons and Prose as the classifier's group "
        "shares per statement and stance level, reweighted to the same stance distribution. "
        "Positive values are shares a group wins beyond its position relative to the "
        "center. " + methods_note)
    tex = render(minus_null(shares, nulls)[parties], parties, args.columns, caption,
                 "tab:argmax_shares_minus_null", signed=True)
    null_output = output.with_name(f"{output.stem}_minus_null{output.suffix}")
    null_output.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"Wrote {null_output}")


if __name__ == "__main__":
    main()
