#!/usr/bin/env python3
"""Per-model argmax shares as a LaTeX table: the print-size replacement for
argmax_all_models_party_distribution.png.

One block per model, one row per evaluation method plus the mean over the methods,
one column per party (cluster or EP group). The winning party of every row is set in
bold, so the argmax reads off the table directly instead of off bar heights.

Reads <results_dir>/plots/argmax_share_methods.csv, which plot_argmax_shares.py writes,
and the nulls of every method through analysis/null_baseline.py -- the same numbers the
figures subtract. The blocks are laid out side by side in `--columns` panels
so the table fills the page width rather than running down a whole column.

The main table is every share minus its null, in percentage points, once per null in
--null (gaussian = 0-centered normal, uniform): vaa_null_model_argmax.py's for the Direct
and Indirect rows, classifier_null_model.py's (the same stance distribution, carried over
to the classifier) for the classifier rows (Prose, and Reasons when
analysis.core.CLASSIFIED_TRACKS includes them). The Mean row is the mean of the methods'
differences. A positive cell is a share a group wins beyond what random stances would
already give it. Each goes to the sibling folder of --output's named after its null:
--output tables/argmax_share_table.tex writes tables_gaussian/argmax_share_table.tex and
tables_uniform/argmax_share_table.tex (tables_gaussian_sigma0.25/ for a non-default
--sigma). The raw shares go to --output's own folder, as <stem>_absolute.tex.

Next to each minus-null table goes argmax_tighter.tex, the same numbers in a single text
column (render_tight): the method labels once per row, two-letter column heads, and every
cell above the null shaded by its value in the group's colour, the ones below left white.
The colours are xcolor names (RadLeft, ...) the paper's preamble has to define; the file
opens with those lines, commented out.

--sigma-range 0.25 0.5 0.75 1 adds a sensitivity table over the gaussian null's sigma:
every cell is the mean over those sigmas of share minus null, with the smallest and largest
in small brackets after it; the Mean row is built per sigma first and then summarised the
same way. The ranges make the cells too wide for side-by-side panels, so this table is one
panel, split into floats of RANGE_MODELS_PER_TABLE models (the later ones captioned as a
continuation). It goes to tables_gaussian/<stem>_sigma_range.tex and needs each sigma's
null files (vaa_null_model_argmax.py and classifier_null_model.py --sigma <value>).

--measure probability tabulates the companion: every method's mean probability per group
instead of its argmax share, from plots/meanprob_share_methods.csv (plot_argmax_shares.py
--measure probability), minus each method's mean-probability null. Every output is named
meanprob_share_table* in place of argmax_share_table*, and meanprob_tighter.tex.

    python -m analysis.tables.argmax_share_table --dataset euandi_2024
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core import model_display_name, party_sort_key, short_party
from analysis.plotting.plot_argmax_shares import AVERAGE_LABEL, METHODS
from analysis.tables.render import latex_escape
from analysis.null_baseline import (
    CLASSIFIER_METHOD, ENVELOPE, SUBTRACTED_NULLS, VAA_METHOD, NullBaseline,
    null_description, output_dir_name,
)
from analysis.vaa_null_model import DEFAULT_SIGMA
from analysis.vaa_null_model_argmax import null_suffix
from analysis.vaa_scoring import POSTERIOR_FLOOR, vaa_scoring
from utils import FALLBACK_PARTY_COLOR, PARTY_COLORS, PARTY_SHORT

# plot_argmax_shares' method labels -> the shorter row labels used in the tables.
ROW_LABEL = {
    "direct Likert": "Direct",
    "indirect Likert": "Indirect",
    "classified reasons": "Reasons",
    "classified open-ended": "Prose",
    AVERAGE_LABEL: "Mean",
}
METHOD_ROWS = [label for _, label, _ in METHODS]
# The classifier rows in play and the text each reads, for the captions.
CLASSIFIER_ROWS = [ROW_LABEL[label] for _, label, kind in METHODS if kind == "classifier"]
CLASSIFIER_TEXT = {"Reasons": "Likert reasons", "Prose": "open-ended answers"}
# Models per float of the --sigma-range table: twelve one-panel blocks are taller than a page.
RANGE_MODELS_PER_TABLE = 6
# What the table holds; main() sets it from --measure: "argmax" shares or "probability"
# means, with the CSV column, nulls, file names and captions that go with each.
MEASURE = "argmax"
VALUE_COLUMN = {"argmax": "argmax_share", "probability": "mean_probability"}
# The tight table (render_tight): xcolor tint, in percent, of the cell furthest above the
# null. Black digits stay readable on the darkest group colour at this strength.
TIGHT_TINT = 60
METHOD_KEY = {label: key for key, label, _ in METHODS}


def load_shares(csv_path, variant):
    df = pd.read_csv(csv_path)
    df = df[df["variant"] == variant]
    if df.empty:
        raise SystemExit(f"No rows for variant '{variant}' in {csv_path}")
    # only the methods in play (a CSV written with the classified reasons may still hold them)
    df = df[df["method"].isin(METHOD_ROWS)]
    return df.pivot_table(index=["model", "method"], columns="ep_group",
                          values=VALUE_COLUMN[MEASURE], aggfunc="first")


def load_nulls(null_dir, positions, variant, sigma=DEFAULT_SIGMA, kind="gaussian"):
    """Null share per method label (rows) and group (columns), over all statements, as
    analysis/null_baseline.py serves it to the figures. Raises when a null is missing."""
    baseline = NullBaseline.load(null_dir, positions, sigma, kind)
    keys = [k for k in [*VAA_METHOD.values(), *CLASSIFIER_METHOD.values()] if k in METHOD_KEY.values()]
    label = {key: label for label, key in METHOD_KEY.items()}
    null = baseline.mean_probability if MEASURE == "probability" else baseline.share
    return pd.DataFrame({label[key]: null(key, variant) for key in keys}).T


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


def format_row(values, signed=False, lows=None, highs=None):
    """Percentages to one decimal, every maximum of the row in bold. Ties are judged on the
    unrounded shares, so two printed 31.0s are only both bold if they really tie. `signed`
    prints differences from the null with their sign: +4.2, -3.1. With `lows` and `highs`
    every cell also gets its range, [low, high], in small type after the value."""
    finite = values[np.isfinite(values)]
    top = finite.max() if finite.size else np.nan
    cells = []
    for i, value in enumerate(values):
        if not np.isfinite(value):
            cells.append("--")
            continue
        text = signed_text(value) if signed else f"{100 * value:.1f}"
        text = rf"\textbf{{{text}}}" if np.isclose(value, top) else text
        if lows is not None:
            text += (rf"\,{{\scriptsize[{signed_text(lows[i])}, "
                     rf"{signed_text(highs[i])}]}}")
        cells.append(text)
    return cells


RULE = None   # marks the dashed rule between the methods and their mean


def model_block(model, shares, parties, signed=False, bounds=None):
    """The rows of one model: a name line, the methods, a dashed rule, the mean.
    A missing method still gets its (empty-valued) row, so every block has the same
    height and the dashed rules of side-by-side panels fall on the same line. `bounds`
    is an optional (low, high) pair of frames shaped like `shares`, printed as ranges."""
    ncols = len(parties) + 1
    name = latex_escape(model_display_name(model))
    rows = [rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{name}}}}}"]
    for method in METHOD_ROWS + [AVERAGE_LABEL]:
        if method == AVERAGE_LABEL:
            rows.append(RULE)
        lows = highs = None
        if (model, method) in shares.index:
            values = shares.loc[(model, method), parties].to_numpy(dtype=float)
            if bounds is not None:
                lows, highs = (frame.loc[(model, method), parties].to_numpy(dtype=float)
                               for frame in bounds)
        else:
            values = np.full(len(parties), np.nan)
        rows.append(" & ".join([ROW_LABEL[method]] + format_row(values, signed, lows, highs)))
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


def sorted_models(shares):
    """The models of a shares frame in display-name order, as every table lists them."""
    return sorted(shares.index.get_level_values("model").unique(),
                  key=lambda m: model_display_name(m).casefold())


def render(shares, parties, panels, caption, label, signed=False, bounds=None, models=None,
           size=r"\small", tabcolsep="4pt"):
    """One table* float. `models` restricts it to those models (default: all of them, in
    display-name order); a `label` of None leaves the float unlabelled."""
    models = models if models is not None else sorted_models(shares)
    blocks = [model_block(model, shares, parties, signed, bounds) for model in models]
    ncols = len(parties) + 1
    panels = min(panels, len(blocks))
    header = " & ".join(["Method"] + [latex_escape(short_party(p)) for p in parties])
    # Panels are joined by an empty spacer column, which the rules below skip.
    spec = "@{}" + "@{\\hspace{1.5em}}c@{}".join(["l" + "r" * len(parties)] * panels) + "@{}"
    midrules = "".join(r"\cmidrule(lr){%d-%d}" % panel_columns(p, ncols) for p in range(panels))
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        f"  {size}",
        rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}",
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
        *([rf"  \label{{{label}}}"] if label else []),
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def group_initials(party):
    """RL for Rad-left: the two-letter column heads of the tight table. A group without a
    nickname (the EP groups) keeps its name."""
    if party not in PARTY_SHORT:
        return latex_escape(party)
    return "".join(word[0] for word in PARTY_SHORT[party].split("-")).upper()


def color_name(party):
    """The xcolor name the tight table shades a group's cells with: RadLeft for Rad-left,
    GreensEFA for Greens/EFA. The paper's preamble defines it (color_definition)."""
    words = re.split(r"[^A-Za-z]+", short_party(party))
    return "".join(word[0].upper() + word[1:] for word in words if word)


def color_definition(party):
    """The preamble line that gives color_name its PARTY_COLORS colour."""
    red, green, blue = bytes.fromhex(PARTY_COLORS.get(party, FALLBACK_PARTY_COLOR).lstrip("#"))
    return rf"\definecolor{{{color_name(party)}}}{{RGB}}{{{red},{green},{blue}}}"


def tight_cells(values, parties, scale):
    """format_row's signed cells, those above the null shaded in the group's colour: at
    TIGHT_TINT when the value is `scale`. A value below the null stays white."""
    cells = []
    for value, party, text in zip(values, parties, format_row(values, signed=True)):
        if np.isfinite(value) and value > 0 and scale > 0:
            tint = round(TIGHT_TINT * min(1.0, value / scale))
            if tint:
                text = rf"\cellcolor{{{color_name(party)}!{tint}}}{text}"
        cells.append(text)
    return cells


def tight_block(model, shares, parties, scale, labelled):
    """model_block for the tight table: shaded cells, and the method labels only when
    `labelled` -- the first panel carries them for the whole row."""
    name = latex_escape(model_display_name(model))
    span = len(parties) + labelled
    rows = [rf"\multicolumn{{{span}}}{{@{{}}l}}{{\textit{{{name}}}}}"]
    for method in METHOD_ROWS + [AVERAGE_LABEL]:
        if method == AVERAGE_LABEL:
            rows.append(RULE)
        if (model, method) in shares.index:
            values = shares.loc[(model, method), parties].to_numpy(dtype=float)
        else:
            values = np.full(len(parties), np.nan)
        rows.append(" & ".join([ROW_LABEL[method]] * labelled
                               + tight_cells(values, parties, scale)))
    return rows


def tight_columns(panel, nparties):
    """1-based tabular columns of a tight-table panel; only the first has a label column."""
    return (1 if panel == 0 else panel * nparties + 2), (panel + 1) * nparties + 1


def render_tight(shares, parties, panels, caption, label):
    """The minus-null table squeezed into one text column: a `table` float rather than a
    `table*`, the method labels in the first panel only, two-letter column heads, a narrow
    column gap, and every cell above the null shaded by its value (tight_cells) on one
    scale for the whole table, its largest value. The group colours come from the paper's
    preamble; the file opens with the lines that define them, commented out."""
    models = sorted_models(shares)
    panels = min(panels, len(models))
    per_panel = int(np.ceil(len(models) / panels))
    scale = float(np.nanmax(shares[parties].to_numpy(dtype=float)))
    columns = [sum((tight_block(model, shares, parties, scale, labelled=panel == 0)
                    for model in models[panel * per_panel:(panel + 1) * per_panel]), [])
               for panel in range(panels)]
    empty = " & ".join([""] * len(parties))
    body = []
    for index in range(len(columns[0])):
        cells = [column[index] if index < len(column) else empty for column in columns]
        if RULE in cells:
            body.append("".join(
                r"\cdashline{%d-%d}" % tight_columns(panel, len(parties))
                for panel, cell in enumerate(cells) if cell is RULE))
        else:
            body.append(" & ".join(cells) + r" \\")
    heads = " & ".join(rf"\cellcolor{{{color_name(p)}!{TIGHT_TINT}}}{group_initials(p)}"
                       for p in parties)
    # The gap between panels is an @-expression, which drops the column padding on both
    # sides of it; the shading still overhangs by \tabcolsep, so the gap is twice that
    # wider than the white it leaves. No closing @{}: the shading would run past the rules.
    spec = "@{}l" + "@{\\hspace{7pt}}".join(["r" * len(parties)] * panels)
    lines = [
        "% Needs these group colours in the preamble:",
        *(f"% {color_definition(p)}" for p in parties),
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{1.5pt}",
        rf"  \begin{{tabular}}{{{spec}}}",
        r"    \toprule",
        "    & " + " & ".join([heads] * panels) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + line for line in body]
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def sigma_words(sigma):
    """The null's sigma for the caption, in Likert steps (one step = 0.5 stance units)."""
    steps = sigma / 0.5
    if np.isclose(steps, 1):
        return "one Likert step"
    return f"{steps:g} Likert steps" if steps > 1 else f"{steps:g} of a Likert step"


def null_words(kind, sigma):
    """The null's stance distribution for the caption."""
    if kind in ("uniform", ENVELOPE):
        return null_description(kind)
    return f"a normal distribution around the neutral answer ($\\sigma$ = {sigma_words(sigma)})"


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
                        help="Default: <results_dir>/tables/argmax_share_table.tex. The "
                             "minus-null tables go to the sibling folder named after each "
                             "null (tables_gaussian/argmax_share_table.tex, ...); the raw "
                             "shares to this folder, as <stem>_absolute.tex.")
    parser.add_argument("--positions", default="ep-group",
                        help="Basis the VAA null was run on; picks its file name.")
    parser.add_argument("--null_dir", type=Path, default=None,
                        help="Default: <results_dir>/tables/null_model")
    parser.add_argument("--null", nargs="+", default=list(SUBTRACTED_NULLS),
                        choices=SUBTRACTED_NULLS,
                        help="The nulls to subtract, one table each (default: %(default)s). "
                             "max = per group the largest of the normal nulls at sigma "
                             "0.25/0.5/0.75/1 and the uniform null.")
    parser.add_argument("--measure", default="argmax", choices=list(VALUE_COLUMN),
                        help="argmax: argmax shares (default); probability: mean "
                             "probabilities, the companion table (meanprob_share_table*).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Gaussian-null sigma to subtract (stance units, %(default)s = "
                             "one Likert step). A non-default sigma reads null files tagged "
                             "_sigma<value> and writes into tables_gaussian_sigma<value>/.")
    parser.add_argument("--sigma-range", nargs="+", type=float, default=None,
                        metavar="SIGMA",
                        help="Also write tables_gaussian/<stem>_sigma_range.tex: share minus "
                             "the gaussian null, mean over these sigmas with [min, max], "
                             "e.g. --sigma-range 0.25 0.5 0.75 1.")
    return parser.parse_args()


def sigma_range(shares, parties, null_dir, positions, variant, sigmas):
    """(mean, low, high) over `sigmas` of shares minus the gaussian null, each a frame
    shaped like minus_null's. The Mean row is the per-sigma Mean row, summarised."""
    stacked = pd.concat(
        {sigma: minus_null(shares, load_nulls(null_dir, positions, variant, sigma, "gaussian"))
         [parties] for sigma in sigmas}, names=["sigma"])
    grouped = stacked.groupby(level=["model", "method"])
    return grouped.mean(), grouped.min(), grouped.max()


def main():
    global MEASURE
    args = parse_args()
    MEASURE = args.measure
    probability = MEASURE == "probability"
    # The names and words that change with the measure.
    stem = "meanprob_share" if probability else "argmax_share"
    noun = "Mean probability" if probability else "Argmax share"
    label_stem = "tab:meanprob" if probability else "tab:argmax_shares"
    unit_noun = "probabilities" if probability else "shares"
    results_dir = args.results_dir or Path("data") / f"{args.dataset}_results"
    plots_dir = results_dir / "plots"
    csv_path = plots_dir / f"{stem}_methods.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run plot_argmax_shares.py"
                         + (" --measure probability" if probability else "") + " first.")
    shares = load_shares(csv_path, args.variant)
    parties = sorted(shares.columns, key=party_sort_key)   # left to right, as the figures
    legend = "; ".join(f"{short}: {full}" for full, short in PARTY_SHORT.items()
                       if full in parties)
    scoring = vaa_scoring(args.positions)
    if probability and scoring != "posterior":
        print("--measure probability: skipped -- the VAA tracks are on the distance score "
              f"(--positions {args.positions}), so they have no probability to average.")
        return
    direct = ("Likert answers, each statement's stance read as cluster membership "
              "probabilities under one Gaussian per cluster and statement (member parties' "
              f"MEP-weighted mean and SD, SD floored at {POSTERIOR_FLOOR:g}), "
              + ("averaged over the cells" if probability
                 else "the most probable cluster winning the cell")
              if scoring == "posterior"
              else "Likert answers scored against party positions")
    classifier = ("mean softmax probability of the classifier" if probability
                  else "classifier")
    methods_note = (
        f"Direct: {direct}; Indirect: stance of the "
        f"open-ended answers scored the same way; {' / '.join(CLASSIFIER_ROWS)}: {classifier} "
        f"on the {' / '.join(CLASSIFIER_TEXT[row] for row in CLASSIFIER_ROWS)}."
        + (f" {latex_escape(legend)}." if legend else ""))
    output = args.output or results_dir / "tables" / "argmax_share_table.tex"
    output = output.with_name(output.name.replace("argmax_share", stem, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Loaded first, so a missing null stops the run before anything is written.
    null_dir = args.null_dir or results_dir / "tables" / "null_model"
    nulls = {kind: load_nulls(null_dir, args.positions, args.variant, args.sigma, kind)
             for kind in dict.fromkeys(args.null)}

    caption = (f"{noun} (\\%) of each group per evaluation method, and their mean, for "
               f"every model; the largest value in each row is in bold. " + methods_note)
    tex = render(shares, parties, args.columns, caption, f"{label_stem}_absolute")
    absolute_output = output.with_name(f"{output.stem}_absolute{output.suffix}")
    absolute_output.write_text(tex, encoding="utf-8")
    print(f"Wrote {absolute_output}")

    null_how = ("for Direct and Indirect as random Likert answers scored the same way, for "
                f"{' and '.join(CLASSIFIER_ROWS)} as the classifier's group {unit_noun} per "
                "statement and "
                "stance level, reweighted to the same stance distribution")
    for kind, null in nulls.items():
        caption = (
            f"{noun} of each group minus its "
            f"{dict(gaussian='0-centered', max='largest').get(kind, kind)} "
            "null (percentage points), per evaluation method and their mean; the largest "
            "value in each row is in bold. The null draws every stance from "
            f"{null_words(kind, args.sigma)}: {null_how}. Positive values are {unit_noun} "
            "a group gets beyond its position relative to the center. "
            + ("The null is a maximum per group, so the four values of a row sum to less "
               "than zero and a positive value is a lower bound. " if kind == ENVELOPE else "")
            + methods_note)
        difference = minus_null(shares, null)[parties]
        tex = render(difference, parties, args.columns, caption,
                     label_stem + ("-max" if kind == ENVELOPE
                                   else null_suffix(kind, args.sigma).replace("_", "-")),
                     signed=True)
        null_output = (output.parent.parent / output_dir_name(output.parent.name, kind,
                                                              args.sigma) / output.name)
        null_output.parent.mkdir(parents=True, exist_ok=True)
        null_output.write_text(tex, encoding="utf-8")
        print(tex)
        print(f"Wrote {null_output}")

        # The same numbers in one text column; the caption is cut down with the table and
        # leaves the methods to the text.
        furthest = signed_text(np.nanmax(difference.to_numpy(dtype=float)))
        initials = "; ".join(f"{group_initials(p)}: {latex_escape(p)}" for p in parties
                             if p in PARTY_SHORT)
        tight_caption = (
            f"{noun} of each group minus its "
            f"{dict(gaussian='0-centered', max='largest').get(kind, kind)} "
            "null (percentage points), per evaluation method and their mean; the largest "
            "value in each row is in bold. Values above the null are shaded in the "
            f"group's color, darker the larger they are (up to {furthest}). The null draws "
            f"every stance from {null_words(kind, args.sigma)}. "
            + ("It is a maximum per group, so the four values of a row sum to less than "
               "zero and a positive value is a lower bound. " if kind == ENVELOPE else "")
            + (f"{initials}." if initials else "")).rstrip()
        tight_output = null_output.with_name(f"{stem.removesuffix('_share')}_tighter.tex")
        tight_output.write_text(
            render_tight(difference, parties, args.columns if len(parties) <= 4 else 1,
                         tight_caption, f"{label_stem}-{kind}-tighter"),
            encoding="utf-8")
        print(f"Wrote {tight_output}")

    if args.sigma_range:
        sigmas = sorted(set(args.sigma_range))
        mean, low, high = sigma_range(shares, parties, null_dir, args.positions,
                                      args.variant, sigmas)
        listed = ", ".join(f"{sigma:g}" for sigma in sigmas)
        caption = (
            f"{noun} of each group minus its 0-centered null (percentage points), per "
            "evaluation method and their mean, averaged over the null's spread: the null draws "
            "every stance from a normal distribution around the neutral answer with "
            f"$\\sigma \\in \\{{{listed}\\}}$ (stance units; 0.5 is one Likert step), each "
            "value is the mean over these, and the brackets give the smallest and largest. "
            f"The largest mean in each row is in bold. The null: {null_how}. " + methods_note)
        label = f"{label_stem}-sigma-range"
        models = sorted_models(mean)
        chunks = [models[i:i + RANGE_MODELS_PER_TABLE]
                  for i in range(0, len(models), RANGE_MODELS_PER_TABLE)]
        tex = "\n".join(
            render(mean, parties, 1,
                   caption if index == 0 else
                   f"Continued from Table~\\ref{{{label}}}: {noun.lower()} minus the "
                   f"0-centered null, mean over $\\sigma \\in \\{{{listed}\\}}$ [min, max].",
                   label if index == 0 else None, signed=True, bounds=(low, high),
                   models=chunk, size=r"\footnotesize", tabcolsep="3pt")
            for index, chunk in enumerate(chunks))
        range_output = (output.parent.parent / output_dir_name(output.parent.name, "gaussian")
                        / f"{output.stem}_sigma_range{output.suffix}")
        range_output.parent.mkdir(parents=True, exist_ok=True)
        range_output.write_text(tex, encoding="utf-8")
        print(tex)
        print(f"Wrote {range_output}")


if __name__ == "__main__":
    main()
