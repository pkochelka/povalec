"""LaTeX and Markdown rendering for the rank-consistency tables.

Formatting only: turning computed metrics and their bootstrap replicates into table
cells, captions and complete `tabular` environments. Nothing here reads a file, computes
a statistic, or knows what the CLI was asked for.

It was ~600 lines in the middle of `rank_consistency_tables.py`, between the bootstrap
and the CLI, so changing how a number is *printed* meant editing the module that decides
what the number *is*.
"""
import itertools
import re

import numpy as np
import pandas as pd

from analysis.core import model_display_name
from analysis.stats import (
    kendall_pvalue,
    pair_indices,
    percentile_interval,
    spearman_brown,
)
from analysis.tables.labels import (
    BYMODEL_FACTORS,
    DIMENSION_HEADER,
    MATRIX_LABEL,
    METHOD_LABEL,
    POOLED,
    POOLED_ROW_LABEL,
    short_model_name,
)

# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}


def latex_escape(text):
    return "".join(LATEX_ESCAPES.get(character, character) for character in str(text))


def format_metric(observed, replicates, key, percent=False):
    value = observed[key]
    text = f"{100 * value:.0f}\\%" if percent else f"{value:.3f}"
    if not replicates:
        return text
    low, high = percentile_interval(replicates[key])
    if percent:
        return f"{text} [{100 * low:.0f}, {100 * high:.0f}]"
    return f"{text} [{low:.3f}, {high:.3f}]"


def scope_labels(spec, scope):
    """The scope tuple as prose: method codes become their display names."""
    return [value if dimension != "method" or value == POOLED else METHOD_LABEL[value]
            for (dimension, _), value in zip(spec["dimensions"], scope)]


def row_prefix(job, runs):
    raters = len(job["levels"])
    return [str(raters), str(runs[0]) if len(set(runs)) == 1 else f"{min(runs)}--{max(runs)}"]


def concordance_body(spec, jobs, observed, replicates, parties, args):
    """W / ICC(3,1) / top-1 share, for a rater set that all rank the same EP
    groups: the methods, languages or prompt-variants tables."""
    dimensions = len(spec["dimensions"])
    header = [*(DIMENSION_HEADER[dimension] for dimension, _ in spec["dimensions"]),
              "$m$", "runs/rater", "Kendall's $W$", "ICC(3,1)", "Top-1 share",
              "Modal top-1 group"]
    if args.pvalues:
        header.insert(dimensions + 3, "$p$")
    rows = []
    for job, values, sampled in zip(jobs, observed, replicates):
        raters, runs = len(job["levels"]), job["runs"]
        labels = scope_labels(spec, job["scope"])
        cells = [
            *labels, *row_prefix(job, runs),
            format_metric(values, sampled, "w"),
            format_metric(values, sampled, "icc"),
            format_metric(values, sampled, "top1", percent=True),
            parties[values["winner"]],
        ]
        if args.pvalues:
            cells.insert(dimensions + 3,
                         format_pvalue(kendall_pvalue(values["w"], raters, len(parties))))
        rows.append(cells)
        print(f"  {' / '.join(labels):52s} m={raters:3d}  W={values['w']:.3f}  "
              f"ICC={values['icc']:.3f}  top-1={values['top1']:.0%} "
              f"({parties[values['winner']]})")
    return header, rows, set(range(dimensions)) | {len(header) - 1}


def method_pair_lookup(job):
    """{frozenset of the two method codes: index into job['rho']} -- a job may not
    carry every requested method (--allow-partial-models), so pairs are matched by
    name, not by position in the job's own (possibly shorter) method list."""
    return {frozenset((job["levels"][a], job["levels"][b])): local
            for local, (a, b) in enumerate(pair_indices(len(job["levels"])))}


def rho_marker(sampled, key, local):
    """'*' when the pair's CI excludes zero -- the conventional reading. An
    earlier version inverted this to flag the rarer non-significant case, but a
    star is pattern-matched to "significant" faster than any caption is read,
    so the convention wins."""
    if not sampled:
        return ""
    low, high = percentile_interval(sampled[key][:, local])
    return "" if low <= 0 <= high else "*"


def format_correlation(value):
    """APA-style correlation formatting: no leading zero (bounded by +-1, so it
    is redundant), no + sign, and "1.0" rather than "1.00" for a perfect
    correlation (2 decimals would otherwise round 0.995+ up to a misleading
    "1.00" that reads as exact)."""
    rounded = round(value, 2)
    if rounded >= 1.0:
        return "1.0"
    if rounded <= -1.0:
        return "-1.0"
    text = f"{abs(value):.2f}".lstrip("0")
    return f"-{text}" if value < 0 else text


# Faint vertical rule between tabular columns; needs \usepackage{xcolor}.
GRAY_COLUMN_RULE = r"!{\color{gray!25}\vrule}"


def format_sd(value):
    """Between-party SD formatting: same no-leading-zero convention. Always
    non-negative, so no sign handling needed."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}".lstrip("0") or "0.00"


def with_sd_superscript(text, sd):
    """"<value>\\textsuperscript{<sd>}" -- the between-party SD rides along as a
    superscript rather than a parenthetical, so the primary coefficient stays
    the thing the eye lands on in a dense grid."""
    return text if sd is None else f"{text}\\textsuperscript{{{format_sd(sd)}}}"


SUPERSCRIPT_PATTERN = re.compile(r"\\textsuperscript\{([^}]*)\}")


def markdown_cell(text):
    """A LaTeX cell rendered readably in the markdown preview: superscripts
    become ^x, escaped percents unescape."""
    return SUPERSCRIPT_PATTERN.sub(r"^\1", str(text)).replace("\\%", "%")


MATRIX_FRAMINGS = ["base", "negated"]  # upper triangle, lower triangle


def matrix_cells(methods, by_framing):
    """(labels, labels, grid) for one model's method x method correlation matrix.

    A correlation matrix is symmetric, so the lower triangle would otherwise
    repeat the upper. It carries the second framing instead: the UPPER triangle
    is the base framing, the LOWER the negated one. Each triangle is computed
    from its own framing's runs -- pooling them first would average two
    rankings that, for Indirect, are close to opposite. "n/a" marks a pair the
    job is missing a method for (e.g. --allow-partial-models).

    A grid entry is the (correlation text, significance marker) pair rather than
    one joined string, so each renderer can mark the coefficient up on its own
    terms -- LaTeX wraps it in \\Corr and leaves the star outside."""
    labels = [MATRIX_LABEL[method] for method in methods]
    grid = []
    for row in range(len(methods)):
        cells = []
        for col in range(len(methods)):
            if row == col:
                cells.append(("1.0", ""))
                continue
            framing = MATRIX_FRAMINGS[0] if row < col else MATRIX_FRAMINGS[1]
            job, values, sampled = by_framing[framing]
            local = method_pair_lookup(job).get(frozenset((methods[row], methods[col])))
            if local is None:
                cells.append(("n/a", ""))
                continue
            key = "rho_lang" if "rho_lang" in values else "rho"
            cells.append((format_correlation(values[key][local]),
                          rho_marker(sampled, key, local)))
        grid.append(cells)
    return labels, labels, grid


NOT_AVAILABLE = "n/a"


def latex_correlation_cell(text, marker):
    """\\Corr{<rho>} so the document can style every coefficient at once (shading
    by magnitude, say) from one macro. The significance star stays outside the
    braces: it qualifies the coefficient, it is not part of the number. "n/a" is
    not a coefficient, so it is left bare."""
    return text + marker if text == NOT_AVAILABLE else f"\\Corr{{{text}}}{marker}"


def markdown_correlation_cell(text, marker):
    return text + marker


def matrix_title(model_scope):
    """Table heading: the model's official name, as the figures write it."""
    return "All models" if model_scope == POOLED else model_display_name(model_scope)


def slug(text):
    return re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-").lower()


def matrix_summary_sentence(methods, by_framing, draws, explain=True):
    """The closing lines: every whole-set figure the pairwise matrix cannot show --
    Kendall's W, ICC(3,1) and the top-1 share over all the methods at once, per
    framing. All three are computed here anyway (pair_metrics), and a reader who has
    only the table in front of them should not have to go to the per-model table or
    the CSV for the two that were previously dropped.

    `explain=False` drops the sentence defining W / ICC / top-1: the per-model
    tables are a block of a dozen otherwise identical captions, so the
    definitions are stated once, on the pooled table they all sit under."""
    def metrics(framing):
        observed, sampled = by_framing[framing][1], by_framing[framing][2]
        return (f"$W$ {format_metric(observed, sampled, 'w')}, "
                f"ICC {format_metric(observed, sampled, 'icc')}, "
                f"top-1 {format_metric(observed, sampled, 'top1', percent=True)}")

    jointly = " jointly" if explain else ""
    sentence = (f"Over all {len(methods)} methods{jointly}, base framing: "
                f"{metrics('base')}; negated: {metrics('negated')}.")
    if not explain:
        return sentence
    steps = ", ".join(f"{100 * (step + 1) // len(methods)}"
                      for step in range(len(methods)))
    return (f"{sentence} "
            f"$W$ is the concordance of the {len(methods)} orderings (1 = identical, "
            f"0 = unrelated); ICC is ICC(3,1), consistency form, over the methods' "
            f"z-scored rank profiles, so a method whose raw scale is compressed is "
            f"not charged for that; top-1 is the share of methods whose closest "
            f"group is the modal one, ties split evenly, which on {len(methods)} "
            f"methods can only be {steps}\\%.")


def matrix_caption(spec, methods, model_scope, by_framing, draws):
    """The pooled table carries the full explanation of what the matrix and the
    summary figures are; the per-model tables that follow it repeat only their
    own numbers, since a reader meets the explanation once and then wants the
    dozen model tables to be scannable."""
    title = latex_escape(matrix_title(model_scope))
    if model_scope == POOLED:
        return (f"\\textbf{{{title}.}} {spec['caption']} "
                f"{matrix_summary_sentence(methods, by_framing, draws)}")
    return (f"\\textbf{{{title}.}} "
            f"{matrix_summary_sentence(methods, by_framing, draws, explain=False)}")


def matrix_label_slug(model_scope):
    """Slug from the DIRECTORY name, not the display title -- the title now
    carries the official model name ("Kimi K2.7 Code"), and cross-references in
    the thesis should not move because a model's marketing name gained a word."""
    return "all-models" if model_scope == POOLED else slug(model_scope)


def render_matrix_latex(spec, methods, model_scope, by_framing, draws, provenance):
    """One small booktabs table per model: methods x methods, base framing in the
    upper triangle and negated in the lower."""
    row_labels, col_labels, grid = matrix_cells(methods, by_framing)
    caption = matrix_caption(spec, methods, model_scope, by_framing, draws)
    alignment = "l" + "r" * len(col_labels)
    lines = [f"% {line}" for line in provenance]
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{{alignment}}}",
        r"    \toprule",
        "    " + " & ".join(["", *(latex_escape(label) for label in col_labels)]) + r" \\",
        r"    \midrule",
    ]
    for row_label, cells in zip(row_labels, grid):
        rendered = [latex_correlation_cell(text, marker) for text, marker in cells]
        lines.append("    " + " & ".join([latex_escape(row_label), *rendered]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
             f"  \\caption{{{caption}}}",
             f"  \\label{{{spec['label']}-{matrix_label_slug(model_scope)}}}",
             r"\end{table}"]
    return "\n".join(lines)


def render_matrix_markdown(methods, model_scope, by_framing, draws):
    row_labels, col_labels, grid = matrix_cells(methods, by_framing)
    header = ["", *col_labels]
    body = [[row_label, *(markdown_cell(markdown_correlation_cell(text, marker))
                          for text, marker in cells)]
            for row_label, cells in zip(row_labels, grid)]
    widths = [max(len(str(row[index])) for row in [header, *body])
             for index in range(len(header))]
    summary = markdown_cell(matrix_summary_sentence(methods, by_framing, draws))
    lines = [f"#### {matrix_title(model_scope)}  (upper = base, lower = negated)", "",
             summary, "",
             "| " + " | ".join(c.ljust(w) for c, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
             for row in body]
    return "\n".join(lines)


def bymodel_factor_spec(factor):
    """A minimal spec for one column of the per-model table: same row scoping as
    "methods" (pooled + one row per model), but the raters are this factor's
    levels with every method pooled together -- safe here because concordance
    metrics operate on ranks (scale-free), unlike the crosslang table's JSD."""
    return dict(name="bymodel", factor=factor, dimensions=[("model", True)],
               kind="concordance")


def reliability_spec():
    """Raters = framing (base, negated) for one method at a time -- self-
    negation invariance is inherently a per-method question, so unlike bymodel's
    factors, method is never pooled here. Rows = model scopes (pooled + one
    per model), matching every other table. Built whenever "internal" or
    "negation" is requested: the internal table's disattenuated ratio needs
    these r_xx values even if the user only asked to see the matrix."""
    return dict(name="negation", factor="framing",
               dimensions=[("method", False), ("model", True)], kind="pairs")


def reliability_r_xx(values):
    """The single base-vs-negated pair's language-blocked rho -- with exactly 2
    raters (framings), pair_indices gives exactly one pair, and that value IS
    the negation-invariance statistic for this (method, model scope). Only
    valid on a negation job; an
    internal job also carries "rho_lang" but with six pairs, of which [0] is a
    cross-method correlation."""
    return values["rho_lang"][0] if "rho_lang" in values else np.nan


def reliability_cell(values, method, model_scope, sd_lookup):
    """r_xx with the between-party SD superscripted -- the SD comes from the
    internal table's own per-method score profile (see pair_metrics), and is
    co-located so a reader can immediately check whether a low r_xx comes with
    a low SD (restriction of range, e.g. semantic refusals scored as neutral)
    rather than genuine test-retest noise."""
    if values is None:
        return "n/a"
    return with_sd_superscript(format_correlation(reliability_r_xx(values)),
                               sd_lookup.get((method, model_scope)))


def render_reliability_latex(spec, methods, jobs, observed, provenance, sd_lookup):
    """Grid: rows = model scopes, columns = methods, cell = that method's own
    r_xx at that scope, next to its between-party SD. Pivoted from the flat
    (method, model) job list built by reliability_spec, the same way
    render_bymodel_latex pivots three factor job-lists -- here there is one
    job-list, keyed by job["scope"] instead."""
    by_scope = {job["scope"]: values for job, values in zip(jobs, observed)}
    model_scopes = sorted({job["scope"][1] for job in jobs},
                          key=lambda value: (value != POOLED, value))
    header = ["Model", *(MATRIX_LABEL[method] for method in methods)]
    alignment = "l" + "r" * len(methods)
    lines = [f"% {line}" for line in provenance]
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\begin{{tabular}}{{{alignment}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for index, model_scope in enumerate(model_scopes):
        if index == 1:
            lines.append(r"    \midrule")
        row_label = POOLED_ROW_LABEL if model_scope == POOLED else short_model_name(model_scope)
        cells = [reliability_cell(by_scope.get((method, model_scope)), method, model_scope,
                                  sd_lookup)
                for method in methods]
        lines.append("    " + " & ".join([latex_escape(row_label), *cells]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
             f"  \\caption{{{spec['caption']}}}",
             f"  \\label{{{spec['label']}}}",
             r"\end{table}"]
    return "\n".join(lines)


def render_reliability_markdown(methods, jobs, observed, sd_lookup):
    by_scope = {job["scope"]: values for job, values in zip(jobs, observed)}
    model_scopes = sorted({job["scope"][1] for job in jobs},
                          key=lambda value: (value != POOLED, value))
    header = ["Model", *(MATRIX_LABEL[method] for method in methods)]
    body = []
    for model_scope in model_scopes:
        row_label = POOLED_ROW_LABEL if model_scope == POOLED else short_model_name(model_scope)
        cells = [markdown_cell(reliability_cell(by_scope.get((method, model_scope)), method,
                                                model_scope, sd_lookup))
                for method in methods]
        body.append([row_label, *cells])
    widths = [max(len(str(row[i])) for row in [header, *body]) for i in range(len(header))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
             for row in body]
    return "\n".join(lines)


def bymodel_cell(values):
    """ICC(3,1) / Kendall's W / argmax share, all as bare fractions in the same
    no-leading-zero style as the correlation matrices -- three numbers on one
    scale read faster than a mix of decimals and percentages, and the cell stays
    narrow enough for a single-column layout."""
    if np.isnan(values["icc"]) or np.isnan(values["w"]):
        return "n/a"
    return "/".join(format_correlation(values[key]) for key in ("icc", "w", "top1"))


def bymodel_row_label(job):
    scope = job["scope"][0]
    return POOLED_ROW_LABEL if scope == POOLED else short_model_name(scope)


def render_bymodel_latex(spec, jobs_by_factor, observed_by_factor, provenance):
    # Every cell is three "/"-joined numbers, so the columns run wide and read
    # as one block without a separator. Tight inter-column padding plus a faint
    # rule between the annotator sets keeps them apart without the heaviness of
    # a full \vline. Needs xcolor in the preamble for \color{gray!25}.
    header = ["Model", *(label for _, label in BYMODEL_FACTORS)]
    alignment = "l" + GRAY_COLUMN_RULE.join("r" * len(BYMODEL_FACTORS))
    lines = [f"% {line}" for line in provenance]
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        f"  \\begin{{tabular}}{{{alignment}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for index, job in enumerate(jobs_by_factor[0]):
        if index == 1:
            lines.append(r"    \midrule")
        cells = [bymodel_cell(observed_by_factor[column][index])
                for column in range(len(BYMODEL_FACTORS))]
        lines.append("    " + " & ".join([latex_escape(bymodel_row_label(job)), *cells])
                     + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
             f"  \\caption{{{spec['caption']}}}",
             f"  \\label{{{spec['label']}}}",
             r"\end{table}"]
    return "\n".join(lines)


def render_bymodel_markdown(spec, jobs_by_factor, observed_by_factor):
    header = ["Model", *(label for _, label in BYMODEL_FACTORS)]
    body = []
    for index, job in enumerate(jobs_by_factor[0]):
        cells = [bymodel_cell(observed_by_factor[column][index])
                for column in range(len(BYMODEL_FACTORS))]
        body.append([bymodel_row_label(job), *cells])
    widths = [max(len(str(row[i])) for row in [header, *body]) for i in range(len(header))]
    lines = [f"#### {spec['name']}", "",
             "| " + " | ".join(c.ljust(w) for c, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
             for row in body]
    return "\n".join(lines)


def divergence_body(spec, jobs, observed, replicates, args):
    """Mean pairwise JSD, W, and the two outlier raters -- the cross-language and
    (if reused elsewhere) cross-rater distributional-agreement table."""
    dimensions = len(spec["dimensions"])
    header = [*(DIMENSION_HEADER[dimension] for dimension, _ in spec["dimensions"]),
              "$m$", "runs/rater", "Mean JSD (bits)", "Kendall's $W$",
              "Most divergent", "Most typical"]
    rows = []
    for job, values, sampled in zip(jobs, observed, replicates):
        labels = scope_labels(spec, job["scope"])
        runs = job["runs"]
        divergent, typical = job["levels"][values["divergent"]], job["levels"][values["typical"]]
        cells = [
            *labels, *row_prefix(job, runs),
            format_metric(values, sampled, "jsd"),
            format_metric(values, sampled, "w"),
            f"{divergent} ({values['rater_jsd'][values['divergent']]:.3f})",
            f"{typical} ({values['rater_jsd'][values['typical']]:.3f})",
        ]
        rows.append(cells)
        print(f"  {' / '.join(labels):52s} m={len(job['levels']):3d}  "
              f"JSD={values['jsd']:.3f}  W={values['w']:.3f}  "
              f"divergent={divergent}  typical={typical}")
    return header, rows, set(range(dimensions)) | {len(header) - 2, len(header) - 1}


def table_body(spec, jobs, observed, replicates, parties, args):
    """(header, rows, prose_columns): the column indices render_latex must escape,
    as opposed to the metric cells it must not (they already carry \\% and $...$).
    Only concordance and divergence kinds use this path -- pairs renders one small
    matrix table per job instead (see render_matrix_latex/_markdown)."""
    if spec["kind"] == "divergence":
        return divergence_body(spec, jobs, observed, replicates, args)
    return concordance_body(spec, jobs, observed, replicates, parties, args)


def format_pvalue(value):
    return "$<10^{-4}$" if value < 1e-4 else f"{value:.4f}"


def block_breaks(rows):
    """Row indices to precede with a rule: where the outermost scope changes, unless
    that scope is one row per block throughout and so needs no separating at all."""
    blocks = [(label, len(list(group)))
              for label, group in itertools.groupby(row[0] for row in rows)]
    breaks, position = set(), 0
    for index, (label, size) in enumerate(blocks):
        previous = blocks[index - 1] if index else None
        if previous and (size > 1 or previous[1] > 1 or previous[0] == POOLED):
            breaks.add(position)
        position += size
    return breaks


def render_latex(spec, header, rows, prose_columns, provenance):
    # prose_columns are free text (scope labels, party/language names) and must be
    # escaped; every other cell is written as LaTeX (\% and maths) on purpose and
    # would be double-escaped if run through latex_escape again.
    dimensions = len(spec["dimensions"])
    alignment = "l" * dimensions + "rr" + "l" * (len(header) - dimensions - 2)
    lines = [f"% {line}" for line in provenance]
    lines += [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\begin{{tabular}}{{{alignment}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    breaks = block_breaks(rows)
    for index, row in enumerate(rows):
        if index in breaks:
            lines.append(r"    \midrule")
        cells = [latex_escape(cell) if position in prose_columns else cell
                 for position, cell in enumerate(row)]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
             f"  \\caption{{{spec['caption']}}}",
             f"  \\label{{{spec['label']}}}",
             r"\end{table}"]
    return "\n".join(lines)


def render_markdown(spec, header, rows):
    plain = [column.replace("$", "").replace("\\", "") for column in header]
    body = [[str(cell).replace("\\%", "%") for cell in row] for row in rows]
    widths = [max(len(row[index]) for row in [plain, *body]) for index in range(len(plain))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(plain, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in body]
    return f"### {spec['name']}\n\n" + "\n".join(lines)


def scope_record(spec, job):
    """The scope columns common to every row of every table: one per possible
    dimension, POOLED where a table has no such dimension, so the per-table frames
    concatenate into one tidy CSV."""
    scope = dict(zip((dimension for dimension, _ in spec["dimensions"]), job["scope"]))
    record = {dimension: scope.get(dimension, POOLED) for dimension in DIMENSION_HEADER}
    record.update({"raters": len(job["levels"]), "levels": "|".join(map(str, job["levels"])),
                   "runs_per_rater_min": min(job["runs"]),
                   "runs_per_rater_max": max(job["runs"])})
    return record


def tidy_frame(spec, jobs, observed, replicates, parties):
    """The same numbers as a long CSV, for reuse outside the tex file. Columns are
    the union of what any table kind produces; a kind that does not produce a given
    metric leaves it NaN via the outer join in main()'s pd.concat."""
    records = []
    for job, values, sampled in zip(jobs, observed, replicates):
        record = {"table": spec["name"], "factor": spec["factor"], **scope_record(spec, job)}
        for key in ("w", "icc", "top1", "jsd"):
            if key not in values:
                continue
            record[key] = values[key]
            if sampled and key in sampled:
                record[f"{key}_low"], record[f"{key}_high"] = percentile_interval(sampled[key])
        if "winner" in values:
            record["modal_top1_group"] = parties[values["winner"]]
        if "divergent" in values:
            record["most_divergent"] = job["levels"][values["divergent"]]
            record["most_divergent_jsd"] = values["rater_jsd"][values["divergent"]]
            record["most_typical"] = job["levels"][values["typical"]]
            record["most_typical_jsd"] = values["rater_jsd"][values["typical"]]
        if spec["name"] == "negation" and "rho_lang" in values:
            # Keyed on the table, not just on "rho_lang" being present: the
            # internal jobs carry that key too, but with six pairs, where [0]
            # is a cross-method correlation. Here there are exactly 2 raters
            # (base, negated), so the single pair is the negation-invariance
            # statistic and fits the one-row-per-job schema as a scalar. Both
            # it and its Spearman-Brown step-up are exported: the raw value is
            # what the table reports, the stepped-up one is what the internal
            # matrix divides by.
            record["r_neg"] = reliability_r_xx(values)
            record["r_xx_spearman_brown"] = spearman_brown(record["r_neg"])
            if sampled and "rho_lang" in sampled:
                record["r_neg_low"], record["r_neg_high"] = percentile_interval(
                    sampled["rho_lang"][:, 0])
        records.append(record)
    return pd.DataFrame.from_records(records)


def pairs_frame(spec, jobs, observed, replicates):
    """Long (table, scope..., method_a, method_b, rho[, rho_low, rho_high][,
    rho_lang, rho_lang_low, rho_lang_high]) -- the full pairwise detail the
    internal-consistency table only shows a marker for."""
    records = []
    for job, values, sampled in zip(jobs, observed, replicates):
        for local, (a, b) in enumerate(pair_indices(len(job["levels"]))):
            record = {"table": spec["name"], **scope_record(spec, job),
                      "method_a": job["levels"][a], "method_b": job["levels"][b],
                      "rho": values["rho"][local]}
            if sampled:
                record["rho_low"], record["rho_high"] = percentile_interval(
                    sampled["rho"][:, local])
            if "rho_lang" in values:
                record["rho_lang"] = values["rho_lang"][local]
                if sampled:
                    record["rho_lang_low"], record["rho_lang_high"] = percentile_interval(
                        sampled["rho_lang"][:, local])
            records.append(record)
    return pd.DataFrame.from_records(records)
