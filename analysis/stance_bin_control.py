#!/usr/bin/env python3
"""Is the EP-group classifier stance-sensitive, or is it only reading the topic?

The paper's negation-invariance result (Table~\\ref{tab:negation-invariance}) can be read
two ways. Either the classifier tracks *what the text argues* and the invariance is a
robustness property, or it tracks *what the text is about* and the invariance is a
symptom: a topic detector cannot move when you flip the stance, because the topic did not
change.

This script separates the two readings with data already in the repo, holding the topic
fixed by construction. For one statement at a time it takes every opinionated text the
models wrote about that statement, splits those texts by the *regressor's* reading of
them -- the cross-encoder indirect-Likert stance, which is a separate model from the
classifier -- into a strongly-agree bin and a strongly-disagree bin, and compares the
EP-group distribution the classifier assigns to each bin. Same statement, same topic,
same languages, opposite opinions.

Two numbers per statement:

  magnitude   the total-variation distance between the agree-bin and disagree-bin group
              distributions, against a permutation null that shuffles the bin labels
              among the same texts. The null is not zero: bins differ in which models,
              languages and paraphrases land in them, and that alone moves the
              distribution. TVD at or near the null means the classifier did not notice
              the opinion flip.
  direction   the correlation, across the six groups, between the per-group probability
              shift (agree minus disagree) and the groups' official EU&I positions on
              that statement -- the per-statement data behind Table 4. Positive means the
              agree bin shifts toward the groups that actually agree. For "immigrants
              must accept our culture and values" that predicts the agree bin moving to
              ECR+ID and PPE and the disagree bin to GUE/NGL and Greens/EFA.

A classifier that is stance-sensitive scores high on both. One that is mostly a topic
detector scores at the null on magnitude, and its direction correlation is noise around
zero. The two are reported separately on purpose: a large shift in the wrong direction
and a tiny shift in the right one are different failures.

Inputs, all already produced by the main pipeline (nothing is downloaded and no model is
run): `speeches_*_scored.csv` for the cross-encoder stance and `speeches_*_classified.csv`
for the per-statement group probabilities, paired cell by cell on (statement, language,
paraphrase, framing); `euandi_2024_parties.jsonl` for the official positions. The negated
framing is folded onto the base statement's scale by flipping the stance sign, exactly as
`analysis/core/results.py` does, so both framings contribute opinions about the same
proposition; --framings base drops it if you want the check without that step.

This script is standalone: it reads the results tree and writes its own report, and
nothing else in the repository imports it or is modified by it.

Usage, from the repository root:
  python analysis/stance_bin_control.py
  python analysis/stance_bin_control.py --threshold 0.75 --plot
  python analysis/stance_bin_control.py --models kimi-k3,grok-4.5 --framings base
"""
import argparse
import difflib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.core.labels import use_short_party_labels
use_short_party_labels()   # cluster nicknames in every label (utils.PARTY_SHORT)

from analysis.analyze_all import MODEL_DIRS
from analysis.core import (
    DEFAULT_POSITIONS,
    FRAMING_ORIENTATION,
    PARTY_PROB_COLUMN,
    SLUG_TO_LABEL,
    SPEECHES_CLASSIFIED_CSV,
    SPEECHES_SCORED_CSV,
    STANCE_COLUMN,
    find_csvs,
    model_display_name,
)
from analysis.core.positions import load_party_positions, positions_path
from analysis.plotting.plot_topical_parties import statement_rows_per_axis
from utils import configure_stdout

configure_stdout()

CELL_KEYS = ["model", "framing", "language", "paraphrase", "statement_idx"]
DEFAULT_DATASET = "euandi_2024"
DEFAULT_REPORT = "results_stance_bin_control.txt"
# The immigration statement the reviewer's example names; matched case-insensitively
# against the English statement text, and only to pick which statements get a detail
# block. Nothing in the statistics depends on it.
DEFAULT_DETAIL_MATCH = "culture and values"


# --------------------------------------------------------------------------- #
# loading: one row per (model, framing, language, paraphrase, statement)
# --------------------------------------------------------------------------- #

def stance_cells(path, framing):
    """Cross-encoder stance, signed onto the base statement's scale."""
    header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns
    columns = [c for c in header if STANCE_COLUMN.match(c)]
    if not columns:
        return pd.DataFrame()
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=columns)
    orientation = FRAMING_ORIENTATION[framing]
    frames = []
    for column in columns:
        match = STANCE_COLUMN.match(column)
        frames.append(pd.DataFrame({
            "statement_idx": np.arange(len(raw)),
            "language": match.group("language"),
            "paraphrase": int(match.group("paraphrase")),
            "stance": orientation * pd.to_numeric(raw[column], errors="coerce").to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def probability_cells(path):
    """Per-statement group probabilities, wide: one column per EP group.

    `core.results.classifier_scores` averages the statement axis away -- that is the
    axis this script needs, so the parsing is repeated here rather than reused.
    """
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    blocks = {}
    for column in raw.columns:
        match = PARTY_PROB_COLUMN.match(column)
        if not match:
            continue
        probability = pd.to_numeric(raw[column], errors="coerce")
        if probability.isna().all():
            continue
        slug = match.group("slug")
        group = SLUG_TO_LABEL.get(slug, slug.replace("_", " "))
        key = (match.group("language"), int(match.group("paraphrase")))
        blocks.setdefault(key, {})[group] = probability.to_numpy(dtype=float)
    frames = []
    for (language, paraphrase), groups in blocks.items():
        frame = pd.DataFrame(groups)
        frame.insert(0, "statement_idx", np.arange(len(raw)))
        frame.insert(1, "language", language)
        frame.insert(2, "paraphrase", paraphrase)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def answer_texts(path):
    """{(language, paraphrase): texts} -- the opinionated prose both files were built
    from, used only to verify that the two files are row-aligned."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    pattern = re.compile(r"^answer_(?P<language>[a-z]{2})_v(?P<paraphrase>\d+)$")
    return {(m.group("language"), int(m.group("paraphrase"))): raw[c].astype(str).to_numpy()
            for c in raw.columns if (m := pattern.match(c))}


def check_alignment(scored_path, classified_path):
    """The two files are joined positionally, so confirm they hold the same texts.

    Both are written from the same speeches CSV, but nothing in the tree records that,
    and a silent misalignment would put one text's stance next to another's group
    probabilities -- which is exactly the pairing this whole script rests on.
    """
    left, right = answer_texts(scored_path), answer_texts(classified_path)
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0, 0
    mismatched = sum(1 for key in shared if not np.array_equal(left[key], right[key]))
    return len(shared), mismatched


def load_model(model_dir, framings):
    """Stance and group probabilities for one model, paired cell by cell."""
    scored = find_csvs(model_dir, SPEECHES_SCORED_CSV)
    classified = find_csvs(model_dir, SPEECHES_CLASSIFIED_CSV)
    frames, checked, mismatched = [], 0, 0
    for framing in framings:
        if framing not in scored or framing not in classified:
            continue
        columns, bad = check_alignment(scored[framing], classified[framing])
        checked += columns
        mismatched += bad
        stances = stance_cells(scored[framing], framing)
        probabilities = probability_cells(classified[framing])
        if stances.empty or probabilities.empty:
            continue
        merged = stances.merge(probabilities, on=["statement_idx", "language", "paraphrase"],
                               how="inner")
        frames.append(merged.assign(model=model_dir.name, framing=framing))
    if not frames:
        return pd.DataFrame(), checked, mismatched
    return pd.concat(frames, ignore_index=True), checked, mismatched


def load_cells(dataset, models, framings):
    frames, checked, mismatched = [], 0, 0
    for name in models:
        model_dir = Path("data") / f"{dataset}_results" / name
        if not model_dir.is_dir():
            print(f"  {name}: no results directory, skipping.")
            continue
        cells, columns, bad = load_model(model_dir, framings)
        checked += columns
        mismatched += bad
        if cells.empty:
            print(f"  {name}: no paired scored/classified files, skipping.")
            continue
        frames.append(cells)
        print(f"  {name}: {len(cells)} texts over "
              f"{cells['language'].nunique()} languages, "
              f"{cells['paraphrase'].nunique()} paraphrases, "
              f"{cells['framing'].nunique()} framing(s)")
    if not frames:
        raise SystemExit("No model produced a paired scored/classified pair; nothing to do.")
    if mismatched:
        raise SystemExit(f"{mismatched} of {checked} (language, paraphrase) columns differ "
                         "between the scored and classified files: they are not row-aligned, "
                         "so stance and group probabilities cannot be paired positionally.")
    print(f"  row alignment verified on {checked} (language, paraphrase) answer columns")
    return pd.concat(frames, ignore_index=True)


def load_statements(dataset):
    path = Path("data") / f"{dataset}_data" / "statements.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["statement"]["en"] for line in f if line.strip()]


def normalize(text):
    return " ".join(str(text).split())


def match_statement(key, pool, cutoff, margin=0.10):
    """`key`'s row in the positions file, exactly if possible and by wording if not.

    Two statements were reworded between the positions file and `statements.jsonl`
    ("EU member states" became "European Union"), so an exact text join drops them. The
    fallback is deliberately narrow: the best candidate has to clear `cutoff` *and* beat
    the runner-up by `margin`, so a genuinely ambiguous statement raises instead of being
    quietly paired with the wrong policy. Every inexact match is printed.
    """
    if key in pool:
        return key
    scored = sorted(((difflib.SequenceMatcher(None, key, candidate).ratio(), candidate)
                     for candidate in pool), reverse=True)
    if not scored or scored[0][0] < cutoff:
        raise SystemExit(f"No positions row matches {key!r} (best "
                         f"{scored[0][0]:.2f}: {scored[0][1]!r}); the positions file and "
                         "statements.jsonl have drifted too far to join on text.")
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        raise SystemExit(f"{key!r} matches two positions rows about equally well "
                         f"({scored[0][1]!r} at {scored[0][0]:.2f}, {scored[1][1]!r} at "
                         f"{scored[1][0]:.2f}); refusing to guess.")
    print(f"  matched by wording ({scored[0][0]:.2f}): {key!r}\n"
          f"                  -> {scored[0][1]!r}")
    return scored[0][1]


def load_positions(positions, groups, statements, cutoff):
    """Official positions in the results files' row order: (statements, groups).

    **The join is on statement text, not on `statement_idx`.** The two are not the same
    key. `euandi_2024_parties.jsonl` numbers statements in the EU&I codebook order, the
    same order `euandi_2024_questionnaire.jsonl` uses, while every results CSV is written
    in `statements.jsonl` order -- and the two orders agree on only 11 of the 30
    statements. `core.questionnaire.load_axis_directions` already joins the axis coding on
    text for exactly this reason; the positions are the same shape of problem, so they are
    joined the same way here. The mismatch count is reported at load time rather than
    silently absorbed.

    ECR and ID are averaged into ECR+ID to match the classifier's collapsed label set.
    """
    # `core.positions_frame` would drop the statement text, which is the join key here,
    # so the frame is built from the loader underneath it and the ECR+ID collapse that
    # `positions_frame` applies is repeated below.
    party_df = load_party_positions(positions_path(positions), positions)
    party_df = party_df.dropna(subset=["ep_group"]).copy()
    party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    by_text = (party_df.assign(key=party_df["statement"].map(normalize))
               .groupby(["key", "ep_group"])["normalized_answer"].mean()
               .unstack("ep_group"))
    missing_groups = [g for g in groups if g not in by_text.columns]
    if missing_groups:
        raise SystemExit(f"No official positions for {missing_groups}; the classifier's "
                         f"label set and the {positions} positions basis do not line up.")

    keys = [match_statement(normalize(s), list(by_text.index), cutoff)
            for s in statements]
    duplicated = {k for k in keys if keys.count(k) > 1}
    if duplicated:
        raise SystemExit(f"{len(duplicated)} positions row(s) matched more than one "
                         f"statement, so the join is not one-to-one; first: "
                         f"{sorted(duplicated)[0]!r}")
    positional = party_df.groupby("statement_idx")["statement"].first().map(normalize)
    agree = sum(1 for i, key in enumerate(keys) if positional.get(i) == key)
    print(f"  positions joined on statement text; a positional join on statement_idx would "
          f"have matched {agree}/{len(keys)} statements")
    return by_text.loc[keys, groups].to_numpy(dtype=float), agree


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else np.nan


def spearman(x, y):
    """Rank correlation without a scipy dependency."""
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return pearson(x[ok].rank().to_numpy(), y[ok].rank().to_numpy())


def sign_test(positive, total):
    """Two-sided binomial p against p=0.5, for 'how many statements shift the right way'."""
    if total == 0:
        return np.nan
    extreme = max(positive, total - positive)
    tail = sum(math.comb(total, k) for k in range(extreme, total + 1)) / 2 ** total
    return float(min(1.0, 2 * tail))


def stratified_labels(labels, strata, rng):
    """`labels` reshuffled inside each stratum, so every stratum keeps its own bin sizes.

    This is what makes the null a control rather than a formality. The bins are badly
    confounded with framing -- models mostly agree with whatever proposition they are
    shown, so the negated framing supplies most of the disagree bin -- and an unstratified
    shuffle destroys that imbalance, crediting the classifier for a difference that is
    really "negated-framing prose looks different". Permuting within strata reproduces the
    observed composition exactly, so the excess over the null is what stance adds on top.
    """
    shuffled = labels.copy()
    for stratum in strata:
        shuffled[stratum] = rng.permutation(labels[stratum])
    return shuffled


def statement_record(probabilities, stance, strata_codes, positions, threshold, draws, rng):
    """Magnitude and direction for one statement, with their permutation nulls."""
    agree = stance >= threshold
    disagree = stance <= -threshold
    n_agree, n_disagree = int(agree.sum()), int(disagree.sum())
    record = dict(n_agree=n_agree, n_disagree=n_disagree, n_texts=len(stance))
    if n_agree == 0 or n_disagree == 0:
        return record, None, None

    agree_mean = probabilities[agree].mean(axis=0)
    disagree_mean = probabilities[disagree].mean(axis=0)
    shift = agree_mean - disagree_mean
    record["tvd"] = float(np.abs(shift).sum() / 2)
    record["r_pearson"] = pearson(shift, positions)
    record["r_spearman"] = spearman(shift, positions)
    # Argmax share reproduces what `classify_speeches.py` actually deploys (the hard
    # label), and the softmax here is near-degenerate, so the two rarely disagree.
    winners = probabilities.argmax(axis=1)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(winners)), winners] = 1.0
    argmax_shift = onehot[agree].mean(axis=0) - onehot[disagree].mean(axis=0)
    record["tvd_argmax"] = float(np.abs(argmax_shift).sum() / 2)
    record["r_pearson_argmax"] = pearson(argmax_shift, positions)

    # The null keeps the same texts and only forgets which bin each was in, stratum by
    # stratum, so it carries the heterogeneity a bin split induces even when the
    # classifier ignores stance entirely.
    pool = np.flatnonzero(agree | disagree)
    labels = agree[pool]
    codes = strata_codes[pool]
    strata = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    strata = [s for s in strata if 0 < labels[s].sum() < len(s)]
    pooled_probabilities = probabilities[pool]
    null_tvd, null_r = np.empty(draws), np.empty(draws)
    for draw in range(draws):
        shuffled = stratified_labels(labels, strata, rng)
        null_shift = (pooled_probabilities[shuffled].mean(axis=0)
                      - pooled_probabilities[~shuffled].mean(axis=0))
        null_tvd[draw] = np.abs(null_shift).sum() / 2
        null_r[draw] = pearson(null_shift, positions)
    record["n_strata"] = len(strata)
    record["tvd_null"] = float(null_tvd.mean())
    record["tvd_p"] = float((np.count_nonzero(null_tvd >= record["tvd"]) + 1) / (draws + 1))
    if np.isfinite(record["r_pearson"]):
        finite = null_r[np.isfinite(null_r)]
        record["r_p"] = float((np.count_nonzero(np.abs(finite) >= abs(record["r_pearson"])) + 1)
                              / (len(finite) + 1))
    return record, (agree_mean, disagree_mean, shift), null_r


def strata_codes(cells, columns):
    """An integer stratum id per text, from the cell columns the null holds fixed."""
    if not columns:
        return np.zeros(len(cells), dtype=int)
    keys = cells[columns].astype(str).agg("|".join, axis=1)
    return pd.factorize(keys)[0]


def analyse(cells, groups, positions, statements, args, rng, strata=None):
    """One record per statement, plus the per-statement group vectors for the report."""
    probabilities = cells[groups].to_numpy(dtype=float)
    stance = cells["stance"].to_numpy(dtype=float)
    index = cells["statement_idx"].to_numpy()
    codes = strata_codes(cells, args.strata if strata is None else strata)

    records, vectors, nulls = [], {}, {}
    for statement_idx in range(len(statements)):
        rows = np.flatnonzero(index == statement_idx)
        if len(rows) == 0:
            continue
        record, vector, null_r = statement_record(
            probabilities[rows], stance[rows], codes[rows], positions[statement_idx],
            args.threshold, args.draws, rng)
        record.update(statement_idx=statement_idx, statement=statements[statement_idx],
                      position_spread=float(np.nanstd(positions[statement_idx])))
        records.append(record)
        if vector is not None:
            vectors[statement_idx] = vector
            nulls[statement_idx] = null_r
    table = pd.DataFrame.from_records(records)
    # The two headline statistics do not have the same requirements, and collapsing them
    # into one flag understates the magnitude result. Magnitude only needs both bins
    # filled. Direction additionally needs the groups to actually disagree on the
    # statement: on a few of them every group holds the identical official position, so
    # the reference vector is constant and the correlation is undefined -- but the
    # distributions still move, and that movement is evidence.
    table["has_bins"] = ((table["n_agree"] >= args.min_bin)
                         & (table["n_disagree"] >= args.min_bin))
    table["usable"] = table["has_bins"] & (table["position_spread"] > 0)
    return table, vectors, nulls


def pooled_direction(table, vectors, positions, groups, draws, rng):
    """Correlation of shift against position over every (statement, group) cell.

    Both sides are centred within a statement first. Without that, the correlation is
    carried by the group main effects -- PPE is a common prediction and a frequent
    agreer, which would score as stance sensitivity no matter how the bins were drawn.
    """
    shifts, wanted = [], []
    for statement_idx in table.loc[table["usable"], "statement_idx"]:
        _, _, shift = vectors[statement_idx]
        position = positions[statement_idx]
        shifts.append(shift - shift.mean())
        wanted.append(position - np.nanmean(position))
    if not shifts:
        return np.nan, np.nan, 0
    shifts, wanted = np.concatenate(shifts), np.concatenate(wanted)
    observed = pearson(shifts, wanted)
    # Permuting the group labels within a statement is the null for "the shift lands on
    # the groups the positions point to" while keeping each statement's shift magnitudes.
    blocks = shifts.reshape(-1, len(groups))
    null = np.empty(draws)
    for draw in range(draws):
        permuted = np.concatenate([rng.permutation(block) for block in blocks])
        null[draw] = pearson(permuted, wanted)
    p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (draws + 1))
    return observed, p_value, len(blocks)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def fmt(value, places=3):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{places}f}"


def markdown_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = lambda cells: "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), line(["-" * w for w in widths]),
                      *(line(r) for r in rows)])


def truncate(text, width=58):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + "…"


def headline_section(table, pooled, pooled_p, n_blocks, groups, args):
    magnitude = table[table["has_bins"]]
    usable = table[table["usable"]]
    lines = ["=" * 78, "HEADLINE", "=" * 78, ""]
    if magnitude.empty:
        return "\n".join(lines + [f"No statement has {args.min_bin}+ texts in both bins at "
                                  f"threshold {args.threshold}; lower --threshold or --min_bin."])

    tvd, null = magnitude["tvd"].mean(), magnitude["tvd_null"].mean()
    beaten = int((magnitude["tvd_p"] < 0.05).sum())
    constant = len(magnitude) - len(usable)
    lines += [
        f"Statements with >= {args.min_bin} texts in both bins: {len(magnitude)} of "
        f"{len(table)}. Of those, {len(usable)} also have groups that differ on the "
        f"statement, which the direction test needs and the magnitude test does not"
        + (f" ({constant} statement(s) carry an identical position for every group)."
           if constant else "."),
        "",
        f"MAGNITUDE ({len(magnitude)} statements) -- do the two bins get different "
        "group distributions?",
        f"  mean TVD, agree bin vs disagree bin   {fmt(tvd)}",
        f"  mean TVD under the shuffled-bin null  {fmt(null)}",
        f"  excess over null                      {fmt(tvd - null)}"
        f"  ({fmt(tvd / null, 2) if null > 0 else 'n/a'}x)",
        f"  statements beating their own null     {beaten}/{len(magnitude)} at p<0.05",
        "  (TVD is 0 when the bins get identical distributions and 1 when they share none.)",
        "",
        f"DIRECTION ({len(usable)} statements) -- does the shift go where the official "
        "positions say it should?",
        f"  pooled r over {n_blocks * len(groups)} (statement, group) cells, "
        f"both sides centred within statement: {fmt(pooled)}",
        f"  permutation p (group labels shuffled within statement): {fmt(pooled_p, 4)}",
    ]
    per_statement = usable["r_pearson"].dropna()
    if len(per_statement):
        positive = int((per_statement > 0).sum())
        lines += [
            f"  per-statement r: mean {fmt(per_statement.mean())}, "
            f"median {fmt(per_statement.median())}",
            f"  statements shifting the right way: {positive}/{len(per_statement)} "
            f"(sign test p = {fmt(sign_test(positive, len(per_statement)), 4)})",
        ]
    lines += ["", f"Hard-label cross-check (argmax, as deployed): mean TVD "
                  f"{fmt(magnitude['tvd_argmax'].mean())} over {len(magnitude)}, mean "
                  f"per-statement r {fmt(usable['r_pearson_argmax'].mean())} over "
                  f"{len(usable)}."]
    return "\n".join(lines)


def group_section(table, vectors, positions, groups):
    """Where the probability mass moves, per group, averaged over statements.

    Split by the sign of the group's own position, since a group that agrees with a
    statement should gain in the agree bin and lose in the disagree bin.
    """
    usable = list(table.loc[table["usable"], "statement_idx"])
    rows = []
    for column, group in enumerate(groups):
        pro, against = [], []
        for statement_idx in usable:
            _, _, shift = vectors[statement_idx]
            position = positions[statement_idx][column]
            if not np.isfinite(position):
                continue
            (pro if position > 0 else against).append(shift[column])
        rows.append([group, len(pro), fmt(np.mean(pro)) if pro else "n/a",
                     len(against), fmt(np.mean(against)) if against else "n/a"])
    body = markdown_table(
        ["group", "n stmt +", "mean shift", "n stmt -", "mean shift"], rows)
    return "\n".join([
        "=" * 78, "PER-GROUP SHIFT", "=" * 78, "",
        "Mean (agree bin - disagree bin) probability for each group, split by whether the",
        "group's own official position on the statement is positive or negative. Stance",
        "sensitivity means the '+' column is positive and the '-' column negative.", "",
        body])


def axis_section(table, dataset, n_statements):
    rows = []
    for axis, statement_rows in statement_rows_per_axis(dataset, n_statements).items():
        sub = table[table["statement_idx"].isin(statement_rows) & table["usable"]]
        if sub.empty:
            rows.append([axis, 0, "n/a", "n/a", "n/a"])
            continue
        rows.append([axis, len(sub), fmt(sub["tvd"].mean()), fmt(sub["tvd_null"].mean()),
                     fmt(sub["r_pearson"].mean())])
    return "\n".join(["=" * 78, "BY TOPIC", "=" * 78, "",
                      markdown_table(["axis", "n stmt", "mean TVD", "null TVD", "mean r"],
                                     rows)])


def composition_section(cells, args):
    """How the two bins are made up, per framing -- the confound the null stratifies on."""
    binned = np.where(cells["stance"] >= args.threshold, "agree bin",
                      np.where(cells["stance"] <= -args.threshold, "disagree bin",
                               "unbinned (mid)"))
    counts = pd.crosstab(cells["framing"], binned)
    rows = [[framing, *(f"{counts.loc[framing].get(column, 0)}" for column in counts.columns)]
            for framing in counts.index]
    rows.append(["all", *(f"{counts[column].sum()}" for column in counts.columns)])
    return "\n".join([
        "=" * 78, "BIN COMPOSITION", "=" * 78, "",
        "Models tend to agree with whatever proposition they are shown, so the negated",
        "framing supplies most of the disagree bin. That imbalance would move the group",
        "distribution on its own; it is why the null permutes bin labels within",
        f"{' x '.join(args.strata) if args.strata else 'nothing (unstratified)'} rather "
        "than freely, and why the per-framing", "table below is worth reading.", "",
        markdown_table(["framing", *counts.columns], rows)])


def framing_section(cells, groups, positions, statements, args, rng):
    """The same check inside a single framing, where the confound cannot arise at all.

    Within one framing every text answers the identical prompt, so an agree/disagree
    split there is as clean as this data gets. The price is bin size: the base framing's
    disagree bin is thin, hence its own --min_bin_framing floor.
    """
    inner = argparse.Namespace(**{**vars(args), "min_bin": args.min_bin_framing})
    rows = []
    for framing, sub in cells.groupby("framing"):
        table, vectors, _ = analyse(sub, groups, positions, statements, inner, rng, strata=[])
        usable = table[table["usable"]]
        if usable.empty:
            rows.append([framing, 0, "n/a", "n/a", "n/a"])
            continue
        pooled, _, _ = pooled_direction(table, vectors, positions, groups, args.draws, rng)
        rows.append([framing, len(usable), fmt(usable["tvd"].mean()),
                     fmt(usable["tvd_null"].mean()), fmt(pooled)])
    return "\n".join(["=" * 78, "WITHIN ONE FRAMING", "=" * 78, "",
                      f"Statements need {args.min_bin_framing}+ texts in both bins here.", "",
                      markdown_table(["framing", "n stmt", "mean TVD", "null TVD", "pooled r"],
                                     rows)])


def model_section(cells, groups, positions, statements, args, rng):
    """The same headline numbers one model at a time, to show the effect is not one
    model's quirk (and that no single model is carrying it)."""
    rows = []
    for model, sub in cells.groupby("model"):
        table, vectors, _ = analyse(sub, groups, positions, statements, args, rng)
        magnitude = table[table["has_bins"]]
        if magnitude.empty:
            rows.append([model_display_name(model), 0, "n/a", "n/a", "n/a"])
            continue
        pooled, _, _ = pooled_direction(table, vectors, positions, groups, args.draws, rng)
        rows.append([model_display_name(model), len(magnitude),
                     fmt(magnitude["tvd"].mean()),
                     fmt(magnitude["tvd_null"].mean()), fmt(pooled)])
    return "\n".join(["=" * 78, "BY MODEL", "=" * 78, "",
                      markdown_table(["model", "n stmt", "mean TVD", "null TVD", "pooled r"],
                                     rows)])


def statement_section(table):
    rows = []
    for _, row in table.sort_values("statement_idx").iterrows():
        rows.append([
            row["statement_idx"], truncate(row["statement"]),
            row["n_agree"], row["n_disagree"],
            fmt(row.get("tvd")), fmt(row.get("tvd_null")), fmt(row.get("r_pearson")),
            fmt(row.get("r_spearman")),
            "" if row["usable"] else
            ("thin bins" if not row["has_bins"] else "no position spread"),
        ])
    return "\n".join(["=" * 78, "PER STATEMENT", "=" * 78, "",
                      markdown_table(["#", "statement", "n agr", "n dis", "TVD", "null",
                                      "r", "rho", ""], rows)])


def detail_section(table, vectors, positions, groups, args):
    """Full group distributions for a few statements, so the direction claim is legible
    rather than a single correlation."""
    usable = table[table["usable"]].copy()
    if usable.empty:
        return ""
    wanted = list(usable.nlargest(args.detail, "tvd")["statement_idx"])
    if args.detail_match:
        matched = usable[usable["statement"].str.contains(args.detail_match, case=False,
                                                          regex=False)]
        wanted = list(matched["statement_idx"]) + [i for i in wanted
                                                   if i not in set(matched["statement_idx"])]
    blocks = ["=" * 78, "STATEMENT DETAIL", "=" * 78]
    for statement_idx in wanted[:max(args.detail, 1)]:
        row = usable[usable["statement_idx"] == statement_idx].iloc[0]
        agree_mean, disagree_mean, shift = vectors[statement_idx]
        order = np.argsort(-shift)
        body = markdown_table(
            ["group", "agree bin", "disagree bin", "shift", "official position"],
            [[groups[i], fmt(agree_mean[i]), fmt(disagree_mean[i]), f"{shift[i]:+.3f}",
              fmt(positions[statement_idx][i], 2)] for i in order])
        blocks += [
            "", f"[{statement_idx}] {truncate(row['statement'], 100)}",
            f"  {row['n_agree']} texts in the agree bin, {row['n_disagree']} in the "
            f"disagree bin (|stance| >= {args.threshold})",
            f"  TVD {fmt(row['tvd'])} (null {fmt(row['tvd_null'])}, p={fmt(row['tvd_p'], 4)})"
            f"   r {fmt(row['r_pearson'])}   rho {fmt(row['r_spearman'])}",
            "", body,
        ]
    return "\n".join(blocks)


def verdict_section(table, pooled, pooled_p):
    magnitude = table[table["has_bins"]]
    if magnitude.empty:
        return ""
    tvd, null = magnitude["tvd"].mean(), magnitude["tvd_null"].mean()
    excess = tvd - null
    moved = excess > 0.05 and tvd > 1.5 * null if null > 0 else excess > 0.05
    aimed = np.isfinite(pooled) and pooled > 0.15 and pooled_p < 0.05
    if moved and aimed:
        verdict = ("The bins get substantially different group distributions and the shift "
                   "lands on the groups that hold the matching position. On this evidence "
                   "the classifier is stance-sensitive, and negation invariance is a "
                   "robustness property rather than a topic artefact.")
    elif moved and not aimed:
        verdict = ("The bins do get different group distributions, but the shift is not "
                   "aligned with the groups' official positions. Something other than "
                   "stance separates the bins -- register, length or which model wrote the "
                   "text are the candidates -- so this does not support a stance reading.")
    elif aimed:
        verdict = ("The shift points the right way but is small against the null. The "
                   "classifier registers stance weakly; on the scale the paper uses it, "
                   "topic dominates the prediction.")
    else:
        verdict = ("Splitting the texts by opinion barely moves the group distribution and "
                   "what movement there is does not follow the official positions. The "
                   "classifier is mostly reading the topic, and negation invariance is a "
                   "symptom of that, not a strength.")
    return "\n".join(["=" * 78, "READING", "=" * 78, "", verdict])


def plot(table, vectors, positions, groups, path):
    usable = table[table["usable"]]
    if usable.empty:
        return
    x, y = [], []
    for statement_idx in usable["statement_idx"]:
        _, _, shift = vectors[statement_idx]
        position = positions[statement_idx]
        x.append(position - np.nanmean(position))
        y.append(shift - shift.mean())
    x, y = np.concatenate(x), np.concatenate(y)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    left.axhline(0, color="0.8", lw=0.8)
    left.axvline(0, color="0.8", lw=0.8)
    left.scatter(x, y, s=18, alpha=0.6, color="#3b6ea5", edgecolor="none")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2 and np.std(x[ok]) > 0:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        grid = np.linspace(x[ok].min(), x[ok].max(), 2)
        left.plot(grid, slope * grid + intercept, color="#b4451f", lw=1.6)
    left.set_xlabel("official position on the statement (centred within statement)")
    left.set_ylabel("probability shift, agree bin - disagree bin")
    left.set_title(f"Direction: r = {fmt(pearson(x, y))}")

    magnitude = table[table["has_bins"]]
    order = np.argsort(-magnitude["tvd"].to_numpy())
    tvd = magnitude["tvd"].to_numpy()[order]
    null = magnitude["tvd_null"].to_numpy()[order]
    positions_x = np.arange(len(tvd))
    right.bar(positions_x, tvd, color="#3b6ea5", label="observed")
    right.bar(positions_x, null, color="none", edgecolor="#b4451f", hatch="///",
              label="shuffled-bin null")
    right.set_xlabel("statements, sorted by TVD")
    right.set_ylabel("total-variation distance between bins")
    right.set_title("Magnitude")
    right.legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--models", default=",".join(MODEL_DIRS),
                        help="comma-separated result directory names")
    parser.add_argument("--framings", default="base,negated",
                        help="which framings to pool; negated stances are flipped onto the "
                             "base statement's scale before binning")
    parser.add_argument("--languages", default="",
                        help="comma-separated language codes (default: all)")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS,
                        help="positions basis for the official group answers")
    parser.add_argument("--match_cutoff", type=float, default=0.75,
                        help="similarity a reworded statement needs to be joined to a "
                             "positions row when the texts are not identical")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="|stance| a text needs to enter a bin (default 0.5)")
    parser.add_argument("--min_bin", type=int, default=25,
                        help="texts a statement needs in *both* bins to be analysed")
    parser.add_argument("--strata", default="framing",
                        help="comma-separated cell columns the permutation null holds "
                             "fixed (framing, model, language, paraphrase; '' for none). "
                             "Framing is stratified by default because it is the bins' "
                             "main confound")
    parser.add_argument("--min_bin_framing", type=int, default=10,
                        help="the same floor for the within-framing table, which has "
                             "smaller bins to work with")
    parser.add_argument("--draws", type=int, default=2000, help="permutation draws")
    parser.add_argument("--detail", type=int, default=6,
                        help="statements to print full group distributions for")
    parser.add_argument("--detail_match", default=DEFAULT_DETAIL_MATCH,
                        help="always show statements whose English text contains this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--csv", default="", help="also write the per-statement table here")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_path", default="stance_bin_control.png")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    unknown = [f for f in framings if f not in FRAMING_ORIENTATION]
    if unknown:
        raise SystemExit(f"Unknown framing(s) {unknown}; choose from "
                         f"{sorted(FRAMING_ORIENTATION)}.")
    allowed = {"model", "framing", "language", "paraphrase"}
    args.strata = [c.strip() for c in args.strata.split(",") if c.strip()]
    if not allowed.issuperset(args.strata):
        raise SystemExit(f"--strata may only name {sorted(allowed)}; "
                         f"got {sorted(set(args.strata) - allowed)}.")
    if len(framings) < 2 and "framing" in args.strata:
        args.strata = [c for c in args.strata if c != "framing"]

    print(f"Loading {args.dataset} results ({', '.join(framings)} framing)")
    cells = load_cells(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()],
                       framings)
    if args.languages:
        wanted = {l.strip() for l in args.languages.split(",") if l.strip()}
        cells = cells[cells["language"].isin(wanted)]
    cells = cells[cells["stance"].notna()].reset_index(drop=True)

    groups = sorted(c for c in cells.columns if c not in
                    {*CELL_KEYS, "stance", "model", "framing"})
    statements = load_statements(args.dataset)
    if cells["statement_idx"].max() >= len(statements):
        raise SystemExit(f"Results have more statements than "
                         f"data/{args.dataset}_data/statements.jsonl lists.")
    positions, positional_matches = load_positions(args.positions, groups, statements,
                                                   args.match_cutoff)
    print(f"\n{len(cells)} opinionated texts, {len(groups)} EP groups "
          f"({', '.join(groups)}), {len(statements)} statements")

    table, vectors, _ = analyse(cells, groups, positions, statements, args, rng)
    pooled, pooled_p, n_blocks = pooled_direction(table, vectors, positions, groups,
                                                  args.draws, rng)

    header = "\n".join([
        "Stance-bin control for the EP-group classifier",
        f"dataset={args.dataset}  positions={args.positions}  framings={','.join(framings)}",
        f"threshold=|stance|>={args.threshold}  min_bin={args.min_bin}  draws={args.draws}  "
        f"seed={args.seed}",
        f"null strata={'x'.join(args.strata) if args.strata else 'none (unstratified)'}",
        f"{len(cells)} texts from {cells['model'].nunique()} model(s), "
        f"{cells['language'].nunique()} language(s), "
        f"{cells['paraphrase'].nunique()} paraphrase(s)",
        f"positions joined on statement text ({positional_matches}/{len(statements)} "
        f"statements would also have matched a positional statement_idx join)",
        "",
    ])
    sections = [
        header,
        headline_section(table, pooled, pooled_p, n_blocks, groups, args),
        composition_section(cells, args),
        group_section(table, vectors, positions, groups),
        framing_section(cells, groups, positions, statements, args, rng)
        if len(framings) > 1 else "",
        axis_section(table, args.dataset, len(statements)),
        model_section(cells, groups, positions, statements, args, rng),
        detail_section(table, vectors, positions, groups, args),
        statement_section(table),
        verdict_section(table, pooled, pooled_p),
    ]
    report = "\n\n".join(s for s in sections if s)
    print("\n" + report)
    Path(args.report).write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {args.report}")

    if args.csv:
        table.to_csv(args.csv, sep=";", index=False, encoding="utf-8-sig")
        print(f"Wrote {args.csv}")
    if args.plot:
        plot(table, vectors, positions, groups, args.plot_path)


if __name__ == "__main__":
    main()
