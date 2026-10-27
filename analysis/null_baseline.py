"""The null of every argmax method, gaussian or uniform, read back as one lookup.

vaa_null_model_argmax.py writes the null of the two VAA methods (direct and indirect
Likert), classifier_null_model.py that of the two classifier methods (classified reasons,
classified open-ended). Both are argmax shares on the scale of plot_argmax_shares'
shares, so a share minus its null is a difference in percentage points: what a group
wins beyond what random stances around the neutral answer would already give it.

This is the one place their files are read, so the figures and the table subtract the
same numbers:

  NullBaseline.load(null_dir, positions, sigma, kind).share(method, variant, scope, slice)

  kind     "gaussian" (the 0-centered normal null, sigma `sigma`), "uniform" (every
           stance iid uniform over the five Likert points; `sigma` is ignored) or "max"
           (per group the largest of the normal nulls at vaa_null_model.ENVELOPE_SIGMAS
           and the uniform null; needs every one of those nulls' files)

  method   plot_argmax_shares' keys: likert_vaa, speeches_vaa, reasons_clf, speeches_clf
  variant  base, negated or pooled. The VAA null is framing-free -- a 0-centered null is
           symmetric under negation -- so it serves every variant; the classifier null
           is read per variant, since the classifier's response to a stance is not.
  scope    "all" (slice WHOLE), "topic" (slice = axis name) or "language" (slice = code).
           The VAA null of one language is its null over all statements: random answers
           do not depend on the language, and the positions do not either.

plot_classified_parties' combined figure shows neither argmax shares nor anything per
slice, but mean VAA agreement and mean classifier probability; .agreement(source) and
.mean_probability(method, variant) serve the same null in those units.

A missing null raises rather than returning nothing, so a figure can never quietly fall
back to the raw share under a "minus null" label.

Figures and tables drawn minus a null go to <results_dir>/plots_<kind> and
tables_<kind> (output_dir_name), so the two nulls never overwrite each other or the raw
outputs in plots/ and tables/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.classifier_null_model import (
    ALL_SCOPE, LEVELS, level_probabilities, output_stem,
)
from analysis.vaa_null_model import (
    ALL_MODELS, DEFAULT_SIGMA, ENVELOPE, ENVELOPE_SIGMAS, SUBTRACTED_NULLS,
)
from analysis.vaa_null_model_argmax import (   # noqa: F401  (NULL_KINDS re-exported)
    NULL_KINDS, NULL_SIDE, WHOLE, null_suffix, output_suffix,
)
from analysis.vaa_scoring import level_posteriors, scoring_suffix, vaa_scoring

VAA_METHOD = {"likert": "likert_vaa", "speeches": "speeches_vaa"}
CLASSIFIER_METHOD = {"classified reasons": "reasons_clf",
                     "classified open-ended": "speeches_clf"}
ANY_VARIANT = "any"   # the VAA null is framing-free


def output_dir_name(base, kind, sigma=DEFAULT_SIGMA):
    """plots_gaussian / tables_uniform: where the outputs drawn minus a null go, next to
    the raw plots/ and tables/. A non-default sigma tags the gaussian folder
    (plots_gaussian_sigma0.25), as it tags the null's own files."""
    return f"{base}_{kind}{null_suffix(kind, sigma) if kind == 'gaussian' else ''}"


def null_description(kind, sigma=DEFAULT_SIGMA):
    """The null in words, for captions: "a normal distribution around the neutral answer,
    $\\sigma$ = 0.5" or "a uniform distribution over the five Likert points"."""
    if kind == "uniform":
        return "a uniform distribution over the five Likert points"
    if kind == ENVELOPE:
        sigmas = ", ".join(f"{s:g}" for s in ENVELOPE_SIGMAS)
        return ("the least favourable of five distributions, taken per group: normal "
                f"distributions around the neutral answer with $\\sigma \\in \\{{{sigmas}\\}}$ "
                "and a uniform distribution over the five Likert points")
    return f"a normal distribution around the neutral answer, $\\sigma$ = {sigma:g}"


def envelope_components():
    """(sigma, kind) of every null the "max" null is the per-group maximum of."""
    return [*((sigma, "gaussian") for sigma in ENVELOPE_SIGMAS), (DEFAULT_SIGMA, "uniform")]


def _largest(series):
    """Per group, the largest value over a list of per-group Series."""
    return pd.concat(series, axis=1).max(axis=1)


class NullBaseline:
    def __init__(self, table, agreement, source):
        # {(method, variant or ANY_VARIANT, scope, slice): null value per group}, for the
        # argmax share and, classifier rows only, the mean probability.
        keyed = {key: frame.set_index("ep_group")
                 for key, frame in table.groupby(["method", "variant", "scope", "slice"])}
        self._shares = {key: frame["null_share"] for key, frame in keyed.items()}
        self._probabilities = {key: frame["null_mean_probability"]
                               for key, frame in keyed.items()
                               if frame["null_mean_probability"].notna().all()}
        # {"likert" / "speeches": null mean agreement per group}
        self._agreement = {source: frame.set_index("ep_group")["null_agreement"]
                           for source, frame in agreement.groupby("method")}
        self.source = source

    @classmethod
    def envelope(cls, parts):
        """The "max" null of several loaded nulls: every lookup returns, per group, the
        largest value any of them has. A key missing from one of them is dropped, so a
        lookup of it raises rather than quietly taking the maximum of fewer nulls."""
        combined = cls.__new__(cls)
        for name in ("_shares", "_probabilities", "_agreement"):
            tables = [getattr(part, name) for part in parts]
            shared = set(tables[0]).intersection(*tables[1:])
            setattr(combined, name,
                    {key: _largest([table[key] for table in tables]) for key in shared})
        combined.source = f"{ENVELOPE} of [" + "; ".join(part.source for part in parts) + "]"
        return combined

    @classmethod
    def load(cls, null_dir, positions, sigma=DEFAULT_SIGMA, kind="gaussian"):
        if kind == ENVELOPE:
            return cls.envelope([cls.load(null_dir, positions, part_sigma, part_kind)
                                 for part_sigma, part_kind in envelope_components()])
        null_dir = Path(null_dir)
        # Both VAA nulls sit in one shares file, written by every vaa_null_model_argmax
        # run; the uniform one is read from the default-sigma run's. The VAA files carry
        # the scoring they were computed under (analysis/vaa_scoring.py), the one this run
        # scores the models with.
        tag = scoring_suffix(vaa_scoring(positions))
        vaa_sigma = sigma if kind == "gaussian" else DEFAULT_SIGMA
        vaa_path = null_dir / f"null_model_argmax{output_suffix(positions, vaa_sigma)}{tag}_shares.csv"
        classifier_path = null_dir / f"{output_stem(positions, sigma, kind)}_shares.csv"
        agreement_path = null_dir / f"null_model_agreement{output_suffix(positions, sigma, kind)}{tag}.csv"
        missing = [path.name for path in (vaa_path, classifier_path, agreement_path)
                   if not path.exists()]
        if missing:
            raise SystemExit(
                f"No {kind} null model: {', '.join(missing)} not in {null_dir} -- run "
                f"vaa_null_model_argmax.py and classifier_null_model.py "
                f"(--positions {positions} --null {kind}"
                + (f" --sigma {sigma:g}" if kind == "gaussian" else "")
                + ") first, or draw the raw shares with --absolute.")

        vaa = pd.read_csv(vaa_path)
        vaa = vaa[(vaa["side"] == NULL_SIDE[kind]) & (vaa["model"] == ALL_MODELS)
                  & vaa["method"].isin(VAA_METHOD)]
        vaa = pd.DataFrame({
            "method": vaa["method"].map(VAA_METHOD),
            "variant": ANY_VARIANT,
            "scope": np.where(vaa["topic"] == WHOLE, ALL_SCOPE, "topic"),
            "slice": vaa["topic"],
            "ep_group": vaa["ep_group"],
            "null_share": vaa["argmax_share"],
            # The mean cell score, a mean cluster probability under the posterior scoring;
            # files written before it was kept have none.
            "null_mean_probability": vaa.get("mean_score", np.nan),
        })

        classifier = pd.read_csv(classifier_path)
        if not {"scope", "null_mean_probability"} <= set(classifier.columns):
            raise SystemExit(f"{classifier_path.name} predates the per-slice and "
                             "mean-probability nulls -- rerun classifier_null_model.py.")
        classifier = classifier[classifier["method"].isin(CLASSIFIER_METHOD)]
        classifier = classifier.assign(method=classifier["method"].map(CLASSIFIER_METHOD))
        table = pd.concat([vaa, classifier[["method", "variant", "scope", "slice", "ep_group",
                                            "null_share", "null_mean_probability"]]],
                          ignore_index=True)
        return cls(table, pd.read_csv(agreement_path),
                   f"{kind}: {vaa_path.name} + {classifier_path.name} + {agreement_path.name}")

    def share(self, method, variant, scope=ALL_SCOPE, slice_name=WHOLE):
        """The null share per group of one method, framing and slice."""
        if method in VAA_METHOD.values():
            if scope == "language":
                scope, slice_name = ALL_SCOPE, WHOLE
            key = (method, ANY_VARIANT, scope, slice_name)
        else:
            key = (method, variant, scope, slice_name)
        if key not in self._shares:
            raise SystemExit(f"No null for {key} in {self.source}.")
        return self._shares[key]

    def mean_probability(self, method, variant, scope=ALL_SCOPE, slice_name=WHOLE):
        """The null mean probability per group: the classifier's, or for a VAA method the
        mean cluster probability (posterior scoring), framing-free like its share."""
        if method in VAA_METHOD.values():
            if scope == "language":
                scope, slice_name = ALL_SCOPE, WHOLE
            key = (method, ANY_VARIANT, scope, slice_name)
        else:
            key = (method, variant, scope, slice_name)
        if key not in self._probabilities:
            raise SystemExit(f"No mean-probability null for {key} in {self.source}.")
        return self._probabilities[key]

    def agreement(self, source):
        """The null mean VAA agreement per group of one track, "likert" or "speeches".
        Framing-free, like the VAA argmax null."""
        if source not in self._agreement:
            raise SystemExit(f"No agreement null for {source!r} in {self.source}.")
        return self._agreement[source]


def vaa_statement_null(party_df, sigma=DEFAULT_SIGMA, kind="gaussian", scoring="distance"):
    """Per (ep_group, statement_idx): the score one random answer would get, in closed
    form -- sum over the five stance levels s of pi(s) times the score of s: the mean over
    the group's parties of 1 - |position - s| / 2 under "distance", the cluster's
    probability at s under "posterior". A single answer, not a paraphrase average: the
    per-statement scores rank_consistency_tables ranks come from one run each. Holds for
    every VAA track, since the null draws stances whatever the answers were scored by.
    kind "max" is the largest of its component nulls per (group, statement)."""
    if kind == ENVELOPE:
        parts = [vaa_statement_null(party_df, part_sigma, part_kind, scoring)
                 for part_sigma, part_kind in envelope_components()]
        return (pd.concat(parts).groupby(["ep_group", "statement_idx"], as_index=False)
                ["null_score"].max())
    pi = level_probabilities(kind, sigma)
    if scoring == "posterior":
        groups = sorted(party_df["ep_group"].dropna().unique())
        expected = np.einsum("l,lgt->gt", pi, level_posteriors(LEVELS, groups))
        return pd.DataFrame([{"ep_group": group, "statement_idx": t, "null_score": value}
                             for g, group in enumerate(groups)
                             for t, value in enumerate(expected[g])])
    frames = []
    for level, weight in zip(LEVELS, pi):
        agreement = 1 - (party_df["normalized_answer"] - level).abs() / 2
        frames.append(party_df.assign(null_score=weight * agreement)
                      .groupby(["ep_group", "statement_idx"])["null_score"].mean())
    return sum(frames).rename("null_score").reset_index()


def classifier_statement_null(null_dir, positions, sigma=DEFAULT_SIGMA, kind="gaussian"):
    """Per (method key, variant, language, statement_idx, ep_group): the classifier's
    mean-probability null for one text, from classifier_null_model's _statements.csv.
    kind "max" is the largest of its component nulls per cell."""
    if kind == ENVELOPE:
        parts = [classifier_statement_null(null_dir, positions, part_sigma, part_kind)
                 for part_sigma, part_kind in envelope_components()]
        keys = ["method", "variant", "language", "statement_idx", "ep_group"]
        return (pd.concat(parts).groupby(keys, as_index=False)["null_mean_probability"].max())
    path = Path(null_dir) / f"{output_stem(positions, sigma, kind)}_statements.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run classifier_null_model.py "
                         f"(--positions {positions} --null {kind}"
                         + (f" --sigma {sigma:g}" if kind == "gaussian" else "")
                         + ") first, or pass --absolute.")
    table = pd.read_csv(path)
    return table.assign(method=table["method"].map(CLASSIFIER_METHOD))


def minus_null(share, null):
    """Share minus null over the groups of either: a group the model never lands on but
    the null gives some share to comes out negative rather than missing."""
    groups = share.index.union(null.index)
    return share.reindex(groups, fill_value=0.0) - null.reindex(groups, fill_value=0.0)
