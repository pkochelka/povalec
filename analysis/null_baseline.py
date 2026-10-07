"""The 0-centered normal null of every argmax method, read back as one lookup.

vaa_null_model_argmax.py writes the null of the two VAA methods (direct and indirect
Likert), classifier_null_model.py that of the two classifier methods (classified reasons,
classified open-ended). Both are argmax shares on the scale of plot_argmax_shares'
shares, so a share minus its null is a difference in percentage points: what a group
wins beyond what random stances around the neutral answer would already give it.

This is the one place their files are read, so the figures and the table subtract the
same numbers:

  NullBaseline.load(null_dir, positions, sigma).share(method, variant, scope, slice)

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
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.classifier_null_model import (
    ALL_SCOPE, LEVELS, null_level_probabilities, output_stem,
)
from analysis.vaa_null_model import ALL_MODELS, DEFAULT_SIGMA
from analysis.vaa_null_model_argmax import WHOLE, output_suffix

NULL_SIDE = "normal"
VAA_METHOD = {"likert": "likert_vaa", "speeches": "speeches_vaa"}
CLASSIFIER_METHOD = {"classified reasons": "reasons_clf",
                     "classified open-ended": "speeches_clf"}
ANY_VARIANT = "any"   # the VAA null is framing-free


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
    def load(cls, null_dir, positions, sigma=DEFAULT_SIGMA):
        null_dir = Path(null_dir)
        vaa_path = null_dir / f"null_model_argmax{output_suffix(positions, sigma)}_shares.csv"
        classifier_path = null_dir / f"{output_stem(positions, sigma)}_shares.csv"
        agreement_path = null_dir / f"null_model_agreement{output_suffix(positions, sigma)}.csv"
        missing = [path.name for path in (vaa_path, classifier_path, agreement_path)
                   if not path.exists()]
        if missing:
            raise SystemExit(
                f"No null model: {', '.join(missing)} not in {null_dir} -- run "
                f"vaa_null_model_argmax.py and classifier_null_model.py "
                f"(--positions {positions} --sigma {sigma:g}) first, or draw the raw "
                f"shares with --absolute.")

        vaa = pd.read_csv(vaa_path)
        vaa = vaa[(vaa["side"] == NULL_SIDE) & (vaa["model"] == ALL_MODELS)
                  & vaa["method"].isin(VAA_METHOD)]
        vaa = pd.DataFrame({
            "method": vaa["method"].map(VAA_METHOD),
            "variant": ANY_VARIANT,
            "scope": np.where(vaa["topic"] == WHOLE, ALL_SCOPE, "topic"),
            "slice": vaa["topic"],
            "ep_group": vaa["ep_group"],
            "null_share": vaa["argmax_share"],
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
                   f"{vaa_path.name} + {classifier_path.name} + {agreement_path.name}")

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
        """The null mean classifier probability per group (classifier methods only)."""
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


def vaa_statement_null(party_df, sigma=DEFAULT_SIGMA):
    """Per (ep_group, statement_idx): the agreement one random answer would get, in closed
    form -- sum over the five stance levels s of pi(s) times the mean over the group's
    parties of 1 - |position - s| / 2. A single answer, not a paraphrase average: the
    per-statement scores rank_consistency_tables ranks come from one run each. Holds for
    every VAA track, since the null draws stances whatever the answers were scored by."""
    pi = null_level_probabilities(sigma)
    frames = []
    for level, weight in zip(LEVELS, pi):
        agreement = 1 - (party_df["normalized_answer"] - level).abs() / 2
        frames.append(party_df.assign(null_score=weight * agreement)
                      .groupby(["ep_group", "statement_idx"])["null_score"].mean())
    return sum(frames).rename("null_score").reset_index()


def classifier_statement_null(null_dir, positions, sigma=DEFAULT_SIGMA):
    """Per (method key, variant, language, statement_idx, ep_group): the classifier's
    mean-probability null for one text, from classifier_null_model's _statements.csv."""
    path = Path(null_dir) / f"{output_stem(positions, sigma)}_statements.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run classifier_null_model.py "
                         f"(--positions {positions} --sigma {sigma:g}) first, or pass "
                         "--absolute.")
    table = pd.read_csv(path)
    return table.assign(method=table["method"].map(CLASSIFIER_METHOD))


def minus_null(share, null):
    """Share minus null over the groups of either: a group the model never lands on but
    the null gives some share to comes out negative rather than missing."""
    groups = share.index.union(null.index)
    return share.reindex(groups, fill_value=0.0) - null.reindex(groups, fill_value=0.0)
