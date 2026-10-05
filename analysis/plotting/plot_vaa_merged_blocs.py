"""Cross-model VAA agreement box plots against the four merged blocs.

The same figure as the "VAA agreement" panel of plot_classified_parties.py (one box per
bloc x model over a statement bootstrap, box = IQR, whiskers = 1.5 x IQR, line = median),
but scored against the blocs of analysis/vaa_merged_groups.py instead of the six groups:

    GUE/NGL, S&D+Greens/EFA, ALDE+PPE, ECR+ID

Each bloc's position is the mean of all its national member parties, weighted by the
seats each party's list won in 2024 (`--weights seats`, the default) or equally
(`--weights equal`). All 30 statements are used; a party that gave no answer on a
statement drops out of that statement's mean.

Figures go to data/{dataset}_results/plots/vaa_blocs/, one per scope (pooled, per
framing, per source, per source x framing), all on a shared y-axis. The top bloc per
model and scope is printed and written next to them as a CSV.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.core.labels import use_short_party_labels
use_short_party_labels()   # cluster nicknames in every label (utils.PARTY_SHORT)
import numpy as np
import pandas as pd

from analysis.core import model_display_name
from analysis.plotting.plot_classified_parties import (
    LEGEND_BELOW, LEGEND_FONTSIZE, LEGEND_TITLE_FONTSIZE, MEAN_AGREEMENT_LABEL,
    MODEL_LEGEND_NCOL, STANCE_SOURCE_FOR_CLASSIFIER_SOURCE, VARIANT_LABEL_TO_SUFFIX,
    cross_model_scopes, draw_models_party_boxes, model_legend_handles,
    percent_box_limits, save_figure, scoped_replicates_per_model, type_scale,
)
from analysis.vaa_agreement_ci import agreement_matrices, bootstrap_group_means, stance_frame_for
from analysis.vaa_merged_groups import BLOCS, NATIONAL_WEIGHTS, bloc_positions, national_positions
from utils import ALL_LANGS_STR, configure_stdout

configure_stdout()

VARIANT_LABELS = ["base", "negated"]
SOURCES = ["reasons", "speeches"]


def load_bloc_replicates(model_dirs, bloc_df, languages, bootstrap_draws, seed):
    """{(model, source, variant): mean agreement per bloc per bootstrap draw}, every cell
    on the same statement draws so pooled scopes can be averaged draw by draw."""
    statement_index = np.sort(bloc_df["statement_idx"].unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(statement_index), size=(bootstrap_draws, len(statement_index)))

    replicates, observed = {}, {}
    for model_dir in model_dirs:
        for source in SOURCES:
            for variant_label in VARIANT_LABELS:
                stance_df, langs = stance_frame_for(
                    str(model_dir.parent), model_dir.name,
                    STANCE_SOURCE_FOR_CLASSIFIER_SOURCE[source],
                    VARIANT_LABEL_TO_SUFFIX[variant_label], languages)
                if stance_df is None:
                    continue
                matrices = agreement_matrices(stance_df, langs, bloc_df, BLOCS, statement_index)
                if not matrices[1].any():
                    continue
                point, draws_per_bloc = bootstrap_group_means(matrices, draws)
                key = (model_dir.name, source, variant_label)
                replicates[key] = pd.DataFrame(draws_per_bloc, index=BLOCS)
                observed[key] = pd.Series(point, index=BLOCS)
    return replicates, observed


def plot_bloc_boxes(replicates_per_model, output_path, ylim, title):
    """plot_models_party_boxes with the bloc order fixed: the shared PARTY_DISPLAY_ORDER
    knows GUE/NGL and ECR+ID but not the two merged centre blocs, and would sort them
    to the end."""
    models = sorted(replicates_per_model)
    model_cmap = plt.get_cmap("tab20", max(len(models), 2))
    figure_width = max(13.0, len(BLOCS) * len(models) * 0.34)
    scale = type_scale(figure_width)
    fig, ax = plt.subplots(figsize=(figure_width, 7.5 * scale))
    draw_models_party_boxes(ax, replicates_per_model, BLOCS, models, model_cmap,
                            MEAN_AGREEMENT_LABEL, ylim, scale=scale)
    ax.set_title(title, fontsize=LEGEND_TITLE_FONTSIZE * scale)
    ax.legend(
        handles=model_legend_handles(models, model_cmap, model_display_name),
        **{**LEGEND_BELOW,
           "fontsize": LEGEND_FONTSIZE * scale,
           "title_fontsize": LEGEND_TITLE_FONTSIZE * scale},
        bbox_to_anchor=(0.5, -0.34), ncol=min(len(models), MODEL_LEGEND_NCOL), title="model",
    )
    save_figure(fig, output_path, bbox_inches="tight")


def top_blocs(observed):
    rows = []
    for (model, source, variant), series in sorted(observed.items()):
        ranked = series.sort_values(ascending=False)
        rows.append({"model": model, "source": source, "variant": variant,
                     "top": ranked.index[0], "agreement": round(ranked.iloc[0], 4),
                     "second": ranked.index[1],
                     "margin": round(ranked.iloc[0] - ranked.iloc[1], 4)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--weights", default="seats", choices=sorted(NATIONAL_WEIGHTS))
    parser.add_argument("--bootstrap", default=10000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    results_dir = Path("data") / f"{args.dataset}_results"
    model_dirs = sorted(p for p in results_dir.iterdir()
                        if p.is_dir() and p.name not in {"plots", "tables"})
    bloc_df = bloc_positions(national_positions(), NATIONAL_WEIGHTS[args.weights], "short_name")

    replicates, observed = load_bloc_replicates(
        model_dirs, bloc_df, args.languages, args.bootstrap, args.seed)
    if len({model for model, _, _ in replicates}) < 2:
        raise SystemExit("Need at least two models with VAA results.")

    out_dir = results_dir / "plots" / "vaa_blocs"
    stem = f"vaa_blocs_{args.weights}_all_models_mean_agreement"
    ylim = percent_box_limits(replicates)
    weighting = "2024 seat-weighted" if args.weights == "seats" else "equal-weighted"
    for source, variant_label, suffix, scope_label in cross_model_scopes(replicates):
        scoped = scoped_replicates_per_model(replicates, source, variant_label)
        if len(scoped) < 2:
            continue
        plot_bloc_boxes(scoped, out_dir / f"{stem}{suffix}.png", ylim,
                        f"National parties, {weighting} blocs -- {scope_label}")

    tops = top_blocs(observed)
    tops_path = out_dir / f"vaa_blocs_{args.weights}_top_match.csv"
    tops.to_csv(tops_path, index=False)
    print(tops.pivot(index="model", columns=["source", "variant"], values="top").to_string())
    print(f"\nTop-match counts:\n{tops['top'].value_counts().to_string()}")
    print(f"  Wrote {tops_path}")


if __name__ == "__main__":
    main()
