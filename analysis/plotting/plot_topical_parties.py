#!/usr/bin/env python3
"""Per-topic versions of the plot_classified_parties figures.

plot_classified_parties.py pools all 30 euandi statements into one party
distribution, which hides that a model can lean GUE/NGL on Economy and PPE on
Immigration. euandi_2024_questionnaire.jsonl codes every statement on seven
topical axes; a statement belongs to an axis whenever its value there is
non-zero. This slices both the classifier predictions and the recomputed euandi
(VAA) agreement by axis and re-emits the core figures per axis, plus an
axis x EP-group overview heatmap.

The axes overlap -- the same statement can load Economy, Ecology and Left-Right
at once -- so the per-axis figures do not partition the pooled ones.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import plotting  # noqa: F401  (package import for the module below)
from analysis.plotting import plot_classified_parties as pcp

from analysis.core.questionnaire import axis_directions_in_admin_order
from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS, POSITION_CHOICES, load_party_positions, positions_path,
)
from analysis.vaa_agreement_ci import (
    agreement_matrices, bootstrap_group_means, stance_frame_for,
)
from utils import AXES, configure_stdout

configure_stdout()

TOPICS_DIRNAME = "topics"
OVERVIEW_FILENAME = "topics_axis_party_heatmap.png"


@contextmanager
def figure_footnote(text):
    """Stamp every figure saved inside the block with `text`.

    The plotting helpers reused from plot_classified_parties call save_figure
    themselves and take no title argument, so the note is handed over through the
    module rather than threaded through a dozen signatures."""
    previous = pcp.FIGURE_FOOTNOTE
    pcp.FIGURE_FOOTNOTE = text
    try:
        yield
    finally:
        pcp.FIGURE_FOOTNOTE = previous


def statement_rows_per_axis(dataset, n_rows):
    """{axis: 0-based statement rows loading on it}, in administration order.

    That row position is what the classified CSVs are indexed by (load_long_predictions
    stores it as row_idx) and what evaluate_euandi calls statement_idx, so one mapping
    drives both the classifier and the VAA side."""
    questionnaire_path = f"data/{dataset}_data/{dataset}_questionnaire.jsonl"
    directions = axis_directions_in_admin_order(questionnaire_path, dataset, n_rows)
    return {
        axis: np.flatnonzero(directions[axis].to_numpy() != 0)
        for axis in AXES
    }


def filter_long_by_axis(long_df, rows):
    return long_df[long_df["row_idx"].isin(rows)]


def statement_count_note(axis, rows):
    return f"{axis}: {len(rows)} statement(s)"


def load_full_agreement_matrices(model_dirs, positions):
    """{(model, source, variant): (sums, counts)} over every statement, plus the
    statement index and EP groups the matrices are laid out on.

    agreement_matrices aggregates per (group, statement), so an axis is just a
    column slice of the full matrices -- the raw stance frames are read once
    instead of once per axis."""
    vaa_csvs_per_model = {model_dir: pcp.discover_vaa_csvs(model_dir) for model_dir in model_dirs}
    every_vaa_csv = [path for csvs in vaa_csvs_per_model.values() for path in csvs.values()]
    if not every_vaa_csv:
        return {}, np.array([]), []

    party_df = load_party_positions(positions_path(positions), positions)
    if pcp.vaa_groups_are_collapsed(every_vaa_csv):
        party_df["ep_group"] = party_df["ep_group"].replace({"ECR": "ECR+ID", "ID": "ECR+ID"})
    party_df = party_df.dropna(subset=["ep_group"])
    statement_index = np.sort(party_df["statement_idx"].unique())
    groups = sorted(party_df["ep_group"].unique())

    matrices_per_key = {}
    for model_dir, vaa_csvs in vaa_csvs_per_model.items():
        for (source, variant_label), path in vaa_csvs.items():
            stance_df, languages = stance_frame_for(
                str(model_dir.parent), model_dir.name,
                pcp.STANCE_SOURCE_FOR_CLASSIFIER_SOURCE[source],
                pcp.VARIANT_LABEL_TO_SUFFIX[variant_label],
                pcp.languages_in_vaa_filename(path),
            )
            if stance_df is None:
                continue
            matrices = agreement_matrices(
                stance_df, languages, party_df, groups, statement_index)
            if not matrices[1].any():
                continue
            matrices_per_key[(model_dir.name, source, variant_label)] = matrices
    return matrices_per_key, statement_index, groups


def axis_column_positions(statement_index, rows):
    """Where the axis' statements sit in the columns of the full matrices."""
    return np.flatnonzero(np.isin(statement_index, rows))


def axis_vaa_results(matrices_per_key, statement_index, groups, rows, bootstrap_draws, seed):
    """({key: observed mean agreement per group}, {key: bootstrap replicates}) for
    one axis. The draws are redrawn per axis from the same seed, so every model and
    scope is scored on shared draws within the axis and the replicates stay
    comparable across the cells a plotted scope pools over."""
    positions = axis_column_positions(statement_index, rows)
    if positions.size == 0 or not matrices_per_key:
        return {}, {}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, positions.size, size=(bootstrap_draws, positions.size))

    observed_per_key, replicates_per_key = {}, {}
    for key, (sums, counts) in matrices_per_key.items():
        axis_matrices = (sums[:, positions], counts[:, positions])
        if not axis_matrices[1].any():
            continue
        observed, replicates = bootstrap_group_means(axis_matrices, draws)
        observed_per_key[key] = pd.Series(observed, index=groups).dropna()
        replicates_per_key[key] = pd.DataFrame(replicates, index=groups)
    return observed_per_key, replicates_per_key


def axis_mean_probability_per_key(predictions_by_model, rows):
    """{(model, source, variant): mean probability per party}, restricted to one axis."""
    return {
        (model_dir.name, source, variant_label): pcp.mean_probability_per_party(filtered)
        for model_dir, long_by_source_variant in predictions_by_model.items()
        for (source, variant_label), long_df in long_by_source_variant.items()
        if not (filtered := filter_long_by_axis(long_df, rows)).empty
    }


def run_wide_percent_limits(predictions_by_model, axis_rows, matrices_per_key,
                            statement_index, groups, bootstrap_draws, seed):
    """(bar ylim, box ylim) covering every axis this run plots.

    The cross-model helpers already share one scale between the scopes of a single call,
    but each axis is a separate call, so the limits are worked out up front. The
    replicates are recomputed afterwards rather than kept: holding every axis' draws at
    once would run to hundreds of MB, and the same seed redraws the same numbers."""
    bar_limits, box_limits = [], []
    for rows in axis_rows.values():
        bar_limits.append(pcp.percent_bar_limits(
            axis_mean_probability_per_key(predictions_by_model, rows)))
        _, replicates = axis_vaa_results(
            matrices_per_key, statement_index, groups, rows, bootstrap_draws, seed)
        box_limits.append(pcp.percent_box_limits(replicates))
    return pcp.widest_limits(bar_limits), pcp.widest_limits(box_limits)


def format_limits(limits):
    return "n/a" if limits is None else f"{limits[0]:.1f}% .. {limits[1]:.1f}%"


def plot_axis_party_heatmap(mean_probability_per_axis, statement_counts, output_path):
    """Axes x EP groups of mean class probability: all seven topics in one panel."""
    axes_present = [axis for axis in AXES if axis in mean_probability_per_axis]
    if not axes_present:
        return
    parties = pcp.ordered_parties_present(
        set().union(*(mean_probability_per_axis[axis].index for axis in axes_present)))
    if not parties:
        return

    matrix = np.array([
        mean_probability_per_axis[axis].reindex(parties).to_numpy(dtype=float)
        for axis in axes_present
    ])
    row_labels = [f"{axis} ({statement_counts[axis]})" for axis in axes_present]

    fig, ax = plt.subplots(figsize=(max(8, len(parties) * 1.1),
                                    max(4, len(axes_present) * 0.6)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis",
                      vmin=0.0, vmax=max(0.4, np.nanmax(matrix)))
    fig.colorbar(image, ax=ax, label=pcp.MEAN_PROBABILITY_LABEL, fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(parties)), parties, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(axes_present)), row_labels, fontsize=10)
    for row_index in range(len(axes_present)):
        for column_index in range(len(parties)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if value > 0.45 else "black")
    pcp.save_figure(fig, output_path)


def axis_plots_dir(model_dir, axis):
    return pcp.plots_dir_for(model_dir) / TOPICS_DIRNAME / axis


def plot_axis_for_model(model_dir, axis, long_by_source_variant, observed_vaa_per_key):
    """The core plot_classified_parties figures, restricted to one axis."""
    plots_dir = axis_plots_dir(model_dir, axis)
    comparison_rows = []
    mean_probability_by_source = {}

    for source in pcp.SOURCE_LABELS:
        mean_probability_by_variant = {}
        for (csv_source, variant_label), long_df in sorted(long_by_source_variant.items()):
            if csv_source != source:
                continue
            mean_probability = pcp.mean_probability_per_party(long_df)
            argmax_share = pcp.predicted_party_share(long_df)
            mean_probability_by_variant[variant_label] = mean_probability

            pcp.plot_party_distribution_bar(
                mean_probability, argmax_share,
                plots_dir / f"classified_{source}_{variant_label}_party_distribution.png",
            )
            comparison_row = axis_vaa_comparison(
                model_dir.name, axis, source, variant_label,
                mean_probability, argmax_share,
                observed_vaa_per_key.get((model_dir.name, source, variant_label)),
                plots_dir,
            )
            if comparison_row is not None:
                comparison_rows.append(comparison_row)

        if len(mean_probability_by_variant) > 1:
            pcp.plot_party_distribution_across_variants(
                mean_probability_by_variant,
                plots_dir / f"classified_{source}_party_distribution_variants.png",
            )
        frames = [long_df for (csv_source, _), long_df in long_by_source_variant.items()
                  if csv_source == source]
        if frames:
            mean_probability_by_source[source] = pcp.mean_probability_per_party(
                pd.concat(frames, ignore_index=True))

    if len(mean_probability_by_source) >= 2:
        pcp.plot_speeches_vs_reasons(
            mean_probability_by_source,
            plots_dir / "classified_speeches_vs_reasons_party_distribution.png",
        )
    return comparison_rows


def axis_vaa_comparison(model_name, axis, source, variant_label,
                        mean_probability, argmax_share, vaa_mean_agreement, plots_dir):
    if vaa_mean_agreement is None or vaa_mean_agreement.empty:
        return None
    comparison_row = pcp.build_vaa_comparison_row(
        model_name, source, variant_label, mean_probability, argmax_share, vaa_mean_agreement)
    pcp.plot_classifier_vs_vaa_scatter(
        mean_probability, vaa_mean_agreement,
        comparison_row["spearman_rho_mean_prob_vs_vaa"],
        plots_dir / f"classified_vs_vaa_{source}_{variant_label}_scatter.png",
    )
    return {"axis": axis, **comparison_row}


def pooled_mean_probability(long_by_source_variant):
    frames = [long_df for long_df in long_by_source_variant.values() if not long_df.empty]
    if not frames:
        return pd.Series(dtype=float)
    return pcp.mean_probability_per_party(pd.concat(frames, ignore_index=True))


def write_summary_csv(rows, output_path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Wrote {output_path}")


def rows_in_use(axis_rows, selected_axes):
    return {axis: axis_rows[axis] for axis in selected_axes}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None)
    parser.add_argument("--axes", default=",".join(AXES),
                        help="Comma-separated topical axes to plot.")
    parser.add_argument("--positions", default=DEFAULT_POSITIONS, choices=POSITION_CHOICES,
                        help="Which euandi answers stand for an EP group when the per-topic "
                             "VAA agreement is recomputed; must match the basis "
                             "evaluate_euandi.py wrote the vaa*.csv files with.")
    parser.add_argument("--bootstrap", default=10000, type=int,
                        help="Bootstrap draws behind the per-topic VAA agreement boxes.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    args.axes = [axis.strip() for axis in args.axes.split(",") if axis.strip()]
    unknown = [axis for axis in args.axes if axis not in AXES]
    if unknown:
        raise SystemExit(f"Unknown axis/axes {unknown}; available: {AXES}")
    return args


def main():
    args = parse_args()
    results_dir = Path("data") / f"{args.dataset}_results"
    if not results_dir.exists():
        raise SystemExit(f"Directory not found: {results_dir}")

    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    predictions_by_model = {}
    for model_dir in model_dirs:
        long_by_source_variant = pcp.load_model_predictions(model_dir)
        if long_by_source_variant:
            predictions_by_model[model_dir] = long_by_source_variant
    if not predictions_by_model:
        raise SystemExit("No classified predictions found.")

    n_rows = 1 + max(
        int(long_df["row_idx"].max())
        for long_by_source_variant in predictions_by_model.values()
        for long_df in long_by_source_variant.values()
    )
    axis_rows = rows_in_use(statement_rows_per_axis(args.dataset, n_rows), args.axes)

    print(f"\n{n_rows} statements, {len(axis_rows)} topical axes:")
    for axis, rows in axis_rows.items():
        print(f"  {axis}: {len(rows)} statement(s) -> rows {rows.tolist()}")
    print("Axes overlap: a statement can load several of them, so the per-topic "
          "figures do not partition the pooled ones.")

    print("\nRecomputing euandi agreement per statement (once per model/scope):")
    matrices_per_key, statement_index, groups = load_full_agreement_matrices(
        model_dirs, args.positions)
    if not matrices_per_key:
        print("  no usable VAA inputs; the VAA half of the topic figures is skipped.")

    bar_ylim, box_ylim = run_wide_percent_limits(
        predictions_by_model, axis_rows, matrices_per_key, statement_index, groups,
        args.bootstrap, args.seed)
    print(f"  Shared y-axis: bars {format_limits(bar_ylim)}, "
          f"agreement boxes {format_limits(box_ylim)}.")

    dataset_plots_dir = results_dir / "plots"
    mean_probability_per_model_axis = {}
    summary_rows = []

    for axis, rows in axis_rows.items():
        print(f"\n=== {axis} ({len(rows)} statements) ===")
        observed_vaa, replicates_vaa = axis_vaa_results(
            matrices_per_key, statement_index, groups, rows, args.bootstrap, args.seed)

        axis_long_by_model = {
            model_dir: {
                key: filtered
                for key, long_df in long_by_source_variant.items()
                if not (filtered := filter_long_by_axis(long_df, rows)).empty
            }
            for model_dir, long_by_source_variant in predictions_by_model.items()
        }
        axis_long_by_model = {m: v for m, v in axis_long_by_model.items() if v}

        with figure_footnote(f"{axis}: {len(rows)} of {n_rows} statements"):
            axis_summary_rows = []
            for model_dir, axis_long in axis_long_by_model.items():
                axis_summary_rows.extend(
                    plot_axis_for_model(model_dir, axis, axis_long, observed_vaa))
                mean_probability_per_model_axis[(model_dir.name, axis)] = (
                    pooled_mean_probability(axis_long))
            summary_rows.extend(axis_summary_rows)

            mean_probability_per_model_source_variant = {
                (model_dir.name, source, variant_label): pcp.mean_probability_per_party(long_df)
                for model_dir, axis_long in axis_long_by_model.items()
                for (source, variant_label), long_df in axis_long.items()
            }
            axis_dataset_dir = dataset_plots_dir / TOPICS_DIRNAME / axis
            if len({model for model, _, _ in mean_probability_per_model_source_variant}) > 1:
                pcp.plot_cross_model_party_bars(
                    mean_probability_per_model_source_variant, pcp.MEAN_PROBABILITY_LABEL,
                    axis_dataset_dir, "classified_all_models_party_distribution", bar_ylim,
                )
            if len({model for model, _, _ in replicates_vaa}) > 1:
                pcp.plot_cross_model_vaa_boxes(
                    replicates_vaa, pcp.MEAN_AGREEMENT_LABEL,
                    axis_dataset_dir, "vaa_all_models_mean_agreement", box_ylim,
                )

    statement_counts = {axis: len(rows) for axis, rows in axis_rows.items()}
    print("\nAxis x party overviews:")
    for model_dir in predictions_by_model:
        per_axis = {
            axis: mean_probability_per_model_axis[(model_dir.name, axis)]
            for axis in axis_rows
            if (model_dir.name, axis) in mean_probability_per_model_axis
        }
        plot_axis_party_heatmap(
            per_axis, statement_counts, pcp.plots_dir_for(model_dir) / OVERVIEW_FILENAME)

    across_models = {}
    for (model_name, axis), series in mean_probability_per_model_axis.items():
        across_models.setdefault(axis, []).append(series)
    plot_axis_party_heatmap(
        {axis: pd.concat(series_list).groupby(level=0).mean()
         for axis, series_list in across_models.items()},
        statement_counts, dataset_plots_dir / OVERVIEW_FILENAME,
    )

    for model_dir in predictions_by_model:
        model_rows = [row for row in summary_rows if row["model"] == model_dir.name]
        write_summary_csv(
            model_rows,
            pcp.plots_dir_for(model_dir) / TOPICS_DIRNAME / "classified_vs_vaa_summary.csv")
    if len(predictions_by_model) > 1:
        write_summary_csv(
            summary_rows,
            dataset_plots_dir / TOPICS_DIRNAME / "classified_vs_vaa_models_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
