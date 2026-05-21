import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import flip_likert, likert_to_stance

NUM_VARIANTS = 8
NEUTRAL_LIKERT = 3
FRAMING_SUFFIX = {"base": "", "negated": "_negated", "question": "_question"}
FRAMING_FOR_VARIANT = {"": "base", "_negated": "negated", "_question": "question"}
FRAMING_COLOR = {"base": "#1f77b4", "negated": "#d62728", "question": "#2ca02c"}
RESPONSE_CSV = re.compile(r"^(?P<langs>[a-z]{2}(?:,[a-z]{2})+)(?P<variant>|_negated|_question)\.csv$")
SPEECH_CSV = re.compile(r"^speeches_(?P<langs>[a-z]{2}(?:,[a-z]{2})+)(?P<variant>|_negated|_question)_scored\.csv$")
RUN_KEYS = ["language", "variant_idx", "framing", "source"]

SOURCE_POINT_MARKER = {"likert": "o", "speeches": "^"}
SOURCE_MEAN_MARKER = {"likert": "*", "speeches": "X"}
SOURCE_LINESTYLE = {"likert": "-", "speeches": "--"}
SOURCE_VIOLIN_SIDE = {"likert": "low", "speeches": "high"}
SOURCE_VIOLIN_ALPHA = {"likert": 0.6, "speeches": 0.3}

DIMENSION_POLES = {
    "Ukraine": ("Less EU support", "More EU support"),
    "Ecology": ("Environmental protection", "Economic growth"),
    "Immigration": ("Lenient immigration policy", "Strict immigration policy"),
    "Values": ("Liberal", "Traditional"),
    "Economy": ("State intervention", "Free market"),
    "Europe": ("More national autonomy", "More European integration"),
    "Left-Right": ("Left - progressive", "Right - conservative"),
}


def load_questionnaire(dataset):
    path = Path("data") / f"{dataset}_data" / f"{dataset}_questionnaire.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).sort_values("statement_idx").reset_index(drop=True)
    dims = [c for c in df.columns if c not in ("statement_idx", "statement")]
    one_sided = {d for d in dims if len(set(np.sign(df[d])) - {0.0}) < 2}
    return df, dims, one_sided


def find_csvs(model_dir, pattern):
    found = {}
    for path in sorted(model_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        framing = FRAMING_FOR_VARIANT[match.group("variant")]
        if framing not in found or len(match.group("langs")) > len(found[framing].stem):
            found[framing] = path
    return found


def detect_languages(columns, prefix, suffix):
    pattern = re.compile(rf"^{prefix}([a-z]{{2}}){re.escape(suffix)}_v0$")
    return sorted(m.group(1) for col in columns if (m := pattern.match(col)))


def load_likert_stances(model_dir):
    frames = []
    for framing, path in find_csvs(model_dir, RESPONSE_CSV).items():
        suffix = FRAMING_SUFFIX[framing]
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        for lang in detect_languages(df.columns, "choice_", suffix):
            for v in range(NUM_VARIANTS):
                col = f"choice_{lang}{suffix}_v{v}"
                if col not in df.columns:
                    continue
                likert = pd.to_numeric(df[col], errors="coerce").fillna(NEUTRAL_LIKERT)
                if framing == "negated":
                    likert = flip_likert(likert)
                frames.append(pd.DataFrame({
                    "row_idx": np.arange(len(df)),
                    "language": lang,
                    "variant_idx": v,
                    "framing": framing,
                    "source": "likert",
                    "stance": likert_to_stance(likert).to_numpy(),
                }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_speech_stances(model_dir):
    frames = []
    for framing, path in find_csvs(model_dir, SPEECH_CSV).items():
        suffix = FRAMING_SUFFIX[framing]
        orientation = -1.0 if framing == "negated" else 1.0
        header = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns
        stance_cols = [c for c in header if re.match(rf"^stance_[a-z]{{2}}{re.escape(suffix)}_v\d+$", c)]
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=stance_cols)
        for lang in detect_languages(stance_cols, "stance_", suffix):
            v = 0
            while (col := f"stance_{lang}{suffix}_v{v}") in df.columns:
                frames.append(pd.DataFrame({
                    "row_idx": np.arange(len(df)),
                    "language": lang,
                    "variant_idx": v,
                    "framing": framing,
                    "source": "speeches",
                    "stance": orientation * pd.to_numeric(df[col], errors="coerce").to_numpy(),
                }))
                v += 1
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def project_onto_dimensions(stances, questionnaire, dims):
    signs = questionnaire[dims].reset_index(names="row_idx")
    merged = stances.merge(signs, on="row_idx", how="left")
    for d in dims:
        merged[d] = merged["stance"] * merged[d]
    return merged


def per_run_positions(projected, dims):
    return projected.groupby(RUN_KEYS)[dims].mean().reset_index()


def per_response_positions(projected, dims):
    long = projected.melt(id_vars=["framing", "source"], value_vars=dims,
                          var_name="dimension", value_name="position")
    return long[long["position"].notna()]


def covariance_ellipse(points, n_std, **kwargs):
    mean = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    angle = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues[::-1], 0))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def setup_compass(ax, x_dim, y_dim, one_sided):
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(axis_label(x_dim, one_sided), fontsize=9)
    ax.set_ylabel(axis_label(y_dim, one_sided), fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def axis_label(dim, one_sided):
    negative, positive = DIMENSION_POLES.get(dim, ("disagree", "agree"))
    flag = "  (one-sided!)" if dim in one_sided else ""
    return f"← {negative}      {dim}{flag}      {positive} →"


def plot_compass(runs, x_dim, y_dim, one_sided, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_compass(ax, x_dim, y_dim, one_sided)

    all_points = runs[[x_dim, y_dim]].dropna().to_numpy()
    if len(all_points) >= 5:
        density = gaussian_kde(all_points.T)
        gx, gy = np.mgrid[-1:1:120j, -1:1:120j]
        grid = density(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contourf(gx, gy, grid, levels=12, cmap="Greys", alpha=0.35)

    sources = [s for s in SOURCE_POINT_MARKER if s in set(runs["source"])]
    for framing, color in FRAMING_COLOR.items():
        for source in sources:
            points = runs[(runs["framing"] == framing) & (runs["source"] == source)][[x_dim, y_dim]].dropna().to_numpy()
            if len(points) < 3:
                continue
            ax.scatter(points[:, 0], points[:, 1], s=12, color=color, alpha=0.3,
                       marker=SOURCE_POINT_MARKER[source])
            ax.add_patch(covariance_ellipse(points, 2, edgecolor=color, facecolor="none",
                                             lw=1.4, ls=SOURCE_LINESTYLE[source]))
            mean = points.mean(axis=0)
            ax.scatter(*mean, s=170, color=color, edgecolor="black",
                       marker=SOURCE_MEAN_MARKER[source], zorder=4)

    ax.set_title(title, fontsize=11, pad=8)
    add_framing_source_legends(ax, set(runs["framing"]), sources)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def add_framing_source_legends(ax, framings, sources):
    framing_handles = [Line2D([], [], color=c, marker="s", ls="", label=f)
                       for f, c in FRAMING_COLOR.items() if f in framings]
    source_handles = [Line2D([], [], color="gray", marker=SOURCE_MEAN_MARKER[s],
                             ls=SOURCE_LINESTYLE[s], label=f"{s} (★/X = mean, 2σ ellipse)")
                      for s in sources]
    first = ax.legend(handles=framing_handles, loc="upper left", fontsize=8, title="framing")
    ax.add_artist(first)
    ax.legend(handles=source_handles, loc="upper right", fontsize=8, title="source")


def violin_series(responses, framing, source, dims, positions_for_dim):
    positions, datasets = [], []
    subset = responses[(responses["framing"] == framing) & (responses["source"] == source)]
    for pos, d in zip(positions_for_dim, dims):
        values = subset[subset["dimension"] == d]["position"].to_numpy()
        if len(values):
            positions.append(pos)
            datasets.append(values)
    return positions, datasets


def plot_violins(responses, dims, one_sided, title, out_path):
    framings = [f for f in FRAMING_COLOR if f in set(responses["framing"])]
    sources = [s for s in SOURCE_POINT_MARKER if s in set(responses["source"])]
    split = len(sources) > 1
    x = np.arange(len(dims))
    band_width = 0.8 / max(len(framings), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(dims) * 1.5), 6))
    ax.axhline(0, color="gray", lw=0.8)
    for i, framing in enumerate(framings):
        offset = -0.4 + band_width * (i + 0.5)
        for source in sources:
            positions, datasets = violin_series(responses, framing, source, dims, x + offset)
            if not datasets:
                continue
            parts = ax.violinplot(datasets, positions=positions, widths=band_width * 0.9,
                                  showmeans=True, showextrema=False,
                                  side=SOURCE_VIOLIN_SIDE[source] if split else "both")
            for body in parts["bodies"]:
                body.set_facecolor(FRAMING_COLOR[framing])
                body.set_alpha(SOURCE_VIOLIN_ALPHA[source])
            parts["cmeans"].set_color("black")

    ax.set_xticks(x)
    ax.set_xticklabels([d + ("\n(one-sided!)" if d in one_sided else "") for d in dims], fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("position  (← disagree   ·   agree →)")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title(title, fontsize=11, pad=8)

    framing_handles = [Line2D([], [], color=c, marker="s", ls="", label=f)
                       for f, c in FRAMING_COLOR.items() if f in set(framings)]
    source_note = "  ".join(f"{SOURCE_VIOLIN_SIDE[s]} half = {s}" for s in sources) if split else ""
    legend = ax.legend(handles=framing_handles, loc="upper right", fontsize=8,
                       title="framing (black bar = mean)")
    if source_note:
        ax.text(0.99, 0.02, source_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_models_compass(runs_by_model, x_dim, y_dim, one_sided, title, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    setup_compass(ax, x_dim, y_dim, one_sided)
    cmap = plt.get_cmap("tab10" if len(runs_by_model) <= 10 else "tab20")
    all_runs = pd.concat(runs_by_model.values(), ignore_index=True)
    sources = [s for s in SOURCE_POINT_MARKER if s in set(all_runs["source"])]
    model_handles = []
    for i, (model, runs) in enumerate(sorted(runs_by_model.items())):
        color = cmap(i % cmap.N)
        model_handles.append(Line2D([], [], color=color, marker="s", ls="", label=model))
        for source in sources:
            points = runs[runs["source"] == source][[x_dim, y_dim]].dropna().to_numpy()
            if len(points) < 3:
                continue
            ax.add_patch(covariance_ellipse(points, 2, edgecolor=color, facecolor="none",
                                             lw=1.3, ls=SOURCE_LINESTYLE[source]))
            ax.scatter(*points.mean(axis=0), s=110, color=color, edgecolor="black",
                       marker=SOURCE_MEAN_MARKER[source], zorder=4)
    ax.set_title(title, fontsize=11, pad=8)
    source_handles = [Line2D([], [], color="gray", marker=SOURCE_MEAN_MARKER[s],
                             ls=SOURCE_LINESTYLE[s], label=f"{s} (mean, 2σ)") for s in sources]
    first = ax.legend(handles=model_handles, loc="upper left", fontsize=8, title="model")
    ax.add_artist(first)
    ax.legend(handles=source_handles, loc="upper right", fontsize=8, title="source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def process_model(model_dir, questionnaire, dims, one_sided, x_dim, y_dim):
    print(f"\nProcessing: {model_dir.name}")
    stances = pd.concat(
        [load_likert_stances(model_dir), load_speech_stances(model_dir)],
        ignore_index=True,
    )
    if stances.empty:
        print("  No response CSVs found, skipping.")
        return None

    projected = project_onto_dimensions(stances, questionnaire, dims)
    runs = per_run_positions(projected, dims)
    responses = per_response_positions(projected, dims)

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_compass(runs, x_dim, y_dim, one_sided,
                 f"{model_dir.name} – political compass ({x_dim} × {y_dim}, per-run)",
                 out_dir / "political_compass.png")
    plot_violins(responses, dims, one_sided,
                 f"{model_dir.name} – per-dimension stance distribution (per-response)",
                 out_dir / "dimension_violins.png")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None)
    parser.add_argument("--x-dim", default="Left-Right")
    parser.add_argument("--y-dim", default="Europe")
    args = parser.parse_args()

    questionnaire, dims, one_sided = load_questionnaire(args.dataset)
    for d in (args.x_dim, args.y_dim):
        if d not in dims:
            raise SystemExit(f"Dimension {d!r} not in questionnaire. Available: {dims}")
    if one_sided:
        print(f"One-sided dimensions (agreement bias confounded): {sorted(one_sided)}")

    results_dir = Path("data") / f"{args.dataset}_results"
    model_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name != "plots")
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    if not model_dirs:
        raise SystemExit("No model directories found.")

    runs_by_model = {}
    for model_dir in model_dirs:
        runs = process_model(model_dir, questionnaire, dims, one_sided, args.x_dim, args.y_dim)
        if runs is not None:
            runs_by_model[model_dir.name] = runs

    if len(runs_by_model) > 1:
        out_dir = results_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        plot_models_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            f"Political compass across models ({args.x_dim} × {args.y_dim}, per-run)",
                            out_dir / "political_compass_models.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
