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
FRAMING_SUFFIX = {"base": "", "negated": "_negated"}
FRAMING_FOR_VARIANT = {"": "base", "_negated": "negated"}
FRAMING_ORIENTATION = {"base": 1.0, "negated": -1.0}
FRAMING_COLOR = {"base": "#2ca02c", "negated": "#d62728"}
FRAMING_MARKER = {"base": "o", "negated": "X"}
SOURCE_FILLED = {"likert": True, "speeches": False}
SOURCE_LABEL = {"likert": "likert", "speeches": "open-ended"}
LLM_SOURCE_LABEL = {"likert": "reasons (judged)", "speeches": "open-ended (judged)"}
RESPONSE_CSV = re.compile(r"^(?P<langs>[a-z]{2}(?:,[a-z]{2})+)(?P<variant>|_negated)\.csv$")
SPEECH_CSV = re.compile(r"^speeches_(?P<langs>[a-z]{2}(?:,[a-z]{2})+)(?P<variant>|_negated)_scored\.csv$")
RUN_KEYS = ["language", "variant_idx", "framing", "source"]

# --stance-source llm: take both tracks from the LLM stance judge instead of the
# model's own Likert choice (likert) and the NLI pool (speeches).
LLM_STANCE_CSV = {"likert": "reasons_llm_stance.csv", "speeches": "speeches_llm_stance.csv"}
LLM_STANCE_COLUMNS = ["paraphrase", "language", "variant", "statement", "llm_stance"]
STANCE_SOURCE_SUFFIX = {"nli": "", "llm": "_llmjudge"}
STANCE_SOURCE_TAG = {"nli": "", "llm": ", LLM-judged stance"}

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


def language_colors(languages):
    palette = []
    for name in ("tab20", "tab20b", "tab20c"):
        palette.extend(plt.get_cmap(name).colors)
    return {lang: palette[i % len(palette)] for i, lang in enumerate(sorted(languages))}


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
        orientation = FRAMING_ORIENTATION[framing]
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


def load_llm_judge_stances(model_dir):
    """Both tracks straight from judge_all_speeches.py, in place of the Likert
    choice columns and the NLI-scored speeches. Each text was judged against the
    statement as it was shown to the model, so the negated framing is flipped back
    onto the base orientation exactly like the NLI speech stances are."""
    frames = []
    for source, filename in LLM_STANCE_CSV.items():
        path = model_dir / filename
        if not path.exists():
            print(f"  no {filename}, skipping the {SOURCE_LABEL[source]} track.")
            continue
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=LLM_STANCE_COLUMNS)
        df = df[df["paraphrase"].isin(FRAMING_ORIENTATION)]
        orientation = df["paraphrase"].map(FRAMING_ORIENTATION)
        frames.append(pd.DataFrame({
            "row_idx": df["statement"].to_numpy(),
            "language": df["language"].to_numpy(),
            "variant_idx": df["variant"].to_numpy(),
            "framing": df["paraphrase"].to_numpy(),
            "source": source,
            "stance": (orientation * pd.to_numeric(df["llm_stance"], errors="coerce")).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_stances(model_dir, stance_source):
    if stance_source == "llm":
        return load_llm_judge_stances(model_dir)
    return pd.concat(
        [load_likert_stances(model_dir), load_speech_stances(model_dir)],
        ignore_index=True,
    )


def project_onto_dimensions(stances, questionnaire, dims):
    signs = questionnaire[dims].reset_index(names="row_idx")
    merged = stances.merge(signs, on="row_idx", how="left")
    for d in dims:
        merged[d] = merged["stance"] * merged[d].replace(0, np.nan)
    return merged


def aggregate_positions(projected, dims, granularity):
    if granularity == "none":
        return projected[["framing", "source", *dims]].copy()
    return projected.groupby(RUN_KEYS)[dims].mean().reset_index()


def melt_positions(positions, dims):
    long = positions.melt(id_vars=["framing", "source"], value_vars=dims,
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


NEUTRAL_COLOR = "#1f77b4"


def plot_compass(runs, x_dim, y_dim, one_sided, title, out_path, color_map=None, aggregate=False):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_compass(ax, x_dim, y_dim, one_sided)

    all_points = runs[[x_dim, y_dim]].dropna().to_numpy()
    if len(all_points) >= 5:
        try:
            density = gaussian_kde(all_points.T)
            gx, gy = np.mgrid[-1:1:120j, -1:1:120j]
            grid = density(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            ax.contourf(gx, gy, grid, levels=12, cmap="Greys", alpha=0.25)
            ax.contour(gx, gy, grid, levels=6, colors="gray", linewidths=0.5, alpha=0.5)
        except np.linalg.LinAlgError:
            pass

    sources = [s for s in SOURCE_FILLED if s in set(runs["source"])]
    framings = [f for f in FRAMING_MARKER if f in set(runs["framing"])]

    def draw_mean(points, marker, color):
        face = color if SOURCE_FILLED[source] else "none"
        ax.scatter(*points.mean(axis=0), s=260, marker=marker, facecolor=face,
                   edgecolor="black" if SOURCE_FILLED[source] else color,
                   linewidths=1.8, zorder=5)

    for framing in framings:
        marker = FRAMING_MARKER[framing]
        for source in sources:
            if aggregate:
                framing_color = FRAMING_COLOR[framing]
                face = framing_color if SOURCE_FILLED[source] else "none"
                points = runs[(runs["framing"] == framing) &
                              (runs["source"] == source)][[x_dim, y_dim]].dropna().to_numpy()
                if len(points):
                    ax.scatter(points[:, 0], points[:, 1], s=10, marker=marker,
                               facecolor=face, edgecolor=framing_color, alpha=0.6, linewidths=0.6)
                    draw_mean(points, marker, framing_color)
                continue
            for lang in sorted(runs["language"].dropna().unique()):
                color = color_map[lang] if color_map else NEUTRAL_COLOR
                face = color if SOURCE_FILLED[source] else "none"
                points = runs[(runs["language"] == lang) &
                              (runs["framing"] == framing) &
                              (runs["source"] == source)][[x_dim, y_dim]].dropna().to_numpy()
                if not len(points):
                    continue
                ax.scatter(points[:, 0], points[:, 1], s=18, marker=marker,
                           facecolor=face, edgecolor=color, alpha=0.45, linewidths=0.8)
                draw_mean(points, marker, color)

    ax.set_title(title, fontsize=11, pad=8)
    show_languages = color_map if color_map and len(color_map) > 1 else None
    add_language_framing_legends(ax, show_languages, framings, sources, aggregate=aggregate)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def add_language_framing_legends(ax, colors, framings, sources, aggregate=False):
    if colors:
        lang_handles = [Line2D([], [], color=c, marker="o", ls="", label=lang)
                        for lang, c in colors.items()]
        ax.legend(handles=lang_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=7, title="language")
        return
    framing_color = (lambda f: FRAMING_COLOR[f]) if aggregate else (lambda f: "gray")
    style_handles = [Line2D([], [], color=framing_color(f), marker=FRAMING_MARKER[f], ls="",
                            label=f"{f}  (small = run, large = mean)") for f in framings]
    style_handles += [Line2D([], [], color="gray", marker="o", ls="",
                             markerfacecolor="gray" if SOURCE_FILLED[s] else "none",
                             label=f"{SOURCE_LABEL[s]}  ({'filled' if SOURCE_FILLED[s] else 'hollow'})")
                      for s in sources]
    ax.legend(handles=style_handles, loc="upper right", fontsize=8, title="framing / source")


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
    source_note = "  ".join(f"{SOURCE_VIOLIN_SIDE[s]} half = {SOURCE_LABEL[s]}" for s in sources) if split else ""
    legend = ax.legend(handles=framing_handles, loc="upper right", fontsize=8,
                       title="framing (black bar = mean)")
    if source_note:
        ax.text(0.99, 0.02, source_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_models_scatter_compass(runs_by_model, x_dim, y_dim, one_sided, title, out_path, split_by=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_compass(ax, x_dim, y_dim, one_sided)
    cmap = plt.get_cmap("tab10" if len(runs_by_model) <= 10 else "tab20")
    all_runs = pd.concat(runs_by_model.values(), ignore_index=True)
    framings = [f for f in FRAMING_MARKER if f in set(all_runs["framing"])]
    sources = [s for s in SOURCE_FILLED if s in set(all_runs["source"])]

    def draw(points, color, marker, filled):
        if not len(points):
            return
        face = color if filled else "none"
        ax.scatter(points[:, 0], points[:, 1], s=12, marker=marker,
                   facecolor=face, edgecolor=color, alpha=0.8, linewidths=0.5)
        ax.scatter(*points.mean(axis=0), s=240, marker=marker, facecolor=face,
                   edgecolor="black" if filled else color, linewidths=1.8, zorder=5)

    model_handles = []
    for i, (model, runs) in enumerate(sorted(runs_by_model.items())):
        color = cmap(i % cmap.N)
        model_handles.append(Line2D([], [], color=color, marker="o", ls="", label=model))
        if split_by == "framing":
            for framing in framings:
                points = runs[runs["framing"] == framing][[x_dim, y_dim]].dropna().to_numpy()
                draw(points, color, FRAMING_MARKER[framing], True)
        elif split_by == "source":
            for source in sources:
                points = runs[runs["source"] == source][[x_dim, y_dim]].dropna().to_numpy()
                draw(points, color, "o", SOURCE_FILLED[source])
        else:
            draw(runs[[x_dim, y_dim]].dropna().to_numpy(), color, "o", True)

    ax.set_title(title, fontsize=11, pad=8)
    model_legend = ax.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                             fontsize=8, title="model  (large = mean)")
    ax.add_artist(model_legend)
    if split_by == "framing":
        style_handles = [Line2D([], [], color="gray", marker=FRAMING_MARKER[f], ls="", label=f)
                         for f in framings]
        ax.legend(handles=style_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  fontsize=8, title="framing")
    elif split_by == "source":
        style_handles = [Line2D([], [], color="gray", marker="o", ls="",
                                 markerfacecolor="gray" if SOURCE_FILLED[s] else "none",
                                 label=f"{SOURCE_LABEL[s]}  ({'filled' if SOURCE_FILLED[s] else 'hollow'})")
                         for s in sources]
        ax.legend(handles=style_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  fontsize=8, title="source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
                             ls=SOURCE_LINESTYLE[s], label=f"{SOURCE_LABEL[s]} (mean, 2σ)") for s in sources]
    first = ax.legend(handles=model_handles, loc="upper left", fontsize=8, title="model")
    ax.add_artist(first)
    ax.legend(handles=source_handles, loc="upper right", fontsize=8, title="source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


VIOLIN_GRANULARITIES = {"run": "per-run", "none": "per-answer"}


def process_model(model_dir, questionnaire, dims, one_sided, x_dim, y_dim, stance_source):
    print(f"\nProcessing: {model_dir.name}")
    stances = load_stances(model_dir, stance_source)
    if stances.empty:
        print("  No response CSVs found, skipping.")
        return None
    suffix = STANCE_SOURCE_SUFFIX[stance_source]
    tag = STANCE_SOURCE_TAG[stance_source]

    projected = project_onto_dimensions(stances, questionnaire, dims)
    run_positions = aggregate_positions(projected, dims, "run")

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    compass_dir = out_dir / "political_compasses"
    compass_dir.mkdir(exist_ok=True)
    shared_compass_dir = model_dir.parent / "plots" / "political_compasses"
    shared_compass_dir.mkdir(parents=True, exist_ok=True)

    base_title = f"{model_dir.name} – political compass ({x_dim} × {y_dim}, per-run{tag})"
    plot_compass(run_positions, x_dim, y_dim, one_sided,
                 f"{base_title}, all languages",
                 shared_compass_dir / f"{model_dir.name}{suffix}.png", aggregate=True)

    color_map = language_colors(run_positions["language"].dropna().unique())
    plot_compass(run_positions, x_dim, y_dim, one_sided,
                 f"{base_title}, by language",
                 compass_dir / f"political_compass_by_language{suffix}.png", color_map=color_map)
    for lang in sorted(color_map):
        plot_compass(run_positions[run_positions["language"] == lang], x_dim, y_dim, one_sided,
                     f"{base_title}, {lang}",
                     compass_dir / f"political_compass_{lang}{suffix}.png",
                     color_map={lang: color_map[lang]})
    for granularity, label in VIOLIN_GRANULARITIES.items():
        responses = melt_positions(aggregate_positions(projected, dims, granularity), dims)
        plot_violins(responses, dims, one_sided,
                     f"{model_dir.name} – per-dimension stance distribution ({label}{tag})",
                     out_dir / f"dimension_violins_{granularity}{suffix}.png")
    return run_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model dir; default processes every model.")
    parser.add_argument("--x-dim", default="Left-Right")
    parser.add_argument("--y-dim", default="Europe")
    parser.add_argument("--stance-source", default="nli", choices=sorted(STANCE_SOURCE_SUFFIX),
                        help="nli: Likert choice columns + NLI-scored speeches (*_scored.csv). "
                             "llm: both tracks from {reasons,speeches}_llm_stance.csv.")
    args = parser.parse_args()
    if args.stance_source == "llm":
        SOURCE_LABEL.update(LLM_SOURCE_LABEL)

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
        runs = process_model(model_dir, questionnaire, dims, one_sided,
                             args.x_dim, args.y_dim, args.stance_source)
        if runs is not None:
            runs_by_model[model_dir.name] = runs

    if len(runs_by_model) > 1:
        out_dir = results_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        suffix = STANCE_SOURCE_SUFFIX[args.stance_source]
        tag = STANCE_SOURCE_TAG[args.stance_source]
        source_pair = " vs ".join(SOURCE_LABEL[s] for s in SOURCE_FILLED)
        plot_models_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            f"Political compass across models ({args.x_dim} × {args.y_dim}, per-run{tag})",
                            out_dir / f"political_compass_models{suffix}.png")
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            f"Political compass across models ({args.x_dim} × {args.y_dim}, per-run runs{tag})",
                            out_dir / f"political_compass_models_scatter{suffix}.png")
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            f"Political compass across models ({args.x_dim} × {args.y_dim}, per-run runs, base vs negated{tag})",
                            out_dir / f"political_compass_models_scatter_framing{suffix}.png", split_by="framing")
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            f"Political compass across models ({args.x_dim} × {args.y_dim}, per-run runs, {source_pair}{tag})",
                            out_dir / f"political_compass_models_scatter_source{suffix}.png", split_by="source")

    print("\nDone.")


if __name__ == "__main__":
    main()
