import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.evaluate_euandi import (
    DEFAULT_POSITIONS,
    POSITION_CHOICES,
    load_party_positions,
    positions_path,
)
from analysis.plotting.plot_classified_parties import model_display_name
from utils import flip_likert, likert_to_stance

NUM_VARIANTS = 8
NEUTRAL_LIKERT = 3
# Figure sizes are fixed, so text is sized to survive the ~3x downscale to an
# ACL one-column figure. The axis labels carry the long DIMENSION_POLES strings
# and are the one element that cannot take the full size without clipping.
FONTSIZE = 15
AXIS_FONTSIZE = 12
LEGEND_FONTSIZE_SMALL = 7  # the 21-entry language lookup: scanned, not read
# The compasses are dense in the upper right (PPE/ALDE anchors, the pro-EU lobe),
# so the main legend sits in the lower left and its colour dots are drawn large
# enough to be told apart after the downscale.
MAIN_LEGEND_LOC = "lower left"
MODEL_LEGEND_TITLE = "model  (large = mean)"
LEGEND_MARKERSIZE = 12
# The 21-entry language box would grow out of the panel at the full size.
LANG_LEGEND_MARKERSIZE = 9
FRAMING_SUFFIX = {"base": "", "negated": "_negated"}
FRAMING_FOR_VARIANT = {"": "base", "_negated": "negated"}
FRAMING_ORIENTATION = {"base": 1.0, "negated": -1.0}
FRAMING_COLOR = {"base": "#2ca02c", "negated": "#d62728"}
FRAMING_MARKER = {"base": "o", "negated": "X"}
# On top of the shape, each split has a hollow half: the negated framing and the
# open-ended source are drawn unfilled, so which side of a split a mark belongs to
# survives the downscale even where the two shapes overlap.
SOURCE_MARKER = {"likert": "o", "speeches": "s"}
HOLLOW_FRAMING = "negated"
HOLLOW_SOURCE = "speeches"
FRAMING_SOURCE_MARKER = {
    ("base", "likert"): "o", ("base", "speeches"): "s",
    ("negated", "likert"): "X", ("negated", "speeches"): "D",
}
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
    "Left-Right": ("Left", "Right"),
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
    # axis columns only: the file also carries `statement` and the `axis_source`
    # provenance string written by rebuild_questionnaire_axes.py
    dims = [c for c in df.columns
            if c not in ("statement_idx", "statement") and pd.api.types.is_numeric_dtype(df[c])]
    one_sided = {d for d in dims if len(set(np.sign(df[d])) - {0.0}) < 2}
    df["admin_idx"] = administration_order(dataset, df["statement"])
    return df, dims, one_sided


def administration_order(dataset, statements):
    """Row each statement occupies in the result CSVs.

    The questionnaire is in codebook order (s1..s36); every result CSV is written in
    statements.jsonl order, and the two differ for 19 of the 30 statements. The
    stance loaders index rows by CSV position, so projecting them onto the axes
    requires this permutation -- merging on raw position silently attaches the wrong
    statement's axes.
    """
    path = Path("data") / f"{dataset}_data" / "statements.jsonl"
    administered = [json.loads(line)["statement"]["en"]
                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    position = {" ".join(s.split()): i for i, s in enumerate(administered)}
    keys = [" ".join(str(s).split()) for s in statements]
    missing = [k for k in keys if k not in position]
    if missing:
        raise SystemExit(f"{len(missing)} questionnaire statement(s) are not in {path}; "
                         f"first: {missing[0]!r}")
    return [position[k] for k in keys]


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


# Reference anchors: the EP groups' own euandi answers, projected onto the same
# dimensions as the model runs. A star is used nowhere else on these plots, so a
# black star always means "this is a party, not a run".
PARTY_MARKER = "*"
PARTY_COLOR = "black"
PARTY_LABEL_COLOR = "#1a1a1a"


def party_anchors(questionnaire, dims, positions=DEFAULT_POSITIONS):
    """Mean position per EP group on every dimension, indexed by group name."""
    party_df = load_party_positions(positions_path(positions), positions)
    party_df = party_df.dropna(subset=["ep_group"])
    wide = party_df.pivot_table(index=["ep_group", "short_name"], columns="statement_idx",
                                values="normalized_answer")

    signs = questionnaire[dims].to_numpy(dtype=float)[wide.columns.to_numpy()]
    answers = wide.to_numpy(dtype=float)
    projected = pd.DataFrame(index=wide.index)
    for i, dim in enumerate(dims):
        loading = signs[:, i] != 0
        projected[dim] = np.nanmean(answers[:, loading] * signs[loading, i], axis=1)
    # For ep-group positions this is one europarty per group and averages nothing;
    # for national/group-mean it pools the member parties.
    return projected.groupby("ep_group").mean()


def draw_party_anchors(ax, anchors, x_dim, y_dim, size=260, fontsize=FONTSIZE,
                       color=PARTY_COLOR):
    """The label ink stays dark whatever the marker colour: a light star (EU
    yellow, say) is fine as a mark but unreadable as text on a white panel."""
    if anchors is None:
        return None
    edge = "white" if color == PARTY_COLOR else PARTY_LABEL_COLOR
    for label, row in anchors.iterrows():
        if not (np.isfinite(row[x_dim]) and np.isfinite(row[y_dim])):
            continue
        ax.scatter(row[x_dim], row[y_dim], s=size, marker=PARTY_MARKER,
                   color=color, edgecolor=edge, linewidths=1.2, zorder=8)
        # Bold, with a fatter white halo: the group names are read on top of the
        # per-run dot cloud, which regular weight disappears into.
        ax.annotate(label, (row[x_dim], row[y_dim]), textcoords="offset points",
                    xytext=(9, 4), fontsize=fontsize, color=PARTY_LABEL_COLOR, zorder=9,
                    fontweight="bold",
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=3.5, foreground="white")])
    return Line2D([], [], color=color, marker=PARTY_MARKER, ls="",
                  markersize=LEGEND_MARKERSIZE, markeredgecolor=edge, label="EP group")


def project_onto_dimensions(stances, questionnaire, dims):
    # row_idx from the stance loaders is a position in the result CSV, so the axis
    # signs are keyed by admin_idx, not by the questionnaire's own row order
    signs = questionnaire[[*dims, "admin_idx"]].rename(columns={"admin_idx": "row_idx"})
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
    ax.set_xlabel(axis_label(x_dim, one_sided), fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(axis_label(y_dim, one_sided), fontsize=AXIS_FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True, linestyle=":", alpha=0.3)


def axis_label(dim, one_sided):
    negative, positive = DIMENSION_POLES.get(dim, ("disagree", "agree"))
    flag = "  (one-sided!)" if dim in one_sided else ""
    return f"← {negative}      {dim}{flag}      {positive} →"


NEUTRAL_COLOR = "#1f77b4"


def is_hollow(framing=None, source=None):
    return framing == HOLLOW_FRAMING or source == HOLLOW_SOURCE


def mark_style(color, hollow):
    """facecolor/edgecolor pair for a scatter; the colour moves to the outline."""
    return {"facecolor": "none" if hollow else color, "edgecolor": color}


# The likert track is a lattice: a run is a mean of stances that are multiples of
# 0.5, so every run lands on a multiple of 0.5/n_loading and dozens of them stack on
# one vertex. Only the drawn copy is nudged, and only inside its own cell -- the
# means and the anchors keep using the exact values.
JITTERED_SOURCE = "likert"
JITTER_FRACTION = 0.3  # of one grid cell, so a dot stays nearer its own vertex
JITTER_SEED = 20240  # fixed: the same run must reproduce the same figure


def grid_steps(questionnaire, dims):
    """Lattice spacing of the likert runs on each dimension (0 where nothing loads)."""
    return np.array([0.5 / n if (n := int((questionnaire[d] != 0).sum())) else 0.0
                     for d in dims])


def jitter_likert(frame, x_dim, y_dim, steps, rng):
    """Drawing copy of a run frame's points, with the likert lattice broken up."""
    points = frame[[x_dim, y_dim]].to_numpy()
    if rng is None or steps is None:
        return points
    mask = (frame["source"] == JITTERED_SOURCE).to_numpy()
    points = points.copy()
    points[mask] += rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, (int(mask.sum()), 2)) * steps
    return points


def legend_mark(color, marker, label, hollow=False, markersize=LEGEND_MARKERSIZE, ls="", **kwargs):
    return Line2D([], [], color=color, marker=marker, ls=ls, label=label,
                  markersize=markersize, markeredgecolor=color,
                  markerfacecolor="none" if hollow else color, **kwargs)


def plot_compass(runs, x_dim, y_dim, one_sided, out_path, color_map=None, aggregate=False,
                 anchors=None):
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

    sources = [s for s in SOURCE_MARKER if s in set(runs["source"])]
    framings = [f for f in FRAMING_MARKER if f in set(runs["framing"])]

    # Means stay filled whatever the run marks do: hollow at this size reads as a
    # ring the eye loses against the dot cloud.
    def draw_mean(points, marker, color):
        ax.scatter(*points.mean(axis=0), s=260, marker=marker, facecolor=color,
                   edgecolor="black", linewidths=1.8, zorder=5)

    for framing in framings:
        for source in sources:
            marker = FRAMING_SOURCE_MARKER[(framing, source)]
            hollow = is_hollow(framing, source)
            if aggregate:
                framing_color = FRAMING_COLOR[framing]
                points = runs[(runs["framing"] == framing) &
                              (runs["source"] == source)][[x_dim, y_dim]].dropna().to_numpy()
                if len(points):
                    ax.scatter(points[:, 0], points[:, 1], s=10, marker=marker,
                               **mark_style(framing_color, hollow),
                               alpha=0.5, linewidths=0.6)
                    draw_mean(points, marker, framing_color)
                continue
            for lang in sorted(runs["language"].dropna().unique()):
                color = color_map[lang] if color_map else NEUTRAL_COLOR
                points = runs[(runs["language"] == lang) &
                              (runs["framing"] == framing) &
                              (runs["source"] == source)][[x_dim, y_dim]].dropna().to_numpy()
                if not len(points):
                    continue
                ax.scatter(points[:, 0], points[:, 1], s=18, marker=marker,
                           **mark_style(color, hollow), alpha=0.35, linewidths=0.8)
                draw_mean(points, marker, color)

    party_handle = draw_party_anchors(ax, anchors, x_dim, y_dim)
    show_languages = color_map if color_map and len(color_map) > 1 else None
    add_language_framing_legends(ax, show_languages, framings, sources,
                                 aggregate=aggregate, party_handle=party_handle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def add_language_framing_legends(ax, colors, framings, sources, aggregate=False,
                                 party_handle=None):
    if colors:
        lang_handles = [Line2D([], [], color=c, marker="o", ls="", label=lang,
                               markersize=LANG_LEGEND_MARKERSIZE)
                        for lang, c in colors.items()]
        legend = ax.legend(handles=lang_handles, loc=MAIN_LEGEND_LOC, title="language",
                           fontsize=LEGEND_FONTSIZE_SMALL, title_fontsize=LEGEND_FONTSIZE_SMALL,
                           ncol=2 if len(lang_handles) > 8 else 1, framealpha=0.9)
        if party_handle is not None:
            ax.add_artist(legend)
            ax.legend(handles=[party_handle], loc="lower right", fontsize=FONTSIZE,
                      framealpha=0.9)
        return
    framing_color = (lambda f: FRAMING_COLOR[f]) if aggregate else (lambda f: "gray")
    style_handles = [legend_mark(framing_color(f), FRAMING_SOURCE_MARKER[(f, s)],
                                 f"{f} · {SOURCE_LABEL[s]}", hollow=is_hollow(f, s))
                     for f in framings for s in sources]
    if party_handle is not None:
        style_handles.append(party_handle)
    ax.legend(handles=style_handles, loc=MAIN_LEGEND_LOC, fontsize=FONTSIZE, framealpha=0.9,
              title_fontsize=FONTSIZE,
              title="framing · source  (small = run, large = mean)")


def violin_series(responses, framing, source, dims, positions_for_dim):
    positions, datasets = [], []
    subset = responses[(responses["framing"] == framing) & (responses["source"] == source)]
    for pos, d in zip(positions_for_dim, dims):
        values = subset[subset["dimension"] == d]["position"].to_numpy()
        if len(values):
            positions.append(pos)
            datasets.append(values)
    return positions, datasets


def plot_violins(responses, dims, one_sided, out_path):
    framings = [f for f in FRAMING_COLOR if f in set(responses["framing"])]
    sources = [s for s in SOURCE_MARKER if s in set(responses["source"])]
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
    ax.set_xticklabels([d + ("\n(one-sided!)" if d in one_sided else "") for d in dims],
                       fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("position  (← disagree   ·   agree →)", fontsize=AXIS_FONTSIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    framing_handles = [Line2D([], [], color=c, marker="s", ls="", label=f)
                       for f, c in FRAMING_COLOR.items() if f in set(framings)]
    source_note = "  ".join(f"{SOURCE_VIOLIN_SIDE[s]} half = {SOURCE_LABEL[s]}" for s in sources) if split else ""
    legend = ax.legend(handles=framing_handles, loc="upper right", fontsize=FONTSIZE,
                       title_fontsize=FONTSIZE, title="framing (black bar = mean)")
    if source_note:
        ax.text(0.99, 0.02, source_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=FONTSIZE)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def draw_models_scatter_panel(ax, runs_by_model, x_dim, y_dim, split_by, anchors, steps):
    """One cross-model scatter panel, legends left to the caller.

    Returns (model handles, style handles, style title): a single-panel figure puts both
    legends inside the axes, the stacked figure keeps the style legend in its panel and
    draws one shared model legend under both."""
    cmap = plt.get_cmap("tab10" if len(runs_by_model) <= 10 else "tab20")
    all_runs = pd.concat(runs_by_model.values(), ignore_index=True)
    framings = [f for f in FRAMING_MARKER if f in set(all_runs["framing"])]
    sources = [s for s in SOURCE_MARKER if s in set(all_runs["source"])]
    rng = np.random.default_rng(JITTER_SEED)

    def draw(frame, color, marker, hollow=False):
        frame = frame.dropna(subset=[x_dim, y_dim])
        if frame.empty:
            return
        drawn = jitter_likert(frame, x_dim, y_dim, steps, rng)
        ax.scatter(drawn[:, 0], drawn[:, 1], s=12, marker=marker,
                   **mark_style(color, hollow), alpha=0.7, linewidths=0.5)
        # the mean is taken from the exact values, never from the jittered copy
        points = frame[[x_dim, y_dim]].to_numpy()
        ax.scatter(*points.mean(axis=0), s=240, marker=marker, facecolor=color,
                   edgecolor="black", linewidths=1.8, zorder=5)

    model_handles = []
    for i, (model, runs) in enumerate(sorted(runs_by_model.items())):
        color = cmap(i % cmap.N)
        model_handles.append(legend_mark(color, "o", model_display_name(model)))
        if split_by == "framing":
            for framing in framings:
                draw(runs[runs["framing"] == framing], color, FRAMING_MARKER[framing],
                     is_hollow(framing=framing))
        elif split_by == "source":
            for source in sources:
                draw(runs[runs["source"] == source], color, SOURCE_MARKER[source],
                     is_hollow(source=source))
        else:
            draw(runs, color, "o")

    party_handle = draw_party_anchors(ax, anchors, x_dim, y_dim)
    style_handles, style_title = [], None
    if split_by == "framing":
        style_handles = [legend_mark("gray", FRAMING_MARKER[f], f, hollow=is_hollow(framing=f))
                         for f in framings]
        style_title = "framing"
    elif split_by == "source":
        style_handles = [legend_mark("gray", SOURCE_MARKER[s], SOURCE_LABEL[s],
                                     hollow=is_hollow(source=s))
                         for s in sources]
        style_title = "source"
    if party_handle is not None:
        style_handles.append(party_handle)
    return model_handles, style_handles, style_title


def place_style_legend(ax, style_handles, style_title, loc="lower right"):
    if style_handles:
        ax.legend(handles=style_handles, loc=loc, fontsize=FONTSIZE,
                  title_fontsize=FONTSIZE, framealpha=0.9, title=style_title)


def plot_models_scatter_compass(runs_by_model, x_dim, y_dim, one_sided, out_path,
                                split_by=None, anchors=None, steps=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_compass(ax, x_dim, y_dim, one_sided)
    model_handles, style_handles, style_title = draw_models_scatter_panel(
        ax, runs_by_model, x_dim, y_dim, split_by, anchors, steps)
    model_legend = ax.legend(handles=model_handles, loc=MAIN_LEGEND_LOC, fontsize=FONTSIZE,
                             title_fontsize=FONTSIZE, framealpha=0.9,
                             title=MODEL_LEGEND_TITLE)
    ax.add_artist(model_legend)
    place_style_legend(ax, style_handles, style_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_models_scatter_panels(runs_by_model, x_dim, y_dim, one_sided, out_path,
                               splits=("source", "framing"), anchors=None, steps=None):
    """The split-by-source and split-by-framing scatters as one figure.

    The two are the same runs under two splits, so the 13-entry model legend is the same
    on both -- and inside the axes it covers a quarter of the compass and prints through
    the EP-group labels. Here it is drawn once, under the bottom panel, and only the
    2-3 entry style legend stays in the panel it belongs to."""
    fig, axes = plt.subplots(len(splits), 1, figsize=(9, 9 * len(splits)))
    shared_handles = None
    for ax, split_by in zip(np.atleast_1d(axes), splits):
        setup_compass(ax, x_dim, y_dim, one_sided)
        model_handles, style_handles, style_title = draw_models_scatter_panel(
            ax, runs_by_model, x_dim, y_dim, split_by, anchors, steps)
        ax.set_title(f"split by {split_by}", fontsize=FONTSIZE)
        # Lower left, the corner the model legend just vacated: on the right it sits on
        # top of the ID anchor's label, which is the one group out in that corner.
        place_style_legend(ax, style_handles, style_title, loc=MAIN_LEGEND_LOC)
        # Same models, same order, same colours on every panel, so the first panel's
        # handles stand for all of them.
        shared_handles = shared_handles or model_handles

    fig.tight_layout()
    # A figure legend is not laid out by tight_layout, so the panels keep the whole canvas
    # and bbox_inches="tight" grows the saved image to take the legend in below them.
    fig.legend(handles=shared_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=FONTSIZE, title=MODEL_LEGEND_TITLE,
               title_fontsize=FONTSIZE, frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_models_compass(runs_by_model, x_dim, y_dim, one_sided, out_path, anchors=None):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    setup_compass(ax, x_dim, y_dim, one_sided)
    cmap = plt.get_cmap("tab10" if len(runs_by_model) <= 10 else "tab20")
    all_runs = pd.concat(runs_by_model.values(), ignore_index=True)
    sources = [s for s in SOURCE_MARKER if s in set(all_runs["source"])]
    model_handles = []
    for i, (model, runs) in enumerate(sorted(runs_by_model.items())):
        color = cmap(i % cmap.N)
        model_handles.append(legend_mark(color, "s", model_display_name(model)))
        for source in sources:
            points = runs[runs["source"] == source][[x_dim, y_dim]].dropna().to_numpy()
            if len(points) < 3:
                continue
            ax.add_patch(covariance_ellipse(points, 2, edgecolor=color, facecolor="none",
                                             lw=1.3, ls=SOURCE_LINESTYLE[source]))
            ax.scatter(*points.mean(axis=0), s=110, color=color, edgecolor="black",
                       marker=SOURCE_MARKER[source], zorder=4)
    # This panel plots means only -- nothing hollow to key, the sources split by
    # marker and by the ellipse line style.
    party_handle = draw_party_anchors(ax, anchors, x_dim, y_dim, size=200)
    source_handles = [legend_mark("gray", SOURCE_MARKER[s], f"{SOURCE_LABEL[s]} (mean, 2σ)",
                                  ls=SOURCE_LINESTYLE[s])
                      for s in sources]
    if party_handle is not None:
        source_handles.append(party_handle)
    first = ax.legend(handles=model_handles, loc=MAIN_LEGEND_LOC, fontsize=FONTSIZE,
                      title_fontsize=FONTSIZE, framealpha=0.9, title="model")
    ax.add_artist(first)
    ax.legend(handles=source_handles, loc="lower right", fontsize=FONTSIZE,
              title_fontsize=FONTSIZE, framealpha=0.9, title="source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# One row per model on the per-axis strips, with the split's two halves on their own half
# of the row: pooled into one band their clouds sit on top of each other, and the gap
# between the halves is what these figures are read for. Either split can be drawn -- the
# markers, the hollow half and the labels are the ones the compasses already use.
SOURCE_BAND = {"likert": 0.2, "speeches": -0.2}
FRAMING_BAND = {"base": 0.2, "negated": -0.2}
STRIP_SPLITS = {
    "source": (SOURCE_BAND, SOURCE_MARKER, SOURCE_LABEL, "framings"),
    "framing": (FRAMING_BAND, FRAMING_MARKER, {f: f for f in FRAMING_MARKER}, "sources"),
}
BAND_SPREAD = 0.13  # vertical jitter inside a half-row, short of the neighbouring band


STRIP_NCOLS = 2
STRIP_PANEL_SIZE = (9.0, 0.46)  # inches: panel width, and height per model row


def draw_axis_strip_panel(ax, runs_by_model, dim, one_sided, split_by="source", step=None,
                          model_names=True):
    """Every model's runs on one questionnaire axis, one row per model.

    The compass spends both of its dimensions on Left-Right and Europe, which leaves the
    topical axes with no figure of their own. A row per model reads them one axis at a
    time: where each model sits, how wide its runs spread, and how far the two halves of
    `split_by` are apart -- against the EP groups, drawn as reference lines rather than as
    points."""
    bands, markers, labels, _ = STRIP_SPLITS[split_by]
    models = sorted(runs_by_model)
    all_runs = pd.concat(runs_by_model.values(), ignore_index=True)
    halves = [h for h in markers if h in set(all_runs[split_by])]
    cmap = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")
    rng = np.random.default_rng(JITTER_SEED)

    for index, model in enumerate(models):
        color = cmap(index % cmap.N)
        runs = runs_by_model[model]
        for half in halves:
            rows = runs[runs[split_by] == half].dropna(subset=[dim])
            if rows.empty:
                continue
            values = rows[dim].to_numpy()
            band = index + bands[half]
            x = values.copy()
            if step:
                # Under the framing split a band holds both tracks, so the lattice is
                # broken up by the run's own source rather than by the band it is in.
                lattice = (rows["source"] == JITTERED_SOURCE).to_numpy()
                x[lattice] += rng.uniform(-JITTER_FRACTION, JITTER_FRACTION,
                                          int(lattice.sum())) * step
            ax.scatter(x, band + rng.uniform(-BAND_SPREAD, BAND_SPREAD, len(x)), s=9,
                       marker=markers[half], alpha=0.45, linewidths=0.5,
                       **mark_style(color, is_hollow(**{split_by: half})))
            # the mean is taken from the exact values, never from the jittered copy
            ax.scatter(values.mean(), band, s=150, marker=markers[half],
                       facecolor=color, edgecolor="black", linewidths=1.5, zorder=5)

    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(len(models) - 0.5, -0.5)  # first model on top
    ax.set_yticks(range(len(models)),
                  [model_display_name(m) for m in models] if model_names else [],
                  fontsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE)
    ax.set_xlabel(axis_label(dim, one_sided), fontsize=AXIS_FONTSIZE)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    return [legend_mark("gray", markers[h], labels[h],
                        hollow=is_hollow(**{split_by: h})) for h in halves]


def plot_axis_strips(runs_by_model, dims, one_sided, out_path, anchors=None,
                     steps_by_dim=None, split_by="source", ncols=STRIP_NCOLS):
    """The per-axis strips as one figure, the legend taking the leftover cell.

    Five axes in a two-column grid leave one cell empty, which is exactly the room the
    legend needs -- so the whole set is one figure with one legend rather than five
    figures each repeating it."""
    models = sorted(runs_by_model)
    nrows = -(-(len(dims) + 1) // ncols)  # +1: the legend occupies a cell of its own
    panel_width, row_height = STRIP_PANEL_SIZE
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharey=True,
                             figsize=(panel_width * ncols,
                                      (row_height * len(models) + 1.6) * nrows))
    flat = axes.ravel()
    handles = []
    for cell, dim in enumerate(dims):
        # Names on the left column only; sharey keeps every panel on the same rows.
        handles = draw_axis_strip_panel(flat[cell], runs_by_model, dim, one_sided,
                                        split_by=split_by,
                                        step=(steps_by_dim or {}).get(dim),
                                        model_names=cell % ncols == 0)
    for cell in range(len(dims), len(flat)):
        flat[cell].axis("off")

    if anchors is not None:
        handles.append(Line2D([], [], color=PARTY_COLOR, ls=":", lw=1.2, label="EP group"))
    pooled = STRIP_SPLITS[split_by][3]
    title = f"{split_by}  (large = mean, runs pooled over {pooled})"
    flat[-1].legend(handles=handles, loc="center", fontsize=FONTSIZE, frameon=False,
                    title=title, title_fontsize=FONTSIZE)

    # The group names are packed by measuring them, so the panels have to be where they
    # will finally sit before they are drawn -- hence after the layout, not before.
    fig.tight_layout()
    reserve_anchor_lanes(fig, flat[:len(dims)], dims, anchors)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def reserve_anchor_lanes(fig, panel_axes, dims, anchors):
    """Draw the group names, then open up enough row spacing for the lanes they took.

    A name sits above its own panel, so on a grid the deepest stack -- five groups pile up
    at the top of the Ukraine axis -- would print into the panel above it. The lanes are
    only known once the names have been measured, so they are drawn, counted, cleared, and
    drawn again against the spacing that count asks for; that keeps the row gaps equal
    rather than padding one row and leaving the rest tight."""
    if anchors is None:
        return
    drawn = [draw_axis_anchors(ax, anchors, dim) for ax, dim in zip(panel_axes, dims)]
    deepest = max((lanes for _, lanes in drawn), default=0)
    if deepest < 2:
        return
    renderer = fig.canvas.get_renderer()
    panel_height = panel_axes[0].get_window_extent(renderer).height / fig.dpi
    extra = (deepest - 1) * ANCHOR_LANE_STEP * AXIS_FONTSIZE / 72.0
    for artists, _ in drawn:
        for artist in artists:
            artist.remove()
    fig.subplots_adjust(hspace=fig.subplotpars.hspace + extra / panel_height)
    for ax, dim in zip(panel_axes, dims):
        draw_axis_anchors(ax, anchors, dim)


ANCHOR_LANE_PAD = 4  # points of clear space demanded between two names in one lane
ANCHOR_LANE_STEP = 1.35  # line height, in multiples of the label's own font size


def draw_axis_anchors(ax, anchors, dim):
    """The EP groups as vertical reference lines, named in lanes above the strip.

    On the topical axes the groups are not spread out the way they are on Left-Right --
    three of them land within a few hundredths on Ecology -- so their names have to be
    packed rather than simply placed. Each name is drawn, measured, and pushed up a lane
    at a time until it clears everything already in that lane; the line itself always
    stays exactly on the group's position, and a figure with no crowding still gets a
    single row of names.

    Returns (every artist drawn, how many lanes the names took), so a caller stacking
    several panels can clear them and make room for the deepest of them."""
    if anchors is None:
        return [], 0
    figure = ax.figure
    figure.canvas.draw()  # positions must be final before anything is measured
    renderer = figure.canvas.get_renderer()
    lanes, artists = [], []
    panel = ax.get_window_extent(renderer)
    for label, position in anchors[dim].dropna().sort_values().items():
        artists.append(
            ax.axvline(position, color=PARTY_COLOR, ls=":", lw=1.2, alpha=0.7, zorder=1))
        annotation = ax.annotate(
            label, (position, 1.0), xycoords=("data", "axes fraction"),
            textcoords="offset points", xytext=(0, 5), ha="center", va="bottom",
            fontsize=AXIS_FONTSIZE, color=PARTY_LABEL_COLOR, fontweight="bold",
            path_effects=[matplotlib.patheffects.withStroke(
                linewidth=3.0, foreground="white")])
        box = annotation.get_window_extent(renderer)
        # A group sitting at either end of the axis would otherwise hang off the panel --
        # and on a grid, straight into the neighbouring one. Pull it back inside; the
        # line it names is right there, so a centred name is not worth a collision.
        shift = max(0.0, panel.x0 - box.x0) - max(0.0, box.x1 - panel.x1)
        span = (box.x0 + shift - ANCHOR_LANE_PAD, box.x1 + shift + ANCHOR_LANE_PAD)
        lane = next((i for i, taken in enumerate(lanes)
                     if all(span[1] < low or span[0] > high for low, high in taken)),
                    len(lanes))
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append(span)
        annotation.xyann = (shift * 72.0 / figure.dpi,
                            5 + lane * ANCHOR_LANE_STEP * AXIS_FONTSIZE)
        artists.append(annotation)
    return artists, len(lanes)


VIOLIN_GRANULARITIES = ["run", "none"]  # per-run mean vs. every single answer


def process_model(model_dir, questionnaire, dims, one_sided, x_dim, y_dim, stance_source,
                  anchors=None):
    print(f"\nProcessing: {model_dir.name}")
    stances = load_stances(model_dir, stance_source)
    if stances.empty:
        print("  No response CSVs found, skipping.")
        return None
    suffix = STANCE_SOURCE_SUFFIX[stance_source]

    projected = project_onto_dimensions(stances, questionnaire, dims)
    run_positions = aggregate_positions(projected, dims, "run")

    out_dir = model_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    compass_dir = out_dir / "political_compasses"
    compass_dir.mkdir(exist_ok=True)
    shared_compass_dir = model_dir.parent / "plots" / "political_compasses"
    shared_compass_dir.mkdir(parents=True, exist_ok=True)

    plot_compass(run_positions, x_dim, y_dim, one_sided,
                 shared_compass_dir / f"{model_dir.name}{suffix}.png", aggregate=True,
                 anchors=anchors)

    color_map = language_colors(run_positions["language"].dropna().unique())
    plot_compass(run_positions, x_dim, y_dim, one_sided,
                 compass_dir / f"political_compass_by_language{suffix}.png",
                 color_map=color_map, anchors=anchors)
    for lang in sorted(color_map):
        plot_compass(run_positions[run_positions["language"] == lang], x_dim, y_dim, one_sided,
                     compass_dir / f"political_compass_{lang}{suffix}.png",
                     color_map={lang: color_map[lang]}, anchors=anchors)
    for granularity in VIOLIN_GRANULARITIES:
        responses = melt_positions(aggregate_positions(projected, dims, granularity), dims)
        plot_violins(responses, dims, one_sided,
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
    parser.add_argument("--parties", default=DEFAULT_POSITIONS,
                        choices=[*POSITION_CHOICES, "none"],
                        help="Which euandi answers the black EP-group anchors come from; "
                             "'none' draws no anchors.")
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

    anchors = None if args.parties == "none" else party_anchors(questionnaire, dims, args.parties)
    if anchors is not None:
        print(f"\nEP-group anchors ({args.parties}):")
        print(anchors[[args.x_dim, args.y_dim]].round(3).to_string())

    runs_by_model = {}
    for model_dir in model_dirs:
        runs = process_model(model_dir, questionnaire, dims, one_sided,
                             args.x_dim, args.y_dim, args.stance_source, anchors)
        if runs is not None:
            runs_by_model[model_dir.name] = runs

    if len(runs_by_model) > 1:
        out_dir = results_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        suffix = STANCE_SOURCE_SUFFIX[args.stance_source]
        steps = grid_steps(questionnaire, [args.x_dim, args.y_dim])
        plot_models_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            out_dir / f"political_compass_models{suffix}.png", anchors=anchors)
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            out_dir / f"political_compass_models_scatter{suffix}.png",
                            anchors=anchors, steps=steps)
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            out_dir / f"political_compass_models_scatter_framing{suffix}.png",
                            split_by="framing", anchors=anchors, steps=steps)
        plot_models_scatter_compass(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            out_dir / f"political_compass_models_scatter_source{suffix}.png",
                            split_by="source", anchors=anchors, steps=steps)
        plot_models_scatter_panels(runs_by_model, args.x_dim, args.y_dim, one_sided,
                            out_dir / f"political_compass_models_scatter_panels{suffix}.png",
                            anchors=anchors, steps=steps)
        # The compass spends its two dimensions on x_dim and y_dim; every other axis gets
        # a strip panel instead, the whole set on one figure.
        strip_dims = [d for d in dims if d not in (args.x_dim, args.y_dim)]
        strip_steps = {d: grid_steps(questionnaire, [d])[0] for d in strip_dims}
        for split_by in STRIP_SPLITS:
            plot_axis_strips(runs_by_model, strip_dims, one_sided,
                             out_dir / f"axis_strips_{split_by}{suffix}.png",
                             anchors=anchors, steps_by_dim=strip_steps,
                             split_by=split_by)

    print("\nDone.")


if __name__ == "__main__":
    main()
