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
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import flip_likert, likert_to_stance  # noqa: E402

NUM_VARIANTS = 8
NEUTRAL_LIKERT = 3
FRAMING_SUFFIX = {"base": "", "negated": "_negated", "question": "_question"}
FRAMING_COLOR = {"base": "#1f77b4", "negated": "#d62728", "question": "#2ca02c"}
RESPONSE_CSV = re.compile(r"^(?P<langs>[a-z]{2}(?:,[a-z]{2})+)(?P<variant>|_negated|_question)\.csv$")
RUN_KEYS = ["language", "variant_idx", "framing"]

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


def find_response_csvs(model_dir):
    framing_for = {"": "base", "_negated": "negated", "_question": "question"}
    found = {}
    for path in sorted(model_dir.glob("*.csv")):
        match = RESPONSE_CSV.match(path.name)
        if not match:
            continue
        framing = framing_for[match.group("variant")]
        if framing not in found or len(match.group("langs")) > len(found[framing].stem):
            found[framing] = path
    return found


def load_stances(model_dir):
    frames = []
    for framing, path in find_response_csvs(model_dir).items():
        suffix = FRAMING_SUFFIX[framing]
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        languages = sorted(
            m.group(1) for col in df.columns
            if (m := re.match(rf"^choice_([a-z]{{2}}){re.escape(suffix)}_v0$", col))
        )
        for lang in languages:
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
                    "stance": likert_to_stance(likert).to_numpy(),
                }))
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
    long = projected.melt(id_vars=["framing"], value_vars=dims,
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
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    setup_compass(ax, x_dim, y_dim, one_sided)

    all_points = runs[[x_dim, y_dim]].dropna().to_numpy()
    if len(all_points) >= 5:
        density = gaussian_kde(all_points.T)
        gx, gy = np.mgrid[-1:1:120j, -1:1:120j]
        grid = density(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contourf(gx, gy, grid, levels=12, cmap="Greys", alpha=0.45)

    for framing, color in FRAMING_COLOR.items():
        points = runs[runs["framing"] == framing][[x_dim, y_dim]].dropna().to_numpy()
        if len(points) < 3:
            continue
        ax.scatter(points[:, 0], points[:, 1], s=14, color=color, alpha=0.4, label=framing)
        for n_std, style in ((1, "-"), (2, "--")):
            ax.add_patch(covariance_ellipse(points, n_std, edgecolor=color,
                                             facecolor="none", lw=1.4, ls=style))
        mean = points.mean(axis=0)
        ax.scatter(*mean, s=170, color=color, edgecolor="black", marker="*", zorder=4)

    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=8, title="framing (★ = mean, ellipses = 1σ/2σ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_violins(responses, dims, one_sided, title, out_path):
    framings = [f for f in FRAMING_COLOR if f in set(responses["framing"])]
    x = np.arange(len(dims))
    band_width = 0.8 / max(len(framings), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(dims) * 1.5), 6))
    ax.axhline(0, color="gray", lw=0.8)
    legend = {}
    for i, framing in enumerate(framings):
        offset = -0.4 + band_width * (i + 0.5)
        positions, datasets = [], []
        for j, d in enumerate(dims):
            values = responses[(responses["framing"] == framing) &
                               (responses["dimension"] == d)]["position"].to_numpy()
            if len(values):
                positions.append(x[j] + offset)
                datasets.append(values)
        if not datasets:
            continue
        parts = ax.violinplot(datasets, positions=positions, widths=band_width * 0.9,
                              showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(FRAMING_COLOR[framing])
            body.set_alpha(0.5)
        parts["cmeans"].set_color("black")
        legend[framing] = parts["bodies"][0]

    ax.set_xticks(x)
    ax.set_xticklabels([d + ("\n(one-sided!)" if d in one_sided else "") for d in dims], fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("position  (← disagree   ·   agree →)")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(legend.values(), legend.keys(), loc="upper right", fontsize=8,
              title="framing (black bar = mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_models_compass(runs_by_model, x_dim, y_dim, one_sided, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_compass(ax, x_dim, y_dim, one_sided)
    cmap = plt.get_cmap("tab10" if len(runs_by_model) <= 10 else "tab20")
    for i, (model, runs) in enumerate(sorted(runs_by_model.items())):
        points = runs[[x_dim, y_dim]].dropna().to_numpy()
        if len(points) < 3:
            continue
        color = cmap(i % cmap.N)
        ax.add_patch(covariance_ellipse(points, 2, edgecolor=color, facecolor=color, lw=1.0, alpha=0.12))
        ax.add_patch(covariance_ellipse(points, 2, edgecolor=color, facecolor="none", lw=1.3))
        ax.scatter(*points.mean(axis=0), s=110, color=color, edgecolor="black", label=model)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=8, title="model (● = mean, ellipse = 2σ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def process_model(model_dir, questionnaire, dims, one_sided, x_dim, y_dim):
    print(f"\nProcessing: {model_dir.name}")
    stances = load_stances(model_dir)
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
