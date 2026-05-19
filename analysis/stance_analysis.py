"""
Analysis: correlate NLI stance scores with direct model choices + sanity checks.

Expects two scored CSVs (e.g. one for variant="" and one for variant="_negated")
and a separate CSV/JSONL of the model's 1-5 choice answers to the same statements,
with columns choice_{lang}_v{j} (one per prompt variant per language).
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, likert_to_stance, load_dataframe


def build_matrices(df, likert_df, languages, stance_col_fmt, choice_agree_is_low=True, choice_variant=""):
    stance_mat = np.vstack([df[stance_col_fmt.format(l=lang)].values for lang in languages])
    rows = []
    for lang in languages:
        vcols = [c for c in likert_df.columns if c.startswith(f"choice_{lang}{choice_variant}_v")]
        mean_choice = likert_df[vcols].mean(axis=1).values
        aligned = likert_to_stance(mean_choice) if choice_agree_is_low else -likert_to_stance(mean_choice)
        rows.append(aligned)
    choice_mat = np.vstack(rows)
    return stance_mat, choice_mat


def plot_negation_inconsistency_heatmaps(s_mat, s_mat_n, c_mat, c_mat_n,
                                         languages, q_labels, outfile, suptitle=None):
    """Two-panel heatmap of negation INCONSISTENCY: base + negated.
    
    If responses are negation-consistent, base ≈ -negated, so the sum ≈ 0.
    Non-zero values mean the model agrees (or disagrees) with both a statement
    and its negation — i.e. an incoherent stance.
    """
    stance_sum = s_mat + s_mat_n
    choice_sum = c_mat + c_mat_n
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.55 * len(q_labels)), 7),
                             constrained_layout=True)
    panels = [
        ("Stance inconsistency: Base + Negated (0 = consistent)", stance_sum),
        ("Choice inconsistency: Base + Negated (0 = consistent)", choice_sum),
    ]
    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap="PuOr_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(languages)
        ax.set_xticks(range(len(q_labels)))
        ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > 1.2 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_heatmaps(stance_mat, choice_mat, languages, q_labels,
                  outfile, suptitle=None):
    diff_mat = choice_mat - stance_mat
    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.55 * len(q_labels)), 9),
                             constrained_layout=True)
    panels = [
        ("Stance (speeches, NLI)", stance_mat, "RdBu_r", (-1, 1)),
        ("Choice (direct Likert)", choice_mat, "RdBu_r", (-1, 1)),
        ("Diff: Choice - Stance",  diff_mat,   "PuOr_r", (-2, 2)),
    ]
    for ax, (title, mat, cmap, (vmin, vmax)) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(languages)
        ax.set_xticks(range(len(q_labels)))
        ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

        thresh = 0.6 * max(abs(vmin), abs(vmax))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > thresh else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")

def bootstrap_ci(x, y, stat_fn, n_boot=2000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat_fn(x[idx], y[idx])[0]
    lo, hi = np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi


def report_correlation(name, x, y):
    mask = ~(pd.isna(x) | pd.isna(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    rho, p = spearmanr(x, y)
    lo, hi = bootstrap_ci(x, y, spearmanr)
    print(f"  {name}: n={len(x)}  Spearman rho = {rho:.3f}  "
          f"[{lo:.3f}, {hi:.3f}]  p = {p:.2e}")
    return rho, (lo, hi), p


def attach_choice_means(df, likert_df, languages):
    """Compute mean choice score per language and overall, attach to df."""
    all_cols = []
    for lang in languages:
        vcols = [c for c in likert_df.columns if c.startswith(f"choice_{lang}_v")]
        if not vcols:
            continue
        df[f"choice_{lang}_mean"] = likert_df[vcols].mean(axis=1).values
        all_cols.extend(vcols)
    if all_cols:
        df["choice_mean"] = likert_df[all_cols].mean(axis=1).values


def resolve_paths(args):
    results_dir = args.results_dir or f"./data/euandi_2024_results/{args.model}"
    langs = args.languages
    scored     = args.scored     or f"{results_dir}/speeches_{langs}_scored.csv"
    likert     = args.likert     or f"{results_dir}/{langs}.csv"
    scored_neg = args.scored_negated or f"{results_dir}/speeches_{langs}_negated_scored.csv"
    likert_neg = args.likert_negated or (f"{results_dir}/{langs}_negated.csv" if scored_neg else None)
    hm_base    = args.heatmap_out_base          or f"{results_dir}/heatmaps_base.png"
    hm_negated = args.heatmap_out_negated       or f"{results_dir}/heatmaps_negated.png"
    hm_diff    = args.heatmap_out_negation_diff or f"{results_dir}/heatmaps_negation_diff.png"
    os.makedirs(results_dir, exist_ok=True)
    return langs.split(","), scored, likert, scored_neg, likert_neg, hm_base, hm_negated, hm_diff


def run_stance_vs_choice(df, languages):
    print("=" * 60)
    print("1. STANCE vs CHOICE correlation (per language)")
    print("=" * 60)
    for lang in languages:
        stance_col, choice_col = f"stance_{lang}_mean", f"choice_{lang}_mean"
        if stance_col not in df or choice_col not in df:
            continue
        report_correlation(lang, df[stance_col], df[choice_col])


def run_cross_language_agreement(df, languages):
    print("\n" + "=" * 60)
    print("2. Cross-language choice agreement (should be high)")
    print("=" * 60)
    stance_cols = [f"choice_{lang}_mean" for lang in languages if f"choice_{lang}_mean" in df]
    print(df[stance_cols].corr(method="spearman").round(3))


def run_prompt_stability(df, languages):
    print("\n" + "=" * 60)
    print("3. Prompt-variant stability (mean pairwise rho within language)")
    print("=" * 60)
    for lang in languages:
        vcols = [c for c in df.columns if c.startswith(f"choice_{lang}_v")]
        if len(vcols) < 2:
            continue
        mat = df[vcols].corr(method="spearman").values
        off_diag = mat[np.triu_indices_from(mat, k=1)]
        print(f"  {lang}: mean rho across {len(vcols)} prompt variants = {off_diag.mean():.3f}")


def run_negation_analysis(df, scored_neg, languages):
    if not scored_neg:
        return
    print("\n" + "=" * 60)
    print("4. Negation consistency: stance(orig) vs -stance(negated)")
    print("=" * 60)
    df_neg = load_dataframe(scored_neg)
    for lang in languages:
        c_o, c_n = f"stance_{lang}_mean", f"stance_{lang}_negated_mean"
        if c_o in df and c_n in df_neg:
            report_correlation(lang, df[c_o], -df_neg[c_n])


def build_q_labels(df, statement_col):
    if statement_col and statement_col in df.columns:
        return [s[:45] + ("…" if len(s) > 45 else "") for s in df[statement_col]]
    return [f"Q{i+1}" for i in range(len(df))]


def generate_heatmaps(df, likert_df, scored_neg, likert_neg,
                      languages, q_labels, agree_low,
                      hm_base, hm_negated, hm_diff):
    s_mat, c_mat = build_matrices(df, likert_df, languages, "stance_{l}_mean", agree_low)
    plot_heatmaps(s_mat, c_mat, languages, q_labels, hm_base, suptitle="Base statements")

    if not (scored_neg and likert_neg):
        return
    df_neg = load_dataframe(scored_neg)
    likert_neg_df = load_dataframe(likert_neg)
    s_mat_n, c_mat_n = build_matrices(df_neg, likert_neg_df, languages,
                                      "stance_{l}_negated_mean", agree_low,
                                      choice_variant="_negated")
    plot_heatmaps(s_mat_n, c_mat_n, languages, q_labels, hm_negated, suptitle="Negated statements")
    plot_negation_inconsistency_heatmaps(s_mat, s_mat_n, c_mat, c_mat_n, languages, q_labels, hm_diff,
                                     suptitle="Negation inconsistency: Base + Negated (≈0 means consistent)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-122b", type=str, choices=["gpt-oss-120b", "qwen3.5-122b"],
                    help="Model name; sets the default results directory")
    ap.add_argument("--results_dir", default="./data/euandi_2024_results/qwen3.5-122b/",
                    help="Model results directory (default: ./data/euandi_2019_results/{model})")
    ap.add_argument("--scored", default=None,
                    help="Scored CSV (default: {results_dir}/speeches_{languages}_scored.csv)")
    ap.add_argument("--scored_negated", default=None,
                    help="Optional: scored CSV for _negated variant")
    ap.add_argument("--likert", default=None,
                    help="CSV/JSONL with choice columns (default: {results_dir}/{languages}.csv)")
    ap.add_argument("--likert_negated", default=None,
                    help="Likert CSV/JSONL for the _negated variant")
    ap.add_argument("--languages", default=ALL_LANGS_STR)
    ap.add_argument("--statement_col", default=None,
                    help="Column in --scored CSV holding the question text (optional)")
    ap.add_argument("--choice_agree_high", action="store_true",
                    help="Set if Likert 5 = agree (default assumes 1 = agree)")
    ap.add_argument("--heatmap_out_base", default=None)
    ap.add_argument("--heatmap_out_negated", default=None)
    ap.add_argument("--heatmap_out_negation_diff", default=None)
    args = ap.parse_args()

    languages, scored, likert_path, scored_neg, likert_neg, hm_base, hm_negated, hm_diff = resolve_paths(args)
    df = load_dataframe(scored)
    likert_df = load_dataframe(likert_path)
    attach_choice_means(df, likert_df, languages)

    run_stance_vs_choice(df, languages)
    run_cross_language_agreement(likert_df, languages)
    run_prompt_stability(likert_df, languages)
    run_negation_analysis(df, scored_neg, languages)

    q_labels = build_q_labels(df, args.statement_col)
    generate_heatmaps(df, likert_df, scored_neg, likert_neg, languages, q_labels,
                      not args.choice_agree_high, hm_base, hm_negated, hm_diff)


if __name__ == "__main__":
    main()