"""
Analysis: correlate NLI stance scores with direct model choices + sanity checks.

Expects two scored CSVs (e.g. one for variant="" and one for variant="_negated")
and a separate CSV/JSONL of the model's 1-5 choice answers to the same statements,
with columns choice_{lang}_v{j} (one per prompt variant per language).
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def stance_to_likert(score):
    """Map [-1, 1] stance score to [1, 5] Likert scale (linear, for plotting only)."""
    return 1 + 2 * (score + 1)

def build_matrices(df, likert_df, languages, stance_col_fmt, choice_agree_is_low=True, choice_variant=""):
    stance_mat = np.vstack([df[stance_col_fmt.format(l=l)].values for l in languages])
    rows = []
    for l in languages:
        vcols = [c for c in likert_df.columns if c.startswith(f"choice_{l}{choice_variant}_v")]
        mean_choice = likert_df[vcols].mean(axis=1).values  # 1..5
        aligned = (3 - mean_choice) / 2 if choice_agree_is_low else (mean_choice - 3) / 2
        rows.append(aligned)
    choice_mat = np.vstack(rows)
    return stance_mat, choice_mat


def plot_negation_diff_heatmaps(s_mat, s_mat_n, c_mat, c_mat_n,
                                languages, q_labels, outfile, suptitle=None):
    """Two-panel heatmap: stance(base)−stance(negated) and choice(base)−choice(negated)."""
    stance_diff = s_mat - s_mat_n
    choice_diff = c_mat - c_mat_n
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.55 * len(q_labels)), 7),
                             constrained_layout=True)
    panels = [
        ("Stance diff: Base − Negated", stance_diff),
        ("Choice diff: Base − Negated", choice_diff),
    ]
    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap="PuOr_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(languages))); ax.set_yticklabels(languages)
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
        ("Diff: Choice − Stance",  diff_mat,   "PuOr_r", (-2, 2)),
    ]
    for ax, (title, mat, cmap, (vmin, vmax)) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(languages))); ax.set_yticklabels(languages)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="./data/euandi_2019_results/gpt-oss-120b_speeches_en,de,el,es,fr,it.csv_scored",
                    help="Scored CSV (original variant, output of nli_stance_scoring.py)")
    ap.add_argument("--scored_negated", default="./data/euandi_2019_results/gpt-oss-120b_speeches_speeches_en,de,el,es,fr,it_negated.csv_scored",
                    help="Optional: scored CSV for _negated variant (for sanity check)")
    ap.add_argument("--likert", default="./data/euandi_2019_results/gpt-oss-120b_en,de,el,es,fr,it.csv",
                    help="CSV/JSONL with choice_{lang}_v{j} columns aligned row-wise to the scored file")
    ap.add_argument("--likert_negated", default="./data/euandi_2019_results/gpt-oss-120b_en,de,el,es,fr,it_negated.csv",
                    help="Likert CSV/JSONL for the _negated variant")
    ap.add_argument("--languages", default="en,de,el,es,fr,it")
    ap.add_argument("--statement_col", default=None,
                    help="Column in --scored CSV holding the question text (optional)")
    ap.add_argument("--heatmap_out", default="heatmaps.png")
    ap.add_argument("--choice_agree_high", action="store_true",
                    help="Set if Likert 5 = agree (default assumes 1 = agree)")
    ap.add_argument("--heatmap_out_base", default="heatmaps_base.png")
    ap.add_argument("--heatmap_out_negated", default="heatmaps_negated.png")
    ap.add_argument("--heatmap_out_negation_diff", default="heatmaps_negation_diff.png")
    args = ap.parse_args()

    languages = args.languages.split(",")
    df = pd.read_csv(args.scored, sep=";", encoding="utf-8-sig")

    if args.likert.endswith(".jsonl"):
        likert_df = pd.read_json(args.likert, lines=True)
    else:
        likert_df = pd.read_csv(args.likert, sep=";", encoding="utf-8-sig")
    attach_choice_means(df, likert_df, languages)

    print("=" * 60)
    print("1. STANCE vs CHOICE correlation (per language)")
    print("=" * 60)
    for lang in languages:
        stance_col = f"stance_{lang}_mean"
        choice_col = f"choice_{lang}_mean"
        if stance_col not in df or choice_col not in df:
            continue
        report_correlation(f"{lang}", df[stance_col], df[choice_col])

    print("\n" + "=" * 60)
    print("2. Cross-language stance agreement (should be high)")
    print("=" * 60)
    stance_cols = [f"stance_{lang}_mean" for lang in languages if f"stance_{lang}_mean" in df]
    corr_matrix = df[stance_cols].corr(method="spearman")
    print(corr_matrix.round(3))

    print("\n" + "=" * 60)
    print("3. Prompt-variant stability (mean pairwise rho within language)")
    print("=" * 60)
    for lang in languages:
        vcols = [c for c in df.columns if c.startswith(f"stance_{lang}_v")]
        if len(vcols) < 2:
            continue
        mat = df[vcols].corr(method="spearman").values
        off_diag = mat[np.triu_indices_from(mat, k=1)]
        print(f"  {lang}: mean rho across {len(vcols)} prompt variants = {off_diag.mean():.3f}")

    if args.scored_negated:
        print("\n" + "=" * 60)
        print("4. Negation consistency: stance(orig) vs -stance(negated)")
        print("=" * 60)
        df_neg = pd.read_csv(args.scored_negated, sep=";", encoding="utf-8-sig")
        for lang in languages:
            c_o = f"stance_{lang}_mean"
            c_n = f"stance_{lang}_negated_mean"
            if c_o in df and c_n in df_neg:
                report_correlation(f"{lang}", df[c_o], -df_neg[c_n])

    q_labels = ([s[:45] + ("…" if len(s) > 45 else "") for s in df[args.statement_col]]
                if args.statement_col and args.statement_col in df.columns
                else [f"Q{i+1}" for i in range(len(df))])
    agree_low = not args.choice_agree_high

    # Base
    s_mat, c_mat = build_matrices(df, likert_df, languages,
                                  "stance_{l}_mean", agree_low)
    plot_heatmaps(s_mat, c_mat, languages, q_labels,
                  args.heatmap_out_base, suptitle="Base statements")

    # Negated
    if args.scored_negated and args.likert_negated:
        df_neg = pd.read_csv(args.scored_negated, sep=";", encoding="utf-8-sig")
        if args.likert_negated.endswith(".jsonl"):
            likert_neg_df = pd.read_json(args.likert_negated, lines=True)
        else:
            likert_neg_df = pd.read_csv(args.likert_negated, sep=";", encoding="utf-8-sig")
        s_mat_n, c_mat_n = build_matrices(df_neg, likert_neg_df, languages,
                                          "stance_{l}_negated_mean", agree_low,
                                          choice_variant="_negated")
        plot_heatmaps(s_mat_n, c_mat_n, languages, q_labels,
                      args.heatmap_out_negated, suptitle="Negated statements")
        plot_negation_diff_heatmaps(s_mat, s_mat_n, c_mat, c_mat_n,
                                    languages, q_labels,
                                    args.heatmap_out_negation_diff,
                                    suptitle="Negation consistency: Base − Negated")


if __name__ == "__main__":
    main()