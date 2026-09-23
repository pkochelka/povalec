#!/usr/bin/env python3
"""K-means over every EU&I 2024 party that holds at least one MEP: a data-driven
"fictional EP" to hold up against the real group structure.

Input is the raw EU&I party dataset, `data/euandi_2024_raw/EUandI_2024_party_dataset.csv`,
not `euandi_2024_parties.jsonl` -- the jsonl covers five countries, the raw file covers
all 27. Parsing follows `plot_political_bias.training_mix_party_positions`, except for
the statement mapping (that function assumes s<i> is questionnaire row i, which only
holds up to s10):

  * `;`-separated, decimal commas, UTF-8 with BOM; the file ends in empty padding rows.
  * s1..s36 are 0/25/50/75/100 in codebook order, `-1` = "no answer"; they are mapped to
    the project's [-1, 1] scale with `raw / 50 - 1`.
  * A party's seats are MEPs_1 + MEPs_2 + MEPs_3: ten national lists split their MEPs
    across groups (PSD-PNL S&D/EPP, GL-PvdA G/EFA/S&D, ...). The party's "real" group is
    the one holding most of its seats; the seat crosstab splits them exactly.

**Features.** All 36 statements, unstandardised -- every item is already on the same
five-point scale, and z-scoring would up-weight near-consensus items. The repo has text
for 30 of them (the questionnaire file), and those 30 are not s1..s30 -- see
RAW_COLUMN_FOR_QUESTIONNAIRE. The other six are clustered on but reported by number.

**Missing answers.** ~8% of cells for the MEP-holding parties, concentrated in a handful
of parties (one misses 25 of 36). k-means needs complete rows, so gaps are filled by
KNN imputation (5 nearest parties, nan-euclidean). Filling with 0 would drag sparse
parties toward the centre and manufacture a "moderate" cluster. `--max-missing` drops
sparse parties instead, which is the sensitivity check for this choice.

**Choosing k (2..7).** Silhouette, Calinski-Harabasz, Davies-Bouldin, the inertia
elbow, and bootstrap stability: refit on resampled parties, label the full set, and
report the ARI against the full-data fit. The pick is the highest-silhouette k among
the stable ones (mean ARI >= `--min-stability`); `--k` overrides it.

Outputs go to `data/party_kmeans/`: the two figures, per-k metrics, party assignments.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

RAW_CSV = Path("data/euandi_2024_raw/EUandI_2024_party_dataset.csv")
QUESTIONNAIRE = Path("data/euandi_2024_data/euandi_2024_questionnaire_codebook_order.jsonl")
COUNTRY_BY_ISO = {"de": "Germany", "fr": "France", "it": "Italy", "es": "Spain", "gr": "Greece"}
PARTIES_CODEBOOK = Path("data/euandi_2024_data/euandi_2024_parties_codebook_order.jsonl")
OUT_DIR = Path("data/party_kmeans")

N_STATEMENTS = 36
S_COLS = [f"s{i}" for i in range(1, N_STATEMENTS + 1)]
# The raw file has 36 statements; the repo's questionnaire keeps 30 of them, and they are
# NOT s1..s30: s11, s14, s21, s26, s32 and s33 are the six the questionnaire leaves out.
# Recovered by matching every raw column against the per-party answers in the
# codebook-order parties jsonl (21 parties, 100% agreement on each of the 30, runner-up
# column <= 74%); `check_raw_columns` re-verifies it on every run.
RAW_COLUMN_FOR_QUESTIONNAIRE = [f"s{i}" for i in (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17,
    18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 34, 35, 36)]
SEAT_COLS = [("POLITICAL_GROUP_1", "MEPs_1"), ("POLITICAL_GROUP_2", "MEPs_2"),
             ("POLITICAL_GROUP_3", "MEPs_3")]
# The raw file's own 0-100 summary placements, used to read the clusters, not to fit them.
SUMMARY_COLS = {"LR_ECON": "econ right", "GALTAN": "TAN", "PRO/ANTI-EU": "pro-EU"}
# The tenth-term groups as the raw file spells them, left to right.
EP_GROUPS = ["LEFT", "G/EFA", "S&D", "RENEW", "EPP", "ECR", "PFE", "ESN", "NI"]
K_RANGE = range(2, 8)
# Names for the fictional groups, read off the printed cluster profiles. Keyed by k, in
# the LR_ECON order `order_clusters` fixes; they only hold for the default seed/imputation.
CLUSTER_NAMES = {
    2: ["Pro-European mainstream", "Sovereigntist right"],
    3: ["Progressive left", "Sovereigntist right", "Liberal-conservative centre-right"],
    4: ["Radical left", "Progressive federalists", "Sovereigntist right",
        "Liberal-conservative centre-right"],
}

# Reference categorical palette (dataviz skill, light mode), in fixed slot order. A
# scatter can only keep ~3 hues apart for every reader, so each cluster also gets its
# own marker and every cluster is direct-labelled at its centroid.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_parties(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",", dtype=str)
    df = df.dropna(subset=["COUNTRY"]).reset_index(drop=True)
    for col in S_COLS + list(SUMMARY_COLS):
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    seats = {}
    for group_col, mep_col in SEAT_COLS:
        # One MEPs_3 cell holds a stray "Obrázok" (an image placeholder); coerce drops it.
        meps = pd.to_numeric(df[mep_col], errors="coerce").fillna(0)
        groups = df[group_col].fillna("").str.strip()
        for i in np.flatnonzero(meps > 0):
            seats.setdefault(i, {}).setdefault(groups[i] or "NI", 0)
            seats[i][groups[i] or "NI"] += int(meps[i])
    df["seats_by_group"] = [seats.get(i, {}) for i in range(len(df))]
    df["meps"] = [sum(s.values()) for s in df["seats_by_group"]]
    df["ep_group"] = [max(s, key=s.get) if s else None for s in df["seats_by_group"]]
    df["label"] = df["ABBREVIATON"].str.strip() + " (" + df["COUNTRY"].str.strip().str[:3] + ")"
    return df


def answer_matrix(df: pd.DataFrame) -> np.ndarray:
    raw = df[S_COLS].to_numpy(dtype=float)
    raw[raw < 0] = np.nan
    return raw / 50.0 - 1.0


def statement_texts(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        rows = sorted((json.loads(line) for line in f if line.strip()),
                      key=lambda r: r["statement_idx"])
    return {RAW_COLUMN_FOR_QUESTIONNAIRE[i]: r["statement"] for i, r in enumerate(rows)}


def check_raw_columns(df: pd.DataFrame, path: Path = PARTIES_CODEBOOK) -> None:
    """Fail loudly if RAW_COLUMN_FOR_QUESTIONNAIRE stops matching the per-party answers
    in the codebook-order jsonl, for the parties whose abbreviation appears in both."""
    with open(path, encoding="utf-8") as f:
        parties = [json.loads(line) for line in f if line.strip()]
    # Keyed on country too: abbreviations collide across countries (Portugal's CDU).
    raw = df.set_index([df["COUNTRY"].str.strip(), df["ABBREVIATON"].str.strip().str.lower()])
    raw = raw[~raw.index.duplicated(keep=False)]
    mismatches, checked = [], 0
    for party in parties:
        key = (COUNTRY_BY_ISO.get(party["country_iso"]), party["short_name"].lower())
        if key not in raw.index:
            continue
        for r in party["responses"]:
            value, col = r.get("normalized_answer"), RAW_COLUMN_FOR_QUESTIONNAIRE[r["statement_idx"]]
            answer = raw.at[key, col]
            if isinstance(value, (int, float)) and answer >= 0:
                checked += 1
                if abs(answer / 50.0 - 1.0 - value) > 1e-9:
                    mismatches.append((party["short_name"], col))
    if not checked or mismatches:
        raise SystemExit(f"raw s-column mapping disagrees with {path} on "
                         f"{len(mismatches)}/{checked} answers, e.g. {mismatches[:5]}")


def fit(X: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)


def stability(X: np.ndarray, k: int, reference: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        sample = rng.choice(len(X), size=len(X), replace=True)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed + b).fit(X[sample])
        scores.append(adjusted_rand_score(reference, model.predict(X)))
    return np.array(scores)


def select_k(X: np.ndarray, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    rows, models = [], {}
    for k in K_RANGE:
        model = fit(X, k, seed)
        models[k] = model
        ari = stability(X, k, model.labels_, n_boot, seed)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, model.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, model.labels_),
            "davies_bouldin": davies_bouldin_score(X, model.labels_),
            "stability_ari": ari.mean(),
            "stability_ari_lo": np.percentile(ari, 5),
            "stability_ari_hi": np.percentile(ari, 95),
            "smallest_cluster": np.bincount(model.labels_).min(),
        })
    return pd.DataFrame(rows).set_index("k"), models


def order_clusters(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Renumber clusters left to right by mean LR_ECON, so cluster ids (and so colours)
    do not depend on k-means' arbitrary initialisation order."""
    means = pd.Series(df["LR_ECON"].to_numpy()).groupby(labels).mean().sort_values()
    remap = {old: new for new, old in enumerate(means.index)}
    return np.array([remap[l] for l in labels])


# --------------------------------------------------------------------------- plots

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_k_selection(metrics: pd.DataFrame, best_k: int, min_stability: float, path: Path):
    panels = [
        ("silhouette", "Silhouette", "higher is better"),
        ("stability_ari", "Bootstrap stability (ARI)", "higher is better"),
        ("inertia", "Inertia (elbow)", "look for the bend"),
        ("calinski_harabasz", "Calinski-Harabasz", "higher is better"),
        ("davies_bouldin", "Davies-Bouldin", "lower is better"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.6), facecolor=SURFACE)
    ks = metrics.index.to_numpy()
    for ax, (col, title, hint) in zip(axes, panels):
        style(ax)
        if col == "stability_ari":
            ax.fill_between(ks, metrics["stability_ari_lo"], metrics["stability_ari_hi"],
                            color=CLUSTER_COLORS[0], alpha=0.15, lw=0)
            ax.axhline(min_stability, color=INK_MUTED, lw=1, ls="--")
            ax.annotate(f"stability floor {min_stability}", (ks[-1], min_stability),
                        xytext=(0, 4), textcoords="offset points", ha="right",
                        fontsize=8, color=INK_MUTED)
        ax.plot(ks, metrics[col], color=CLUSTER_COLORS[0], lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(best_k, metrics.loc[best_k, col], marker="o", ms=12, color="none",
                markeredgecolor=INK, markeredgewidth=1.8)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_ylabel(hint, fontsize=9, color=INK_MUTED)
        ax.set_xticks(ks)
        ax.set_xlabel("k", color=INK_MUTED)
    fig.suptitle(f"K-means on EU&I 2024 parties with at least one MEP; chosen k = {best_k} (ringed)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_clusters(df: pd.DataFrame, names: list[str], path: Path, n_labels: int = 25):
    k = len(names)
    fig, (ax, hm) = plt.subplots(1, 2, figsize=(17, 7.5), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    style(ax)
    halo = [matplotlib.patheffects.withStroke(linewidth=3, foreground=SURFACE)]
    for c in range(k):
        sub = df[df["cluster"] == c]
        ax.scatter(sub["LR_ECON"], sub["GALTAN"], s=18 + 7 * sub["meps"],
                   color=CLUSTER_COLORS[c], marker=CLUSTER_MARKERS[c], alpha=0.8,
                   edgecolor=SURFACE, linewidth=1.2, label=f"{c + 1}: {names[c]}", zorder=3)
    # Only the biggest delegations are named; the full list is in the CSV.
    for _, row in df.nlargest(n_labels, "meps").iterrows():
        ax.annotate(row["label"], (row["LR_ECON"], row["GALTAN"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5, color=INK_MUTED,
                    path_effects=halo, zorder=4)
    for c in range(k):
        sub = df[df["cluster"] == c]
        ax.annotate(str(c + 1), (sub["LR_ECON"].mean(), sub["GALTAN"].mean()), fontsize=15,
                    fontweight="bold", color=INK, ha="center", va="center", zorder=5,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground=SURFACE)])
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("LR_ECON  (0 = economic left, 100 = economic right)", color=INK_MUTED)
    ax.set_ylabel("GALTAN  (0 = green/alternative/libertarian, 100 = trad./authoritarian/nationalist)",
                  color=INK_MUTED)
    ax.set_title("Parties on EU&I's own summary axes, coloured by cluster (size = MEPs;\n"
                 "numbers are cluster centroids; the axes were not used in the fit)",
                 fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, markerscale=0.8, labelcolor=INK)

    groups = [g for g in EP_GROUPS if g in set().union(*df["seats_by_group"])]
    seats = np.zeros((k, len(groups)), dtype=int)
    for _, row in df.iterrows():
        for g, n in row["seats_by_group"].items():
            seats[row["cluster"], groups.index(g)] += n
    hm.imshow(seats, cmap="Blues", aspect="auto", vmin=0)
    hm.set_xticks(range(len(groups)), groups, fontsize=9, color=INK)
    hm.set_yticks(range(k), [f"{c + 1}: {names[c]}\n({seats[c].sum()} MEPs)" for c in range(k)],
                  fontsize=9, color=INK)
    threshold = seats.max() * 0.55
    for i in range(k):
        for j in range(len(groups)):
            if seats[i, j]:
                hm.text(j, i, seats[i, j], ha="center", va="center", fontsize=9,
                        color="white" if seats[i, j] > threshold else INK)
    for side in hm.spines.values():
        side.set_visible(False)
    hm.set_title("MEP seats: fictional group (rows) x real EP group (columns)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report

def describe_clusters(df: pd.DataFrame, X: np.ndarray, texts: dict[str, str], top: int):
    overall = X.mean(axis=0)
    k = df["cluster"].max() + 1
    for c in range(k):
        mask = (df["cluster"] == c).to_numpy()
        sub = df[mask]
        seats = {}
        for s in sub["seats_by_group"]:
            for g, n in s.items():
                seats[g] = seats.get(g, 0) + n
        seat_str = ", ".join(f"{g} {n}" for g, n in sorted(seats.items(), key=lambda x: -x[1]))
        summary = ", ".join(f"{label} {sub[col].mean():.0f}" for col, label in SUMMARY_COLS.items())
        print(f"\n=== Cluster {c + 1}: {mask.sum()} parties, {sub['meps'].sum()} MEPs ===")
        print(f"  real groups (seats): {seat_str}")
        print(f"  EU&I summary axes (0-100 means): {summary}")
        centroid = X[mask].mean(axis=0)
        diff = centroid - overall
        print("  most distinctive statements (cluster mean vs all, on -1..1):")
        for i in np.argsort(-np.abs(diff))[:top]:
            s = S_COLS[i]
            print(f"    {s:>4} {centroid[i]:+.2f} ({diff[i]:+.2f})  {texts.get(s, '[text not in repo]')}")
        biggest = sub.nlargest(12, "meps")
        print("  largest members: " + ", ".join(f"{r.label} {r.meps}" for r in biggest.itertuples()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, help="override the automatic choice of k")
    parser.add_argument("--min-stability", type=float, default=0.75,
                        help="mean bootstrap ARI a k must reach to be eligible")
    parser.add_argument("--max-missing", type=int, default=N_STATEMENTS,
                        help="drop parties with more unanswered statements than this")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-statements", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = load_parties(RAW_CSV)
    check_raw_columns(df)
    df = df[df["meps"] >= 1].reset_index(drop=True)
    raw = answer_matrix(df)
    n_missing = np.isnan(raw).sum(axis=1)
    keep = n_missing <= args.max_missing
    df, raw = df[keep].reset_index(drop=True), raw[keep]
    print(f"{len(df)} parties with >= 1 MEP ({df['meps'].sum()} MEPs); "
          f"{np.isnan(raw).mean():.1%} of answers missing, KNN-imputed; "
          f"{(~keep).sum()} dropped by --max-missing {args.max_missing}")
    X = KNNImputer(n_neighbors=5).fit_transform(raw)

    metrics, models = select_k(X, args.n_boot, args.seed)
    print("\n" + metrics.round(3).to_string())
    stable = metrics[metrics["stability_ari"] >= args.min_stability]
    auto_k = int((stable if len(stable) else metrics)["silhouette"].idxmax())
    best_k = args.k or auto_k
    print(f"\nhighest-silhouette k with stability >= {args.min_stability}: {auto_k}"
          + (f"; using --k {best_k}" if args.k else ""))

    df["cluster"] = order_clusters(models[best_k].labels_, df)
    texts = statement_texts(QUESTIONNAIRE)
    describe_clusters(df, X, texts, args.top_statements)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir / "k_selection_metrics.csv")
    plot_k_selection(metrics, best_k, args.min_stability, args.out_dir / "k_selection.png")
    names = CLUSTER_NAMES.get(best_k) or [f"cluster {c + 1}" for c in range(best_k)]
    plot_clusters(df, names, args.out_dir / f"clusters_k{best_k}.png")
    out = df[["COUNTRY", "ABBREVIATON", "PARTY_NAME_ENG", "PARTYFAMILY_1", "ep_group", "meps",
              "LR_ECON", "GALTAN", "PRO/ANTI-EU"]].assign(cluster=df["cluster"] + 1)
    out.to_csv(args.out_dir / f"assignments_k{best_k}.csv", index=False)
    return df, best_k


if __name__ == "__main__":
    main()
