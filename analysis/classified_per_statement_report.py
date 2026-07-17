"""Per-statement breakdown of classifier predictions for the LLM speeches.

Analogue of embed_llm_speeches.per_statement_report, but instead of nearest-prototype
cosine geometry it pools the *_classified.csv predicted_party columns over
language/variant/track and reports, for every statement (i.e. every euandi topic),
the ordered counts of predicted EP party. Lets you see how the predicted leaning
depends on the topic.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "data" / "euandi_2024_results"
PARTIES_FILE = REPO_ROOT / "data" / "euandi_2024_data" / "euandi_2024_parties.jsonl"

# predicted_party_<lang>[_negated|_question]_v<variant>
PRED_RE = re.compile(r"^predicted_party_([a-z]{2})(?:_(negated|question))?_v(\d+)$")
# stance_<lang>[_negated|_question]_v<variant>  (speech stance score in [-1, +1])
STANCE_RE = re.compile(r"^stance_([a-z]{2})(?:_(negated|question))?_v(\d+)$")

# classifier EP-group label -> euandi EU-level party (country_iso == "eu") that voices it
GROUP_TO_EUANDI = {
    "PPE": "EPP", "S&D": "PES", "ALDE": "ALDE", "ECR": "ECR",
    "Greens/EFA": "EGP", "ID": "ID", "GUE/NGL": "PEL",
}


def discover_classified_files(model_dir):
    files = []
    for path in sorted(model_dir.glob("speeches_*_classified.csv")):
        name = path.name
        track = "negated" if "_negated" in name else "question" if "_question" in name else "base"
        files.append((track, path))
    return files


def extract_predictions(track, path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    labels = df["original_text_en"] if "original_text_en" in df.columns else None
    rows = []
    for column in df.columns:
        match = PRED_RE.match(column)
        if not match:
            continue
        lang, col_track, variant = match.group(1), match.group(2), int(match.group(3))
        for statement_idx, value in df[column].items():
            if not isinstance(value, str):
                continue
            party = value.strip()
            if not party or party.upper().startswith("FAILED"):
                continue
            rows.append({
                "track": col_track or track,
                "language": lang,
                "variant": f"v{variant}",
                "statement_idx": int(statement_idx),
                "predicted_party": party,
            })
    return rows, labels


def discover_scored_files(model_dir):
    files = []
    for path in sorted(model_dir.glob("speeches_*_scored.csv")):
        name = path.name
        track = "negated" if "_negated" in name else "question" if "_question" in name else "base"
        files.append((track, path))
    return files


def extract_scores(track, path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    rows = []
    for column in df.columns:
        match = STANCE_RE.match(column)
        if not match:
            continue
        lang, col_track, variant = match.group(1), match.group(2), int(match.group(3))
        values = pd.to_numeric(df[column], errors="coerce")
        for statement_idx, value in values.items():
            if pd.isna(value):
                continue
            rows.append({
                "track": col_track or track,
                "language": lang,
                "variant": f"v{variant}",
                "statement_idx": int(statement_idx),
                "speech_score": float(value),
            })
    return rows


def load_group_stances():
    """{EP-group label: {statement_idx: normalized_answer}} from the euandi EU-level parties.

    statement_idx here is 0-based, matching the classified/scored CSV row index.
    None ("no opinion") answers are stored as NaN.
    """
    eu_parties = {}
    for line in PARTIES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        party = json.loads(line)
        if party.get("country_iso") == "eu":
            eu_parties[party["short_name"]] = party
    stances = {}
    for group, euandi_name in GROUP_TO_EUANDI.items():
        party = eu_parties.get(euandi_name)
        if party is None:
            continue
        answers = {}
        for resp in party["responses"]:
            value = resp.get("normalized_answer")
            answers[int(resp["statement_idx"])] = float(value) if value is not None else np.nan
        stances[group] = answers
    return stances


def _corr(x, y):
    """Pearson r over paired arrays, ignoring NaNs; None if too few / no variance."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def party_score_correlation(pred_df, score_df, group_stances, statement_labels=None, out_path=None):
    merged = pred_df.merge(score_df, on=["statement_idx", "track", "language", "variant"], how="inner")
    if merged.empty:
        print("\n[party-stance vs score] no overlapping (statement, track, language, variant) keys")
        return None

    merged["party_stance"] = [
        group_stances.get(p, {}).get(s, np.nan)
        for p, s in zip(merged["predicted_party"], merged["statement_idx"])
    ]
    usable = merged.dropna(subset=["party_stance", "speech_score"])
    dropped = len(merged) - len(usable)

    def report_corr(label, sub):
        r, k = _corr(sub["party_stance"], sub["speech_score"])
        rho, _ = _corr(sub["party_stance"].rank(), sub["speech_score"].rank())
        if r is None:
            print(f"{label:<22} Pearson r = n/a (insufficient variance), n={k:,}")
        else:
            print(f"{label:<22} Pearson r = {r:+.3f}   Spearman rho = {rho:+.3f}   n={k:,}")

    print("\n=== assigned-party stance vs speech score ===")
    print(f"{len(usable):,} speeches usable ({dropped:,} dropped: party 'no opinion' or unscored)")
    tracks = sorted(usable["track"].unique())
    # party_stance is track-invariant while the speech score flips sign per track, so a
    # base+negated pool cancels out; report each track separately as well as pooled.
    if len(tracks) > 1:
        for track in tracks:
            report_corr(f"track={track}", usable[usable["track"] == track])
        print("(pooling tracks cancels the signal -- prefer the per-track rows above)")
    report_corr("pooled" if len(tracks) > 1 else f"track={tracks[0]}", usable)

    # per-statement correlation across language/variant/track -> which topics track best
    rows = []
    for stmt, grp in usable.groupby("statement_idx"):
        r, k = _corr(grp["party_stance"], grp["speech_score"])
        rows.append({
            "statement_idx": stmt, "r": r, "n": k,
            "mean_party_stance": grp["party_stance"].mean(),
            "mean_speech_score": grp["speech_score"].mean(),
        })
    per_stmt = pd.DataFrame(rows).sort_values("r", ascending=False, na_position="last")

    print("\n=== per-statement correlation (assigned-party stance vs speech score) ===")
    for _, row in per_stmt.iterrows():
        r_str = f"{row['r']:+.2f}" if pd.notna(row["r"]) else " n/a"
        label = short(statement_labels.get(row["statement_idx"])) if statement_labels else ""
        print(f"[{int(row['statement_idx']):>2}] r={r_str} n={int(row['n']):<4} "
              f"party_stance={row['mean_party_stance']:+.2f} score={row['mean_speech_score']:+.2f}"
              + (f"  | {label}" if label else ""))

    if out_path:
        if statement_labels:
            per_stmt.insert(1, "statement_en",
                            [short(statement_labels.get(s), 120) for s in per_stmt["statement_idx"]])
        per_stmt.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved per-statement correlation to {out_path}")
    return per_stmt


def short(text, width=70):
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def same_party_under_negation(df, score_df=None, statement_labels=None, out_path=None):
    """Per statement: of the base/negated speech pairs (same language+variant), how many
    were classified under the *same* party.

    With score_df, pairs are first restricted to those where the base and negated speech
    scores have the SAME sign. The score is agreement with the *original* statement, so this
    keeps pairs where the model held the same stance in both tracks (agreed with the statement
    both times, or disagreed both times) despite the negated prompt -- then asks whether it
    still got the same party."""
    if not {"base", "negated"} <= set(df["track"].unique()):
        print("\n[negation consistency] skipped: need both base and negated tracks "
              "(run with --tracks base,negated)")
        return None

    key = ["statement_idx", "language", "variant"]
    base = df[df.track == "base"][key + ["predicted_party"]].rename(columns={"predicted_party": "party_base"})
    neg = df[df.track == "negated"][key + ["predicted_party"]].rename(columns={"predicted_party": "party_negated"})
    pairs = base.merge(neg, on=key, how="inner")

    filtered_note = ""
    if score_df is not None:
        sb = (score_df[score_df.track == "base"][key + ["speech_score"]]
              .rename(columns={"speech_score": "score_base"}))
        sn = (score_df[score_df.track == "negated"][key + ["speech_score"]]
              .rename(columns={"speech_score": "score_negated"}))
        before = len(pairs)
        pairs = pairs.merge(sb, on=key, how="inner").merge(sn, on=key, how="inner")
        same_sign = pairs["score_base"] * pairs["score_negated"] > 0
        strong = pairs["score_base"].abs() + pairs["score_negated"].abs() > 1.0
        pairs = pairs[same_sign & strong]
        filtered_note = (f" [filtered to same-sign & strong (|base|+|negated|>1.0) score "
                         f"(stance unchanged): {len(pairs):,} of {before:,} pairs]")

    pairs["same"] = pairs["party_base"] == pairs["party_negated"]
    same_pairs = pairs[pairs["same"]]

    def breakdown(party_series):
        vc = party_series.value_counts()
        return "  ".join(f"{party}: {count}" for party, count in vc.items())

    grouped = pairs.groupby("statement_idx")
    rows = []
    for stmt in sorted(grouped.groups):
        g = grouped.get_group(stmt)
        rows.append({"statement_idx": stmt, "same": int(g["same"].sum()), "pairs": len(g),
                     "kept_parties": breakdown(g[g["same"]]["party_base"])})
    per_stmt = pd.DataFrame(rows)
    per_stmt["share"] = per_stmt["same"] / per_stmt["pairs"]

    print("\n=== same party assigned to base & negated speech, per statement ===" + filtered_note)
    print(f"{int(per_stmt['same'].sum()):,} of {int(per_stmt['pairs'].sum()):,} pairs "
          f"got the SAME party ({per_stmt['same'].sum() / per_stmt['pairs'].sum():.0%} overall)")
    print("kept parties (overall): " + breakdown(same_pairs["party_base"]))
    for _, row in per_stmt.iterrows():
        label = short(statement_labels.get(row["statement_idx"])) if statement_labels else ""
        print(f"[{int(row['statement_idx']):>2}] same={int(row['same']):>3}/{int(row['pairs']):<3} "
              f"({row['share']:>4.0%})" + (f"  | {label}" if label else ""))
        if row["kept_parties"]:
            print("       " + row["kept_parties"])

    if out_path:
        if statement_labels:
            per_stmt.insert(1, "statement_en",
                            [short(statement_labels.get(s), 120) for s in per_stmt["statement_idx"]])
        per_stmt.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved per-statement negation consistency to {out_path}")
    return per_stmt


def per_statement_report(df, statement_labels=None, out_path=None, relative=False):
    parties = sorted(df["predicted_party"].unique())

    # counts of predicted party per statement (pooled over language/variant/track)
    counts = (df.groupby(["statement_idx", "predicted_party"])
                .size().unstack(fill_value=0).reindex(columns=parties, fill_value=0))
    totals = counts.sum(axis=1)

    # per-statement share, and each party's overall mean share (its base rate)
    shares = counts.div(totals, axis=0)
    base_rate = counts.sum(axis=0) / counts.sum().sum()      # P(party) pooled over all statements
    lift = shares.div(base_rate, axis=1)                      # >1 => over-represented on this topic

    if relative:
        print("\n=== per-statement party LIFT vs overall base rate (share / mean share) ===")
        print("base rate per party: " + "  ".join(f"{p}:{base_rate[p]:.0%}" for p in parties))
        rank_metric, top_party = lift, [lift.columns[i] for i in lift.to_numpy().argmax(1)]
    else:
        print("\n=== per-statement predicted-party counts (pooled over lang/variant/track) ===")
        rank_metric, top_party = counts, [counts.columns[i] for i in counts.to_numpy().argmax(1)]

    for stmt in counts.index:
        n = int(totals.loc[stmt])
        order = rank_metric.loc[stmt].sort_values(ascending=False)
        order = order[counts.loc[stmt, order.index] > 0]      # drop parties with zero count
        label = short(statement_labels.get(stmt)) if statement_labels else ""
        head = f"[{stmt:>2}] n={n:<4} top={order.index[0]}"
        head += f" ({lift.loc[stmt, order.index[0]]:.1f}x)" if relative else f" ({shares.loc[stmt, order.index[0]]:.0%})"
        if label:
            head += f"  | {label}"
        print(head)
        if relative:
            print("       " + "  ".join(
                f"{p}:{int(counts.loc[stmt, p])}({lift.loc[stmt, p]:.1f}x)" for p in order.index))
        else:
            print("       " + "  ".join(f"{p}:{int(counts.loc[stmt, p])}" for p in order.index))

    print("\n=== top party across statements"
          f" ({'by lift' if relative else 'modal'}) ===")
    modal = pd.Series(top_party, index=counts.index)
    print("distinct top parties:", modal.nunique(), "of", len(parties))
    print(modal.value_counts().to_string())

    if out_path:
        out = (lift if relative else counts).copy()
        out.insert(0, "n", totals)
        out.insert(1, "top_party", top_party)
        if statement_labels:
            out.insert(0, "statement_en", [short(statement_labels.get(s), 120) for s in counts.index])
        out.to_csv(out_path)
        print(f"\nSaved per-statement {'lift' if relative else 'counts'} to {out_path}")

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(RESULTS_ROOT / "mistral-medium-3.5"))
    parser.add_argument("--tracks", default="base,negated", help="comma list of base,negated,question")
    parser.add_argument("--relative", action="store_true",
                        help="show each party's count as lift vs its overall base rate (share / mean share)")
    parser.add_argument("--score-corr", action="store_true",
                        help="also correlate the assigned party's euandi stance with the speech "
                             "stance score from speeches_*_scored.csv")
    parser.add_argument("--negation", action="store_true",
                        help="per statement, count base/negated speech pairs assigned the SAME party "
                             "(same party for two opposite stances)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    keep = set(args.tracks.split(","))

    records, labels = [], None
    for track, path in discover_classified_files(model_dir):
        if track not in keep:
            continue
        rows, file_labels = extract_predictions(track, path)
        records.extend(rows)
        if labels is None and file_labels is not None:
            labels = file_labels
        print(f"[{model_dir.name}] {track:<8} {path.name}: {len(rows):,} predictions")
    if not records:
        raise SystemExit(f"no classified predictions found in {model_dir} for tracks {sorted(keep)}")

    df = pd.DataFrame(records)
    print(f"[{model_dir.name}] total {len(df):,} predictions over "
          f"{df['language'].nunique()} languages, {df['statement_idx'].nunique()} statements, "
          f"{df['track'].nunique()} tracks")

    statement_labels = labels.to_dict() if labels is not None else None
    default_name = "classified_per_statement_lift.csv" if args.relative else "classified_per_statement.csv"
    out_path = Path(args.out) if args.out else model_dir / default_name
    per_statement_report(df, statement_labels=statement_labels, out_path=out_path, relative=args.relative)

    score_df = None
    if args.score_corr or args.negation:
        score_records = []
        for track, path in discover_scored_files(model_dir):
            if track not in keep:
                continue
            rows = extract_scores(track, path)
            score_records.extend(rows)
            print(f"[{model_dir.name}] {track:<8} {path.name}: {len(rows):,} scores")
        if not score_records:
            raise SystemExit(f"no scored speeches found in {model_dir} for tracks {sorted(keep)}")
        score_df = pd.DataFrame(score_records)

    if args.negation:
        same_party_under_negation(
            df, score_df=score_df, statement_labels=statement_labels,
            out_path=model_dir / "classified_negation_consistency.csv",
        )

    if args.score_corr:
        party_score_correlation(
            df, score_df, load_group_stances(), statement_labels=statement_labels,
            out_path=model_dir / "classified_party_stance_vs_score.csv",
        )


if __name__ == "__main__":
    main()
