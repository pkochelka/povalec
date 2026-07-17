"""Score LLM answers as calibrated party-agreement strengths in the fine-tuned
IdeoEncoder space.

Each answer is compared ONLY to the (statement x party) prototypes of its own
statement. Two steps turn raw cosines into an interpretable scale:

  contrast  : per answer, center the similarities across the parties of its
              statement, so offsets shared by all parties (language, LLM domain,
              anisotropy) cancel.
  calibrate : on EuroParl dev speeches, measure the contrasted similarity that a
              party's own speeches vs other parties' speeches reach in each
              (statement, party) cell. An answer is placed on that scale:
              0 = as close as a rival party's typical speech,
              1 = as close as the party's own typical speech.

Usage:
  python analysis/score_llm_ideo.py --llm-dir data/euandi_2024_results/kimi-k2.6 \
      --model-dir models/harrier-ideo --tag final --metric b
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed_llm_speeches import RESULTS_ROOT, discover_speech_files, extract_speeches, party_token
from eval_ideo_encoder import InMemoryCache, attach_text, encode, sample_per_cell
from party_prototype_embeddings import DATA_DIR, PARTY_COLUMN, FeatureTransform
from statement_cutoff_sweep import build_prototypes, keep_by_quantile
from train_ideo_encoder import IdeoEncoder

AGREEMENT_CLIP = (-0.5, 1.5)
MIN_CALIBRATION_ROWS = 20


def load_llm_answers(llm_dir, tracks):
    rows = []
    for track, path in discover_speech_files(llm_dir):
        if track in tracks:
            rows.extend(extract_speeches(track, path))
    if not rows:
        raise SystemExit(f"no LLM answers found in {llm_dir}")
    return pd.DataFrame(rows)


def encode_metric(encoder, texts, batch_size, metric):
    metric_a, metric_b = encode(encoder, texts, batch_size)
    return metric_a if metric == "a" else metric_b


def build_statement_prototypes(encoder, args):
    train = keep_by_quantile(pd.read_parquet(args.train), args.quantile)
    train = attach_text(train, args.train_split, args.data_dir)
    train = sample_per_cell(train, args.per_cell, args.seed).reset_index(drop=True)
    print(f"[prototypes] {len(train):,} train speeches "
          f"(quantile={args.quantile}, per_cell<={args.per_cell})")
    vectors = encode_metric(encoder, train["text"].tolist(), args.batch_size, args.metric)
    cache = InMemoryCache(train["id"].tolist(), vectors)
    transform = FeatureTransform(kind="none").fit(np.zeros((1, vectors.shape[1])))
    return build_prototypes(cache, train, transform)


def contrasted_similarities(vectors, statement_ids, prototypes, parties):
    column = {party: j for j, party in enumerate(parties)}
    contrasts = np.full((len(vectors), len(parties)), np.nan, dtype=np.float32)
    for statement in np.unique(statement_ids):
        present = [p for p in parties if (int(statement), p) in prototypes]
        if len(present) < 2:
            continue
        proto_matrix = np.stack([prototypes[(int(statement), p)] for p in present])
        rows = np.where(statement_ids == statement)[0]
        similarities = vectors[rows] @ proto_matrix.T
        centered = similarities - similarities.mean(axis=1, keepdims=True)
        for j, party in enumerate(present):
            contrasts[rows, column[party]] = centered[:, j]
    return contrasts


def fit_calibration(contrasts, statement_ids, speaker_parties, parties):
    records = []
    for statement in np.unique(statement_ids):
        in_statement = statement_ids == statement
        for j, party in enumerate(parties):
            values = contrasts[in_statement, j]
            speakers = speaker_parties[in_statement]
            own = values[(speakers == party) & ~np.isnan(values)]
            other = values[(speakers != party) & ~np.isnan(values)]
            if len(own) < MIN_CALIBRATION_ROWS or len(other) < MIN_CALIBRATION_ROWS:
                continue
            records.append({
                "statement_idx": int(statement), "party": party,
                "mu_own": float(own.mean()), "mu_other": float(other.mean()),
                "n_own": len(own), "n_other": len(other),
            })
    calibration = pd.DataFrame(records)
    separated = calibration["mu_own"] > calibration["mu_other"]
    if (~separated).any():
        print(f"[calibration] dropping {(~separated).sum()} cells where dev speeches "
              f"of the party sit no closer than rivals (no signal to scale by)")
    return calibration[separated].set_index(["statement_idx", "party"])


def apply_calibration(contrasts, statement_ids, calibration, parties):
    column = {party: j for j, party in enumerate(parties)}
    agreement = np.full_like(contrasts, np.nan)
    for (statement, party), cell in calibration.iterrows():
        rows = statement_ids == statement
        scale = cell["mu_own"] - cell["mu_other"]
        agreement[rows, column[party]] = (contrasts[rows, column[party]] - cell["mu_other"]) / scale
    return np.clip(agreement, *AGREEMENT_CLIP)


def summarize_per_statement(answers, parties):
    agreement_columns = [f"agreement_{party_token(p)}" for p in parties]
    return answers.groupby(["track", "statement_idx"])[agreement_columns].mean()


def report_stance_flip(summary):
    tracks = set(summary.index.get_level_values("track"))
    if not {"base", "negated"} <= tracks:
        return
    base, negated = summary.loc["base"], summary.loc["negated"]
    shared = base.index.intersection(negated.index)
    correlation = pd.Series({s: base.loc[s].corr(negated.loc[s]) for s in shared})
    print("\n=== stance flip: corr(base, negated) agreement profile per statement ===")
    print(correlation.round(3).to_string())
    print(f"\nanti-correlated (corr < 0, encoder reads stance): {(correlation < 0).mean():.0%} "
          f"of statements; the rest read topic there -> interpret with caution")


def main():
    args = build_parser().parse_args()
    llm_dir = Path(args.llm_dir)
    answers = load_llm_answers(llm_dir, set(args.tracks.split(",")))
    print(f"[{llm_dir.name}] {len(answers):,} answers | {answers['language'].nunique()} languages "
          f"| tracks: {sorted(answers['track'].unique())}")

    encoder = IdeoEncoder.load(args.model_dir, args.tag)
    parties = sorted(encoder.meta["parties"])

    prototypes = build_statement_prototypes(encoder, args)

    dev = attach_text(pd.read_parquet(args.dev), args.dev_split, args.data_dir)
    if len(dev) > args.dev_sample:
        dev = dev.sample(args.dev_sample, random_state=args.seed).reset_index(drop=True)
    dev_vectors = encode_metric(encoder, dev["text"].tolist(), args.batch_size, args.metric)
    dev_contrasts = contrasted_similarities(
        dev_vectors, dev["statement_idx"].to_numpy(), prototypes, parties)
    calibration = fit_calibration(
        dev_contrasts, dev["statement_idx"].to_numpy(), dev[PARTY_COLUMN].to_numpy(), parties)
    print(f"[calibration] {len(calibration)} usable (statement, party) cells "
          f"from {len(prototypes)} prototypes")

    answer_vectors = encode_metric(encoder, answers["text"].tolist(), args.batch_size, args.metric)
    statements = answers["statement_idx"].to_numpy()
    contrasts = contrasted_similarities(answer_vectors, statements, prototypes, parties)
    agreement = apply_calibration(contrasts, statements, calibration, parties)
    for j, party in enumerate(parties):
        answers[f"agreement_{party_token(party)}"] = agreement[:, j]

    stem = f"ideo_agreement_{args.tag}_metric{args.metric}"
    answers.drop(columns=["text"]).to_parquet(llm_dir / f"{stem}.parquet", index=False)
    calibration.to_csv(llm_dir / f"{stem}_calibration.csv")
    summary = summarize_per_statement(answers, parties)
    summary.to_csv(llm_dir / f"{stem}_per_statement.csv")
    print(f"\nSaved row-level scores, calibration table and per-statement summary to {llm_dir}")

    base = answers[answers["track"] == "base"]
    print("\n=== mean agreement per party (base track; 0 = rival-like, 1 = own-party-like) ===")
    for party in parties:
        print(f"  {party:<12} {base[f'agreement_{party_token(party)}'].mean():.3f}")
    report_stance_flip(summary)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-dir", default=str(RESULTS_ROOT / "mistral-medium-3.5"))
    p.add_argument("--model-dir", default="models/harrier-ideo")
    p.add_argument("--tag", default="final")
    p.add_argument("--metric", choices=["a", "b"], default="b",
                   help="a = backbone cosine, b = projection-head learned metric")
    p.add_argument("--tracks", default="base,negated")
    p.add_argument("--train", default=str(DATA_DIR / "topics_train_statements.parquet"))
    p.add_argument("--dev", default=str(DATA_DIR / "topics_dev_statements.parquet"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--dev-split", default="dev")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--quantile", type=float, default=0.5,
                   help="per-statement margin keep-quantile for prototype speeches "
                        "(the knee from statement_cutoff_sweep)")
    p.add_argument("--per-cell", type=int, default=300,
                   help="max train speeches per (statement, party) prototype")
    p.add_argument("--dev-sample", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main()
