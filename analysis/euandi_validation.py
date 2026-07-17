"""Does the fine-tuned geometry encode IDEOLOGY or just a party fingerprint?

Test: can a single linear "agree/disagree readout" direction, learned on some
EU&I statements, predict the parties' signed positions on UNSEEN statements?

  ground truth : euandi_2024_parties.jsonl normalized_answer (-1..1), national
                 parties aggregated to EP group via EP_GROUP_BY_PARTY.
  recovered    : (group x statement) prototypes in the fine-tuned space.

Per statement we CENTER across groups (remove the topic component -> leaves the
group's stance deviation), then run leave-one-statement-out CV: fit PCA+Ridge on
train statements, predict the held-out statement. The readout learns its sign from
the training statements, so the per-statement correlation on held-out statements is
honest (no orientation leakage). A shuffled-label baseline gives the chance level.

Usage:
  python analysis/euandi_validation.py --model-dir models/harrier-ideo --tag final
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import DATA_DIR, PARTY_COLUMN, FeatureTransform
from statement_cutoff_sweep import build_prototypes
from eval_ideo_encoder import InMemoryCache, attach_text, encode, sample_per_cell
from train_ideo_encoder import IdeoEncoder
from evaluate_euandi import PARTY_POSITIONS_PATH, load_party_positions


def euandi_matrix(parties, statements):
    """(group x statement) mean normalized_answer in [-1, 1]; NaN where no party answered."""
    pp = load_party_positions(PARTY_POSITIONS_PATH).dropna(subset=["ep_group"])
    gt = pp.groupby(["ep_group", "statement_idx"])["normalized_answer"].mean()
    return np.array([[gt.get((p, s), np.nan) for s in statements] for p in parties])


def center_per_statement(V, Y):
    """Remove the per-statement (across-group) mean from both -> stance deviations."""
    G, S = Y.shape
    R = np.full(V.shape, np.nan)
    Ytil = np.full((G, S), np.nan)
    for j in range(S):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 3:
            continue
        R[m, j] = V[m, j] - V[m, j].mean(0, keepdims=True)
        Ytil[m, j] = Y[m, j] - Y[m, j].mean()
    return R, Ytil


def loo_predict(R, Ytil, n_pca, alpha):
    """Leave-one-statement-out: predict each statement's stance deviations from the rest."""
    G, S, _ = R.shape
    preds = np.full((G, S), np.nan)
    for test_j in range(S):
        Xtr, ytr = [], []
        for j in range(S):
            if j == test_j:
                continue
            m = ~np.isnan(Ytil[:, j])
            Xtr.append(R[m, j])
            ytr.append(Ytil[m, j])
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        m = ~np.isnan(Ytil[:, test_j])
        if m.sum() < 2:
            continue
        pca = PCA(n_components=min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])).fit(Xtr)
        ridge = Ridge(alpha=alpha).fit(pca.transform(Xtr), ytr)
        preds[m, test_j] = ridge.predict(pca.transform(R[m, test_j]))
    return preds


def score(preds, Ytil):
    flat_p, flat_t, per_stmt = [], [], []
    for j in range(Ytil.shape[1]):
        m = ~np.isnan(preds[:, j]) & ~np.isnan(Ytil[:, j])
        if m.sum() < 2:
            continue
        flat_p += preds[m, j].tolist()
        flat_t += Ytil[m, j].tolist()
        if m.sum() >= 3 and np.ptp(Ytil[m, j]) > 0:
            rho, _ = spearmanr(preds[m, j], Ytil[m, j])
            per_stmt.append(rho)
    return {
        "pooled_spearman": spearmanr(flat_p, flat_t)[0],
        "pooled_pearson": pearsonr(flat_p, flat_t)[0],
        "per_statement_spearman": float(np.nanmean(per_stmt)),
        "n_pairs": len(flat_p),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/harrier-ideo")
    p.add_argument("--tag", default="final")
    p.add_argument("--train", default=str(DATA_DIR / "topics_train_statements.parquet"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--space", choices=["A", "B"], default="A", help="A=backbone cosine, B=projection")
    p.add_argument("--per-cell", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pca", type=int, default=20)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    encoder = IdeoEncoder.load(args.model_dir, args.tag)
    parties = sorted(encoder.meta["parties"])

    proto_df = attach_text(pd.read_parquet(args.train), args.train_split, args.data_dir)
    proto_df = sample_per_cell(proto_df, args.per_cell, args.seed).reset_index(drop=True)
    print(f"prototypes from {len(proto_df):,} train speeches | space {args.space}")
    a_vecs, b_vecs = encode(encoder, proto_df["text"].tolist(), args.batch_size)
    cache = InMemoryCache(proto_df["id"].tolist(), a_vecs if args.space == "A" else b_vecs)
    protos = build_prototypes(cache, proto_df, FeatureTransform(kind="none").fit(np.zeros((1, a_vecs.shape[1]))))

    statements = sorted({s for s, _ in protos})
    V = np.stack([[protos[(s, p)] for s in statements] for p in parties])   # (G, S, d)
    Y = euandi_matrix(parties, statements)                                  # (G, S)
    valid = ~np.isnan(Y)
    print(f"groups={len(parties)} statements={len(statements)} "
          f"| euandi cells: {valid.sum()}/{Y.size}")

    R, Ytil = center_per_statement(V, Y)
    metrics = score(loo_predict(R, Ytil, args.pca, args.alpha), Ytil)

    # shuffled-label chance level: permute groups within each statement, same CV
    rng = np.random.default_rng(args.seed)
    Ysh = Ytil.copy()
    for j in range(Ysh.shape[1]):
        m = ~np.isnan(Ysh[:, j])
        Ysh[m, j] = rng.permutation(Ysh[m, j])
    shuffled = score(loo_predict(R, Ysh, args.pca, args.alpha), Ysh)

    print("\n=== EU&I validation (orientation-fold, leave-one-statement-out) ===")
    print(f"pooled Spearman (held-out stance) : {metrics['pooled_spearman']:.3f}")
    print(f"pooled Pearson                    : {metrics['pooled_pearson']:.3f}")
    print(f"mean per-statement Spearman       : {metrics['per_statement_spearman']:.3f}  "
          f"(rank of {len(parties)} groups within each unseen statement)")
    print(f"shuffled-label baseline (Spearman): {shuffled['pooled_spearman']:.3f}")
    print(f"n held-out (group,statement) pairs: {metrics['n_pairs']}")


if __name__ == "__main__":
    main()
