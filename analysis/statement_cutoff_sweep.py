"""Pick the per-statement margin cutoff by the DOWNSTREAM objective, not by eye.

For each candidate keep-quantile q, we:
  1. keep, within each statement, the top-q fraction of train speeches by
     `statement_margin` (top1 - top2 sim; the anisotropy offset cancels),
  2. build per-(statement, party) prototypes in the FROZEN Harrier space
     (optionally language-centered, matching the existing pipeline),
  3. score dev speeches against the 7 party prototypes OF THEIR OWN STATEMENT and
     report within-party vs between-party cosine + nearest-party top-1.

The quantity you want maximized is `gap = own - other` (small within-party
distance, large between-party distance). Tightening q raises the gap but shrinks
coverage and risks keeping only prototypical exemplars -> pick the knee, then
spot-check. This runs on the frozen embeddings, before any fine-tuning.

Usage:
  python analysis/statement_cutoff_sweep.py \
    --train "data/EuroParl Custom/topics_train_statements.parquet" \
    --dev   "data/EuroParl Custom/topics_dev_statements.parquet" \
    --quantiles 1.0,0.75,0.5,0.25 --transform lang_center
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import (
    CACHE_DIR,
    DATA_DIR,
    PARTY_COLUMN,
    EmbeddingCache,
    FeatureTransform,
)


def load_topics(path, cache):
    df = pd.read_parquet(path)
    if "statement_margin" not in df.columns:
        raise SystemExit(f"{path} has no 'statement_margin' -> rerun assign_topics.py "
                         f"(updated to store the margin)")
    cached = df["id"].map(cache.has)
    if not cached.all():
        print(f"[{Path(path).name}] dropping {(~cached).sum():,} uncached rows")
        df = df[cached]
    return df.reset_index(drop=True)


def fit_transform(cache, train, kind, fit_sample, seed=0):
    if kind == "none":
        return FeatureTransform(kind="none").fit(np.zeros((1, cache.vectors.shape[1])))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(train), min(fit_sample, len(train)), replace=False)
    sub = train.iloc[sel]
    emb = cache.get(sub["id"].tolist())
    return FeatureTransform(kind=kind).fit(emb, groups=sub["language"].to_numpy())


def keep_by_quantile(train, q):
    """Within each statement, keep the top-q fraction by margin."""
    if q >= 1.0:
        return train
    thr = train.groupby("statement_idx")["statement_margin"].transform(
        lambda s: s.quantile(1.0 - q)
    )
    return train[train["statement_margin"] >= thr]


def build_prototypes(cache, df, transform, chunk=100000):
    """(statement_idx, party) -> unit prototype, accumulated in chunks."""
    ids = df["id"].to_numpy()
    langs = df["language"].to_numpy()
    stmts = df["statement_idx"].to_numpy()
    parts = df[PARTY_COLUMN].to_numpy()
    sums, counts = {}, {}
    for start in range(0, len(df), chunk):
        sl = slice(start, min(start + chunk, len(df)))
        emb = transform.apply(cache.get(ids[sl].tolist()), langs[sl]).astype(np.float64)
        cs, cp = stmts[sl], parts[sl]
        for s, p in set(zip(cs.tolist(), cp.tolist())):
            m = (cs == s) & (cp == p)
            key = (int(s), p)
            sums[key] = sums.get(key, 0.0) + emb[m].sum(0)
            counts[key] = counts.get(key, 0) + int(m.sum())
    protos = {}
    for key, v in sums.items():
        vec = v / counts[key]
        protos[key] = vec / (np.linalg.norm(vec) or 1.0)
    return protos


def evaluate(cache, dev, transform, protos, parties):
    ids = dev["id"].to_numpy()
    langs = dev["language"].to_numpy()
    stmts = dev["statement_idx"].to_numpy()
    parts = dev[PARTY_COLUMN].to_numpy()
    own_sum = other_sum = 0.0
    correct = total = 0
    for s in np.unique(stmts):
        present = [p for p in parties if (int(s), p) in protos]
        if len(present) < 2:
            continue
        proto_mat = np.stack([protos[(int(s), p)] for p in present])   # (k, d)
        pcol = {p: i for i, p in enumerate(present)}
        rows = np.where(stmts == s)[0]
        valid = np.array([parts[r] in pcol for r in rows])
        rows = rows[valid]
        if len(rows) == 0:
            continue
        emb = transform.apply(cache.get(ids[rows].tolist()), langs[rows]).astype(np.float64)
        sims = emb @ proto_mat.T                                       # (n, k)
        jcol = np.array([pcol[parts[r]] for r in rows])
        own = sims[np.arange(len(sims)), jcol]
        other = (sims.sum(1) - own) / (len(present) - 1)
        own_sum += own.sum()
        other_sum += other.sum()
        correct += int((sims.argmax(1) == jcol).sum())
        total += len(rows)
    return {
        "own": own_sum / total, "other": other_sum / total,
        "gap": (own_sum - other_sum) / total, "top1": correct / total, "n_dev": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=str(DATA_DIR / "topics_train_statements.parquet"))
    p.add_argument("--dev", default=str(DATA_DIR / "topics_dev_statements.parquet"))
    p.add_argument("--cache", default=str(CACHE_DIR))
    p.add_argument("--transform", choices=["none", "lang_center"], default="lang_center")
    p.add_argument("--quantiles", default="1.0,0.75,0.5,0.25")
    p.add_argument("--fit-sample", type=int, default=300000)
    args = p.parse_args()

    cache = EmbeddingCache(args.cache)
    train = load_topics(args.train, cache)
    dev = load_topics(args.dev, cache)
    parties = sorted(train[PARTY_COLUMN].unique())
    transform = fit_transform(cache, train, args.transform, args.fit_sample)

    print(f"\nsweep | transform={args.transform} | parties={len(parties)} | "
          f"train={len(train):,} dev={len(dev):,}")
    print(f"{'quantile':>9}{'train_kept':>12}{'own_cos':>9}{'other_cos':>11}"
          f"{'gap':>8}{'top1':>8}{'n_dev':>9}")
    print("-" * 66)
    for q in [float(x) for x in args.quantiles.split(",")]:
        kept = keep_by_quantile(train, q)
        protos = build_prototypes(cache, kept, transform)
        m = evaluate(cache, dev, transform, protos, parties)
        print(f"{q:>9.2f}{len(kept) / len(train):>12.1%}{m['own']:>9.4f}"
              f"{m['other']:>11.4f}{m['gap']:>8.4f}{m['top1']:>8.4f}{m['n_dev']:>9,}")


if __name__ == "__main__":
    main()